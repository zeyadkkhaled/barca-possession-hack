"""One command: the hackathon submission from the raw match data and the saved C3-T model (no training).

    python run.py [--out outputs/submission.csv] [--no-validate]
    python run.py --pathcrf published|<checkpoint> --out outputs/<name>.csv      # the PathCRF combo (CPU, fp32)

Steps: read the match (Metrica EPTS tracking + metadata, official readers) -> player-slot clean-up as the model was trained
(models/c3/decoder.json: feature_mode, dedup) -> ball: smoother + ball-from-players (models/bfp) -> C3-T features ->
layer 1 P(dead), layer 2 P(home | alive), 3-state Viterbi (models/c3) -> post-rules saved with the model (SET PIECES tags
from Smart Tagging) -> one label per frame: H / A / D on the graded halves, X elsewhere -> official validator.
Deterministic: same inputs and models give the same file (LightGBM prediction is deterministic on CPU).
Prints only the label shares, the file's sha256 and the validator result.

--pathcrf (the step-2b combo): C3-T layer 1 decides DEAD (its P(D) and its decoder, unchanged); PathCRF decides the team:
its P(sender is home | sender is a player) replaces C3-T's layer-2 P(home | alive) inside the same Viterbi decoder.
PathCRF's input is built exactly as on the HPC (duplicate rule, ball fill, trim to 11, Hungarian re-linking into 22
lanes; pathcrf_adapter.build.pathcrf_arrays). CPU, fp32, torch deterministic algorithms, fixed threads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PYTHONUTF8", "1")

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

from barca_possession import ballfill as BF  # noqa: E402
from barca_possession import c3t as T  # noqa: E402
from barca_possession import postrules as PR  # noqa: E402
from barca_possession.load import read_metrica  # noqa: E402
from barca_possession.paths import MODELS, OUT, REF  # noqa: E402
from barca_train import pipeline as PL  # noqa: E402

LETTERS = np.array(["H", "A", "D"])


def load_c3t():
    d = MODELS / "c3"
    dec = json.loads((d / "decoder.json").read_text())
    model = dict(l1=lgb.Booster(model_file=str(d / "layer1_dead.txt")), l2=lgb.Booster(model_file=str(d / "layer2_home.txt")),
                 C=np.array(dec["C"]), w=dec["w"], beta=dec["beta"])
    bfp = [lgb.Booster(model_file=str(MODELS / "bfp" / nm)) for nm in ("bfp_dx.txt", "bfp_dy.txt")]
    return model, dec, bfp


def pathcrf_q(m_raw: dict, bfp, spec: str, rows, threads: int, pred_stride_s: float):
    """PathCRF's team probability at C3-T's rows for the hackathon match (CPU, fp32)."""
    import torch
    from pathcrf_adapter import adapter as A
    from pathcrf_adapter.build import pathcrf_arrays
    from pathcrf_adapter.predict import load_model, predict_match
    from pathcrf_adapter.score import q_at_rows
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    m = T.prepare_players(m_raw, "current", dedup=True)                 # as the HPC build
    arr, st = pathcrf_arrays(m, BF.fill_ball(m, bfp))
    model, ball_on = load_model(spec, "cpu")
    r = predict_match(model, A.Match(arr), torch.device("cpu"), False, stride=max(1, int(round(pred_stride_s * 5))), batch=8,
                      ball=ball_on)
    print(f"PathCRF ({spec}, ball input {'on' if ball_on else 'off'}): {len(r['frames'])} steps, uncovered "
          f"{(r['covered'] == 0).mean():.4f}; lanes: both teams >= 10 in {100 * st['window']['share_both_teams_ge10']:.1f} % of windows")
    return q_at_rows(r["frames"], r["q_home"], rows)


def predict_labels(pathcrf: str | None = None, threads: int = 4, pred_stride_s: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    model, dec, bfp = load_c3t()
    mode = dec.get("feature_mode", "current")
    m_raw = read_metrica()
    m_raw["pitch"] = np.array(m_raw["pitch"], np.float32)
    m = T.prepare_players(m_raw, mode, dedup=bool(dec.get("dedup", False)))
    ball = BF.fill_ball(m, bfp)
    rows, X, names = T.features(m, ball, mode=mode)
    if names != dec["features"]:
        raise SystemExit("feature names differ from the saved model (models/c3/decoder.json)")
    f = dict(rows=rows, X=X, prow=m["period"][rows], period=m["period"])
    if pathcrf:
        from pathcrf_adapter.score import combo
        p_dead = model["l1"].predict(X)
        q = pathcrf_q(m_raw, bfp, pathcrf, rows, threads, pred_stride_s)
        pred = combo(p_dead, q, f["prow"], rows, len(m["period"]), m["period"], model["C"], model["w"], model["beta"])
    else:
        pred = PL.predict(model, f)
    rules = tuple(dec.get("rules", ()))
    if rules:
        tags, ko = PL.metrica_tags(m["teams"][0])
        pred = PR.apply(pred, PL.context(m, ball, tags=tags, kickoff_tags=ko), rules)
    graded = m["period"] > 0
    return np.where(graded, LETTERS[np.clip(pred, 0, 2)], "X"), graded


def peak_ram_gb() -> float:
    """Peak resident memory of this process (Windows: PeakWorkingSetSize; elsewhere ru_maxrss)."""
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        c = PMC(); c.cb = ctypes.sizeof(PMC)
        k32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            raise OSError("GetProcessMemoryInfo failed")
        return c.PeakWorkingSetSize / 2**30
    except (AttributeError, OSError):
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def sha256(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT / "submission.csv"))
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--pathcrf", help="'published' or a PathCRF checkpoint: the combo (C3-T DEAD + PathCRF team)")
    ap.add_argument("--threads", type=int, default=4, help="torch CPU threads (--pathcrf)")
    ap.add_argument("--pred-stride", type=float, default=1.0, help="PathCRF window stride in seconds (--pathcrf)")
    a = ap.parse_args()
    t0 = time.time()
    labels, graded = predict_labels(a.pathcrf, a.threads, a.pred_stride)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="\n") as fh:
        fh.write("frame,label\n")
        fh.writelines(f"{i},{v}\n" for i, v in enumerate(labels))
    print(f"wrote {out}: {len(labels)} rows; graded label share " +
          ", ".join(f"{k} {100 * np.mean(labels[graded] == k):.1f} %" for k in ("H", "A", "D")))
    print(f"sha256 {sha256(out)}")
    print(f"runtime {time.time() - t0:.0f} s; peak RAM {peak_ram_gb():.2f} GB")
    if not a.no_validate:
        vr = subprocess.run([sys.executable, str(REF / "evaluation" / "validate_submission.py"), str(out)], capture_output=True, text=True)
        print(f"validator: {'PASS' if vr.returncode == 0 else 'FAIL'} (exit {vr.returncode})")
        return vr.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
