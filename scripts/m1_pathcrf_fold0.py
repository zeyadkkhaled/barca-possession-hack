"""Quick check: M1's PathCRF fold-0 predictions (trained on clean PFF in M1's cycle 1) vs C3-T, on the same 6 held-out
matches (fold 0: 3813 3815 3818 3819 3849 3858), CLEAN input, our H/A/D scorer (LOOSE -> last team). No PathCRF training.

- C3-T: trained here on the clean 24 other learning matches (folds 1-4; outputs/pff_clean + outputs/ballfill_clean,
  feature mode current, no duplicate rule, seed 0, 1 thread: the step-1b clean CV's fold 0).
- PathCRF has no dead-ball class. Its team probability q = P(sender is home) / P(sender is a player) (from p_src over
  the package's player columns) replaces C3-T's layer-2 P(H | alive) inside C3-T's decoder, with C3-T's P(D), transition
  counts, w and beta unchanged. Each 25 fps row takes the PathCRF step (29.97 fps / 6) nearest in time in the same period.
- blend: the mean of the two team probabilities.
Prints aggregates only; writes reports/auto/m1_pathcrf_fold0.json.

    .venv/Scripts/python.exe scripts/m1_pathcrf_fold0.py [--preds "D:/m1_tmp/hpc/results cucle1/preds"]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from barca_possession import c3t as T
from barca_possession.paths import AUTO, OUT, PFF_PACKAGE
from barca_train import pipeline as PL

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv_c3t as CV  # noqa: E402

VARIANTS = ("c3t", "pathcrf", "blend")


def pathcrf_q(mid, preds_dir, t25, per25, rows):
    """PathCRF team probability at C3-T's rows, and the worst time gap of the row -> step match (s)."""
    with np.load(PFF_PACKAGE / "matches" / f"{mid}.npz") as z:
        t0, side = z["t"].astype(np.float64), z["player_side"]
    with np.load(Path(preds_dir) / f"{mid}.npz") as z:
        steps, sper, p_home, p_src = z["steps"], z["period"], z["p_home"].astype(np.float64), z["p_src"].astype(np.float64)
    P = len(side)
    p_away = p_src[:, :P][:, side == 1].sum(1)
    q = p_home / np.maximum(p_home + p_away, 1e-6)
    ts = t0[steps]
    out = np.full(len(rows), 0.5)
    gap = 0.0
    for p in np.unique(per25[rows]):
        r = np.flatnonzero(per25[rows] == p)
        s = np.flatnonzero(sper == p)
        j = np.clip(np.searchsorted(ts[s], t25[rows[r]]), 1, len(s) - 1)
        j = np.where(np.abs(ts[s][j - 1] - t25[rows[r]]) <= np.abs(ts[s][j] - t25[rows[r]]), j - 1, j)
        out[r] = q[s][j]
        gap = max(gap, float(np.abs(ts[s][j] - t25[rows[r]]).max()))
    return out, gap


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", default="D:/m1_tmp/hpc/results cucle1/preds")
    a = ap.parse_args()
    t0 = time.time()
    CV._set_dirs(OUT / "pff_clean", OUT / "ballfill_clean", ("current", False))
    folds = json.loads((PFF_PACKAGE / "folds.json").read_text())["folds"]
    held = sorted(folds[0])
    mids = sorted(int(p.stem) for p in (OUT / "pff_clean").glob("*.npz"))
    F = dict(map(CV.job_features, [(m, 0) for m in mids]))
    print(f"features for {len(mids)} clean matches ({time.time() - t0:.0f} s)", flush=True)
    train = [m for m in mids if m not in held]
    model = PL.train(F, train, 0, 1, log=lambda s: print(s, flush=True))
    per_v = {v: [] for v in VARIANTS}
    ex = {v: [] for v in VARIANTS}
    gaps = []
    for m in held:
        f = F[m]
        with np.load(OUT / "pff_clean" / f"{m}.npz") as z:
            t25 = z["t"].astype(np.float64)
        p_dead = model["l1"].predict(f["X"])
        q_c3 = model["l2"].predict(T.l2_inputs(f["X"], p_dead, f["prow"]))
        q_pc, gap = pathcrf_q(m, a.preds, t25, f["period"], f["rows"])
        gaps.append(gap)
        for v, q in (("c3t", q_c3), ("pathcrf", q_pc), ("blend", 0.5 * (q_c3 + q_pc))):
            path = T.decode(p_dead, q, f["prow"], model["C"], model["w"], model["beta"])
            pred = T.rows_to_frames(path, f["rows"], len(f["period"]), f["period"])
            s = PL.score_hda(f["gt"], f["gmask"], f["period"], pred)
            per_v[v].append((s["spell"], s["macro"]))
            ex[v].append(PL.extra_metrics(f["gt"], f["gmask"], f["period"], pred))
    res = dict(held=held, decoder=dict(w=model["w"], beta=model["beta"]), max_row_step_gap_s=max(gaps), preds=a.preds)
    print(f"\nfold 0, clean input, 6 matches, mean per match (row -> PathCRF step gap <= {max(gaps):.3f} s)")
    print("variant  | spellF1 | macroF1 | short-spell recall | restart-team acc (matched share)")
    for v in VARIANTS:
        arr = np.array(per_v[v])
        e = PL.merge_extra(ex[v])
        res[v] = dict(spell=float(arr[:, 0].mean()), macro=float(arr[:, 1].mean()), extra=e)
        print(f"{v:8s} | {res[v]['spell']:.4f}  | {res[v]['macro']:.4f}  | {e['short_spell_recall']:.3f}              | "
              f"{e['restart_team_acc']:.3f} ({e['restart_matched_share']:.3f})")
    AUTO.mkdir(parents=True, exist_ok=True)
    (AUTO / "m1_pathcrf_fold0.json").write_text(json.dumps(res, indent=1))
    print(f"done in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
