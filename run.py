"""One command: the hackathon submission from the raw match data and the saved C3-T model (no training).

    python run.py [--out outputs/submission.csv] [--no-validate]

Steps: read the match (Metrica EPTS tracking + metadata, official readers) -> player-slot clean-up as the model was trained
(models/c3/decoder.json: feature_mode, dedup) -> ball: smoother + ball-from-players (models/bfp) -> C3-T features ->
layer 1 P(dead), layer 2 P(home | alive), 3-state Viterbi (models/c3) -> post-rules saved with the model (SET PIECES tags
from Smart Tagging) -> one label per frame: H / A / D on the graded halves, X elsewhere -> official validator.
Deterministic: same inputs and models give the same file (LightGBM prediction is deterministic on CPU).
Prints only the label shares, the file's sha256 and the validator result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
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


def predict_labels() -> tuple[np.ndarray, np.ndarray]:
    model, dec, bfp = load_c3t()
    mode = dec.get("feature_mode", "current")
    m = read_metrica()
    m["pitch"] = np.array(m["pitch"], np.float32)
    m = T.prepare_players(m, mode, dedup=bool(dec.get("dedup", False)))
    ball = BF.fill_ball(m, bfp)
    rows, X, names = T.features(m, ball, mode=mode)
    if names != dec["features"]:
        raise SystemExit("feature names differ from the saved model (models/c3/decoder.json)")
    pred = PL.predict(model, dict(rows=rows, X=X, prow=m["period"][rows], period=m["period"]))
    rules = tuple(dec.get("rules", ()))
    if rules:
        tags, ko = PL.metrica_tags(m["teams"][0])
        pred = PR.apply(pred, PL.context(m, ball, tags=tags, kickoff_tags=ko), rules)
    graded = m["period"] > 0
    return np.where(graded, LETTERS[np.clip(pred, 0, 2)], "X"), graded


def sha256(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT / "submission.csv"))
    ap.add_argument("--no-validate", action="store_true")
    a = ap.parse_args()
    labels, graded = predict_labels()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="\n") as fh:
        fh.write("frame,label\n")
        fh.writelines(f"{i},{v}\n" for i, v in enumerate(labels))
    print(f"wrote {out}: {len(labels)} rows; graded label share " +
          ", ".join(f"{k} {100 * np.mean(labels[graded] == k):.1f} %" for k in ("H", "A", "D")))
    print(f"sha256 {sha256(out)}")
    if not a.no_validate:
        vr = subprocess.run([sys.executable, str(REF / "evaluation" / "validate_submission.py"), str(out)], capture_output=True, text=True)
        print(f"validator: {'PASS' if vr.returncode == 0 else 'FAIL'} (exit {vr.returncode})")
        return vr.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
