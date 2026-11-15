"""Compare our M1 H/A/D scorer (barca_labels.scoring.score_had / merge_had) with the official one
(reference/more-than-a-hack-2027/evaluation/possession_metrics.py) on the same label arrays.

Inputs: the public 5-minute ground truth and the four committed baselines (all in the public reference repo), plus
random synthetic H/A/D sequences over two halves. No match data is read.

H / A / D map to our states as CONTROL of team 0 (home) / CONTROL of team 1 / DEAD; graded frames (GT != X) get
period 1 or 2 (by run) and score_mask True; everything else period 0. Predictions here contain no X (ours has no X).

    .venv/Scripts/python.exe scripts/check_scorer_parity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "reference" / "more-than-a-hack-2027" / "evaluation"
sys.path.insert(0, str(REF))
import possession_metrics as pm  # noqa: E402

from barca_labels.labeller import CONTROL, DEAD  # noqa: E402
from barca_labels.scoring import merge_had, score_had  # noqa: E402


def ours(gt: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    period = np.zeros(len(gt), np.int64)
    for k, (a, b) in enumerate(pm.valid_runs(gt), start=1):
        period[a:b] = k

    def arrays(codes):
        state = np.where(codes == pm.D, DEAD, CONTROL)
        team = np.where(codes == pm.H, 0, np.where(codes == pm.A, 1, -1))
        return state, team

    st, tm = arrays(gt)
    ps, pt = arrays(pred)
    label = dict(state=st, team_id=tm, score_mask=gt != pm.X, period=period)
    m = merge_had([score_had(label, ps, pt, home_team=0)])
    return m["macro_f1"], m["spell_f1"]


def official(gt: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    r = pm.evaluate(gt, pred)
    return r["macro_f1"], r["spell_f1_mean"]


def synthetic(rng: np.random.Generator, n: int = 40000) -> tuple[np.ndarray, np.ndarray]:
    def seq(mean_len):
        out, i = [], 0
        while i < n:
            L = int(rng.exponential(mean_len)) + 1
            out.append(np.full(L, rng.choice([pm.H, pm.A, pm.D], p=[0.4, 0.35, 0.25]), np.int8))
            i += L
        return np.concatenate(out)[:n]

    gt = seq(300)
    if rng.random() < 0.5:
        pred = seq(150)                  # unrelated prediction
    else:                                # near-correct prediction: shifted truth plus short wrong blips
        pred = np.roll(gt, int(rng.integers(-30, 30)))
        for s in rng.integers(0, n - 40, 60):
            pred[s:s + int(rng.integers(5, 40))] = rng.integers(0, 3)
    pred = pred.copy()
    flip = rng.random(n) < 0.0005        # rare single-frame noise
    pred = np.where(flip, rng.integers(0, 3, n), pred).astype(np.int8)
    gt[:1000] = pm.X                     # pre-match, halftime, post-match are ungraded
    gt[18000:21000] = pm.X
    gt[-800:] = pm.X
    return gt, pred


def main() -> int:
    gt = pm.label_codes(pm.read_label_csv(REF / "gt" / "BAR-ATH_possession_gt_public.csv"))
    rows = []
    for p in sorted((REF / "results").glob("baseline_*.csv")):
        pred = pm.label_codes(pm.read_label_csv(p))
        rows.append((p.stem, official(gt, pred), ours(gt, pred)))
    rng = np.random.default_rng(0)
    for i in range(20):
        g, q = synthetic(rng)
        rows.append((f"synthetic_{i:02d}", official(g, q), ours(g, q)))
    worst_m = worst_s = 0.0
    print(f"{'case':22s} {'macroF1 off':>11s} {'ours':>8s} {'spellF1 off':>11s} {'ours':>8s}")
    for name, (om, os_), (um, us) in rows:
        worst_m, worst_s = max(worst_m, abs(om - um)), max(worst_s, abs(os_ - us))
        print(f"{name:22s} {om:11.6f} {um:8.6f} {os_:11.6f} {us:8.6f}")
    print(f"max |diff|: macroF1 {worst_m:.2e}, spellF1 {worst_s:.2e} ({len(rows)} cases, no X in predictions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
