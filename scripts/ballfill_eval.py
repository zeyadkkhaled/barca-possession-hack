"""Step 2b on degraded PFF: train ball-from-players per fold (5 folds of the 30 learning matches), fill every match with its
out-of-fold model (used by the C3-T CV), report the ball position error by source against PFF's true ball, and fit the
final model on all 30 (models/bfp/, used for the hackathon match).

    .venv/Scripts/python.exe scripts/ballfill_eval.py [--workers 4] [--threads 1]

Prints aggregate errors only; writes reports/auto/ballfill_eval.json.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

from barca_possession import ballfill as BF
from barca_possession.paths import AUTO, MODELS, OUT, PFF_PACKAGE

DEG = OUT / "pff_degraded"      # matches to fill (--deg)
TRAIN = OUT / "pff_degraded"    # matches the ball-from-players model is trained on (--train-deg)
FILL = OUT / "ballfill"


def load(mid, d=None):
    with np.load((d or DEG) / f"{mid}.npz") as z:
        return {k: z[k] for k in z.files}


def job_samples(args):
    mid, seed = args
    m = load(mid, TRAIN)
    return mid, BF.bfp_samples(m, np.random.default_rng(seed + mid))


def _set_dirs(deg, train, fill):
    global DEG, TRAIN, FILL
    DEG, TRAIN, FILL = deg, train, fill


def job_fill(args):
    mid, models = args
    m = load(mid)
    mods = [lgb.Booster(model_str=s) for s in models] if models else None
    b = BF.fill_ball(m, mods)
    FILL.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FILL / f"{mid}.npz", **b)
    tx, ty = m["ball_true_x"], m["ball_true_y"]
    ok = np.isfinite(tx) & (m["period"] > 0)
    err = np.hypot(b["x"] - tx, b["y"] - ty)
    # "last seen" baseline for the estimated frames: smoothed ball forward-filled
    r, src = BF.smooth_stage(m["t"], m["period"], m["bx"], m["by"])
    pres = np.isfinite(r.x)
    li = np.maximum.accumulate(np.where(pres, np.arange(len(pres)), 0))
    base = np.hypot(r.x[li] - tx, r.y[li] - ty)
    out = {}
    for k, nm in enumerate(BF.SOURCE_NAMES):
        sel = ok & (b["source"] == k) & np.isfinite(err)
        out[nm] = err[sel].astype(np.float32)
    out["estimated_last_seen_baseline"] = base[ok & (b["source"] == BF.SRC_ESTIMATED) & np.isfinite(base)].astype(np.float32)
    share = {nm: float(((b["source"] == k) & (m["period"] > 0)).sum() / max((m["period"] > 0).sum(), 1))
             for k, nm in enumerate(BF.SOURCE_NAMES)}
    return mid, out, share


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deg", default=str(DEG), help="matches to fill")
    ap.add_argument("--train-deg", help="matches to train ball-from-players on (default: --deg)")
    ap.add_argument("--fill", default=str(FILL), help="output folder of the fills")
    ap.add_argument("--no-final", action="store_true", help="do not refit / save models/bfp (check runs)")
    ap.add_argument("--out-name", default="ballfill_eval.json")
    a = ap.parse_args()
    dirs = (Path(a.deg), Path(a.train_deg or a.deg), Path(a.fill))
    _set_dirs(*dirs)
    t0 = time.time()
    folds = json.loads((PFF_PACKAGE / "folds.json").read_text())["folds"]
    mids = sorted(int(p.stem) for p in DEG.glob("*.npz"))
    with multiprocessing.Pool(a.workers, initializer=_set_dirs, initargs=dirs) as pool:
        S = dict(pool.map(job_samples, [(m, a.seed) for m in mids]))
    print(f"samples: {sum(len(s[1]) for s in S.values())} ({time.time() - t0:.0f} s)", flush=True)
    jobs = []
    for k, fold in enumerate(folds):
        tr = [m for m in mids if m not in fold]
        mods = BF.fit_bfp([S[m] for m in tr], a.seed, a.threads)
        strs = [x.model_to_string() for x in mods]
        jobs += [(m, strs) for m in fold if m in S]
        print(f"fold {k} bfp trained ({time.time() - t0:.0f} s)", flush=True)
    with multiprocessing.Pool(a.workers, initializer=_set_dirs, initargs=dirs) as pool:
        res = pool.map(job_fill, jobs)
    errs = {k: np.concatenate([r[1][k] for r in res]) for k in res[0][1]}
    shares = {k: float(np.mean([r[2][k] for r in res])) for k in res[0][2]}
    table = {k: dict(n=int(len(v)), median_m=float(np.median(v)) if len(v) else None,
                     p90_m=float(np.percentile(v, 90)) if len(v) else None) for k, v in errs.items()}
    print("ball position error on degraded PFF (out-of-fold), metres:")
    for k, v in table.items():
        if v["n"]:
            print(f"  {k:30s} n={v['n']:8d}  median {v['median_m']:.2f}  p90 {v['p90_m']:.2f}")
    print("share of in-period frames by source: " + ", ".join(f"{k} {100 * v:.1f} %" for k, v in shares.items()))
    if not a.no_final:
        final = BF.fit_bfp([S[m] for m in mids], a.seed, a.threads)
        (MODELS / "bfp").mkdir(parents=True, exist_ok=True)
        for nm, mdl in zip(("bfp_dx.txt", "bfp_dy.txt"), final):
            mdl.save_model(str(MODELS / "bfp" / nm))
    AUTO.mkdir(parents=True, exist_ok=True)
    (AUTO / a.out_name).write_text(json.dumps(dict(error=table, source_share=shares, n_matches=len(mids)), indent=1))
    print(f"final bfp -> models/bfp/; fills -> outputs/ballfill/ ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
