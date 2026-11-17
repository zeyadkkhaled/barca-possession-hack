"""The ONE final run on the 16 locked PFF test matches: the frozen C3-T (models/c3 + models/bfp), exactly as
scripts/stress_eval.py scores the stress matches (labels v2, LOOSE -> last team; degraded with the training degradation,
seed 0; then clean). Mean per match + a per-match bootstrap 95 % interval for spellF1.

    .venv/Scripts/python.exe scripts/locked_test.py [--workers 2] [--variants degraded clean]

Prints aggregate scores only; writes reports/auto/locked_test.json. Each use must be logged in docs/dev_log/LOCKED_USED.md.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import time

import lightgbm as lgb
import numpy as np

from barca_possession import ballfill as BF
from barca_possession import c3t as T
from barca_possession.paths import AUTO, MODELS, OUT
from barca_train import degrade_pff as DP
from barca_train import pipeline as PL
from barca_train.stress import LOCKED_TEST, load_locked_test_match

import stress_eval as SE  # noqa: E402  (scripts/ is on sys.path when run as a script)


def job(args):
    mid, seed, variant = args
    model, dec = SE.load_model()
    bfp = [lgb.Booster(model_file=str(MODELS / "bfp" / nm)) for nm in ("bfp_dx.txt", "bfp_dy.txt")]
    stats = json.loads((OUT / "load_stats.json").read_text())
    deg = json.loads((AUTO / "degrade_stats.json").read_text())["degraded"]
    d = DP.load_package_match_dict(load_locked_test_match(mid, unlock=True))
    if variant == "clean":
        m = DP.clean_match(d)
    else:
        m = DP.degrade_match(d, stats, seed, dead_weight=deg["dead_weight"], extra=dec.get("extra_slots"))
    m = T.prepare_players(m, dec.get("feature_mode", "current"), dedup=bool(dec.get("dedup", False)))
    ball = BF.fill_ball(m, bfp)
    rows, X, names = T.features(m, ball, mode=dec.get("feature_mode", "current"))
    assert names == dec["features"], "feature names differ from the saved model"
    pred = PL.predict(model, dict(rows=rows, X=X, prow=m["period"][rows], period=m["period"]))
    s = PL.score_hda(m["label_had"], m["label_mask"], m["period"], pred)
    return mid, dict(spell=s["spell"], macro=s["macro"], raw=s["raw"],
                     extra=PL.extra_metrics(m["label_had"], m["label_mask"], m["period"], pred))


def bootstrap_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    means = x[rng.integers(0, len(x), size=(n, len(x)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", nargs="+", default=["degraded", "clean"])
    a = ap.parse_args()
    t0 = time.time()
    res = {}
    out = AUTO / "locked_test.json"
    for v in a.variants:
        with multiprocessing.Pool(a.workers) as pool:
            R = dict(pool.map(job, [(m, a.seed, v) for m in LOCKED_TEST]))
        rs = [R[m] for m in LOCKED_TEST]
        sp = [r["spell"] for r in rs]
        e = PL.merge_extra([r["extra"] for r in rs])
        res[v] = dict(spell_mean=float(np.mean(sp)), spell_ci95=bootstrap_ci(sp),
                      macro_mean=float(np.mean([r["macro"] for r in rs])),
                      short_spell_recall=e["short_spell_recall"], restart_team_acc=e["restart_team_acc"],
                      restart_matched_share=e["restart_matched_share"], pooled=PL.pooled([r["raw"] for r in rs]),
                      n_matches=len(rs), minutes=round((time.time() - t0) / 60, 1))
        r = res[v]
        print(f"locked test {v:8s}: spellF1 {r['spell_mean']:.4f} (95% CI {r['spell_ci95'][0]:.3f}-{r['spell_ci95'][1]:.3f}), "
              f"macroF1 {r['macro_mean']:.4f}, short-spell recall {r['short_spell_recall']:.3f}, restart team "
              f"{r['restart_team_acc']:.3f} (matched {r['restart_matched_share']:.3f}); {r['minutes']} min", flush=True)
        AUTO.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1))          # written after each variant, so a stop keeps what finished
    print(f"done in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
