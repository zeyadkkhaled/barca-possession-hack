"""Score the saved C3-T (models/c3 + models/bfp) on the 6 M1 stress-eval matches, clean and degraded (the same
degradation as training: load_stats.json rates, the calibrated dead_weight, extra-slot simulation if the model was
trained with it). Never the 16 locked test matches (barca_train.stress refuses them).

    .venv/Scripts/python.exe scripts/stress_eval.py [--workers 3] [--out-name stress_eval.json]

Prints aggregate scores only; writes reports/auto/<out-name>.
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
from barca_train import degrade_pff as DP
from barca_train import pipeline as PL
from barca_train.stress import STRESS_EVAL, load_stress_match
from barca_possession.paths import AUTO, MODELS, OUT


def load_model():
    d = MODELS / "c3"
    dec = json.loads((d / "decoder.json").read_text())
    return dict(l1=lgb.Booster(model_file=str(d / "layer1_dead.txt")), l2=lgb.Booster(model_file=str(d / "layer2_home.txt")),
                C=np.array(dec["C"]), w=dec["w"], beta=dec["beta"]), dec


def job(args):
    mid, seed = args
    model, dec = load_model()
    bfp = [lgb.Booster(model_file=str(MODELS / "bfp" / nm)) for nm in ("bfp_dx.txt", "bfp_dy.txt")]
    stats = json.loads((OUT / "load_stats.json").read_text())
    deg = json.loads((AUTO / "degrade_stats.json").read_text())["degraded"]     # calibrated dead_weight (step 1)
    d = DP.load_package_match_dict(load_stress_match(mid))
    out = {}
    for variant in ("clean", "degraded"):
        if variant == "clean":
            m = DP.clean_match(d)
        else:
            m = DP.degrade_match(d, stats, seed, dead_weight=deg["dead_weight"], extra=dec.get("extra_slots"))
        m = T.prepare_players(m, dec.get("feature_mode", "current"), dedup=bool(dec.get("dedup", False)))
        ball = BF.fill_ball(m, bfp)
        rows, X, names = T.features(m, ball, mode=dec.get("feature_mode", "current"))
        assert names == dec["features"], "feature names differ from the saved model"
        f = dict(rows=rows, X=X, prow=m["period"][rows], period=m["period"])
        pred = PL.predict(model, f)
        s = PL.score_hda(m["label_had"], m["label_mask"], m["period"], pred)
        out[variant] = dict(spell=s["spell"], macro=s["macro"], raw=s["raw"],
                            extra=PL.extra_metrics(m["label_had"], m["label_mask"], m["period"], pred))
    return mid, out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-name", default="stress_eval.json")
    a = ap.parse_args()
    t0 = time.time()
    with multiprocessing.Pool(a.workers) as pool:
        R = dict(pool.map(job, [(m, a.seed) for m in STRESS_EVAL]))
    res = {}
    for v in ("clean", "degraded"):
        rs = [R[m][v] for m in STRESS_EVAL]
        res[v] = dict(spell_mean=float(np.mean([r["spell"] for r in rs])), macro_mean=float(np.mean([r["macro"] for r in rs])),
                      pooled=PL.pooled([r["raw"] for r in rs]), extra=PL.merge_extra([r["extra"] for r in rs]),
                      n_matches=len(rs))
        e = res[v]["extra"]
        print(f"stress-eval {v:8s}: spellF1 {res[v]['spell_mean']:.4f} / macroF1 {res[v]['macro_mean']:.4f} (mean per match), "
              f"pooled {res[v]['pooled']['spell']:.4f} / {res[v]['pooled']['macro']:.4f}; restart team {e['restart_team_acc']:.3f} "
              f"(matched {e['restart_matched_share']:.3f}); short-spell recall {e['short_spell_recall']:.3f}")
    _, dec = load_model()
    res["model"] = dict(feature_mode=dec.get("feature_mode", "current"), extra_slots=dec.get("extra_slots"))
    AUTO.mkdir(parents=True, exist_ok=True)
    (AUTO / a.out_name).write_text(json.dumps(res, indent=1))
    print(f"done in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
