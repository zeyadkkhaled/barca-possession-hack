"""Step 6: fit C3-T on all 30 degraded PFF learning matches, save it to models/c3/, predict the hackathon match, write
outputs/submission_c3.csv (X only on ungraded frames), run the official validator and score the public 5 minutes
(a sanity check only: never used to choose anything).

Post-rules: the ones whose solo spellF1 gain in the CV was positive (reports/auto/cv_c3t.json "rules_kept"); --rules
overrides. Deterministic: LightGBM deterministic=True, fixed seed, --threads 1 by default.

    .venv/Scripts/python.exe scripts/fit_predict.py [--workers 4] [--threads 1] [--rules ...]

Prints aggregate numbers only; writes reports/auto/fit_predict.json (aggregates).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import subprocess
import sys
import time

import lightgbm as lgb
import numpy as np

from barca_possession import ballfill as BF
from barca_possession import c3t as T
from barca_train import pipeline as PL
from barca_possession import postrules as PR
from barca_possession.load import read_metrica
from barca_possession.paths import AUTO, MODELS, OUT, REF

sys.path.insert(0, str(REF / "evaluation"))
import possession_metrics as pm  # noqa: E402

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import cv_c3t as CV  # noqa: E402
from cv_c3t import job_features  # noqa: E402

LETTERS = np.array(["H", "A", "D"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rules", nargs="*", choices=PR.RULES)
    ap.add_argument("--out", default=str(OUT / "submission_c3.csv"))
    ap.add_argument("--deg", default=str(CV.DEG), help="training matches (PFF in the Metrica form)")
    ap.add_argument("--fill", default=str(CV.FILL), help="their ball fills")
    ap.add_argument("--mode", default="current", choices=("current", "robust", "trim11"))
    ap.add_argument("--dedup", action="store_true")
    ap.add_argument("--degrade-stats", default="degrade_stats.json",
                    help="reports/auto file of the training data's degradation (its extra-slot config goes to decoder.json)")
    a = ap.parse_args()
    dirs = (__import__("pathlib").Path(a.deg), __import__("pathlib").Path(a.fill), (a.mode, a.dedup))
    CV._set_dirs(*dirs)
    xdeg = json.loads((AUTO / a.degrade_stats).read_text())["degraded"] if (AUTO / a.degrade_stats).exists() else {}
    t0 = time.time()
    cvf = AUTO / "cv_c3t.json"
    if a.rules is not None:
        rules = tuple(a.rules)
    elif cvf.exists():
        rules = tuple(json.loads(cvf.read_text())["rules_kept"])
    else:
        rules = ()
        print("WARNING: no reports/auto/cv_c3t.json -> no post-rules (run scripts/cv_c3t.py first)")
    print(f"post-rules: {list(rules)}")

    # 1. fit on all 30
    mids = sorted(int(p.stem) for p in CV.DEG.glob("*.npz"))
    with multiprocessing.Pool(a.workers, initializer=CV._set_dirs, initargs=dirs) as pool:
        F = dict(pool.map(job_features, [(m, a.seed) for m in mids]))
    names = F[mids[0]]["names"]
    print(f"training rows: {sum(len(f['rows']) for f in F.values())} from {len(mids)} matches ({time.time() - t0:.0f} s)", flush=True)
    model = PL.train(F, mids, a.seed, a.threads, log=lambda s: print(s, flush=True))
    d = MODELS / "c3"
    d.mkdir(parents=True, exist_ok=True)
    model["l1"].save_model(str(d / "layer1_dead.txt"))
    model["l2"].save_model(str(d / "layer2_home.txt"))
    (d / "decoder.json").write_text(json.dumps(dict(C=model["C"].tolist(), w=model["w"], beta=model["beta"], step=T.STEP,
                                                     features=names, rules=list(rules), seed=a.seed, threads=a.threads,
                                                     feature_mode=a.mode, dedup=a.dedup, train_dir=str(a.deg),
                                                     extra_slots=xdeg.get("extra")), indent=1))
    print(f"saved models/c3/ ({time.time() - t0:.0f} s)", flush=True)
    del F

    # 2. hackathon match
    m = read_metrica()
    m["pitch"] = np.array(m["pitch"], np.float32)
    m = T.prepare_players(m, a.mode, dedup=a.dedup)
    bfp = [lgb.Booster(model_file=str(MODELS / "bfp" / nm)) for nm in ("bfp_dx.txt", "bfp_dy.txt")]
    ball = BF.fill_ball(m, bfp)
    rows, X, nm2 = T.features(m, ball, mode=a.mode)
    assert nm2 == names, "feature names differ between training and the hackathon match"
    f = dict(rows=rows, X=X, prow=m["period"][rows], period=m["period"])
    pred = PL.predict(model, f)
    tags, ko = PL.metrica_tags(m["teams"][0])
    ctx = PL.context(m, ball, tags=tags, kickoff_tags=ko)
    pred = PR.apply(pred, ctx, rules) if rules else pred
    graded = m["period"] > 0
    labels = np.where(graded, LETTERS[np.clip(pred, 0, 2)], "X")
    out = a.out
    with open(out, "w", newline="\n") as fh:
        fh.write("frame,label\n")
        fh.writelines(f"{i},{v}\n" for i, v in enumerate(labels))

    # 3. validator (the official checks) and the public 5 minutes (sanity check only)
    vr = subprocess.run([sys.executable, str(REF / "evaluation" / "validate_submission.py"), str(out)], capture_output=True, text=True)
    sub = pm.read_label_csv(out)
    reason = pm.validate_submission(sub, len(labels))
    gt = pm.label_codes(pm.read_label_csv(REF / "evaluation" / "gt" / "BAR-ATH_possession_gt_public.csv"))
    r = pm.evaluate(gt, pm.label_codes(sub))
    codes = pm.label_codes(sub)
    runs = pm.valid_runs(np.where(graded, 0, pm.X).astype(np.int8))          # the two graded halves
    stats = pm.possession_stats(codes, runs)
    rep = dict(rules=list(rules), validator_exit=vr.returncode, validator_ok=reason is None and vr.returncode == 0,
               validator_reason=reason, rows=int(len(sub)),
               label_share_graded={k: float(np.mean(labels[graded] == k)) for k in ("H", "A", "D", "X")},
               n_pred_spells=int(stats["n_spells"]), mean_spell_s=float(stats["mean_duration_s"]),
               public=dict(spell_f1=r["spell_f1_mean"], macro_f1=r["macro_f1"], n_gt_spells=r["n_gt_spells"],
                           n_pred_spells=r["n_pred_spells"]),
               baseline_smooth50=dict(public_spell_f1=0.3256, public_macro_f1=0.6207, full_spell_f1=0.2433),
               tags=dict(set_pieces=len(tags), team_H=sum(t["team"] == 0 for t in tags), team_A=sum(t["team"] == 1 for t in tags),
                         team_NA=sum(t["team"] is None for t in tags), kickoff_halves=sorted(ko)),
               ball_source_share={nm: float(np.mean(ball["source"][graded] == k)) for k, nm in enumerate(BF.SOURCE_NAMES)},
               decoder=dict(w=model["w"], beta=model["beta"]), feature_mode=a.mode, dedup=a.dedup, train_dir=str(a.deg))
    print(f"validator: {'PASS' if rep['validator_ok'] else 'FAIL'} (exit {vr.returncode}, reason {reason}), rows {rep['rows']}")
    print(f"graded label share: " + ", ".join(f"{k} {100 * v:.1f} %" for k, v in rep["label_share_graded"].items()))
    print(f"predicted possessions (full match): {rep['n_pred_spells']}, mean {rep['mean_spell_s']:.1f} s")
    print(f"Smart Tagging set pieces: {rep['tags']}")
    print(f"public 5 min (sanity check only, not used for any choice): spellF1 {r['spell_f1_mean']:.4f}, macroF1 {r['macro_f1']:.4f} "
          f"(baseline_smooth50 on the same window: 0.3256 / 0.6207; its full-match floor 0.2433)")
    AUTO.mkdir(parents=True, exist_ok=True)
    (AUTO / "fit_predict.json").write_text(json.dumps(rep, indent=1))
    print(f"wrote {out} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
