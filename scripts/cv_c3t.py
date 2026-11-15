"""Step 5: 5-fold CV of C3-T on the 30 degraded PFF learning matches (folds.json of the M1 package).

Per fold: train on 4 folds (layer 1, layer 2 with stacking, decoder w / beta from inner out-of-fold predictions),
predict the held-out fold, score per match (spellF1, macroF1 with our scorer = the official one without X), without
post-rules, with each post-rule alone and with all of them. The SET PIECES rule uses simulated tags on PFF
(postrules.simulate_tags), so its effect here is only indicative.

    .venv/Scripts/python.exe scripts/cv_c3t.py [--workers 4] [--threads 1] [--folds 0 1 2 3 4]

Prints aggregate scores only; writes reports/auto/cv_c3t.json (aggregates).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from pathlib import Path

import numpy as np

from barca_possession import c3t as T
from barca_train import pipeline as PL
from barca_possession import postrules as PR
from barca_possession.paths import AUTO, OUT, PFF_PACKAGE

DEG = OUT / "pff_degraded"
FILL = OUT / "ballfill"
VARIANTS = {"none": (), **{f"only {r}": (r,) for r in PR.RULES}, "all": PR.RULES}


PREP = ("current", False)       # (feature mode, dedup)


def _set_dirs(deg, fill, prep=("current", False)):
    global DEG, FILL, PREP
    DEG, FILL, PREP = deg, fill, prep


def load_match(mid):
    with np.load(DEG / f"{mid}.npz") as z:
        m = T.prepare_players({k: z[k] for k in z.files}, PREP[0], dedup=PREP[1])
    with np.load(FILL / f"{mid}.npz") as z:
        b = {k: z[k] for k in z.files}
    return m, b


def job_features(args):
    mid, seed = args
    m, b = load_match(mid)
    rows, X, names = T.features(m, b, mode=PREP[0])
    gt, gmask = m["label_had"], m["label_mask"]
    tags = PR.simulate_tags(gt, m["period"], m["ball_true_y"], float(m["pitch"][1]) / 2, np.random.default_rng(seed + mid))
    ctx = PL.context(m, b, tags=tags)
    return mid, dict(rows=rows, X=X, names=names, lab=gt[rows], mask=gmask[rows], prow=m["period"][rows], gt=gt,
                     gmask=gmask, period=m["period"], ctx=ctx)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, nargs="*")
    ap.add_argument("--out-name", default="cv_c3t.json")
    ap.add_argument("--deg", default=str(DEG), help="PFF matches in the Metrica form")
    ap.add_argument("--fill", default=str(FILL), help="their ball fills")
    ap.add_argument("--mode", default="current", choices=("current", "robust", "trim11"),
                    help="features: current / robust (option A) / trim11 (option B)")
    ap.add_argument("--dedup", action="store_true", help="apply the duplicate rule (load.remove_duplicates)")
    a = ap.parse_args()
    dirs = (Path(a.deg), Path(a.fill), (a.mode, a.dedup))
    _set_dirs(*dirs)
    t0 = time.time()
    folds = json.loads((PFF_PACKAGE / "folds.json").read_text())["folds"]
    mids = sorted(int(p.stem) for p in DEG.glob("*.npz"))
    with multiprocessing.Pool(a.workers, initializer=_set_dirs, initargs=dirs) as pool:
        F = dict(pool.map(job_features, [(m, a.seed) for m in mids]))
    names = F[mids[0]]["names"]
    print(f"features: {len(names)} per row, {sum(len(f['rows']) for f in F.values())} rows ({time.time() - t0:.0f} s)",
          flush=True)
    res = {"folds": [], "variants": list(VARIANTS)}
    per_match = {v: [] for v in VARIANTS}
    extra_all = []
    raws = {v: [] for v in VARIANTS}
    for k in (a.folds if a.folds is not None else range(len(folds))):
        test = [m for m in folds[k] if m in F]
        train = [m for m in mids if m not in test]
        print(f"fold {k}: train {len(train)}, test {len(test)}", flush=True)
        model = PL.train(F, train, a.seed, a.threads, log=lambda s: print(s, flush=True))
        fr = dict(fold=k, w=model["w"], beta=model["beta"], inner_spell=model["inner_spell"], n_test=len(test))
        sc = {v: [] for v in VARIANTS}
        rw = {v: [] for v in VARIANTS}
        ex = []
        for m in test:
            f = F[m]
            pred = PL.predict(model, f)
            ex.append(PL.extra_metrics(f["gt"], f["gmask"], f["period"], pred))
            for v, rules in VARIANTS.items():
                p = PR.apply(pred, f["ctx"], rules) if rules else pred
                s = PL.score_hda(f["gt"], f["gmask"], f["period"], p)
                sc[v].append((s["spell"], s["macro"])); rw[v].append(s["raw"])
        for v in VARIANTS:
            arr = np.array(sc[v])
            fr[v] = dict(spell=float(arr[:, 0].mean()), macro=float(arr[:, 1].mean()), pooled=PL.pooled(rw[v]))
            per_match[v] += sc[v]; raws[v] += rw[v]
        fr["extra"] = PL.merge_extra(ex)
        extra_all += ex
        res["folds"].append(fr)
        print(f"  fold {k}: spellF1 {fr['none']['spell']:.4f} / macroF1 {fr['none']['macro']:.4f} without post-rules; "
              f"{fr['all']['spell']:.4f} / {fr['all']['macro']:.4f} with all ({time.time() - t0:.0f} s)", flush=True)
    res["mean"] = {v: dict(spell=float(np.mean([x[0] for x in per_match[v]])), macro=float(np.mean([x[1] for x in per_match[v]])),
                           pooled=PL.pooled(raws[v]), n_matches=len(per_match[v])) for v in VARIANTS}
    res["extra"] = PL.merge_extra(extra_all)
    res["mode"], res["dedup"], res["deg"] = a.mode, a.dedup, str(a.deg)
    base = res["mean"]["none"]
    res["rule_delta"] = {r: dict(spell=res["mean"][f"only {r}"]["spell"] - base["spell"],
                                 macro=res["mean"][f"only {r}"]["macro"] - base["macro"]) for r in PR.RULES}
    res["rules_kept"] = [r for r in PR.RULES if res["rule_delta"][r]["spell"] > 0]
    res["n_features"] = len(names)
    res["threads"], res["seed"] = a.threads, a.seed
    print("\nfold | spellF1 none | macroF1 none | spellF1 all | macroF1 all")
    for fr in res["folds"]:
        print(f"{fr['fold']:4d} | {fr['none']['spell']:.4f} | {fr['none']['macro']:.4f} | {fr['all']['spell']:.4f} | {fr['all']['macro']:.4f}")
    print(f"mean | {base['spell']:.4f} | {base['macro']:.4f} | {res['mean']['all']['spell']:.4f} | {res['mean']['all']['macro']:.4f}")
    print("\npost-rule alone vs none (mean over matches):")
    for r, dv in res["rule_delta"].items():
        print(f"  {r:14s} spellF1 {dv['spell']:+.4f}  macroF1 {dv['macro']:+.4f}")
    print(f"rules kept for the submission (spellF1 gain > 0): {res['rules_kept']}")
    e = res["extra"]
    print(f"mode {a.mode}, dedup {a.dedup}: restart-team accuracy {e['restart_team_acc']:.3f} (matched {e['restart_matched_share']:.3f}), "
          f"short-spell (<2 s) recall {e['short_spell_recall']:.3f}")
    AUTO.mkdir(parents=True, exist_ok=True)
    (AUTO / a.out_name).write_text(json.dumps(res, indent=1))
    print(f"done in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
