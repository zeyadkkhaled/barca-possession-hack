"""Score PathCRF against C3-T on degraded + relinked PFF held-out matches (H/A/D, our scorer = the official one without X).

Variants (all with C3-T's P(D), transition counts and Viterbi decoder, so only the team probability changes):
  c3t        C3-T alone (its layer-2 P(H | alive))
  pathcrf    PathCRF's team probability q_home (sender mass on home lanes / on all player lanes)
  blend      the mean of the two
Metrics: spellF1 and macroF1 (mean per match, and pooled), restart-team accuracy (matched share), short-spell recall.

    python -m pathcrf_adapter.score --data BUILT --c3t RUNS/c3t_fold0 --preds RUN/preds --out RUNS/score_<name>.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from barca_possession import c3t as T
from barca_train import pipeline as PL

VARIANTS = ("c3t", "pathcrf", "blend")


def score(data, c3t_dir, preds_dir) -> dict:
    dec = json.loads((Path(c3t_dir) / "decoder.json").read_text())
    C = np.array(dec["C"])
    per_v = {v: [] for v in VARIANTS}
    raws = {v: [] for v in VARIANTS}
    extra = {v: [] for v in VARIANTS}
    missing = []
    for m in dec["held"]:
        with np.load(Path(data) / "c3t" / f"{m}.npz") as z:
            gt, gmask, period = z["gt"], z["gmask"], z["period"]
        with np.load(Path(c3t_dir) / f"{m}.npz") as z:
            rows, p_dead, q_c3, prow = z["rows"], z["p_dead"], z["q_home"], z["prow"]
        pp = Path(preds_dir) / f"{m}.npz"
        q_pc = None
        if pp.exists():
            with np.load(pp) as z:
                fr, qh = z["frames"], z["q_home"]
            pos = np.searchsorted(fr, rows)
            assert (fr[np.clip(pos, 0, len(fr) - 1)] == rows).all(), "PathCRF steps and C3-T rows differ"
            q_pc = qh[pos].astype(np.float64)
        else:
            missing.append(m)
        qs = dict(c3t=q_c3, pathcrf=q_pc, blend=None if q_pc is None else 0.5 * (q_c3 + q_pc))
        for v, q in qs.items():
            if q is None:
                continue
            path = T.decode(p_dead, q, prow, C, dec["w"], dec["beta"])
            pred = T.rows_to_frames(path, rows, len(period), period)
            s = PL.score_hda(gt, gmask, period, pred)
            per_v[v].append((s["spell"], s["macro"])); raws[v].append(s["raw"])
            extra[v].append(PL.extra_metrics(gt, gmask, period, pred))
    res = {}
    for v in VARIANTS:
        if len(per_v[v]) != len(dec["held"]):
            continue
        a = np.array(per_v[v])
        res[v] = dict(spell=float(a[:, 0].mean()), macro=float(a[:, 1].mean()), pooled=PL.pooled(raws[v]),
                      extra=PL.merge_extra(extra[v]), n_matches=len(a))
    res["_meta"] = dict(held=dec["held"], fold=dec["fold"], preds=str(preds_dir), missing_preds=missing)
    return res


def table(res: dict, title: str) -> str:
    L = [f"### {title}", "", "| variant | spellF1 | macroF1 | spellF1 pooled | restart-team acc (matched) | short-spell recall |",
         "|---|---|---|---|---|---|"]
    for v in VARIANTS:
        if v in res:
            r, e = res[v], res[v]["extra"]
            L.append(f"| {v} | {r['spell']:.4f} | {r['macro']:.4f} | {r['pooled']['spell']:.4f} | "
                     f"{e['restart_team_acc']:.3f} ({e['restart_matched_share']:.3f}) | {e['short_spell_recall']:.3f} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--c3t", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    a = ap.parse_args(argv)
    res = score(a.data, a.c3t, a.preds)
    Path(a.out).write_text(json.dumps(res, indent=1))
    md = table(res, a.title or Path(a.out).stem)
    Path(a.out).with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
