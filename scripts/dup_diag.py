"""Step 1c diagnostic: what are the extra slots on the match's frames where a team has more than 11? Aggregates only.

Per graded frame with a team over 11 (that team only) vs frames with exactly 11: slots over 11, nearest same-team pair
distance and its velocity difference, how many slots are off the pitch, and how long the extra slots' presence runs are.

    .venv/Scripts/python.exe scripts/dup_diag.py

Prints percentages / quantiles only; writes reports/auto/dup_diag.json.
"""
from __future__ import annotations

import json

import numpy as np

from barca_possession.load import read_metrica, segment_length, slot_velocity
from barca_possession.paths import AUTO, FPS, PITCH


def q(v, ps=(5, 25, 50, 75, 95)):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return {f"p{p}": round(float(np.percentile(v, p)), 2) for p in ps} if len(v) else None


def main() -> int:
    m = read_metrica()
    px, py, ts, per = m["px"], m["py"], m["team_sign"], m["period"]
    pres = np.isfinite(px)
    vx, vy = slot_velocity(px, per), slot_velocity(py, per)
    seg = segment_length(pres) / FPS
    L, W = PITCH
    off = (np.abs(px) > L / 2 + 1) | (np.abs(py) > W / 2 + 1)
    out = {}
    for name, cond in (("over11", lambda c: c > 11), ("exactly11", lambda c: c == 11)):
        n_extra, dmin, dvmin, n_off, seg_short, seg_all, n_fr = [], [], [], [], [], [], 0
        for side in (1.0, -1.0):
            cols = np.flatnonzero(ts == side)
            cnt = pres[:, cols].sum(1)
            fr = np.flatnonzero((per > 0) & cond(cnt))
            rng = np.random.default_rng(0)
            fr_s = fr if len(fr) <= 20000 else rng.choice(fr, 20000, replace=False)
            n_fr += len(fr)
            for f in fr_s:
                c = cols[pres[f, cols]]
                n_extra.append(len(c) - 11)
                i, j = np.triu_indices(len(c), 1)
                d = np.hypot(px[f, c[i]] - px[f, c[j]], py[f, c[i]] - py[f, c[j]])
                k = int(np.argmin(d))
                dmin.append(d[k])
                dvmin.append(np.hypot(vx[f, c[i[k]]] - vx[f, c[j[k]]], vy[f, c[i[k]]] - vy[f, c[j[k]]]))
                n_off.append(int(off[f, c].sum()))
                s = np.sort(seg[f, c])
                seg_short.append(s[0])
                seg_all.extend(s.tolist())
        dmin, dvmin = np.array(dmin), np.array(dvmin)
        out[name] = dict(
            n_team_frames=int(n_fr),
            slots_over_11=dict(zip(*[x.tolist() for x in np.unique(n_extra, return_counts=True)])) if name == "over11" else None,
            nearest_pair_m=q(dmin), nearest_pair_dv_mps=q(dvmin),
            share_nearest_lt_0_5m=float((dmin < 0.5).mean()), share_nearest_lt_1m=float((dmin < 1).mean()),
            share_nearest_lt_2m=float((dmin < 2).mean()),
            share_rule_pair=float(((dmin < 0.5) & (dvmin < 1)).mean()),
            share_nearest_lt_1m_dv_lt_2=float(((dmin < 1) & (dvmin < 2)).mean()),
            share_with_slot_off_pitch=float((np.array(n_off) > 0).mean()),
            shortest_run_s=q(seg_short), all_runs_s=q(seg_all))
    for k, v in out.items():
        print(f"== {k}: {v['n_team_frames']} team-frames")
        for kk, vv in v.items():
            if kk != "n_team_frames":
                print(f"  {kk}: {vv}")
    AUTO.mkdir(parents=True, exist_ok=True)
    (AUTO / "dup_diag.json").write_text(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
