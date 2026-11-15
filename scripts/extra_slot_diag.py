"""Step 1c-v2, task 1: what is the extra slot on the match's frames where a team has more than 11? Aggregates only.

The extra slot in such a frame is the team's most recently born slot (its presence run started last). Per graded frame
with a team over 11:
- the other team's slot count (<=10 / 11 / >11) and the total (<=22 / 23 / >23), and the classification
  (a) extra person: total 23 with the other team at 11; (b) wrong team label: total 22 with the other team at 10; other;
- per extra-slot run: share of its frames within 15 m of the play centre (the ball where seen, else the centroid of all
  slots), its mean speed, its length, all vs regular slot runs;
- team label: fixed per slot in the metadata XML, so a flip can only show as a hand-over between slots: does the extra
  run start within 2 m / 1 s of where a slot of the OTHER team (or of the same team) just ended?
Writes reports/auto/extra_slot_diag.json (the run-length histogram is what degrade_pff copies).

    .venv/Scripts/python.exe scripts/extra_slot_diag.py
"""
from __future__ import annotations

import json

import numpy as np

from barca_possession.load import GAP_BIN_NAMES, GAP_EDGES_S, read_metrica, runs, slot_velocity
from barca_possession.paths import AUTO, FPS

NEAR_PLAY_M = 15.0
HANDOVER_M, HANDOVER_S = 2.0, 1.0


def q(v, ps=(10, 25, 50, 75, 90)):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return {f"p{p}": round(float(np.percentile(v, p)), 2) for p in ps} if len(v) else None


def main() -> int:
    m = read_metrica()
    px, py, ts, per = m["px"], m["py"], m["team_sign"], m["period"]
    n, S = px.shape
    pres = np.isfinite(px)
    sp = np.hypot(slot_velocity(px, per), slot_velocity(py, per))
    born = np.full((n, S), -1, np.int64)                       # start frame of the run containing (frame, slot)
    run_list = []                                              # (slot, a, b)
    for s in range(S):
        for a, b in runs(pres[:, s]):
            born[a:b, s] = a
            run_list.append((s, a, b))
    with np.errstate(all="ignore"):
        cx = np.where(np.isfinite(m["bx"]), m["bx"], np.nanmean(px, 1))
        cy = np.where(np.isfinite(m["by"]), m["by"], np.nanmean(py, 1))
    near = np.hypot(px - cx[:, None], py - cy[:, None]) < NEAR_PLAY_M
    g = per > 0
    nh, na = pres[:, ts > 0].sum(1), pres[:, ts < 0].sum(1)
    extra = np.zeros((n, S), bool)
    cls = {"a_extra_person": 0, "b_wrong_label": 0, "other": 0}
    other_cnt = {"<=10": 0, "11": 0, ">11": 0}
    total_cnt = {"<=22": 0, "23": 0, ">23": 0}
    n_tf = 0
    for side, mine, oth in ((1.0, nh, na), (-1.0, na, nh)):
        cols = np.flatnonzero(ts == side)
        for f in np.flatnonzero(g & (mine > 11)):
            n_tf += 1
            c = cols[pres[f, cols]]
            extra[f, c[np.argmax(born[f, c])]] = True
            o, tot = int(oth[f]), int(mine[f] + oth[f])
            other_cnt["<=10" if o <= 10 else "11" if o == 11 else ">11"] += 1
            total_cnt["<=22" if tot <= 22 else "23" if tot == 23 else ">23"] += 1
            if tot == 23 and o == 11:
                cls["a_extra_person"] += 1
            elif tot == 22 and o == 10:
                cls["b_wrong_label"] += 1
            else:
                cls["other"] += 1
    # runs that are the extra slot at some frame vs regular runs (never extra)
    ex_runs = [(s, a, b) for s, a, b in run_list if extra[a:b, s].any()]
    reg_runs = [(s, a, b) for s, a, b in run_list if not extra[a:b, s].any() and (per[a:b] > 0).any()]

    def run_stats(rr):
        return dict(n_runs=len(rr), length_s=q([(b - a) / FPS for _, a, b in rr]),
                    share_near_play=q([near[a:b, s].mean() for s, a, b in rr]),
                    mean_speed_mps=q([np.nanmean(sp[a:b, s]) for s, a, b in rr]),
                    share_of_run_as_extra=q([extra[a:b, s].mean() for s, a, b in rr]))

    # hand-over: does the run start within 2 m / 1 s of where another slot's run just ended?
    ends = {}
    for s, a, b in run_list:
        ends.setdefault(ts[s], []).append((b - 1, s))
    w = int(round(HANDOVER_S * FPS))

    def handover(rr, same: bool):
        hits = 0
        for s, a, b in rr:
            team = ts[s] if same else -ts[s]
            ok = False
            for e, s2 in ends.get(team, []):
                if s2 != s and a - w <= e <= a and np.hypot(px[e, s2] - px[a, s], py[e, s2] - py[a, s]) <= HANDOVER_M:
                    ok = True
                    break
            hits += ok
        return hits / max(len(rr), 1)

    rng = np.random.default_rng(0)
    reg_sample = [reg_runs[i] for i in rng.choice(len(reg_runs), min(len(reg_runs), 400), replace=False)]
    ex_len = [(b - a) / FPS for _, a, b in ex_runs]
    hist, _ = np.histogram(ex_len, bins=np.array(GAP_EDGES_S))
    res = dict(team_frames_over11=n_tf, graded_frames=int(g.sum()),
               frames_with_team_over11_frac=float((g & ((nh > 11) | (na > 11))).sum() / g.sum()),
               other_team_count={k: v / max(n_tf, 1) for k, v in other_cnt.items()},
               total_count={k: v / max(n_tf, 1) for k, v in total_cnt.items()},
               classification={k: v / max(n_tf, 1) for k, v in cls.items()},
               extra_runs=run_stats(ex_runs), regular_runs=run_stats(reg_runs),
               team_label_flip_within_slot=0.0,
               handover_from_other_team=dict(extra=handover(ex_runs, False), regular=handover(reg_sample, False)),
               handover_from_same_team=dict(extra=handover(ex_runs, True), regular=handover(reg_sample, True)),
               extra_run_length_hist=dict(bins=list(GAP_BIN_NAMES), counts=hist.tolist()),
               extra_share_of_run_frames_mean=float(np.mean([extra[a:b, s].mean() for s, a, b in ex_runs])) if ex_runs else None)
    print(f"graded frames with a team > 11: {100 * res['frames_with_team_over11_frac']:.1f} % ({n_tf} team-frames)")
    print("other team's slots: " + ", ".join(f"{k} {100 * v:.1f} %" for k, v in res["other_team_count"].items()))
    print("total slots: " + ", ".join(f"{k} {100 * v:.1f} %" for k, v in res["total_count"].items()))
    print("classification: " + ", ".join(f"{k} {100 * v:.1f} %" for k, v in res["classification"].items()))
    for k in ("extra_runs", "regular_runs"):
        print(f"{k}: {res[k]}")
    print("team label: fixed per slot in the metadata (0 % flips within a slot)")
    print(f"run starts within {HANDOVER_M} m / {HANDOVER_S} s of a just-ended OTHER-team slot: extra "
          f"{100 * res['handover_from_other_team']['extra']:.1f} %, regular {100 * res['handover_from_other_team']['regular']:.1f} %")
    print(f"... of a just-ended SAME-team slot: extra {100 * res['handover_from_same_team']['extra']:.1f} %, "
          f"regular {100 * res['handover_from_same_team']['regular']:.1f} %")
    print(f"extra-run lengths: {dict(zip(GAP_BIN_NAMES, hist.tolist()))}")
    AUTO.mkdir(parents=True, exist_ok=True)
    (AUTO / "extra_slot_diag.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
