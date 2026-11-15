"""M1: score contestants on labels v2 (and v1 for comparison), cross-validated on the 30 learning matches only.

Stages
- baselines: C0, C1-tuned (fold parameters chosen on v1 in compare.py), C2-tuned and C3 (old, final custom-1110) from their
  saved predictions / pass lists, scored on evalset (v1) and evalset_v2 -> compare/v2_baselines.json
- dbp: databallpy's own possession function (outputs of dbp_run.py) mapped to the four states; default parameters and
  a per-fold grid choice (frame accuracy on the other 4 folds, labels v2) -> compare/v2_dbp.json

State mapping for databallpy [OURS] (it only outputs a possessor): possessor -> CONTROL (player, team); no possessor and
our ball-alive flag false -> DEAD; no possessor, alive -> TRAVELLING with the last possessor's team within the period
(LOOSE before the first possession of the period). Passes = end of a possession followed by another player's possession
in the same period (as in C1).

    python scripts/m1/score_v2.py baselines | dbp [--workers 3]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from compare import full_score  # noqa: E402
from barca_labels.scoring import score_had, score_passes  # noqa: E402
from barca_possession import SmoothedBall  # noqa: E402
from barca_possession.baselines import c0_naive, c1_finish, c1_prepare  # noqa: E402
from barca_possession.splits import FOLDS, LEARNING_MATCHES  # noqa: E402

BASE = Path("data/processed/m1_possession")
CMP = BASE / "compare"
HOMES = {int(k): v for k, v in json.loads((BASE / "home_team.json").read_text()).items()}
EVDIR = {"v1": BASE / "evalset", "v2": BASE / "evalset_v2"}
FOLD_OF = {m: k for k, f in enumerate(FOLDS) for m in f}


def load_inputs(mid):
    z = dict(np.load(BASE / "inputs" / f"{mid}.npz"))
    b = SmoothedBall(z["bx"], z["by"], z["bz"], z["bvx"], z["bvy"], np.zeros(len(z["t"])), np.hypot(z["bvx"], z["bvy"]), z["bflag"])
    return z, b


def load_eval(mid, ver="v2"):
    return dict(np.load(EVDIR[ver] / f"{mid}.npz"))


def full(e, mid, state, player, team, fps, passes=None, pass_scores=None):
    """passes: list of (frame, passer); pass_scores: precomputed pass scores (they do not depend on the score mask)."""
    s = full_score(e, state, player, team, None, fps)
    s["passes"] = pass_scores if pass_scores is not None else (
        score_passes([p[0] for p in passes], [p[1] for p in passes], e["pass_frame"], e["pass_player"], fps) if passes is not None else None)
    s["had"] = score_had(e, state, team, HOMES[mid])
    return json.loads(json.dumps(s, default=lambda o: o.item()))


def base_job(mid):
    warnings.simplefilter("ignore")
    z, b = load_inputs(mid)
    fps = float(z["fps"])
    R = json.loads((CMP / "results.json").read_text())
    F = json.loads((BASE / "c3" / "final_custom-1110.json").read_text())
    c1p = R["c1_choice"][str(FOLD_OF[mid])]
    preds = {}
    r = c0_naive(z["t"], z["period"], b, z["px"], z["py"], z["player_id"], z["team_id"])
    preds["C0"] = (r.state, r.player, r.team, None, None)
    zone, ang, sp, spa = c1p
    prep = c1_prepare(z["t"], z["period"], b, z["px"], z["py"], z["player_id"], z["team_id"], *z["pitch"], zone_m=zone, spatials=(spa,))
    r = c1_finish(prep, ang, sp, spa)
    preds["C1-tuned"] = (r.state, r.player, r.team, [(p[0], p[1]) for p in r.passes], None)
    p = np.load(CMP / "c2_pred" / f"{mid}.npz")
    preds["C2-tuned"] = (p["state"], p["player"], p["team"], None, R["res"]["C2-tuned"][str(mid)]["passes"])
    p = np.load(BASE / "c3" / "pred" / "final_custom-1110" / f"{mid}.npz")
    preds["C3 (old)"] = (p["state"], p["player"], p["team"], None, F["res"][str(mid)]["passes"])
    out = {}
    for ver in ("v1", "v2"):
        e = load_eval(mid, ver)
        for name, (st, pl, tm, ps, pscore) in preds.items():
            out.setdefault(name, {}).setdefault(ver, {})[str(mid)] = full(e, mid, st, pl, tm, fps, ps, pscore)
    return out


def dbp_states(owner, alive, period, player_id, team_id):
    n = len(owner)
    state = np.where(alive, 3, 0).astype(np.int8)
    has = owner >= 0
    state[has] = 1
    player = np.where(has, player_id[np.maximum(owner, 0)], -1).astype(np.int64)
    team = np.where(has, team_id[np.maximum(owner, 0)], -1).astype(np.int64)
    # last possessor within the period (forward fill)
    src = np.where(has, np.arange(n), -1)
    src = np.maximum.accumulate(src)
    ok = (~has) & alive & (src >= 0)
    ok[ok] = period[src[ok]] == period[ok]
    state[ok] = 2
    team[ok] = team[src[ok]]
    # passes: possession end followed by another player's possession in the same period
    starts = np.flatnonzero(has & ~np.r_[False, has[:-1]] | (has & np.r_[False, owner[1:] != owner[:-1]]))
    passes = []
    for a, b_ in zip(starts[:-1], starts[1:]):
        if period[a] == period[b_] and owner[a] != owner[b_]:
            e = b_ - 1
            while e > a and owner[e] != owner[a]:
                e -= 1
            passes.append((int(e), int(player_id[owner[a]])))
    return state, player, team, passes


def dbp_job(mid):
    warnings.simplefilter("ignore")
    z = np.load(BASE / "inputs" / f"{mid}.npz")
    d = np.load(BASE / "dbp" / f"{mid}.npz")
    e = load_eval(mid, "v2")
    st, sc = e["state"], e["score_mask"] & (e["state"] <= 3) & (e["period"] > 0)
    per = z["period"].astype(int)
    acc = []
    for k in range(len(d["combos"])):
        s, *_ = dbp_states(d["owner"][k], d["alive"], per, z["player_id"], z["team_id"])
        acc.append((int((s[sc] == st[sc]).sum()), int(sc.sum())))
    return mid, acc


def dbp_full(args):
    mid, k = args
    warnings.simplefilter("ignore")
    z = np.load(BASE / "inputs" / f"{mid}.npz")
    d = np.load(BASE / "dbp" / f"{mid}.npz")
    s, p, t, passes = dbp_states(d["owner"][k], d["alive"], z["period"].astype(int), z["player_id"], z["team_id"])
    np.savez_compressed(BASE / "dbp" / f"pred_{mid}_k{k}.npz", state=s, player=p, team=t)
    return mid, k, {ver: full(load_eval(mid, ver), mid, s, p, t, float(z["fps"]), passes) for ver in ("v1", "v2")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("baselines", "dbp"))
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    from barca_possession.config import use_scratch
    use_scratch()                                          # temp files to M1_TMP, default D:/m1_tmp (config.py)
    with multiprocessing.Pool(a.workers, maxtasksperchild=2) as pool:
        if a.stage == "baselines":
            res = {}
            for out in pool.map(base_job, LEARNING_MATCHES):
                for name, vv in out.items():
                    for ver, mm in vv.items():
                        res.setdefault(name, {}).setdefault(ver, {}).update(mm)
            (CMP / "v2_baselines.json").write_text(json.dumps(res))
            print("baselines scored")
        else:
            acc = dict(pool.map(dbp_job, LEARNING_MATCHES))
            combos = np.load(BASE / "dbp" / f"{LEARNING_MATCHES[0]}.npz")["combos"].tolist()
            choice = {}
            for k, fold in enumerate(FOLDS):
                train = [m for m in LEARNING_MATCHES if m not in fold]
                a_ = [sum(acc[m][c][0] for m in train) / sum(acc[m][c][1] for m in train) for c in range(len(combos))]
                best = int(np.argmax(a_))
                choice[str(k)] = dict(k=best, combo=combos[best], train_acc=a_[best], train_acc_default=a_[0])
                print(f"fold {k}: default train acc {a_[0]:.4f}; chosen {combos[best]} -> {a_[best]:.4f}")
            jobs = [(m, 0) for m in LEARNING_MATCHES] + [(m, choice[str(FOLD_OF[m])]["k"]) for m in LEARNING_MATCHES]
            res = {"databallpy-default": {}, "databallpy-tuned": {}}
            for m, k, sc in pool.map(dbp_full, jobs):
                if k == 0:
                    for ver, s in sc.items():
                        res["databallpy-default"].setdefault(ver, {})[str(m)] = s
                if k == choice[str(FOLD_OF[m])]["k"]:
                    for ver, s in sc.items():
                        res["databallpy-tuned"].setdefault(ver, {})[str(m)] = s
            allc = [sum(acc[m][c][0] for m in LEARNING_MATCHES) / sum(acc[m][c][1] for m in LEARNING_MATCHES) for c in range(len(combos))]
            (CMP / "v2_dbp.json").write_text(json.dumps(dict(res=res, choice=choice, combos=combos, acc_all=allc)))
            print("databallpy scored; best on all 30:", combos[int(np.argmax(allc))], round(max(allc), 4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
