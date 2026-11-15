"""M1 C3, round 5 (2026-11-11): the round-4 restart model as C3 features. Cross-validated by match on the 30 learning matches
(5 folds); the test and stress matches are not used. Inputs tag-free; labels v2 only as training target and for scoring.

Restart detector (scripts/m1/c3v4.find_restarts, unchanged): at the end of each DEAD stretch of a decoded path, kick = our smoothed
ball crossing 6 m/s after <= 3 m/s, taker = nearest player, T2 type + sanity rules -> restart team. Here the path is the base
model's decoded path (the same one the stacking feature uses): inner out-of-fold predictions for the 24 training matches of fold
k, held-out predictions for its 6 matches, so no fold sees its own labels.

Features (per 10 Hz row, relative to the last restart kick in the same period):
- restart type one-hot (kick_off, corner, throw_in, goal_kick, free_kick); restart team sign (+1 home / -1 away / 0 none);
  time since the kick (s, NaN if none); inside the first 5 s after a restart (0/1);
- restart team relative to each of the 5 candidates' teams (+1 same / -1 other / 0 none).
Layer 2 (CONTROL / TRAVELLING / LOOSE) gets all of them on top of the combo inputs. The decoder's team input (owner ranker ->
q_home) gets, per candidate: same-team-as-restart, the same times the 5 s flag, time since the kick. Layer 1 (alive / dead) is
the frozen combo model. LightGBM settings: the per-fold settings of the frozen C3; decoder: per-transition offsets (round 3 "alt").

Variants scored vs the frozen C3 (c3v4 "base_pt"): rf (features), rf_both (features + round-4 "b + c": DEAD ends at the kick, the
restart team forced). Keep rule as before.

    python scripts/m1/c3v5.py [--workers 4]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np

import c3v4 as X
from c3v4 import O, V, W

TAG = "v5_rf"
RF = V.SCR.parent / "c3v5" / "rf"
OUT5 = X.BASE / "c3v5"
KINDS = ("kick_off", "corner", "throw_in", "goal_kick", "free_kick")
IN_S = 5.0


def job_rf(args):
    """Row features of one (fold, match, kind) from the base decoded path; cached."""
    k, mid, kind = args
    out = RF / f"k{k}_{mid}_{kind}.npz"
    if out.exists():
        return
    warnings.simplefilter("ignore")
    pred = V.load_pred("base", k, mid, kind)
    z = np.load(O.INP / f"{mid}.npz")
    per, fps, t = z["period"], float(z["fps"]), z["t"]
    rows = pred["rows"]
    path = V.decode(pred, per[rows], fps, V.hmm_of(k), {})
    rs = X.find_restarts(z, pred, path, X.HOMES[mid])
    n = len(rows)
    row_f = np.zeros((n, len(KINDS) + 3), np.float32)
    since = np.full(n, np.nan, np.float32)
    team = np.zeros(n, np.float32)
    if rs:
        kf = np.array([r["f"] for r in rs])
        j = np.searchsorted(kf, rows, "right") - 1
        ok = (j >= 0) & (per[kf[np.maximum(j, 0)]] == per[rows])
        jj = np.maximum(j, 0)
        since = np.where(ok, t[rows] - t[kf[jj]], np.nan).astype(np.float32)
        team = np.where(ok, np.array([r["team"] for r in rs])[jj], 0).astype(np.float32)
        kind_i = np.array([KINDS.index(r["kind"]) for r in rs])[jj]
        row_f[np.flatnonzero(ok), kind_i[ok]] = 1.0
    row_f[:, len(KINDS)] = team
    row_f[:, len(KINDS) + 1] = since
    row_f[:, len(KINDS) + 2] = np.nan_to_num(since, nan=1e9) <= IN_S
    rel = (np.asarray(pred["ts"]) * team[:, None]).astype(np.float32)          # 5 candidates
    RF.mkdir(parents=True, exist_ok=True)
    np.savez(out, row=np.hstack([row_f, rel]), rel=rel, in5=row_f[:, -1], since=since, n=len(rs))


def rf(k, mid, kind):
    return np.load(RF / f"k{k}_{mid}_{kind}.npz")


def cand_extra(R):
    """(rows, 5, 3): same team as the restart, same team x inside 5 s, time since the kick."""
    rel, in5, since = R["rel"], R["row"][:, len(KINDS) + 2], R["row"][:, len(KINDS) + 1]
    return np.stack([rel, rel * in5[:, None], np.repeat(since[:, None], 5, 1)], 2).astype(np.float32)


def fit(train, k, aux, kind_of):
    p = V.lgbp(k, loose_weight=W.LOOSE_W2)
    X2, y2, w2, X3, y3 = [], [], [], [], []
    for m in train:
        R = rf(k, m, kind_of(m))
        sel, yy, ww, _, _, _ = W.weighted_rows(m, 2, None, W.LOOSE_W2)
        X2.append(np.hstack([W.xmat3(m, k, sel, aux[m], False), R["row"][sel]])); y2.append(yy); w2.append(ww)
        F = O.load_feat(V.CONFIG, m)
        s3 = np.flatnonzero(F["y_scored"])
        s3 = s3[(s3 % O.TRAIN_EVERY == 0)]
        s3 = s3[(F["y_state"][s3] == 1) & F["y_cand"][s3].any(1)]
        C = np.concatenate([F["Cb"][s3], F[f"Cc{k}"][s3], cand_extra(R)[s3]], 2)
        X3.append(C.reshape(-1, C.shape[2])); y3.append(F["y_cand"][s3].reshape(-1))
    m2 = lgb.LGBMClassifier(**O.lgb_params(p)).fit(np.concatenate(X2), np.concatenate(y2), sample_weight=np.concatenate(w2))
    del X2
    m3 = lgb.LGBMClassifier(**O.lgb_params(dict(p, n_estimators=min(p["n_estimators"], 200)))).fit(np.concatenate(X3), np.concatenate(y3))
    return m2, m3


def predict(mid, k, m2, m3, aux):
    pred = dict(V.load_pred("combo", k, mid, "test"))             # frozen layer 1 (p_dead), rows, touches, candidates
    R = rf(k, mid, "test")
    F = O.load_feat(V.CONFIG, mid)
    pred["p_state"] = m2.predict_proba(np.hstack([W.xmat3(mid, k, slice(None), aux, False), R["row"]]))
    C = np.concatenate([F["Cb"], F[f"Cc{k}"], cand_extra(R)], 2)
    pc = m3.predict_proba(C.reshape(-1, C.shape[2]))[:, 1].reshape(C.shape[:2])
    tot = pc.sum(1)
    pred["pc"] = pc
    pred["q_home"] = np.where(tot > 0, (pc * (pred["ts"] > 0)).sum(1) / np.maximum(tot, 1e-9), 0.5)
    return pred


def main() -> int:
    from barca_possession.config import use_scratch
    use_scratch()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    jobs = [(k, m, "test" if m in f else "oof") for k, f in enumerate(X.FOLDS) for m in X.LEARNING_MATCHES]
    with multiprocessing.Pool(a.workers, maxtasksperchild=20) as pool:
        pool.map(job_rf, jobs)
    n = [int(rf(k, m, kd)["n"]) for k, m, kd in jobs]
    print(f"restart features: {len(jobs)} (fold, match) tables, {np.mean(n):.0f} restarts per match ({time.time() - t0:.0f}s)", flush=True)
    for k, fold in enumerate(X.FOLDS):
        if (V.SCR / TAG / f"k{k}_{fold[-1]}_test.pkl").exists():
            continue
        train = [m for m in X.LEARNING_MATCHES if m not in fold]
        aux = {m: dict(stack=v["stack"]) for m, v in W.stack_cached(k).items()}
        m2, m3 = fit(train, k, aux, lambda m: "oof")
        imp = m2.booster_.feature_importance("gain")
        share = float(imp[-(len(KINDS) + 3 + 5):].sum() / max(imp.sum(), 1e-9))
        for m in fold:
            V.save_pred(TAG, k, m, "test", predict(m, k, m2, m3, aux[m]))
        print(f"  fold {k} trained ({time.time() - t0:.0f}s); restart features' share of layer-2 gain {share:.3f}", flush=True)
    X.OUT4, X.STAGE[0] = OUT5, "rf"
    R = X.load_res()
    R4 = {}
    for f in sorted((X.BASE / "c3v4").glob("results_*.json")):
        R4.update(json.loads(f.read_text()))
    R["frozen"] = R4["base_pt"]
    R["restart_both"] = R4["restart_both"]
    with multiprocessing.Pool(a.workers, maxtasksperchild=50) as pool:
        if "rf" not in R:
            X.record(R, "rf", X.run_eval(pool, TAG, {k: X.dec_opts(k) for k in range(5)}))
        if "rf_both" not in R:
            X.record(R, "rf_both", X.run_eval(pool, TAG, {k: X.dec_opts(k, restart="both") for k in range(5)}))
    for name in ("restart_both", "rf", "rf_both"):
        ok, da, ds = X.kept(R, name, "frozen")
        print(f"{name}: d_acc {100 * da:+.2f} pts, d_spell {100 * ds:+.2f} pts -> {'KEEP' if ok else 'not kept'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
