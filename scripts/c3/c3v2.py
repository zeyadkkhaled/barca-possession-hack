"""M1 C3 on labels v2 with the Part-3 improvements (2026-11-09), cross-validated by match on the 30 learning matches only.

Same inputs as the final C3 ('custom-1110': player smoothing, medium gap filling, gap features; tag-free) and the same tuned
settings per outer fold (LightGBM parameters and decoder settings chosen by the nested Optuna run in c3/final_custom-1110.json);
no new nested Optuna. Labels: evalset_v2 (training targets and scoring). Never runs on the test or stress matches.

Stages
- features : base feature tables (scripts/m1/c3.py, targets from evalset_v2) + the extra blocks of m1_possession.c3_extra
- oof      : per outer fold k: base models on the 24 training matches -> held-out predictions for the 6 fold matches, and
             inner out-of-fold (OOF) predictions for the 24 training matches (4 inner folds, each trained on 18). A binary
             LOOSE specialist is trained the same way. OOF predictions are used ONLY to tune the decoder-level settings and to
             build the stacking / specialist features of the training matches.
- items    : with / without each item a-k against the base (C3 retrained on v2). Decoder-level items (a stuck fix, b timing
             fix, c switch penalty, j tiny-spell merge, k override) are tuned per fold on the OOF predictions of that fold's
             training matches; feature items (d LOOSE weight, e-h extra blocks, i stacking, k specialist as a feature) retrain
             layers 1-2 (the owner ranker is unchanged and reused).
- combo    : everything that helped, together.

    python scripts/m1/c3v2.py features | oof | items | combo [--workers 3]
"""
from __future__ import annotations

import argparse
import functools
import json
import multiprocessing
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import c3 as O  # noqa: E402
from compare import full_score  # noqa: E402
from barca_labels.scoring import merge, merge_had, score_had, score_passes  # noqa: E402
from barca_possession import c3 as C3  # noqa: E402
from barca_possession.c3_extra import extra_features  # noqa: E402
from barca_possession.c3_post import merge_tiny, passes_from, timing_fix  # noqa: E402
from barca_possession.smoother import FLAG_MISSING  # noqa: E402
from barca_possession.splits import FOLDS, LEARNING_MATCHES  # noqa: E402
from barca_possession.upgrades import fill_medium_gaps, smooth_players  # noqa: E402

BASE = Path("data/processed/m1_possession")
OUT2 = BASE / "c3v2"
EVS2 = BASE / "evalset_v2"
CONFIG = "custom-1110"
SCR = Path(os.environ.get("M1_TMP", r"D:\m1_tmp")) / "c3v2"
O.EVS, O.OUT = EVS2, OUT2                 # base feature tables: targets from labels v2, written under c3v2/
FINAL = json.loads((BASE / "c3" / f"final_{CONFIG}.json").read_text())["choice"]
HOMES = O.homes()
FOLD_OF = {m: k for k, f in enumerate(FOLDS) for m in f}
INNER_EVERY = 4
DEAD_WIN_S, CTRL_NEAR_M, TIMING_S = 5.0, 2.0, 1.0
GRIDS = dict(stuck_N=(1.0, 2.0, 3.0, 4.0, 5.0), pen_switch=(0.0, -1.0, -2.0, -4.0, -6.0, -9.0), merge_M=(0.5, 1.0, 1.5, 2.0),
             loose_thr=(0.5, 0.6, 0.7, 0.8, 0.9), loose_w=(1.0, 2.0, 4.0, 8.0),
             pen_switch_pos=(0.0, 1.0, 2.0, 4.0))
TUNE_METRIC = dict(stuck_N="acc", pen_switch="spell", merge_M="spell", loose_thr="acc")
KEEP_DACC, KEEP_DSPELL = 0.0005, -0.005   # kept: frame accuracy +0.05 pts or more and spell-F1 down by at most 0.5 pts,
KEEP_SPELL = 0.003                         # or spell-F1 +0.3 pts or more without lower frame accuracy


def lgbp(k, **over):
    p = dict(FINAL[str(k)]["lgb"])
    p.update(over)
    return p


def hmm_of(k):
    return dict(FINAL[str(k)]["hmm"])


@functools.lru_cache(maxsize=1)
def meta():
    return json.loads((OUT2 / "feat" / CONFIG / "meta.json").read_text())


# ---------------------------------------------------------------- features

def job_feat(args):
    mid, home, c2p = args
    warnings.simplefilter("ignore")
    O.EVS, O.OUT = EVS2, OUT2
    _, m = O.job_features((mid, CONFIG, home, c2p))
    z = dict(np.load(O.INP / f"{mid}.npz"))
    t, per = z["t"], z["period"]
    ts = np.where(z["team_id"] == home, 1.0, -1.0).astype(np.float32)
    px, py = smooth_players(z["px"], z["py"], per)
    bx, by, bz, bvx, bvy, fl, _ = fill_medium_gaps(t, per, z["bx"], z["by"], z["bz"], z["bvx"], z["bvy"], z["bflag"])
    P = C3.Prepared(t, per, bx, by, bz, bvx, bvy, fl, px, py, ts, z["pitch"], z["bflag"] != FLAG_MISSING)
    rows = np.flatnonzero((np.arange(len(t)) % C3.STEP == 0) & (per > 0))
    X, names, blocks = extra_features(P, rows)
    (OUT2 / "extra").mkdir(parents=True, exist_ok=True)
    np.save(OUT2 / "extra" / f"{mid}.npy", X)
    return mid, dict(m, extra=names, blocks=blocks)


def run_features(workers):
    c2p = O.c2v2_params()
    t0 = time.time()
    with multiprocessing.Pool(workers, maxtasksperchild=1) as pool:
        metas = dict(pool.map(job_feat, [(m, HOMES[m], c2p) for m in LEARNING_MATCHES]))
    mt = next(iter(metas.values()))
    (OUT2 / "feat" / CONFIG / "meta.json").write_text(json.dumps(mt))
    print(f"features: {len(mt['nb'])} base + {len(mt['nc'])} c2 + {len(mt['extra'])} extra ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------- tables, models

def ecols(blocks):
    mt = meta()
    return [mt["extra"].index(c) for b in blocks for c in mt["blocks"][b]]


def xmat(mid, k, sel, blocks=(), aux=None):
    F = O.load_feat(CONFIG, mid)
    parts = [F["Xb"][sel], F[f"Xc{k}"][sel]]
    if blocks:
        parts.append(np.load(OUT2 / "extra" / f"{mid}.npy", mmap_mode="r")[sel][:, ecols(blocks)])
    if aux is not None:
        if "stack" in aux:
            parts.append(aux["stack"][sel])
        if "ploose" in aux:
            parts.append(aux["ploose"][sel][:, None])
    return np.hstack(parts).astype(np.float32)


def xnames(blocks=(), aux_keys=()):
    mt = meta()
    nm = mt["nb"] + mt["nc"] + [c for b in blocks for c in mt["blocks"][b]]
    if "stack" in aux_keys:
        nm += [f"stack_{a}@{o:+g}s" for o in STACK_OFFS for a in ("state", "team")]
    if "ploose" in aux_keys:
        nm += ["p_loose_specialist"]
    return nm


def train_sel(mid, every, layer):
    F = O.load_feat(CONFIG, mid)
    sel = np.flatnonzero(F["y_scored"])
    sel = sel[sel % every == 0]
    if layer == 2:
        sel = sel[F["y_state"][sel] > 0]
    return sel, F


def fit_l1(mids, k, p, blocks=(), aux=None, every=O.TRAIN_EVERY):
    X, y = [], []
    for m in mids:
        sel, F = train_sel(m, every, 1)
        X.append(xmat(m, k, sel, blocks, aux and aux[m])); y.append(F["y_reason"][sel])
    return lgb.LGBMClassifier(**O.lgb_params(p)).fit(np.concatenate(X), np.concatenate(y))


def fit_l2(mids, k, p, blocks=(), aux=None, every=O.TRAIN_EVERY):
    X, y = [], []
    for m in mids:
        sel, F = train_sel(m, every, 2)
        X.append(xmat(m, k, sel, blocks, aux and aux[m])); y.append(F["y_state"][sel] - 1)
    y = np.concatenate(y)
    return lgb.LGBMClassifier(**O.lgb_params(p)).fit(np.concatenate(X), y, sample_weight=np.where(y == 2, p["loose_weight"], 1.0))


def fit_l3(mids, k, p, every=O.TRAIN_EVERY):
    X, y = [], []
    for m in mids:
        F = O.load_feat(CONFIG, m)
        sel = np.flatnonzero(F["y_scored"])
        sel = sel[(sel % every == 0)]
        sel = sel[(F["y_state"][sel] == 1) & F["y_cand"][sel].any(1)]
        C = np.concatenate([F["Cb"][sel], F[f"Cc{k}"][sel]], 2)
        X.append(C.reshape(-1, C.shape[2])); y.append(F["y_cand"][sel].reshape(-1))
    return lgb.LGBMClassifier(**O.lgb_params(dict(p, n_estimators=min(p["n_estimators"], 200)))).fit(np.concatenate(X), np.concatenate(y))


def fit_spec(mids, k, p, every=O.TRAIN_EVERY):
    X, y = [], []
    for m in mids:
        sel, F = train_sel(m, every, 1)
        X.append(xmat(m, k, sel)); y.append(F["y_state"][sel] == 3)
    return lgb.LGBMClassifier(**O.lgb_params(p)).fit(np.concatenate(X), np.concatenate(y))


def pred_full(mid, k, m1, m2, m3, spec=None, blocks=(), aux=None):
    F = O.load_feat(CONFIG, mid)
    mt = meta()
    sel = slice(None)
    X = xmat(mid, k, sel, blocks, aux)
    pr = m1.predict_proba(X)
    out = dict(rows=F["rows"], p_dead=1 - pr[:, list(m1.classes_).index(0)], p_state=m2.predict_proba(X),
               tf=F[f"tf{k}"], tc=F[f"tc{k}"], y_state=F["y_state"], y_scored=F["y_scored"], cand=F["cand"].astype(int))
    ci = mt["cnb"].index("c_team")
    out["ts"] = F["Cb"][:, :, ci]
    if m3 is not None:
        C = np.concatenate([F["Cb"], F[f"Cc{k}"]], 2)
        pc = m3.predict_proba(C.reshape(-1, C.shape[2]))[:, 1].reshape(C.shape[:2])
        tot = pc.sum(1)
        out["pc"] = pc
        out["q_home"] = np.where(tot > 0, (pc * (out["ts"] > 0)).sum(1) / np.maximum(tot, 1e-9), 0.5)
    if spec is not None:
        out["p_loose"] = spec.predict_proba(xmat(mid, k, sel))[:, 1]
    # evidence for the stuck fix
    xb, xc = F["Xb"], F[f"Xc{k}"]
    beyond = xb[:, mt["nb"].index("since_beyond_s")] < 1.0
    restart = xc[:, mt["nc"].index("since_restart_s")] < 1.0
    out["ev_dead"] = beyond | restart
    cd = F["Cb"][:, :, mt["cnb"].index("c_d")]
    near = np.nan_to_num(cd, nan=99.0) <= CTRL_NEAR_M
    out["near_h"] = (near & (out["ts"] > 0)).any(1)
    out["near_a"] = (near & (out["ts"] < 0)).any(1)
    return out


def save_pred(tag, k, mid, kind, pred):
    d = SCR / tag
    d.mkdir(parents=True, exist_ok=True)
    (d / f"k{k}_{mid}_{kind}.pkl").write_bytes(pickle.dumps(pred, protocol=4))


def load_pred(tag, k, mid, kind):
    return pickle.loads((SCR / tag / f"k{k}_{mid}_{kind}.pkl").read_bytes())


def run_oof(workers):
    t0 = time.time()
    for k, fold in enumerate(FOLDS):
        if (SCR / "base" / f"k{k}_{fold[-1]}_test.pkl").exists():
            print(f"fold {k} exists", flush=True)
            continue
        train = [m for m in LEARNING_MATCHES if m not in fold]
        p = lgbp(k)
        m1, m2, m3, sp = fit_l1(train, k, p), fit_l2(train, k, p), fit_l3(train, k, p), fit_spec(train, k, p)
        for m in fold:
            save_pred("base", k, m, "test", pred_full(m, k, m1, m2, m3, sp))
        imp = O.importance((m1, m2, m3), xnames(), meta()["cnb"] + meta()["cnc"])
        (SCR / "base" / f"k{k}_importance.json").write_text(json.dumps(imp))
        del m1, m2, m3, sp
        inner = [[m for m in f if m in train] for f in FOLDS if any(m in train for m in f)]
        for val in inner:
            tr = [m for m in train if m not in val]
            m1, m2, m3, sp = fit_l1(tr, k, p), fit_l2(tr, k, p), fit_l3(tr, k, p), fit_spec(tr, k, p)
            for m in val:
                save_pred("base", k, m, "oof", pred_full(m, k, m1, m2, m3, sp))
        print(f"fold {k} oof done ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------- decoding variants

def runs_(mask):
    d = np.diff(np.r_[0, np.asarray(mask, np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def decode(pred, period_rows, fps, hmm, opts):
    h = dict(hmm)
    ps = opts.get("pen_switch") or 0.0
    for key in ("pen_leave", "pen_ctrl", "pen_tl"):
        h[key] += ps
    logE = C3.emissions(pred["p_dead"], pred["p_state"], pred["q_home"])
    ok = C3.touch_mask(pred["tf"], pred["rows"], fps, h["tol_s"])
    path = C3.viterbi(logE, ok, period_rows, h)
    N = opts.get("stuck_N")
    if N:
        rate = 1.0 / (C3.STEP / fps)
        bad = np.zeros((len(path), 5), bool)
        w = int(round(DEAD_WIN_S * rate))
        starts = set(np.r_[0, np.flatnonzero(period_rows[1:] != period_rows[:-1]) + 1].tolist())
        for a, b in runs_(path == C3.S_DEAD) if not opts.get("stuck_ctrl_only") else []:
            if a in starts:
                continue
            if not pred["ev_dead"][max(0, a - w):b + w].any():
                bad[a:b, C3.S_DEAD] = True
        nN = int(round(N * rate))
        for s_, near in ((C3.S_CH, pred["near_h"]), (C3.S_CA, pred["near_a"])):
            for a, b in runs_(path == s_):
                last = a
                for i in range(a, b):
                    if near[i]:
                        last = i
                    elif i - last > nN:
                        bad[i, s_] = True
        if bad.any():
            logE = np.where(bad, np.log(1e-6), logE)
            path = C3.viterbi(logE, ok, period_rows, h)
    thr = opts.get("loose_thr")
    if thr is not None and "p_loose" in pred:
        path = path.copy()
        path[(pred["p_loose"] > thr) & np.isin(path, (C3.S_CH, C3.S_CA, C3.S_TRAV))] = C3.S_LOOSE
    return path


def eval_one(args):
    """Decode + finish + post-process + score one match. args: (mid, k, tag, kind, opts, save_dir)."""
    mid, k, tag, kind, opts, save_dir = args
    warnings.simplefilter("ignore")
    pred = load_pred(tag, k, mid, kind)
    if "pc" not in pred:                                         # layer 3 reused from the base predictions
        b = load_pred("base", k, mid, kind)
        pred.update(pc=b["pc"], q_home=b["q_home"])
        if "p_loose" not in pred and "p_loose" in b:
            pred["p_loose"] = b["p_loose"]
    z = np.load(O.INP / f"{mid}.npz")
    e = dict(np.load(EVS2 / f"{mid}.npz"))
    fps, per = float(z["fps"]), z["period"]
    hmm = hmm_of(k)
    path = decode(pred, per[pred["rows"]], fps, hmm, opts)
    home = HOMES[mid]
    touches = [dict(frame=int(f), col=int(c)) for f, c in zip(pred["tf"], pred["tc"])]
    state, player, team, _ = C3.finish(path, pred["rows"], len(z["t"]), per, touches, z["team_id"], np.where(z["team_id"] == home, 1, -1),
                                       z["player_id"], pred["cand"], pred["pc"], pred["ts"], fps, hmm["tol_s"])
    state[per <= 0] = 0
    if opts.get("timing"):
        state, player, team = timing_fix(state, player, team, per, pred["tf"], fps, opts.get("timing_tol", TIMING_S))
    if opts.get("merge_M"):
        state, player, team = merge_tiny(state, player, team, per, home, fps, opts["merge_M"])
    passes = passes_from(state, player)
    s = full_score(e, state, player, team, None, fps)
    s["passes"] = score_passes([p[0] for p in passes], [p[1] for p in passes], e["pass_frame"], e["pass_player"], fps)
    s["had"] = score_had(e, state, team, home)
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(Path(save_dir) / f"{mid}.npz", state=state, player=player, team=team)
    return mid, json.loads(json.dumps(s, default=lambda o: o.item()))


def summ(scores):
    m = merge(scores)
    h = merge_had([s["had"] for s in scores])
    c = np.asarray(m["confusion"], float)
    return dict(acc=m["accuracy"], loose_rec=c[3, 3] / max(c[3].sum(), 1), spell=h["spell_f1"], macro=h["macro_f1"],
                player=m["player_acc"], team=m["team_acc"])


def tune(pool, k, key, grid, base_opts, tag="base", metric=None):
    train = [m for m in LEARNING_MATCHES if m not in FOLDS[k]]
    best, best_v, log = None, -1, []
    for v in grid:
        opts = dict(base_opts, **{key: v})
        sc = [s for _, s in pool.map(eval_one, [(m, k, tag, "oof", opts, None) for m in train])]
        r = summ(sc)
        val = r["spell"] if (metric or TUNE_METRIC[key]) == "spell" else r["acc"]
        log.append((v, round(val, 4)))
        if val > best_v:
            best, best_v = v, val
    return best, log


def outer(pool, opts_by_fold, tag="base", save=None):
    jobs = [(m, k, tag, "test", opts_by_fold[k], save) for k, f in enumerate(FOLDS) for m in f]
    return dict(pool.map(eval_one, jobs))


STACK_OFFS = (0.0, 5.0, 10.0)


def stack_aux(k, kind_by_match):
    """Stacking features from the base decoded paths: decoded state and team sign now, +5 s and +10 s."""
    aux = {}
    for m, kind in kind_by_match.items():
        pred = load_pred("base", k, m, kind)
        z = np.load(O.INP / f"{m}.npz")
        path = decode(pred, z["period"][pred["rows"]], float(z["fps"]), hmm_of(k), {})
        rs = O.ROW_STATE[path].astype(np.float32)
        tsg = np.where(path == C3.S_CH, 1.0, np.where(path == C3.S_CA, -1.0, 0.0)).astype(np.float32)
        per = z["period"][pred["rows"]]
        cols = []
        for off in STACK_OFFS:
            j = np.arange(len(path)) + int(round(off * 10))
            jc = np.clip(j, 0, len(path) - 1)
            okj = (j < len(path)) & (per[jc] == per)
            cols += [np.where(okj, rs[jc], np.nan), np.where(okj, tsg[jc], np.nan)]
        aux[m] = dict(stack=np.stack(cols, 1))
    return aux


def retrain(tag, blocks=(), loose_w=None, use_stack=False, use_ploose=False):
    """Outer models with extra inputs (layers 1-2 only); the base owner ranker is reused."""
    for k, fold in enumerate(FOLDS):
        if (SCR / tag / f"k{k}_{fold[-1]}_test.pkl").exists():
            continue
        train = [m for m in LEARNING_MATCHES if m not in fold]
        p = lgbp(k, **({"loose_weight": loose_w} if loose_w is not None else {}))
        aux_tr = aux_te = None
        if use_stack or use_ploose:
            aux_tr = {m: {} for m in train}
            aux_te = {m: {} for m in fold}
            if use_stack:
                for m, v in stack_aux(k, {m: "oof" for m in train}).items():
                    aux_tr[m].update(v)
                for m, v in stack_aux(k, {m: "test" for m in fold}).items():
                    aux_te[m].update(v)
            if use_ploose:
                for m in train:
                    aux_tr[m]["ploose"] = load_pred("base", k, m, "oof")["p_loose"]
                for m in fold:
                    aux_te[m]["ploose"] = load_pred("base", k, m, "test")["p_loose"]
        m1, m2 = fit_l1(train, k, p, blocks, aux_tr), fit_l2(train, k, p, blocks, aux_tr)
        names = xnames(blocks, ("stack",) * use_stack + ("ploose",) * use_ploose)
        imp = O.importance((m1, m2, fit_dummy()), names, [])
        for m in fold:
            pr = pred_full(m, k, m1, m2, None, None, blocks, aux_te and aux_te[m])
            save_pred(tag, k, m, "test", pr)
        (SCR / tag / f"k{k}_importance.json").write_text(json.dumps(imp))


def fit_dummy():
    class _D:
        class booster_:
            @staticmethod
            def feature_importance(_):
                return np.zeros(0)
    return _D()


def importance_of(tag):
    imp = {}
    for k in range(5):
        p = SCR / tag / f"k{k}_importance.json"
        if p.exists():
            for a, b in json.loads(p.read_text()).items():
                imp[a] = imp.get(a, 0.0) + b / 5
    return imp


# ---------------------------------------------------------------- items / combo

FEATURE_ITEMS = dict(e=("who_first",), f=("kick_target",), g=("bounce",), h=("team_shape",))


def run_items(workers):
    path = OUT2 / "items.json"
    R = json.loads(path.read_text()) if path.exists() else {}
    t0 = time.time()

    def done(name, res, extra=None):
        R[name] = dict(res=res, summary=summ(list(res.values())), **(extra or {}))
        path.write_text(json.dumps(R))
        s = R[name]["summary"]
        print(f"{name}: acc {s['acc']:.4f} loose_rec {s['loose_rec']:.3f} spell {s['spell']:.3f} ({time.time() - t0:.0f}s)", flush=True)

    with multiprocessing.Pool(workers, maxtasksperchild=20) as pool:
        none = {k: {} for k in range(5)}
        if "base" not in R:
            done("base", outer(pool, none, save=OUT2 / "pred" / "base"))
        for item, key in (("a", "stuck_N"), ("c", "pen_switch"), ("j", "merge_M"), ("k_override", "loose_thr")):
            if item in R:
                continue
            ch, logs = {}, {}
            for k in range(5):
                ch[k], logs[k] = tune(pool, k, key, GRIDS[key], {})
            done(item, outer(pool, {k: {key: ch[k]} for k in range(5)}), dict(choice=ch, tune_log=logs))
        if "b" not in R:
            done("b", outer(pool, {k: {"timing": True} for k in range(5)}))
    for item, blocks in FEATURE_ITEMS.items():
        if item not in R:
            retrain(f"item_{item}", blocks)
            with multiprocessing.Pool(workers, maxtasksperchild=20) as pool:
                done(item, outer(pool, {k: {} for k in range(5)}, tag=f"item_{item}"), dict(importance=importance_of(f"item_{item}")))
    for w in GRIDS["loose_w"]:
        name = f"d_w{w:g}"
        if name not in R:
            retrain(f"item_{name}", loose_w=w)
            with multiprocessing.Pool(workers, maxtasksperchild=20) as pool:
                done(name, outer(pool, {k: {} for k in range(5)}, tag=f"item_{name}"))
    with multiprocessing.Pool(workers, maxtasksperchild=20) as pool:
        if "a_ctrl" not in R:                                    # a without the DEAD-evidence half
            ch, logs = {}, {}
            for k in range(5):
                ch[k], logs[k] = tune(pool, k, "stuck_N", GRIDS["stuck_N"], {"stuck_ctrl_only": True})
            done("a_ctrl", outer(pool, {k: {"stuck_N": ch[k], "stuck_ctrl_only": True} for k in range(5)}),
                 dict(choice=ch, tune_log=logs))
        if "b_0.3" not in R:                                     # b with a 0.3 s window
            done("b_0.3", outer(pool, {k: {"timing": True, "timing_tol": 0.3} for k in range(5)}))
        if "c_pos" not in R:                                     # c with looser (positive) penalties
            ch, logs = {}, {}
            for k in range(5):
                ch[k], logs[k] = tune(pool, k, "pen_switch", GRIDS["pen_switch_pos"], {}, metric="spell")
            done("c_pos", outer(pool, {k: {"pen_switch": ch[k]} for k in range(5)}), dict(choice=ch, tune_log=logs))
    for name, kw in (("i", dict(use_stack=True)), ("k_feature", dict(use_ploose=True))):
        if name not in R:
            retrain(f"item_{name}", **kw)
            with multiprocessing.Pool(workers, maxtasksperchild=20) as pool:
                done(name, outer(pool, {k: {} for k in range(5)}, tag=f"item_{name}"), dict(importance=importance_of(f"item_{name}")))


def run_combo(workers):
    R = json.loads((OUT2 / "items.json").read_text())
    b = R["base"]["summary"]

    def helps(n):
        if n not in R:
            return False
        da, ds = R[n]["summary"]["acc"] - b["acc"], R[n]["summary"]["spell"] - b["spell"]
        return (da >= KEEP_DACC and ds >= KEEP_DSPELL) or (ds >= KEEP_SPELL and da >= 0)
    blocks = tuple(bl for it, bls in FEATURE_ITEMS.items() if helps(it) for bl in bls)
    ws = [w for w in GRIDS["loose_w"] if helps(f"d_w{w:g}")]
    best_w = max(ws, key=lambda w: R[f"d_w{w:g}"]["summary"]["acc"]) if ws else None
    use_stack, use_ploose = helps("i"), helps("k_feature")
    dec, extra = {}, {}
    for items, key in ((("a", "a_ctrl"), "stuck_N"), (("c", "c_pos"), "pen_switch"), (("j",), "merge_M"),
                       (("k_override",), "loose_thr")):
        ok = [it for it in items if helps(it)]
        if ok:
            it = max(ok, key=lambda x: R[x]["summary"]["acc"])
            dec[key] = {int(k): v for k, v in R[it]["choice"].items()}
            if it == "a_ctrl":
                extra["stuck_ctrl_only"] = True
    tb = [it for it in ("b", "b_0.3") if helps(it)]
    timing = bool(tb)
    if timing:
        extra["timing_tol"] = 0.3 if max(tb, key=lambda x: R[x]["summary"]["acc"]) == "b_0.3" else TIMING_S
    kept = dict(blocks=blocks, loose_w=best_w, stack=use_stack, ploose=use_ploose, decoder={k: v for k, v in dec.items()},
                timing=timing, extra=extra)
    print("kept:", kept, flush=True)
    tag = "combo"
    import shutil
    shutil.rmtree(SCR / tag, ignore_errors=True)
    retrain(tag, blocks, best_w, use_stack, use_ploose)
    opts = {k: dict({key: dec[key][k] for key in dec}, timing=timing, **extra) for k in range(5)}
    with multiprocessing.Pool(workers, maxtasksperchild=20) as pool:
        res = outer(pool, opts, tag=tag, save=OUT2 / "pred" / "combo")
    R["combo"] = dict(res=res, summary=summ(list(res.values())), kept=kept, importance=importance_of(tag))
    (OUT2 / "items.json").write_text(json.dumps(R))
    s = R["combo"]["summary"]
    print(f"combo: acc {s['acc']:.4f} loose_rec {s['loose_rec']:.3f} spell {s['spell']:.3f}", flush=True)


def main() -> int:
    from barca_possession.config import use_scratch
    use_scratch()                                          # temp files to M1_TMP, default D:/m1_tmp (config.py)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("features", "oof", "items", "combo"))
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    dict(features=run_features, oof=run_oof, items=run_items, combo=run_combo)[a.stage](a.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
