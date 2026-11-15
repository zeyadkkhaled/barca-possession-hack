"""M1 C3, round 3 (2026-11-09 c): softer labels near switches and catching short possessions. Cross-validated by match on the
30 learning matches only (5 folds); never the test or stress matches. Inputs tag-free; labels v2 only as training target and for
scoring. Starts from the final C3 of round 2 ("combo" in scripts/m1/c3v2.py: extra blocks who_first + team_shape, LOOSE weight 2,
stacking, per-fold switch penalty) with the same tuned LightGBM / decoder settings per fold; no new nested Optuna.

Stages
- soft    : Part 2. Retrain layers 1-2 with (i) training rows within +-0.1 / 0.2 / 0.3 s of a state or team change in the answer
            key down-weighted to 0.25 / 0.5, (ii) label smoothing at switches (a row near a change gets the label of the other
            side of the change as a second row with weight eps = 0.5 * (1 - d / (W + 1)), d = frames to the change, W = window).
            The owner ranker (layer 3) is reused.
- decoder : Parts 3a / 3b on a given model's held-out predictions. 3a confidence-gated switching: where P(CONTROL by the other
            team) > thr for >= T consecutive 10 Hz rows, the decoder is forced to give those rows to the other team (it still
            has to obey the touch rule). 3b per-transition penalties: a 7-state decoder (TRAVELLING and LOOSE remember the team
            of the last CONTROL) with separate offsets for alive -> dead, dead -> alive, CONTROL -> TRAVELLING, CONTROL -> LOOSE,
            back to CONTROL of the same team, CONTROL of the other team, TRAVELLING <-> LOOSE. With all offsets 0 it decodes
            exactly as the 5-state decoder. Settings for fold k are tuned on the held-out predictions of the OTHER 4 folds' 24
            matches (cross-fitting; fold k's labels are never used to choose its settings): the highest spell-F1 whose accuracy
            is not lower than the fold baseline by more than 0.05 points.
- feature : Part 3c. "Short-control evidence" (m1_possession.c3_extra.short_control) as an extra block; retrain layers 1-2.
- final   : Part 4. Everything kept, together; full scoring, tolerant accuracy, spell recall by length, both LOOSE mappings.

Kept (same rule as before): frame accuracy +0.05 points with spell-F1 down at most 0.5 points, or spell-F1 +0.3 points with no
accuracy loss.

    python scripts/m1/c3v3.py soft | decoder [--tag combo] | feature | final  [--workers 10]
"""
from __future__ import annotations

import argparse
import functools
import json
import multiprocessing
import pickle
import shutil
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import c3 as O  # noqa: E402
import c3v2 as V  # noqa: E402
from compare import full_score  # noqa: E402
from diag_v2 import tolerant_hits  # noqa: E402
from barca_labels.scoring import merge, merge_had, merge_passes, score_had, score_passes  # noqa: E402
from barca_possession import c3 as C3  # noqa: E402
from barca_possession.c3_extra import short_control  # noqa: E402
from barca_possession.c3_post import passes_from  # noqa: E402
from barca_possession.smoother import FLAG_MISSING  # noqa: E402
from barca_possession.splits import FOLDS, LEARNING_MATCHES  # noqa: E402
from barca_possession.upgrades import fill_medium_gaps, smooth_players  # noqa: E402
from spells_v3 import merge_recall, recall_by_length  # noqa: E402

BASE = Path("data/processed/m1_possession")
OUT3 = BASE / "c3v3"
SCR = V.SCR                                    # D:/m1_tmp/c3v2; this round's tags start with "v3_"
EVS2 = V.EVS2
ITEMS = json.loads((V.OUT2 / "items.json").read_text())
KEPT2 = ITEMS["combo"]["kept"]                 # round-2 final: blocks, loose_w, stack, decoder pen_switch per fold
BLOCKS2 = tuple(KEPT2["blocks"])
LOOSE_W2 = KEPT2["loose_w"]
PEN_SWITCH = {int(k): v for k, v in KEPT2["decoder"]["pen_switch"].items()}
FOLD_OF = {m: k for k, f in enumerate(FOLDS) for m in f}
FPS = 29.97
SOFT = [("down", w, x) for w in (0.1, 0.2, 0.3) for x in (0.25, 0.5)] + [("smooth", w, None) for w in (0.2, 0.3)]
OFFS = ("dead", "restart", "leave_trav", "leave_loose", "ctrl_same", "ctrl_other", "tl")
OFF_GRID = (-3.0, -1.5, 0.0, 1.5, 3.0)
GRID = [OFF_GRID]           # --wide: (-4.5 ... 4.5)
GATE_THR, GATE_T = (0.3, 0.4, 0.5, 0.6, 0.7), (1, 2, 3, 5)
TUNE_ACC_TOL = 0.0005
TUNE = [TUNE_ACC_TOL]       # --acc-tol (strict run: 0.0)
KEEP_DACC, KEEP_DSPELL, KEEP_SPELL = V.KEEP_DACC, V.KEEP_DSPELL, V.KEEP_SPELL
NEG = C3.NEG if hasattr(C3, "NEG") else -1e9


def scheme_name(s):
    kind, w, x = s
    return f"down_{w:g}s_w{x:g}" if kind == "down" else f"smooth_{w:g}s"


# ---------------------------------------------------------------- labels near changes (training weights only)

@functools.lru_cache(maxsize=40)
def change_info(mid):
    """Per frame: frames to the nearest state / team change in the answer key (same period), the frame on the other side of
    that change, the layer-1 target (alive 0 / dead reason) and state of every frame, and whether the frame is scored."""
    e = np.load(EVS2 / f"{mid}.npz")
    st, tm, per = e["state"], e["team_id"], e["period"]
    n = len(st)
    ch = np.flatnonzero(((st[1:] != st[:-1]) | (tm[1:] != tm[:-1])) & (per[1:] == per[:-1])) + 1
    f = np.arange(n)
    dist = np.full(n, 10 ** 9, np.int64)
    other = f.copy()
    if len(ch):
        j = np.searchsorted(ch, f)
        cl, cr = ch[np.clip(j - 1, 0, len(ch) - 1)], ch[np.clip(j, 0, len(ch) - 1)]
        c = np.where(np.abs(f - cl) <= np.abs(cr - f), cl, cr)
        ok = per[c] == per
        dist = np.where(ok, np.abs(f - c), 10 ** 9)
        other = np.where(f < c, c, c - 1)
    rn = e["rule_names"]
    rmap = np.array([O.REASON.get(str(x), 4) for x in rn])
    reason = np.where(st == 0, rmap[e["rule"]], 0)
    return dist, other, reason, st, e["score_mask"] & (st <= 3)


def weighted_rows(mid, layer, scheme, loose_w):
    sel, F = V.train_sel(mid, O.TRAIN_EVERY, layer)
    y = F["y_reason"][sel] if layer == 1 else F["y_state"][sel] - 1
    w = np.ones(len(sel))
    dsel, dy, dw = np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    if scheme is not None:
        kind, win, wt = scheme
        fr = F["rows"][sel]
        dist, other, reason, st, ok = change_info(mid)
        W = int(round(win * FPS))
        d = dist[fr]
        near = d <= W
        if kind == "down":
            w[near] = wt
        else:
            eps = np.where(near, 0.5 * (1 - d / (W + 1)), 0.0)
            of = other[fr]
            if layer == 1:
                oy = reason[of]
                valid = near & ok[of] & (oy != y)
            else:
                oy = st[of].astype(int) - 1
                valid = near & ok[of] & (st[of] >= 1) & (st[of] <= 3) & (oy != y)
            w[valid] = 1 - eps[valid]
            dsel, dy, dw = sel[valid], oy[valid], eps[valid]
    if layer == 2:
        w = w * np.where(y == 2, loose_w, 1.0)
        dw = dw * np.where(dy == 2, loose_w, 1.0)
    return sel, y, w, dsel, dy, dw


# ---------------------------------------------------------------- features of this round

def job_sc(mid):
    out = OUT3 / "sc" / f"{mid}.npy"
    if out.exists():
        return mid, None
    warnings.simplefilter("ignore")
    z = dict(np.load(O.INP / f"{mid}.npz"))
    t, per = z["t"], z["period"]
    ts = np.where(z["team_id"] == V.HOMES[mid], 1.0, -1.0).astype(np.float32)
    px, py = smooth_players(z["px"], z["py"], per)
    bx, by, bz, bvx, bvy, fl, _ = fill_medium_gaps(t, per, z["bx"], z["by"], z["bz"], z["bvx"], z["bvy"], z["bflag"])
    P = C3.Prepared(t, per, bx, by, bz, bvx, bvy, fl, px, py, ts, z["pitch"], z["bflag"] != FLAG_MISSING)
    rows = np.flatnonzero((np.arange(len(t)) % C3.STEP == 0) & (per > 0))
    cols, names = short_control(P, rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, np.stack(cols, 1).astype(np.float32))
    (OUT3 / "sc_names.json").write_text(json.dumps(names))
    return mid, float(np.mean(cols[3]))


def xmat3(mid, k, sel, aux, use_sc):
    X = V.xmat(mid, k, sel, BLOCKS2, aux)
    if use_sc:
        X = np.hstack([X, np.load(OUT3 / "sc" / f"{mid}.npy", mmap_mode="r")[sel]])
    return X.astype(np.float32)


def stack_cached(k):
    p = SCR / "v3_stack" / f"k{k}.pkl"
    if p.exists():
        return pickle.loads(p.read_bytes())
    fold = FOLDS[k]
    train = [m for m in LEARNING_MATCHES if m not in fold]
    aux = V.stack_aux(k, {m: "oof" for m in train})
    aux.update(V.stack_aux(k, {m: "test" for m in fold}))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(pickle.dumps(aux, protocol=4))
    return aux


def fit_layer(mids, k, p, layer, aux, use_sc, scheme):
    X, y, w = [], [], []
    for m in mids:
        sel, yy, ww, dsel, dy, dw = weighted_rows(m, layer, scheme, LOOSE_W2)
        X.append(xmat3(m, k, sel, aux[m], use_sc)); y.append(yy); w.append(ww)
        if len(dsel):
            X.append(xmat3(m, k, dsel, aux[m], use_sc)); y.append(dy); w.append(dw)
    return lgb.LGBMClassifier(**O.lgb_params(p)).fit(np.concatenate(X), np.concatenate(y), sample_weight=np.concatenate(w))


def pred3(mid, k, m1, m2, aux, use_sc):
    F = O.load_feat(V.CONFIG, mid)
    X = xmat3(mid, k, slice(None), aux, use_sc)
    pr = m1.predict_proba(X)
    return dict(rows=F["rows"], p_dead=1 - pr[:, list(m1.classes_).index(0)], p_state=m2.predict_proba(X),
                tf=F[f"tf{k}"], tc=F[f"tc{k}"], y_state=F["y_state"], y_scored=F["y_scored"], cand=F["cand"].astype(int),
                ts=F["Cb"][:, :, V.meta()["cnb"].index("c_team")])


def retrain(tag, scheme=None, use_sc=False):
    t0 = time.time()
    for k, fold in enumerate(FOLDS):
        if (SCR / tag / f"k{k}_{fold[-1]}_test.pkl").exists():
            continue
        train = [m for m in LEARNING_MATCHES if m not in fold]
        p = V.lgbp(k, loose_weight=LOOSE_W2)
        aux = {m: dict(stack=v["stack"]) for m, v in stack_cached(k).items()}
        m1 = fit_layer(train, k, p, 1, aux, use_sc, scheme)
        m2 = fit_layer(train, k, p, 2, aux, use_sc, scheme)
        for m in fold:
            V.save_pred(tag, k, m, "test", pred3(m, k, m1, m2, aux[m], use_sc))
        print(f"  {tag} fold {k} trained ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------- decoder (5 states, or 7 with team-aware TRAVELLING / LOOSE)

D7, CH7, CA7, TH7, TA7, LH7, LA7 = range(7)
TO5 = np.array([C3.S_DEAD, C3.S_CH, C3.S_CA, C3.S_TRAV, C3.S_TRAV, C3.S_LOOSE, C3.S_LOOSE], np.int8)


def trans7(h, off, at_touch):
    o = {x: off.get(x, 0.0) for x in OFFS}
    A = np.full((7, 7), NEG)
    np.fill_diagonal(A, 0.0)
    for a in range(1, 7):
        A[a, D7] = h["pen_dead"] + o["dead"]
        if at_touch:
            A[D7, a] = h["pen_restart"] + o["restart"]
    for c, tr, lo, tr_o, lo_o, c_o in ((CH7, TH7, LH7, TA7, LA7, CA7), (CA7, TA7, LA7, TH7, LH7, CH7)):
        A[c, tr] = h["pen_leave"] + o["leave_trav"]
        A[c, lo] = h["pen_leave"] + o["leave_loose"]
        A[tr, lo] = A[lo, tr] = h["pen_tl"] + o["tl"]
        if at_touch:
            A[tr, c] = A[lo, c] = h["pen_ctrl"] + o["ctrl_same"]
            A[tr, c_o] = A[lo, c_o] = h["pen_ctrl"] + o["ctrl_other"]
    return A


def viterbi_m(logE, touch_ok, period, w, At, Ao):
    S = logE.shape[1]
    T = len(logE)
    path = np.zeros(T, np.int8)
    starts = np.r_[0, np.flatnonzero(period[1:] != period[:-1]) + 1, T]
    for a, b in zip(starts[:-1], starts[1:]):
        delta = w * logE[a] + np.r_[0.0, np.full(S - 1, NEG)]
        back = np.zeros((b - a, S), np.int8)
        for i in range(a + 1, b):
            m = delta[:, None] + (At if touch_ok[i] else Ao)
            back[i - a] = m.argmax(0)
            delta = m.max(0) + w * logE[i]
        s = int(np.argmax(delta))
        for i in range(b - 1, a - 1, -1):
            path[i] = s
            s = back[i - a, s] if i > a else s
    return path


def team_of_path(path, per):
    """+1 home / -1 away / 0 per row: CONTROL's team, carried through TRAVELLING / LOOSE; 0 when DEAD."""
    sg = np.where(path == C3.S_CH, 1, np.where(path == C3.S_CA, -1, 0)).astype(np.int8)
    idx = np.maximum.accumulate(np.where(sg != 0, np.arange(len(sg)), -1))
    starts = np.r_[0, np.flatnonzero(per[1:] != per[:-1]) + 1]
    pstart = starts[np.searchsorted(starts, np.arange(len(sg)), "right") - 1]
    cur = np.where(idx >= pstart, sg[np.maximum(idx, 0)], 0)
    cur[path == C3.S_DEAD] = 0
    return cur


def decode3(pred, period_rows, fps, hmm, opts):
    h = dict(hmm)
    ps = opts.get("pen_switch") or 0.0
    for key in ("pen_leave", "pen_ctrl", "pen_tl"):
        h[key] += ps
    logE = C3.emissions(pred["p_dead"], pred["p_state"], pred["q_home"])
    ok = C3.touch_mask(pred["tf"], pred["rows"], fps, h["tol_s"])
    off = opts.get("off") or {}
    seven = any(v != 0 for v in off.values())

    def run(le):
        if not seven:
            return C3.viterbi(le, ok, period_rows, h)
        le7 = le[:, [0, 1, 2, 3, 3, 4, 4]]
        return TO5[viterbi_m(le7, ok, period_rows, h["w_emit"], trans7(h, off, True), trans7(h, off, False))]

    path = run(logE)
    gate = opts.get("gate")
    if gate:
        thr, T = gate
        cur = team_of_path(path, period_rows)
        E = np.exp(logE)
        p_other = np.where(cur > 0, E[:, C3.S_CA], np.where(cur < 0, E[:, C3.S_CH], 0.0))
        cond = p_other > thr
        force = np.zeros(len(path), bool)
        for a, b in V.runs_(cond):
            if b - a >= T:
                force[a:b] = True
        if force.any():
            le = logE.copy()
            tgt = np.where(cur > 0, C3.S_CA, C3.S_CH)
            keep = np.zeros_like(le, bool)
            keep[np.arange(len(le)), tgt] = True
            le[force[:, None] & ~keep] = np.log(1e-6)
            path = run(le)
    return path


def eval3(args):
    """args: (mid, k, tag, opts, save_dir, full). Returns (mid, scores)."""
    mid, k, tag, opts, save_dir, full = args
    warnings.simplefilter("ignore")
    pred = V.load_pred(tag, k, mid, "test")
    b = V.load_pred("base", k, mid, "test")
    pred.setdefault("pc", b["pc"]); pred.setdefault("q_home", b["q_home"])
    z = np.load(O.INP / f"{mid}.npz")
    e = dict(np.load(EVS2 / f"{mid}.npz"))
    fps, per = float(z["fps"]), z["period"]
    hmm = V.hmm_of(k)
    path = decode3(pred, per[pred["rows"]], fps, hmm, opts)
    home = V.HOMES[mid]
    touches = [dict(frame=int(f), col=int(c)) for f, c in zip(pred["tf"], pred["tc"])]
    state, player, team, _ = C3.finish(path, pred["rows"], len(z["t"]), per, touches, z["team_id"], np.where(z["team_id"] == home, 1, -1),
                                       z["player_id"], pred["cand"], pred["pc"], pred["ts"], fps, hmm["tol_s"])
    state[per <= 0] = 0
    s = dict(confusion=np.asarray(full_score(e, state, player, team, None, fps)["confusion"]).tolist()) if not full else \
        full_score(e, state, player, team, None, fps)
    s["had"] = score_had(e, state, team, home)
    if full:
        passes = passes_from(state, player)
        s["passes"] = score_passes([p[0] for p in passes], [p[1] for p in passes], e["pass_frame"], e["pass_player"], fps)
        s["had_gap"] = score_had(e, state, team, home, "gap")
        okm = e["score_mask"] & (e["state"] <= 3) & (per > 0)
        s["tol"] = dict(n=int(okm.sum()), **{f"{t:g}": int((tolerant_hits(e["state"], state, okm, int(round(t * fps))) & okm).sum())
                                             for t in (0.2, 0.5)})
        s["rec_len"] = recall_by_length(e, state, team, home, fps, "last")
        s["rec_len_gap"] = recall_by_length(e, state, team, home, fps, "gap")
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(Path(save_dir) / f"{mid}.npz", state=state, player=player, team=team)
    return mid, json.loads(json.dumps(s, default=lambda o: o.item()))


def quick(scores):
    conf = np.sum([np.asarray(s["confusion"]) for s in scores], 0)
    return float(np.trace(conf) / conf.sum()), merge_had([s["had"] for s in scores])["spell_f1"]


def summary(res):
    sc = list(res.values())
    m = merge(sc) if "possession_change" in sc[0] else None
    acc, spell = quick(sc)
    out = dict(acc=acc, spell=spell)
    if m is None:
        return out
    c = np.asarray(m["confusion"], float)
    hl, hg = merge_had([s["had"] for s in sc]), merge_had([s["had_gap"] for s in sc])
    n = sum(s["tol"]["n"] for s in sc)
    out.update(tol_0_2=sum(s["tol"]["0.2"] for s in sc) / n, tol_0_5=sum(s["tol"]["0.5"] for s in sc) / n,
               spell_gap=hg["spell_f1"], macro=hl["macro_f1"], macro_gap=hg["macro_f1"], player=m["player_acc"], team=m["team_acc"],
               recall_by_state={s: c[i, i] / max(c[i].sum(), 1) for i, s in enumerate(("DEAD", "CONTROL", "TRAVELLING", "LOOSE"))},
               poss_change_f1=m["possession_change"]["f1"], dead_start_2s=m["dead_start_2s"]["f1"], dead_end_2s=m["dead_end_2s"]["f1"],
               dead_start_0_3s=m["dead_start_0.3s"]["f1"], dead_end_0_3s=m["dead_end_0.3s"]["f1"],
               pass_f1_2s=merge_passes([s["passes"] for s in sc])["2s"]["f1"],
               spell_by_iou=hl["spell_f1_by_iou"],
               rec_len={k: v["recall"] for k, v in merge_recall([s["rec_len"] for s in sc]).items()},
               rec_len_gap={k: v["recall"] for k, v in merge_recall([s["rec_len_gap"] for s in sc]).items()},
               n_rec_len={k: v["n"] for k, v in merge_recall([s["rec_len"] for s in sc]).items()})
    return out


def outer(pool, tag, opts_by_fold, save=None):
    jobs = [(m, k, tag, opts_by_fold[k], save, True) for k, f in enumerate(FOLDS) for m in f]
    return dict(pool.map(eval3, jobs))


def base_opts(k, **extra):
    return dict(pen_switch=PEN_SWITCH[k], **extra)


# ---------------------------------------------------------------- results file

STAGE = ["soft"]          # each stage writes its own results_<stage>.json (stages may run at the same time); reads merge all


def load_res():
    R = {}
    for f in sorted(OUT3.glob("results_*.json")):
        R.update(json.loads(f.read_text()))
    return R


def save_res(R):
    f = OUT3 / f"results_{STAGE[0]}.json"
    mine = json.loads(f.read_text()) if f.exists() else {}
    mine.update({k: v for k, v in R.items() if k in OWN})
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(mine, indent=1))


OWN: set = set()


def record(R, name, res, **extra):
    R[name] = dict(summary=summary(res), res=res, **extra)
    OWN.add(name)
    save_res(R)
    s = R[name]["summary"]
    print(f"{name}: acc {s['acc']:.4f} tol0.2 {s['tol_0_2']:.4f} tol0.5 {s['tol_0_5']:.4f} spell {s['spell']:.4f} "
          f"(gap {s['spell_gap']:.4f}) rec<2s {s['rec_len']['< 2 s']:.3f} 2-5 {s['rec_len']['2-5 s']:.3f}", flush=True)


def kept(R, name, ref):
    a, b = R[name]["summary"], R[ref]["summary"]
    da, ds = a["acc"] - b["acc"], a["spell"] - b["spell"]
    return bool((da >= KEEP_DACC and ds >= KEEP_DSPELL) or (ds >= KEEP_SPELL and da >= 0)), da, ds


# ---------------------------------------------------------------- tuning (cross-fitting over folds)

class Tuner:
    """Evaluates decoder settings on all 30 held-out predictions once (cached); fold k picks using the other 24 matches."""

    def __init__(self, pool, tag, pool_size):
        self.pool, self.tag, self.cache, self.n = pool, tag, {}, pool_size

    def scores(self, setting):
        key = json.dumps(setting, sort_keys=True)
        if key not in self.cache:
            jobs = [(m, FOLD_OF[m], self.tag, base_opts(FOLD_OF[m], **setting), None, False) for m in LEARNING_MATCHES]
            self.cache[key] = dict(self.pool.map(eval3, jobs))
        return self.cache[key]

    def fold_val(self, setting, k):
        r = self.scores(setting)
        return quick([v for m, v in r.items() if FOLD_OF[m] != k])


def pick(cands, base):
    """cands: [(setting, (acc, spell))]; the best spell with acc >= base acc - TUNE_ACC_TOL."""
    ok = [c for c in cands if c[1][0] >= base[0] - TUNE[0]]
    return max(ok, key=lambda c: (round(c[1][1], 6), c[1][0]))


def tune_gate(tn, k, fixed=None):
    fixed = fixed or {}
    base = tn.fold_val(fixed, k)
    cands = [(fixed, base)] + [(dict(fixed, gate=[thr, T]), tn.fold_val(dict(fixed, gate=[thr, T]), k))
                               for thr in GATE_THR for T in GATE_T]
    return pick(cands, base)


def tune_offsets(tn, k, fixed=None, rounds=2):
    fixed = fixed or {}
    cur = {x: 0.0 for x in OFFS}
    base = tn.fold_val(fixed, k)
    best = (dict(fixed), base)
    log = []
    for _ in range(rounds):
        for x in OFFS:
            cands = [best]
            for v in GRID[0]:
                trial = dict(cur, **{x: v})
                st = dict(fixed, off={a: b for a, b in trial.items() if b != 0}) if any(trial.values()) else dict(fixed)
                cands.append((st, tn.fold_val(st, k)))
            best = pick(cands, base)
            cur = {a: best[0].get("off", {}).get(a, 0.0) for a in OFFS}
            log.append((x, cur[x], round(best[1][0], 4), round(best[1][1], 4)))
    return best, log


# ---------------------------------------------------------------- stages

def run_soft(workers):
    R = load_res()
    with multiprocessing.Pool(workers, maxtasksperchild=40) as pool:
        if "baseline" not in R:
            record(R, "baseline", outer(pool, "combo", {k: base_opts(k) for k in range(5)}, save=OUT3 / "pred" / "baseline"))
    for sch in SOFT:
        name = scheme_name(sch)
        if name in R:
            continue
        retrain(f"v3_{name}", scheme=sch)
        with multiprocessing.Pool(workers, maxtasksperchild=40) as pool:
            record(R, name, outer(pool, f"v3_{name}", {k: base_opts(k) for k in range(5)}))


def run_decoder(workers, tag="combo", prefix="", items=("gate", "pertrans", "gate+pertrans")):
    R = load_res()
    with multiprocessing.Pool(workers, maxtasksperchild=200) as pool:
        tn = Tuner(pool, tag, workers)
        for item in (items if not prefix else ("pertrans",)):
            name = f"{prefix}{item}"
            if name in R:
                continue
            t0 = time.time()
            ch, logs = {}, {}
            for k in range(5):
                g = tune_gate(tn, k)[0] if "gate" in item else {}
                if "pertrans" in item:
                    (ch[k], _), logs[k] = tune_offsets(tn, k, fixed=g)
                else:
                    ch[k] = g
            print(f"{name} tuned ({time.time() - t0:.0f}s, {len(tn.cache)} settings): {ch}", flush=True)
            record(R, name, outer(pool, tag, {k: base_opts(k, **ch[k]) for k in range(5)}), choice=ch, tune_log=logs)


def run_feature(workers):
    R = load_res()
    with multiprocessing.Pool(min(workers, 4), maxtasksperchild=1) as pool:
        for mid, ev in pool.imap_unordered(job_sc, LEARNING_MATCHES):
            print(f"  sc {mid}: evidence share {ev}", flush=True)
    if "short_ctrl" not in R:
        retrain("v3_short_ctrl", use_sc=True)
        with multiprocessing.Pool(workers, maxtasksperchild=40) as pool:
            record(R, "short_ctrl", outer(pool, "v3_short_ctrl", {k: base_opts(k) for k in range(5)}))


def run_final(workers):
    R = load_res()
    ref = "baseline"
    soft_ok = [n for n in map(scheme_name, SOFT) if n in R and kept(R, n, ref)[0]]
    best_soft = max(soft_ok, key=lambda n: (R[n]["summary"]["acc"], R[n]["summary"]["spell"])) if soft_ok else None
    use_sc = "short_ctrl" in R and kept(R, "short_ctrl", ref)[0]
    dec_ok = [n for n in ("gate", "pertrans", "gate+pertrans", "strict_pertrans") if n in R and kept(R, n, ref)[0]]
    print(f"kept: soft {best_soft}, short_ctrl {use_sc}, decoder {dec_ok}", flush=True)
    scheme = dict((scheme_name(s), s) for s in SOFT).get(best_soft)
    if scheme is None and not use_sc:
        tag = "combo"
    elif scheme is not None and not use_sc:
        tag = f"v3_{best_soft}"
    elif scheme is None:
        tag = "v3_short_ctrl"
    else:
        tag = "v3_final"
        shutil.rmtree(SCR / tag, ignore_errors=True)
        retrain(tag, scheme=scheme, use_sc=True)
    with multiprocessing.Pool(workers, maxtasksperchild=200) as pool:
        if "final_model" not in R:
            record(R, "final_model", outer(pool, tag, {k: base_opts(k) for k in range(5)}), tag=tag)
        ch = {k: {} for k in range(5)}
        if dec_ok:
            tn = Tuner(pool, tag, workers)
            item = max(dec_ok, key=lambda n: R[n]["summary"]["spell"])
            if item.startswith("strict_"):
                TUNE[0], GRID[0] = 0.0, (-4.5, -3.0, -1.5, 0.0, 1.5, 3.0, 4.5)
            for k in range(5):
                g = tune_gate(tn, k)[0] if "gate" in item else {}
                ch[k] = tune_offsets(tn, k, fixed=g)[0][0] if "pertrans" in item else g
            print(f"final decoder ({item}): {ch}", flush=True)
        shutil.rmtree(OUT3 / "pred" / "final", ignore_errors=True)
        record(R, "final", outer(pool, tag, {k: base_opts(k, **ch[k]) for k in range(5)}, save=OUT3 / "pred" / "final"),
               tag=tag, decoder=ch, kept=dict(soft=best_soft, short_ctrl=use_sc, decoder=dec_ok))
        # the best spell-F1 candidate among the decoder runs (not kept by the rule unless listed above), saved for the analysis
        # robust to the LOOSE definition: spell-F1 with LOOSE as a gap must not fall below the previous C3's
        alts = [n for n in ("pertrans", "gate+pertrans", "strict_pertrans", "sc_pertrans")
                if n in R and R[n]["summary"]["spell_gap"] >= R[ref]["summary"]["spell_gap"]]
        if alts:
            alt = max(alts, key=lambda n: R[n]["summary"]["spell"])
            atag = "v3_short_ctrl" if alt.startswith("sc_") else "combo"
            ach = {int(k): v for k, v in R[alt]["choice"].items()}
            shutil.rmtree(OUT3 / "pred" / "alt", ignore_errors=True)
            record(R, "alt", outer(pool, atag, {k: base_opts(k, **ach[k]) for k in range(5)}, save=OUT3 / "pred" / "alt"),
                   tag=atag, decoder=ach, source=alt)


def main() -> int:
    from barca_possession.config import use_scratch
    use_scratch()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("soft", "decoder", "feature", "final"))
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--tag", default="combo")
    ap.add_argument("--acc-tol", type=float, default=TUNE_ACC_TOL)
    ap.add_argument("--wide", action="store_true")
    ap.add_argument("--prefix", default="")
    a = ap.parse_args()
    STAGE[0] = a.stage + (f"_{a.prefix.strip('_')}" if a.prefix else "")
    TUNE[0] = a.acc_tol
    if a.wide:
        GRID[0] = (-4.5, -3.0, -1.5, 0.0, 1.5, 3.0, 4.5)
    if a.stage == "decoder":
        run_decoder(a.workers, a.tag, a.prefix)
    else:
        dict(soft=run_soft, feature=run_feature, final=run_final)[a.stage](a.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
