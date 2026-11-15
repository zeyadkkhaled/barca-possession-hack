"""M1 C3, round 4 (2026-11-10): restart model and extra training data from the stress matches. Cross-validated by match on the
30 learning matches (5 folds); the 16 test matches are never touched. Inputs tag-free; labels v2 only as training target and for
scoring. Stress split: m1_possession.splits STRESS_EXTRA (12, extra training only, never scored) / STRESS_EVAL (6, scored
separately with each fold's model; numbers are the merge over the 5 fold models).

New baseline: C3 "combo" + the per-transition decoder ("alt" of round 3, per-fold offsets in c3v3/results_final.json).

Stages
- base    : Part 1. Rescore the new baseline (full scoring + restart metrics), save predictions.
- restart : Part 2. At the end of each predicted DEAD stretch: the restart kick = the frame where our smoothed ball speed
            crosses KICK_MPS (6 m/s) after being slow (min speed <= SLOW_MPS = 3 m/s from -1.0 to -0.2 s, or not seen), the
            crossing nearest the predicted end within [end - 3 s, end + 2 s] and >= 0.5 s after the DEAD start. (The C2 touches
            were too late / sparse at restarts: 31 % within 0.3 s of the answer key's restart vs 53 % for this detector; the two
            thresholds were picked on folds 0-1 only and then fixed.) The kick counts as a touch for the decoder. TAKER = the
            player nearest the ball 0.2 s before the crossing (<= 3 m). Restart type from the ball position (T2 patterns):
            kick-off (<= 2.5 m from the centre), corner (<= 3 m from a corner), throw-in (|y| >= half width - 2.5 m), goal kick
            (goal area + 1 m), else free kick. Restart team = taker's team, with sanity rules: goal kick -> the team defending
            that goal (its keeper's team), corner -> the attacking team (not defending that goal line), throw-in -> the team of
            the player nearest the ball who is at / beyond the touchline (|y| >= half width - 0.5 m) if any.
              end  : DEAD rows up to the kick forced DEAD, the kick row forced alive (b: DEAD ends exactly at the kick)
              team : the first alive row after the DEAD stretch (or the kick row if later) forced CONTROL of the restart team (c)
              both : end + team at the kick row; frames from the kick to the next row get CONTROL of the taker (or the nearest
                     player of the restart team)
            Forcing = emissions of the other states set to log(1e-9) (soft; never infeasible). Thresholds fixed (not tuned).
- extra   : Part 3. Per fold k: base models (24 learning matches) predict the 18 stress matches (stacking inputs + owner
            ranker); layers 1-2 of the combo model retrained without (v4_noextra) and with (v4_extra) the 12 STRESS_EXTRA
            matches (frames with score_mask True and no far-control flag); both predict fold k and the 6 STRESS_EVAL matches.
            The owner ranker is reused (base, 24 learning matches).
- final   : Part 4. Kept items together; full scoring on the 30 learning matches and on the 6 stress-eval matches.

Keep rule: (acc >= -0.05 pts and spell-F1 >= +0.3 pts) or (acc >= +0.05 pts and spell-F1 >= -0.5 pts) vs the new baseline.

    python scripts/m1/c3v4.py base | restart | extra | final  [--workers 4]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import pickle
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import c3 as O  # noqa: E402
import c3v2 as V  # noqa: E402
import c3v3 as W  # noqa: E402
from compare import full_score  # noqa: E402
from diag_v2 import tolerant_hits  # noqa: E402
from barca_labels.scoring import _spells, had, merge_had, score_had, score_passes  # noqa: E402
from barca_possession import c3 as C3  # noqa: E402
from barca_possession.c3_post import passes_from  # noqa: E402
from barca_possession.smoother import FLAG_MISSING  # noqa: E402
from barca_possession.splits import FOLDS, LEARNING_MATCHES, STRESS_EVAL, STRESS_EXTRA  # noqa: E402
from spells_v3 import matched_true, recall_by_length  # noqa: E402

BASE = Path("data/processed/m1_possession")
OUT4 = BASE / "c3v4"
SCR = V.SCR
EVS2 = V.EVS2
STRESS = tuple(sorted(STRESS_EXTRA + STRESS_EVAL))
ALT = {int(k): v for k, v in json.loads((W.OUT3 / "results_final.json").read_text())["alt"]["decoder"].items()}
KEEP = ((-0.0005, 0.003), (0.0005, -0.005))       # (min d_acc, min d_spell) pairs, either one
SLOW_MPS, KICK_MPS, TAKER_M, LOG_OFF = 3.0, 6.0, 3.0, np.log(1e-9)
MODES = ("end", "team", "both")


def homes():
    return {int(k): v for k, v in json.loads((BASE / "home_team.json").read_text()).items()}


HOMES = homes()


def dec_opts(k, **extra):
    return W.base_opts(k, **ALT[k], **extra)


def load(tag, k, mid):
    p = V.load_pred(tag, k, mid, "test")
    b = V.load_pred("base", k, mid, "test")
    p.setdefault("pc", b["pc"]); p.setdefault("q_home", b["q_home"])
    return p


# ---------------------------------------------------------------- Part 2: restart model

def own_signs(px, team_sign, per):
    """(period, team sign) -> +1 / -1: the half (x sign) the team stands in = its own half."""
    out = {}
    for p in np.unique(per[per > 0]):
        sel = per == p
        for s in (1, -1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                med = np.nanmedian(px[sel][:, team_sign == s], axis=0)
            if np.isfinite(med).any():
                out[(int(p), s)] = -1 if np.nanmean(med) < 0 else 1
    return out


def find_restarts(z, pred, path, home):
    """One entry per predicted DEAD stretch with a kick found: dict(ra, rb, kick_row, f, team (+1 home / -1 away), col, kind)."""
    rows, per = pred["rows"], z["period"]
    fps = float(z["fps"])
    n = len(z["t"])
    miss = z["bflag"] == FLAG_MISSING
    bx, by = np.where(miss, np.nan, z["bx"]), np.where(miss, np.nan, z["by"])
    speed = np.where(miss, np.nan, np.hypot(z["bvx"], z["bvy"]))
    px, py = z["px"], z["py"]
    tsign = np.where(z["team_id"] == home, 1, -1)
    own = own_signs(px, tsign, per)
    hl, hw = z["pitch"][0] / 2, z["pitch"][1] / 2
    prow = per[rows]
    dead = path == C3.S_DEAD
    cr = np.flatnonzero(np.r_[False, (speed[1:] >= KICK_MPS) & ~(speed[:-1] >= KICK_MPS)])
    w1, w0 = int(1.0 * fps), int(0.2 * fps)
    cr = np.array([f for f in cr if not np.isfinite(speed[max(0, f - w1):max(1, f - w0)]).any()
                   or np.nanmin(speed[max(0, f - w1):max(1, f - w0)]) <= SLOW_MPS], int)
    out = []
    for ra, rb in V.runs_(dead):
        if rb >= len(rows) or prow[rb] != prow[ra] or prow[ra] <= 0:
            continue
        fa, fb = rows[ra], rows[rb]
        lo, hi = max(fa + int(0.5 * fps), fb - int(3 * fps)), fb + int(2 * fps)
        c = cr[(cr >= lo) & (cr <= hi)]
        c = c[per[c] == prow[ra]]
        if not len(c):
            continue
        f = int(c[np.argmin(np.abs(c - fb))])
        p_ = int(per[f])
        g0 = max(0, f - w0)
        x, y = (bx[g0], by[g0]) if np.isfinite(bx[g0]) else (bx[f], by[f])
        dd = np.nan_to_num(np.hypot(px[g0] - x, py[g0] - y), nan=np.inf)
        if not np.isfinite(dd).any() or dd.min() > TAKER_M:
            continue
        col = int(np.argmin(dd))
        team = int(tsign[col])
        g = 1 if x > 0 else -1
        defending = next((s for s in (1, -1) if own.get((p_, s)) == g), None)
        if np.hypot(x, y) <= 2.5:
            kind = "kick_off"
        elif min(np.hypot(abs(x) - hl, abs(y) - hw), 9.0) <= 3.0:
            kind = "corner"
            if defending is not None:
                team = -defending
        elif abs(y) >= hw - 2.5:
            kind = "throw_in"
            d = np.hypot(px[f] - x, py[f] - y)
            ok = np.isfinite(d) & (np.abs(py[f]) >= hw - 0.5) & (d <= 3.0)
            if ok.any():
                team = int(tsign[np.flatnonzero(ok)[np.argmin(d[ok])]])
        elif abs(x) >= hl - 6.5 and abs(y) <= 10.2:
            kind = "goal_kick"
            if defending is not None:
                team = defending
        else:
            kind = "free_kick"
        if tsign[col] != team:                     # taker of the restart team: the nearest such player
            d = np.where(tsign == team, np.hypot(px[f] - x, py[f] - y), np.inf)
            col = int(np.nanargmin(np.nan_to_num(d, nan=np.inf))) if np.isfinite(np.nan_to_num(d, nan=np.inf)).any() else col
        out.append(dict(ra=int(ra), rb=int(rb), kick_row=int(np.searchsorted(rows, f)), f=f, team=team, col=col, kind=kind))
    return out


def decode4(pred, per_rows, fps, hmm, opts, forced=None, kicks=()):
    """c3v3.decode3 (per-transition decoder) with forced rows: forced = {row: allowed 5-state columns}; kicks = extra touch
    frames (restart kicks)."""
    h = dict(hmm)
    ps = opts.get("pen_switch") or 0.0
    for key in ("pen_leave", "pen_ctrl", "pen_tl"):
        h[key] += ps
    logE = C3.emissions(pred["p_dead"], pred["p_state"], pred["q_home"])
    if forced:
        logE = logE.copy()
        for r, allowed in forced.items():
            if r < len(logE):
                m = np.ones(logE.shape[1], bool)
                m[list(allowed)] = False
                logE[r, m] = LOG_OFF
    ok = C3.touch_mask(np.sort(np.r_[np.asarray(pred["tf"], int), np.asarray(kicks, int)]), pred["rows"], fps, h["tol_s"])
    off = opts.get("off") or {}
    if not any(v != 0 for v in off.values()):
        return C3.viterbi(logE, ok, per_rows, h)
    le7 = logE[:, [0, 1, 2, 3, 3, 4, 4]]
    return W.TO5[W.viterbi_m(le7, ok, per_rows, h["w_emit"], W.trans7(h, off, True), W.trans7(h, off, False))]


ALIVE = (C3.S_CH, C3.S_CA, C3.S_TRAV, C3.S_LOOSE)


def forced_rows(rs, mode):
    fr = {}
    for r in rs:
        ctrl = (C3.S_CH,) if r["team"] > 0 else (C3.S_CA,)
        if mode in ("end", "both"):
            for i in range(r["ra"], r["kick_row"]):
                fr[i] = (C3.S_DEAD,)
            fr[r["kick_row"]] = ctrl if mode == "both" else ALIVE
        if mode == "team":
            fr[max(r["rb"], r["kick_row"])] = ctrl
    return fr


def restart_metrics(e, state, team, home, fps):
    """Restart team (answer key: the team of the first frame after a DEAD stretch, H/A view) vs the prediction: (i) at the
    predicted DEAD end nearest the true one within +-2 s (matched), (ii) the predicted H/A 0.5 s after the true end.
    Also the number of extra (unmatched at IoU 0.5) predicted spells that start right after a predicted DEAD."""
    per, mask = e["period"], e["score_mask"]
    lv, pv = had(e["state"], e["team_id"], per, home), had(state, team, per, home)

    def ends(v):
        return np.flatnonzero((v[:-1] == 2) & (v[1:] != 2) & (per[1:] == per[:-1]) & (per[1:] > 0)) + 1

    te, pe = ends(lv), ends(pv)
    te = te[np.isin(lv[te], (0, 1)) & mask[te]]
    tol, half = int(round(2 * fps)), int(round(0.5 * fps))
    n_m = n_mc = n_at = 0
    for b in te:
        j = np.searchsorted(pe, b)
        c = [pe[i] for i in (j - 1, j) if 0 <= i < len(pe) and abs(pe[i] - b) <= tol]
        if c:
            p0 = min(c, key=lambda x: abs(x - b))
            n_m += 1
            n_mc += int(pv[p0] == lv[b])
        q = min(b + half, len(pv) - 1)
        n_at += int(per[q] == per[b] and pv[q] == lv[b])
    idx = np.flatnonzero(per > 0)
    lv2, pv2, pr2 = lv[idx], pv[idx], per[idx]
    lb, le_, lc = _spells(lv2, pr2, (0, 1))
    pb, pe_, pc = _spells(pv2, pr2, (0, 1))
    okp = matched_true(pb, pe_, pc, lb, le_, lc)
    after = np.array([b > 0 and pv2[b - 1] == 2 for b in pb], bool)
    return dict(n=int(len(te)), matched=n_m, matched_ok=n_mc, at_ok=n_at, extra_after_dead=int((~okp & after).sum()),
                extra=int((~okp).sum()), pred_spells=int(len(pb)))


def eval4(args):
    """args: (mid, k, tag, opts, save_dir, full). opts may carry restart=<mode>."""
    mid, k, tag, opts, save_dir, full = args
    warnings.simplefilter("ignore")
    pred = load(tag, k, mid)
    z = np.load(O.INP / f"{mid}.npz")
    e = dict(np.load(EVS2 / f"{mid}.npz"))
    fps, per = float(z["fps"]), z["period"]
    hmm = V.hmm_of(k)
    home = HOMES[mid]
    o = {a: b for a, b in opts.items() if a != "restart"}
    prow = per[pred["rows"]]
    path = decode4(pred, prow, fps, hmm, o)
    mode = opts.get("restart")
    rs = []
    if mode:
        rs = find_restarts(z, pred, path, home)
        path = decode4(pred, prow, fps, hmm, o, forced_rows(rs, mode), [r["f"] for r in rs])
    touches = sorted([dict(frame=int(f), col=int(c)) for f, c in zip(pred["tf"], pred["tc"])] +
                     [dict(frame=r["f"], col=r["col"]) for r in rs], key=lambda x: x["frame"])
    state, player, team, _ = C3.finish(path, pred["rows"], len(z["t"]), per, touches, z["team_id"], np.where(z["team_id"] == home, 1, -1),
                                       z["player_id"], pred["cand"], pred["pc"], pred["ts"], fps, hmm["tol_s"])
    if mode in ("end", "both"):
        rows = pred["rows"]
        for r in rs:
            kr = r["kick_row"]
            if kr >= len(rows) or path[kr] == C3.S_DEAD:
                continue
            nxt = rows[kr + 1] if kr + 1 < len(rows) else len(state)
            if mode == "both":
                state[r["f"]:nxt], player[r["f"]:nxt], team[r["f"]:nxt] = 1, z["player_id"][r["col"]], z["team_id"][r["col"]]
            else:
                state[r["f"]:rows[kr]] = state[rows[kr]]
                team[r["f"]:rows[kr]] = team[rows[kr]]
                player[r["f"]:rows[kr]] = player[rows[kr]]
    state[per <= 0] = 0
    fs = full_score(e, state, player, team, None, fps)
    s = fs if full else dict(confusion=np.asarray(fs["confusion"]).tolist())
    s["had"] = score_had(e, state, team, home)
    s["restart"] = restart_metrics(e, state, team, home, fps)
    s["restart"]["found"] = len(rs)
    s["restart"]["kinds"] = {kd: sum(r["kind"] == kd for r in rs) for kd in ("kick_off", "corner", "throw_in", "goal_kick", "free_kick")}
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
        np.savez_compressed(Path(save_dir) / f"k{k}_{mid}.npz", state=state, player=player, team=team)
    return f"{mid}:{k}", json.loads(json.dumps(s, default=lambda x: x.item()))


def summary4(res):
    out = W.summary(res)
    r = [v["restart"] for v in res.values()]
    tot = {key: sum(x[key] for x in r) for key in ("n", "matched", "matched_ok", "at_ok", "extra_after_dead", "extra", "found")}
    out.update(restart_acc_matched=tot["matched_ok"] / max(tot["matched"], 1), restart_matched_share=tot["matched"] / max(tot["n"], 1),
               restart_acc_at=tot["at_ok"] / max(tot["n"], 1), n_restarts=tot["n"], extra_after_dead=tot["extra_after_dead"],
               extra_spells=tot["extra"], restarts_found=tot["found"],
               kinds={kd: sum(x["kinds"][kd] for x in r) for kd in r[0]["kinds"]})
    return out


def run_eval(pool, tag, opts_by_fold, matches="cv", save=None):
    if matches == "cv":
        jobs = [(m, k, tag, opts_by_fold[k], save, True) for k, f in enumerate(FOLDS) for m in f]
    else:
        jobs = [(m, k, tag, opts_by_fold[k], save, True) for k in range(5) for m in STRESS_EVAL]
    return dict(pool.map(eval4, jobs))


# ---------------------------------------------------------------- results

STAGE = ["base"]


def load_res():
    R = {}
    for f in sorted(OUT4.glob("results_*.json")):
        R.update(json.loads(f.read_text()))
    return R


def record(R, name, res, **extra):
    R[name] = dict(summary=summary4(res), res=res, **extra)
    f = OUT4 / f"results_{STAGE[0]}.json"
    mine = json.loads(f.read_text()) if f.exists() else {}
    mine[name] = R[name]
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(mine, indent=1))
    s = R[name]["summary"]
    print(f"{name}: acc {s['acc']:.4f} tol0.2 {s['tol_0_2']:.4f} tol0.5 {s['tol_0_5']:.4f} spell {s['spell']:.4f} (gap {s['spell_gap']:.4f}) "
          f"dead_end 0.3s {s['dead_end_0_3s']:.4f} 2s {s['dead_end_2s']:.4f} restart-team {s['restart_acc_matched']:.3f} "
          f"(matched {s['restart_matched_share']:.3f}; at +0.5s {s['restart_acc_at']:.3f}) extra-after-DEAD {s['extra_after_dead']} "
          f"rec<2s {s['rec_len']['< 2 s']:.3f}", flush=True)


def kept(R, name, ref):
    a, b = R[name]["summary"], R[ref]["summary"]
    da, ds = a["acc"] - b["acc"], a["spell"] - b["spell"]
    return any(da >= x and ds >= y for x, y in KEEP), da, ds


# ---------------------------------------------------------------- Part 3: extra training data

def sel_rows(mid, layer):
    """Training rows: scored (score_mask True, states 0-3); stress matches additionally without the far-control flag."""
    sel, F = V.train_sel(mid, O.TRAIN_EVERY, layer)
    if mid in STRESS_EXTRA:
        far = np.load(EVS2 / f"{mid}.npz")["far_control"]
        sel = sel[~far[F["rows"][sel]]]
    return sel, F


def fit4(mids, k, p, layer, aux):
    X, y, w = [], [], []
    for m in mids:
        sel, F = sel_rows(m, layer)
        yy = F["y_reason"][sel] if layer == 1 else F["y_state"][sel] - 1
        X.append(W.xmat3(m, k, sel, aux[m], False)); y.append(yy)
        w.append(np.where(yy == 2, W.LOOSE_W2, 1.0) if layer == 2 else np.ones(len(sel)))
    return lgb.LGBMClassifier(**O.lgb_params(p)).fit(np.concatenate(X), np.concatenate(y), sample_weight=np.concatenate(w))


def run_extra():
    t0 = time.time()
    for k, fold in enumerate(FOLDS):
        train = [m for m in LEARNING_MATCHES if m not in fold]
        if not all((SCR / "base" / f"k{k}_{m}_test.pkl").exists() for m in STRESS):
            p = V.lgbp(k)
            m1, m2, m3 = V.fit_l1(train, k, p), V.fit_l2(train, k, p), V.fit_l3(train, k, p)
            for m in STRESS:
                V.save_pred("base", k, m, "test", V.pred_full(m, k, m1, m2, m3))
            del m1, m2, m3
            print(f"  fold {k}: base models -> stress predictions ({time.time() - t0:.0f}s)", flush=True)
        aux = {m: dict(stack=v["stack"]) for m, v in W.stack_cached(k).items()}
        aux.update({m: dict(stack=v["stack"]) for m, v in V.stack_aux(k, {m: "test" for m in STRESS}).items()})
        p = V.lgbp(k, loose_weight=W.LOOSE_W2)
        for tag, mids in (("v4_noextra", train), ("v4_extra", train + list(STRESS_EXTRA))):
            if (SCR / tag / f"k{k}_{STRESS_EVAL[-1]}_test.pkl").exists():
                continue
            m1, m2 = fit4(mids, k, p, 1, aux), fit4(mids, k, p, 2, aux)
            for m in list(fold) + list(STRESS_EVAL):
                V.save_pred(tag, k, m, "test", W.pred3(m, k, m1, m2, aux[m], False))
            del m1, m2
            print(f"  fold {k}: {tag} trained on {len(mids)} matches ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------- stages

def main() -> int:
    from barca_possession.config import use_scratch
    use_scratch()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("base", "restart", "extra", "final"))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    STAGE[0] = a.stage
    R = load_res()
    if a.stage == "extra":
        run_extra()
    with multiprocessing.Pool(a.workers, maxtasksperchild=50) as pool:
        if a.stage == "base" and "base_pt" not in R:
            record(R, "base_pt", run_eval(pool, "combo", {k: dec_opts(k) for k in range(5)}, save=OUT4 / "pred" / "base_pt"))
        if a.stage == "restart":
            for mode in MODES:
                if f"restart_{mode}" not in R:
                    record(R, f"restart_{mode}", run_eval(pool, "combo", {k: dec_opts(k, restart=mode) for k in range(5)}))
        if a.stage == "extra":
            for tag in ("v4_noextra", "v4_extra"):
                if tag not in R:
                    record(R, tag, run_eval(pool, tag, {k: dec_opts(k) for k in range(5)}))
        if a.stage == "final":
            ref = "base_pt"
            rk = [m for m in MODES if f"restart_{m}" in R and kept(R, f"restart_{m}", ref)[0]]
            mode = max(rk, key=lambda m: R[f"restart_{m}"]["summary"]["spell"]) if rk else None
            ex = "v4_extra" in R and kept(R, "v4_extra", ref)[0]
            tag = "v4_extra" if ex else "combo"
            print(f"kept: restart {rk} -> {mode}, extra data {ex}", flush=True)
            opts = {k: dec_opts(k, **({"restart": mode} if mode else {})) for k in range(5)}
            record(R, "final", run_eval(pool, tag, opts, save=OUT4 / "pred" / "final"), tag=tag, restart=mode, extra=ex)
            # stress-eval: the new baseline (combo model of each fold, re-fitted in the same code path) and the final version
            record(R, "stress_base", run_eval(pool, "v4_noextra", {k: dec_opts(k) for k in range(5)}, "stress"))
            ftag = "v4_extra" if ex else "v4_noextra"
            record(R, "stress_final", run_eval(pool, ftag, opts, "stress"), tag=ftag, restart=mode)
            # context: the two not-kept items on stress-eval too
            if "stress_extra" not in R:
                record(R, "stress_extra", run_eval(pool, "v4_extra", {k: dec_opts(k) for k in range(5)}, "stress"))
            if "stress_restart_both" not in R:
                record(R, "stress_restart_both", run_eval(pool, "v4_noextra", {k: dec_opts(k, restart="both") for k in range(5)}, "stress"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
