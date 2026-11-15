"""Glue for the hackathon pipeline: scoring on H/A/D codes, post-rule contexts, training one C3-T fold, Smart Tagging tags."""
from __future__ import annotations

import numpy as np

from barca_labels.labeller import CONTROL, DEAD
from barca_labels.scoring import merge_had, score_had

from barca_possession import c3t as T
from barca_possession.ballfill import SRC_SEEN, SRC_SMOOTHED

H, A, D = 0, 1, 2


def score_hda(gt, mask, period, pred) -> dict:
    """spellF1 / macroF1 of one match, H/A/D codes; identical to the official scorer when the prediction has no X
    (checked by scripts/check_scorer_parity.py). Spells run over all in-period frames, macroF1 over masked frames."""
    per = np.where(period > 0, period, 0)

    def arrays(c):
        c = np.where(c < 0, D, c)
        return np.where(c == D, DEAD, CONTROL), np.where(c == H, 0, np.where(c == A, 1, -1))

    st, tm = arrays(gt)
    ps, pt = arrays(pred)
    s = score_had(dict(state=st, team_id=tm, score_mask=mask & (per > 0), period=per), ps, pt, home_team=0)
    r = merge_had([s])
    return dict(spell=r["spell_f1"], macro=r["macro_f1"], raw=s)


def _matched_true(lb, le, lc, pb, pe, pc, thr=0.5):
    """One-to-one greedy matching at IoU >= thr (the spell-F1 rule); a bool per true spell (M1's matched_true)."""
    pairs = []
    for i, (b, e, c) in enumerate(zip(lb, le, lc)):
        lo, hi = np.searchsorted(pe, b, "right"), np.searchsorted(pb, e, "left")
        for j in range(lo, hi):
            if pc[j] == c:
                inter = min(e, pe[j]) - max(b, pb[j])
                if inter > 0:
                    pairs.append((inter / (max(e, pe[j]) - min(b, pb[j])), i, j))
    pairs.sort(reverse=True)
    ok, used = np.zeros(len(lb), bool), set()
    for iou, i, j in pairs:
        if iou < thr:
            break
        if not ok[i] and j not in used:
            ok[i] = True; used.add(j)
    return ok


def extra_metrics(gt, mask, period, pred, fps: float = 25.0) -> dict:
    """Counts for M1's two PathCRF-strength metrics, on H/A/D codes:
    restart team (matched): true D -> H/A transitions on scored frames; the predicted D end nearest within +-2 s, team
    compared (M1 c3v4.restart_metrics "matched"); short spells: true H/A spells < 2 s matched at IoU >= 0.5 (recall)."""
    lv, pv = np.where(gt < 0, D, gt), np.where(pred < 0, D, pred)
    per = np.where(period > 0, period, 0)

    def ends(v):
        return np.flatnonzero((v[:-1] == D) & (v[1:] != D) & (per[1:] == per[:-1]) & (per[1:] > 0)) + 1

    te, pe = ends(lv), ends(pv)
    te = te[np.isin(lv[te], (H, A)) & mask[te]]
    tol = int(round(2 * fps))
    n_m = n_ok = 0
    for b in te:
        j = np.searchsorted(pe, b)
        c = [pe[i] for i in (j - 1, j) if 0 <= i < len(pe) and abs(pe[i] - b) <= tol]
        if c:
            p0 = min(c, key=lambda x: abs(x - b))
            n_m += 1
            n_ok += int(pv[p0] == lv[b])
    idx = np.flatnonzero(per > 0)
    l2, p2, r2 = lv[idx], pv[idx], per[idx]

    def spells(v):
        b = np.flatnonzero(np.r_[True, (v[1:] != v[:-1]) | (r2[1:] != r2[:-1])])
        e = np.r_[b[1:], len(v)]
        s = np.isin(v[b], (H, A))
        return b[s], e[s], v[b[s]]

    lb, le, lc = spells(l2)
    pb, pe2, pc = spells(p2)
    ok = _matched_true(lb, le, lc, pb, pe2, pc)
    short = (le - lb) < 2 * fps
    return dict(restarts=int(len(te)), restarts_matched=n_m, restarts_team_ok=n_ok,
                short_spells=int(short.sum()), short_matched=int(ok[short].sum()))


def merge_extra(rows: list) -> dict:
    t = {k: sum(r[k] for r in rows) for k in rows[0]} if rows else {}
    return dict(restart_team_acc=t["restarts_team_ok"] / max(t["restarts_matched"], 1),
                restart_matched_share=t["restarts_matched"] / max(t["restarts"], 1),
                short_spell_recall=t["short_matched"] / max(t["short_spells"], 1), **t) if t else {}


def pooled(raws: list) -> dict:
    r = merge_had(raws)
    return dict(spell=r["spell_f1"], macro=r["macro_f1"])


def context(m: dict, ball: dict, tags=None, kickoff_tags=None) -> dict:
    sm = np.isin(ball["source"], (SRC_SEEN, SRC_SMOOTHED))
    return dict(missing=ball["ball_missing"], bx=ball["x"], by=ball["y"], raw_by=m["by"],
                speed=np.where(sm, np.hypot(ball["vx"], ball["vy"]), np.nan), period=m["period"],
                half_width=float(m["pitch"][1]) / 2, tags=tags, kickoff_tags=kickoff_tags)


# ---------------------------------------------------------------- training

def _cat(feats, mids, key):
    return np.concatenate([feats[m][key] for m in mids])


def _rows_labels(feats, mids):
    X = _cat(feats, mids, "X")
    y = _cat(feats, mids, "lab")
    ok = _cat(feats, mids, "mask")
    return X, y, ok


def train(feats: dict, mids: list, seed: int = 0, threads: int = 1, log=print) -> dict:
    """feats[m] = dict(rows, X, lab (row labels), mask (row scored), prow (row period), gt, gmask, period (frames)).
    Two inner groups give out-of-fold P(D) for stacking and out-of-fold emissions for choosing the decoder's w / beta."""
    mids = sorted(mids)
    groups = [mids[0::2], mids[1::2]]
    pd_oof, q_oof = {}, {}

    def l1_data(ms):
        X, y, ok = _rows_labels(feats, ms)
        return X[ok], (y[ok] == D).astype(int)

    for g in (0, 1):
        X, y = l1_data(groups[1 - g])
        mdl = T.fit_l1(X, y, seed, threads)
        for m in groups[g]:
            pd_oof[m] = mdl.predict(feats[m]["X"])
    log("  layer 1 inner fits done")
    l1 = T.fit_l1(*l1_data(mids), seed, threads)

    def l2_data(ms):
        Xs, ys = [], []
        for m in ms:
            f = feats[m]
            ok = f["mask"] & (f["lab"] != D)
            Xs.append(T.l2_inputs(f["X"], pd_oof[m], f["prow"])[ok]); ys.append((f["lab"][ok] == H).astype(int))
        return np.concatenate(Xs), np.concatenate(ys)

    for g in (0, 1):
        mdl = T.fit_l2(*l2_data(groups[1 - g]), seed, threads)
        for m in groups[g]:
            q_oof[m] = mdl.predict(T.l2_inputs(feats[m]["X"], pd_oof[m], feats[m]["prow"]))
    log("  layer 2 inner fits done")
    l2 = T.fit_l2(*l2_data(mids), seed, threads)
    C = T.transition_counts([feats[m]["lab"] for m in mids], [feats[m]["prow"] for m in mids])
    best = None
    for w in T.W_GRID:
        for beta in T.BETA_GRID:
            sc = []
            for m in mids:
                f = feats[m]
                path = T.decode(pd_oof[m], q_oof[m], f["prow"], C, w, beta)
                pred = T.rows_to_frames(path, f["rows"], len(f["period"]), f["period"])
                sc.append(score_hda(f["gt"], f["gmask"], f["period"], pred)["spell"])
            key = (float(np.mean(sc)), -beta, w)
            if best is None or key > best[0]:
                best = (key, w, beta)
    log(f"  decoder: w={best[1]} beta={best[2]} (inner OOF spellF1 {best[0][0]:.4f})")
    return dict(l1=l1, l2=l2, C=C, w=best[1], beta=best[2], inner_spell=best[0][0])


def predict(model: dict, f: dict) -> np.ndarray:
    p_dead = model["l1"].predict(f["X"])
    q = model["l2"].predict(T.l2_inputs(f["X"], p_dead, f["prow"]))
    path = T.decode(p_dead, q, f["prow"], model["C"], model["w"], model["beta"])
    return T.rows_to_frames(path, f["rows"], len(f["period"]), f["period"])


# ---------------------------------------------------------------- Smart Tagging (hackathon match)

def metrica_tags(home_name: str, fps: float = 25.0):
    """SET PIECES tags as {start_frame, end_frame, team (0 H / 1 A / None for N/A), type} and the Kick Off (Start)
    intervals per half, from the official starter reader."""
    from barca_possession.paths import official
    official()
    import load_smart_tagging as LST
    ev = LST.load()
    home_key = (home_name or "").strip().lower()
    tags, ko = [], {}
    for e in ev:
        if (e["code"] or "").strip().lower() != "set pieces":
            continue
        team_txt = (e["tags"].get("Team") or "").strip().lower()
        if team_txt in ("", "n/a", "na"):
            team = None
        elif home_key and (home_key in team_txt or team_txt in home_key):
            team = H
        else:
            team = A
        typ = (e["tags"].get("Type") or "").strip().lower()
        tags.append(dict(start_frame=e["start_frame"], end_frame=e["end_frame"], team=team, type=typ))
        if typ == "kick off (start)":
            half = (e["tags"].get("Half") or "").strip().lower()
            p = 1 if half.startswith("1") else 2 if half.startswith("2") else None
            if p:
                ko[p] = (e["start_frame"], e["end_frame"])
    return tags, ko
