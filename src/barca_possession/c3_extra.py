"""M1 C3 extra feature blocks (Part 3 of 2026-11-09). Tag-free, numpy only; rows = the C3 row frames.

e. who_first – "who gets there first" (pitch-control idea, Spearman 2018; Fernandez & Bornn 2018): every player keeps their
   velocity for a reaction time T_REACT and then runs straight at VMAX to the ball's current position and to its position
   predicted 0.5 s and 1 s ahead (ball velocity, 0 when unknown). Per target: the fastest time of the team of the player nearest
   the ball ("same") and of the other team, their gap, and how many players of each team arrive within 1 s.
f. kick_target – "was there a target" (pass-probability idea, Spearman 2018): at each kick (a local maximum of our kick signal
   >= KICK_ACC with the ball seen) points along the ball's direction every 3 m from 6 m to 45 m; the ball reaches a point at
   distance / speed after the kick; a player reaches it in time t (model above); reception chance = logistic((t_ball - t) /
   SIGMA). Best chance of a team-mate of the kicker (nearest player at the kick) vs the best opponent; carried forward for
   up to 5 s.
g. bounce – over the last 1 s: number of ups and downs of the ball height, summed / max change of vertical speed, max
   height and height spread.
h. team_shape – per team (the team of the player nearest the forward-filled ball = "same", the other = "opp"): width
   (y p90 - p10), length (x p90 - p10), spread (mean distance to the centroid), centroid x (orientation: the home team
   defends -x), their change over the last 2 s, the same-minus-opp differences, and home-minus-away spread (signed).
"""
from __future__ import annotations

import warnings

import numpy as np

VMAX, T_REACT, SIGMA = 7.0, 0.3, 0.45
KICK_ACC, KICK_MIN_SPEED = 10.0, 3.0
BLOCKS = ("who_first", "kick_target", "bounce", "team_shape")


def _reach(px, py, vx, vy, tx, ty):
    rx = px + np.nan_to_num(vx) * T_REACT
    ry = py + np.nan_to_num(vy) * T_REACT
    return T_REACT + np.hypot(tx - rx, ty - ry) / VMAX


def _nearest_ff(P, f):
    dff = np.hypot(P.px[f] - P.bx_ff[f, None], P.py[f] - P.by_ff[f, None])
    dff = np.where(np.isfinite(dff), dff, np.inf)
    j = np.argmin(dff, 1)
    ok = np.isfinite(dff[np.arange(len(f)), j])
    return np.where(ok, P.ts[j], 0.0).astype(np.float32)


def who_first(P, f):
    cols, names = [], []
    same = _nearest_ff(P, f)
    home = P.ts > 0
    bvx = np.nan_to_num(P.bvx[f]); bvy = np.nan_to_num(P.bvy[f])
    for dt in (0.0, 0.5, 1.0):
        tx, ty = P.bx_ff[f] + bvx * dt, P.by_ff[f] + bvy * dt
        T = _reach(P.px[f], P.py[f], P.vxp[f], P.vyp[f], tx[:, None], ty[:, None])
        th = np.nanmin(np.where(home[None, :], T, np.nan), 1)
        ta = np.nanmin(np.where(~home[None, :], T, np.nan), 1)
        nh = np.sum((T <= 1.0) & home[None, :], 1); na = np.sum((T <= 1.0) & ~home[None, :], 1)
        ts_, to_ = np.where(same >= 0, th, ta), np.where(same >= 0, ta, th)
        ns_, no_ = np.where(same >= 0, nh, na), np.where(same >= 0, na, nh)
        tag = f"{dt:g}s"
        for nm, v in ((f"race_same_t@{tag}", ts_), (f"race_opp_t@{tag}", to_), (f"race_gap@{tag}", to_ - ts_),
                      (f"race_same_n1s@{tag}", ns_), (f"race_opp_n1s@{tag}", no_)):
            cols.append(v); names.append(nm)
        if dt == 0.0:
            cols.append(ta - th); names.append("race_away_minus_home@0s")
    return cols, names


def kick_target(P, f):
    n = P.n
    k = P.kacc
    kk = np.nan_to_num(k)
    loc = (kk >= KICK_ACC) & (kk >= np.r_[0, kk[:-1]]) & (kk > np.r_[kk[1:], 0]) & P.real_present
    kicks = np.flatnonzero(loc)
    home = P.ts > 0
    out = np.full((len(kicks), 5), np.nan, np.float32)        # best mate, best opp, diff, speed, kicker team sign
    pts = np.arange(6.0, 46.0, 3.0)
    for i, kf in enumerate(kicks):
        g = min(kf + 6, n - 1)
        if not (np.isfinite(P.bx[kf]) and np.isfinite(P.bx[g])) or P.period[g] != P.period[kf]:
            continue
        dx, dy = P.bx[g] - P.bx[kf], P.by[g] - P.by[kf]
        nrm = np.hypot(dx, dy)
        sp = P.sa[kf] if np.isfinite(P.sa[kf]) else nrm / max(P.t[g] - P.t[kf], 1e-3)
        if nrm < 0.05 or not np.isfinite(sp) or sp < KICK_MIN_SPEED:
            continue
        ux, uy = dx / nrm, dy / nrm
        X, Y = P.bx[kf] + ux * pts, P.by[kf] + uy * pts
        tb = pts / sp
        T = _reach(P.px[kf][:, None], P.py[kf][:, None], P.vxp[kf][:, None], P.vyp[kf][:, None], X[None, :], Y[None, :])
        pr = 1 / (1 + np.exp(-(tb[None, :] - T) * np.pi / np.sqrt(3) / SIGMA))
        best = np.nanmax(np.where(np.isfinite(T), pr, np.nan), 1)
        d0 = np.hypot(P.px[kf] - P.bx[kf], P.py[kf] - P.by[kf])
        if not np.isfinite(d0).any():
            continue
        kicker = int(np.nanargmin(np.where(np.isfinite(d0), d0, np.inf)))
        team = P.ts[kicker]
        mate = (P.ts == team) & (np.arange(len(P.ts)) != kicker)
        bm = np.nanmax(np.where(mate, best, np.nan)) if np.isfinite(np.where(mate, best, np.nan)).any() else 0.0
        bo = np.nanmax(np.where(P.ts != team, best, np.nan)) if np.isfinite(np.where(P.ts != team, best, np.nan)).any() else 0.0
        out[i] = (bm, bo, bm - bo, sp, team)
    li = np.searchsorted(kicks, f, "right") - 1
    ok = li >= 0
    lic = np.maximum(li, 0)
    kf = kicks[lic] if len(kicks) else np.zeros(len(f), int)
    since = np.where(ok, P.t[f] - P.t[np.clip(kf, 0, n - 1)], np.inf)
    ok &= (since <= 5.0) & (P.period[np.clip(kf, 0, n - 1)] == P.period[f])
    vals = out[lic] if len(kicks) else np.full((len(f), 5), np.nan, np.float32)
    vals = np.where(ok[:, None], vals, np.nan)
    names = ["kick_since_s", "kick_best_mate", "kick_best_opp", "kick_mate_minus_opp", "kick_speed", "kick_team"]
    return [np.where(ok, since, 5.0)] + [vals[:, j] for j in range(5)], names


def bounce(P, f):
    fps = P.fps
    w = int(round(fps))
    z = np.nan_to_num(P.bz, nan=0.0)
    dz = np.r_[0.0, np.diff(z)]
    sgn = np.where(np.abs(dz) > 0.3 / fps, np.sign(dz), 0.0)
    nz = np.flatnonzero(sgn)
    last = np.maximum.accumulate(np.where(sgn != 0, np.arange(len(sgn)), -1))
    prev_s = np.where(last > 0, sgn[np.maximum(np.r_[-1, last[:-1]], 0)], 0.0)
    flip = (sgn != 0) & (prev_s != 0) & (sgn != prev_s)
    vz = dz * fps
    dvz = np.abs(np.r_[0.0, np.diff(vz)])
    dvz[~np.isfinite(P.bz)] = 0.0
    cf, cd = np.r_[0, np.cumsum(flip)], np.r_[0, np.cumsum(dvz)]
    lo = np.clip(f - w, 0, None)
    hi = f + 1
    pad = np.r_[np.full(w, np.nan), P.bz]
    win = np.lib.stride_tricks.sliding_window_view(pad, w + 1)[f]
    pdv = np.r_[np.zeros(w), dvz]
    wdv = np.lib.stride_tricks.sliding_window_view(pdv, w + 1)[f]
    del nz
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return [cf[hi] - cf[lo], cd[hi] - cd[lo], wdv.max(1), np.nanmax(win, 1), np.nanstd(win, 1)], \
            ["bounce_flips_1s", "bounce_dvz_sum_1s", "bounce_dvz_max_1s", "ball_z_max_1s", "ball_z_std_1s"]


def _shape(P, g, team_mask):
    x = np.where(team_mask[None, :], P.px[g], np.nan)
    y = np.where(team_mask[None, :], P.py[g], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        qx = np.nanquantile(x, [0.1, 0.9], axis=1)
        qy = np.nanquantile(y, [0.1, 0.9], axis=1)
        cx, cy = np.nanmean(x, 1), np.nanmean(y, 1)
        spread = np.nanmean(np.hypot(x - cx[:, None], y - cy[:, None]), 1)
    return qy[1] - qy[0], qx[1] - qx[0], spread, cx * P.s[g]


def team_shape(P, f):
    home = P.ts > 0
    g2 = np.clip(f - int(round(2 * P.fps)), 0, P.n - 1)
    H, A = _shape(P, f, home), _shape(P, f, ~home)
    H2, A2 = _shape(P, g2, home), _shape(P, g2, ~home)
    same = _nearest_ff(P, f) >= 0
    cols, names = [], []
    for j, nm in enumerate(("width", "length", "spread", "cx")):
        s_, o_ = np.where(same, H[j], A[j]), np.where(same, A[j], H[j])
        cols += [s_, o_]; names += [f"shape_same_{nm}", f"shape_opp_{nm}"]
        if nm != "cx":
            s2, o2 = np.where(same, H2[j], A2[j]), np.where(same, A2[j], H2[j])
            cols += [s_ - s2, o_ - o2, s_ - o_]
            names += [f"shape_same_d{nm}_2s", f"shape_opp_d{nm}_2s", f"shape_same_minus_opp_{nm}"]
    cols += [H[2] - A[2], (H[2] - H2[2]) - (A[2] - A2[2])]
    names += ["shape_home_minus_away_spread", "shape_home_minus_away_dspread_2s"]
    return cols, names


FUNCS = dict(who_first=who_first, kick_target=kick_target, bounce=bounce, team_shape=team_shape)


def extra_features(P, frames):
    """Returns X (rows, k) float32, names, and block -> list of column names."""
    f = np.asarray(frames)
    cols, names, blocks = [], [], {}
    for b in BLOCKS:
        c, nm = FUNCS[b](P, f)
        cols += [np.asarray(v, np.float32) for v in c]
        names += nm
        blocks[b] = nm
    return np.stack(cols, 1), names, blocks


# ---------------------------------------------------------------- short-control evidence (2026-11-09 c, item 3c)

SC_RADIUS, SC_MIN_S, SC_DROP, SC_PRE_S, SC_CAP = 1.5, 0.3, 1.0, 0.3, 10.0


def short_control(P, f):
    """"Short-control evidence": a run where the SAME player is nearest to the (really seen) ball AND within SC_RADIUS, lasting
    >= SC_MIN_S while the ball slows down (ball speed drop >= SC_DROP m/s from the SC_PRE_S before the run to the run's minimum),
    by a player of the other team than the previous such run (an opponent of the last team near the ball). Not in BLOCKS (kept
    out of extra_features so older caches stay valid); computed separately by scripts/m1/c3v3.py."""
    n = P.n
    close = P.real_present & np.isfinite(P.dmin) & (P.dmin <= SC_RADIUS)
    j = np.where(close, P.near, -1)
    brk = np.r_[True, (j[1:] != j[:-1]) | (P.period[1:] != P.period[:-1])]
    st = np.flatnonzero(brk)
    en = np.r_[st[1:], n]
    run_s = np.zeros(n, np.float32)
    drop = np.zeros(n, np.float32)
    switch = np.full(n, np.nan, np.float32)
    ev = np.zeros(n, bool)
    ev_sign = np.zeros(n, np.float32)
    sp = np.where(np.isfinite(P.bsp), P.bsp, np.nan)
    pre = int(round(SC_PRE_S * P.fps))
    last_team, last_per = 0.0, None
    for a, b in zip(st, en):
        if j[a] < 0:
            continue
        if P.period[a] != last_per:
            last_team, last_per = 0.0, P.period[a]
        L = P.t[b - 1] - P.t[a]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            dr = np.nanmax(sp[max(0, a - pre):a + 1]) - np.nanmin(sp[a:b])
        dr = float(dr) if np.isfinite(dr) else 0.0
        team = float(P.ts[j[a]])
        run_s[a:b], drop[a:b] = L, dr
        if last_team != 0.0:
            switch[a:b] = float(team != last_team)
        if L >= SC_MIN_S:
            if last_team != 0.0 and team != last_team and dr >= SC_DROP:
                ev[a:b] = True
                ev_sign[a:b] = team
            last_team = team
    li = np.maximum.accumulate(np.where(ev, np.arange(n), -1))
    last_sign = np.where(li >= 0, ev_sign[np.maximum(li, 0)], 0.0)
    cols = [run_s[f], drop[f], switch[f], ev[f].astype(np.float32), P.since(ev, f, SC_CAP), P.since(ev, f, SC_CAP, future=True),
            last_sign[f]]
    names = ["sc_run_s", "sc_drop", "sc_switch", "sc_evidence", "sc_since_ev_s", "sc_to_ev_s", "sc_last_ev_team"]
    return cols, names
