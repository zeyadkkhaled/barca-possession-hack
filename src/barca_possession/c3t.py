"""C3-T: M1's C3 adapted to anonymous, team-labelled tracking at 25 fps (the hackathon form). Tag-free inputs.

What is kept from C3 (c3.py): the per-row base features (ball position / speed / kick signal / out-of-bounds / still
ball / time since seen, pitch geometry, the players around the ball as a crowd, team activity with context at -4 ... +4 s,
ball / distance context), two LightGBM layers with stacking, and a Viterbi decoder.

What is removed: everything per player -- the 5-candidate owner ranker, C2's touches / restarts / per-player flags, the
touch-gated decoder ("CONTROL starts only at a touch", "the owner is fixed between touches") and the owner assignment.
They need persistent player identities, which the hackathon tracking does not have.

What replaces it (per frame, team level): nearest home / away distance to the ball and the gap between them (with context),
who gets there first (time-to-ball per team), who is nearest the ball's projected position 0.5 / 1 s ahead, team shape
(centroid, spread, depth, players in the opponent half) and team speeds; the ball-filling features (source, missing,
time since seen).

Target (3 states, as the hackathon scores): CONTROL / TRAVELLING -> that team, LOOSE -> the last team, DEAD -> D.
- layer 1: P(D) per row;
- layer 2: P(H | alive) per row, with layer-1 context (P(D) at -4 ... +4 s) as stacked inputs;
- decoder: 3-state Viterbi (D, H, A) per period on hybrid scaled likelihoods (posterior / prior) with transition
  probabilities counted on the training labels; emission weight w and switch-penalty scale beta chosen on the training
  matches' inner out-of-fold predictions (never on the held-out fold).
Rows every STEP = 5 frames (5 Hz); each frame takes the nearest row's state.
"""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np

from .ballfill import SRC_ESTIMATED, SRC_SEEN, SRC_SMOOTHED, lgb_params
from .c3 import Prepared, base_features
from .rules import _kick_signal

STEP = 5
H, A, D = 0, 1, 2
STACK_OFFS_S = (-4.0, -2.0, -1.0, -0.4, 0.4, 1.0, 2.0, 4.0)
DIFF_OFFS_S = (-2.0, -1.0, -0.4, 0.4, 1.0, 2.0)
W_GRID = (0.5, 1.0, 1.5, 2.0)
BETA_GRID = (0.5, 0.75, 1.0, 1.5, 2.0)
V_RUN = 7.0          # m/s, time-to-ball speed


def rows_of(period: np.ndarray) -> np.ndarray:
    """Row frames: every STEP frames from each period's first frame (period > 0 only)."""
    out = []
    for p in [p for p in np.unique(period) if p > 0]:
        idx = np.flatnonzero(period == p)
        out.append(idx[::STEP])
    return np.concatenate(out) if out else np.zeros(0, int)


def prepare(m: dict, ball: dict) -> Prepared:
    warnings.simplefilter("ignore", RuntimeWarning)
    n = len(m["t"])
    P = Prepared(m["t"], m["period"], ball["x"], ball["y"], np.full(n, np.nan, np.float32), ball["vx"], ball["vy"],
                 ball["source"].astype(np.float32), m["px"], m["py"], m["team_sign"], m["pitch"], ~ball["ball_missing"])
    sm = np.isin(ball["source"], (SRC_SEEN, SRC_SMOOTHED))          # kick signal on the smoothed ball only
    P.kacc, P.sa, P.sb, P.ang = _kick_signal(P.t, np.where(sm, ball["x"], np.nan), np.where(sm, ball["y"], np.nan), 3)
    return P


def team_features(P: Prepared, ball: dict, f: np.ndarray):
    cols, names = [], []

    def add(nm, v):
        cols.append(np.asarray(v, np.float32)); names.append(nm)

    home = P.ts > 0
    add("dh", P.dh[f]); add("da", P.da[f]); add("dh_minus_da", P.dh[f] - P.da[f])
    diff = P.dh - P.da
    for off in DIFF_OFFS_S:
        add(f"dh_minus_da@{off:+g}", P.at(diff, f, off))
    d = P.d[f]
    di = np.where(np.isfinite(d), d, np.inf)
    tti = np.maximum(di - 1.0, 0) / V_RUN
    th, ta = tti[:, home].min(1), tti[:, ~home].min(1)
    add("tti_home", np.where(np.isfinite(th), th, np.nan)); add("tti_away", np.where(np.isfinite(ta), ta, np.nan))
    add("tti_diff", np.where(np.isfinite(th - ta), th - ta, np.nan))
    for dt in (0.5, 1.0):
        qx = P.bx[f] + np.nan_to_num(P.bvx[f]) * dt
        qy = P.by[f] + np.nan_to_num(P.bvy[f]) * dt
        dq = np.hypot(P.px[f] - qx[:, None], P.py[f] - qy[:, None])
        dq = np.where(np.isfinite(dq), dq, np.inf)
        h_, a_ = dq[:, home].min(1), dq[:, ~home].min(1)
        add(f"proj{dt:g}_dh_minus_da", np.where(np.isfinite(h_ - a_), h_ - a_, np.nan))
    X, Y = P.px[f], P.py[f]
    for side, nm in ((home, "home"), (~home, "away")):
        add(f"{nm}_cx", np.nanmean(X[:, side], 1)); add(f"{nm}_cy", np.nanmean(Y[:, side], 1))
        add(f"{nm}_sdx", np.nanstd(X[:, side], 1)); add(f"{nm}_sdy", np.nanstd(Y[:, side], 1))
        add(f"{nm}_minx", np.nanmin(X[:, side], 1)); add(f"{nm}_maxx", np.nanmax(X[:, side], 1))
        add(f"{nm}_n", np.isfinite(X[:, side]).sum(1))
        add(f"{nm}_speed", np.nanmean(P.sp[f][:, side], 1))
        add(f"{nm}_cvx", np.nanmean(P.vxp[f][:, side], 1))
    add("home_n_opp_half", np.sum(X[:, home] > 0, 1)); add("away_n_opp_half", np.sum(X[:, ~home] < 0, 1))
    add("cx_diff", cols[names.index("home_cx")] - cols[names.index("away_cx")])
    add("cvx_diff", cols[names.index("home_cvx")] - cols[names.index("away_cvx")])
    add("ball_minus_home_cx", P.bx[f] - cols[names.index("home_cx")])
    add("ball_minus_away_cx", P.bx[f] - cols[names.index("away_cx")])
    add("ball_source", ball["source"][f]); add("ball_missing", ball["ball_missing"][f]); add("gap_s", ball["gap_s"][f])
    add("estimated", ball["source"][f] == SRC_ESTIMATED)
    return np.stack(cols, 1), names


def features(m: dict, ball: dict):
    """(rows, X, names) for one match."""
    rows = rows_of(m["period"])
    P = prepare(m, ball)
    Xb, nb, *_ = base_features(P, rows)
    Xt, nt = team_features(P, ball, rows)
    return rows, np.concatenate([Xb, Xt], 1), nb + nt


def stack_features(p_dead: np.ndarray, period_rows: np.ndarray) -> np.ndarray:
    """Layer-1 context on the row grid: P(D) at the offsets and its mean over +-2 s (never across a period)."""
    n = len(p_dead)
    cols = []
    for off in STACK_OFFS_S:
        k = int(round(off * 25 / STEP))
        g = np.clip(np.arange(n) + k, 0, n - 1)
        ok = (np.arange(n) + k >= 0) & (np.arange(n) + k < n) & (period_rows[g] == period_rows)
        cols.append(np.where(ok, p_dead[g], np.nan))
    w = int(round(2 * 25 / STEP))
    cs = np.r_[0, np.cumsum(p_dead)]
    lo, hi = np.clip(np.arange(n) - w, 0, n), np.clip(np.arange(n) + w + 1, 0, n)
    cols.append((cs[hi] - cs[lo]) / np.maximum(hi - lo, 1))
    cols.append(p_dead)
    return np.stack(cols, 1).astype(np.float32)


STACK_NAMES = [f"pdead@{o:+g}" for o in STACK_OFFS_S] + ["pdead_mean4s", "pdead"]


# ---------------------------------------------------------------- models

def fit_l1(X, y, seed, threads):
    return lgb.LGBMClassifier(**lgb_params(seed, threads, n_estimators=300)).fit(X, y).booster_


def fit_l2(X, y, seed, threads):
    return lgb.LGBMClassifier(**lgb_params(seed, threads, n_estimators=300)).fit(X, y).booster_


def l2_inputs(X, p_dead, prow):
    return np.concatenate([X, stack_features(p_dead, prow)], 1)


# ---------------------------------------------------------------- decoder

def transition_counts(label_rows_list, period_rows_list) -> np.ndarray:
    C = np.ones((3, 3))
    for lab, per in zip(label_rows_list, period_rows_list):
        same = per[1:] == per[:-1]
        np.add.at(C, (lab[:-1][same], lab[1:][same]), 1)
    return C


def log_trans(C: np.ndarray, beta: float) -> np.ndarray:
    A = C / C.sum(1, keepdims=True)
    L = np.log(A)
    off = ~np.eye(3, dtype=bool)
    L[off] *= beta
    return L


def priors(C: np.ndarray) -> np.ndarray:
    p = C.sum(1)
    return p / p.sum()


def log_emissions(p_dead, q_home, prior, w):
    eps = 1e-6
    E = np.stack([(1 - p_dead) * q_home, (1 - p_dead) * (1 - q_home), p_dead], 1)
    return w * (np.log(np.clip(E, eps, 1)) - np.log(prior)[None, :])


def viterbi(logE, logA, period_rows) -> np.ndarray:
    T = len(logE)
    path = np.zeros(T, np.int8)
    starts = np.r_[0, np.flatnonzero(period_rows[1:] != period_rows[:-1]) + 1, T]
    for a, b in zip(starts[:-1], starts[1:]):
        delta = logE[a].copy()
        back = np.zeros((b - a, 3), np.int8)
        for i in range(a + 1, b):
            m = delta[:, None] + logA
            bi = m.argmax(0)
            back[i - a] = bi
            delta = m[bi, (0, 1, 2)] + logE[i]
        s = int(np.argmax(delta))
        for i in range(b - 1, a - 1, -1):
            path[i] = s
            if i > a:
                s = back[i - a, s]
    return path


def rows_to_frames(path_rows, rows, n, period) -> np.ndarray:
    """Each frame takes the nearest row of the same period; frames outside periods get -1."""
    out = np.full(n, -1, np.int8)
    for p in [p for p in np.unique(period) if p > 0]:
        idx = np.flatnonzero(period == p)
        r = np.flatnonzero(period[rows] == p)
        if len(r) == 0:
            continue
        k = np.clip(np.round((idx - idx[0]) / STEP).astype(int), 0, len(r) - 1)
        out[idx] = path_rows[r[k]]
    return out


def decode(p_dead, q_home, prow, C, w, beta):
    return viterbi(log_emissions(p_dead, q_home, priors(C), w), log_trans(C, beta), prow)
