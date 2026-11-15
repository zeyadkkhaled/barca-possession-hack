"""Re-link anonymous slots into 11 persistent tracks ("lanes") per team, for PathCRF's 22 player nodes.

1. trim each team to 11 slots per frame (load.trim_to_11: the newest-born slots go), so at most 11 runs of a team are
   present at any frame;
2. per team and period, walk through the presence runs in start order. A run that starts while its slot is new takes a
   free lane (a lane whose previous run has ended). All runs starting on the same frame are assigned together with the
   Hungarian method on the distance between the lane's predicted position (last position + last velocity x gap, the
   extrapolation capped at 1 s) and the run's first position:
   - a lane never used in this period costs UNUSED_COST_M (so a far-away free lane loses to opening a fresh lane);
   - a pair outside the gate (GATE_M + VMAX_MPS x gap) is still allowed (a team never has more than 11 runs at once,
     so every run gets a lane) but costs FORCED_COST_M more and is counted as forced.
3. a link = a run taking a lane that already held an earlier run in the period (a lane's first run is an opening).

No identities are used: the result is only geometry. Deterministic (no random numbers).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .load import runs, trim_to_11

TEAM = 11
GATE_M = 3.0
VMAX_MPS = 8.0
EXTRAP_MAX_S = 1.0
UNUSED_COST_M = 15.0
FORCED_COST_M = 1000.0
VEL_FRAMES = 5


def _end_state(px, py, s, a, b, fps):
    """Last position and velocity (mean over the last VEL_FRAMES frames) of run (s, a, b)."""
    x1, y1 = float(px[b - 1, s]), float(py[b - 1, s])
    k = max(a, b - 1 - VEL_FRAMES)
    if b - 1 > k:
        dt = (b - 1 - k) / fps
        return x1, y1, (x1 - float(px[k, s])) / dt, (y1 - float(py[k, s])) / dt
    return x1, y1, 0.0, 0.0


def relink_team(px, py, cols, period, fps: float = 25.0):
    """Lanes of one team. Returns lane_slot (n, 11) int32 (slot column per frame, -1 = none) and link records
    (frame, gap_s, dist_m, forced) for every link (not for lane openings)."""
    n = px.shape[0]
    lane_slot = np.full((n, TEAM), -1, np.int32)
    links = []
    pres = np.isfinite(px)
    for p in [p for p in np.unique(period) if p > 0]:
        idx = np.flatnonzero(period == p)
        lo, hi = int(idx[0]), int(idx[-1]) + 1
        rr = []
        for s in cols:
            for a, b in runs(pres[lo:hi, s]):
                rr.append((lo + int(a), lo + int(b), int(s)))
        rr.sort()
        end = np.full(TEAM, -1)                     # end frame of the lane's last run (exclusive); -1 = never used
        state = [None] * TEAM                       # (x, y, vx, vy) at the end of the lane's last run
        i = 0
        while i < len(rr):
            a0 = rr[i][0]
            group = []
            while i < len(rr) and rr[i][0] == a0:
                group.append(rr[i]); i += 1
            free = [k for k in range(TEAM) if end[k] <= a0]
            if len(free) < len(group):              # cannot happen after trim_to_11; keep the earliest-indexed runs
                group = group[:len(free)]
            C = np.zeros((len(group), len(free)))
            forced = np.zeros_like(C, bool)
            dist = np.zeros_like(C)
            for r, (a, b, s) in enumerate(group):
                x0, y0 = float(px[a, s]), float(py[a, s])
                for c, k in enumerate(free):
                    if end[k] < 0:
                        C[r, c] = UNUSED_COST_M
                        continue
                    gap = (a - end[k] + 1) / fps
                    lx, ly, vx, vy = state[k]
                    h = min(gap, EXTRAP_MAX_S)
                    d = float(np.hypot(x0 - (lx + vx * h), y0 - (ly + vy * h)))
                    dist[r, c] = d
                    forced[r, c] = d > GATE_M + VMAX_MPS * gap
                    C[r, c] = d + (FORCED_COST_M if forced[r, c] else 0.0)
            ri, ci = linear_sum_assignment(C)
            for r, c in zip(ri, ci):
                a, b, s = group[r]
                k = free[c]
                if end[k] >= 0:
                    links.append((a, (a - end[k] + 1) / fps, dist[r, c], bool(forced[r, c])))
                lane_slot[a:b, k] = s
                end[k] = b
                state[k] = _end_state(px, py, s, a, b, fps)
    return lane_slot, links


def relink(px, py, team_sign, period, fps: float = 25.0, trim: bool = True) -> dict:
    """22 lanes: columns 0-10 the +1 team (home), 11-21 the -1 team. Returns lane_x, lane_y (n, 22) float32 (NaN = no
    run), lane_slot (n, 22) int32, links (frame, gap_s, dist_m, forced, team) and the trimmed share."""
    if trim:
        px, py, dropped = trim_to_11(px, py, team_sign, period)
    else:
        dropped = np.zeros(px.shape, bool)
    n = px.shape[0]
    lane_slot = np.full((n, 2 * TEAM), -1, np.int32)
    links = []
    for t_i, side in enumerate((1.0, -1.0)):
        cols = np.flatnonzero(team_sign == side)
        ls, lk = relink_team(px, py, cols, period, fps)
        lane_slot[:, t_i * TEAM:(t_i + 1) * TEAM] = ls
        links += [(*x, t_i) for x in lk]
    has = lane_slot >= 0
    rows = np.arange(n)[:, None]
    sl = np.where(has, lane_slot, 0)
    lane_x = np.where(has, px[rows, sl], np.nan).astype(np.float32)
    lane_y = np.where(has, py[rows, sl], np.nan).astype(np.float32)
    return dict(lane_x=lane_x, lane_y=lane_y, lane_slot=lane_slot,
                links=np.array(links, np.float64).reshape(-1, 5), trimmed_share=float(dropped.sum() / max(np.isfinite(px).sum() + dropped.sum(), 1)))


def window_stats(lane_slot, period, links, fps: float = 25.0, win_s: float = 10.0, cover: float = 0.8) -> dict:
    """Per 10 s window (non-overlapping, inside a period): number of lanes per team present on >= cover of the frames,
    and links per window. Aggregates only."""
    w = int(round(win_s * fps))
    n_ok10, n_ok11, n_win, n_links = 0, 0, 0, []
    lk_frames = links[:, 0].astype(int) if len(links) else np.zeros(0, int)
    lanes_per_team = []
    for p in [p for p in np.unique(period) if p > 0]:
        idx = np.flatnonzero(period == p)
        for s in range(int(idx[0]), int(idx[-1]) + 1 - w + 1, w):
            share = (lane_slot[s:s + w] >= 0).mean(0)
            c = [(share[:TEAM] >= cover).sum(), (share[TEAM:] >= cover).sum()]
            lanes_per_team += c
            n_win += 1
            n_ok10 += int(min(c) >= 10)
            n_ok11 += int(min(c) >= 11)
            n_links.append(int(((lk_frames >= s) & (lk_frames < s + w)).sum()))
    return dict(windows=n_win, share_both_teams_ge10=n_ok10 / max(n_win, 1), share_both_teams_eq11=n_ok11 / max(n_win, 1),
                mean_lanes_per_team=float(np.mean(lanes_per_team)) if lanes_per_team else float("nan"),
                mean_links_per_window=float(np.mean(n_links)) if n_links else float("nan"),
                cover=cover, win_s=win_s)


def link_stats(links) -> dict:
    if not len(links):
        return dict(n_links=0)
    return dict(n_links=int(len(links)), forced_share=float(links[:, 3].mean()),
                gap_s_p50=float(np.median(links[:, 1])), dist_m_p50=float(np.median(links[:, 2])),
                dist_m_p90=float(np.percentile(links[:, 2], 90)))
