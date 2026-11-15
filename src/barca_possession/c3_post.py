"""M1 C3 frame-level post-processing (Part 3 of 2026-11-09). Numpy only.

- timing_fix (item b): every possession switch (into CONTROL, out of CONTROL, a team change, a restart) is moved to the
  nearest detected touch within ``tol_s`` (same period, never across another switch).
- merge_tiny (item j; source: coderhema, Barca hackathon): H / A spells shorter than ``M`` seconds between two spells of the
  other team become that team's TRAVELLING (no owner).
- passes_from: passes = CONTROL -> TRAVELLING transitions with a known owner.
States: 0 DEAD, 1 CONTROL, 2 TRAVELLING, 3 LOOSE.
"""
from __future__ import annotations

import numpy as np


def _had(state, team, period, home_team):
    """Team-level H / A / D view (0 H, 1 A, 2 D) as used for scoring; kept local, this package stays tag-free."""
    v = np.full(len(state), -1, np.int8)
    own = np.isin(state, (1, 2)) & (team >= 0)
    v[own] = np.where(team[own] == home_team, 0, 1)
    idx = np.maximum.accumulate(np.where(own, np.arange(len(v)), -1))
    starts = np.r_[0, np.flatnonzero(period[1:] != period[:-1]) + 1]
    pstart = starts[np.searchsorted(starts, np.arange(len(v)), "right") - 1]
    fill = ~own & (idx >= pstart)
    v[fill] = v[idx[fill]]
    v[state == 0] = 2
    return v


def timing_fix(state, player, team, period, touch_frames, fps, tol_s=1.0):
    state, player, team = np.array(state), np.array(player), np.array(team)
    n = len(state)
    tf = np.sort(np.asarray(touch_frames, int))
    if not len(tf) or n < 2:
        return state, player, team
    ch = np.flatnonzero((state[1:] != state[:-1]) | (team[1:] != team[:-1])) + 1
    tol = int(round(tol_s * fps))
    keep = [i for i in ch if state[i] != 0 and period[i] == period[i - 1]
            and (state[i] == 1 or state[i - 1] in (0, 1) or team[i] != team[i - 1])]
    chs = np.r_[0, ch, n]
    s0, p0, t0 = state.copy(), player.copy(), team.copy()
    for i in keep:
        j = np.searchsorted(tf, i)
        cands = [x for x in (tf[j - 1] if j > 0 else None, tf[j] if j < len(tf) else None) if x is not None and abs(x - i) <= tol]
        if not cands:
            continue
        x = int(min(cands, key=lambda v: abs(v - i)))
        q = int(np.searchsorted(chs, i))
        lo, hi = chs[q - 1], (chs[q + 1] if q + 1 < len(chs) else n)
        if not (lo < x < hi) or period[x] != period[i]:
            continue
        if x < i:
            state[x:i], player[x:i], team[x:i] = s0[i], p0[i], t0[i]
        elif x > i:
            state[i:x], player[i:x], team[i:x] = s0[i - 1], p0[i - 1], t0[i - 1]
    return state, player, team


def merge_tiny(state, player, team, period, home_team, fps, M):
    state, player, team = np.array(state), np.array(player), np.array(team)
    v = _had(state, team, period, home_team)
    b = np.flatnonzero(np.r_[True, (v[1:] != v[:-1]) | (period[1:] != period[:-1])])
    e = np.r_[b[1:], len(v)]
    lim = M * fps
    for i in range(1, len(b) - 1):
        if not (v[b[i]] in (0, 1) and e[i] - b[i] < lim and v[b[i - 1]] == v[b[i + 1]] and v[b[i - 1]] in (0, 1)
                and v[b[i - 1]] != v[b[i]] and period[b[i - 1]] == period[b[i + 1]]):
            continue
        idx = np.flatnonzero(team[b[i - 1]:b[i]] >= 0)
        if not len(idx):
            continue
        prev_team = team[b[i - 1] + idx[-1]]
        sel = np.arange(b[i], e[i])
        sel = sel[state[sel] != 0]
        state[sel], team[sel], player[sel] = 2, prev_team, -1
    return state, player, team


def passes_from(state, player):
    state, player = np.asarray(state), np.asarray(player)
    i = np.flatnonzero((state[1:] == 2) & (state[:-1] == 1) & (player[:-1] >= 0))
    return [(int(x), int(player[x])) for x in i]
