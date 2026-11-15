"""Post-rules on the decoded per-frame labels (0 H, 1 A, 2 D; -1 outside the periods), applied after the decoder.
Parameters are fixed here a priori (not tuned on any data); their effect is measured on degraded PFF in the CV.

Order: gap_same_team -> gap_sideline -> set_pieces -> kickoff (later rules override earlier ones).
- gap_same_team: a raw-ball-missing run with the same team just before and just after -> that team throughout.
- gap_sideline : a missing run of >= 2 s whose ball reappears within 1.5 m of a touchline and slow (< 3 m/s over the
                 next 0.4 s; a throw-in being taken) -> D during the run.
- set_pieces   : Smart Tagging SET PIECES (frame = round(s * 25)): the restart kick = the first ball-speed crossing of
                 6 m/s after a slow ball (<= 3 m/s or unseen over the previous 0.8 s) within [start - 1 s, end + 2 s];
                 D from the ball stopping (the last frame with a moving seen ball, >= 3 m/s, at most 30 s before) until
                 the kick; the first spell from the kick gets the restart team: the Team tag, or for N/A (every throw-in)
                 the team that did not have the ball before the stoppage.
- kickoff      : D from each period's first frame until its kick-off kick (the first kick within 3 m of the centre spot,
                 inside the Kick Off tag's interval +- 2 s when there is one).
"""
from __future__ import annotations

import numpy as np

H, A, D = 0, 1, 2
KICK_MPS, SLOW_MPS, MOVING_MPS = 6.0, 3.0, 3.0
MAX_DEAD_ADD_S = 8.0       # set pieces: D added at most this long before the kick (an unseen ball is not a stopped ball)
KICKOFF_SEARCH_S = 180.0   # kick-off: searched in the first 3 min of a period
RULES = ("gap_same_team", "gap_sideline", "set_pieces", "kickoff")


def _runs(mask):
    e = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))


def kick_frames(speed, period, fps=25.0) -> np.ndarray:
    """Frames where the (smoothed, seen) ball speed crosses KICK_MPS after a slow or unseen ball over the last 0.8 s."""
    sp = np.asarray(speed, float)
    slow = ~(sp > SLOW_MPS)                                   # NaN (unseen) counts as slow
    w = int(round(0.8 * fps))
    cs = np.r_[0, np.cumsum(slow)]
    i = np.arange(len(sp))
    lo = np.clip(i - w, 0, len(sp))
    any_slow = (cs[i] - cs[lo]) > 0
    up = (sp >= KICK_MPS) & ~(np.r_[np.inf, sp[:-1]] >= KICK_MPS) & any_slow & (period > 0)
    return np.flatnonzero(up)


def gap_same_team(lab, missing, period, fps=25.0):
    """Inside the gap: the other team -> that team; D runs shorter than 2 s -> that team; longer D runs (a real stoppage
    with the same team restarting) are kept."""
    out = lab.copy()
    n = len(lab)
    for a, b in _runs(missing & (period > 0)):
        if a > 0 and b < n and period[a - 1] == period[b] and out[a - 1] == out[b] and out[b] in (H, A):
            team = out[b]
            seg = out[a:b].copy()
            seg[seg == 1 - team] = team
            for c, e in _runs(seg == D):
                if e - c < 2 * fps:
                    seg[c:e] = team
            out[a:b] = seg
    return out


def gap_sideline(lab, missing, by, speed, period, half_width, fps=25.0):
    out = lab.copy()
    n = len(lab)
    k = int(round(0.4 * fps))
    for a, b in _runs(missing & (period > 0)):
        if b - a < 2 * fps or b >= n or period[b] != period[a] or not np.isfinite(by[b]):
            continue
        if abs(by[b]) >= half_width - 1.5 and np.nanmax(np.r_[speed[b:b + k], 0.0]) < SLOW_MPS:
            out[a:b] = D
    return out


def set_pieces(lab, tags, kicks, speed, period, fps=25.0):
    """tags: [{start_frame, end_frame, team: 0 / 1 / None}]"""
    out = lab.copy()
    n = len(lab)
    sp = np.asarray(speed, float)
    for tg in tags:
        lo, hi = max(tg["start_frame"] - int(fps), 1), min(tg["end_frame"] + int(2 * fps), n - 1)
        ks = kicks[(kicks >= lo) & (kicks <= hi)]
        if not len(ks):
            continue
        # anchor: the decoder's own restart (end of a D run) inside the window, else the tag start; the kick nearest it
        # (the first kick in the window can be the one that sent the ball out)
        w = out[lo:hi + 1]
        ends = np.flatnonzero((w[:-1] == D) & (w[1:] != D)) + lo + 1
        anchor = int(ends[0]) if len(ends) else int(tg["start_frame"])
        k = int(ks[np.argmin(np.abs(ks - anchor))])
        p = period[k]
        start = max(k - int(MAX_DEAD_ADD_S * fps), int(np.flatnonzero(period == p)[0]))
        seg = sp[start:k - int(0.5 * fps)]
        mov = np.flatnonzero(seg >= MOVING_MPS)
        stop = start + int(mov[-1]) + 1 if len(mov) else start
        team = tg.get("team")
        if team is None:
            before = out[stop - 1] if stop > 0 and period[stop - 1] == p else -1
            team = (1 - before) if before in (H, A) else None
        out[stop:k] = D
        if team is None:
            continue
        g = k
        while g < n and period[g] == p and out[g] == D and g - k < 3 * fps:
            g += 1
        if g >= n or period[g] != p or out[g] == D:
            continue
        e, first = g, out[g]
        while e < n and period[e] == p and out[e] == first:
            e += 1
        out[k:e] = team
    return out


def kickoff(lab, period, kicks, bx, by, kickoff_tags=None, fps=25.0):
    """kickoff_tags: {period: (start_frame, end_frame)} (optional)."""
    out = lab.copy()
    for p in [p for p in np.unique(period) if p > 0]:
        idx = np.flatnonzero(period == p)
        a = int(idx[0])
        ks = kicks[(kicks >= a) & (kicks <= idx[-1]) & (kicks < a + KICKOFF_SEARCH_S * fps)]
        prev = np.clip(ks - 1, 0, len(bx) - 1)
        near = np.isfinite(bx[prev]) & (np.hypot(bx[prev], by[prev]) <= 3.0)
        ks = ks[near]
        tg = (kickoff_tags or {}).get(int(p))
        if tg is not None:
            sel = ks[(ks >= tg[0] - 2 * fps) & (ks <= tg[1] + 2 * fps)]
            ks = sel if len(sel) else ks
        if len(ks):
            out[a:int(ks[0])] = D
    return out


def apply(lab, ctx: dict, rules=RULES):
    """ctx: missing, bx, by (filled), raw_by, speed (smoothed, NaN when not seen), period, half_width, tags,
    kickoff_tags."""
    out = lab.copy()
    kicks = kick_frames(ctx["speed"], ctx["period"])
    if "gap_same_team" in rules:
        out = gap_same_team(out, ctx["missing"], ctx["period"])
    if "gap_sideline" in rules:
        out = gap_sideline(out, ctx["missing"], ctx["raw_by"], ctx["speed"], ctx["period"], ctx["half_width"])
    if "set_pieces" in rules and ctx.get("tags"):
        out = set_pieces(out, ctx["tags"], kicks, ctx["speed"], ctx["period"])
    if "kickoff" in rules:
        out = kickoff(out, ctx["period"], kicks, ctx["bx"], ctx["by"], ctx.get("kickoff_tags"))
    return out


def simulate_tags(lab, period, by, half_width, rng, fps=25.0, keep=0.6, team_ok=0.9):
    """PFF has no Smart Tagging: SET PIECES-like tags simulated from the answer key's restarts (D -> H/A after >= 2 s of D),
    kept with probability `keep`, an interval of [restart - U(1, 4) s, restart + U(2, 6) s], the right team with
    probability `team_ok`, and N/A when the restart is within 2 m of a touchline (throw-ins). Evaluation only."""
    tags = []
    for a, b in _runs((lab == D) & (period > 0)):
        if b >= len(lab) or b - a < 2 * fps or period[b] != period[a] or lab[b] not in (H, A):
            continue
        if rng.random() > keep:
            continue
        team = int(lab[b]) if rng.random() < team_ok else int(1 - lab[b])
        if np.isfinite(by[b]) and abs(by[b]) >= half_width - 2.0:
            team = None
        tags.append(dict(start_frame=int(b - rng.uniform(1, 4) * fps), end_frame=int(b + rng.uniform(2, 6) * fps), team=team))
    return tags
