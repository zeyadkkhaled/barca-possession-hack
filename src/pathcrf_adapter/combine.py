"""PathCRF outputs (5 fps steps, m1pc/predict.py) + the frozen C3 (per frame) -> per-frame state, player id, team id.

C3 always decides DEAD vs alive (PathCRF has no DEAD state; plan: C3's tag-free alive / dead layer). Variants:
- pathcrf     : alive frames from PathCRF's CRF path: a self-link p->p = CONTROL by p; a link A->B = ball between players,
                TRAVELLING or LOOSE by PathCRF's head (P(LOOSE) > 0.5), team = A's team.
- combo_c3tl  : as pathcrf, but TRAVELLING vs LOOSE between players from C3 (C3's TRAVELLING / LOOSE where it says so,
                else TRAVELLING).
- combo_who   : C3's states everywhere; PathCRF only names the player (and team) in C3's CONTROL frames: its self-link
                player, else its most likely ball holder.
Each 5 fps step holds for the frames up to the next step; frames outside PathCRF's coverage keep C3's output.
"""
from __future__ import annotations

import numpy as np

VARIANTS = ("pathcrf", "combo_c3tl", "combo_who")


def combine(pred: dict, c3: dict, period: np.ndarray, variant: str):
    n = len(period)
    steps = pred["steps"].astype(int)
    P = len(pred["player_id"])
    pid, ptm = pred["player_id"].astype(np.int64), pred["player_team_id"].astype(np.int64)
    j = np.searchsorted(steps, np.arange(n), "right") - 1
    ok = (j >= 0) & (period > 0)
    jj = np.clip(j, 0, len(steps) - 1)
    ok &= pred["period"][jj] == period
    ok &= pred["covered"][jj] > 0
    cs, cp, ct = (np.asarray(c3[k]).astype(np.int64) for k in ("state", "player", "team"))
    st, pl, tm = cs.copy(), cp.copy(), ct.copy()
    ps, pd = pred["path_src"][jj].astype(int), pred["path_dst"][jj].astype(int)
    is_ctrl = (ps == pd) & (ps >= 0) & (ps < P)
    src_player = (ps >= 0) & (ps < P)
    alive = ok & (cs != 0)
    if variant in ("pathcrf", "combo_c3tl"):
        between = alive & ~is_ctrl
        ctrl = alive & is_ctrl
        st[ctrl] = 1
        pl[ctrl] = pid[ps[ctrl]]
        tm[ctrl] = ptm[ps[ctrl]]
        if variant == "pathcrf":
            st[between] = np.where(pred["p_loose"][jj][between] > 0.5, 3, 2)
        else:
            st[between] = np.where(np.isin(cs[between], (2, 3)), cs[between], 2)
        pl[between] = -1
        tm[between] = np.where(src_player[between], ptm[np.clip(ps[between], 0, P - 1)], -1)
    elif variant == "combo_who":
        c = ok & (cs == 1)
        hold = np.where(is_ctrl, ps, np.argmax(pred["p_has"][jj], 1))
        pl[c] = pid[hold[c]]
        tm[c] = ptm[hold[c]]
    else:
        raise ValueError(variant)
    st[period <= 0] = 0
    return st.astype(np.int8), pl, tm
