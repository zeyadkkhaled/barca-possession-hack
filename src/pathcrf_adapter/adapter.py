"""Our HPC package (.npz per match, 29.97 fps) -> PathCRF inputs and targets (5 fps, 10 s windows, 22 players + 4 outside nodes).

Node features (PathCRF order, model 140): side (-1 left team / +1 right team), keeper flag, x, y, vx, vy, speed, accel, then with
the ball input ON: ball x, ball y (our smoothed ball, broadcast to every node) and ball_seen (1 = observed / short-gap
interpolated, 0 = filled). Coordinates: pitch corner origin, scaled to 105 x 68 m. Left team = the team that defends the left goal
in that period (from home_attacks_right). Missing ball: linear fill inside the period (edges held), ball_seen = 0.

Players: per window, the 11 columns of each team with the most frames on the pitch inside the window (a substitution inside the
window keeps the player with more frames); a gap inside the window is filled from the nearest frame of that player. A team with
fewer than 11 present players gets zero nodes (player_mask 0), as PathCRF's padding does.

Targets per 5 fps step (answer key = labels v2; only training target and scoring, never input):
- CONTROL by player p                       -> self-link p->p
- TRAVELLING or LOOSE between A and B       -> link A->B (A = last controller before, B = next controller after, same period, no
                                               DEAD in between)
- TRAVELLING or LOOSE followed by DEAD      -> link A->outside node (the side where the ball is at the start of the DEAD stretch)
- DEAD, UNKNOWN, unscored, or no A / B, or A / B not among the window's 22 nodes -> masked (no loss); the frame stays in the
  input window as context.
- head target (TRAVELLING vs LOOSE): 0 TRAVELLING, 1 LOOSE, -1 otherwise (only scored frames).
LOOSE -> the A->B link (not masked): PathCRF's own labels already treat every in-between frame (pass, deflection, rebound) as
the link between the two touching players, so the pretrained weights expect it; masking LOOSE would cut the CRF chain at every
loose ball. Telling TRAVELLING from LOOSE is the job of the extra head.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PITCH_X, PITCH_Y = 105.0, 68.0
STEP = 6                      # 29.97 fps -> 4.995 fps
FPS = 29.97
WIN = 50                      # 10 s at 5 fps
TEAM = 11
N_NODES = 2 * TEAM + 4        # + out_left, out_right, out_bottom, out_top
OUT_XY = ((0.0, PITCH_Y / 2), (PITCH_X, PITCH_Y / 2), (PITCH_X / 2, 0.0), (PITCH_X / 2, PITCH_Y))
DEAD, CONTROL, TRAV, LOOSE = 0, 1, 2, 3
K_MASK, K_SELF, K_LINK, K_OUT = 0, 1, 2, 3


def load_match(path) -> dict:
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}
    for k, v in d.items():
        if v.dtype == np.float16:
            d[k] = v.astype(np.float32)
    return d


def folds(root) -> list[list[int]]:
    return json.loads((Path(root) / "folds.json").read_text())["folds"]


def _fill_1d(v, per):
    """Linear fill of NaN inside each period; edges held."""
    out = v.copy()
    for p in np.unique(per):
        idx = np.flatnonzero(per == p)
        ok = np.isfinite(v[idx])
        if ok.sum() == 0:
            out[idx] = PITCH_X / 2 if v is None else 0.0
            continue
        out[idx] = np.interp(idx, idx[ok], v[idx][ok])
    return out


def _side_out(bx, by, L, W):
    """Index of the outside node nearest to a ball position (centre origin): 0 left, 1 right, 2 bottom, 3 top."""
    if not np.isfinite(bx):
        return -1
    d = (bx + L / 2, L / 2 - bx, by + W / 2, W / 2 - by)
    return int(np.argmin(d))


class Match:
    """One match at 5 fps: per-step player positions (all columns), ball, labels, and the slot-level targets."""

    def __init__(self, d: dict):
        self.d = d
        self.mid = int(d["match_id"])
        L, W = (float(x) for x in d["pitch"])
        self.L, self.W = L, W
        per_f = d["period"].astype(int)
        n = len(per_f)
        # steps: every 6th frame of each period (so steps never straddle a period boundary)
        rows = []
        for p in np.unique(per_f[per_f > 0]):
            idx = np.flatnonzero(per_f == p)
            rows.append(idx[::STEP])
        self.frames = np.concatenate(rows) if rows else np.zeros(0, int)
        f = self.frames
        self.n_frames = n
        self.period = per_f[f]
        sx, sy = PITCH_X / L, PITCH_Y / W
        self.px = (d["player_x"][f] + L / 2) * sx                          # (S, P) NaN = absent
        self.py = (d["player_y"][f] + W / 2) * sy
        bx = np.where(np.isfinite(d["ball_x"]), d["ball_x"], np.nan)
        by = np.where(np.isfinite(d["ball_y"]), d["ball_y"], np.nan)
        self.ball_seen = np.isin(d["ball_flag"][f], (1, 2)).astype(np.float32)
        self.bx = (_fill_1d(bx, per_f)[f] + L / 2) * sx
        self.by = (_fill_1d(by, per_f)[f] + W / 2) * sy
        self.bx = np.nan_to_num(self.bx, nan=PITCH_X / 2)
        self.by = np.nan_to_num(self.by, nan=PITCH_Y / 2)
        self.side = d["player_side"].astype(int)                           # 0 home / 1 away per column
        har = np.asarray(d["home_attacks_right"], bool)
        # left team per step: home defends the left goal when it attacks right
        self.home_left = np.array([bool(har[p - 1]) if 1 <= p <= len(har) else True for p in self.period])
        self.keeper = self._keepers()
        self._kinematics()
        self._targets(d, bx, by)

    def _keepers(self):
        """Per (period, side): the column whose mean x is furthest from halfway (PathCRF's detect_keepers rule)."""
        kp = {}
        for p in np.unique(self.period):
            sel = self.period == p
            for s in (0, 1):
                cols = np.flatnonzero(self.side == s)
                with np.errstate(all="ignore"):
                    m = np.nanmean(self.px[sel][:, cols], 0)
                if np.isfinite(m).any():
                    kp[(int(p), s)] = int(cols[np.nanargmax(np.abs(np.nan_to_num(m, nan=PITCH_X / 2) - PITCH_X / 2))])
        return kp

    def _kinematics(self):
        """vx, vy, speed, accel at 5 fps (central differences inside a period, on positions filled per player)."""
        dt = STEP / FPS
        S, P = self.px.shape
        self.vx = np.zeros((S, P), np.float32); self.vy = np.zeros((S, P), np.float32); self.acc = np.zeros((S, P), np.float32)
        for p in np.unique(self.period):
            idx = np.flatnonzero(self.period == p)
            if len(idx) < 3:
                continue
            for arr, out in ((self.px, self.vx), (self.py, self.vy)):
                g = np.gradient(arr[idx], dt, axis=0)
                out[idx] = np.nan_to_num(g)
            sp = np.hypot(self.vx[idx], self.vy[idx])
            self.acc[idx] = np.nan_to_num(np.gradient(sp, dt, axis=0))
        self.speed = np.hypot(self.vx, self.vy)

    def _targets(self, d, bx, by):
        """Per step: kind (mask / self / link / out), a_slot, b_slot (column or outside index), head target, scored."""
        st = d["label_state"].astype(int)
        slot = d["label_player_slot"].astype(int)
        per = d["period"].astype(int)
        scored = d["label_score_mask"] & (st <= 3) & (per > 0)
        n = len(st)
        # last controller before / next controller after each frame (same period, no DEAD in between)
        last = np.full(n, -1); nxt = np.full(n, -1); nxt_out = np.full(n, -1)
        cur = -1
        for i in range(n):
            if i and per[i] != per[i - 1]:
                cur = -1
            if st[i] == DEAD:
                cur = -1
            elif st[i] == CONTROL and slot[i] >= 0:
                cur = slot[i]
            last[i] = cur
        cur, cur_out = -1, -1
        for i in range(n - 1, -1, -1):
            if i < n - 1 and per[i] != per[i + 1]:
                cur, cur_out = -1, -1
            if st[i] == DEAD:
                cur = -1
                cur_out = _side_out(bx[i], by[i], self.L, self.W)       # ball position at the start of the DEAD stretch
            elif st[i] == CONTROL and slot[i] >= 0:
                cur, cur_out = slot[i], -1
            nxt[i], nxt_out[i] = cur, cur_out
        f = self.frames
        s_st, s_slot, s_sc = st[f], slot[f], scored[f]
        kind = np.zeros(len(f), np.int8); a = np.full(len(f), -1, np.int16); b = np.full(len(f), -1, np.int16)
        ctrl = s_sc & (s_st == CONTROL) & (s_slot >= 0)
        kind[ctrl], a[ctrl], b[ctrl] = K_SELF, s_slot[ctrl], s_slot[ctrl]
        mid = s_sc & np.isin(s_st, (TRAV, LOOSE)) & (last[f] >= 0)
        to_p = mid & (nxt[f] >= 0)
        to_o = mid & (nxt[f] < 0) & (nxt_out[f] >= 0)
        kind[to_p], a[to_p], b[to_p] = K_LINK, last[f][to_p], nxt[f][to_p]
        kind[to_o], a[to_o], b[to_o] = K_OUT, last[f][to_o], nxt_out[f][to_o]
        self.kind, self.a, self.b = kind, a, b
        self.tl = np.where(s_sc & (s_st == TRAV), 0, np.where(s_sc & (s_st == LOOSE), 1, -1)).astype(np.int8)
        self.live_scored = s_sc & (s_st != DEAD)                          # a scored live frame (window-skip rule)

    # ------------------------------------------------------------ windows
    def window_starts(self, stride: int, length: int = WIN) -> list[int]:
        """Window start steps inside each period (the last window of a period is aligned to its end, as PathCRF does)."""
        out = []
        for p in np.unique(self.period):
            idx = np.flatnonzero(self.period == p)
            if len(idx) < length:
                continue
            a, e = idx[0], idx[-1] + 1
            st = list(range(a, e - length + 1, stride))
            if st[-1] != e - length:
                st.append(e - length)
            out += st
        return out

    def window(self, s: int, flip_x=False, flip_y=False, length: int = WIN, ball=True):
        """x (T, 26, F), player_mask (T, 26), edge target (T,) as src*26+dst, loss mask (T,), head target (T,),
        node_slot (26,) = player column per node (-1 = outside / padding), left_is_home."""
        w = slice(s, s + length)
        T = length
        home_left = bool(self.home_left[s])
        left_side = 0 if home_left else 1
        px, py = self.px[w], self.py[w]
        pres = np.isfinite(px)
        node_slot = np.full(N_NODES, -1, np.int64)
        pmask = np.zeros((T, N_NODES), np.float32)
        for t_i, side in enumerate((left_side, 1 - left_side)):
            cols = np.flatnonzero(self.side == side)
            cnt = pres[:, cols].sum(0)
            order = cols[np.argsort(-cnt, kind="stable")]
            order = [c for c in order if pres[:, c].any()][:TEAM]
            for j, c in enumerate(order):
                node_slot[t_i * TEAM + j] = c
        F = 8 + (3 if ball else 0)
        x = np.zeros((T, N_NODES, F), np.float32)
        p0 = int(self.period[s])
        kp = {self.keeper.get((p0, 0)), self.keeper.get((p0, 1))}
        for nd in range(2 * TEAM):
            c = node_slot[nd]
            if c < 0:
                continue
            ok = pres[:, c]
            ii = np.arange(T)
            near = ii[ok][np.clip(np.searchsorted(ii[ok], ii), 0, ok.sum() - 1)]          # nearest present step (fill)
            prev = ii[ok][np.clip(np.searchsorted(ii[ok], ii) - 1, 0, ok.sum() - 1)]
            near = np.where(np.abs(prev - ii) < np.abs(near - ii), prev, near)
            x[:, nd, 0] = -1.0 if nd < TEAM else 1.0
            x[:, nd, 1] = 1.0 if c in kp else 0.0
            x[:, nd, 2] = px[near, c]; x[:, nd, 3] = py[near, c]
            x[:, nd, 4] = self.vx[w][near, c]; x[:, nd, 5] = self.vy[w][near, c]
            x[:, nd, 6] = self.speed[w][near, c]; x[:, nd, 7] = self.acc[w][near, c]
            pmask[:, nd] = 1.0
        for k, (ox, oy) in enumerate(OUT_XY):
            x[:, 2 * TEAM + k, 2], x[:, 2 * TEAM + k, 3] = ox, oy
            pmask[:, 2 * TEAM + k] = 1.0
        if ball:
            x[:, :, 8] = self.bx[w][:, None]; x[:, :, 9] = self.by[w][:, None]; x[:, :, 10] = self.ball_seen[w][:, None]
        # targets in node space
        slot2node = {int(c): i for i, c in enumerate(node_slot) if c >= 0}
        kind, a, b = self.kind[w], self.a[w], self.b[w]
        src = np.zeros(T, np.int64); dst = np.zeros(T, np.int64); m = np.zeros(T, bool)
        for i in range(T):
            if kind[i] == K_MASK:
                continue
            na = slot2node.get(int(a[i]))
            if kind[i] == K_OUT:
                nb = 2 * TEAM + int(b[i])
            else:
                nb = slot2node.get(int(b[i]))
            if na is None or nb is None:
                continue
            src[i], dst[i], m[i] = na, nb, True
        tl = self.tl[w].astype(np.int64)
        if flip_x or flip_y:
            x, src, dst, pmask, node_slot = flip(x, src, dst, pmask, node_slot, flip_x, flip_y, ball)
        return dict(x=x, pmask=pmask, src=src, dst=dst, mask=m, tl=tl, node_slot=node_slot,
                    live=self.live_scored[w].copy(), step0=s)


def flip(x, src, dst, pmask, node_slot, fx, fy, ball=True):
    """PathCRF's random flip: x -> 105 - x swaps the team blocks (left team stays first); y -> 68 - y swaps bottom / top."""
    x = x.copy()
    perm = np.arange(N_NODES)
    if fx:
        x[:, :, 2] = PITCH_X - x[:, :, 2]; x[:, :, 4] = -x[:, :, 4]
        if ball:
            x[:, :, 8] = PITCH_X - x[:, :, 8]
        perm = np.r_[np.arange(TEAM, 2 * TEAM), np.arange(TEAM), 2 * TEAM + np.array([1, 0, 2, 3])]
        x[:, :2 * TEAM, 0] = -x[:, :2 * TEAM, 0]
    if fy:
        x[:, :, 3] = PITCH_Y - x[:, :, 3]; x[:, :, 5] = -x[:, :, 5]
        if ball:
            x[:, :, 9] = PITCH_Y - x[:, :, 9]
        p2 = np.arange(N_NODES); p2[2 * TEAM + 2], p2[2 * TEAM + 3] = 2 * TEAM + 3, 2 * TEAM + 2
        perm = perm[p2]
    # new node i holds old node perm[i]; labels map old -> new
    inv = np.empty(N_NODES, np.int64); inv[perm] = np.arange(N_NODES)
    x = x[:, perm]
    x[:, 2 * TEAM:, 2:4] = np.array(OUT_XY, np.float32)[None]               # outside nodes keep their fixed positions
    x[:, 2 * TEAM:, 0] = 0.0
    return x, inv[src], inv[dst], pmask[:, perm], node_slot[perm]


class WindowSet:
    """Torch-style dataset over several matches. skip_dead: drop windows with no scored live frame (training only)."""

    def __init__(self, root, mids, stride: int, train: bool, skip_dead: bool = True, ball: bool = True, matches: dict | None = None):
        self.m = matches if matches is not None else {m: Match(load_match(Path(root) / "matches" / f"{m}.npz")) for m in mids}
        self.train, self.ball = train, ball
        self.index, self.n_all = [], 0
        for m in mids:
            M = self.m[m]
            for s in M.window_starts(stride):
                self.n_all += 1
                if train and skip_dead and not M.live_scored[s:s + WIN].any():
                    continue
                self.index.append((m, s))
        self.skipped_share = 1 - len(self.index) / max(self.n_all, 1)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        m, s = self.index[i]
        fx = fy = False
        if self.train:
            fx, fy = bool(np.random.randint(2)), bool(np.random.randint(2))
        w = self.m[m].window(s, fx, fy, ball=self.ball)
        w["mid"] = m
        return w
