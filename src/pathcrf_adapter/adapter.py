"""Built lane files (pathcrf_adapter.build, 25 fps) -> PathCRF inputs and targets (5 fps, 10 s windows, 22 players + 4
outside nodes). Adapted from M1's adapter (29.97 fps, player ids) to the hackathon form:

- STEP 5 at 25 fps (5 fps), steps = every 5th frame of each period (the same grid as C3-T's rows);
- nodes 0-10 = home lanes, 11-21 = away lanes (barca_possession.relink). Our convention (centre origin, home attacks +x
  in every period) makes home PathCRF's left team in every window, so no per-period side swap is needed;
- coordinates: corner origin, scaled to 105 x 68 m (PathCRF's pitch) from our 105.3 x 68 m;
- node features (model 140 order): side (-1 left / +1 right), keeper flag (per window and team: the lane whose mean x is
  furthest from halfway), x, y, vx, vy, speed, accel; with the ball input ON also ball x, ball y (step-1 ball fill:
  smoother + ball-from-players) and ball_seen (1 = seen or smoothed sample, 0 = estimated);
- a lane absent on some steps of the window is filled from its nearest present step; a lane absent in the whole window
  is a zero node (player_mask 0), as PathCRF's padding does;
- targets come precomputed per frame from build.py (kind / a / b / tl in node space), masked where the answer key has no
  link or its player is in no lane.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PITCH_X, PITCH_Y = 105.0, 68.0
FPS = 25.0
STEP = 5                      # 25 fps -> 5 fps
WIN = 50                      # 10 s at 5 fps
TEAM = 11
N_NODES = 2 * TEAM + 4        # + out_left, out_right, out_bottom, out_top
OUT_XY = ((0.0, PITCH_Y / 2), (PITCH_X, PITCH_Y / 2), (PITCH_X / 2, 0.0), (PITCH_X / 2, PITCH_Y))
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


def match_path(root, mid) -> Path:
    return Path(root) / "pathcrf" / f"{mid}.npz"


class Match:
    """One match at 5 fps: lane positions and kinematics, ball, targets (if the file has labels)."""

    def __init__(self, d: dict):
        self.d = d
        self.mid = int(d["match_id"]) if "match_id" in d else -1
        L, W = (float(x) for x in d["pitch"])
        self.L, self.W = L, W
        per_f = d["period"].astype(int)
        self.n_frames = len(per_f)
        rows = [np.flatnonzero(per_f == p)[::STEP] for p in np.unique(per_f[per_f > 0])]
        self.frames = np.concatenate(rows) if rows else np.zeros(0, int)
        f = self.frames
        self.period = per_f[f]
        sx, sy = PITCH_X / L, PITCH_Y / W
        self.px = (d["lane_x"][f] + L / 2) * sx                         # (S, 22) NaN = lane empty
        self.py = (d["lane_y"][f] + W / 2) * sy
        self.bx = np.nan_to_num((d["ball_x"][f] + L / 2) * sx, nan=PITCH_X / 2).astype(np.float32)
        self.by = np.nan_to_num((d["ball_y"][f] + W / 2) * sy, nan=PITCH_Y / 2).astype(np.float32)
        self.ball_seen = d["ball_seen"][f].astype(np.float32)
        self._kinematics()
        self.labelled = "kind" in d
        if self.labelled:
            self.kind, self.a, self.b = d["kind"][f], d["a"][f], d["b"][f]
            self.tl = d["tl"][f]
            self.live_scored = d["live_scored"][f].astype(bool)
        else:
            S = len(f)
            self.kind = np.zeros(S, np.int8); self.a = self.b = np.full(S, -1, np.int16)
            self.tl = np.full(S, -1, np.int8); self.live_scored = np.zeros(S, bool)

    def _kinematics(self):
        """vx, vy, speed, accel at 5 fps (central differences inside a period, on positions filled per lane)."""
        dt = STEP / FPS
        S, P = self.px.shape
        self.vx = np.zeros((S, P), np.float32); self.vy = np.zeros((S, P), np.float32); self.acc = np.zeros((S, P), np.float32)
        for p in np.unique(self.period):
            idx = np.flatnonzero(self.period == p)
            if len(idx) < 3:
                continue
            for arr, out in ((self.px, self.vx), (self.py, self.vy)):
                out[idx] = np.nan_to_num(np.gradient(arr[idx], dt, axis=0))
            sp = np.hypot(self.vx[idx], self.vy[idx])
            self.acc[idx] = np.nan_to_num(np.gradient(sp, dt, axis=0))
        self.speed = np.hypot(self.vx, self.vy)

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
        """x (T, 26, F), player_mask (T, 26), src / dst (T,) node targets, loss mask (T,), head target (T,),
        node_slot (26,) = lane per node (-1 = outside / empty lane)."""
        w = slice(s, s + length)
        T = length
        px, py = self.px[w], self.py[w]
        pres = np.isfinite(px)
        F = 8 + (3 if ball else 0)
        x = np.zeros((T, N_NODES, F), np.float32)
        pmask = np.zeros((T, N_NODES), np.float32)
        node_slot = np.full(N_NODES, -1, np.int64)
        keep = pres.any(0)
        mx = np.full(px.shape[1], np.nan)
        mx[keep] = np.nanmean(px[:, keep], 0)
        kp = set()
        for t_i in range(2):
            lanes = np.arange(t_i * TEAM, (t_i + 1) * TEAM)
            lanes = lanes[keep[lanes]]
            if len(lanes):
                kp.add(int(lanes[np.argmax(np.abs(mx[lanes] - PITCH_X / 2))]))
        ii = np.arange(T)
        for nd in np.flatnonzero(keep):
            ok = pres[:, nd]
            j = np.searchsorted(ii[ok], ii)
            near = ii[ok][np.clip(j, 0, ok.sum() - 1)]
            prev = ii[ok][np.clip(j - 1, 0, ok.sum() - 1)]
            near = np.where(np.abs(prev - ii) < np.abs(near - ii), prev, near)       # nearest present step
            x[:, nd, 0] = -1.0 if nd < TEAM else 1.0
            x[:, nd, 1] = 1.0 if nd in kp else 0.0
            x[:, nd, 2] = px[near, nd]; x[:, nd, 3] = py[near, nd]
            x[:, nd, 4] = self.vx[w][near, nd]; x[:, nd, 5] = self.vy[w][near, nd]
            x[:, nd, 6] = self.speed[w][near, nd]; x[:, nd, 7] = self.acc[w][near, nd]
            pmask[:, nd] = 1.0
            node_slot[nd] = nd
        for k, (ox, oy) in enumerate(OUT_XY):
            x[:, 2 * TEAM + k, 2], x[:, 2 * TEAM + k, 3] = ox, oy
            pmask[:, 2 * TEAM + k] = 1.0
        if ball:
            x[:, :, 8] = self.bx[w][:, None]; x[:, :, 9] = self.by[w][:, None]; x[:, :, 10] = self.ball_seen[w][:, None]
        kind, a, b = self.kind[w], self.a[w].astype(np.int64), self.b[w].astype(np.int64)
        m = (kind != K_MASK) & (a >= 0) & (b >= 0)
        # a target on a lane that is empty in this window cannot be learnt (its node is padding)
        m &= np.where(a >= 0, keep[np.clip(a, 0, 2 * TEAM - 1)], False)
        m &= np.where(b < 2 * TEAM, keep[np.clip(b, 0, 2 * TEAM - 1)], True)
        src, dst = np.where(m, a, 0), np.where(m, b, 0)
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
    inv = np.empty(N_NODES, np.int64); inv[perm] = np.arange(N_NODES)
    x = x[:, perm]
    x[:, 2 * TEAM:, 2:4] = np.array(OUT_XY, np.float32)[None]               # outside nodes keep their fixed positions
    x[:, 2 * TEAM:, 0] = 0.0
    return x, inv[src], inv[dst], pmask[:, perm], node_slot[perm]


class WindowSet:
    """Torch-style dataset over several matches. skip_dead: drop windows with no scored live frame (training only)."""

    def __init__(self, root, mids, stride: int, train: bool, skip_dead: bool = True, ball: bool = True, matches: dict | None = None):
        self.m = matches if matches is not None else {m: Match(load_match(match_path(root, m))) for m in mids}
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
