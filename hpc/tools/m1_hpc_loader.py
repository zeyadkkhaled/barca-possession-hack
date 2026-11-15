"""Loader for the M1 HPC package (format: README.md next to this file). Needs numpy only; torch is optional.

    from m1_hpc_loader import load_match, fold_split, Windows
    m = load_match("matches/3812.npz")                        # dict of numpy arrays (float32 coordinates)
    train, val = fold_split(".", val_fold=0)                  # lists of match ids
    ds = Windows(".", train, length=300, stride=150)          # fixed-length windows (torch Dataset if torch is installed)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

STATES = ("DEAD", "CONTROL", "TRAVELLING", "LOOSE", "UNKNOWN")
BALL_FLAGS = ("MISSING", "OBSERVED", "INTERPOLATED", "ESTIMATED")   # ball_flag codes 0-3


def load_match(path, float32: bool = True) -> dict:
    """All arrays of one match file; float16 coordinates converted to float32 (NaN = absent / missing)."""
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}
    if float32:
        for k, v in d.items():
            if v.dtype == np.float16:
                d[k] = v.astype(np.float32)
    return d


def fold_split(root, val_fold: int):
    folds = json.loads((Path(root) / "folds.json").read_text())["folds"]
    val = folds[val_fold]
    return [m for k, f in enumerate(folds) if k != val_fold for m in f], list(val)


def frame_features(d: dict) -> np.ndarray:
    """A simple per-frame tensor (frames, slots, 6): for every player column x, y, side (+1 home / -1 away), visible,
    present, and the ball as an extra last slot (x, y, z, 0, observed, present). Absent values are 0 with present = 0."""
    n, p = d["player_x"].shape
    out = np.zeros((n, p + 1, 6), np.float32)
    pres = np.isfinite(d["player_x"])
    out[:, :p, 0] = np.nan_to_num(d["player_x"]); out[:, :p, 1] = np.nan_to_num(d["player_y"])
    out[:, :p, 2] = np.where(d["player_side"] == 0, 1.0, -1.0)[None, :] * pres
    out[:, :p, 3] = d["player_vis"] == 1
    out[:, :p, 4] = pres
    bpres = np.isfinite(d["ball_x"])
    out[:, p, 0] = np.nan_to_num(d["ball_x"]); out[:, p, 1] = np.nan_to_num(d["ball_y"]); out[:, p, 2] = np.nan_to_num(d["ball_z"])
    out[:, p, 4] = d["ball_flag"] == 1
    out[:, p, 5] = bpres
    return out


try:
    from torch.utils.data import Dataset as _Base
except Exception:          # torch not installed: plain Python sequence
    _Base = object


class Windows(_Base):
    """Fixed-length windows over the frames of the given matches (windows never cross a period boundary).
    Item: dict(x=(length, slots, 6) float32, state=(length,) int8, slot=(length,) int8, mask=(length,) bool, match, start)."""

    def __init__(self, root, matches, length: int = 300, stride: int = 150):
        self.data, self.index, self.length = {}, [], length
        for m in matches:
            d = load_match(Path(root) / "matches" / f"{m}.npz")
            self.data[m] = dict(x=frame_features(d), state=d["label_state"], slot=d["label_player_slot"],
                                mask=d["label_score_mask"] & (d["label_state"] <= 3) & (d["period"] > 0))
            per = d["period"]
            starts = np.r_[0, np.flatnonzero(per[1:] != per[:-1]) + 1, len(per)]
            for a, b in zip(starts[:-1], starts[1:]):
                if per[a] <= 0:
                    continue
                for s in range(a, max(a + 1, b - length + 1), stride):
                    if s + length <= b:
                        self.index.append((m, s))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        m, s = self.index[i]
        d, w = self.data[m], slice(s, s + self.length)
        return dict(x=d["x"][w], state=d["state"][w], slot=d["slot"][w], mask=d["mask"][w], match=m, start=s)
