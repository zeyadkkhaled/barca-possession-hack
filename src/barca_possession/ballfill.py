"""Ball filling for Metrica-form tracking (the hackathon match or degraded PFF), in this order:

a. the M1 smoother (repeat / spike removal, adaptive RTS Kalman) on the raw ball at 25 fps, joining gaps up to 1 s;
b. longer gaps: the M1 ball-from-players regressors (ball_infer features: team-level, no player ids), trained on degraded
   PFF where the true ball is known;
c. per frame: ball_missing (raw ball absent), gap_length_so_far (s since the raw ball was last seen), ball_source
   (0 seen, 1 smoothed = joined short gap, 2 estimated by ball-from-players, 3 none).
"""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np

from . import ball_infer as BI
from .config import SmootherConfig
from .smoother import FLAG_INTERPOLATED, smooth_ball

SRC_SEEN, SRC_SMOOTHED, SRC_ESTIMATED, SRC_NONE = 0, 1, 2, 3
SOURCE_NAMES = ("seen", "smoothed", "estimated", "none")
# a. Metrica: no evidence that the ball is held for two frames as in PFF -> every sample is an update; gaps up to 1 s
SMOOTHER_25 = SmootherConfig(repeat_keep_every=1, max_gap_s=1.0, method="rts_adaptive")
CHUNK = 8000


def lgb_params(seed: int = 0, threads: int = 1, **kw) -> dict:
    p = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=50, colsample_bytree=0.8, subsample=0.8,
             subsample_freq=1, verbose=-1, deterministic=True, force_row_wise=True, seed=seed, n_jobs=threads)
    p.update(kw)
    return p


def smooth_stage(t, period, bx, by, cfg: SmootherConfig = SMOOTHER_25):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = smooth_ball(t, bx, by, None, period=period, cfg=cfg)
    raw = np.isfinite(bx) & np.isfinite(by)
    src = np.full(len(t), SRC_NONE, np.int8)
    ok = np.isfinite(r.x)
    src[ok] = SRC_SMOOTHED
    src[ok & raw & (r.flag != FLAG_INTERPOLATED)] = SRC_SEEN
    return r, src


def _bfp_inputs(m: dict, r, frames):
    t, per = m["t"], m["period"]
    present = np.isfinite(r.x)
    last, _ = BI.last_seen(t, per, present)
    last = last[frames]
    s = BI.orient(per, m["px"], m["team_sign"])
    lf = np.maximum(last, 0)
    ok = last >= 0
    lx, ly = np.where(ok, r.x[lf], np.nan), np.where(ok, r.y[lf], np.nan)
    lvx, lvy = np.where(ok, r.vx[lf], np.nan), np.where(ok, r.vy[lf], np.nan)
    out = []
    for c in range(0, len(frames), CHUNK):
        sl = slice(c, c + CHUNK)
        out.append(BI.features(t, per, m["px"], m["py"], m["team_sign"], s, frames[sl], last[sl], lx[sl], ly[sl], lvx[sl], lvy[sl]))
    X = np.concatenate(out) if out else np.zeros((0, 1), np.float32)
    return X, lx, ly, s[frames]


def bfp_samples(m: dict, rng, n_max: int = 6000):
    """Training samples on degraded PFF: frames without a smoothed ball where the true ball is known."""
    r, src = smooth_stage(m["t"], m["period"], m["bx"], m["by"])
    cand = np.flatnonzero((src == SRC_NONE) & (m["period"] > 0) & np.isfinite(m["ball_true_x"]))
    f = np.sort(rng.choice(cand, size=min(n_max, len(cand)), replace=False)) if len(cand) else cand
    X, lx, ly, s = _bfp_inputs(m, r, f)
    lxo, lyo = np.where(np.isfinite(lx), lx, 0.0), np.where(np.isfinite(ly), ly, 0.0)
    dx = ((m["ball_true_x"][f] - lxo) * s).astype(np.float32)
    dy = (m["ball_true_y"][f] - lyo).astype(np.float32)
    return X, dx, dy


def fit_bfp(samples: list, seed: int = 0, threads: int = 1):
    X = np.concatenate([s[0] for s in samples])
    mods = []
    for k in (1, 2):
        y = np.concatenate([s[k] for s in samples])
        mods.append(lgb.LGBMRegressor(**lgb_params(seed, threads, n_estimators=500)).fit(X, y).booster_)
    return mods


def fill_ball(m: dict, bfp_models=None) -> dict:
    """Ball per frame: x, y (filled), vx, vy (NaN when estimated), source, ball_missing, gap_length_so_far."""
    t, per = m["t"], m["period"]
    r, src = smooth_stage(t, per, m["bx"], m["by"])
    x, y = r.x.astype(np.float32).copy(), r.y.astype(np.float32).copy()
    vx, vy = r.vx.astype(np.float32), r.vy.astype(np.float32)
    if bfp_models is not None:
        f = np.flatnonzero((src == SRC_NONE) & (per > 0))
        if len(f):
            X, lx, ly, s = _bfp_inputs(m, r, f)
            dx, dy = (mdl.predict(X) for mdl in bfp_models)
            lxo, lyo = np.where(np.isfinite(lx), lx, 0.0), np.where(np.isfinite(ly), ly, 0.0)
            x[f] = lxo + dx * s
            y[f] = lyo + dy
            src[f] = SRC_ESTIMATED
    raw = np.isfinite(m["bx"])
    last, h = BI.last_seen(t, per, raw)
    gap = np.where(raw, 0.0, np.where(np.isfinite(h), h, 60.0)).astype(np.float32)
    return dict(x=x, y=y, vx=vx, vy=vy, source=src, ball_missing=~raw, gap_s=np.minimum(gap, 60.0))
