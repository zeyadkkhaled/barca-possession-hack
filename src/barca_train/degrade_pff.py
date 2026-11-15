"""Turn the M1 PFF package (30 learning matches, 29.97 fps, clean tracking, player ids) into the Metrica form, so models
trained on it see the same kind of input as the hackathon match.

Per match (seeded: seed + match id):
1. resample to 25 fps inside each period (linear for positions; labels from the nearest original frame);
2. coordinates: PFF centre-origin metres, scaled to 105.3 x 68 m, turned 180 degrees in the periods where home attacks
   -x (home_attacks_right), so home always attacks +x, as load.py does;
3. player ids dropped: one team sign per slot; every player's track is cut at random breaks (a short NaN gap, then a new
   slot) and pairs of same-team players swap identities at random (the slot continues with the other player), at the
   rates load.py measured (slot starts per minute); random player drop-out runs up to the measured share of absent
   players (default ~10 %);
4. ball: only PFF's observed samples are kept, then extra missing runs are cut until the share of missing graded frames
   matches the measurement (~41 %), with run lengths drawn from the measured gap-length histogram of the same ball-speed
   class (slow < 5 m/s / fast) and gap starts drawn so that the slow / fast share of gaps matches too.

Labels (labels v2 -> H/A/D: CONTROL / TRAVELLING -> that team, LOOSE -> last team, DEAD -> D; barca_labels.scoring.had) and
the true ball (observed samples) are kept for training and scoring only.

    python -m barca_train.degrade_pff [--stats outputs/load_stats.json] [--out outputs/pff_degraded] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from barca_labels.scoring import had

from barca_possession.load import (BEFORE_RESTART_S, GAP_EDGES_S, SLOW_MPS, gap_stats, missing_before, print_stats,
                                   slot_stats)
from barca_possession.paths import AUTO, FPS, OUT, PFF_PACKAGE, PITCH

DT = 1.0 / FPS
BREAK_GAP_S = 0.4          # mean NaN gap at a track break (assumption; not measurable without identities)
DROP_RUN_S = 1.0           # mean length of a player drop-out run (assumption)
SWAP_SHARE = 0.25          # identity swaps per track break (assumption; swaps cannot be counted without identities)
MIN_REUSE = 10             # frames a slot stays empty before it is reused by another track
LAST_BIN_MAX_S = 60.0


def load_package_match(path: Path) -> dict:
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}
    return load_package_match_dict(d)


def load_package_match_dict(d: dict) -> dict:
    return {k: (v.astype(np.float32) if getattr(v, "dtype", None) == np.float16 else v) for k, v in d.items()}


def _resample(d: dict):
    """29.97 fps -> 25 fps per period. Returns the index arrays (i0, i1, w, nearest) and the new t / period."""
    t, per = d["t"].astype(np.float64), d["period"]
    T, P, I0, I1, W = [], [], [], [], []
    for p in [p for p in np.unique(per) if p > 0]:
        idx = np.flatnonzero(per == p)
        tp = t[idx]
        tn = np.arange(tp[0], tp[-1] + 1e-9, DT)
        j = np.clip(np.searchsorted(tp, tn, "right"), 1, len(tp) - 1)
        i0, i1 = idx[j - 1], idx[j]
        w = np.clip((tn - t[i0]) / np.maximum(t[i1] - t[i0], 1e-9), 0, 1)
        T.append(tn); P.append(np.full(len(tn), p, np.int8)); I0.append(i0); I1.append(i1); W.append(w)
    i0, i1, w = np.concatenate(I0), np.concatenate(I1), np.concatenate(W)
    near = np.where(w < 0.5, i0, i1)
    return np.concatenate(T), np.concatenate(P), i0, i1, w, near


def _lin(v, i0, i1, w):
    w = w.reshape((-1,) + (1,) * (v.ndim - 1))
    return ((1 - w) * v[i0] + w * v[i1]).astype(np.float32)    # NaN if either side is NaN


def to_25fps(d: dict) -> dict:
    """Resampled, rescaled, oriented PFF match with H/A/D labels (no degradation yet)."""
    t, period, i0, i1, w, near = _resample(d)
    L0, W0 = (float(x) for x in d["pitch"])
    sx, sy = PITCH[0] / L0, PITCH[1] / W0
    px, py = _lin(d["player_x"], i0, i1, w) * sx, _lin(d["player_y"], i0, i1, w) * sy
    # PFF's raw ball is held for two frames, so the package flags alternate observed / interpolated; every frame with a
    # ball (flags 1-3: observed, interpolated <= 0.5 s, estimated) is taken as present. PFF already misses ~27 %.
    obs = d["ball_flag"] != 0
    rbx, rby = np.where(obs, d["ball_x"], np.nan), np.where(obs, d["ball_y"], np.nan)
    bx, by = _lin(rbx, i0, i1, w) * sx, _lin(rby, i0, i1, w) * sy
    bvx, bvy = _lin(d["ball_vx"], i0, i1, w) * sx, _lin(d["ball_vy"], i0, i1, w) * sy
    s = np.ones(len(t), np.float32)
    har = np.asarray(d["home_attacks_right"]).astype(bool)
    for p in np.unique(period):
        if not har[min(int(p), len(har)) - 1]:
            s[period == p] = -1.0
    px, py = px * s[:, None], py * s[:, None]
    bx, by, bvx, bvy = bx * s, by * s, bvx * s, bvy * s
    v = had(d["label_state"], d["label_team_id"], d["period"], int(d["home_team_id"]), loose="last")
    mask0 = d["label_score_mask"] & (d["label_state"] <= 3) & (d["period"] > 0) & (v >= 0)
    lab = np.where(v[near] < 0, 2, v[near]).astype(np.int8)
    return dict(match_id=int(d["match_id"]), fold=int(d["fold"]), t=t, period=period, px=px, py=py,
                team_sign=np.where(d["player_side"] == 0, 1.0, -1.0).astype(np.float32),
                ball_true_x=bx, ball_true_y=by, ball_speed_true=np.hypot(bvx, bvy).astype(np.float32),
                label_had=lab, label_mask=mask0[near], pitch=np.array(PITCH, np.float32))


# ---------------------------------------------------------------- players

def _runs(mask):
    e = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))


def degrade_players(px, py, team_sign, period, rng, rates: dict):
    """Break / swap / drop tracks; returns new (px, py, team_sign) with anonymous slot columns."""
    n, P = px.shape
    gap_f = max(1, int(round(BREAK_GAP_S * FPS)))
    pieces = []                                    # [player, a, b)
    pres = np.isfinite(px)
    for c in range(P):
        for a, b in _runs(pres[:, c]):
            # never across a period boundary
            cuts = [a] + [k for k in range(a + 1, b) if period[k] != period[k - 1]] + [b]
            for a1, b1 in zip(cuts[:-1], cuts[1:]):
                dur = (b1 - a1) / FPS
                k = rng.poisson(rates["break_per_s"] * dur)
                bt = np.sort(rng.integers(a1 + 1, max(a1 + 2, b1), size=k)) if k else np.zeros(0, int)
                s0 = a1
                for x in bt:
                    if x - s0 > 2:
                        pieces.append([c, s0, x])
                    s0 = min(b1, x + 1 + int(rng.exponential(gap_f)))
                if b1 - s0 > 2:
                    pieces.append([c, s0, b1])
    # swaps between two same-team players present around the swap frame
    nxt = {}                                       # piece index -> piece index that continues the same slot
    for side in (1.0, -1.0):
        tot = sum(p[2] - p[1] for p in pieces if team_sign[p[0]] == side) / FPS
        k = rng.poisson(rates["swap_per_player_s"] * tot / 2)
        for _ in range(k):
            tau = int(rng.integers(0, n))
            cand = [i for i, p in enumerate(pieces) if team_sign[p[0]] == side and p[1] + 5 < tau < p[2] - 5
                    and i not in nxt and not any(v == i for v in nxt.values())]
            if len(cand) < 2:
                continue
            ia, ib = rng.choice(cand, 2, replace=False)
            pa, pb = pieces[ia], pieces[ib]
            na, nb = len(pieces), len(pieces) + 1
            pieces.append([pa[0], tau, pa[2]]); pieces.append([pb[0], tau, pb[2]])
            pa[2], pb[2] = tau, tau
            nxt[ia], nxt[ib] = nb, na              # A's slot continues with B's future, and B's with A's
    # chains of linked pieces = tracks (each a contiguous interval)
    heads = set(range(len(pieces))) - set(nxt.values())
    tracks = []
    for h in heads:
        chain, i = [h], h
        while i in nxt:
            i = nxt[i]
            chain.append(i)
        tracks.append(chain)
    tracks.sort(key=lambda ch: pieces[ch[0]][1])
    cols_end, cols_team, assign = [], [], []
    for ch in tracks:
        side, a = team_sign[pieces[ch[0]][0]], pieces[ch[0]][1]
        j = next((k for k in range(len(cols_end)) if cols_team[k] == side and cols_end[k] + MIN_REUSE <= a), None)
        if j is None:
            cols_end.append(0); cols_team.append(side); j = len(cols_end) - 1
        cols_end[j] = pieces[ch[-1]][2]
        assign.append((j, ch))
    nx, ny = np.full((n, len(cols_end)), np.nan, np.float32), np.full((n, len(cols_end)), np.nan, np.float32)
    for j, ch in assign:
        for i in ch:
            c, a, b = pieces[i]
            nx[a:b, j], ny[a:b, j] = px[a:b, c], py[a:b, c]
    # drop-out runs
    drop_f = max(1, int(round(rates.get("drop_len_s", DROP_RUN_S) * FPS)))
    for j in range(nx.shape[1]):
        on = np.isfinite(nx[:, j]).sum()
        k = rng.poisson(rates["drop_per_s"] * on / FPS)
        for a in rng.integers(0, n, size=k):
            b = min(n, a + 1 + int(rng.exponential(drop_f)))
            nx[a:b, j] = ny[a:b, j] = np.nan
    return nx, ny, np.array(cols_team, np.float32)


# ---------------------------------------------------------------- ball

def _bin_sampler(counts, rng):
    c = np.asarray(counts, float)
    p = c / c.sum() if c.sum() > 0 else np.full(len(c), 1 / len(c))
    lo = np.array(GAP_EDGES_S[:-1])
    hi = np.array([*GAP_EDGES_S[1:-1], LAST_BIN_MAX_S])

    def draw():
        k = rng.choice(len(p), p=p)
        return rng.uniform(lo[k], hi[k])
    return draw


def degrade_ball(bx, by, speed_true, period, rng, gap: dict, lab=None, dead_weight: float = 1.0):
    """Cut ball-missing runs until the share of missing in-period frames reaches the measured one. Gap starts on
    answer-key D frames are dead_weight times more likely (the match's ball is missing far more often around restarts;
    the weight is calibrated in calibrate_dead_weight)."""
    bx, by = bx.copy(), by.copy()
    target = gap["missing_frac_graded"]
    nslow, nfast = sum(gap["slow"]["counts"]), sum(gap["fast"]["counts"])
    f_fast = nfast / max(nslow + nfast, 1)
    draw = {False: _bin_sampler(gap["slow"]["counts"], rng), True: _bin_sampler(gap["fast"]["counts"], rng)}
    inper = period > 0
    n_in = inper.sum()
    fast_frame = np.nan_to_num(speed_true) >= SLOW_MPS
    b_fast = max(min(fast_frame[inper & np.isfinite(bx)].mean(), 0.99), 0.01)
    acc = {True: f_fast / b_fast, False: (1 - f_fast) / (1 - b_fast)}
    mx = max(acc.values()) * max(dead_weight, 1.0)
    dead = np.zeros(len(bx), bool) if lab is None else (lab == 2)
    miss = ~np.isfinite(bx)
    cur = miss[inper].sum()
    n = len(bx)
    tries = 0
    while cur / n_in < target and tries < 2000000:
        tries += 1
        a = int(rng.integers(1, n))
        if miss[a] or not inper[a] or miss[a - 1]:
            continue
        fs = bool(fast_frame[a])
        if rng.random() > acc[fs] * (dead_weight if dead[a] else 1.0) / mx:
            continue
        L = max(1, int(round(draw[fs]() * FPS)))
        b = a + L
        while b > a and (b > n or period[b - 1] != period[a]):
            b -= 1
        add = (~miss[a:b]).sum()
        miss[a:b] = True
        cur += add
    bx[miss], by[miss] = np.nan, np.nan
    return bx, by


# ---------------------------------------------------------------- extra slots (step 1c-v2)

REF_LAG_S = 3.0                 # the referee-like path follows the ball with this time constant
REF_OFFSET_M = (5.0, 12.0)


def extra_config(diag: dict, target: float | None = None) -> dict:
    """From reports/auto/extra_slot_diag.json: the frame share of over-11 frames to reach, the episode-length
    histograms of (b) wrong team label and (a) extra person, and their mix (the 'other' frames shared in proportion)."""
    ep = diag["episodes"]
    fb, fa = ep["b_wrong_label"]["frame_frac"], ep["a_extra_person"]["frame_frac"]
    return dict(target_frac=float(target if target is not None else ep["any_over11"]["frame_frac"]),
                share_b=fb / max(fa + fb, 1e-9), b_counts=ep["b_wrong_label"]["counts"], a_counts=ep["a_extra_person"]["counts"])


def _team_counts(px, team_sign):
    pres = np.isfinite(px)
    return pres[:, team_sign > 0].sum(1), pres[:, team_sign < 0].sum(1)


def _referee_path(bx, by, period, rng, fps: float = FPS):
    """A player-like path that follows play: the true ball (held where missing) smoothed with a REF_LAG_S time constant,
    plus a fixed random offset of 5-12 m."""
    x = bx.copy(); y = by.copy()
    for v in (x, y):
        ok = np.isfinite(v)
        if ok.any():
            i = np.maximum.accumulate(np.where(ok, np.arange(len(v)), 0))
            v[:] = v[i]
            v[~ok & (np.arange(len(v)) < np.argmax(ok))] = v[np.argmax(ok)]
    a = 1.0 / (REF_LAG_S * fps)
    sx, sy = x.copy(), y.copy()
    for k in range(1, len(x)):
        if period[k] != period[k - 1]:
            continue
        sx[k] = sx[k - 1] + a * (x[k] - sx[k - 1])
        sy[k] = sy[k - 1] + a * (y[k] - sy[k - 1])
    ang, r = rng.uniform(0, 2 * np.pi), rng.uniform(*REF_OFFSET_M)
    return (sx + r * np.cos(ang)).astype(np.float32), (sy + r * np.sin(ang)).astype(np.float32)


def add_extra_slots(m: dict, rng, cfg: dict) -> dict:
    """Make some team have more than 11 slots on cfg['target_frac'] of the in-period frames, as the match does:
    (b) wrong team label: a real player's track is moved for an episode into a new slot of the OTHER team (that team
        gets 12, his own team 10); (a) extra person: a referee-like path is added to a team as a new slot.
    Episode lengths are drawn from the match's histograms; episodes start only where the receiving team already has
    at least 11 slots (so each one creates over-11 frames). Seeded by the caller's rng."""
    m = dict(m)
    px, py, ts, per = m["px"].copy(), m["py"].copy(), m["team_sign"].copy(), m["period"]
    n = len(per)
    inper = per > 0
    n_in = max(int(inper.sum()), 1)
    draw = {"a": _bin_sampler(cfg["a_counts"], rng), "b": _bin_sampler(cfg["b_counts"], rng)}
    ref = None
    new_x, new_y, new_t = [], [], []
    nh, na = _team_counts(px, ts)
    over = inper & ((nh > 11) | (na > 11))
    cur = int(over.sum())
    tries = 0
    while cur / n_in < cfg["target_frac"] and tries < 200000:
        tries += 1
        a = int(rng.integers(0, n))
        if not inper[a]:
            continue
        side = 1.0 if rng.random() < 0.5 else -1.0
        mine = nh if side > 0 else na
        if mine[a] < 11:
            continue
        kind = "b" if rng.random() < cfg["share_b"] else "a"
        L = max(1, int(round(draw[kind]() * FPS)))
        b = a + L
        while b > a and (b > n or per[b - 1] != per[a]):
            b -= 1
        if b <= a:
            continue
        gx = np.full(n, np.nan, np.float32); gy = np.full(n, np.nan, np.float32)
        if kind == "b":
            cand = np.flatnonzero((ts == -side) & np.isfinite(px[a]))
            if not len(cand):
                continue
            j = int(rng.choice(cand))
            seg = np.isfinite(px[a:b, j])
            b = a + (int(np.argmin(seg)) if not seg.all() else len(seg))
            if b <= a:
                continue
            gx[a:b], gy[a:b] = px[a:b, j], py[a:b, j]
            px[a:b, j] = py[a:b, j] = np.nan
            (na if side > 0 else nh)[a:b] -= 1
        else:
            if ref is None:
                ref = _referee_path(m["ball_true_x"], m["ball_true_y"], per, rng)
            gx[a:b], gy[a:b] = ref[0][a:b], ref[1][a:b]
        new_x.append(gx); new_y.append(gy); new_t.append(side)
        mine[a:b] += 1
        o2 = inper[a:b] & ((nh[a:b] > 11) | (na[a:b] > 11))
        cur += int(o2.sum() - over[a:b].sum())
        over[a:b] = o2
    if new_x:
        # pack the new tracks into as few columns as possible (per team, MIN_REUSE frames apart)
        cols_x, cols_y, cols_t, ends = [], [], [], []
        for gx, gy, side in sorted(zip(new_x, new_y, new_t), key=lambda z: int(np.argmax(np.isfinite(z[0])))):
            fin = np.flatnonzero(np.isfinite(gx))
            s0, e0 = int(fin[0]), int(fin[-1]) + 1
            k = next((i for i in range(len(cols_x)) if cols_t[i] == side and ends[i] + MIN_REUSE <= s0), None)
            if k is None:
                cols_x.append(np.full(n, np.nan, np.float32)); cols_y.append(np.full(n, np.nan, np.float32))
                cols_t.append(side); ends.append(0); k = len(cols_x) - 1
            cols_x[k][s0:e0], cols_y[k][s0:e0] = gx[s0:e0], gy[s0:e0]
            ends[k] = e0
        px = np.concatenate([px, np.stack(cols_x, 1)], 1)
        py = np.concatenate([py, np.stack(cols_y, 1)], 1)
        ts = np.concatenate([ts, np.array(cols_t, np.float32)])
    m["px"], m["py"], m["team_sign"] = px, py, ts
    m["extra_frac"] = np.float32(cur / n_in)
    return m


def rates_from(stats: dict) -> dict:
    s = stats["slots"]
    present_frac = min(max(s["mean_slots_total"] / 22.0, 0.5), 1.0)
    p_drop = 1.0 - present_frac if s["mean_slots_total"] < 22 else 0.10
    # every drop-out run and every break starts a new slot segment: split the measured start rate half / half, and make
    # the drop-out runs as long as they must be to reach the measured share of absent players
    starts_per_player_s = max(s["slot_starts_per_min"] / 60.0 / 22.0, 0.004)
    drop = starts_per_player_s / 2
    brk = starts_per_player_s / 2
    drop_len = float(np.clip(p_drop / drop, 0.2, 60.0))
    return dict(break_per_s=brk, swap_per_player_s=SWAP_SHARE * brk, drop_per_s=drop, drop_len_s=drop_len, p_drop=p_drop)


def restarts(lab, period, fps: float = FPS):
    """Answer-key restarts: D -> H/A after at least 2 s of D, inside a period."""
    out = []
    for a, b in _runs((lab == 2) & (period > 0)):
        if b < len(lab) and b - a >= 2 * fps and period[b] == period[a]:
            out.append(b)
    return out


def clean_match(d: dict) -> dict:
    """No degradation (step-1b check): PFF at 25 fps in the same form, player columns keep their identities, the model
    ball is the true ball."""
    m = to_25fps(d)
    m["bx"], m["by"] = m["ball_true_x"].copy(), m["ball_true_y"].copy()
    m["dead_weight"] = np.float32(np.nan)
    return m


def degrade_match(d: dict, stats: dict, seed: int, dead_weight: float = 1.0, m25: dict | None = None,
                  extra: dict | None = None) -> dict:
    m = dict(m25) if m25 is not None else to_25fps(d)
    rng = np.random.default_rng(seed + m["match_id"])
    r = rates_from(stats)
    m["px"], m["py"], m["team_sign"] = degrade_players(m["px"], m["py"], m["team_sign"], m["period"], rng, r)
    m["bx"], m["by"] = degrade_ball(m["ball_true_x"], m["ball_true_y"], m["ball_speed_true"], m["period"], rng, stats["gap"],
                                    lab=m["label_had"], dead_weight=dead_weight)
    m["dead_weight"] = np.float32(dead_weight)
    if extra:
        m = add_extra_slots(m, rng, extra)
    return m


DEAD_WEIGHTS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)


def calibrate_dead_weight(files, stats: dict, seed: int, n_matches: int = 4, log=print) -> float:
    """The weight whose ball-missing share before the answer key's restarts (on a few PFF matches) is closest to the
    match's share before SET PIECES tags (load.py). Ball only, so players are not degraded here."""
    target = stats.get("missing_before_set_piece")
    if target is None or not np.isfinite(target):
        log("no missing_before_set_piece in the stats -> dead_weight 1")
        return 1.0
    ms = [to_25fps(load_package_match(f)) for f in files[:n_matches]]
    best = None
    for w in DEAD_WEIGHTS:
        v = []
        for m in ms:
            rng = np.random.default_rng(seed + m["match_id"])
            bx, _ = degrade_ball(m["ball_true_x"], m["ball_true_y"], m["ball_speed_true"], m["period"], rng, stats["gap"],
                                 lab=m["label_had"], dead_weight=w)
            v.append(missing_before(~np.isfinite(bx), m["period"], restarts(m["label_had"], m["period"]), BEFORE_RESTART_S))
        val = float(np.mean(v))
        log(f"  dead_weight {w:g}: missing before restarts {100 * val:.1f} % (target {100 * target:.1f} %)")
        if best is None or abs(val - target) < best[0]:
            best = (abs(val - target), w, val)
    return best[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stats", default=str(OUT / "load_stats.json"))
    ap.add_argument("--package", default=str(PFF_PACKAGE))
    ap.add_argument("--out", default=str(OUT / "pff_degraded"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--matches", type=int, nargs="*", help="subset of match ids (default: all 30)")
    ap.add_argument("--auto-name", default="degrade_stats.json", help="aggregate summary file in reports/auto ('' = none)")
    ap.add_argument("--dead-weight", type=float, help="fixed dead-ball gap weight (default: calibrated); 1 = no weighting")
    ap.add_argument("--clean", action="store_true", help="no degradation at all (25 fps, same form, true ball)")
    ap.add_argument("--extra", action="store_true", help="also simulate the match's extra slots (reports/auto/extra_slot_diag.json)")
    a = ap.parse_args()
    t0 = time.time()
    stats = json.loads(Path(a.stats).read_text())
    pkg, out = Path(a.package), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted((pkg / "matches").glob("*.npz"))
    if a.matches:
        files = [f for f in files if int(f.stem) in set(a.matches)]
    print(f"rates: {rates_from(stats)}")
    dw = (float("nan") if a.clean else a.dead_weight if a.dead_weight is not None
          else calibrate_dead_weight(files, stats, a.seed))
    print(f"dead_weight: {dw:g}" + (" (clean: no degradation)" if a.clean else ""))
    xcfg = extra_config(json.loads((AUTO / "extra_slot_diag.json").read_text())) if a.extra else None
    if xcfg:
        print(f"extra slots: target {100 * xcfg['target_frac']:.1f} % of frames, wrong-label share {xcfg['share_b']:.2f}")
    agg_gap, agg_slot, before = [], [], []
    for f in files:
        d = load_package_match(f)
        m = clean_match(d) if a.clean else degrade_match(d, stats, a.seed, dead_weight=dw, extra=xcfg)
        before.append(missing_before(~np.isfinite(m["bx"]), m["period"], restarts(m["label_had"], m["period"]),
                                     BEFORE_RESTART_S))
        np.savez_compressed(out / f"{m['match_id']}.npz", **m)
        agg_gap.append(gap_stats(m["t"], m["period"], m["bx"], m["by"], PITCH[1] / 2))
        agg_slot.append(slot_stats(m["period"], m["px"], m["team_sign"]))
        print(f"  {m['match_id']}: {len(m['t'])} frames, {m['px'].shape[1]} slots ({time.time() - t0:.0f} s)", flush=True)
    # aggregate over matches (sum of counts, mean of rates) vs the target
    g = dict(agg_gap[0])
    for key in [k for k in g if isinstance(g[k], dict) and "counts" in g[k]]:
        g[key] = dict(counts=np.sum([x[key]["counts"] for x in agg_gap], 0).tolist(),
                      seconds=np.sum([x[key]["seconds"] for x in agg_gap], 0).round(1).tolist())
    g["n_gaps"] = int(sum(x["n_gaps"] for x in agg_gap))
    g["missing_frac_graded"] = float(np.mean([x["missing_frac_graded"] for x in agg_gap]))
    g["missing_frac_all"] = float(np.mean([x["missing_frac_all"] for x in agg_gap]))
    s = {k: float(np.mean([x[k] for x in agg_slot])) for k in agg_slot[0]}
    deg = dict(gap=g, slots=s, missing_before_restart=float(np.nanmean(before)), dead_weight=dw, extra=xcfg)
    print("\n== degraded PFF (all matches)")
    print(f"ball missing before restarts: {100 * deg['missing_before_restart']:.1f} % "
          f"(match, before SET PIECES tags: {100 * stats.get('missing_before_set_piece', float('nan')):.1f} %)")
    print_stats(deg)
    print("\n== target (hackathon match)")
    print_stats(stats)
    if a.auto_name:
        AUTO.mkdir(parents=True, exist_ok=True)
        (AUTO / a.auto_name).write_text(json.dumps(dict(degraded=deg, rates=rates_from(stats), n_matches=len(files),
                                                        seed=a.seed), indent=1))
    print(f"done in {time.time() - t0:.0f} s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
