"""Re-linking (barca_possession.relink) and the owner map of degrade_pff, on synthetic tracks only."""
import numpy as np

from barca_possession.relink import TEAM, link_stats, relink, window_stats
from barca_train.degrade_pff import degrade_players

FPS = 25.0


def players(n=1500, seed=0):
    """22 players, smooth random walks (home columns 0-10, away 11-21), well apart."""
    rng = np.random.default_rng(seed)
    base = np.stack([np.repeat(np.linspace(-40, 40, 11), 1), np.linspace(-25, 25, 11)], 1)
    x0 = np.r_[base[:, 0], base[:, 0] + 2.0]
    y0 = np.r_[base[:, 1], base[:, 1] - 2.0]
    v = rng.normal(0, 0.03, (n, 22)).cumsum(0) * 0.1
    px = (x0[None] + np.cumsum(v, 0) * 0.04 + np.linspace(0, 3, n)[:, None]).astype(np.float32)
    py = (y0[None] + np.sin(np.arange(n)[:, None] / 200 + np.arange(22)) * 1.5).astype(np.float32)
    return px, py, np.r_[np.ones(11), -np.ones(11)].astype(np.float32)


def fragment(px, py, ts, cuts):
    """Cut each player into pieces at the given frames (with a 5-frame NaN gap) and put every piece in a new slot."""
    n = px.shape[0]
    cols_x, cols_y, cols_t, owner = [], [], [], []
    for c in range(px.shape[1]):
        edges = [0] + [k for k in cuts[c]] + [n]
        for a, b in zip(edges[:-1], edges[1:]):
            a2 = a + (5 if a else 0)
            x = np.full(n, np.nan, np.float32); y = np.full(n, np.nan, np.float32); o = np.full(n, -1)
            x[a2:b], y[a2:b], o[a2:b] = px[a2:b, c], py[a2:b, c], c
            cols_x.append(x); cols_y.append(y); cols_t.append(ts[c]); owner.append(o)
    return np.stack(cols_x, 1), np.stack(cols_y, 1), np.array(cols_t, np.float32), np.stack(owner, 1)


def test_relink_recovers_players_from_fragments():
    px, py, ts = players()
    n = px.shape[0]
    rng = np.random.default_rng(1)
    cuts = [sorted(rng.choice(np.arange(100, n - 100), 3, replace=False)) for _ in range(22)]
    sx, sy, st, own = fragment(px, py, ts, cuts)
    period = np.ones(n, np.int8)
    r = relink(sx, sy, st, period, FPS)
    ls = r["lane_slot"]
    assert ls.shape == (n, 2 * TEAM)
    # each lane holds one true player for its whole life, and that player is on the lane's team
    rows = np.arange(n)[:, None]
    lo = np.where(ls >= 0, own[rows, np.maximum(ls, 0)], -1)
    for k in range(2 * TEAM):
        ids = np.unique(lo[:, k][lo[:, k] >= 0])
        assert len(ids) == 1, (k, ids)
        assert (ids[0] < 11) == (k < TEAM)
    # 22 lanes cover 22 different players
    assert len({int(np.unique(lo[:, k][lo[:, k] >= 0])[0]) for k in range(22)}) == 22
    st_ = link_stats(r["links"])
    assert st_["n_links"] == 22 * 3 and st_["forced_share"] == 0.0
    w = window_stats(ls, period, r["links"], FPS)
    assert w["windows"] == n // 250 and w["share_both_teams_ge10"] == 1.0


def test_relink_trims_a_twelfth_slot_and_never_exceeds_11_lanes():
    px, py, ts = players(n=600)
    extra_x = np.full((600, 1), np.nan, np.float32); extra_y = extra_x.copy()
    extra_x[200:300, 0], extra_y[200:300, 0] = 0.0, 0.0                 # a 12th home slot for 4 s
    sx, sy = np.concatenate([px, extra_x], 1), np.concatenate([py, extra_y], 1)
    st = np.r_[ts, 1.0].astype(np.float32)
    r = relink(sx, sy, st, np.ones(600, np.int8), FPS)
    assert (r["lane_slot"][:, :TEAM] == 22).sum() == 0                 # the newest-born slot is the one trimmed
    assert (r["lane_slot"][:, :TEAM] >= 0).sum(1).max() == TEAM
    assert r["trimmed_share"] > 0


def test_relink_is_deterministic_and_respects_periods():
    px, py, ts = players(n=800)
    period = np.r_[np.ones(400), np.zeros(10), 2 * np.ones(390)].astype(np.int8)
    px[400:410], py[400:410] = np.nan, np.nan
    a = relink(px, py, ts, period, FPS)
    b = relink(px, py, ts, period, FPS)
    assert np.array_equal(a["lane_slot"], b["lane_slot"])
    assert (a["lane_slot"][400:410] < 0).all()
    assert len(a["links"]) == 0                                         # a new period opens lanes, it does not link


def test_degrade_owner_map_follows_the_tracks():
    px, py, ts = players(n=3000)
    rng = np.random.default_rng(3)
    rates = dict(break_per_s=0.05, swap_per_player_s=0.02, drop_per_s=0.02, drop_len_s=1.0)
    nx, ny, nt, own = degrade_players(px, py, ts, np.ones(3000, np.int8), rng, rates, return_owner=True)
    ok = np.isfinite(nx)
    assert ((own >= 0) == ok).all()
    rows = np.nonzero(ok)
    assert np.allclose(nx[rows], px[rows[0], own[rows]])               # the slot holds the owner's true position
    assert (ts[own[rows]] == nt[rows[1]]).all()                        # and the owner is on the slot's team
    # same random draws with and without the owner map
    nx2, _, _ = degrade_players(px, py, ts, np.ones(3000, np.int8), np.random.default_rng(3), rates)
    assert np.array_equal(np.isfinite(nx), np.isfinite(nx2))
