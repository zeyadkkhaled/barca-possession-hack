"""Synthetic tests for the M1 Part-3 additions (2026-11-09): C3 post-processing and the extra feature blocks (labelled
synthetic data only)."""
import numpy as np

from barca_possession import c3 as C3
from barca_possession.c3_extra import extra_features, who_first
from barca_possession.c3_post import merge_tiny, passes_from, timing_fix
from barca_possession.smoother import FLAG_OBSERVED

FPS = 30.0


def test_timing_fix_moves_a_switch_to_the_nearest_touch():
    n = 100
    state = np.r_[np.full(50, 2), np.full(50, 1)]           # TRAVELLING, then CONTROL from frame 50
    team = np.full(n, 7); player = np.r_[np.full(50, -1), np.full(50, 9)]
    s, p, t = timing_fix(state, player, team, np.ones(n, int), [44], FPS, tol_s=1.0)
    assert (s[44:] == 1).all() and (s[:44] == 2).all() and (p[44:50] == 9).all()
    s2, *_ = timing_fix(state, player, team, np.ones(n, int), [5], FPS, tol_s=1.0)   # too far: unchanged
    assert (s2 == state).all()


def test_merge_tiny_spell_becomes_the_neighbours_possession():
    n = 300
    state = np.full(n, 1); team = np.full(n, 1); player = np.full(n, 5)
    team[150:160] = 2; player[150:160] = 8                   # 10 frames (0.33 s) of the other team
    s, p, t = merge_tiny(state, player, team, np.ones(n, int), 1, FPS, M=1.0)
    assert (t == 1).all() and (s[150:160] == 2).all() and (p[150:160] == -1).all()
    s3, p3, t3 = merge_tiny(state, player, team, np.ones(n, int), 1, FPS, M=0.2)  # shorter limit: kept
    assert (t3[150:160] == 2).all()


def test_passes_from_control_to_travelling():
    state = np.array([1, 1, 2, 2, 1, 1, 3])
    player = np.array([4, 4, -1, -1, 6, 6, -1])
    assert passes_from(state, player) == [(1, 4)]


def _scene(n=200):
    t = np.arange(n) / FPS
    per = np.ones(n, int)
    bx = np.linspace(0, 10, n); by = np.zeros(n); bz = np.zeros(n)
    bvx = np.full(n, 10 / (n / FPS)); bvy = np.zeros(n)
    px = np.stack([bx + 0.5, bx + 6.0, bx - 3.0, bx + 20.0], 1).astype(np.float32)   # home near, home far, away x2
    py = np.zeros((n, 4), np.float32)
    ts = np.array([1, 1, -1, -1], np.float32)
    return C3.Prepared(t, per, bx, by, bz, bvx, bvy, np.full(n, FLAG_OBSERVED), px, py, ts, np.array([105.0, 68.0]),
                       np.ones(n, bool))


def test_who_first_favours_the_team_of_the_nearest_player():
    P = _scene()
    rows = np.arange(10, 190, 3)
    cols, names = who_first(P, rows)
    gap = cols[names.index("race_gap@0s")]
    assert np.all(gap > 0)                                    # the nearest player's team arrives first
    assert np.all(cols[names.index("race_away_minus_home@0s")] > 0)


def test_extra_features_shape_and_blocks():
    P = _scene()
    rows = np.arange(0, 200, 3)
    X, names, blocks = extra_features(P, rows)
    assert X.shape == (len(rows), len(names)) and set(blocks) == {"who_first", "kick_target", "bounce", "team_shape"}
    assert sum(len(v) for v in blocks.values()) == len(names)
