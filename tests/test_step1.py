"""Toy-array tests of the step-1 pipeline pieces (no data)."""
import numpy as np

from barca_possession import c3t as T
from barca_possession import postrules as PR
from barca_train.degrade_pff import degrade_ball, degrade_players, rates_from
from barca_possession.load import gap_stats, slot_stats
from barca_train.pipeline import score_hda

H, A, D = 0, 1, 2


def test_viterbi_follows_clear_emissions_and_respects_periods():
    n = 60
    p_dead = np.r_[np.full(20, 0.95), np.full(40, 0.05)]
    q = np.r_[np.full(40, 0.9), np.full(20, 0.1)]
    prow = np.r_[np.full(30, 1), np.full(30, 2)].astype(np.int8)
    C = np.array([[50, 1, 1], [1, 50, 1], [1, 1, 50]], float)
    path = T.decode(p_dead, q, prow, C, 1.0, 1.0)
    assert (path[:20] == D).all() and (path[25:40] == H).all() and (path[45:] == A).all()


def test_rows_to_frames_nearest_row_within_period():
    period = np.r_[np.zeros(3), np.ones(12), np.zeros(2), np.full(10, 2)].astype(np.int8)
    rows = T.rows_of(period)
    path = np.arange(len(rows)) % 3
    fr = T.rows_to_frames(path.astype(np.int8), rows, len(period), period)
    assert (fr[period == 0] == -1).all()
    assert fr[rows[0]] == path[0] and fr[rows[1]] == path[1]


def test_gap_same_team_fills_other_team_and_short_dead_keeps_long_dead():
    lab = np.array([H] * 5 + [A] * 3 + [D] * 10 + [D] * 60 + [H] * 5, np.int8)
    miss = np.zeros(len(lab), bool)
    miss[5:78] = True
    out = PR.gap_same_team(lab, miss, np.ones(len(lab), np.int8))
    assert (out[5:8] == H).all() and (out[8:78] == D).all()            # the 70-frame D run (2.8 s) is a real stoppage
    lab2 = np.array([H] * 5 + [A] * 3 + [D] * 10 + [H] * 5, np.int8)
    miss2 = np.zeros(len(lab2), bool); miss2[5:18] = True
    assert (PR.gap_same_team(lab2, miss2, np.ones(len(lab2), np.int8))[5:18] == H).all()


def test_kickoff_rule_sets_dead_before_the_centre_kick():
    n = 400
    period = np.ones(n, np.int8)
    speed = np.zeros(n); speed[200:] = 10.0
    bx, by = np.zeros(n), np.zeros(n)
    lab = np.full(n, H, np.int8)
    kicks = PR.kick_frames(speed, period)
    assert kicks[0] == 200
    out = PR.kickoff(lab, period, kicks, bx, by)
    assert (out[:200] == D).all() and (out[200:] == H).all()


def test_set_pieces_restart_team_after_kick():
    n = 1000
    period = np.ones(n, np.int8)
    speed = np.full(n, 1.0); speed[300:320] = 10.0                  # ball moving -> stops -> kick at 600
    speed[600:] = 8.0
    lab = np.r_[np.full(500, A), np.full(80, D), np.full(420, A)].astype(np.int8)
    kicks = PR.kick_frames(speed, period)
    tags = [dict(start_frame=590, end_frame=650, team=H)]
    out = PR.set_pieces(lab, tags, kicks, speed, period)
    assert (out[600:] == H).all() and (out[580:600] == D).all()
    out2 = PR.set_pieces(lab, [dict(start_frame=590, end_frame=650, team=None)], kicks, speed, period)
    assert (out2[600:] == H).all()                                  # N/A -> the team that did not have it (A) -> H


def test_score_hda_perfect_is_one():
    gt = np.r_[np.full(100, D), np.full(200, H), np.full(150, A)].astype(np.int8)
    period = np.ones(len(gt), np.int8)
    s = score_hda(gt, np.ones(len(gt), bool), period, gt.copy())
    assert abs(s["spell"] - 1) < 1e-9 and abs(s["macro"] - 1) < 1e-9


def test_degradation_hits_target_missing_share_and_keeps_teams():
    rng = np.random.default_rng(0)
    n, P = 25 * 600, 22
    t = np.arange(n) / 25
    period = np.ones(n, np.int8)
    px = rng.normal(0, 20, (n, P)).astype(np.float32).cumsum(0) * 0.001
    py = px.copy()
    team = np.r_[np.ones(11), -np.ones(11)].astype(np.float32)
    bx = np.sin(t).astype(np.float32); by = np.cos(t).astype(np.float32)
    gap = dict(slow=dict(counts=[1, 1, 1, 1, 1, 0, 0, 0]), fast=dict(counts=[1, 1, 1, 1, 1, 0, 0, 0]), missing_frac_graded=0.4)
    nbx, nby = degrade_ball(bx, by, np.full(n, 6.0, np.float32), period, rng, gap)
    assert 0.38 < np.isnan(nbx).mean() < 0.45
    stats = dict(slots=dict(mean_slots_total=20.5, slot_starts_per_min=25.0))
    nx, ny, ts = degrade_players(px, py, team, period, rng, rates_from(stats))
    assert set(np.unique(ts)) <= {1.0, -1.0}
    assert np.isfinite(nx).sum(1).mean() < 22
    g = gap_stats(t, period, nbx, nby, 34.0)
    assert g["n_gaps"] > 0 and abs(g["missing_frac_graded"] - np.isnan(nbx).mean()) < 1e-9
    s = slot_stats(period, nx, ts)
    assert s["slot_starts_per_min"] > 0
