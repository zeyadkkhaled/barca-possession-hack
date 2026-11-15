"""PathCRF adapter (no torch needed): targets in lane space, windows, scorer alignment. Synthetic data only."""
import json

import numpy as np

from pathcrf_adapter import adapter as A
from pathcrf_adapter import build as B
from pathcrf_adapter import score as S

FPS = 25


def test_player_targets_and_lane_mapping():
    # player 3 controls, travelling ball, player 15 controls, then LOOSE and DEAD with the ball out on the right
    st = np.array([1, 1, 2, 2, 1, 1, 3, 0, 0])
    pl = np.array([3, 3, -1, -1, 15, 15, -1, -1, -1])
    per = np.ones(9, int)
    bx = np.array([0, 0, 0, 0, 0, 0, 0, 60.0, 60.0]); by = np.zeros(9)
    kind, a, b, tl = B.player_targets(st, pl, np.ones(9, bool), per, bx, by, 105.3, 68.0)
    assert list(kind) == [1, 1, 2, 2, 1, 1, 3, 0, 0]
    assert list(a[:7]) == [3, 3, 3, 3, 15, 15, 15] and list(b[:6]) == [3, 3, 15, 15, 15, 15] and b[6] == 1   # right outside
    assert list(tl) == [-1, -1, 0, 0, -1, -1, 1, -1, -1]
    lane_owner = np.full((9, 22), -1)
    lane_owner[:, 2] = 3                       # player 3 on home lane 2
    lane_owner[:4, 13] = 15                    # player 15 on away lane 13 only for the first 4 frames
    k2, la, lb = B.lanes_to_targets(kind, a, b, lane_owner)
    assert list(k2) == [1, 1, 2, 2, 0, 0, 0, 0, 0]          # 15 unmapped from frame 4 on -> masked
    assert list(la[:4]) == [2, 2, 2, 2] and list(lb[:4]) == [2, 2, 13, 13]


def built_match(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    lx = rng.uniform(-50, 50, (n, 22)).astype(np.float32); ly = rng.uniform(-30, 30, (n, 22)).astype(np.float32)
    lx[:, 21] = np.nan                                       # one away lane empty all match
    lx[100:140, 3] = np.nan
    ly[np.isnan(lx)] = np.nan
    kind = np.ones(n, np.int8); a = np.full(n, 2, np.int16); b = np.full(n, 2, np.int16)
    a[500:600] = 21; b[500:600] = 21                         # a target on the empty lane must be masked
    return dict(lane_x=lx, lane_y=ly, period=np.r_[np.ones(n // 2), 2 * np.ones(n - n // 2)].astype(np.int8),
                t=np.arange(n) / FPS, ball_x=np.zeros(n, np.float32), ball_y=np.zeros(n, np.float32),
                ball_seen=np.ones(n, np.int8), pitch=np.array([105.3, 68.0], np.float32), kind=kind, a=a, b=b,
                tl=np.full(n, -1, np.int8), live_scored=np.ones(n, bool), match_id=np.int64(1), fold=np.int64(0))


def test_windows_shapes_masks_and_flip():
    M = A.Match(built_match())
    assert len(M.frames) == 200 and (M.frames[:3] == [0, 5, 10]).all()
    st = M.window_starts(5)
    assert all(M.period[s] == M.period[s + A.WIN - 1] for s in st)       # windows never straddle a period
    w = M.window(st[0])
    assert w["x"].shape == (A.WIN, A.N_NODES, 11) and np.isfinite(w["x"]).all()
    assert w["pmask"][0, 21] == 0 and w["pmask"][0, :21].all()          # empty lane = padding node
    assert (w["x"][:, :11, 0] == -1).all() and (w["x"][:, 11:21, 0] == 1).all() and (w["x"][:, 21, :8] == 0).all()
    w5 = M.window(100)                                                   # steps 100-149 = frames 500-745
    assert not w5["mask"][:20].any() and w5["mask"][20:].all()
    wf = M.window(st[0], flip_x=True)
    assert (wf["src"][wf["mask"]] == 11 + 2).all()                       # home lane 2 -> node 13 after the team swap


def test_score_aligns_pathcrf_steps_with_c3t_rows(tmp_path):
    n = 2000
    period = np.r_[np.ones(1000), 2 * np.ones(1000)].astype(np.int8)
    gt = np.where((np.arange(n) // 200) % 2 == 0, 0, 1).astype(np.int8)
    gt[900:1100] = 2
    rows = np.r_[np.arange(0, 1000, 5), np.arange(1000, 2000, 5)]
    (tmp_path / "data" / "c3t").mkdir(parents=True)
    (tmp_path / "c3t").mkdir(); (tmp_path / "preds").mkdir()
    np.savez(tmp_path / "data" / "c3t" / "7.npz", gt=gt, gmask=np.ones(n, bool), period=period)
    p_dead = (gt[rows] == 2).astype(np.float32) * 0.9 + 0.05
    q_true = (gt[rows] == 0).astype(np.float32)
    np.savez(tmp_path / "c3t" / "7.npz", rows=rows, p_dead=p_dead, q_home=np.full(len(rows), 0.5, np.float32), prow=period[rows])
    C = np.full((3, 3), 1.0) + np.eye(3) * 200
    (tmp_path / "c3t" / "decoder.json").write_text(json.dumps(dict(C=C.tolist(), w=1.0, beta=1.0, fold=0, held=[7])))
    np.savez(tmp_path / "preds" / "7.npz", frames=rows.astype(np.int32), q_home=0.05 + 0.9 * q_true)
    r = S.score(tmp_path / "data", tmp_path / "c3t", tmp_path / "preds")
    assert r["pathcrf"]["spell"] > 0.9 and r["pathcrf"]["spell"] > r["c3t"]["spell"]
    assert set(r) >= {"c3t", "pathcrf", "blend", "_meta"} and r["_meta"]["missing_preds"] == []
