"""The 6 M1 stress-eval matches (never trained on, in M1 or here) in the M1 package format, built on the fly from M1's own
processed files (the same sources hpc/tools/export_hpc.py uses), so degrade_pff.to_25fps / clean_match / degrade_match
take them unchanged.

Only STRESS_EVAL is allowed: the 16 locked test matches are refused.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

M1_ROOT = Path(os.environ.get("M1_ROOT", r"Z:\GP\Task 2 - match sim"))
STRESS_EVAL = (3816, 3829, 3847, 3852, 10512, 10514)          # M1 src/m1_possession/splits.py
LOCKED_TEST = (3822, 3825, 3827, 3828, 3834, 3836, 3842, 3844, 3848, 10503, 10505, 10508, 10509, 10510, 10511, 10513)


def load_stress_match(mid: int) -> dict:
    if mid in LOCKED_TEST or mid not in STRESS_EVAL:
        raise ValueError(f"match {mid} is not a stress-eval match (locked test matches are never read)")
    base = M1_ROOT / "data" / "processed" / "m1_possession"
    z = np.load(base / "inputs" / f"{mid}.npz")
    lab = np.load(base / "labels" / "v2" / f"{mid}.npz")
    pos = np.load(sorted((M1_ROOT / "data" / "processed" / "event_detection" / "cache").glob(f"{mid}*.positions.npz"))[0])
    meta = json.loads(str(pos["meta"]))
    home = int(json.loads((base / "home_team.json").read_text())[str(mid)])
    li = {int(f): i for i, f in enumerate(lab["frame_num"])}
    il = np.array([li[int(f)] for f in z["frame_num"]])
    side = np.array([0 if k.startswith("home") else 1 for k in meta["player_key"]], np.int8)
    f32 = lambda a: np.asarray(a, np.float32)
    return dict(match_id=np.int32(mid), fold=np.int8(-1), fps=np.float32(z["fps"]), pitch=z["pitch"].astype(np.float32),
                home_team_id=np.int32(home),
                home_attacks_right=np.array([bool(meta["home_attacks_right"].get(str(p), False)) for p in (1, 2, 3, 4)]),
                t=f32(z["t"]), period=z["period"].astype(np.int8),
                ball_x=f32(z["bx"]), ball_y=f32(z["by"]), ball_vx=f32(z["bvx"]), ball_vy=f32(z["bvy"]),
                ball_flag=z["bflag"].astype(np.int8), player_x=f32(z["px"]), player_y=f32(z["py"]), player_side=side,
                label_state=lab["state"][il].astype(np.int8), label_team_id=lab["team_id"][il].astype(np.int32),
                label_score_mask=lab["score_mask"][il])
