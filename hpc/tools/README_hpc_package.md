# M1 HPC package (v1): tracking + labels v2 for sequence models

Built by `scripts/m1/export_hpc.py` (repository *Task 2 – match sim*). Contents: the **30 learning matches** of M1, split into 5 folds
(`folds.json`). The 16 test and 18 stress matches are **not** included and must never be used for training or tuning.

## Inputs vs. answer key (leakage rule)

- Model inputs: `t`, `period`, `ball_*` (our own smoother on the raw PFF ball, no tags) and `player_*` (raw PFF player
  positions + raw visibility). Nothing in them comes from PFF event tags or PFF's smoothed tracking.
- `label_*` = the answer key (labels v2, built from PFF event tags). Use it **only as training target and for scoring** – never
  as an input feature.

## Files

```
folds.json            {"folds": [[match ids of fold 0], ..., [fold 4]]}
matches/<match>.npz   one file per match (np.load; compressed)
m1_hpc_loader.py      load_match, fold_split, frame_features, Windows (torch Dataset if torch is installed)
sizes.json            size per match
```

## One match file (n = frames at 29.97 fps, P = player columns of that match, about 28–32)

| key | dtype | shape | meaning |
|---|---|---|---|
| match_id, fold | int32, int8 | () | PFF game id, fold 0–4 |
| fps | float32 | () | 29.97 |
| pitch | float32 | (2,) | length, width in m |
| home_team_id | int32 | () | PFF team id of the home team |
| home_attacks_right | bool | (4,) | per period 1–4: home attacks +x |
| frame_num | int32 | (n,) | PFF frame number |
| t | float32 | (n,) | video time in s |
| period | int8 | (n,) | 1, 2 (3, 4 extra time) |
| ball_x, ball_y, ball_z | float16 | (n,) | our smoothed raw ball, m, pitch centre = (0, 0); NaN = missing |
| ball_vx, ball_vy | float16 | (n,) | ball velocity, m/s |
| ball_flag | int8 | (n,) | 0 missing, 1 observed, 2 interpolated (short gap), 3 estimated |
| player_x, player_y | float16 | (n, P) | raw PFF player position, m; NaN = player not on the pitch / not tracked this frame |
| player_vis | int8 | (n, P) | raw visibility: 0 absent, 1 VISIBLE, 2 ESTIMATED (tracker guessed, player off camera), 3 other |
| player_side | int8 | (P,) | 0 home, 1 away |
| player_team_id, player_jersey, player_id | int32, int16, int32 | (P,) | per column; player_id -1 = unknown |
| label_state | int8 | (n,) | 0 DEAD, 1 CONTROL, 2 TRAVELLING, 3 LOOSE, 4 UNKNOWN (not scored) |
| label_player_id | int32 | (n,) | player in CONTROL (else -1) |
| label_player_slot | int8 | (n,) | column of that player in `player_*` (-1 = none / not found) |
| label_team_id | int32 | (n,) | team of CONTROL / TRAVELLING (else -1) |
| label_score_mask | bool | (n,) | frame is scored (False = unscored: UNKNOWN, video missing, replay_gap, suspicious_travel, ...) |
| label_mask_rule | int8 | (n,) | why labels v2 unscored it: 0 none, 1 replay_gap, 2 suspicious_travel |

Columns are fixed per match (one column per side + shirt number). At most 22 columns hold a position in a given frame;
substitutes are NaN before they come on, replaced players NaN after they leave. Coordinates are float16
(resolution about 3 cm at 32–64 m).

Scored frames for training and evaluation: `label_score_mask & (label_state <= 3) & (period > 0)`.

## Quick start

```python
from m1_hpc_loader import load_match, fold_split, Windows
train, val = fold_split(".", val_fold=0)
ds = Windows(".", train, length=300, stride=150)   # 10 s windows
item = ds[0]; item["x"].shape                       # (300, P + 1, 6)
```

`Windows` keeps every match's per-frame tensor in RAM (about 1 GB for 24 matches at P ≈ 30).
