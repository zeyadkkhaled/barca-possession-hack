# Step 1b: checks before step 2

All numbers are aggregates from the scripts. CV = the same 5 folds of the 30 PFF learning matches (M1 folds.json),
scorer `score_had` (= the official scorer without X), LOOSE → last team, no post-rules.

## 1. Clean PFF vs degraded PFF vs M1's C3

| run | input | spellF1 mean per match | macroF1 mean per match | spellF1 pooled | macroF1 pooled |
|---|---|---|---|---|---|
| C3-T, clean PFF | 25 fps, player identities kept, true ball (only PFF's own gaps) | 0.6449 | 0.9122 | 0.6418 | 0.9178 |
| C3-T, degraded (dead_weight 8, step 1) | Metrica form | 0.6150 | 0.9018 | 0.6122 | 0.9081 |
| C3-T, degraded, dead_weight 1 | Metrica form, gaps placed without labels | 0.6142 | 0.9007 | 0.6112 | 0.9069 |
| M1 C3 (round 4 final) | 29.97 fps, player ids, smoothed raw ball; held-out per-match counts from M1 | 0.5001 | 0.8730 | 0.4978 | 0.8816 |

Per fold (spellF1 / macroF1, mean per match, no post-rules):

| fold | clean | degraded (step 1) | degraded, dead_weight 1 |
|---|---|---|---|
| 0 | 0.6384 / 0.9103 | 0.6094 / 0.9011 | 0.6136 / 0.9006 |
| 1 | 0.6579 / 0.9154 | 0.6194 / 0.9047 | 0.6172 / 0.9031 |
| 2 | 0.6319 / 0.9011 | 0.6040 / 0.8905 | 0.5979 / 0.8889 |
| 3 | 0.6471 / 0.9173 | 0.6249 / 0.9079 | 0.6301 / 0.9087 |
| 4 | 0.6492 / 0.9168 | 0.6175 / 0.9049 | 0.6123 / 0.9022 |

Ball missing share (graded frames) / missing before restarts / mean slots on the pitch:

- clean: 33.2 % / 54.2 % / 22.00
- degraded, dead_weight 8: 41.0 % / 61.9 % / 20.73
- degraded, dead_weight 1: 40.9 % / 59.4 % / 20.73

Reading:
- Aggregation does not explain it: M1's C3 is 0.4978 pooled and 0.5001 as a mean per match (30 matches).
- Clean minus degraded: spellF1 +0.0298 (mean per match), +0.0295 (pooled).
- C3-T on clean PFF minus M1 C3, same matches, folds and scorer: spellF1 +0.1440 pooled. Two things still differ: 25 vs 29.97 fps (spells are compared on the resampled frames), and C3-T's 3-state decoder has w / beta chosen on inner out-of-fold spellF1 with a team-level layer 2, while M1's C3 decodes a player owner.

## 2. Leak check (raw ball only, no ballsSmoothed, no tags)

**Verdict: no leak in the model inputs.** No smoother, ball-from-players or C3-T feature reads PFF's `ballsSmoothed`
or anything built from event tags. One input is placed with the help of labels on purpose: where the degradation
cuts ball gaps (see item 3).

Path of the ball that the features see, from the PFF file to the features:

| step | file:line | what it reads |
|---|---|---|
| 1 | M1 `src/football_fatigue/ingestion/loaders.py:475-493` | raw `balls[0]` x/y/z go to `BallSample.raw`; `ballsSmoothed` is kept in separate fields |
| 2 | M1 `src/football_fatigue/event_detection/inputs.py:122` | `pf.ball.raw` only (docstring line 6: ballsSmoothed is built from event labels, so it is not used) |
| 3 | M1 `scripts/m1/build_labels_v1.py:49` | raw-ball visibility from the `"balls":[{"visibility":…}` list (VISIBLE / ESTIMATED from the tracker, not from events) |
| 4 | M1 `scripts/m1/build_inputs.py:41` | M1 `smooth_ball` on the step-2 raw ball and its raw visibility; `src/m1_possession/smoother.py` is guarded by `tests/test_m1_smoother.py:116-122` (no `game_event`, `possession_event`, `ballsSmoothed` or `PlayersSmoothed` in the code) |
| 5 | `hpc/tools/export_hpc.py:75,103` | the package's `ball_x/y/vx/vy/flag` = that smoothed raw ball (flag 3 "estimated" = the tracker's ESTIMATED visibility) |
| 6 | `src/barca_train/degrade_pff.py:80-96` | `ball_true_x/y` = package ball where flag != 0 |
| 7 | `src/barca_train/degrade_pff.py:268` | model ball `bx/by` = `ball_true` with extra gaps cut |
| 8 | `src/barca_possession/ballfill.py:66,88,100` | smoother and fill read `m["bx"], m["by"]`, players `px/py/team_sign` |
| 9 | `src/barca_possession/c3t.py:56-57,105-111` | features read only the fill (`ball[...]`) and `t, period, px, py, team_sign, pitch` |

The true ball and the labels are read only as training targets or for scoring:
- `ballfill.py:67-72`: ball-from-players targets.
- `c3t.py:150-155`: decoder transition counts.
- `pipeline.py`: layer targets and scores.

The one label-driven input: `degrade_pff.py:193-208` (`degrade_ball`, `dead = lab == 2`) starts ball gaps
dead_weight times more often on answer-key D frames, so the ball is missing before restarts about as often as in the
hackathon match. This ties ball-missing to D in the training data; item 3 measures how much CV gains from it.

Note on the "raw" ball: the PFF package ball is the raw `balls` field after **M1's own smoother** (repeat and spike
removal, short gap fill, RTS). That is our code, not PFF's tag-based ball. It is still cleaner than the Metrica raw ball,
which our pipeline smooths only once.

## 3. Without the dead-ball gap weighting

| | dead_weight 8 (step 1) | dead_weight 1 | difference |
|---|---|---|---|
| spellF1, mean per match | 0.6150 | 0.6142 | -0.0008 |
| macroF1, mean per match | 0.9018 | 0.9007 | -0.0012 |
| spellF1, pooled | 0.6122 | 0.6112 | -0.0011 |
| macroF1, pooled | 0.9081 | 0.9069 | -0.0012 |

Both runs re-train ball-from-players on their own degraded matches (out of fold), then run the same CV.

## 4. Top 10 features by gain (final models in models/c3, fit on all 30 degraded matches)

Layer 1, P(D), 124 features: `ball_speed@+2`, `ball_speed@+0.8`, `ball_speed@+0.4`, `since_beyond_s`, `top3_sp@+0.8`, `top3_sp@+2`, `top3_sp@+0`, `top3_sp@-0.4`, `ball_out_y`, `top3_sp@+0.4`.

Layer 2, P(H | alive), 134 features: `away_sdy`, `home_sdy`, `dh_minus_da@-2`, `dh_minus_da@-1`, `away_cvx`, `home_cvx`, `dh_minus_da@-0.4`, `home_sdx`, `ball_minus_home_cx`, `dmin@-0.8`.

`@+2` means the value 2 s later (offline, both sides of the frame).

## Commands

```powershell
.\.venv\Scripts\python.exe -m barca_train.degrade_pff --clean --out outputs\pff_clean --auto-name degrade_stats_clean.json
.\.venv\Scripts\python.exe scripts\ballfill_eval.py --deg outputs\pff_clean --train-deg outputs\pff_degraded --fill outputs\ballfill_clean --no-final --out-name ballfill_eval_clean.json
.\.venv\Scripts\python.exe scripts\cv_c3t.py --deg outputs\pff_clean --fill outputs\ballfill_clean --out-name cv_c3t_clean.json
.\.venv\Scripts\python.exe -m barca_train.degrade_pff --dead-weight 1 --out outputs\pff_degraded_dw1 --auto-name degrade_stats_dw1.json
.\.venv\Scripts\python.exe scripts\ballfill_eval.py --deg outputs\pff_degraded_dw1 --fill outputs\ballfill_dw1 --no-final --out-name ballfill_eval_dw1.json
.\.venv\Scripts\python.exe scripts\cv_c3t.py --deg outputs\pff_degraded_dw1 --fill outputs\ballfill_dw1 --out-name cv_c3t_dw1.json
.\.venv\Scripts\python.exe scripts\feature_importance.py
.\.venv\Scripts\python.exe scripts\make_step1b_report.py
```
