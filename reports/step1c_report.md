# Step 1c: extra slots, robust team features, stress matches

Aggregate numbers only; nothing per frame was printed or read by the assistant. CV = 5 folds of the 30 PFF learning matches (M1 folds.json), scorer `score_had` (= the official scorer without X), LOOSE → last team, no post-rules. The 16 locked test matches were never read (`barca_train.stress` refuses them).

## 1. The 0.5 m duplicate rule (kept, minor)

Rule: in a frame where a team has more than 11 slots, a same-team pair under 0.5 m apart with velocities differing by under 1 m/s is merged by dropping the shorter-lived slot.

| | value |
|---|---|
| match: graded frames with a team > 11, before → after | 12.9 % → 12.7 % |
| match: frames with a merge | 0.13 % |
| clean PFF: frames where the rule merges two real teammates | 0.010 % |
| clean PFF: same pair test without the >11 gate | 6.73 % |

The extra slots are not near-copies (nearest same-team pair median 3.6 m on over-11 frames; reports/auto/dup_diag.json), so the rule barely acts. It stays in the pipeline (`--dedup`) because it is safe.

## 2. What the extra slot is (match, frames where a team has > 11)

Graded frames with a team over 11: 12.9 % (20450 team-frames). The extra slot is the team's most recently born slot in that frame.

| measure | value |
|---|---|
| other team's slots ≤10 / 11 / >11 | 83.9 % / 15.9 % / 0.2 % |
| total slots ≤22 / 23 / >23 | 81.5 % / 16.9 % / 1.5 % |
| run share within 15 m of play, median: extra / regular | 0.35 / 0.46 |
| run mean speed, median m/s: extra / regular | 1.57 / 1.57 |
| run length, median s: extra / regular | 21.0 / 39.8 |
| team label flips within a slot | 0 % (the team is fixed per slot in the metadata XML) |
| run starts ≤2 m / ≤1 s after an OTHER-team slot ended: extra / regular | 9.7 % / 10.5 % |
| same, SAME-team slot: extra / regular | 2.4 % / 5.0 % |

**Classification** (share of over-11 team-frames):

- (b) wrong team label (total 22, other team 10): **56.5 %**
- (a) extra person (total 23, other team 11): **15.1 %**
- other (e.g. total ≤ 21 with the other team ≤ 9, or > 23): 28.3 %

Over-11 episodes (consecutive frames, per team): b_wrong_label: 83, median 2.2 s, mean 5.6 s, 7.3 % of graded frames; a_extra_person: 44, median 1.6 s, mean 2.8 s, 1.9 % of graded frames; any_over11: 133, median 2.9 s, mean 6.2 s, 12.9 % of graded frames.

Reading: mostly (b), a player of the other team carried under this team's label for a few seconds (as hand-overs between slots happen to regular slots too, ~10 %, it looks like the tracker's team assignment slipping near opponents). The extra slot moves like a player (same speed) and is somewhat less often near play.

## 3. Simulation in degrade_pff (`--extra`)

On top of the step-1 degradation (dead_weight 8), seeded per match:
- (b) a real player's track moves for an episode into a new slot of the other team;
- (a) a referee-like path is added to a team: the true ball smoothed with a 3 s lag, plus a fixed 5–12 m offset.
- Episode lengths are drawn from the match's histograms and episodes start only where the receiving team already has ≥ 11 slots.

Result: frames with a team > 11 13.0 % (match 12.9 %); slots 20.75 (match 20.49); slot starts 13.02/min (match 11.19); wrong-label share of the episodes 0.79.

## 4. CV on degraded PFF + extra slots (duplicate rule on in every arm)

| features | spellF1 | macroF1 | spellF1 pooled | restart-team acc (matched share) | short-spell (<2 s) recall |
|---|---|---|---|---|---|
| current team features | 0.6129 | 0.9012 | 0.6102 | 0.959 (0.787) | 0.125 |
| A: median centre, IQR spread, p10 / p90 depth, median speeds | 0.6115 (-0.0014) | 0.9004 | 0.6087 | 0.956 (0.778) | 0.120 |
| B: trim each team to 11 per frame (newest-born slots dropped) | 0.6131 (+0.0003) | 0.9006 | 0.6102 | 0.958 (0.779) | 0.121 |

Step 1 for reference (degraded, no extra slots, no rule): spellF1 0.6150, macroF1 0.9018.

Decision rule: an option replaces the current features only if it beats them by ≥ 0.005 spellF1 (mean per match).
Chosen: **current**. Submission regenerated: **no**.

## 5. Stress-eval matches (6 M1 stress matches, never trained on)

| model | input | spellF1 | macroF1 | spellF1 pooled | restart-team acc (matched) | short-spell recall |
|---|---|---|---|---|---|---|
| step 1 (current) | clean | 0.5837 | 0.8892 | 0.5829 | 0.955 (0.748) | 0.107 |
| step 1 (current) | degraded | 0.5536 | 0.8789 | 0.5536 | 0.946 (0.700) | 0.091 |

"degraded" = the same degradation as the model's training data (the step-1 model: without extra slots).

The submission was not regenerated, so there is no new model and no "step 1c final" stress row. outputs/submission_c3.csv is still the step-1 file.

## 6. Files changed

- `src/barca_possession/load.py`: 0.5 m duplicate rule (`remove_duplicates`), `trim_to_11`, duplicate stats in load_stats.json.
- `src/barca_possession/c3t.py`: `prepare_players(mode, dedup)`; `features(mode=...)`, where mode "robust" = option A. Feature names are the same in every mode.
- `src/barca_train/degrade_pff.py`: extra-slot simulation (`extra_config`, `add_extra_slots`, `--extra`).
- `src/barca_train/pipeline.py`: restart-team accuracy and short-spell recall (`extra_metrics`, `merge_extra`).
- `src/barca_train/stress.py`, `scripts/stress_eval.py`: the stress-match loader (it refuses the locked test matches) and the stress evaluation.
- `scripts/cv_c3t.py`, `scripts/ballfill_eval.py`, `scripts/fit_predict.py`: `--mode` / `--dedup` flags; the CV also reports the extra metrics.
- `scripts/dup_diag.py`, `scripts/dup_safety.py`, `scripts/extra_slot_diag.py`, `scripts/make_step1c_report.py`: diagnostics and this report.
- `reports/auto/*.json`: aggregate results only.

## 7. Deviations

- Every CV arm, including "current", has the 0.5 m rule switched on, so "current" differs slightly from step 1. The step-1 reference in section 4 is a different run (no extra slots, no rule).
- Option B gets its own ball-from-players fill, trained on trimmed input, so the comparison is end to end.
- The simulation produces more slot starts than the match (13.0 vs 11.2 per minute). Its wrong-label share (0.79) matches the match's b / (a + b) = 56.5 / 71.6.
- The "other" class (28.3 %) is not simulated: a team over 11 while the other team has 9 or fewer.

## 8. Open questions

- Adding the extra slots costs little: CV goes from 0.6150 (step 1) to 0.6129. The extra slots do not seem to be the bottleneck. The larger gaps are the stress matches (0.584 clean, 0.554 degraded, against about 0.61 in CV) and short spells: spells under 2 s are recalled only about 12 % of the time.
- How large is the gap between the learning and the stress-eval matches? It shows up on clean input too, so it is a generalisation gap rather than a tracking-degradation effect.
