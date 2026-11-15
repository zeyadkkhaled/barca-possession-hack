# Technical document (draft, 2026-11-16)

Our method for the More than a Hack 2027 possession challenge: one label per frame (H home, A away, D dead ball,
X no answer) for FC Barcelona vs Athletic Club, 199,319 frames at 25 fps. Ranked on possession-spell F1 (spellF1),
tie-break macroF1. Only aggregate numbers appear here.

## 1. Approach in one paragraph

The hackathon gives one labelled match, and of it only 5 public minutes, which the rules allow only as a check. So every
supervised model is trained on **another dataset**: the public PFF FC FIFA World Cup 2022 tracking and event data. The
PFF data is **degraded** until it looks like Metrica's automatic tracking of this match: 25 fps, anonymous track slots
that break and swap, a missing ball, extra slots. The model then runs on the hackathon match without ever being fitted
to it. The submitted model is **C3-T**, a team-level two-layer LightGBM with a 3-state Viterbi decoder. **PathCRF**, a
graph model with a CRF, is being evaluated as a replacement for C3-T's team decision (section 6, TODO).

## 2. Training data: PFF World Cup 2022

- **Source:** PFF FC's public FIFA World Cup 2022 dataset (64 matches: broadcast tracking at ~29.97 fps, events,
  rosters, metadata), released free by PFF FC on request. *TODO: add the exact download link and licence wording before
  the repo is made public.* No PFF data is in this repository.
- **Answer key ("labels v2"):** built from PFF's events by `barca_labels.labeller` (states DEAD / CONTROL / TRAVELLING /
  LOOSE, team and player). For the hackathon these map to H/A/D: CONTROL and TRAVELLING go to that team, LOOSE to the
  last team, DEAD to D. The labels are used only as training targets and for scoring, never as model inputs.
- **Splits (from M1):** 30 learning matches in 5 fixed folds (`folds.json`), 18 stress matches and 16 locked test
  matches. The locked test matches were never read, and `barca_train.stress` refuses them. Of the stress matches, 6
  are used as an outside check.
- **Package:** one `.npz` per match (positions in centre-origin metres, ball with an observed / interpolated flag, labels,
  `home_attacks_right`), exported by M1's `hpc/tools/export_hpc.py`. To rebuild it you need the PFF download and the
  M1 labeller.

## 3. Degradation (`barca_train.degrade_pff`)

Every rate comes from aggregate measurements of the hackathon match (`barca_possession.load`, stored in
`reports/auto/load_stats.json`). The random draws are seeded per match (seed 0 + match id).

1. Resample to 25 fps in each period. Rescale to Metrica's 105.3 × 68 m pitch and orient it so home attacks +x.
2. Drop the player ids:
   - tracks break at random into new slots, with a short gap;
   - same-team tracks swap;
   - drop-out runs are cut in, at the match's rate of slot starts per minute and its share of absent players.
3. Ball: only PFF's present samples are kept. Extra missing runs are then cut until the match's 41 % missing share on
   graded frames is reached. Run lengths follow the match's gap-length histogram per ball-speed class. Gaps around
   dead balls get extra weight (dead_weight 8), calibrated so the share missing before restarts matches the match.
4. Extra slots (step 1c), copying the 12.9 % of frames where a team of the match has more than 11 slots:
   - wrong team label: a real player's track is moved into the other team for a few seconds;
   - extra person: a referee-like path is added.
   Episode lengths come from the match.

## 4. C3-T (the submitted model)

- **Input clean-up:** a duplicate rule merges same-team pairs under 0.5 m whose speeds differ by less than 1 m/s, only
  on frames where the team has more than 11 slots (`load.remove_duplicates`). The step-1 model was trained without it.
- **Ball:** a repeat / spike filter, gap filling, an adaptive RTS Kalman smoother, then a **ball-from-players** LightGBM
  that fills the frames that are still missing (`barca_possession.ballfill`). It is trained on degraded PFF.
- **Features:** team-level only, since there are no player ids. Every 5th frame gets team shapes and speeds, distances
  to the ball and kick signals, and windows of these over time (`barca_possession.c3t`).
- **Model:**
  - layer 1: P(dead);
  - layer 2: P(home | alive), stacked on the out-of-fold layer-1 output;
  - a 3-state Viterbi decoder on transition counts, with an emission weight and transition sharpness chosen on inner
    out-of-fold predictions.
  LightGBM runs deterministically (`deterministic=True`, fixed seed, 1 thread).
- **Post-rules:** kept only if their solo spellF1 gain in cross-validation is positive. None are kept for the current
  model.

## 5. Results so far

All CV numbers are 5-fold over the 30 PFF learning matches, mean per match, with our scorer (identical to the official
one without X: `scripts/check_scorer_parity.py`).

| setting | spellF1 | macroF1 |
|---|---|---|
| C3-T, degraded PFF (step 1, submitted model) | 0.6150 | 0.9018 |
| C3-T, clean PFF (no degradation) | 0.6449 | 0.9122 |
| C3-T, degraded + simulated extra slots + duplicate rule (step 1c) | 0.6129 | 0.9012 |
| options A (robust team statistics) / B (trim to 11), same data | 0.6115 / 0.6131 | 0.9004 / 0.9006 |
| 6 stress matches, never trained on: clean / degraded input | 0.584 / 0.554 | 0.889 / 0.879 |

**Hackathon match:**
- **Submission:** `outputs/submission_c3.csv` passes the official validator.
- **Public 5 minutes (sanity check only, never used to choose):** spellF1 0.538, macroF1 0.908. The official
  `baseline_smooth50` scores 0.326 / 0.621 on the same window.
- **Reproduction:** `python run.py` regenerates the file byte for byte (sha256 `8612514a…a8c5`, checked twice).

## 6. PathCRF (step 2) — TODO

PathCRF (Kim et al., KDD 2026; MPL-2.0, vendored in `vendor/pathcrf/` with its licence) models possession as a sequence
of ball links between 22 player nodes and 4 outside nodes, with a dynamic sparse CRF. M1 fine-tuned it on clean PFF.

What step 2a built:
- **Re-linking:** `barca_possession.relink` trims each team to 11 slots per frame, then links slots into 11 persistent
  lanes per team with the Hungarian method. The cost is the distance between a lane's predicted position and a new
  slot's first position, gated by 8 m/s. On the hackathon match, both teams have at least 10 lanes present on ≥ 80 %
  of the frames in 68.8 % of 10 s windows (90.2 % counting any frame), with 1.76 links per window.
- **Adapter:** `pathcrf_adapter`: 25 fps → 5 fps steps, lanes as nodes, our ball fill as an extra input.
- **Training targets:** on degraded PFF, an owner map tracks which true player each slot holds, so the answer key can
  be mapped onto lanes. On one test match, 91.8 % of the targets map to a lane, and lane purity is 99.6 % in 10 s
  windows.
- **The HPC cycle (`hpc/step2/`):**
  - zero-shot published weights;
  - fold-0 fine-tuning with and without the ball;
  - an optional all-30 model.
  Every variant is scored against C3-T with C3-T's dead-ball model and decoder, so only the team decision changes.

*TODO after the HPC run:* fold-0 results, the decision, and whether PathCRF goes into the submission.

## 7. Reproducibility

- `python run.py` turns the raw data plus `models/c3/` and `models/bfp/` into `outputs/submission.csv`. It does no
  training and is deterministic.
- Training from scratch needs the PFF package. The order is:
  1. `scripts/` step-1 commands;
  2. `barca_train.degrade_pff`;
  3. `scripts/ballfill_eval.py`;
  4. `scripts/cv_c3t.py`;
  5. `scripts/fit_predict.py`.
  See `reports/step1_report.md`.
- Python 3.13 with pinned packages (`pyproject.toml`). PathCRF needs PyTorch and runs on the HPC
  (`hpc/step2/HPC_STEP2.md`).
