# Technical document: Visca Barca, More than a Hack 2027 (ball possession)

Team: Zeyad Khaled, Ahmed Hassan. With thanks to Omar Essam. Only aggregate numbers appear in this document.

## 1. Problem

The task is one label per video frame of FC Barcelona vs Athletic Club: 199,319 frames at 25 fps, from Metrica Sports'
automatic tracking. The labels are:
- **H:** home team in possession;
- **A:** away team in possession;
- **D:** dead ball;
- **X:** no answer.

Submissions are ranked on **possession-spell F1** (spellF1), with frame-level macroF1 as the tie-break. spellF1 is the
F1 of predicted vs true possession spells, averaged over IoU thresholds 0.3–0.7.

The hard parts:
- **Anonymous tracking:** the tracking has no player ids. Slots break, swap and sometimes hold referees or a player of
  the other team; a team has more than 11 slots on 12.9 % of frames.
- **Missing ball:** the ball is missing on 40.7 % of the graded frames, and on 66.8 % of the frames just before a set
  piece.
- **Almost no labels:** only one labelled match exists. Of it, only 5 public minutes, which the rules allow only as a
  check.

## 2. Method

### 2.1 Idea

Every model is trained on **another dataset**, the PFF FC World Cup 2022 tracking. That data is **degraded** until it
looks like this match's tracking (section 4). The model never sees the hackathon match during training, and nothing is
tuned on the public 5 minutes.

### 2.2 Ball (`barca_possession.ballfill`)

1. A repeat / spike filter and short gap filling.
2. An adaptive RTS Kalman smoother.
3. A **ball-from-players** LightGBM (`models/bfp`) fills the frames that are still missing. It predicts the ball's
   offset from the players' positions and motion, and is trained on degraded PFF where the true ball is known.

### 2.3 C3-T (`barca_possession.c3t`, `models/c3`)

- **Features:** team-level only, since there are no player ids. Every 5th frame gets 124 features:
  - team shapes and speeds;
  - distances from each team to the ball;
  - ball speed and kick signals;
  - rolling windows of these over time.
- **Layer 1:** a LightGBM classifier for P(dead ball).
- **Layer 2:** a LightGBM classifier for P(home | ball alive). It is stacked on the out-of-fold layer-1 output.
- **Decoder:** a 3-state (H / A / D) Viterbi over the 5-frame rows.
  - Transition costs come from training transition counts.
  - The emission weight w = 1.0 and transition sharpness β = 0.75 were chosen on inner out-of-fold predictions.
  - The rows are then expanded to every frame.
- **Post-rules:** a rule was kept only if it raised spellF1 on its own in cross-validation. None was kept.
- **Determinism:** LightGBM runs with `deterministic=True`, a fixed seed and 1 thread.

### 2.4 Answer key: LOOSE → last team

The PFF answer key ("labels v2", `barca_labels.labeller`) has four states: DEAD, CONTROL, TRAVELLING (a pass in the
air) and LOOSE. They map to H / A / D as follows:
- CONTROL and TRAVELLING go to the team in control;
- LOOSE goes to the **last team that had the ball**;
- DEAD goes to D.

The labels are only training targets and scoring keys, never model inputs.

### 2.5 The possession rules (where we drew the lines)

The rules live in `barca_labels.labeller` and apply only to the PFF training matches. Our model learns them from those
labels; no rule is written per frame of the hackathon game.

- **Control:** a player has the ball from PFF's on-the-ball event until the next event. TRAVELLING (a pass, cross or
  shot in flight) belongs to the team that played it, until the next touch.
- **Loose balls:** a clearance, a rebound, a deflection, a touch that pops away or a carry that loses the ball becomes
  LOOSE, scored as **the last team that touched it**. Possession changes only when the other team controls the ball.
- **Challenges:** a spell that holds only 50-50 or aerial challenges is LOOSE, again given to the last team in
  possession. A challenge does not by itself hand possession to the challenger.
- **Restarts:** the ball is DEAD from the moment it goes out (or the period ends) until the next spell starts. For a
  set piece (throw-in, corner, goal kick, free kick, kick-off) it stays DEAD until the restart is **played** (the first
  pass, cross, shot or clearance). A drop ball is live from its start (Laws of the Game).
- **Not scored:** frames PFF marks as missing video or as a non-event, and frames outside the periods. Any unclear case
  gets its own named rule and is counted, never guessed.

### 2.6 Why this approach

- **No labels on the game:** the provided game cannot be labelled, so a supervised model must learn on other games.
  PFF World Cup 2022 is the largest public tracking set with event data.
- **Training on degraded data:** a model trained on clean PFF and applied to Metrica's tracking would meet missing
  balls and broken slots it never saw. Degrading PFF to the game's measured error rates closes that gap (section 4).
- **Team-level model:** Metrica's slots have no stable identity, so per-player models (M1 earlier tried rule-based and
  per-player variants, C0–C2) lose the ball's owner every time a slot breaks. Team-level features survive slot breaks.
- **LightGBM + Viterbi:** fast, deterministic on CPU and reproducible byte for byte. The decoder turns noisy
  frame-level probabilities into clean spells, which is what spellF1 rewards.

## 2.7 Provided assets: used and dropped

| asset | use |
|---|---|
| Tactical-camera tracking (Metrica EPTS) + metadata | **used**: the model's only input at prediction time |
| Tracking statistics, without labels | **used**: aggregate error rates (ball-missing share, gap lengths, slot breaks, extra slots) set the PFF degradation (section 4). Computed by code, never tied to frames |
| Smart Tagging (Metrica's automatic tags) | **used for calibration only**: the share of frames with the ball missing just before the 59 set-piece tags set the dead-ball gap weight. **Dropped as an input:** post-rules built on set-piece tags did not raise spellF1 in CV, so the model does not read them |
| Homography, broadcast video and its ATD, tactical video | **not used** |
| Public 5-minute ground truth | **checks only** (file format, frame alignment), never training or tuning |
| PathCRF (published graph model, MPL-2.0) | **tried, dropped for the submission**: not finished in time (section 7) |

## 3. Training data

- **Source:** PFF FC's FIFA World Cup 2022 dataset: 64 matches, broadcast tracking at ~29.97 fps, events, rosters and
  metadata.
  - PFF FC released it free for research and analysis. Request access on PFF FC's website.
  - No PFF data is in this repository.
- **Who labelled it and how much:**
  - PFF FC's analysts produced the event data;
  - our code (`barca_labels.labeller`) turns those events into one possession label per frame;
  - **no one on our team hand-labelled anything.**
  - The model is trained on **30 full matches** (about 3,000 minutes of play).
- **Nothing from the provided game:** no labels from it are used for training or validation. Only the unlabelled
  aggregate error rates of its tracking (section 2.7) shape the training data.
- **Package:** one `.npz` per match, built by `scripts/data/export_pff_package.py`. It holds positions in centre-origin
  metres, the ball with an observed / interpolated flag, labels v2 and the attacking direction.
- **Splits** (`barca_possession.splits`, fixed seeds):

  | set | matches | how it is used |
  |---|---|---|
  | learning | 30 | training, in 5 fixed folds for cross-validation |
  | locked test | 16 | read once, at the very end (`docs/dev_log/LOCKED_USED.md`) |
  | stress | 18 | an outside check; 6 are used |

## 4. Degrading PFF to look like the match (`barca_train.degrade_pff`)

- Every target below was measured on the hackathon match by `barca_possession.load` (`reports/auto/load_stats.json`).
- The random draws are seeded per match: seed 0 + match id.
- Achieved values are averages over the 30 learning matches (`reports/auto/degrade_stats.json`).

| step | what it does | target (match) | achieved (degraded PFF) |
|---|---|---|---|
| 1. frame rate and pitch | resample to 25 fps per period; rescale to 105.3 × 68 m; orient home to +x | 25 fps | 25 fps |
| 2. track breaks | a track ends and continues in a new slot after a short gap | 11.2 slot starts per min | 9.7 per min |
| 3. swaps | two same-team tracks exchange slots | rate from the match's slot statistics | 0.0011 per player-second |
| 4. drop-outs | a player is absent for a run | 10.2 / 10.3 mean slots home / away | 10.4 / 10.4 |
| 5. ball missing | keep PFF's present samples, then cut missing runs from the match's gap-length histogram per ball-speed class | 40.7 % of graded frames missing | 41.0 % |
| 6. dead-ball gaps | extra weight on gaps around restarts (dead_weight 8, calibrated) | 66.8 % missing before set pieces | 61.9 % before restarts |

**Extra slots** (12.9 % of the match's frames have a team with more than 11 slots) were studied in step 1c:
- **Simulation:** wrong-team episodes plus referee-like paths.
- **Input rule:** a duplicate rule merges same-team pairs closer than 0.5 m whose speeds differ by less than 1 m/s.
- **Outcome:** neither improved cross-validation (0.6129 vs 0.6150), so the submitted model uses neither.

## 5. Evaluation

All scores are the mean per match. Our scorer equals the official one when the prediction has no X
(`scripts/check_scorer_parity.py`).

| evaluation | input | spellF1 | macroF1 |
|---|---|---|---|
| 5-fold CV, 30 learning matches | clean PFF | 0.645 | 0.912 |
| 5-fold CV, 30 learning matches | degraded PFF | 0.615 | 0.902 |
| 6 stress matches (never trained on) | clean | 0.584 | 0.889 |
| 6 stress matches (never trained on) | degraded | 0.554 | 0.879 |
| 16 locked test matches (one run) | degraded | 0.617 (95% CI 0.598–0.636) | 0.902 |
| 16 locked test matches (one run) | clean | 0.644 (95% CI 0.624–0.665) | 0.911 |
| hackathon public 5 min, official scorer | real | 0.539 | 0.908 |
| official `baseline_smooth125`, same 5 min | real | 0.424 | 0.581 |

Locked test (16 matches, one run, `reports/auto/locked_test.json`; 95 % interval = per-match bootstrap of the mean):
- short-spell recall (true spells under 2 s found at IoU ≥ 0.5): 0.113 degraded, 0.138 clean;
- restart-team accuracy (team after a dead ball, on matched restarts): 0.975 degraded, 0.983 clean.

The locked scores (0.617 degraded, 0.644 clean) match the CV scores (0.615 / 0.645), so the CV did not overfit.

- **Public 5 minutes:** used only to check that the pipeline runs end to end. They contain only 17 possessions, so the
  number is noisy.
- **Model choices:** every choice (features, decoder settings, post-rules, degradation settings) was made on CV over the
  learning matches.

## 6. Limits

- **Degraded vs real:** the degradation copies aggregate rates, not the exact error process of Metrica's tracker.
  Slot breaks are slightly rarer than in the match (9.7 vs 11.2 per minute).
- **Different data source:** PFF comes from broadcast tracking of World Cup matches; the hackathon match is a club
  match from a different tracking system.
- **Team-level model:** it cannot follow individual players, so short possessions (under 2 s) and quick restarts are
  its weakest part.
- **Dead balls:** dead-ball detection depends on the ball fill when the ball is missing around restarts.

## 7. Future work: PathCRF

PathCRF (Kim et al., KDD 2026; MPL-2.0) models possession as ball links between player nodes with a dynamic sparse
CRF.
- **Fold-0 test on clean PFF:** fine-tuned PathCRF reached spellF1 **0.770 vs 0.638** for C3-T (6 matches).
- **Plan:** a combination where C3-T decides dead balls and PathCRF decides the team.
- **What was built:** re-linking the anonymous slots into 22 persistent lanes (Hungarian assignment), an adapter to
  25 fps, and an HPC training cycle.
- **Status:** not finished in time, so it is not part of the submission. That code is kept on the `dev-history` branch.

## 8. Use of AI

- **What the assistants did:** coding assistants wrote code from public documentation (the official repository, library
  docs) and from our instructions.
- **What they never saw:** no match data and no extract of it (rows, coordinates, per-frame labels, plots). Scripts that
  read data were run by the team, and only aggregate numbers (scores, counts, shares) were shared.

## 9. Team

Zeyad Khaled, Ahmed Hassan (team Visca Barca). With thanks to Omar Essam.

## 10. Reproduction

```bash
git clone --recurse-submodules https://github.com/zeyadkkhaled/barca-possession-hack.git
cd barca-possession-hack
python -m pip install -r requirements.txt          # Python 3.13, pinned versions
python run.py --out outputs/submission_c3.csv      # data in data/ or MTAH_DATA_DIR
```

- **What it does:** reads the hackathon data and the committed models (`models/c3`, `models/bfp`), writes the 199,319-row
  file and runs the official validator.
- **Output:** sha256 `8612514a69e9c58045ab5c0ccdad20808aa48bdfd45d0566f8389eb43defa8c5`.
- **Cost:** about 30 s on a laptop CPU, peak 0.6 GB RAM. No training and no GPU.
- **Retraining:** needs the PFF data. The order is in `docs/dev_log/step1_report.md`:
  1. measure the match's rates (`barca_possession.load`);
  2. degrade PFF (`barca_train.degrade_pff`);
  3. ball-from-players (`scripts/ballfill_eval.py`);
  4. CV (`scripts/cv_c3t.py`);
  5. final fit (`scripts/fit_predict.py`).
