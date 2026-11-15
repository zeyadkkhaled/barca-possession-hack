# Handoff for planning: More than a Hack 2027 possession challenge (written 2026-11-15)

## 1. The hackathon in 10 lines
1. **Task:** for each of the **199,319 frames** (25 fps) of one match, FC Barcelona (H) vs Athletic Club (A), LaLiga
   2024-25, write `H` / `A` / `D` (dead ball) / `X` (abstain). 158,857 frames are graded (the two halves).
2. **Possession rule:** while the ball is alive, one team always has it. A **contested or loose ball stays with the last
   team**. Possession changes only on *controlled* possession by the opponent: a touch, a deflection or an aerial duel
   without control does not count. A dead ball is just `D`, with no owner. A possession is a maximal run of H or of A.
3. **The ground truth comes mechanically from Metrica's corrected tracking** (its ball-dead flag and ball-owner team),
   not from events or humans.
4. **Ranking = spellF1:** H/A spells matched one-to-one, same team, greedy by temporal IoU. F1 at IoU 0.3/0.4/0.5/0.6/0.7,
   averaged. The tie-break is macroF1 over H/A/D per frame. A 1 s lag costs about 35 %; gaps below 0.04 are noise.
5. **X** is forward-filled before spellF1 ("repeat my last label") and counts as a miss on macroF1. **It is almost never
   worth using on graded frames.**
6. **Baseline to beat:** spellF1 **0.2433** (nearest player to the ball, then a ±50-frame majority vote). Its full-match
   macroF1 is 0.6724.
7. **Data:**
   - automatic Metrica tracking: 46 track slots, **a team per slot but no player identity**, slots break and swap;
     coordinates normalised 0–1 on a 105.3 × 68 m pitch; **the ball is missing on 41 % of graded frames**;
   - Smart Tagging: 638 automatic phases and set pieces with a team tag; the team is N/A on every throw-in;
   - a homography for every frame; the tactical video; a broadcast video plus its tracking, *not synchronised*.
8. **Labels given:** only the **first 5 minutes** (17 possessions). They are for checking the pipeline, **not for
   training or tuning (a rule)**. The full-match GT stays private, and there is no leaderboard.
9. **Submission:**
   - the CSV (`frame,label`, exactly 199,319 rows);
   - a **public repo that regenerates exactly the same file**;
   - a short technical document that lists all training data.
   Teams have 3–5 members. Hand labels on this game are forbidden, and so is any human-made data about it. Data and
   hand labels about other games are fine if the organisers can access them.
10. ⚠ **Deadline: 17 November 2026, two days away; no time zone stated.** ⚠ The data needs enrolment first, and it is
    not downloaded yet. ⚠ **No match data or extract of it may go to a hosted LLM** (T&Cs), so a chat assistant can
    write code but must not see data rows, plots or per-frame outputs.

## 2. What we already have (built on PFF World Cup 2022: 64 matches, ~30 fps, clean tracking, player ids)
- **Answer key "labels v2"** with 4 states (DEAD / CONTROL / TRAVELLING / LOOSE). Split: 30 learning / 16 locked test /
  18 stress matches.
- **C3** (frozen):
  - LightGBM in 2 layers (alive/dead, then CONTROL / TRAVELLING / LOOSE plus a 5-candidate player ranker), with
    stacking and a per-transition HMM/Viterbi decoder;
  - tag-free inputs (tracking plus our smoothed ball);
  - 5-fold CV on the 30 learning matches: **frame accuracy 0.865, H/A/D macroF1 0.882, spellF1 0.498**;
  - **no saved model files**: the scripts retrain it each run.
- **PathCRF** (KDD 2026 graph model plus a CRF), fine-tuned from its published weights through our adapter, plus a
  travelling-vs-loose head. C3 decides DEAD; PathCRF decides team, player and switches.
  - Fold 0: **spellF1 0.64 vs C3 0.49**, restart team 98 % vs 57 %, team accuracy 94.4 % vs 91.5 %.
  - The model trained on all 30 matches (about 5 of 20 epochs) is on the HPC.
  - Ablation: keeping dead-ball windows, a 1 s stride and more epochs help; fine-tuning beats training from scratch.
- **Ball smoother** (repeat and spike removal, gap fill, adaptive RTS Kalman), a **ball-from-players** tree model (for
  missing ball; median error 2.65 m), a **restart detector** (ball-speed kick, then type and team from geometry), and
  **our scorer**.

## 3. Inventory verdicts (the full table is in the repo report)
| component | verdict | why |
|---|---|---|
| scorer `score_had` (H/A/D, spellF1) | **reuse as-is** | identical to the official scorer (see §4) |
| official scorer and validator | **reuse as-is** | the submodule; runs locally |
| ball smoother | adapt | 25 fps, 0–1 coordinates, ball z unknown, repeat-frame rule tuned to PFF |
| ball-from-players | adapt | very relevant (41 % of the ball missing); expects player columns in metres |
| C3 layer 1 (alive/dead) | adapt | tag-free, but trained on PFF's dead-ball definition and frame rate |
| C3 layer 2, owner ranker, decoder | adapt | the ranker and the "owner fixed between touches" assume persistent player ids |
| C3 stacking and pipelines (scripts) | adapt | load PFF from M1 paths; no saved model; retraining needed |
| restart detector | adapt | logic in m/s and seconds is portable; Smart Tagging SET PIECES is a free restart prior |
| C0 baseline | adapt (sanity only) | about the same as the official baseline |
| C1/C2 rules, labeller, splits | not needed for output | C3 imports C2; only needed if we retrain on PFF |
| PathCRF adapter, head, vendored code | adapt | needs 22 persistent player nodes, a keeper flag and the attacking side; built for 29.97/6 fps |
| HPC bundle (Slurm, run_cycle, setup_env) | adapt | paths and module names are M1's; the env is reusable |
| PathCRF final weights (HPC) | adapt | PFF domain (clean tracking); the inputs must be rebuilt the same way |
| config / scratch-dir helper | reuse as-is | the default is `D:\m1_tmp`; override with `M1_TMP` |

## 4. Does our scorer match the official one?
**Yes.**
- Our `score_had` (with LOOSE → last team) gives the same macroF1 and spellF1 as the official `possession_metrics.py`.
  The difference is 0.0 over 24 cases: the public GT against all 4 baselines, plus 20 synthetic two-half sequences.
- Differences that remain:
  - ours has no X, while theirs forward-fills X;
  - ours pools counts over several matches, while theirs scores one match;
  - the tie order in the greedy matching differs, which only matters for exact IoU ties.
- **Our 4-state labels are not their definition.** Their "controlled" possession and their dead-ball boundaries come
  from Metrica's corrected tracking.

## 5. Data facts that decide the approach
- **Track ids:** *not stable* (46 slots for 22 players, with breaks and swaps). **Team per track: yes.** Player identity:
  none.
- **Frame rate:** 25 fps (PFF was 29.97). **Coordinates:** normalised 0–1, pitch 105.3 × 68 m. The origin and attacking
  direction are not checked yet.
- **Ball:** present on 59 % of graded frames. Whether z exists is unknown.
- **Labelled:** only 5 minutes of this match, which cannot be used for training or tuning. So **all supervised training
  must come from other games** (our PFF set, or public Metrica / other data), followed by **transfer to automatic,
  noisy tracking**.
- **Free priors:** Smart Tagging team phases and set pieces (they are automatic, so allowed), and a homography for
  every frame.

## 6. Open questions and risks (ranked)
1. **Time:** 2 days, the data is not in hand, and the time zone of the deadline is unknown. Has the team (3–5 people)
   enrolled? The output must be produced and reproduced by then.
2. **Domain shift:** PFF is clean, ~30 fps and has ids. Metrica's tracking is automatic, at 25 fps, has no ids and is
   missing 41 % of the ball. Models trained on PFF may transfer badly, and we have no labelled target data to measure
   it.
3. **The definition gap:** Metrica's "controlled possession" and dead-ball timing against our labels v2. It can be
   checked only qualitatively on the 5 public minutes, without tuning on them.
4. **Exact reproducibility:** the repo must regenerate the *same* file. This needs deterministic CPU inference and
   shipped weights, or deterministic retraining. GPU training is not bit-exact.
5. **Licences:**
   - Can PFF data, labels derived from it, and models trained on it be used, and can the organisers access them? A
     public repo must not contain PFF data.
   - PathCRF is MPL-2.0, which is fine when vendored with its licence.
6. **The LLM clause** limits assistant help on the data itself.
7. **Tracklets to players:** PathCRF needs 22 consistent nodes per window. Without re-linking tracklets it has to work
   on per-frame slots.
8. **Unvalidated choices:** the public window cannot choose between versions (one possession is worth about 6 points
   there). Choices must be justified on PFF CV or on principle.

## 7. What will need adapting (listed, NOT done)
1. A loader: Metrica EPTS ATD plus metadata → per-frame arrays (slot, team, x/y in metres, ball). frame_id − 1 → row.
   Use the official `games.py` and `starter/` readers.
2. Frame rate 29.97 → 25 everywhere: smoother parameters, the C3 window features, PathCRF's STEP (6 → 5 for 5 fps), the
   decoder timings.
3. Coordinates: 0–1 corner origin → our centre-origin metres (or the reverse). Detect the attacking side per half.
4. Remove the dependence on player ids: the owner ranker over per-frame candidate slots; team from the slot; optionally
   re-link tracklets into players for PathCRF.
5. A 3-state output: CONTROL/TRAVELLING → team, LOOSE → last team, DEAD → D, ungraded → X; write the CSV and validate it.
6. A dead-ball model for Metrica's definition, plus a Smart Tagging prior (SET PIECES = restarts; the phase team as a
   soft team prior). Handle the 2nd-half pre-kick-off D.
7. Retraining on PFF data degraded to look like Metrica (25 fps resampling, ball drop-out about 41 %, track breaks,
   no ids) so that it transfers. Ball-from-players for the missing ball.
8. A one-command, deterministic `make submission` path, with pinned dependencies, weights hosting, and the technical
   document.
9. HPC scripts: paths, module names (`m1pc` → `pathcrf_adapter`), the new data package format.
