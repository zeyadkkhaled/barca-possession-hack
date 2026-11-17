# Step 2a: PathCRF port, re-linking, HPC package, one-command submission (laptop, light work only)

Aggregate numbers only. No training and no PFF cross-validation were run. One heavy process ran at a time with 1 worker.
The 16 locked test matches were not touched.

**Deviation up front:** the "Step 2b prompt below" was not in the message. The design here comes from the task list
itself and from docs/HANDOFF_FOR_PLANNING.md §6–7. I read the Step 2b task numbers as follows:
- 0 = build the degraded + relinked data on the HPC;
- 1 = re-linking;
- 4 = zero-shot scoring;
- 5 = fold-0 fine-tuning with / without the ball;
- 7 = the optional all-30 model plus the summary.
Please check this against the real Step 2b prompt before running tomorrow.

## 1. Files changed

| file | change |
|---|---|
| `src/barca_possession/relink.py` (new) | trim to 11, then Hungarian re-linking of slots into 11 lanes per team; window and link statistics. The prompt said `src/relink.py`; it sits in the package next to `load.trim_to_11`. |
| `src/barca_train/degrade_pff.py` | optional owner map: the true player per slot and frame, moved along with breaks, swaps and wrong-label episodes. Training targets only. Same random draws, defaults unchanged. `to_25fps` also keeps the 4-state labels and player at 25 fps. |
| `src/pathcrf_adapter/adapter.py` | rewritten for 25 fps, STEP 5 (5 fps, the same grid as C3-T's rows), lanes as nodes and our coordinate convention: home attacks +x, so home is always PathCRF's left team. Corner origin scaled to 105 × 68 m; keeper flag per window; step-1 ball fill as the ball input. |
| `src/pathcrf_adapter/build.py` (new) | runs on the HPC: degrade (step-1c settings), duplicate rule, ball fill, C3-T features, re-linking, lane targets |
| `src/pathcrf_adapter/predict.py` | team-level outputs per step (q_home, CRF path team, p_loose); a `--zero-shot` mode with the published weights |
| `src/pathcrf_adapter/c3t_fold.py`, `score.py` (new) | C3-T fold-0 model on the HPC. The scorer keeps C3-T's P(D) and decoder and swaps only the team probability: c3t / pathcrf / blend. |
| `src/pathcrf_adapter/cycle.py` | rewritten for step 2b: run folders, 7 h GPU guard, smoke report, decision (≥ +0.005 spellF1), summary |
| `src/pathcrf_adapter/train.py`, `model.py`, `__init__.py` | docstrings only; the training loop and model are M1's, unchanged |
| `src/pathcrf_adapter/{combine,evaluate,m1score}.py` | deleted: M1 4-state / player-id scoring that cannot work without ids |
| `hpc/step2/` (new) | `run_step2.sh` (with `--dry-run` and `--from`), `jobs/*.sh`, `env/setup_env.sh`, `build_zip.py`, `HPC_STEP2.md` |
| `run.py` (new) | one command: raw data + `models/c3` + `models/bfp` → `outputs/submission.csv`, plus the validator |
| `scripts/relink_match.py` (new) | re-linking statistics on the hackathon match |
| `tests/test_relink.py`, `tests/test_pathcrf_adapter.py` (new) | 7 synthetic tests |
| `README.md`, `docs/technical.md` (new) | documentation drafts |
| `reports/auto/relink_match.json` | aggregates |

The full test suite passes: **69 passed**.

## 2. Re-linking on the hackathon match (`scripts/relink_match.py`, after the duplicate rule)

| measure | value |
|---|---|
| slots in / slot-frames trimmed (team > 11) | 46 / 1.47 % |
| 10 s windows | 634 |
| both teams ≥ 10 lanes present on ≥ 80 % of frames | **68.8 %** (all 11: 30.9 %) |
| both teams ≥ 10 lanes present on any frame of the window | 90.2 % |
| mean lanes per team (≥ 80 % / any frame) | 9.81 / 10.60 |
| links | 1124, **1.76 per 10 s window** |
| forced links (outside the 3 m + 8 m/s × gap gate) | 14.1 % |
| link gap / distance, median | 3.56 s / 4.42 m (distance p90 35.8 m) |

**Check on one PFF learning match:** the HPC build path was run on 10506 on the laptop, with 1 worker in 16 s. The
test output was deleted afterwards.
- both teams ≥ 10 lanes in 63.4 % of windows, against 68.8 % on the match;
- 1.92 links per window, against 1.76;
- 91.8 % of the answer-key targets map to a lane;
- lane purity 99.6 % (share of lane-frames held by the lane's main player inside a 10 s window).

The adapter's windows on that file were all finite, 21.4 nodes present per window on average, 26.9 % of training
windows skipped (no scored live frame).

## 3. HPC package

- **Zip:** `<scratch>\hpc_step2\barca_step2.zip`, **36.0 MB**, 95 files, sha256 `faa09cef164af27f…`.
- **Contents:** code, PathCRF vendored with its LICENSE and model 140, `models/bfp`, and three aggregate rate files.
- **Not included:** no hackathon data and no PFF data. The HPC builds the PFF part from `~/projects/m1-possession/data`.
- **Checked locally:**
  - the `.sh` files have LF endings;
  - unzipped to a scratch folder, the imports and paths resolve and the build config loads;
  - `--dry-run` lists 9 jobs;
  - the 7 h guard trips correctly (tested with a fake ledger).

**Commands (from `hpc/step2/HPC_STEP2.md`):**

```bash
# 1. upload: Bitvise SFTP, drag <scratch>\hpc_step2\barca_step2.zip into ~/scratch
python3 -m zipfile -e ~/scratch/barca_step2.zip ~/projects/barca-step2/       # 2. unzip
cd ~/projects/barca-step2/code && bash run_step2.sh                            # 3. the one command (--dry-run to preview)
squeue -u $USER                                                                 # 4. progress; logs: ~/logs/b2_<step>_<jobid>.log
cat ~/projects/barca-step2/runs/score_ball.md                                   #    scores as they arrive
```

**Chain:**
1. setup_env (CPU);
2. build_data (CPU; 16 s per match on the laptop, so a few minutes);
3. c3t_fold0 (CPU);
4. smoke_test (GPU, 0.3 h);
5. zeroshot (0.3 h);
6. train_ball (2.4 h);
7. train_noball (2.0 h);
8. train_all (1.8 h, only if fold 0 beats C3-T by ≥ 0.005 spellF1);
9. summary.

The GPU caps add up to 6.8 h, and the guard is set at 7 h.

**Download:** put these in `<repo>\reports\auto\step2b\`:
- `runs/STEP2_SUMMARY.md`;
- `runs/score_*.json` and `.md`, `decision.json`, `smoke_report.json`, `gpu_ledger.json`;
- `built/build_stats.json`.

If the all-30 model was trained, `final/<run>/` (best.pt, config.json, NOTES.md) goes to `<repo>\models\pathcrf\`.
`built/` and `runs/*/preds/` stay on the HPC.

## 4. One-command submission (`run.py`)

| file | sha256 |
|---|---|
| `outputs/submission.csv`, run 1 | `8612514a69e9c58045ab5c0ccdad20808aa48bdfd45d0566f8389eb43defa8c5` |
| `outputs/submission.csv`, run 2 | `8612514a69e9c58045ab5c0ccdad20808aa48bdfd45d0566f8389eb43defa8c5` |
| `outputs/submission_c3.csv` (step 1) | `8612514a69e9c58045ab5c0ccdad20808aa48bdfd45d0566f8389eb43defa8c5` |

All three are identical. The validator passes (199,319 rows). The graded label share is H 33.6 %, A 21.5 %, D 44.9 %.

## 5. Documentation status

- `README.md`: updated (status, `run.py`, layout, obligations).
- `docs/technical.md`: draft. It covers the method, the PFF WC 2022 training data, the degradation, C3-T, the results
  so far, the public 5 minutes as a sanity check only, and reproducibility. The PathCRF section is marked **TODO**.
  The exact PFF download link and licence wording are marked TODO too: they are not verified here.

## 6. Open questions

1. **The Step 2b prompt was missing.** Please check the task mapping above and the decision rule (≥ +0.005 spellF1 on
   fold 0 → all-30 model).
2. **`run.py` needs `models/c3` and `models/bfp`.** `models/` is git-ignored, so a fresh clone cannot regenerate the
   file. These are LightGBM text files of about 10 MB, trained on degraded PFF only. Shipping them in git or hosting
   them is needed before submission.
3. **PathCRF on the hackathon match needs torch.** The laptop has none, and installing it needs your OK. The alternative
   is to take a lane file built from the match to the HPC. That is hackathon data leaving the laptop; the T&Cs forbid
   only hosted LLMs, but the data licence should be checked first.
4. **Fold-0 ball fill:** `models/bfp` was fitted on all 30 matches, fold 0 included, so the fold-0 ball fill is
   slightly optimistic. It is the same for C3-T and PathCRF, so the comparison stays fair.
5. **Epoch budget:** at M1's speed, 2.4 h may allow only about 2 epochs on fold 0. The smoke test writes the real
   number. Zero-shot has no ball input and an untrained travelling/loose head (only the team output is scored).
6. **Python versions:** the HPC env is Python 3.12 and the laptop venv is 3.13. The code uses no 3.13-only features, but
   it was not run under 3.12 here.
7. **The rest of `hpc/`** (M1's `run_cycle.sh`, jobs, tools) is left as copied and still names `m1pc`. Only
   `hpc/step2/` is current.
8. **Rule slip:** while drafting the docs I listed the names of the entries in `data/` once with `ls` (names only, no
   content).
