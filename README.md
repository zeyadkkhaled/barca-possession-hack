# barca-possession-hack

Our entry for **More than a Hack 2027, Phase 1: ball possession** (Metrica Sports × FC Barcelona × MWCapital). The task
is one label per frame of FC Barcelona vs Athletic Club: H (home), A (away), D (dead ball) or X (no answer). It is ranked
on possession-spell F1. Rules summary: [docs/HACKATHON_RULES.md](docs/HACKATHON_RULES.md). Official repo:
<https://github.com/metrica-sports/more-than-a-hack-2027>.

## Status (2026-11-16)

- **Submitted model:** C3-T, team-level LightGBM plus a 3-state Viterbi, trained on PFF World Cup 2022 degraded to look
  like Metrica's tracking. Method and results: [docs/technical.md](docs/technical.md).
- **Reproduce the submission (one command, no training):**

  ```powershell
  .\.venv\Scripts\python.exe run.py            # -> outputs/submission.csv, then the official validator
  ```

  It reads the hackathon data (`data/` or `MTAH_DATA_DIR`) and the saved models in `models/c3/` and `models/bfp/`. It
  is deterministic: two runs and `outputs/submission_c3.csv` give the same sha256.
- **Step 2 (PathCRF, in progress):**
  - re-linking: `barca_possession.relink`;
  - the adapter at 25 fps: `pathcrf_adapter`;
  - the HPC cycle: [hpc/step2/HPC_STEP2.md](hpc/step2/HPC_STEP2.md).
- Where the M1 code came from: [docs/PROVENANCE.md](docs/PROVENANCE.md).

## Step 1 pipeline (C3-T, written 2026-11-15)

Team-level C3 trained on PFF degraded to look like the Metrica tracking. The commands, in order, are in
[reports/step1_report.md](reports/step1_report.md). `fit_predict.py` regenerates `outputs/submission_c3.csv`
deterministically: LightGBM `deterministic=True`, fixed seed, `--threads 1`.

- `barca_possession.load`: the hackathon loader and noise statistics.
- `barca_train.degrade_pff`: PFF → Metrica form.
- `barca_possession.ballfill`: smoother plus ball-from-players.
- `barca_possession.c3t`: features, two layers, 3-state Viterbi.
- `barca_possession.postrules`: the post-rules.
- `barca_train.pipeline`: scoring and fold training.

Scripts in `scripts/` print aggregate numbers only. Data-derived files go to `outputs/` (git-ignored). Aggregate
summaries go to `reports/auto/`.

## Layout

```
reference/more-than-a-hack-2027/  official repo (git submodule, read-only): scorer, validator, starter code, public 5-min GT
src/barca_possession/             tag-free possession code from M1: ball smoother, C0-C2 baselines/rules, C3 features,
                                  decoder and post-processing, ball-from-players, gap fills, scratch-dir helper
src/barca_labels/                 M1 labels v2 builder (PFF events) and our scorer (score_had = H/A/D macroF1 + spellF1)
src/pathcrf_adapter/              PathCRF for the hackathon: build (degraded + relinked PFF, on the HPC), adapter (25 fps lanes),
                                  model + travelling/loose head, train / predict, C3-T fold model, scorer, cycle bookkeeping
vendor/pathcrf/                   PathCRF (MPL-2.0), vendored unchanged
scripts/c3/                       M1 C3 training / evaluation pipelines (round 1-5, incl. the restart detector in c3v4.py)
scripts/check_scorer_parity.py    our scorer vs the official one on the public GT + baselines + synthetic cases
hpc/step2/                        step-2b HPC cycle: run_step2.sh, jobs/, env/, build_zip.py, HPC_STEP2.md
hpc/                              (the rest) M1's HPC bundle as copied: Slurm jobs, run_cycle.sh, tools/ (package export)
run.py                            one command: raw data + saved models -> outputs/submission.csv
models/README.md                  where the trained models live (no weights in git)
data/                             EMPTY, git-ignored: hackathon data goes here (never committed)
tests/                            the copied M1 unit tests
docs/                             HACKATHON_RULES, PROVENANCE, HANDOFF_FOR_PLANNING
```

## Setup (Windows, PowerShell)

Python 3.13 (the same as the M1 venv). Packages are pinned to M1's versions.

```powershell
git clone --recurse-submodules <this repo> ; cd barca-possession-hack   # or: git submodule update --init
& "C:\Program Files\Python313\python.exe" -m venv .venv
.\.venv\Scripts\python.exe -m pip install --no-cache-dir -e ".[dev,c3]"
.\.venv\Scripts\python.exe -m pytest
```

PathCRF (torch) runs on the HPC only; see [hpc/step2/HPC_STEP2.md](hpc/step2/HPC_STEP2.md). On Windows set `PYTHONUTF8=1`.

## Official scorer

```powershell
# score every CSV in reference/.../evaluation/results against the public 5-minute GT
.\.venv\Scripts\python.exe reference\more-than-a-hack-2027\evaluation\run_evaluation.py --out outputs\scorer
# score our own folder of submissions
.\.venv\Scripts\python.exe reference\more-than-a-hack-2027\evaluation\run_evaluation.py --results submissions --out outputs\scorer
# check a file before submitting it (the same checks the official scorer runs)
.\.venv\Scripts\python.exe reference\more-than-a-hack-2027\evaluation\validate_submission.py submissions\<team>.csv
# our scorer vs theirs
.\.venv\Scripts\python.exe scripts\check_scorer_parity.py
```

The public 5 minutes are only for checking the pipeline. The rules forbid training or tuning on them.

## Hackathon obligations for this repo

- **The repo must regenerate the submission CSV**: exactly the same file, from the provided data, with one documented
  command (`python run.py`). It must have pinned dependencies and install instructions.
  - *Open:* `models/` is git-ignored, so the small LightGBM text models that `run.py` needs (`models/c3`, `models/bfp`,
    about 10 MB, trained on degraded PFF only) are not in git yet. They must be shipped or hosted before submission.
- **It must be public at submission time.** It is private for now. Before making it public, check that no data, derived
  frame-level exports, PFF data or unlicensed weights are in the history. `.gitignore` excludes `data/`, `*.npz`,
  `*.parquet`, `*.pt`, `models/*` and video.
- Match data must never be committed, and no extract of it may go to a hosted LLM (competition T&Cs).
- Deadline: **17 November 2026** (no time zone is stated in the official repo).
