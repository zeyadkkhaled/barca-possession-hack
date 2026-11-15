# barca-possession-hack

Our entry for **More than a Hack 2027, Phase 1: ball possession** (Metrica Sports × FC Barcelona × MWCapital). The task
is one label per frame of FC Barcelona vs Athletic Club: H (home), A (away), D (dead ball) or X (no answer). It is ranked
on possession-spell F1. Rules summary: [docs/HACKATHON_RULES.md](docs/HACKATHON_RULES.md). Official repo:
<https://github.com/metrica-sports/more-than-a-hack-2027>.

## Status

**These are components copied from our M1 project (PFF World Cup 2022 possession models). They are not adapted to the
hackathon yet.** Nothing here reads the hackathon data. The copied code still assumes PFF conventions: 29.97 fps,
centre-origin metres, PFF player ids, and M1 data paths. What needs changing is listed in
[docs/HANDOFF_FOR_PLANNING.md](docs/HANDOFF_FOR_PLANNING.md). Where each file came from:
[docs/PROVENANCE.md](docs/PROVENANCE.md).

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
src/pathcrf_adapter/              M1 PathCRF adapter (m1pc): data adapter, multi-task head, train / predict / combine / cycle
vendor/pathcrf/                   PathCRF (MPL-2.0), vendored unchanged
scripts/c3/                       M1 C3 training / evaluation pipelines (round 1-5, incl. the restart detector in c3v4.py)
scripts/check_scorer_parity.py    our scorer vs the official one on the public GT + baselines + synthetic cases
hpc/                              M1 HPC bundle: Slurm jobs, run_cycle.sh, env/setup_env.sh, tools/ (bundle + package export)
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

PathCRF (torch) runs on the HPC only; see `hpc/env/setup_env.sh`.

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
  command. It must have pinned dependencies and install instructions. No such command exists yet.
- **It must be public at submission time.** It is private for now. Before making it public, check that no data, derived
  frame-level exports, PFF data or unlicensed weights are in the history. `.gitignore` excludes `data/`, `*.npz`,
  `*.parquet`, `*.pt`, `models/*` and video.
- Match data must never be committed, and no extract of it may go to a hosted LLM (competition T&Cs).
- Deadline: **17 November 2026** (no time zone is stated in the official repo).
