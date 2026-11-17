# Provenance of copied files

> On `main`, the PathCRF and HPC files listed below (`vendor/pathcrf/`, `hpc/`, `pathcrf_adapter`) were removed after the
> submission was frozen. They are kept on the `dev-history` branch.

Everything below was **copied** (the originals were not moved or edited) on 2026-11-15.

- **M1 source:** `<M1 repo>` (branch `m1-possession`), commit `7e86074c29e74ea884037a86f6d00ab528188307`. Every copied M1 file was clean at that commit: `git status` showed only the untracked `.pytest-tmp-codex/` and `Ultralytics/`.
- **PathCRF source:** `<scratch>\ext\pathcrf`, a clone of <https://github.com/hyunsungkim-ds/pathcrf>, commit `1b7c6f85a6b78c92e4543fde72ca2b86a933f997`. It is vendored unchanged with its MPL-2.0 `LICENSE`, using the same file list as M1's `build_bundle.py`.
- **Edits:** import lines only. The packages were renamed `m1_possession` → `barca_possession`, `m1_labels` → `barca_labels` and `m1pc` → `pathcrf_adapter`. There were three other edits:
  - `tests/test_m1_labels.py` names the packages as strings in its source-scan test, so those strings were renamed too;
  - `src/pathcrf_adapter/model.py`: the default location of the vendored PathCRF on `sys.path` changed from `HERE.parent / "pathcrf"` to `HERE.parents[1] / "vendor" / "pathcrf"` (the `M1PC_PATHCRF` override still works);
  - file paths, data paths, the `<scratch>` scratch default, PFF match ids, the Slurm job scripts and `python -m m1pc...` calls in shell scripts are **unchanged** (they are still M1's).

| new path | M1 source path | import-line edits |
|---|---|---|
| `src/barca_possession/__init__.py` | `src/m1_possession/__init__.py` | 0 |
| `src/barca_possession/ball_infer.py` | `src/m1_possession/ball_infer.py` | 0 |
| `src/barca_possession/baselines.py` | `src/m1_possession/baselines.py` | 0 |
| `src/barca_possession/c3.py` | `src/m1_possession/c3.py` | 0 |
| `src/barca_possession/c3_extra.py` | `src/m1_possession/c3_extra.py` | 0 |
| `src/barca_possession/c3_post.py` | `src/m1_possession/c3_post.py` | 0 |
| `src/barca_possession/config.py` | `src/m1_possession/config.py` | 0 |
| `src/barca_possession/rules.py` | `src/m1_possession/rules.py` | 0 |
| `src/barca_possession/smoother.py` | `src/m1_possession/smoother.py` | 0 |
| `src/barca_possession/splits.py` | `src/m1_possession/splits.py` | 0 |
| `src/barca_possession/upgrades.py` | `src/m1_possession/upgrades.py` | 0 |
| `src/barca_labels/__init__.py` | `src/m1_labels/__init__.py` | 0 |
| `src/barca_labels/labeller.py` | `src/m1_labels/labeller.py` | 0 |
| `src/barca_labels/scoring.py` | `src/m1_labels/scoring.py` | 0 |
| `src/pathcrf_adapter/__init__.py` | `scripts/m1/hpc_pathcrf/code/m1pc/__init__.py` | 0 |
| `src/pathcrf_adapter/adapter.py` | `scripts/m1/hpc_pathcrf/code/m1pc/adapter.py` | 0 |
| `src/pathcrf_adapter/combine.py` | `scripts/m1/hpc_pathcrf/code/m1pc/combine.py` | 0 |
| `src/pathcrf_adapter/cycle.py` | `scripts/m1/hpc_pathcrf/code/m1pc/cycle.py` | 0 |
| `src/pathcrf_adapter/evaluate.py` | `scripts/m1/hpc_pathcrf/code/m1pc/evaluate.py` | 0 |
| `src/pathcrf_adapter/m1score.py` | `scripts/m1/hpc_pathcrf/code/m1pc/m1score.py` | 0 |
| `src/pathcrf_adapter/model.py` | `scripts/m1/hpc_pathcrf/code/m1pc/model.py` | 1 |
| `src/pathcrf_adapter/predict.py` | `scripts/m1/hpc_pathcrf/code/m1pc/predict.py` | 0 |
| `src/pathcrf_adapter/train.py` | `scripts/m1/hpc_pathcrf/code/m1pc/train.py` | 0 |
| `scripts/c3/c3.py` | `scripts/m1/c3.py` | 10 |
| `scripts/c3/c3v2.py` | `scripts/m1/c3v2.py` | 8 |
| `scripts/c3/c3v3.py` | `scripts/m1/c3v3.py` | 8 |
| `scripts/c3/c3v4.py` | `scripts/m1/c3v4.py` | 6 |
| `scripts/c3/c3v5.py` | `scripts/m1/c3v5.py` | 1 |
| `scripts/c3/compare.py` | `scripts/m1/compare.py` | 6 |
| `scripts/c3/diag_v2.py` | `scripts/m1/diag_v2.py` | 3 |
| `scripts/c3/spells_v3.py` | `scripts/m1/spells_v3.py` | 2 |
| `scripts/c3/c3_error_videos.py` | `scripts/m1/c3_error_videos.py` | 3 |
| `scripts/c3/score_v2.py` | `scripts/m1/score_v2.py` | 5 |
| `scripts/c3/bfp.py` | `scripts/m1/bfp.py` | 4 |
| `scripts/c3/run_baselines.py` | `scripts/m1/run_baselines.py` | 4 |
| `hpc/env/setup_env.sh` | `scripts/m1/hpc_pathcrf/code/env/setup_env.sh` | 0 |
| `hpc/jobs/ablation_check.sh` | `scripts/m1/hpc_pathcrf/code/jobs/ablation_check.sh` | 0 |
| `hpc/jobs/common.sh` | `scripts/m1/hpc_pathcrf/code/jobs/common.sh` | 0 |
| `hpc/jobs/decide_and_train_all.sh` | `scripts/m1/hpc_pathcrf/code/jobs/decide_and_train_all.sh` | 0 |
| `hpc/jobs/predict_fold.sh` | `scripts/m1/hpc_pathcrf/code/jobs/predict_fold.sh` | 0 |
| `hpc/jobs/score_fold0.sh` | `scripts/m1/hpc_pathcrf/code/jobs/score_fold0.sh` | 0 |
| `hpc/jobs/smoke_test.sh` | `scripts/m1/hpc_pathcrf/code/jobs/smoke_test.sh` | 0 |
| `hpc/jobs/space_report.sh` | `scripts/m1/hpc_pathcrf/code/jobs/space_report.sh` | 0 |
| `hpc/jobs/summary.sh` | `scripts/m1/hpc_pathcrf/code/jobs/summary.sh` | 0 |
| `hpc/jobs/train_fold.sh` | `scripts/m1/hpc_pathcrf/code/jobs/train_fold.sh` | 0 |
| `hpc/README_HPC.md` | `scripts/m1/hpc_pathcrf/code/README_HPC.md` | 0 |
| `hpc/run_cycle.sh` | `scripts/m1/hpc_pathcrf/code/run_cycle.sh` | 0 |
| `hpc/tools/build_bundle.py` | `scripts/m1/hpc_pathcrf/build_bundle.py` | 0 |
| `hpc/tools/local_checks.py` | `scripts/m1/hpc_pathcrf/local_checks.py` | 4 |
| `hpc/tools/make_m1score.py` | `scripts/m1/hpc_pathcrf/make_m1score.py` | 2 |
| `hpc/tools/export_hpc.py` | `scripts/m1/export_hpc.py` | 2 |
| `hpc/tools/m1_hpc_loader.py` | `scripts/m1/hpc/m1_hpc_loader.py` | 0 |
| `hpc/tools/README_hpc_package.md` | `scripts/m1/hpc/README.md` | 0 |
| `tests/test_m1_smoother.py` | `tests/test_m1_smoother.py` | 2 |
| `tests/test_m1_rules.py` | `tests/test_m1_rules.py` | 2 |
| `tests/test_m1_baselines.py` | `tests/test_m1_baselines.py` | 3 |
| `tests/test_m1_c3.py` | `tests/test_m1_c3.py` | 4 |
| `tests/test_m1_v2.py` | `tests/test_m1_v2.py` | 4 |
| `tests/test_m1_labels.py` | `tests/test_m1_labels.py` | 8 |
| `vendor/pathcrf/LICENSE` | `<scratch>/ext/pathcrf/LICENSE` | 0 |
| `vendor/pathcrf/README.md` | `<scratch>/ext/pathcrf/README.md` | 0 |
| `vendor/pathcrf/requirements.txt` | `<scratch>/ext/pathcrf/requirements.txt` | 0 |
| `vendor/pathcrf/setup.cfg` | `<scratch>/ext/pathcrf/setup.cfg` | 0 |
| `vendor/pathcrf/dataset.py` | `<scratch>/ext/pathcrf/dataset.py` | 0 |
| `vendor/pathcrf/train.py` | `<scratch>/ext/pathcrf/train.py` | 0 |
| `vendor/pathcrf/inference.py` | `<scratch>/ext/pathcrf/inference.py` | 0 |
| `vendor/pathcrf/evaluate.py` | `<scratch>/ext/pathcrf/evaluate.py` | 0 |
| `vendor/pathcrf/models` | `<scratch>/ext/pathcrf/models` | 0 |
| `vendor/pathcrf/set_transformer` | `<scratch>/ext/pathcrf/set_transformer` | 0 |
| `vendor/pathcrf/datatools` | `<scratch>/ext/pathcrf/datatools` | 0 |
| `vendor/pathcrf/saved/140` | `<scratch>/ext/pathcrf/saved/140` | 0 |

Written for this repo (not copied): `scripts/check_scorer_parity.py`, `pyproject.toml`, `.gitignore`, `README.md`, `docs/*`, `models/README.md`, `data/README.md`, `vendor/pathcrf/VENDORED.txt`.
