# models/ (weights git-ignored)

No weights are committed. This lists where the trained models are.

## C3 (LightGBM, two layers + stacking + per-transition HMM/Viterbi), trained on PFF World Cup 2022

**C3 has no saved model files.** The M1 scripts (`scripts/c3/c3.py` → `c3v2.py` → `c3v3.py` → `c3v4.py`; copied here)
retrain the LightGBM layers inside each run and save only predictions. The "frozen C3" is a set of settings plus code,
not a file. It has to be retrained (on PFF) before it can predict anything new. So nothing was copied.
- Frozen C3 settings and results: M1 `data/processed/m1_possession/c3v3/results_final.json` (per-fold decoder offsets) and
  `c3v4/results_*.json`, under `Z:\GP\Task 2 - match sim\`.
- Held-out C3 predictions (PFF frames, 5 folds): M1 `data/processed/m1_possession/c3v4/pred/base_pt/k<fold>_<match>.npz`.
  These are PFF-derived, so they are not copied.

## PathCRF

- **Published model 140** (Set Attention + dynamic sparse CRF, 3.43 M parameters): `vendor/pathcrf/saved/140/model`
  (37 MB). It is present locally but git-ignored; upstream is <https://github.com/hyunsungkim-ds/pathcrf> at commit
  `1b7c6f8`.
- **M1 fine-tuned (fold 0)**, and the **final model trained on all 30 learning matches** (about 5 of 20 planned epochs),
  are on the HPC: `~/projects/m1-possession/final/2026-11-11_pathcrf_final_all`. Both were fine-tuned from model 140
  through `pathcrf_adapter` (M1 `m1pc`) at 5 fps on PFF (29.97 fps / 6). They also have a travelling-vs-loose head.

## Not models, but needed to retrain

- The M1 HPC data package (`m1_hpc_v1`: 30 learning matches as `.npz`, plus `folds.json`) and bundle:
  `D:\m1_tmp\hpc\m1_hpc_v1`, `D:\m1_tmp\hpc\m1_hpc_bundle.zip`. These are PFF-derived and are never committed.
