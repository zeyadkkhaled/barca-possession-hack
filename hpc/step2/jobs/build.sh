#!/bin/bash
# a. Data (CPU work on the GPU slice; keep it short): checks the env (adds lightgbm / scipy if missing), builds the 30
# degraded + relinked learning matches (built/), the 22 eval matches (built_eval/, lanes only, never scored) if
# eval_data/ was uploaded, and C3-T fold 0 (trained on folds 1-4; P(D) for every match).
source "$HOME/projects/barca-step2/code/jobs/common.sh"
step_begin build
python -c "import scipy, lightgbm, torch_geometric" 2>/dev/null || python -m pip install --no-cache-dir "lightgbm==4.7.0" scipy
python -c "import scipy, lightgbm, torch_geometric; print('env OK: lightgbm', lightgbm.__version__, '| scipy', scipy.__version__)"
W="${SLURM_CPUS_PER_TASK:-5}"; W=$((W > 1 ? W - 1 : 1))
[ -f "$BUILT/build_stats.json" ] || python -m pathcrf_adapter.build --package "$PKG" --out "$BUILT" --workers "$W"
if [ -d "$EVAL_PKG/matches" ]; then
    [ -f "$BUILT_EVAL/build_stats.json" ] || python -m pathcrf_adapter.build --eval-package "$EVAL_PKG" --out "$BUILT_EVAL" --workers "$W"
else
    echo "WARNING: $EVAL_PKG/matches missing (m1_hpc_v2_eval.zip not unzipped): the predict_eval step will have nothing to do"
fi
C3T="$(CY run-dir c3t)"
[ -f "$C3T/decoder.json" ] || python -m pathcrf_adapter.c3t_fold --data "$BUILT" --fold 0 --out "$C3T" --threads "${SLURM_CPUS_PER_TASK:-5}"
notes "$BUILT" "Built by the build step: 30 learning matches, degraded (step-1c settings) + duplicate rule + ball fill + trim-to-11 + Hungarian re-linking. Stats: build_stats.json."
