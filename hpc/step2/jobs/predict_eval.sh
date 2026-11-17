#!/bin/bash
# e. PREDICTIONS ONLY with the final model on the 6 stress-eval + 16 locked matches (built_eval/: degraded + relinked the
# same way). Nothing is scored here: the chain never reads their labels for a score. -> RUN/preds_eval/<match>.npz.
source "$HOME/projects/barca-step2/code/jobs/common.sh"
step_begin predict_eval
RUN="$(CY run-dir final)"
if [ ! -d "$BUILT_EVAL/pathcrf" ]; then echo "no built_eval/ (eval package not uploaded): nothing to predict"; exit 1; fi
python -m pathcrf_adapter.predict --data "$BUILT_EVAL" --model "$RUN/checkpoints/best.pt" --out "$RUN/preds_eval"
notes "$RUN/preds_eval" "Predictions only (no scores) of the final model on $(ls "$BUILT_EVAL/pathcrf" | wc -l) eval matches (6 stress + 16 locked)."
