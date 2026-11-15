#!/bin/bash
# GPU job (optional). Trains the all-30 model only if a fine-tuned fold-0 variant beats C3-T by >= 0.005 spellF1
# (runs/decision.json), with that variant's ball flag, for the fold-0 run's best epoch count, capped; copies it to final/.
source "$HOME/projects/barca-step2/code/jobs/common.sh"
activate_env
ledger_on train_all 1
banner train_all
cd "$CODE"
if [ "$(python -m pathcrf_adapter.cycle decide | tail -n 1)" != "TRAIN_ALL" ]; then
    echo "no all-30 model (runs/decision.json)"; exit 0
fi
guard_or_exit train_all all
RUN="$(python -m pathcrf_adapter.cycle run-dir all)"
EP="$(python -m pathcrf_adapter.cycle decision max_epochs)"
FLAGS="$(python -m pathcrf_adapter.cycle decision flags)"
TRAIN_H="$(python -c "import sys; print(max(0.2, float(sys.argv[1]) - 0.2))" "$(python -m pathcrf_adapter.cycle cap all)")"
# shellcheck disable=SC2086
python -m pathcrf_adapter.train --data "$BUILT" --run-dir "$RUN" --all --max-epochs "$EP" --gpu-hours "$TRAIN_H" $FLAGS
python -m pathcrf_adapter.cycle promote "$RUN"
