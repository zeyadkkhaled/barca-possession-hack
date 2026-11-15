#!/bin/bash
# CPU job. Degraded + relinked PFF for the 30 learning matches (pathcrf_adapter.build), from the existing M1 package:
# built/pathcrf/<match>.npz (lanes, ball, targets), built/c3t/<match>.npz (C3-T features), built/build_stats.json.
# Skipped if built/build_stats.json exists (delete it to rebuild).
source "$HOME/projects/barca-step2/code/jobs/common.sh"
activate_env
ledger_on build_data 0
banner build_data
cd "$CODE"
if [ -f "$BUILT/build_stats.json" ]; then echo "built/build_stats.json exists: skipped"; exit 0; fi
python -m pathcrf_adapter.build --package "$PKG" --out "$BUILT" --workers "${B2_BUILD_WORKERS:-6}"
