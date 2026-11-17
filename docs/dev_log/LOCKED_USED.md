# Locked test matches: every use

The 16 locked PFF test matches (`barca_train.stress.LOCKED_TEST`) were held out from every training run, CV and model
choice. Each time they are read is logged here.

| date (Cairo) | what | model | script | result |
|---|---|---|---|---|
| 2026-11-17 22:41 | the one final test of the submitted model, degraded (seed 0) then clean | frozen C3-T (`models/c3`, `models/bfp`), unchanged | `scripts/locked_test.py --workers 2` | `reports/auto/locked_test.json` |
