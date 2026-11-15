# Step 2b on the HPC: upload, one command, download

Everything runs through Slurm, one job after another, from one command. Nothing heavy runs on the login node. The
PFF data is **not** in the zip: the first CPU job builds the degraded and relinked training data on the HPC from the M1
package that is already there (`~/projects/m1-possession/data`). The zip has no hackathon data, only three aggregate
rate files (`reports/auto/*.json`).

## 0. Layout

```
~/projects/barca-step2/code      this zip's code/ (src/, vendor/pathcrf/, models/bfp/, reports/auto/, jobs/, env/, run_step2.sh)
~/projects/barca-step2/built     built by job 2: pathcrf/<match>.npz, c3t/<match>.npz, folds.json, build_stats.json
~/projects/barca-step2/runs      run folders, score_*.json / .md, decision.json, gpu_ledger.json, STEP2_SUMMARY.md
~/projects/barca-step2/final     the all-30 PathCRF model (only if it was trained)
~/projects/m1-possession/data    the existing M1 PFF package (read only, unchanged)
~/envs/m1                        the existing M1 conda env (job 1 adds lightgbm / scipy if missing)
~/logs                           Slurm logs: b2_<step>_<jobid>.log / .err
```

## 1. Upload (Bitvise SFTP window)

Drag `D:\m1_tmp\hpc_step2\barca_step2.zip` into the remote folder `~/scratch` (`/nfs/slurm/cugp038/scratch`).

## 2. Unzip (Python on the login node; light)

```bash
python3 -m zipfile -e ~/scratch/barca_step2.zip ~/projects/barca-step2/
```

If `python3` is not found, use `/nfs/slurm/conda/bin/python` instead.

## 3. The one command

```bash
cd ~/projects/barca-step2/code && bash run_step2.sh
```

Optional, before it: `bash run_step2.sh --dry-run` lists the 9 jobs with their sbatch options and submits nothing.

| # | step | what | node | time limit | GPU cap |
|---|---|---|---|---|---|
| 1 | setup_env | checks `~/envs/m1`; adds lightgbm 4.7.0 and scipy if missing | CPU | 1 h | – |
| 2 | build_data | 30 PFF matches → degraded (step-1c settings: dead_weight 8, extra slots) → duplicate rule → ball fill → C3-T features + trim-to-11 + Hungarian re-linking → lane files and targets | CPU, 8 cores | 4 h | – |
| 3 | c3t_fold0 | C3-T trained on folds 1–4, predictions for fold 0 | CPU, 8 cores | 2 h | – |
| 4 | smoke_test | 200 fine-tuning steps; time per step, memory, max epochs that fit | GPU | 40 min | 0.3 h |
| 5 | zeroshot | published PathCRF model 140 (no ball, no fine-tuning) on fold 0 → `score_zeroshot` | GPU | 40 min | 0.3 h |
| 6 | train_ball | fine-tune on folds 1–4 with the ball input, predict and score fold 0 → `score_ball` | GPU | 3 h | 2.4 h |
| 7 | train_noball | the same with `--no-ball` → `score_noball` | GPU | 2 h 40 | 2.0 h |
| 8 | train_all | only if a fine-tuned variant beats C3-T by ≥ 0.005 spellF1 on fold 0: all 30 matches → `final/` | GPU | 2 h 20 | 1.8 h |
| 9 | summary | `runs/STEP2_SUMMARY.md` and `runs/space_report.txt` | CPU | 15 min | – |

**GPU budget:** a guard stops every GPU step once the GPU hours already used (`runs/gpu_ledger.json`, wall time of the GPU
jobs) plus that step's cap would pass **7 h**. The caps add up to 6.8 h. The reason is written to `runs/guard_stop.json`
and to the summary. Training stops cleanly at its cap and keeps `best.pt`.

## 4. Check progress

```bash
squeue -u $USER                                         # the chain: pending / running
ls -t ~/logs | python3 -c "import sys; [print(l, end='') for l in list(sys.stdin)[:10]]"   # newest logs
cat ~/logs/b2_<step>_<JOBID>.log                        # one job's log (job IDs: run_step2.sh output or squeue)
cat ~/projects/barca-step2/built/build_stats.json       # after job 2: relink and target stats (aggregates)
cat ~/projects/barca-step2/runs/smoke_report.json       # after job 4
cat ~/projects/barca-step2/runs/score_zeroshot.md       # after job 5; score_ball.md after 6, score_noball.md after 7
cat ~/projects/barca-step2/runs/*/logs/epochs.jsonl     # validation after every epoch
```

## 5. If a job fails

Slurm cancels every later job (`--kill-on-invalid-dep=yes`). Read the `.err` log, fix the cause, then restart from that
step. Finished steps are skipped by their own checks, and training resumes from `last.pt`:

```bash
cd ~/projects/barca-step2/code && bash run_step2.sh --from train_ball      # any step name from the table
```

## 6. Download (Bitvise SFTP window) and where it goes on the laptop

| on the HPC | on the laptop | what |
|---|---|---|
| `~/projects/barca-step2/runs/STEP2_SUMMARY.md` | `D:\barca_hack\reports\auto\step2b\` | everything on one page |
| `~/projects/barca-step2/runs/score_*.json`, `score_*.md`, `decision.json`, `smoke_report.json`, `gpu_ledger.json` | `D:\barca_hack\reports\auto\step2b\` | aggregate scores only |
| `~/projects/barca-step2/built/build_stats.json` | `D:\barca_hack\reports\auto\step2b\` | relink / target aggregates |
| `~/projects/barca-step2/final/<date>_pathcrf_final_all/` (`best.pt`, `config.json`, `NOTES.md`) | `D:\barca_hack\models\pathcrf\` | only if it was trained; git-ignored |

Do **not** download `built/` (the degraded PFF data) or `runs/*/preds/` (per-frame predictions). They stay on the HPC.

## 7. What can be deleted afterwards

| folder | delete? |
|---|---|
| `~/scratch/*` | always safe |
| `~/projects/barca-step2/runs/*_smoke` | yes |
| `runs/*_fold0/checkpoints/last.pt` | yes (`best.pt` is the model) |
| `~/projects/barca-step2/built` | after the cycle (job 2 rebuilds it in minutes) |
| `~/projects/barca-step2/final` | **never** (only by hand) |
