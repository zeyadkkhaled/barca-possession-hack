# Step 2b on the HPC (MOHESR): upload, one command, check, download

**The combo:** C3-T layer 1 decides DEAD, and PathCRF decides the team.
- **Fold 0:** this chain fine-tunes M1's PathCRF on degraded + relinked PFF and scores it on fold 0 against C3-T.
- **Final model:** it trains the final model on all 30 matches.
- **Eval matches:** it predicts the 22 stress / locked matches. It does not score them.

**What stays off the HPC:**
- **Hackathon data never goes to the HPC.** PathCRF runs on the hackathon match on the laptop CPU (`run.py --pathcrf ...`).
- **M1's folders are read only:** `~/projects/m1-possession/{data,runs,final}` are read and never written.

**Facts this setup relies on:**
- Every job gets 1 MIG slice (1g.20gb, about 1/7 of an A100), 5 CPUs, 32 GB and `--gres=gpu:1`.
- Every job is charged GPU time, CPU work included, so CPU steps are kept short.
- `head` and `df` are blocked; everything here uses Python.

## 0. Layout

```
~/projects/barca-step2/code        barca_step2.zip: src/, vendor/pathcrf/, models/bfp/, reports/auto/, jobs/, run_step2.sh, status_step2.sh
~/projects/barca-step2/eval_data   m1_hpc_v2_eval.zip: 6 stress + 16 locked PFF matches (predictions only)
~/projects/barca-step2/built       job build: 30 learning matches, degraded + relinked (pathcrf/, c3t/, folds.json, build_stats.json)
~/projects/barca-step2/built_eval  job build: the 22 eval matches (lanes only)
~/projects/barca-step2/runs        run folders, score_*.json/.md, steps/ (markers), gpu_ledger.json, STEP2_SUMMARY.md
~/projects/barca-step2/final       the final model (copied there by job final)
~/projects/m1-possession/...       M1: data (30 matches), runs/2026-11-11_pathcrf_fast_fold0, final/2026-11-11_pathcrf_final_all (READ ONLY)
~/logs                             b2_<step>_<jobid>.log / .err
```

## 1. Before you start

```bash
myquota                                          # monthly quota 200 GPU h; this chain needs at most 10
```

## 2. Upload (Bitvise SFTP window)

Drag both files from `D:\m1_tmp\hpc_step2\` into the remote folder `~/scratch` (`/nfs/slurm/cugp038/scratch`):

| file | size | what |
|---|---|---|
| `barca_step2.zip` | ~36 MB | code, PathCRF (vendored with its licence and published weights), ball-fill model, 3 aggregate rate files |
| `m1_hpc_v2_eval.zip` | ~194 MB | the 22 stress + locked PFF matches (PFF data only, no hackathon data) |

The smoke test has **not** been run on the laptop (training there crashes the machine). Job 2 on the HPC is its first
real run: watch it (section 5) before you go to sleep. It takes about 10–20 min after the build.

## 3. Unzip (login node; light)

```bash
python3 -m zipfile -e ~/scratch/barca_step2.zip ~/projects/barca-step2/
python3 -m zipfile -e ~/scratch/m1_hpc_v2_eval.zip ~/projects/barca-step2/
python3 -c "import os; r=os.path.expanduser('~/projects/barca-step2'); print(sorted(os.listdir(r))); print(len(os.listdir(r+'/eval_data/matches')), 'eval matches')"
```

You should see `['HPC_STEP2.md', 'code', 'eval_data']` and `22 eval matches`. If `python3` is not found, use
`/nfs/slurm/conda/bin/python` instead.

## 4. Dry run, then the one command

```bash
cd ~/projects/barca-step2/code && bash run_step2.sh --dry-run
cd ~/projects/barca-step2/code && bash run_step2.sh
```

The dry run prints the 8 jobs, their dependencies and GPU caps (sum 9.7 GPU h; guard 10 h), and submits nothing.

| # | step | what | after | cap (GPU h) |
|---|---|---|---|---|
| 1 | build | env check (+ lightgbm / scipy if missing); degraded + relinked data for the 30 learning matches and the 22 eval matches; C3-T fold 0 | – | 0.4 |
| 2 | smoke | 300 steps on 2 matches, ball and no-ball, from M1's fold-0 best.pt: checkpoint loads, loss falls, s/step, peak GPU memory, minutes per epoch → `runs/smoke_report.json`. **If unhealthy, it fails and nothing long runs.** | 1 ok | 0.4 |
| 3 | zeroshot | fold 0's 6 matches: M1 fold-0 best.pt and the published weights, each combined with C3-T's DEAD, next to C3-T | 2 ok | 0.4 |
| 4 | ft_ball | fine-tune from M1 fold-0 best.pt on fold 0's training matches, ball input: dead windows kept, stride 3 s, ≤ 6 epochs, patience 2; then predict + score fold 0 | 2 ok | 2.5 |
| 5 | ft_noball | the same with `--no-ball` (runs **in parallel** with 3 and 4) | 2 ok | 2.5 |
| 6 | final | picks the variant with the higher fold-0 spellF1, fine-tunes from M1's final_all on all 30 matches, same settings, copies it to `final/` | 4 and 5 ended (afterany) | 3.0 |
| 7 | predict_eval | final model → predictions (no scores) for the 6 stress + 16 locked matches | 6 ok | 0.4 |
| 8 | summary | `runs/STEP2_SUMMARY.md`: every score, step states, GPU hours per job, chosen variant | always (afterany) | 0.1 |

**GPU guard:** before a job starts, its cap is checked against what has been used and what running jobs have reserved.
If used + reserved + cap would pass 10 h, the job ends cleanly, is marked `skipped`, and the reason goes into
`runs/guard_stop.json` and the summary. Training stops itself before its cap (2.2 h fold 0, 2.75 h final) and keeps
`best.pt`.

## 5. Check while it runs

```bash
bash ~/projects/barca-step2/code/status_step2.sh
```

One screen:
- the Slurm queue;
- each step (not started / running / done / failed / skipped);
- the last 3 epochs of every training run;
- the zero-shot and fold-0 scores so far;
- GPU hours used vs the 10 h guard;
- `myquota`;
- the last 20 lines of the newest log.

Other checks:

```bash
squeue -u $USER                                                   # the jobs and their states (PD pending, R running)
python3 -c "import sys; print(open(sys.argv[1]).read()[-3000:])" ~/logs/b2_ft_ball_<JOBID>.log   # the end of one log
cat ~/projects/barca-step2/runs/*_pathcrf_ball_fold0/logs/epochs.jsonl
```

**A healthy epoch line** in `~/logs/b2_ft_*.log` looks like this:

```
epoch 2: seconds 1210.4, train_edge_acc 0.62, ..., val_loss 6.81, val_edge_acc 0.60, ..., val_spell 0.70, val_macro 0.92, ..., minutes 22.9
```

- `train_loss` and `val_loss` fall over the first epochs;
- `val_spell` rises or holds;
- `minutes` is about 15–30 per epoch on fold 0.

In between, `epoch E batch B/N loss ... (x.xxx s/step)` lines print every 200 batches.

**If a job fails:**
1. `status_step2.sh` shows the step as `failed`.
2. Read its error log, `~/logs/b2_<step>_<JOBID>.err`, and the end of the `.log`. The job ID is in `squeue`, the
   run_step2 output, or `runs/steps/<step>.failed`.
3. Fix the cause, then restart from that step. Finished steps (`runs/steps/<step>.done`) exit at once, and training
   resumes from `last.pt`:

```bash
cd ~/projects/barca-step2/code && bash run_step2.sh --from <step>       # e.g. --from ft_noball
cd ~/projects/barca-step2/code && bash run_step2.sh --only zeroshot     # one step on its own, no dependencies
```

**What happens when a step fails:**
- A failed `smoke` cancels everything after it.
- A failed `ft_*` still lets `final` run on the other variant's score.
- `summary` always runs.

## 6. In the morning: download (Bitvise SFTP) and where it goes on the laptop

| on the HPC (`~/projects/barca-step2/...`) | on the laptop | what |
|---|---|---|
| `runs/STEP2_SUMMARY.md`, `runs/score_*.json` + `.md`, `runs/decision.json`, `runs/smoke_report.json`, `runs/gpu_ledger.json`, `built/build_stats.json` | `D:\barca_hack\reports\auto\step2b\` | aggregates only |
| `final/<date>_pathcrf_final_all/` (`best.pt`, `config.json`, `NOTES.md`) | `D:\barca_hack\models\pathcrf\` | the model for the submission (git-ignored) |
| `runs/<date>_pathcrf_final_all/preds_eval/` (22 `.npz`) | `D:\m1_tmp\hpc_step2\preds_eval\` | stress + locked predictions, scored later on the laptop |
| `runs/<date>_pathcrf_{ball,noball}_fold0/NOTES.md` | `D:\barca_hack\reports\auto\step2b\` | per-run settings, GPU hours, scores |

Do **not** download `built/`, `built_eval/` or `eval_data/` (PFF data). Then, on the laptop:

```powershell
.\.venv\Scripts\python.exe run.py --pathcrf models\pathcrf\best.pt --out outputs\submission_combo.csv
```

## 7. Afterwards

- **Safe to delete:** `~/scratch/*`, `runs/*_smoke_*`, and `runs/*/checkpoints/last.pt`.
- **Keep:** `final/`.
- **Never touch:** `~/projects/m1-possession`.
