# M1 × PathCRF on the HPC – one submission, whole cycle

Everything runs through Slurm, one job after another, from a single command. Nothing heavy runs on the login node.
Blocked shell tools are never used; sizes and unzipping use Python.

## 0. Layout (already created on the HPC)

```
~/envs/m1                              conda env (rebuildable; created by the first job if missing)
~/projects/m1-possession/code          this code: m1pc/ (ours) + pathcrf/ (vendored) + jobs/ + env/ + run_cycle.sh
~/projects/m1-possession/data          the data package (30 learning matches, folds.json) + c3_fold0/ (C3 predictions)
~/projects/m1-possession/runs          one folder per run + cycle files (CYCLE_SUMMARY.md, results_fold0.*, ablation.*)
~/projects/m1-possession/final         only what the final GP uses (never auto-deleted)
~/logs                                 all Slurm logs: m1_<step>_<jobid>.log / .err
~/scratch                              temporary files; always safe to delete
```

## 1. Upload

In Bitvise, open the **SFTP window** and drag `m1_hpc_bundle.zip` (about 0.27 GB, from `D:\m1_tmp\hpc\`) into the remote folder
`/nfs/slurm/cugp038/scratch` (`~/scratch`).

## 2. Unzip (Python, on the login node – it is light)

```bash
cd ~
python3 -m zipfile -e ~/scratch/m1_hpc_bundle.zip ~/projects/m1-possession/
python3 -c "import os; [print(p) for p in sorted(os.listdir(os.path.expanduser('~/projects/m1-possession')))]"
```

You should see `code`, `data`, `README_HPC.md` (and `runs`, `final` if they already exist). If `python3` is not found, use
`/nfs/slurm/conda/bin/python` instead.

## 3. Run the whole cycle (one command)

```bash
cd ~/projects/m1-possession/code
bash run_cycle.sh --dry-run      # optional: prints the 12 jobs and their order, submits nothing
bash run_cycle.sh
```

It prints every job ID. The chain (each job starts only when the previous one finished OK):

| # | step | what it does | time limit | GPU-hour cap |
|---|---|---|---|---|
| 1 | setup_env | creates `~/envs/m1` (Python 3.12, PyTorch CUDA 12.6, PathCRF requirements); skipped if it already works | 1 h | – |
| 2 | smoke_test | ~200 training steps; time per step, GPU memory, % windows skipped, estimate per epoch / fold | 30 min | – |
| 3 | train_fold 0 | main run, fast config, early stopping, auto-resume | 8.5 h | 8 h |
| 4 | predict_fold | per-frame outputs for fold 0's 6 held-out matches | 1.5 h | – |
| 5 | score_fold0 | C3 vs PathCRF vs C3 + PathCRF → `runs/results_fold0.md` | 45 min (CPU) | – |
| 6–9 | abl_ref, abl_a1, abl_a2, abl_a3 | 3 epochs each: fast / --from-scratch / --keep-dead-windows / --stride 1 | 2.25 / 2.25 / 2.25 / 3.25 h | 2 / 2 / 2 / 3 h |
| 10 | abl_a4 | main model with --fp32 inference; writes `runs/ablation.md` | 1.5 h | – |
| 11 | decide_and_train_all | if PathCRF beats C3: one model on all 30 matches → `final/` | 8.75 h | 8 h |
| 12 | summary | `runs/CYCLE_SUMMARY.md` + `runs/space_report.txt` | 15 min (CPU) | – |

Total GPU-hour guard: before each training step, if (GPU hours used so far + that step's cap) > 30 h, the step and all later
training is skipped and the reason is written to `runs/guard_stop.json` and the summary.

## 4. Check progress

```bash
squeue -u $USER                                  # pending / running jobs of the chain
ls -t ~/logs                                     # newest logs first
cat ~/logs/m1_train_fold_<JOBID>.log             # one log (the job ID is in squeue / the run_cycle output)
cat ~/projects/m1-possession/runs/smoke_report.json
cat ~/projects/m1-possession/runs/*/logs/epochs.jsonl      # validation score after every epoch
```

## 5. If a job fails

Slurm cancels every later job automatically (they were submitted with `--kill-on-invalid-dep=yes`). Read the `.err` log, fix the
cause, then restart **from that step** – finished steps are not repeated, and training resumes from its `last.pt`:

```bash
cd ~/projects/m1-possession/code
bash run_cycle.sh --from score_fold0          # any step name from the table above
```

## 6. When it is done – download (Bitvise SFTP window)

- `~/projects/m1-possession/runs/CYCLE_SUMMARY.md` – everything in one page (results, ablation verdicts, decision, GPU hours)
- `~/projects/m1-possession/runs/results_fold0.md` (+ `.json`)
- `~/projects/m1-possession/runs/ablation.md` (+ `.json`)
- `~/projects/m1-possession/runs/<date>_pathcrf_fast_fold0/preds/` – PathCRF fold-0 predictions (6 `.npz`, a few MB), and its `NOTES.md`
- `~/projects/m1-possession/final/` – only if the all-matches model was trained

On the laptop, re-score the downloaded predictions with our normal pipeline:
`python scripts/m1/hpc_rescore.py --preds <downloaded preds folder> --results <downloaded results_fold0.json>`.

## 7. What can I delete?

| folder | delete? |
|---|---|
| `~/scratch/*` | always safe (zip, smoke-test run, pip / conda caches) |
| `~/logs/*` | safe once the cycle is summarised |
| `runs/*_ref3ep_fold0`, `*_scratch3ep_fold0`, `*_keepdead3ep_fold0`, `*_stride1s3ep_fold0` | yes, once `ablation.md` is written (their NOTES.md say "delete once scored") |
| `runs/*_fast_fold0/checkpoints/last.pt` | yes after the cycle (`best.pt` is the model) |
| `runs/*_fast_fold0` (whole) | after you downloaded `preds/` and `NOTES.md` |
| `runs/*_final_all` | after it was promoted to `final/` |
| `~/envs/m1` | yes if space is short (step 1 rebuilds it) |
| `~/projects/m1-possession/data` | only after the GP is finished (the zip in `~/scratch` can recreate it) |
| `~/projects/m1-possession/final` | **never** (only by hand) |

Space check (pure Python): `bash ~/projects/m1-possession/code/jobs/space_report.sh`

## 8. Flags (all on `python -m m1pc.train`; default = fast config)

| item | default | switch back |
|---|---|---|
| fine-tune from PathCRF's published weights (model 140) | on | `--from-scratch` |
| skip training windows with no scored live frame | on | `--keep-dead-windows` |
| window stride | 3 s | `--stride 1` |
| bf16 autocast on the A100 | on | `--fp32` |
| epochs / early stopping | 10 / patience 2 | `--max-epochs N --patience P` |
| ball input (our smoothed ball + seen flag) | on | `--no-ball` |
| travelling-vs-loose head loss weight | 0.3 | `--tl-weight W` |
