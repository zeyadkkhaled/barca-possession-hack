"""Step-2b cycle bookkeeping, decisions, summary and status (called by the Slurm jobs and status_step2.sh). Stdlib only at
import time (numpy is imported inside the one command that needs it), so the login node's base Python can run it.

HPC layout:
~/projects/barca-step2/{code,built,built_eval,eval_data,runs,final}      this cycle
~/projects/m1-possession/{data,runs,final}                                M1 (READ ONLY: data package, M1 checkpoints)

    python -m pathcrf_adapter.cycle init                       runs/CYCLE.json (run folder names; kept on restarts)
    python -m pathcrf_adapter.cycle run-dir KEY                 smoke_ball / smoke_noball / zeroshot / ft_ball / ft_noball / final / c3t
    python -m pathcrf_adapter.cycle cap STEP                    GPU-hour cap of a step
    python -m pathcrf_adapter.cycle guard STEP                  OK (and reserves the cap) or SKIP: used + reserved + cap > 10 h
    python -m pathcrf_adapter.cycle ledger STEP JOBID SECONDS   adds the job's wall time (all jobs are charged as GPU)
    python -m pathcrf_adapter.cycle mark STEP STATUS [JOBID]    runs/steps/STEP.{done,failed,skipped}
    python -m pathcrf_adapter.cycle is-done STEP                prints yes / no
    python -m pathcrf_adapter.cycle smoke-report                checks both smoke runs; writes runs/smoke_report.json; exit 1 if unhealthy
    python -m pathcrf_adapter.cycle choose-variant              runs/decision.json: ball or noball (higher fold-0 spellF1)
    python -m pathcrf_adapter.cycle decision FIELD
    python -m pathcrf_adapter.cycle promote RUN                 best.pt + config.json + NOTES.md -> final/
    python -m pathcrf_adapter.cycle summary                     runs/STEP2_SUMMARY.md
    python -m pathcrf_adapter.cycle status                      one screen: jobs, steps, epochs, scores, GPU hours, quota, newest log
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

HOME = Path(os.environ.get("B2_HOME", Path.home()))
ROOT = Path(os.environ.get("B2_ROOT", HOME / "projects" / "barca-step2"))
CODE, BUILT, BUILT_EVAL, RUNS, FINAL = ROOT / "code", ROOT / "built", ROOT / "built_eval", ROOT / "runs", ROOT / "final"
STEPS_DIR = RUNS / "steps"
M1 = HOME / "projects" / "m1-possession"
M1_FOLD0 = Path(os.environ.get("B2_M1_FOLD0", M1 / "runs" / "2026-11-11_pathcrf_fast_fold0" / "checkpoints" / "best.pt"))
M1_FINAL = Path(os.environ.get("B2_M1_FINAL", M1 / "final" / "2026-11-11_pathcrf_final_all" / "best.pt"))
LOGS = HOME / "logs"
GPU_TOTAL_CAP = 10.0
# GPU-hour cap per job (every job holds a MIG slice and is charged, CPU work included). Sum 9.7 h <= 10 h.
CAPS = dict(build=0.4, smoke=0.4, zeroshot=0.4, ft_ball=2.5, ft_noball=2.5, final=3.0, predict_eval=0.4, summary=0.1)
TRAIN_HOURS = dict(ft_ball=2.2, ft_noball=2.2, final=2.75)      # training stops cleanly here; the rest of the cap: load, predict, score
STEPS = ("build", "smoke", "zeroshot", "ft_ball", "ft_noball", "final", "predict_eval", "summary")


def jread(p, default=None):
    p = Path(p)
    try:
        return json.loads(p.read_text()) if p.exists() else default
    except (OSError, ValueError):
        return default


def jwrite(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(p) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    os.replace(tmp, p)


def init(date=None):
    c = jread(RUNS / "CYCLE.json")
    if c:
        print(f"cycle {c['date']} already initialised (kept)")
        return c
    d = date or _dt.date.today().isoformat()
    c = dict(date=d, runs=dict(smoke_ball=f"{d}_smoke_ball", smoke_noball=f"{d}_smoke_noball", zeroshot=f"{d}_zeroshot_fold0",
                               ft_ball=f"{d}_pathcrf_ball_fold0", ft_noball=f"{d}_pathcrf_noball_fold0",
                               final=f"{d}_pathcrf_final_all", c3t=f"{d}_c3t_fold0"))
    jwrite(RUNS / "CYCLE.json", c)
    print(f"cycle {d} initialised")
    return c


def run_dir(key):
    c = jread(RUNS / "CYCLE.json")
    if c is None:
        raise SystemExit("runs/CYCLE.json missing: run_step2.sh initialises it")
    return RUNS / c["runs"][key]


# ------------------------------------------------------------ ledger, guard, markers

def ledger_add(step, job, seconds):
    L = jread(RUNS / "gpu_ledger.json", [])
    L.append(dict(step=step, job=str(job), hours=float(seconds) / 3600, at=_dt.datetime.now().isoformat(timespec="seconds")))
    jwrite(RUNS / "gpu_ledger.json", L)
    R = jread(RUNS / "reservations.json", {})
    R.pop(step, None)
    jwrite(RUNS / "reservations.json", R)


def gpu_used():
    return sum(x["hours"] for x in jread(RUNS / "gpu_ledger.json", []))


def guard(step):
    if step == "summary":                                   # the summary always runs (it reports the guard's own stops)
        print("OK")
        return
    cap = CAPS[step]
    R = jread(RUNS / "reservations.json", {})
    reserved = sum(v for k, v in R.items() if k != step)
    used = gpu_used()
    if used + reserved + cap > GPU_TOTAL_CAP + 1e-9:
        why = f"before {step}: {used:.2f} h used + {reserved:.2f} h reserved by running jobs + {cap:.1f} h cap > {GPU_TOTAL_CAP:.0f} h"
        S = jread(RUNS / "guard_stop.json", [])
        S.append(dict(step=step, used=used, reserved=reserved, cap=cap, why=why))
        jwrite(RUNS / "guard_stop.json", S)
        print("SKIP")
    else:
        R[step] = cap
        jwrite(RUNS / "reservations.json", R)
        print("OK")


def mark(step, status, job="local"):
    STEPS_DIR.mkdir(parents=True, exist_ok=True)
    for s in ("done", "failed", "skipped", "running"):
        p = STEPS_DIR / f"{step}.{s}"
        if p.exists():
            p.unlink()
    (STEPS_DIR / f"{step}.{status}").write_text(f"{job} {_dt.datetime.now().isoformat(timespec='seconds')}\n")


def step_state(step):
    for s in ("done", "failed", "skipped", "running"):
        if (STEPS_DIR / f"{step}.{s}").exists():
            return s, (STEPS_DIR / f"{step}.{s}").read_text().strip()
    return "not started", ""


# ------------------------------------------------------------ smoke test

def smoke_report():
    """Both smoke runs must load the M1 checkpoint, show a falling loss and fit in memory. Exit 1 otherwise."""
    import numpy as np
    from .adapter import STEP, WIN, folds
    rep, ok = {}, True
    for key, no_ball in (("smoke_ball", False), ("smoke_noball", True)):
        run = run_dir(key)
        s, cfg = jread(run / "logs" / "smoke.json"), jread(run / "config.json", {})
        if not s:
            rep[key] = dict(ok=False, why="no logs/smoke.json (the run failed)"); ok = False
            continue
        init = (cfg.get("init_report") or {})
        n_adapt = len(init.get("adapted", []))
        r = dict(loss_first=s["loss_first"], loss_last=s["loss_last"], steps=s["steps"], seconds_per_step=s["seconds_per_step"],
                 peak_gpu_gb=s["peak_gpu_gb"], device=s["device"], init_from=init.get("path"), adapted_weights=n_adapt)
        why = []
        if not init.get("path"):
            why.append("M1 checkpoint not loaded")
        if no_ball and n_adapt == 0:
            why.append("no-ball run did not narrow the ball inputs")
        if not (math.isfinite(s["loss_last"]) and s["loss_last"] < s["loss_first"]):
            why.append("loss not going down")
        if (s["peak_gpu_gb"] or 0) > 18.5:
            why.append("peak GPU memory too close to the 20 GB slice")
        r.update(ok=not why, why="; ".join(why) or "healthy")
        ok &= not why
        rep[key] = r
    # windows per epoch of the real runs (stride 3 s = 15 steps, dead windows kept), from the built period arrays
    F = folds(BUILT)
    held = set(F[0])
    val = [sorted(f)[0] for k, f in enumerate(F) if k != 0]
    n_win = {}
    for p in sorted((BUILT / "pathcrf").glob("*.npz")):
        with np.load(p) as z:
            per = z["period"]
        n = 0
        for q in np.unique(per[per > 0]):
            steps = int(np.ceil((per == q).sum() / STEP))
            n += 0 if steps < WIN else len(range(0, steps - WIN + 1, 15)) + 1
        n_win[int(p.stem)] = n
    w_fold0 = sum(v for m, v in n_win.items() if m not in held and m not in val)
    w_all = sum(n_win.values())
    sps = max(rep.get("smoke_ball", {}).get("seconds_per_step", 0), rep.get("smoke_noball", {}).get("seconds_per_step", 0))
    rep["windows_per_epoch"] = dict(fold0=w_fold0, all30=w_all)
    if sps:
        m0, ma = w_fold0 / 32 * sps / 60, w_all / 32 * sps / 60
        rep["minutes_per_epoch"] = dict(fold0=m0, all30=ma, note="training only; validation scoring adds a few minutes per epoch")
        rep["epochs_in_cap"] = dict(fold0=TRAIN_HOURS["ft_ball"] * 60 / max(m0 * 1.1, 1e-6),
                                    all30=TRAIN_HOURS["final"] * 60 / max(ma, 1e-6))
    rep["ok"] = ok
    jwrite(RUNS / "smoke_report.json", rep)
    print(json.dumps(rep, indent=1, default=float))
    if not ok:
        raise SystemExit("smoke test unhealthy: the chain stops here (runs/smoke_report.json)")


# ------------------------------------------------------------ decisions

def choose_variant():
    """ball or noball: the fold-0 run with the higher PathCRF spellF1 (combo with C3-T's P(D)). One missing -> the other;
    both missing -> ball (the M1 default), with the reason written."""
    sc = {}
    for v in ("ball", "noball"):
        r = jread(RUNS / f"score_ft_{v}.json")
        if r and "pathcrf" in r:
            sc[v] = r["pathcrf"]["spell"]
    if sc:
        v = max(sc, key=sc.get)
        why = "fold-0 spellF1 " + ", ".join(f"{k} {x:.4f}" for k, x in sc.items()) + f" -> {v}"
    else:
        v, why = "ball", "no fold-0 score from either run: the ball variant (M1's default) is used"
    d = dict(variant=v, flags="--no-ball" if v == "noball" else "", why=why, scores=sc)
    jwrite(RUNS / "decision.json", d)
    print(json.dumps(d, indent=1))


def decision_field(key):
    print((jread(RUNS / "decision.json") or {}).get(key, ""))


def promote(run):
    run = Path(run)
    dst = FINAL / run.name
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("checkpoints/best.pt", "config.json", "NOTES.md"):
        if (run / f).exists():
            shutil.copy(run / f, dst / Path(f).name)
    (dst / "README.txt").write_text("PathCRF fine-tuned from M1's final_all on all 30 degraded + relinked PFF matches (step 2b). "
                                    "Never auto-deleted.\n")
    print(f"promoted to {dst}")


# ------------------------------------------------------------ summary, status

def epochs_log(run):
    p = Path(run) / "logs" / "epochs.jsonl"
    if not p.exists():
        return []
    out = []
    for x in p.read_text().splitlines():
        try:
            out.append(json.loads(x))
        except ValueError:
            pass
    return out


def _f(x, d=4):
    return f"{x:.{d}f}" if isinstance(x, (int, float)) and math.isfinite(x) else "–"


def score_rows(name, title):
    r = jread(RUNS / f"{name}.json")
    if not r:
        return [f"| {title} | not run | | | | |"]
    out = []
    for v in ("c3t", "pathcrf", "blend"):
        if v in r:
            e = r[v]["extra"]
            out.append(f"| {title} / {v} | {_f(r[v]['spell'])} | {_f(r[v]['macro'])} | {_f(e['short_spell_recall'], 3)} | "
                       f"{_f(e['restart_team_acc'], 3)} ({_f(e['restart_matched_share'], 3)}) |")
    return out


SCORES = (("score_zs_published", "zero-shot, published weights (no ball)"), ("score_zs_m1fold0", "zero-shot, M1 fold-0 best.pt"),
          ("score_ft_ball", "fine-tuned fold 0, ball"), ("score_ft_noball", "fine-tuned fold 0, --no-ball"))


def summary():
    c = jread(RUNS / "CYCLE.json", {})
    md = [f"# Step 2b PathCRF cycle {c.get('date', '?')}", "",
          "Fold 0 = 6 held-out matches (3813 3815 3818 3819 3849 3858), degraded + relinked PFF. Every variant uses C3-T's "
          "fold-0 P(D) and decoder; only the team probability changes (c3t / pathcrf / blend). Mean per match.", ""]
    for s in jread(RUNS / "guard_stop.json", []) or []:
        md += [f"**GPU guard skipped a step:** {s['why']}", ""]
    md += ["## Steps", "", "| step | state | job / time |", "|---|---|---|"]
    md += [f"| {s} | {step_state(s)[0]} | {step_state(s)[1]} |" for s in STEPS] + [""]
    b = jread(BUILT / "build_stats.json")
    if b:
        md += ["## Built data", "", f"30 learning matches: both teams >= 10 lanes in {100 * b['share_both_teams_ge10']:.1f} % of 10 s windows, "
               f"{b['mean_links_per_window']:.2f} links / window, targets mapped {100 * b['targets_mapped_share']:.1f} %, lane purity "
               f"{100 * b['lane_purity']:.1f} %"]
        be = jread(BUILT_EVAL / "build_stats.json")
        if be:
            md.append(f"eval matches (stress + locked, predictions only): {be['n_matches']} built, both teams >= 10 lanes in "
                      f"{100 * be['share_both_teams_ge10']:.1f} % of windows")
        md.append("")
    sm = jread(RUNS / "smoke_report.json")
    if sm:
        md += ["## Smoke test", ""]
        for k in ("smoke_ball", "smoke_noball"):
            r = sm.get(k, {})
            md.append(f"- {k}: {r.get('why')}; loss {_f(r.get('loss_first'), 3)} -> {_f(r.get('loss_last'), 3)} over {r.get('steps')} steps, "
                      f"{_f(r.get('seconds_per_step'), 3)} s/step, peak GPU {_f(r.get('peak_gpu_gb'), 1)} GB, init {r.get('init_from')}")
        if "minutes_per_epoch" in sm:
            md.append(f"- estimated minutes per epoch: fold 0 {sm['minutes_per_epoch']['fold0']:.1f}, all 30 {sm['minutes_per_epoch']['all30']:.1f}")
        md.append("")
    md += ["## Fold-0 scores", "", "| model / variant | spellF1 | macroF1 | short-spell recall | restart-team acc (matched) |",
           "|---|---|---|---|---|"]
    for name, title in SCORES:
        md += score_rows(name, title)
    md += ["", "## Training runs (validation = 4 inner matches; spellF1 uses C3-T's in-sample P(D): a monitor)", ""]
    for key in ("ft_ball", "ft_noball", "final"):
        if "runs" not in c:
            break
        ep = epochs_log(RUNS / c["runs"][key])
        st = jread(RUNS / c["runs"][key] / "train_state.json", {})
        md.append(f"- {key}: {len(ep)} epochs, stop {st.get('stop_reason')}, best epoch {st.get('best_epoch')}; " +
                  "; ".join(f"{e['epoch']}: spell {_f(e.get('val_spell'))} macro {_f(e.get('val_macro'))} loss {_f(e.get('val_loss', e.get('train_loss')))} "
                            f"{_f(e.get('minutes'), 1)} min" for e in ep))
    d = jread(RUNS / "decision.json")
    md += ["", "## Chosen variant for the final model", "", (d or {}).get("why", "not reached"), ""]
    if "runs" in c:
        pe = RUNS / c["runs"]["final"] / "preds_eval"
        n = len(list(pe.glob("*.npz"))) if pe.exists() else 0
        md += [f"Eval predictions (6 stress + 16 locked, no scores): {n} files in {pe}", ""]
    L = jread(RUNS / "gpu_ledger.json", [])
    md += ["## GPU hours per job (every job is charged)", "", "| step | job | hours |", "|---|---|---|"]
    md += [f"| {x['step']} | {x['job']} | {x['hours']:.2f} |" for x in L]
    md += [f"| **total** | | **{gpu_used():.2f}** (guard {GPU_TOTAL_CAP:.0f}) |", "",
           "## Download", "", "- runs/STEP2_SUMMARY.md, runs/score_*.json / .md, runs/decision.json, runs/smoke_report.json, "
           "runs/gpu_ledger.json, built/build_stats.json -> D:\\barca_hack\\reports\\auto\\step2b\\",
           "- final/<date>_pathcrf_final_all/ (best.pt, config.json, NOTES.md) -> D:\\barca_hack\\models\\pathcrf\\",
           "- runs/<date>_pathcrf_final_all/preds_eval/ (22 .npz: stress + locked predictions) -> D:\\m1_tmp\\hpc_step2\\preds_eval\\", ""]
    (RUNS / "STEP2_SUMMARY.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.rstrip()
    except (OSError, subprocess.SubprocessError) as e:
        return f"({cmd[0]} unavailable: {e.__class__.__name__})"


def status():
    print("=" * 100)
    print(f"step 2b status  {_dt.datetime.now():%Y-%m-%d %H:%M}   root {ROOT}")
    print("-" * 100 + "\nSlurm (squeue -u $USER):")
    print(_run(["squeue", "-u", os.environ.get("USER", ""), "-o", "%.10i %.22j %.8T %.10M %.12l %R"]))
    print("-" * 100 + "\nSteps:")
    for s in STEPS:
        st, info = step_state(s)
        print(f"  {s:13s} {st:12s} {info}")
    c = jread(RUNS / "CYCLE.json", {})
    print("-" * 100 + "\nTraining runs (last 3 epochs):")
    for key in ("smoke_ball", "smoke_noball", "ft_ball", "ft_noball", "final"):
        if "runs" not in c:
            break
        run = RUNS / c["runs"][key]
        ep = epochs_log(run)
        if not ep and not run.exists():
            continue
        print(f"  {key}: {len(ep)} epochs" + ("" if ep else " (none finished yet)"))
        for e in ep[-3:]:
            print(f"    epoch {e['epoch']}: val spellF1 {_f(e.get('val_spell'))} macroF1 {_f(e.get('val_macro'))} "
                  f"loss {_f(e.get('val_loss', e.get('train_loss')))} edge acc {_f(e.get('val_edge_acc'))} {_f(e.get('minutes'), 1)} min")
    print("-" * 100 + "\nFold-0 scores (spellF1 / macroF1, mean per match):")
    for name, title in SCORES:
        r = jread(RUNS / f"{name}.json")
        if r:
            print(f"  {title}: " + ", ".join(f"{v} {_f(r[v]['spell'])} / {_f(r[v]['macro'])}" for v in ("c3t", "pathcrf", "blend") if v in r))
    R = jread(RUNS / "reservations.json", {})
    print("-" * 100 + f"\nGPU hours: used {gpu_used():.2f}, reserved by running jobs {sum(R.values()):.2f}, guard {GPU_TOTAL_CAP:.0f}")
    for s in jread(RUNS / "guard_stop.json", []) or []:
        print(f"  guard skipped: {s['why']}")
    print("myquota:\n" + _run(["myquota"]))
    logs = sorted(LOGS.glob("b2_*.*"), key=lambda p: p.stat().st_mtime) if LOGS.exists() else []
    if logs:
        p = logs[-1]
        print("-" * 100 + f"\nnewest log: {p}")
        lines = p.read_text(errors="replace").splitlines()
        print("\n".join(lines[-20:]))
    print("=" * 100)


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    a = list(sys.argv[1:] if argv is None else argv)
    if not a:
        raise SystemExit(__doc__)
    cmd, rest = a[0], a[1:]
    if cmd == "init":
        init(rest[1] if len(rest) > 1 and rest[0] == "--date" else None)
    elif cmd == "run-dir":
        print(run_dir(rest[0]))
    elif cmd == "cap":
        print(CAPS[rest[0]])
    elif cmd == "train-hours":
        print(TRAIN_HOURS[rest[0]])
    elif cmd == "guard":
        guard(rest[0])
    elif cmd == "ledger":
        ledger_add(*rest[:3])
    elif cmd == "mark":
        mark(rest[0], rest[1], rest[2] if len(rest) > 2 else "local")
    elif cmd == "is-done":
        print("yes" if (STEPS_DIR / f"{rest[0]}.done").exists() else "no")
    elif cmd == "smoke-report":
        smoke_report()
    elif cmd == "choose-variant":
        choose_variant()
    elif cmd == "decision":
        decision_field(rest[0])
    elif cmd == "promote":
        promote(rest[0])
    elif cmd == "summary":
        summary()
    elif cmd == "status":
        status()
    elif cmd == "m1-ckpt":
        print(M1_FOLD0 if rest[0] == "fold0" else M1_FINAL)
    else:
        raise SystemExit(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
