"""Step-2b cycle bookkeeping and decisions (called by the Slurm job scripts; stdlib only for the commands that run before
the conda env exists). HPC layout:

~/projects/barca-step2/{code,built,runs,final}    this cycle (code = the unzipped bundle)
~/projects/m1-possession/data                     the existing M1 PFF package (matches/*.npz + folds.json), read only
~/envs/m1, ~/logs (b2_<step>_<jobid>.log), ~/scratch

    python -m pathcrf_adapter.cycle init [--date D]           runs/CYCLE.json (run folders; kept on restarts)
    python -m pathcrf_adapter.cycle run-dir KEY               ball / noball / all / smoke / zeroshot / c3t
    python -m pathcrf_adapter.cycle ledger STEP JOBID SECONDS GPU
    python -m pathcrf_adapter.cycle guard STEP CAP_HOURS      OK or SKIP (GPU hours used + CAP > 7 h -> SKIP, reason written)
    python -m pathcrf_adapter.cycle smoke-report SMOKE_RUN    time per step / epoch; runs/recommended_max_epochs.txt
    python -m pathcrf_adapter.cycle max-epochs                10, or the smoke test's recommendation
    python -m pathcrf_adapter.cycle decide                    runs/decision.json: train the all-30 model or not, and with which flags
    python -m pathcrf_adapter.cycle decision FIELD
    python -m pathcrf_adapter.cycle promote RUN               best.pt + config.json + NOTES.md -> final/
    python -m pathcrf_adapter.cycle summary                   runs/STEP2_SUMMARY.md
    python -m pathcrf_adapter.cycle space                     size per folder under ~
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import os
import shutil
import sys
from pathlib import Path

HOME = Path(os.environ.get("B2_HOME", Path.home()))
ROOT = Path(os.environ.get("B2_ROOT", HOME / "projects" / "barca-step2"))
CODE, BUILT, RUNS, FINAL = ROOT / "code", ROOT / "built", ROOT / "runs", ROOT / "final"
PACKAGE = Path(os.environ.get("B2_PACKAGE", HOME / "projects" / "m1-possession" / "data"))
GPU_TOTAL_CAP = 7.0
CAPS = dict(smoke=0.3, zeroshot=0.3, ball=2.4, noball=2.0, all=1.8)     # GPU-hour cap per training / inference step
BEAT_SPELL = 0.005                                                       # the step-1c rule: >= +0.005 spellF1 (mean per match)


def jread(p, default=None):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else default


def jwrite(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=1, default=float))


def cycle():
    c = jread(RUNS / "CYCLE.json")
    if c is None:
        raise SystemExit("runs/CYCLE.json missing: run `python -m pathcrf_adapter.cycle init` (run_step2.sh does this)")
    return c


def init(date=None):
    c = jread(RUNS / "CYCLE.json")
    if c:
        print(f"cycle {c['date']} already initialised (kept)")
        return c
    d = date or _dt.date.today().isoformat()
    c = dict(date=d, runs=dict(ball=f"{d}_pathcrf_ball_fold0", noball=f"{d}_pathcrf_noball_fold0", all=f"{d}_pathcrf_final_all",
                               smoke=f"{d}_smoke", zeroshot=f"{d}_zeroshot_fold0", c3t=f"{d}_c3t_fold0"))
    jwrite(RUNS / "CYCLE.json", c)
    print(f"cycle {d} initialised")
    return c


def run_dir(key):
    return RUNS / cycle()["runs"][key]


def ledger_add(step, job, seconds, gpu):
    L = jread(RUNS / "gpu_ledger.json", [])
    L.append(dict(step=step, job=str(job), hours=float(seconds) / 3600, gpu=bool(int(gpu)), at=_dt.datetime.now().isoformat(timespec="seconds")))
    jwrite(RUNS / "gpu_ledger.json", L)


def gpu_used():
    return sum(x["hours"] for x in jread(RUNS / "gpu_ledger.json", []) if x["gpu"])


def guard(step, cap):
    if jread(RUNS / "guard_stop.json"):
        print("SKIP")
        return
    used = gpu_used()
    if used + float(cap) > GPU_TOTAL_CAP:
        jwrite(RUNS / "guard_stop.json", dict(step=step, used=used, cap=float(cap),
                                              why=f"before {step}: {used:.2f} GPU h used + {float(cap):.1f} h cap > {GPU_TOTAL_CAP:.0f} h total"))
        print("SKIP")
    else:
        print("OK")


def smoke_report(smoke_run):
    s = jread(Path(smoke_run) / "logs" / "smoke.json")
    epoch_s = s["seconds_per_step"] * s["steps_per_epoch"] + s["val_seconds"]
    rep = dict(seconds_per_step=s["seconds_per_step"], steps_per_epoch=s["steps_per_epoch"], peak_gpu_gb=s["peak_gpu_gb"],
               device=s["device"], skipped_share=s["skipped_share"], windows_used=s["windows_used"], windows_all=s["windows_all"],
               epoch_minutes=epoch_s / 60, epochs_in_ball_cap=CAPS["ball"] * 3600 / epoch_s,
               epochs_in_noball_cap=CAPS["noball"] * 3600 / epoch_s, val=s.get("val"))
    rec = max(1, min(10, int(CAPS["ball"] * 0.95 * 3600 // epoch_s)))
    rep["recommended_max_epochs"] = rec
    (RUNS / "recommended_max_epochs.txt").write_text(str(rec))
    jwrite(RUNS / "smoke_report.json", rep)
    print(json.dumps(rep, indent=1, default=float))


def max_epochs(default=10):
    p = RUNS / "recommended_max_epochs.txt"
    print(int(p.read_text().strip()) if p.exists() else default)


def epochs_log(run):
    p = Path(run) / "logs" / "epochs.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def decide():
    """Train the all-30 model if a fine-tuned fold-0 PathCRF variant (pathcrf or blend) beats C3-T by >= 0.005 spellF1."""
    d = dict(train_all=False, checks=[])
    best = None
    for key in ("ball", "noball"):
        r = jread(RUNS / f"score_{key}.json")
        if not r or "c3t" not in r:
            d["checks"].append(f"{key}: no score (training skipped or failed)")
            continue
        for v in ("pathcrf", "blend"):
            if v not in r:
                continue
            ds = r[v]["spell"] - r["c3t"]["spell"]
            ok = ds >= BEAT_SPELL
            d["checks"].append(f"{key} / {v}: spellF1 {r[v]['spell']:.4f} vs C3-T {r['c3t']['spell']:.4f} ({ds:+.4f}) -> "
                               f"{'beats C3-T' if ok else 'does not beat C3-T'}")
            if ok and (best is None or ds > best[2]):
                best = (key, v, ds)
    if best:
        ep = epochs_log(run_dir(best[0]))
        best_ep = (min(range(len(ep)), key=lambda i: ep[i].get("val_loss", math.inf)) + 1) if ep else 2
        d.update(train_all=True, run=best[0], variant=best[1], flags="--no-ball" if best[0] == "noball" else "",
                 max_epochs=int(best_ep),
                 why=f"{best[0]} / {best[1]} beats C3-T by {best[2]:+.4f} spellF1 on fold 0: train on all 30 for {best_ep} epochs "
                     f"(the fold-0 run's best epoch), GPU cap {CAPS['all']} h")
    else:
        d["why"] = "no fine-tuned PathCRF variant beats C3-T by >= 0.005 spellF1 on fold 0: no all-30 model"
    jwrite(RUNS / "decision.json", d)
    print(json.dumps(d, indent=1))
    print("TRAIN_ALL" if d["train_all"] else "NO_TRAIN")


def decision_field(key):
    print((jread(RUNS / "decision.json") or {}).get(key, ""))


def promote(run):
    run = Path(run)
    dst = FINAL / run.name
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("checkpoints/best.pt", "config.json", "NOTES.md"):
        if (run / f).exists():
            shutil.copy(run / f, dst / Path(f).name)
    (dst / "README.txt").write_text("PathCRF fine-tuned on all 30 degraded + relinked PFF learning matches (step 2b). Never auto-deleted.\n")
    print(f"promoted to {dst}")


def summary():
    c = jread(RUNS / "CYCLE.json", {})
    md = [f"# Step 2b PathCRF cycle {c.get('date', '?')}", ""]
    stop = jread(RUNS / "guard_stop.json")
    if stop:
        md += [f"**GPU-hour guard stopped the cycle:** {stop['why']}", ""]
    b = jread(BUILT / "build_stats.json")
    if b:
        md += ["## Built data (degraded + relinked PFF, 30 matches)", "",
               f"both teams >= 10 lanes in {100 * b['share_both_teams_ge10']:.1f} % of 10 s windows; links per window "
               f"{b['mean_links_per_window']:.2f}; forced links {100 * b['forced_link_share']:.1f} %; targets mapped to a lane "
               f"{100 * b['targets_mapped_share']:.1f} %; lane purity {100 * b['lane_purity']:.1f} %", ""]
    sm = jread(RUNS / "smoke_report.json")
    if sm:
        md += ["## Smoke test", "", f"{sm['device']}: {sm['seconds_per_step']:.3f} s/step, {sm['steps_per_epoch']} steps/epoch, "
               f"epoch ≈ {sm['epoch_minutes']:.1f} min, peak GPU memory {(sm['peak_gpu_gb'] or 0):.1f} GB, windows skipped "
               f"{100 * sm['skipped_share']:.1f} %, max epochs {sm['recommended_max_epochs']}", ""]
    md += ["## Fold 0 (6 held-out matches; C3-T's P(D) and decoder in every variant)", ""]
    for key, title in (("zeroshot", "zero-shot (published weights, no ball)"), ("ball", "fine-tuned, ball input"),
                       ("noball", "fine-tuned, --no-ball")):
        p = RUNS / f"score_{key}.md"
        md += [p.read_text().replace("### ", "### " + title + ": ", 1) if p.exists() else f"### {title}: not run\n", ""]
    for key in ("ball", "noball", "all"):
        ep = epochs_log(RUNS / c.get("runs", {}).get(key, "_"))
        if ep:
            md.append(f"- {key}: {len(ep)} epochs; " + ", ".join(f"{e['epoch']}: val loss {e.get('val_loss', float('nan')):.4f} / "
                                                                 f"edge acc {e.get('val_edge_acc', float('nan')):.4f}" for e in ep))
    d = jread(RUNS / "decision.json")
    md += ["", "## Decision", "", (d or {}).get("why", "not reached"), ""] + [f"- {x}" for x in (d or {}).get("checks", [])]
    L = jread(RUNS / "gpu_ledger.json", [])
    md += ["", "## Hours per step", "", "| step | job | hours | GPU |", "|---|---|---|---|"]
    md += [f"| {x['step']} | {x['job']} | {x['hours']:.2f} | {'yes' if x['gpu'] else 'no'} |" for x in L]
    md += [f"| **total GPU** | | **{gpu_used():.2f}** (cap {GPU_TOTAL_CAP:.0f}) | |", "",
           "## Download", "", "- runs/STEP2_SUMMARY.md, runs/score_*.json / .md, runs/decision.json, runs/smoke_report.json, built/build_stats.json",
           "- final/ (only if the all-30 model was trained): best.pt, config.json, NOTES.md", ""]
    (RUNS / "STEP2_SUMMARY.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


def space():
    rows = []
    for sub in (HOME, ROOT, RUNS, BUILT, FINAL, HOME / "envs", HOME / "scratch", HOME / "logs"):
        if sub.exists():
            for p in sub.iterdir():
                if p.is_dir() and not p.is_symlink():
                    rows.append((_size(p), str(p)))
    rows.sort(reverse=True)
    print(f"{'size':>10}  folder")
    for s, p in rows:
        print(f"{s / 2**30:8.2f} GB  {p}")
    print(f"total under ~: {_size(HOME) / 2**30:.2f} GB")


def _size(p: Path) -> int:
    tot = 0
    for dp, dn, fn in os.walk(p, onerror=lambda e: None):
        for f in fn:
            try:
                tot += os.lstat(os.path.join(dp, f)).st_size
            except OSError:
                pass
    return tot


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
    elif cmd == "ledger":
        ledger_add(*rest[:4])
    elif cmd == "guard":
        guard(rest[0], rest[1])
    elif cmd == "cap":
        print(CAPS[rest[0]])
    elif cmd == "smoke-report":
        smoke_report(rest[0])
    elif cmd == "max-epochs":
        max_epochs()
    elif cmd == "decide":
        decide()
    elif cmd == "decision":
        decision_field(rest[0])
    elif cmd == "promote":
        promote(rest[0])
    elif cmd == "summary":
        summary()
    elif cmd == "space":
        space()
    else:
        raise SystemExit(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
