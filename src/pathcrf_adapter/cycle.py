"""Cycle bookkeeping and decisions (called by the Slurm job scripts). Paths follow the HPC layout:

~/projects/m1-possession/{code,data,runs,final}, ~/logs, ~/scratch

    python -m m1pc.cycle init [--date D]              runs/CYCLE.json (run folders of this cycle; kept on restarts)
    python -m m1pc.cycle run-dir KEY                  prints the run folder of main / ref / a1 / a2 / a3 / all
    python -m m1pc.cycle ledger STEP JOBID SECONDS GPU   adds a line to runs/gpu_ledger.json
    python -m m1pc.cycle guard STEP CAP_HOURS         prints OK or SKIP (total GPU hours + CAP > 30 -> SKIP, writes the reason)
    python -m m1pc.cycle smoke-report SMOKE_RUN       estimates per epoch / fold; writes runs/recommended_max_epochs.txt if > 8 h
    python -m m1pc.cycle score RUN [--preds preds] [--out NAME]   scores C3 and the C3 + PathCRF variants on fold 0
    python -m m1pc.cycle ablation                     runs/ablation.json + ablation.md (verdict per speed cut)
    python -m m1pc.cycle decide                       runs/decision.json: train the all-matches model or not, and its flags
    python -m m1pc.cycle promote RUN                  copies best.pt, config.json, NOTES.md to final/
    python -m m1pc.cycle summary                      runs/CYCLE_SUMMARY.md
    python -m m1pc.cycle space                        size per folder under ~, largest first
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import os
import shutil
import sys
from pathlib import Path

HOME = Path(os.environ.get("M1_HOME", Path.home()))
ROOT = Path(os.environ.get("M1_ROOT", HOME / "projects" / "m1-possession"))
CODE, RUNS, FINAL = ROOT / "code", ROOT / "runs", ROOT / "final"
DATA = Path(os.environ.get("M1_DATA", ROOT / "data"))
GPU_TOTAL_CAP = 30.0
FOLD0_BUDGET_H = 8.0
BEAT_ACC, BEAT_SPELL, WORST = 0.003, 0.005, 0.01           # +0.3 pts acc OR +0.5 pts spell-F1; no metric worse by > 1 pt
HARM_ACC, HARM_SPELL = 0.003, 0.005
ABL = dict(ref=("fast config", ""), a1=("fine-tuning", "--from-scratch"), a2=("dead-window skip", "--keep-dead-windows"),
           a3=("3 s stride", "--stride 1"))


def jread(p, default=None):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else default


def jwrite(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=1, default=float))


# ------------------------------------------------------------ cycle file, ledger, guard

def cycle():
    c = jread(RUNS / "CYCLE.json")
    if c is None:
        raise SystemExit("runs/CYCLE.json missing: run `python -m m1pc.cycle init` (run_cycle.sh does this)")
    return c


def init(date=None):
    c = jread(RUNS / "CYCLE.json")
    if c:
        print(f"cycle {c['date']} already initialised (kept)")
        return c
    d = date or _dt.date.today().isoformat()
    c = dict(date=d, runs=dict(main=f"{d}_pathcrf_fast_fold0", ref=f"{d}_pathcrf_ref3ep_fold0", a1=f"{d}_pathcrf_scratch3ep_fold0",
                               a2=f"{d}_pathcrf_keepdead3ep_fold0", a3=f"{d}_pathcrf_stride1s3ep_fold0", all=f"{d}_pathcrf_final_all"))
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
    stop = jread(RUNS / "guard_stop.json")
    if stop:
        print("SKIP")
        return
    used = gpu_used()
    if used + float(cap) > GPU_TOTAL_CAP:
        jwrite(RUNS / "guard_stop.json", dict(step=step, used=used, cap=float(cap),
                                              why=f"before {step}: {used:.1f} GPU h used + {float(cap):.1f} h cap > {GPU_TOTAL_CAP:.0f} h total"))
        print("SKIP")
    else:
        print("OK")


# ------------------------------------------------------------ smoke test

def smoke_report(smoke_run):
    s = jread(Path(smoke_run) / "logs" / "smoke.json")
    cfg = jread(Path(smoke_run) / "config.json")
    epoch_s = s["seconds_per_step"] * s["steps_per_epoch"] + s["val_seconds"]
    max_ep = 10
    fold_h = epoch_s * max_ep / 3600
    rep = dict(seconds_per_step=s["seconds_per_step"], steps_per_epoch=s["steps_per_epoch"], peak_gpu_gb=s["peak_gpu_gb"], device=s["device"],
               skipped_share=s["skipped_share"], windows_used=s["windows_used"], windows_all=s["windows_all"],
               epoch_minutes=epoch_s / 60, fold_hours_at_10_epochs=fold_h, ablation_hours_3_epochs=3 * epoch_s / 3600,
               ablation_stride1_hours_3_epochs=3 * epoch_s * 3 / 3600, val=s.get("val"))
    if fold_h > FOLD0_BUDGET_H:
        rec = max(1, int(FOLD0_BUDGET_H * 0.95 * 3600 // epoch_s))
        rep["recommended_max_epochs"] = rec
        (RUNS / "recommended_max_epochs.txt").write_text(str(rec))
    jwrite(RUNS / "smoke_report.json", rep)
    print(json.dumps(rep, indent=1, default=float))


def max_epochs_for_main(default=10):
    p = RUNS / "recommended_max_epochs.txt"
    print(int(p.read_text().strip()) if p.exists() else default)


# ------------------------------------------------------------ scoring

def score_run(run, preds="preds", out=None, write_md=True):
    import numpy as np
    from .adapter import folds, load_match
    from .combine import VARIANTS, combine
    from .evaluate import METRICS, score_match, summarize
    run = Path(run)
    fold0 = folds(DATA)[0]
    sc = {k: [] for k in ("C3",) + VARIANTS}
    for m in fold0:
        d = load_match(DATA / "matches" / f"{m}.npz")
        c3 = dict(np.load(DATA / "c3_fold0" / f"{m}.npz"))
        sc["C3"].append(score_match(d, c3["state"], c3["player"], c3["team"]))
        pp = run / preds / f"{m}.npz"
        if not pp.exists():
            continue
        p = dict(np.load(pp))
        for v in VARIANTS:
            st, pl, tm = combine(p, c3, d["period"].astype(int), v)
            sc[v].append(score_match(d, st, pl, tm))
    res = {k: summarize(v) for k, v in sc.items() if len(v) == len(fold0)}
    name = out or "score"
    jwrite(run / f"{name}.json", res)
    if write_md:
        (run / f"{name}.md").write_text(results_md(res, f"{run.name} ({preds})"), encoding="utf-8")
    return res


def results_md(res, title):
    from .evaluate import METRICS
    cols = list(res)
    md = [f"# Fold 0 held-out matches: {title}", "", "C3 = the frozen C3 (C3 + per-transition decoder). Variants: m1pc/combine.py.", "",
          "| metric | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for k, nm in METRICS:
        md.append(f"| {nm} | " + " | ".join(f"{res[c][k]:.4f}" for c in cols) + " |")
    if "C3" in res:
        md += ["", "Δ vs C3 (points):", "", "| metric | " + " | ".join(c for c in cols if c != "C3") + " |", "|---|" + "---|" * (len(cols) - 1)]
        for k, nm in METRICS:
            md.append(f"| {nm} | " + " | ".join(f"{100 * (res[c][k] - res['C3'][k]):+.2f}" for c in cols if c != "C3") + " |")
    return "\n".join(md) + "\n"


def score_fold0():
    run = run_dir("main")
    res = score_run(run, "preds", "results_fold0")
    shutil.copy(run / "results_fold0.json", RUNS / "results_fold0.json")
    shutil.copy(run / "results_fold0.md", RUNS / "results_fold0.md")
    notes_score(run, res)
    print((RUNS / "results_fold0.md").read_text())


def notes_score(run, res):
    p = Path(run) / "NOTES.md"
    if p.exists() and "pathcrf" in res:
        s = res["pathcrf"]
        txt = p.read_text().replace("- keep/delete:", f"- fold-0 score (PathCRF alone): acc {s['acc']:.4f}, spell-F1 {s['spell']:.4f}\n- keep/delete:", 1)
        p.write_text(txt)


# ------------------------------------------------------------ ablation

def epochs_log(run):
    p = Path(run) / "logs" / "epochs.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def ablation():
    out, md = {}, ["# Ablation of the speed cuts (fold 0, 3 epochs each)", "",
                   f"A cut is HARMFUL if putting it back improves frame accuracy by >= {100 * HARM_ACC:.1f} points or spell-F1 by "
                   f">= {100 * HARM_SPELL:.1f} points (PathCRF alone, fold-0 held-out matches).", ""]
    sc = {k: jread(run_dir(k) / "score.json") for k in ABL}
    ref = (sc["ref"] or {}).get("pathcrf")
    md += ["| run | flag | epochs | frame acc | spell-F1 | Δacc vs REF | Δspell vs REF | verdict |", "|---|---|---|---|---|---|---|---|"]
    for k, (what, flag) in ABL.items():
        ep = epochs_log(run_dir(k))
        s = (sc[k] or {}).get("pathcrf")
        row = dict(cut=what, flag=flag, epochs=len(ep), score=s)
        if k == "ref":
            row["verdict"] = "reference"
        elif s is None or ref is None:
            row["verdict"] = "not run (skipped or failed)"
        else:
            da, ds = s["acc"] - ref["acc"], s["spell"] - ref["spell"]
            row.update(d_acc=da, d_spell=ds)
            row["verdict"] = "HARMFUL" if (da >= HARM_ACC or ds >= HARM_SPELL) else "safe"
            ref_ep = epochs_log(run_dir("ref"))
            if k == "a3" and len(ep) < 3 and ep and len(ref_ep) >= len(ep):
                e = len(ep)
                dv = ep[e - 1].get("val_edge_acc", float("nan")) - ref_ep[e - 1].get("val_edge_acc", float("nan"))
                row["note"] = f"hit the GPU cap after {e} epochs; validation edge accuracy vs REF at epoch {e}: {100 * dv:+.2f} points"
                row["verdict"] = "HARMFUL" if dv >= HARM_ACC else row["verdict"]
        out[k] = row
        f = lambda x: f"{x:.4f}" if isinstance(x, float) else "-"
        md.append(f"| {k.upper()} | `{flag or '(fast)'}` | {len(ep)} | {f((s or {}).get('acc'))} | {f((s or {}).get('spell'))} | "
                  f"{100 * row['d_acc']:+.2f} | {100 * row['d_spell']:+.2f} | {row['verdict']} |" if "d_acc" in row else
                  f"| {k.upper()} | `{flag or '(fast)'}` | {len(ep)} | {f((s or {}).get('acc'))} | {f((s or {}).get('spell'))} | - | - | {row['verdict']} |")
        if row.get("note"):
            md.append(f"| | | | | | | | {row['note']} |")
    # A4: bf16 vs fp32 inference of the main model
    main = run_dir("main")
    b, f32 = jread(main / "results_fold0.json"), jread(main / "score_fp32.json")
    if b and f32 and "pathcrf" in b and "pathcrf" in f32:
        da, ds = f32["pathcrf"]["acc"] - b["pathcrf"]["acc"], f32["pathcrf"]["spell"] - b["pathcrf"]["spell"]
        out["a4"] = dict(cut="bf16 inference", flag="--fp32", d_acc=da, d_spell=ds, verdict="HARMFUL" if (da >= HARM_ACC or ds >= HARM_SPELL) else "safe")
        md += ["", f"A4 bf16: the main model with --fp32 inference vs bf16: Δacc {100 * da:+.2f}, Δspell {100 * ds:+.2f} points -> **{out['a4']['verdict']}**"]
    else:
        out["a4"] = dict(cut="bf16 inference", flag="--fp32", verdict="not run")
    # epoch cap: was validation still improving at the last epoch of the main run?
    ep = epochs_log(main)
    st = jread(main / "train_state.json", {})
    if ep:
        best = min(range(len(ep)), key=lambda i: ep[i].get("val_loss", math.inf)) + 1
        improving = best == len(ep) and st.get("stop_reason") in ("max_epochs", "gpu_cap")
        out["epochs"] = dict(cut="epoch cap", flag="--max-epochs", epochs=len(ep), best_epoch=best, stop=st.get("stop_reason"),
                             verdict="HARMFUL" if improving else "safe")
        md += ["", f"Epoch cap: main run {len(ep)} epochs, best epoch {best}, stop reason {st.get('stop_reason')} -> "
                   f"**{'still improving at the cap (HARMFUL)' if improving else 'safe'}**",
               "", "Validation per epoch (main run): " + ", ".join(f"{e['epoch']}: loss {e.get('val_loss', float('nan')):.4f} / edge acc "
                                                                   f"{e.get('val_edge_acc', float('nan')):.4f}" for e in ep)]
    jwrite(RUNS / "ablation.json", out)
    (RUNS / "ablation.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


# ------------------------------------------------------------ decision

def decide():
    from .evaluate import METRICS
    res = jread(RUNS / "results_fold0.json") or {}
    abl = jread(RUNS / "ablation.json") or {}
    d = dict(train_all=False)
    if "C3" not in res or len(res) < 2:
        d["why"] = "no fold-0 results for PathCRF (an earlier step was skipped or failed)"
    else:
        c3 = res["C3"]
        best, why = None, []
        for v, s in res.items():
            if v == "C3":
                continue
            da, ds = s["acc"] - c3["acc"], s["spell"] - c3["spell"]
            worst = min((s[k] - c3[k], nm) for k, nm in METRICS)
            ok = (da >= BEAT_ACC or ds >= BEAT_SPELL) and worst[0] >= -WORST
            why.append(f"{v}: Δacc {100 * da:+.2f}, Δspell {100 * ds:+.2f}, worst {worst[1]} {100 * worst[0]:+.2f} points -> "
                       f"{'beats C3' if ok else 'does not beat C3'}")
            if ok and (best is None or (da, ds) > best[1]):
                best = (v, (da, ds))
        d["checks"] = why
        if best:
            flags = []
            for k, fl in (("a1", "--from-scratch"), ("a2", "--keep-dead-windows"), ("a3", "--stride 1"), ("a4", "--fp32")):
                if abl.get(k, {}).get("verdict") == "HARMFUL":
                    flags.append(fl)
            ep = epochs_log(run_dir("main"))
            best_ep = (min(range(len(ep)), key=lambda i: ep[i].get("val_loss", math.inf)) + 1) if ep else 10
            if abl.get("epochs", {}).get("verdict") == "HARMFUL":
                best_ep = min(2 * best_ep, 20)
            d.update(train_all=True, variant=best[0], flags=" ".join(flags), max_epochs=int(best_ep),
                     why=f"{best[0]} beats C3 on fold 0; train one model on all 30 learning matches for {best_ep} epochs "
                         f"(the main run's best epoch{', doubled: still improving at the cap' if abl.get('epochs', {}).get('verdict') == 'HARMFUL' else ''}) "
                         f"with {'flags ' + ' '.join(flags) if flags else 'the fast config (no harmful cut)'}")
        else:
            d["why"] = "PathCRF did not beat C3"
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
    (dst / "README.txt").write_text("Final PathCRF model for the M1 locked test (trained on all 30 learning matches). Never auto-deleted.\n")
    print(f"promoted to {dst}")


# ------------------------------------------------------------ summary, space

def summary():
    c = jread(RUNS / "CYCLE.json", {})
    md = [f"# M1 PathCRF cycle {c.get('date', '?')}", ""]
    stop = jread(RUNS / "guard_stop.json")
    if stop:
        md += [f"**GPU-hour guard stopped the cycle:** {stop['why']}", ""]
    sm = jread(RUNS / "smoke_report.json")
    if sm:
        md += ["## Smoke test", "", f"{sm['device']}: {sm['seconds_per_step']:.3f} s/step, {sm['steps_per_epoch']} steps/epoch, "
               f"epoch ≈ {sm['epoch_minutes']:.1f} min, fold at 10 epochs ≈ {sm['fold_hours_at_10_epochs']:.1f} h, peak GPU memory "
               f"{(sm['peak_gpu_gb'] or 0):.1f} GB, windows skipped (no scored live frame) {100 * sm['skipped_share']:.1f} %"
               + (f"; recommended --max-epochs {sm['recommended_max_epochs']}" if "recommended_max_epochs" in sm else ""), ""]
    for f in ("results_fold0.md", "ablation.md"):
        if (RUNS / f).exists():
            md += [(RUNS / f).read_text().replace("# ", "## ", 1), ""]
    d = jread(RUNS / "decision.json")
    md += ["## Decision", "", (d or {}).get("why", "not reached"), ""]
    if d and d.get("checks"):
        md += [f"- {x}" for x in d["checks"]] + [""]
    L = jread(RUNS / "gpu_ledger.json", [])
    md += ["## GPU hours per step", "", "| step | job | hours | GPU |", "|---|---|---|---|"]
    md += [f"| {x['step']} | {x['job']} | {x['hours']:.2f} | {'yes' if x['gpu'] else 'no'} |" for x in L]
    md += [f"| **total GPU** | | **{gpu_used():.2f}** | |", "",
           "## Download", "", "- runs/CYCLE_SUMMARY.md, runs/results_fold0.md, runs/results_fold0.json, runs/ablation.md, runs/ablation.json",
           f"- runs/{c.get('runs', {}).get('main', '<main run>')}/preds/*.npz (PathCRF fold-0 predictions) and NOTES.md",
           "- final/ (only if the all-matches model was trained)", ""]
    (RUNS / "CYCLE_SUMMARY.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


def space():
    rows = []
    for base in (HOME,):
        for p in sorted(base.iterdir()):
            if p.is_dir() and not p.is_symlink():
                rows.append((_size(p), str(p)))
        for sub in (ROOT / "runs", ROOT / "data", ROOT / "final", HOME / "envs", HOME / "scratch", HOME / "logs"):
            if sub.exists():
                for p in sub.iterdir():
                    if p.is_dir():
                        rows.append((_size(p), str(p)))
    rows.sort(reverse=True)
    print(f"{'size':>10}  folder")
    for s, p in rows:
        print(f"{s / 2**30:8.2f} GB  {p}")
    print(f"quota 30 GB; total under ~: {_size(HOME) / 2**30:.2f} GB")


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
    cmd, rest = a[0], a[1:]
    if cmd == "init":
        init(rest[1] if len(rest) > 1 and rest[0] == "--date" else None)
    elif cmd == "run-dir":
        print(run_dir(rest[0]))
    elif cmd == "ledger":
        ledger_add(*rest[:4])
    elif cmd == "guard":
        guard(rest[0], rest[1])
    elif cmd == "smoke-report":
        smoke_report(rest[0])
    elif cmd == "max-epochs":
        max_epochs_for_main()
    elif cmd == "score":
        run = rest[0]
        preds = rest[rest.index("--preds") + 1] if "--preds" in rest else "preds"
        out = rest[rest.index("--out") + 1] if "--out" in rest else None
        r = score_run(run, preds, out)
        print(json.dumps({k: {m: round(v[m], 4) for m in ("acc", "spell")} for k, v in r.items()}))
    elif cmd == "score-fold0":
        score_fold0()
    elif cmd == "ablation":
        ablation()
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
