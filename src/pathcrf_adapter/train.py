"""Train PathCRF (+ travelling/loose head) on our data. One run = one folder:

<run_dir>/checkpoints/{last.pt,best.pt}   only these two are kept
<run_dir>/logs/epochs.jsonl              one line per epoch (train + validation scores)
<run_dir>/config.json, NOTES.md          settings; NOTES.md is rewritten after training (and again after scoring)

    python -m m1pc.train --data DATA --run-dir RUN --fold 0 [flags]       # train on the other 4 folds (24 matches)
    python -m m1pc.train --data DATA --run-dir RUN --all --max-epochs E    # all 30 learning matches, fixed epochs

Fast config (default) and the flags that switch each item back:
  fine-tune from PathCRF's published weights      --from-scratch
  skip windows with no scored live frame          --keep-dead-windows
  window stride 3 s                               --stride 1   (seconds; 1 s = 5 steps)
  bf16 autocast on CUDA                           --fp32
  max 10 epochs, early stopping patience 2        --max-epochs / --patience
Inner validation (early stopping, per-epoch score): with --fold k, the first match (sorted) of each of the other 4 folds; those
4 matches are not trained on. With --all there is no validation; the run trains exactly --max-epochs epochs.
GPU-hour cap (--gpu-hours, default 8): wall time on the GPU summed over restarts; when the next epoch would exceed it, training
stops cleanly (last.pt and best.pt kept). Auto-resume from last.pt.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import adapter as A
from .model import M1PathCRF, losses

KEYS = ("x", "pmask", "src", "dst", "mask", "tl")


def collate(items):
    out = {k: torch.as_tensor(np.stack([it[k] for it in items])) for k in KEYS}
    out["mid"] = [it["mid"] for it in items]
    out["step0"] = [it["step0"] for it in items]
    out["node_slot"] = np.stack([it["node_slot"] for it in items])
    return out


def split(data, fold, all_matches):
    F = A.folds(data)
    if all_matches:
        return sorted(m for f in F for m in f), [], []
    held = list(F[fold])
    others = [sorted(f) for k, f in enumerate(F) if k != fold]
    val = [f[0] for f in others]
    train = [m for f in others for m in f if m not in val]
    return train, val, held


def variant_name(a) -> str:
    parts = []
    if a.from_scratch:
        parts.append("scratch")
    if a.keep_dead_windows:
        parts.append("keepdead")
    if a.stride != 3:
        parts.append(f"stride{a.stride:g}s")
    if a.fp32:
        parts.append("fp32")
    if a.no_ball:
        parts.append("noball")
    return "+".join(parts) or "fast"


def write_notes(run: Path, cfg: dict, hist: list, state: dict, score: dict | None = None):
    best = min(hist, key=lambda h: h.get("val_loss", math.inf)) if hist and "val_loss" in hist[0] else (hist[-1] if hist else {})
    lines = [f"# {run.name}", "",
             f"- model: PathCRF model 140 (Set Attention + dynamic sparse CRF) + travelling/loose head, ball input {'off' if cfg['no_ball'] else 'on'}",
             f"- variant: {cfg['variant']}   fold: {'all 30 learning matches' if cfg['all'] else cfg['fold']}",
             f"- settings: {json.dumps({k: cfg[k] for k in ('from_scratch', 'keep_dead_windows', 'stride', 'fp32', 'max_epochs', 'patience', 'batch', 'lr', 'tl_weight', 'seed')})}",
             f"- training matches: {len(cfg['train'])}, inner validation: {cfg['val']}, held out: {cfg['held']}",
             f"- windows: {cfg.get('n_windows')} used of {cfg.get('n_windows_all')} ({100 * cfg.get('skipped_share', 0):.1f} % skipped: no scored live frame)",
             f"- epochs run: {len(hist)}; stop reason: {state.get('stop_reason')}; best epoch: {best.get('epoch')}",
             f"- best validation: loss {best.get('val_loss', float('nan')):.4f}, edge acc {best.get('val_edge_acc', float('nan')):.4f}, "
             f"travel/loose acc {best.get('val_tl_acc', float('nan')):.4f}",
             f"- GPU hours used (training): {state.get('gpu_seconds', 0) / 3600:.2f}"]
    if score:
        lines.append(f"- final score (held-out matches): {json.dumps(score)}")
    keep = cfg.get("keep", "keep" if not cfg.get("ablation") else "delete once scored (ablation)")
    lines += [f"- keep/delete: {keep}", ""]
    (run / "NOTES.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--run-dir", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fold", type=int)
    g.add_argument("--all", action="store_true")
    ap.add_argument("--from-scratch", action="store_true")
    ap.add_argument("--keep-dead-windows", action="store_true")
    ap.add_argument("--stride", type=float, default=3.0, help="window stride in seconds (5 steps = 1 s)")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--no-ball", action="store_true")
    ap.add_argument("--max-epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None, help="default 3e-4 fine-tuning, 1e-3 from scratch")
    ap.add_argument("--tl-weight", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--gpu-hours", type=float, default=8.0)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many optimiser steps (smoke test)")
    ap.add_argument("--matches", type=int, nargs="*", help="restrict training matches (CPU checks)")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--ablation", action="store_true", help="NOTES.md says: delete once scored")
    a = ap.parse_args(argv)
    t_start = time.time()
    run = Path(a.run_dir)
    (run / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run / "logs").mkdir(exist_ok=True)
    (run / "preds").mkdir(exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = dev.type == "cuda" and not a.fp32
    lr = a.lr or (1e-3 if a.from_scratch else 3e-4)
    train_m, val_m, held = split(a.data, a.fold, a.all)
    if a.matches:
        train_m = [m for m in train_m if m in a.matches] or list(a.matches)
        val_m = val_m[:1]
    stride = max(1, int(round(a.stride * 5)))
    ball = not a.no_ball
    t0 = time.time()
    ds = A.WindowSet(a.data, train_m, stride, train=True, skip_dead=not a.keep_dead_windows, ball=ball)
    vs = A.WindowSet(a.data, val_m, 15, train=False, skip_dead=True, ball=ball) if val_m else None
    print(f"data: {len(train_m)} train matches, {len(ds)} windows used of {ds.n_all} ({100 * ds.skipped_share:.1f} % skipped), "
          f"{len(vs) if vs else 0} validation windows ({time.time() - t0:.0f}s)", flush=True)
    cfg = dict(vars(a), variant=variant_name(a), lr=lr, train=train_m, val=val_m, held=held, n_windows=len(ds), n_windows_all=ds.n_all,
               skipped_share=ds.skipped_share, all=a.all, keep=None if not a.ablation else "delete once scored (ablation)")
    (run / "config.json").write_text(json.dumps(cfg, indent=1))
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers, collate_fn=collate, drop_last=True,
                    persistent_workers=a.workers > 0, pin_memory=dev.type == "cuda")
    vl = DataLoader(vs, batch_size=a.batch, shuffle=False, num_workers=min(a.workers, 2), collate_fn=collate) if vs else None
    model = M1PathCRF(ball=ball, pretrained=not a.from_scratch).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, a.max_epochs), eta_min=lr / 100)
    state = dict(epoch=0, best=math.inf, best_epoch=0, bad=0, gpu_seconds=0.0, stop_reason=None, steps=0)
    hist = []
    last = run / "checkpoints" / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        state.update(ck["state"]); hist = ck["hist"]
        print(f"resumed from {last}: epoch {state['epoch']}, GPU hours so far {state['gpu_seconds'] / 3600:.2f}", flush=True)
        if state.get("stop_reason") in ("early_stop", "max_epochs", "gpu_cap"):
            print(f"run already finished ({state['stop_reason']})", flush=True)
            write_notes(run, cfg, hist, state)
            (run / "train_state.json").write_text(json.dumps(dict(state, epochs=len(hist)), indent=1))
            return 0
    cap_s = a.gpu_hours * 3600
    session0 = time.time()

    def used():
        return state["gpu_seconds"] + (time.time() - session0)

    epoch_times = [h["seconds"] for h in hist if "seconds" in h]
    stop = None
    while state["epoch"] < a.max_epochs:
        est = np.mean(epoch_times) if epoch_times else 0
        if epoch_times and used() + est > cap_s:
            stop = "gpu_cap"
            break
        ep = state["epoch"] + 1
        model.train()
        te = time.time()
        agg, n = {}, 0
        tw = None
        for i, b in enumerate(dl):
            if i == 10:
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                tw = time.time()
            bt = {k: v.to(dev, non_blocking=True) for k, v in b.items() if k in KEYS}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                loss, info = losses(model, bt, a.tl_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            state["steps"] += 1
            for k, v in info.items():
                if isinstance(v, float) and math.isfinite(v):
                    agg[k] = agg.get(k, 0.0) + v
            n += 1
            if i % 200 == 0:
                print(f"  epoch {ep} batch {i}/{len(dl)} loss {info['loss']:.3f} edge_acc {info['edge_acc']:.3f} "
                      f"({(time.time() - te) / (i + 1):.3f} s/step)", flush=True)
            if a.max_steps and state["steps"] >= a.max_steps:
                stop = "max_steps"
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                sps = (time.time() - tw) / max(i - 9, 1) if tw else (time.time() - te) / (i + 1)
                tv = time.time()
                vrow = validate(model, vl, dev, amp, a.tl_weight) if vl is not None else {}
                (run / "logs" / "smoke.json").write_text(json.dumps(dict(
                    seconds_per_step=sps, steps_per_epoch=len(dl), batch=a.batch, windows_used=len(ds), windows_all=ds.n_all,
                    skipped_share=ds.skipped_share, val_seconds=time.time() - tv, val=vrow,
                    peak_gpu_gb=torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else None,
                    device=torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"), indent=1))
                break
            if used() > cap_s:
                stop = "gpu_cap"
                break
        if stop == "gpu_cap" and n < len(dl):
            break                                   # partial epoch: not counted, last.pt stays at the previous epoch
        sched.step()
        row = dict(epoch=ep, seconds=time.time() - te, **{f"train_{k}": v / max(n, 1) for k, v in agg.items()})
        if vl is not None:
            row.update(validate(model, vl, dev, amp, a.tl_weight))
        state["epoch"] = ep
        epoch_times.append(row["seconds"])
        hist.append(row)
        with open(run / "logs" / "epochs.jsonl", "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"epoch {ep}: " + ", ".join(f"{k} {v:.4f}" for k, v in row.items() if isinstance(v, float)), flush=True)
        metric = row.get("val_loss", -ep)                 # --all: no validation, the last epoch is the best
        if metric < state["best"]:
            state["best"], state["best_epoch"], state["bad"] = metric, ep, 0
            torch.save(dict(model=model.state_dict(), epoch=ep, cfg=cfg), run / "checkpoints" / "best.pt")
        else:
            state["bad"] += 1
        if stop == "max_steps":
            break
        if vl is not None and state["bad"] >= a.patience:
            stop = "early_stop"
        elif state["epoch"] >= a.max_epochs:
            stop = "max_epochs"
        save_last(last, model, opt, sched, state, hist, used())
        if stop:
            break
    state["stop_reason"] = stop or ("max_epochs" if state["epoch"] >= a.max_epochs else None)
    save_last(last, model, opt, sched, state, hist, used())
    for p in (run / "checkpoints").glob("*.pt"):        # keep only last.pt and best.pt
        if p.name not in ("last.pt", "best.pt"):
            p.unlink()
    write_notes(run, cfg, hist, dict(state, gpu_seconds=used()))
    (run / "train_state.json").write_text(json.dumps(dict(state, gpu_seconds=used(), epochs=len(hist)), indent=1))
    if dev.type == "cuda":
        print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
    print(f"done: {state['stop_reason']}, {state['epoch']} epochs, best epoch {state['best_epoch']}, "
          f"GPU hours {used() / 3600:.2f} (this job {(time.time() - t_start) / 3600:.2f})", flush=True)
    return 0


def save_last(path, model, opt, sched, state, hist, used_s):
    st = dict(state, gpu_seconds=used_s)
    tmp = path.with_suffix(".tmp")
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), state=st, hist=hist), tmp)
    os.replace(tmp, path)


@torch.no_grad()
def validate(model, vl, dev, amp, tl_weight):
    model.eval()
    agg, n = {}, 0
    for b in vl:
        bt = {k: v.to(dev) for k, v in b.items() if k in KEYS}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            _, info = losses(model, bt, tl_weight)
        for k in ("loss", "edge_acc", "tl_acc", "crf"):
            v = info[k]
            if math.isfinite(v):
                agg[k] = agg.get(k, 0.0) + v * info["n_frames"]
        n += info["n_frames"]
    return {f"val_{k}": v / max(n, 1) for k, v in agg.items()}


if __name__ == "__main__":
    raise SystemExit(main())
