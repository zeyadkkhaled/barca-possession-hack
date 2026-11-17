"""Per-step PathCRF outputs for held-out matches (all windows, DEAD ones included). Writes <out>/<match>.npz:

frames (S,)            25 fps frame index of each 5 fps step (= C3-T's row grid)
period (S,)
p_src_home (S,)        probability mass of the sender / controller being a home lane (nodes 0-10)
p_src_away (S,)        ... an away lane (nodes 11-21); the rest is an outside node
q_home (S,)            p_src_home / (p_src_home + p_src_away): PathCRF's team probability (what replaces C3-T's layer 2)
path_team (S,)         team of the CRF-decoded (Viterbi) sender, from the window where the step is most central:
                       0 home, 1 away, 2 outside
p_loose (S,)           travelling-vs-loose head: P(LOOSE | ball between players)
covered (S,)           number of windows that saw the step
Probabilities are averaged over all windows that contain the step (prediction stride --pred-stride, default 1 s).

    python -m pathcrf_adapter.predict --data BUILT --run-dir RUN [--matches ...] [--fp32] [--out-subdir preds]
    python -m pathcrf_adapter.predict --data BUILT --model published|CKPT --out DIR [--matches ...]   # zero-shot / any checkpoint
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import adapter as A
from .model import N, M1PathCRF, checkpoint_state, init_from


@torch.no_grad()
def predict_match(model, M: A.Match, dev, amp, stride=5, batch=32, ball=True):
    S = len(M.frames)
    acc = dict(p_src_home=np.zeros(S), p_src_away=np.zeros(S), p_loose=np.zeros(S))
    cov = np.zeros(S)
    best_c = np.full(S, 1e9)
    path_team = np.full(S, -1, np.int8)
    net = model.net
    node_team = np.r_[np.zeros(A.TEAM), np.ones(A.TEAM), 2 * np.ones(4)].astype(np.int8)
    starts = M.window_starts(stride)
    for i in range(0, len(starts), batch):
        ws = [M.window(s, ball=ball) for s in starts[i:i + batch]]
        x = torch.as_tensor(np.stack([w["x"] for w in ws])).to(dev)
        pm = torch.as_tensor(np.stack([w["pmask"] for w in ws])).to(dev)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            el, eh, tl = model(x, pm)
        el = el.float(); eh = eh.float(); tl = tl.float()
        B, T = el.shape[:2]
        pe = torch.softmax(el, -1).reshape(B, T, N, N)
        dec = net.crf.decode(el.index_select(2, net.valid_orig_edge_ids), eh.index_select(2, net.valid_orig_edge_ids))
        d_src = net.comp_src[dec].cpu().numpy()
        src_m = pe.sum(3).cpu().numpy()                                  # (B, T, N) sender mass
        ploose = torch.softmax(tl, -1)[..., 1].cpu().numpy()
        cen = np.abs(np.arange(T) - (T - 1) / 2)
        for b, w in enumerate(ws):
            s0 = w["step0"]
            sl = slice(s0, s0 + T)
            acc["p_src_home"][sl] += src_m[b][:, :A.TEAM].sum(1)
            acc["p_src_away"][sl] += src_m[b][:, A.TEAM:2 * A.TEAM].sum(1)
            acc["p_loose"][sl] += ploose[b]
            cov[sl] += 1
            upd = cen < best_c[sl]
            idx = np.arange(s0, s0 + T)[upd]
            best_c[idx] = cen[upd]
            path_team[idx] = node_team[d_src[b][upd]]
    c = np.maximum(cov, 1)
    out = {k: (v / c).astype(np.float32) for k, v in acc.items()}
    out["q_home"] = (out["p_src_home"] / np.maximum(out["p_src_home"] + out["p_src_away"], 1e-6)).astype(np.float32)
    out.update(frames=M.frames.astype(np.int32), period=M.period.astype(np.int8), path_team=path_team,
               covered=cov.astype(np.int16), n_frames=np.int64(M.n_frames))
    return out


def load_model(spec, dev="cpu"):
    """(model, ball) from "published" (PathCRF's model 140 as is: no ball input, untrained head) or a checkpoint path
    (ours or M1's: the ball input follows the checkpoint's run config)."""
    if spec == "published":
        return M1PathCRF(ball=False, pretrained=True).to(dev).eval(), False
    _, cfg = checkpoint_state(spec)
    ball = not cfg.get("no_ball", False)
    model = M1PathCRF(ball=ball, pretrained=False)
    init_from(model, spec)
    return model.to(dev).eval(), ball


def val_scores(model, matches: dict, data, c3t_dir, dev, amp, ball, stride=15) -> dict:
    """Combo scores (C3-T P(D) + decoder, PathCRF team) on validation matches; mean per match. stride in 5 fps steps."""
    from .score import combo, q_at_rows, score_codes
    dec = json.loads((Path(c3t_dir) / "decoder.json").read_text())
    sc, ex = [], []
    for mid, M in matches.items():
        r = predict_match(model, M, dev, amp, stride=stride, ball=ball)
        with np.load(Path(data) / "c3t" / f"{mid}.npz") as z:
            gt, gmask, period = z["gt"], z["gmask"], z["period"]
        with np.load(Path(c3t_dir) / f"{mid}.npz") as z:
            rows, p_dead, prow = z["rows"], z["p_dead"], z["prow"]
        q = q_at_rows(r["frames"], r["q_home"], rows)
        s = score_codes(gt, gmask, period, combo(p_dead, q, prow, rows, len(period), period, dec["C"], dec["w"], dec["beta"]))
        sc.append((s["spell"], s["macro"])); ex.append(s["extra"])
    a = np.array(sc)
    from barca_train.pipeline import merge_extra
    e = merge_extra(ex)
    return dict(val_spell=float(a[:, 0].mean()), val_macro=float(a[:, 1].mean()), val_short_recall=e["short_spell_recall"],
                val_restart_acc=e["restart_team_acc"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="built folder (pathcrf/<match>.npz)")
    ap.add_argument("--run-dir", help="use RUN/checkpoints/<checkpoint>; output RUN/<out-subdir>")
    ap.add_argument("--model", help="'published' or a checkpoint path (instead of --run-dir)")
    ap.add_argument("--zero-shot", action="store_true", help="same as --model published")
    ap.add_argument("--out", help="output folder (default RUN/<out-subdir>)")
    ap.add_argument("--matches", type=int, nargs="*", help="default: the run's held-out matches, else every built match")
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--pred-stride", type=float, default=1.0, help="seconds")
    ap.add_argument("--out-subdir", default="preds")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = torch default)")
    a = ap.parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = dev.type == "cuda" and not a.fp32
    spec = "published" if a.zero_shot else a.model or str(Path(a.run_dir) / "checkpoints" / a.checkpoint)
    if spec != "published" and not Path(spec).exists():
        print(f"no checkpoint {spec} (training skipped or failed): nothing to predict")
        return 0
    model, ball = load_model(spec, dev)
    if a.matches:
        mids = a.matches
    elif a.run_dir and (Path(a.run_dir) / "config.json").exists() and json.loads((Path(a.run_dir) / "config.json").read_text()).get("held"):
        mids = json.loads((Path(a.run_dir) / "config.json").read_text())["held"]
    else:
        mids = sorted(int(p.stem) for p in (Path(a.data) / "pathcrf").glob("*.npz"))
    out = Path(a.out) if a.out else Path(a.run_dir) / a.out_subdir
    out.mkdir(parents=True, exist_ok=True)
    for m in mids:
        M = A.Match(A.load_match(A.match_path(a.data, m)))
        r = predict_match(model, M, dev, amp, stride=max(1, int(round(a.pred_stride * 5))), ball=ball)
        np.savez_compressed(out / f"{m}.npz", **r)
        print(f"{m}: {len(r['frames'])} steps, uncovered {(r['covered'] == 0).mean():.4f}", flush=True)
    (out / "README.txt").write_text(f"PathCRF predictions: model {spec} (ball input {'on' if ball else 'off'}), "
                                    f"{'fp32' if not amp else 'bf16'} inference; format: pathcrf_adapter/predict.py docstring\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
