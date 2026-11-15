"""Per-frame PathCRF outputs for held-out matches (all windows, DEAD ones included). Writes <run>/preds/<match>.npz:

steps (S,)                frame index (29.97 fps) of each 5 fps step; a step holds for the frames up to the next step
period (S,)
p_has (S, P)              per player column: probability that the player has the ball (self-link)
p_src (S, P+4)            last controller / sender (player columns, then out_left, out_right, out_bottom, out_top)
p_dst (S, P+4)            next controller / receiver
p_home (S,)               team probability: P(the sender is a home player)
link_src, link_dst (S, 3), link_p (S, 3)   the 3 most likely links (column or P+k for an outside node) and their probabilities
path_src, path_dst (S,)   PathCRF's Viterbi (CRF) decoded link, from the window in which the step is most central
p_loose (S,)              travelling-vs-loose head: P(LOOSE | ball between players)
covered (S,)              number of windows that saw the step
Probabilities are averaged over all windows that contain the step (prediction stride --pred-stride, default 1 s).

    python -m m1pc.predict --data DATA --run-dir RUN [--matches ...] [--fp32] [--out-subdir preds]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import adapter as A
from .model import N, M1PathCRF


@torch.no_grad()
def predict_match(model, M: A.Match, dev, amp, stride=5, batch=32, ball=True):
    S, P = M.px.shape
    acc = dict(p_has=np.zeros((S, P)), p_src=np.zeros((S, P + 4)), p_dst=np.zeros((S, P + 4)), p_home=np.zeros(S), p_loose=np.zeros(S))
    cov = np.zeros(S)
    best_c = np.full(S, 1e9)
    path_src = np.full(S, -1, np.int16); path_dst = np.full(S, -1, np.int16)
    link_src = np.full((S, 3), -1, np.int16); link_dst = np.full((S, 3), -1, np.int16); link_p = np.zeros((S, 3), np.float32)
    starts = M.window_starts(stride)
    net = model.net
    home = (M.side == 0)
    for i in range(0, len(starts), batch):
        ws = [M.window(s, ball=ball) for s in starts[i:i + batch]]
        x = torch.as_tensor(np.stack([w["x"] for w in ws])).to(dev)
        pm = torch.as_tensor(np.stack([w["pmask"] for w in ws])).to(dev)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            el, eh, tl = model(x, pm)
        el = el.float(); eh = eh.float(); tl = tl.float()
        B, T = el.shape[:2]
        pe = torch.softmax(el, -1).reshape(B, T, N, N)
        emis = el.index_select(2, net.valid_orig_edge_ids)
        emb = eh.index_select(2, net.valid_orig_edge_ids)
        dec = net.crf.decode(emis, emb)                                   # (B, T) compressed ids
        d_src, d_dst = net.comp_src[dec].cpu().numpy(), net.comp_dst[dec].cpu().numpy()
        pe = pe.cpu().numpy(); ploose = torch.softmax(tl, -1)[..., 1].cpu().numpy()
        top_v, top_i = torch.topk(torch.softmax(el, -1), 3, dim=-1)
        top_v, top_i = top_v.cpu().numpy(), top_i.cpu().numpy()
        for b, w in enumerate(ws):
            s0 = w["step0"]
            ns = w["node_slot"]                                           # node -> column (-1: outside / padding)
            col = np.where(ns >= 0, ns, -1)
            col[2 * A.TEAM:] = P + np.arange(4)
            real = col >= 0
            sl = slice(s0, s0 + T)
            diag = pe[b][:, np.arange(N), np.arange(N)]                   # (T, N)
            src_m, dst_m = pe[b].sum(2), pe[b].sum(1)
            for nd in np.flatnonzero(real):
                c = col[nd]
                if c < P:
                    acc["p_has"][sl, c] += diag[:, nd]
                acc["p_src"][sl, c] += src_m[:, nd]
                acc["p_dst"][sl, c] += dst_m[:, nd]
            hn = np.array([real[nd] and col[nd] < P and home[col[nd]] for nd in range(N)])
            acc["p_home"][sl] += src_m[:, hn].sum(1)
            acc["p_loose"][sl] += ploose[b]
            cov[sl] += 1
            cen = np.abs(np.arange(T) - (T - 1) / 2)
            upd = cen < best_c[sl]
            idx = np.arange(s0, s0 + T)[upd]
            best_c[idx] = cen[upd]
            path_src[idx] = col[d_src[b][upd]]
            path_dst[idx] = col[d_dst[b][upd]]
            ls, ld = top_i[b] // N, top_i[b] % N
            link_src[idx] = col[ls[upd]]; link_dst[idx] = col[ld[upd]]; link_p[idx] = top_v[b][upd]
    c = np.maximum(cov, 1)
    out = {k: (v / (c[:, None] if v.ndim == 2 else c)).astype(np.float32) for k, v in acc.items()}
    out.update(steps=M.frames.astype(np.int32), period=M.period.astype(np.int8), path_src=path_src, path_dst=path_dst,
               link_src=link_src, link_dst=link_dst, link_p=link_p, covered=cov.astype(np.int16),
               player_id=M.d["player_id"], player_team_id=M.d["player_team_id"], n_frames=np.int64(M.n_frames))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--matches", type=int, nargs="*", help="default: the run's held-out matches")
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--pred-stride", type=float, default=1.0, help="seconds")
    ap.add_argument("--out-subdir", default="preds")
    a = ap.parse_args(argv)
    run = Path(a.run_dir)
    cfg = json.loads((run / "config.json").read_text())
    mids = a.matches or cfg["held"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = dev.type == "cuda" and not a.fp32
    ball = not cfg.get("no_ball", False)
    model = M1PathCRF(ball=ball, pretrained=False).to(dev)
    ck = torch.load(run / "checkpoints" / a.checkpoint, map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    out = run / a.out_subdir
    out.mkdir(parents=True, exist_ok=True)
    for m in mids:
        M = A.Match(A.load_match(Path(a.data) / "matches" / f"{m}.npz"))
        r = predict_match(model, M, dev, amp, stride=max(1, int(round(a.pred_stride * 5))), ball=ball)
        np.savez_compressed(out / f"{m}.npz", **r)
        print(f"{m}: {len(r['steps'])} steps, uncovered {(r['covered'] == 0).mean():.4f}", flush=True)
    (out / "README.txt").write_text(f"PathCRF predictions, checkpoint {a.checkpoint} (epoch {ck.get('epoch')}), "
                                    f"{'fp32' if not amp else 'bf16'} inference; format: m1pc/predict.py docstring\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
