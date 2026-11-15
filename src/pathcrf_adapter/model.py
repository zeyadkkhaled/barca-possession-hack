"""PathCRF (model 140: Set Attention + dynamic sparse CRF) with our changes, without editing the vendored PathCRF code:

- ball input ON: node_in_dim 8 -> 11 (ball x, y, ball_seen). Fine-tuning: every pretrained weight whose input width grows by 3
  gets 3 zero columns inserted right after the 8 player features, so the model starts exactly as published and learns how to use
  the ball. --from-scratch: no pretrained weights.
- travelling-vs-loose head (always starts fresh): reads PathCRF's per-frame node embeddings (output of the last temporal block
  micro_sab_t2, caught with a forward hook), pools over the 26 nodes (mean + max, player_mask-aware) and classifies TRAVELLING vs
  LOOSE with a small MLP. Trained only on frames where the answer key is TRAVELLING or LOOSE; LOOSE has class weight 2 (as in C3).
- loss with a frame mask: DEAD / unscored frames stay in the input but carry no loss. PathCRF's emission losses (edge, src, dst
  cross-entropy) are averaged over unmasked frames; the CRF negative log-likelihood is computed per contiguous run of unmasked
  frames inside each window (the chain restarts after a gap), using PathCRF's own transition scorer and allowed-transition table.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PATHCRF = Path(os.environ.get("M1PC_PATHCRF", HERE.parents[1] / "vendor" / "pathcrf"))     # vendored PathCRF (override for local tests)
if str(PATHCRF) not in sys.path:
    sys.path.insert(0, str(PATHCRF))

from models import utils as pc_utils  # noqa: E402

N = 26
BASE_FEAT = 8


def model_args(ball: bool = True) -> dict:
    a = json.loads((PATHCRF / "saved" / "140" / "args.json").read_text())
    a["node_in_dim"] = BASE_FEAT + (3 if ball else 0)
    return a


def widen_state_dict(sd: dict, model: nn.Module, extra: int) -> tuple[dict, list]:
    """Insert `extra` zero input columns at position BASE_FEAT for every weight whose last dim grew by `extra`."""
    own = model.state_dict()
    out, widened = {}, []
    for k, v in sd.items():
        if k not in own:
            continue
        w = own[k]
        if v.shape == w.shape:
            out[k] = v
        elif v.dim() == 2 and v.shape[0] == w.shape[0] and w.shape[1] - v.shape[1] == extra:
            z = torch.zeros(v.shape[0], extra, dtype=v.dtype)
            out[k] = torch.cat([v[:, :BASE_FEAT], z, v[:, BASE_FEAT:]], 1)
            widened.append(k)
        else:
            raise ValueError(f"cannot widen {k}: {tuple(v.shape)} -> {tuple(w.shape)}")
    return out, widened


class TLHead(nn.Module):
    def __init__(self, dim: int = 128, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * dim, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, 2))

    def forward(self, h, pmask):
        """h: (B, T, N, D), pmask: (B, T, N) -> logits (B, T, 2)."""
        m = pmask.unsqueeze(-1)
        mean = (h * m).sum(2) / m.sum(2).clamp(min=1.0)
        mx = h.masked_fill(m == 0, -1e4).max(2).values
        return self.mlp(torch.cat([mean, mx], -1))


class M1PathCRF(nn.Module):
    def __init__(self, ball: bool = True, pretrained: bool = True):
        super().__init__()
        self.args = model_args(ball)
        self.net = pc_utils.build_model(self.args, torch.device("cpu"))
        self.head = TLHead(self.args["seq_dim"])
        self.load_report = None
        if pretrained:
            sd = torch.load(PATHCRF / "saved" / "140" / "model" / "state_dict_best_acc.pt", map_location="cpu", weights_only=True)
            sd, widened = widen_state_dict(sd, self.net, self.args["node_in_dim"] - BASE_FEAT)
            res = self.net.load_state_dict(sd, strict=False)
            self.load_report = dict(widened=widened, missing=list(res.missing_keys), unexpected=list(res.unexpected_keys))
        self._h = None
        self.net.micro_sab_t2.register_forward_hook(self._grab)

    def _grab(self, module, inp, out):
        self._h = out                                   # (B*N, T, D): temporal SAB runs on (batch, time, dim)

    def forward(self, x, pmask):
        """x (B, T, N, F), pmask (B, T, N) -> edge_logits (B, T, N*N), edge_embeds (B, T, N*N, D), tl_logits (B, T, 2)."""
        B, T = x.shape[:2]
        _, edge_logits, edge_h = self.net(x)
        h = self._h.reshape(B, N, T, -1).transpose(1, 2)            # (B, T, N, D)
        return edge_logits, edge_h, self.head(h, pmask)


# ------------------------------------------------------------ losses

def crf_nll_segments(crf, emissions, edge_embeds, tags, mask):
    """Sum over contiguous unmasked runs of (logZ - gold), per window; PathCRF's dynamic sparse CRF pieces. Returns (B,) and the
    number of unmasked steps per window."""
    B, T, K = emissions.shape
    D = crf.edge_embed_dim
    mask = mask.bool()
    inc_idx = crf.inc_idx.clamp(min=0)
    M = inc_idx.size(1)
    inc_mask = crf.inc_mask
    alpha = emissions[:, 0]
    total = emissions.new_zeros(B)
    gold = torch.where(mask[:, 0], emissions[:, 0].gather(1, tags[:, :1]).squeeze(1), emissions.new_zeros(B))
    inc_idx_alpha = inc_idx.unsqueeze(0).expand(B, -1, -1)
    inc_idx_edge = inc_idx.view(1, K, -1, 1).expand(B, -1, -1, D)
    inc_mask_b = inc_mask.unsqueeze(0).expand(B, -1, -1)
    forbid = crf.forbid_score
    for t in range(1, T):
        cont = mask[:, t] & mask[:, t - 1]
        start = mask[:, t] & ~mask[:, t - 1]
        end = ~mask[:, t] & mask[:, t - 1]
        total = total + torch.where(end, torch.logsumexp(alpha, 1), emissions.new_zeros(B))
        prev_alpha = alpha.unsqueeze(1).expand(-1, K, -1).gather(2, inc_idx_alpha)
        prev_e = edge_embeds[:, t - 1].unsqueeze(1).expand(-1, K, -1, -1).gather(2, inc_idx_edge)
        curr_e = edge_embeds[:, t].unsqueeze(2).expand(-1, -1, M, -1)
        trans = crf.linear(torch.cat([prev_e, curr_e], -1)).squeeze(-1)
        x = torch.where(inc_mask_b, prev_alpha + trans, torch.full_like(prev_alpha, forbid))
        step_alpha = torch.logsumexp(x, 2) + emissions[:, t]
        alpha = torch.where(cont.unsqueeze(1), step_alpha, torch.where(start.unsqueeze(1), emissions[:, t], alpha))
        # gold
        prev, curr = tags[:, t - 1], tags[:, t]
        emit = emissions[:, t].gather(1, curr.unsqueeze(1)).squeeze(1)
        pe = edge_embeds[:, t - 1].gather(1, prev.view(B, 1, 1).expand(-1, 1, D)).squeeze(1)
        ce = edge_embeds[:, t].gather(1, curr.view(B, 1, 1).expand(-1, 1, D)).squeeze(1)
        tr = crf.linear(torch.cat([pe, ce], -1)).squeeze(-1)
        allowed = ((inc_idx[curr] == prev.unsqueeze(1)) & inc_mask[curr]).any(1)
        tr = torch.where(allowed, tr, torch.full_like(tr, forbid))
        gold = gold + torch.where(cont, emit + tr, torch.where(start, emit, torch.zeros_like(emit)))
    total = total + torch.where(mask[:, -1], torch.logsumexp(alpha, 1), emissions.new_zeros(B))
    return total - gold, mask.sum(1)


def losses(model: M1PathCRF, batch: dict, tl_weight: float, loose_w: float = 2.0):
    """PathCRF 140 losses (emit + src + dst cross-entropy, CRF NLL; macro_type poss_edge is unused by model 140's loss config:
    it trains with emit_loss, src_loss, dst_loss and crf_weight 1) restricted to unmasked frames, plus the head loss."""
    net = model.net
    x, pm, src, dst, mask, tl = (batch[k] for k in ("x", "pmask", "src", "dst", "mask", "tl"))
    edge_logits, edge_h, tl_logits = model(x, pm)
    B, T, _ = edge_logits.shape
    m = mask.reshape(-1)
    edge_labels = (src * N + dst).reshape(-1)
    el = edge_logits.reshape(B * T, N, N).float()
    out = {}
    if m.any():
        emit = F.cross_entropy(el.reshape(B * T, -1)[m], edge_labels[m])
        s_l = F.cross_entropy(torch.logsumexp(el, 2)[m], src.reshape(-1)[m])
        d_l = F.cross_entropy(torch.logsumexp(el, 1)[m], dst.reshape(-1)[m])
        tags = net.orig2comp[(src * N + dst)].clamp(min=0)
        emissions = edge_logits.index_select(2, net.valid_orig_edge_ids).float()
        embeds = edge_h.index_select(2, net.valid_orig_edge_ids).float()
        nll, cnt = crf_nll_segments(net.crf, emissions, embeds, tags, mask)
        crf_l = nll.sum() / cnt.sum().clamp(min=1)
        with torch.no_grad():
            out["edge_acc"] = (el.reshape(B * T, -1)[m].argmax(1) == edge_labels[m]).float().mean().item()
    else:
        emit = s_l = d_l = crf_l = edge_logits.sum() * 0
        out["edge_acc"] = float("nan")
    tm = tl.reshape(-1) >= 0
    if tm.any():
        tl_l = F.cross_entropy(tl_logits.reshape(-1, 2).float()[tm], tl.reshape(-1)[tm],
                               weight=torch.tensor([1.0, loose_w], device=tl_logits.device))
        with torch.no_grad():
            out["tl_acc"] = (tl_logits.reshape(-1, 2)[tm].argmax(1) == tl.reshape(-1)[tm]).float().mean().item()
    else:
        tl_l = tl_logits.sum() * 0
        out["tl_acc"] = float("nan")
    loss = emit + s_l + d_l + crf_l + tl_weight * tl_l
    out.update(loss=loss.item(), emit=float(emit), crf=float(crf_l), tl=float(tl_l), n_frames=int(m.sum()), n_tl=int(tm.sum()))
    return loss, out
