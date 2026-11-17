"""C3-T for one fold on the HPC (CPU): train on the other folds' built features (build.py -> <data>/c3t/), predict the
held-out matches, save per match the row-level P(D) and P(H | alive) plus the decoder, so score.py can swap C3-T's team
probability for PathCRF's under the same dead-ball model and the same Viterbi decoder.

    python -m pathcrf_adapter.c3t_fold --data BUILT --fold 0 --out RUNS/c3t_fold0 [--threads 4]

<out>/<match>.npz for all 30 matches (in_sample = trained on; the held-out ones are the honest scores): rows, p_dead,
q_home, prow; <out>/decoder.json: C, w, beta, train / held matches.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from barca_possession import c3t as T
from barca_train import pipeline as PL

from .adapter import folds


def load_feats(data, mid) -> dict:
    with np.load(Path(data) / "c3t" / f"{mid}.npz") as z:
        return {k: z[k] for k in z.files}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    t0 = time.time()
    F = folds(a.data)
    held = sorted(F[a.fold])
    train = sorted(m for k, f in enumerate(F) if k != a.fold for m in f)
    feats = {m: load_feats(a.data, m) for m in train + held}
    print(f"fold {a.fold}: train {len(train)}, held out {len(held)} ({time.time() - t0:.0f} s)", flush=True)
    model = PL.train(feats, train, a.seed, a.threads, log=lambda s: print(s, flush=True))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sc = []
    for m in train + held:                       # training matches too: PathCRF's inner validation uses their P(D) (in-sample)
        f = feats[m]
        p_dead = model["l1"].predict(f["X"])
        q = model["l2"].predict(T.l2_inputs(f["X"], p_dead, f["prow"]))
        np.savez_compressed(out / f"{m}.npz", rows=f["rows"], p_dead=p_dead.astype(np.float32), q_home=q.astype(np.float32),
                            prow=f["prow"], in_sample=np.bool_(m in train))
        if m in held:
            sc.append(PL.score_hda(f["gt"], f["gmask"], f["period"], PL.predict(model, f))["spell"])
    (out / "decoder.json").write_text(json.dumps(dict(C=model["C"].tolist(), w=model["w"], beta=model["beta"], fold=a.fold,
                                                      train=train, held=held, inner_spell=model["inner_spell"]), indent=1))
    print(f"C3-T fold {a.fold}: held-out spellF1 {np.mean(sc):.4f} (mean per match) ({time.time() - t0:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
