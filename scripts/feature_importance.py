"""Top features by total split gain of the saved C3-T models (models/c3, fit on all 30 degraded PFF matches).

    .venv/Scripts/python.exe scripts/feature_importance.py [--top 10]

Reads only the model files and decoder.json (feature names); prints names and gain shares; writes
reports/auto/feature_importance.json.
"""
from __future__ import annotations

import argparse
import json

import lightgbm as lgb
import numpy as np

from barca_possession.c3t import STACK_NAMES
from barca_possession.paths import AUTO, MODELS


def top(booster: lgb.Booster, names: list, k: int) -> list:
    g = booster.feature_importance("gain")
    assert len(g) == len(names), (len(g), len(names))
    o = np.argsort(-g)[:k]
    return [dict(name=names[i], gain_share=float(g[i] / g.sum())) for i in o]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args()
    d = MODELS / "c3"
    names = json.loads((d / "decoder.json").read_text())["features"]
    res = dict(layer1=top(lgb.Booster(model_file=str(d / "layer1_dead.txt")), names, a.top),
               layer2=top(lgb.Booster(model_file=str(d / "layer2_home.txt")), names + STACK_NAMES, a.top),
               n_features_l1=len(names), n_features_l2=len(names) + len(STACK_NAMES))
    for lay in ("layer1", "layer2"):
        print(f"{lay} top {a.top} by gain: " + ", ".join(f"{r['name']} ({100 * r['gain_share']:.1f} %)" for r in res[lay]))
    AUTO.mkdir(parents=True, exist_ok=True)
    (AUTO / "feature_importance.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
