"""CPU checks of the built HPC bundle (D:/m1_tmp/hpc/m1_hpc_bundle) before zipping is trusted. Run with a torch environment
(D:/m1_tmp/torch_venv), never the project .venv. No training beyond a few CPU steps.

1 adapter round-trip of one match (windows -> slot-space targets == the match's targets; flip + unflip == original)
2 CPU forward + backward on 2 windows (pretrained weights, ball input, travelling/loose head)
3 every flag on CPU (train a few steps per flag set; resume; --all; predict bf16-path and --fp32; cycle score / ablation /
  decide / summary on the result)
4 the HPC scorer reproduces the frozen C3's fold-0 numbers exactly (vs the repository's results file)
5 run_cycle.sh --dry-run prints the 12-job chain
6 the shell scripts use no blocked command (df, head)
7 bundle zip < 2 GB

    D:/m1_tmp/torch_venv/Scripts/python.exe scripts/m1/hpc_pathcrf/local_checks.py
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

BUNDLE = Path("D:/m1_tmp/hpc/m1_hpc_bundle")
ZIP = Path("D:/m1_tmp/hpc/m1_hpc_bundle.zip")
CHECK = Path("D:/m1_tmp/hpc/check_root")
CODE, DATA = BUNDLE / "code", BUNDLE / "data"
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE))
os.environ.update(M1_ROOT=str(CHECK), M1_DATA=str(DATA), PYTHONPATH=str(CODE), PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
RESULTS = {}


def ok(name, cond, detail=""):
    RESULTS[name] = dict(ok=bool(cond), detail=detail)
    print(f"[{'OK' if cond else 'FAIL'}] {name}  {detail}", flush=True)


def run(args, **kw):
    t = time.time()
    r = subprocess.run([sys.executable, *args], capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=CODE, **kw)
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:])
    return r, time.time() - t


def main() -> int:
    import numpy as np
    import torch
    from pathcrf_adapter import adapter as A
    from pathcrf_adapter.model import M1PathCRF, losses
    quick = "--from-cycle" in sys.argv          # re-run only the cycle-tool checks and 4-7 on the existing flag_fast run
    if not quick:
        shutil.rmtree(CHECK, ignore_errors=True)
    CHECK.mkdir(parents=True, exist_ok=True)
    M = A.Match(A.load_match(DATA / "matches" / "3813.npz"))
    if quick:
        prev = json.loads((CHECK / "local_checks.json").read_text())
        RESULTS.update({k: v for k, v in prev.items() if not k.startswith(("3 cycle guard", "4 ", "5 ", "6 ", "7 "))})
        for k in ("runs/CYCLE.json", "runs/gpu_ledger.json", "runs/guard_stop.json", "home"):
            p_ = CHECK / k
            shutil.rmtree(p_, ignore_errors=True) if p_.is_dir() else p_.unlink(missing_ok=True)
        for p_ in (CHECK / "runs").glob("2026-*"):
            shutil.rmtree(p_)
        rd = CHECK / "runs" / "flag_fast"
        return finish(rd, A, np)
    # 1 adapter round-trip
    M = A.Match(A.load_match(DATA / "matches" / "3813.npz"))
    bad = flips = 0
    n_t = 0
    for s in M.window_starts(A.WIN):
        w = M.window(s)
        ns = w["node_slot"]
        for i in np.flatnonzero(w["mask"]):
            k = M.kind[s + i]
            a_ = ns[w["src"][i]]
            b_ = ns[w["dst"][i]] if k != A.K_OUT else w["dst"][i] - 2 * A.TEAM
            bad += int(a_ != M.a[s + i] or b_ != M.b[s + i])
            n_t += 1
        if s % (A.WIN * 7) == 0:
            for fx, fy in ((1, 0), (0, 1), (1, 1)):
                wf = M.window(s, bool(fx), bool(fy))
                same = all(ns[w["src"][i]] == wf["node_slot"][wf["src"][i]] for i in np.flatnonzero(w["mask"]))
                pos = np.allclose(np.sort(wf["x"][:, :22, 2], 1), np.sort((105 - w["x"][:, :22, 2]) if fx else w["x"][:, :22, 2], 1), atol=1e-4)
                flips += int(not (same and pos and (wf["mask"] == w["mask"]).all()))
    lost = int(M.kind.astype(bool).sum() - n_t)
    ok("1 adapter round-trip (match 3813)", bad == 0 and flips == 0,
       f"{n_t} target steps rebuilt exactly; {lost} steps not representable (player outside the window's 22 nodes); flip mismatches {flips}")
    # 2 forward + backward
    ws = M.window_starts(15)
    w2 = [M.window(ws[40]), M.window(ws[41], True, False)]
    b = {k: torch.as_tensor(np.stack([x[k] for x in w2])) for k in ("x", "pmask", "src", "dst", "mask", "tl")}
    m = M1PathCRF(ball=True, pretrained=True)
    loss, info = losses(m, b, 0.3)
    loss.backward()
    g_head = m.head.mlp[0].weight.grad.abs().sum().item()
    g_ball = m.net.macro_fpe_enc.embed[0].weight.grad[:, 8:11].abs().sum().item()
    ok("2 CPU forward + backward on 2 windows", np.isfinite(info["loss"]) and g_head > 0 and g_ball > 0,
       f"loss {info['loss']:.3f} (emit {info['emit']:.3f}, crf {info['crf']:.3f}, head {info['tl']:.3f}); head grad {g_head:.3g}; "
       f"ball-input grad {g_ball:.3g}; widened {len(m.load_report['widened'])} layers, missing {len(m.load_report['missing'])}")
    # 3 flags
    tiny = ["--data", str(DATA), "--matches", "3820", "--batch", "2", "--workers", "0", "--max-steps", "3"]
    sets = {"fast": [], "scratch": ["--from-scratch"], "keepdead": ["--keep-dead-windows"], "stride1": ["--stride", "1"],
            "fp32": ["--fp32"], "noball": ["--no-ball"], "epochs": ["--max-epochs", "1", "--patience", "1"]}
    for name, fl in sets.items():
        rd = CHECK / "runs" / f"flag_{name}"
        r, dt = run(["-m", "m1pc.train", "--run-dir", str(rd), "--fold", "0", *tiny, *fl])
        cfg = json.loads((rd / "config.json").read_text()) if (rd / "config.json").exists() else {}
        ok(f"3 flag {name} {' '.join(fl)}", r.returncode == 0 and (rd / "checkpoints" / "last.pt").exists() and (rd / "NOTES.md").exists(),
           f"{dt:.0f}s; variant {cfg.get('variant')}; windows used {cfg.get('n_windows')} of {cfg.get('n_windows_all')}")
    rd = CHECK / "runs" / "flag_fast"
    r, dt = run(["-m", "m1pc.train", "--run-dir", str(rd), "--fold", "0", *tiny])
    ok("3 auto-resume from last.pt", r.returncode == 0 and "resumed from" in r.stdout, r.stdout.strip().splitlines()[0][:120] if r.stdout else "")
    r, dt = run(["-m", "m1pc.train", "--run-dir", str(CHECK / "runs" / "flag_all"), "--all", "--max-epochs", "1", *tiny])
    ok("3 --all (no held-out fold)", r.returncode == 0, f"{dt:.0f}s")
    # predictions for the 6 fold-0 matches (coarse stride on CPU), bf16 path and --fp32, then the cycle tools
    r, dt = run(["-m", "m1pc.predict", "--data", str(DATA), "--run-dir", str(rd), "--pred-stride", "10", "--checkpoint", "last.pt"])
    p = np.load(rd / "preds" / "3813.npz")
    ok("3 predict (held-out fold-0 matches)", r.returncode == 0 and len(list((rd / "preds").glob("*.npz"))) == 6,
       f"{dt:.0f}s; keys {sorted(p.files)}")
    r, dt = run(["-m", "m1pc.predict", "--data", str(DATA), "--run-dir", str(rd), "--pred-stride", "10", "--checkpoint", "last.pt",
                 "--fp32", "--matches", "3813", "--out-subdir", "preds_fp32"])
    ok("3 predict --fp32", r.returncode == 0, f"{dt:.0f}s")
    r, _ = run(["-m", "m1pc.cycle", "score", str(rd)])
    sc = json.loads((rd / "score.json").read_text()) if (rd / "score.json").exists() else {}
    ok("3 cycle score (C3 + 3 combine variants)", r.returncode == 0 and set(sc) == {"C3", "pathcrf", "combo_c3tl", "combo_who"},
       "; ".join(f"{k} acc {v['acc']:.3f} spell {v['spell']:.3f}" for k, v in sc.items()) + " (3 CPU steps: not a real score)")
    return finish(rd, A, np)


def finish(rd, A, np):
    # cycle bookkeeping end to end on fake run folders
    env = dict(os.environ)
    run(["-m", "m1pc.cycle", "init", "--date", "2026-11-12"])
    cyc = json.loads((CHECK / "runs" / "CYCLE.json").read_text())
    for key in ("main", "ref", "a1", "a2", "a3"):
        shutil.copytree(rd, CHECK / "runs" / cyc["runs"][key])
    run(["-m", "m1pc.cycle", "score-fold0"])
    for key in ("ref", "a1", "a2", "a3"):
        run(["-m", "m1pc.cycle", "score", str(CHECK / "runs" / cyc["runs"][key])])
    run(["-m", "m1pc.cycle", "ledger", "train_fold0", "1", "3600", "1"])
    g1, _ = run(["-m", "m1pc.cycle", "guard", "x", "8"])
    run(["-m", "m1pc.cycle", "ledger", "big", "2", str(25 * 3600), "1"])
    g2, _ = run(["-m", "m1pc.cycle", "guard", "y", "8"])
    ra, _ = run(["-m", "m1pc.cycle", "ablation"])
    rdc, _ = run(["-m", "m1pc.cycle", "decide"])
    rs, _ = run(["-m", "m1pc.cycle", "summary"])
    rsp, _ = run(["-m", "m1pc.cycle", "space"])
    ok("3 cycle guard / ablation / decide / summary / space", g1.stdout.strip() == "OK" and g2.stdout.strip() == "SKIP" and
       all(x.returncode == 0 for x in (ra, rdc, rs, rsp)) and (CHECK / "runs" / "CYCLE_SUMMARY.md").exists(),
       f"guard {g1.stdout.strip()} then {g2.stdout.strip()} at 26 h used; decision: {json.loads((CHECK / 'runs' / 'decision.json').read_text())['why'][:80]}")
    # 4 scorer reproduces the frozen C3 fold-0 numbers
    sys.path.insert(0, str(REPO / "scripts" / "m1"))
    from pathcrf_adapter.evaluate import score_match, summarize
    sc = [score_match(A.load_match(DATA / "matches" / f"{m_}.npz"), *(np.load(DATA / "c3_fold0" / f"{m_}.npz")[k] for k in ("state", "player", "team")))
          for m_ in A.folds(DATA)[0]]
    mine = summarize(sc)
    R = {}
    for f in (REPO / "data/processed/m1_possession/c3v4").glob("results_*.json"):
        R.update(json.loads(f.read_text()))
    f0 = A.folds(DATA)[0]
    ref = {k: v for k, v in R["base_pt"]["res"].items() if int(k.split(":")[0]) in f0}
    import numpy as _np
    from pathcrf_adapter.m1score import merge, merge_had, merge_recall
    conf = _np.asarray(merge(list(ref.values()))["confusion"], float)
    refv = dict(acc=_np.trace(conf) / conf.sum(), spell=merge_had([v["had"] for v in ref.values()])["spell_f1"],
                spell_gap=merge_had([v["had_gap"] for v in ref.values()])["spell_f1"], team=merge(list(ref.values()))["team_acc"],
                loose_recall=conf[3, 3] / conf[3].sum(), rec_lt2=merge_recall([v["rec_len"] for v in ref.values()])["< 2 s"]["recall"])
    diff = max(abs(mine[k] - v) for k, v in refv.items())
    ok("4 HPC scorer == frozen C3 fold-0 numbers", diff == 0.0, f"acc {mine['acc']:.4f}, spell {mine['spell']:.4f}, max diff {diff:.1e}")
    # 5 dry run
    fake_home = CHECK / "home"
    (fake_home / "projects" / "m1-possession").mkdir(parents=True)
    bash = shutil.which("bash") or "C:/Program Files/Git/bin/bash.exe"
    env5 = dict(os.environ, HOME=str(fake_home), M1_BOOT_PY=sys.executable, M1_DRY_DATE="2026-11-12", M1_ROOT=str(fake_home / "projects" / "m1-possession"))
    r = subprocess.run([bash, "run_cycle.sh", "--dry-run"], cwd=CODE, capture_output=True, text=True, env=env5)
    lines = [x for x in r.stdout.splitlines() if x.startswith("[")]
    order = [re.match(r"\[\d+\] (\S+)", x).group(1) for x in lines]
    deps_ok = all(f"afterok:DRY{i}" in lines[i] for i in range(1, len(lines))) and "dependency" not in lines[0]
    ok("5 run_cycle.sh --dry-run", r.returncode == 0 and len(lines) == 12 and deps_ok, " -> ".join(order))
    (CHECK / "dry_run.txt").write_text(r.stdout + r.stderr)
    r2 = subprocess.run([bash, "run_cycle.sh", "--dry-run", "--from", "score_fold0"], cwd=CODE, capture_output=True, text=True, env=env5)
    l2 = [x for x in r2.stdout.splitlines() if x.startswith("[")]
    ok("5 run_cycle.sh --from score_fold0 --dry-run", r2.returncode == 0 and len(l2) == 8 and "dependency" not in l2[0], f"{len(l2)} jobs")
    # 6 blocked commands
    bad = []
    for sh in list(CODE.rglob("*.sh")) + [CODE / "README_HPC.md"]:
        in_block = False
        for i, line in enumerate(sh.read_text(encoding="utf-8").splitlines(), 1):
            if sh.suffix == ".md":                      # README: only the shell code blocks are commands
                if line.startswith("```"):
                    in_block = not in_block
                    continue
                if not in_block:
                    continue
            code = line.split("#", 1)[0]
            if re.search(r"(^|[\s|;&(`$])(df|head)(\s|$)", code):
                bad.append(f"{sh.name}:{i}")
    ok("6 no blocked commands (df, head) in scripts / README", not bad, ", ".join(bad) or "none")
    # 7 size
    size = ZIP.stat().st_size
    ok("7 bundle < 2 GB", size < 2 * 2**30, f"{size / 2**20:.1f} MB")
    (CHECK / "local_checks.json").write_text(json.dumps(RESULTS, indent=1))
    return 0 if all(v["ok"] for v in RESULTS.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
