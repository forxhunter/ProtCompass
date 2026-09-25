"""Shuffled-3Di control, downstream scoring (PRESPEC_shuf.md step 2): ab_score.score_one unchanged, new source only.

ab_score.py is imported, not edited. Its ROOTS dict gets one entry at run time; score_one writes the per-protein test
outputs to ab_modality/predictions/<task>__saprot_shuf3di.npz. Score rows go to ab_modality/score_parts_shuf/ (never
score_parts/, which ab_score.merge globs), merged into scores_ab_long_shuf.csv and scores_ab_selected_shuf.csv.

  python ab_shuf_score.py --tasks stability ... [--threads 6]
  python ab_shuf_score.py --merge-only
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch

AB = Path((PROTCOMPASS_ROOT + "/protcompass/extract/ab_modality"))
sys.path.insert(0, str(AB))
import ab_score                                                          # noqa: E402
from ab_score import TASKS                                               # noqa: E402

KEY = "saprot_shuf3di"
ab_score.ROOTS[KEY] = AB / "embeddings"
PARTS = AB / "score_parts_shuf"
PRED = ab_score.PRED
TIMES = AB / "stage_times_shuf.csv"


def merge():
    parts = [pd.read_csv(f) for f in sorted(PARTS.glob("ab_shuf_long_*.csv"))]
    if not parts:
        return
    d = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["task", "encoder", "probe", "seed"], keep="last")
    d.to_csv(AB / "scores_ab_long_shuf.csv", index=False)
    probe_val = d.groupby(["task", "encoder", "probe"]).val_loss.mean().reset_index()
    best = probe_val.loc[probe_val.groupby(["task", "encoder"]).val_loss.idxmin(), ["task", "encoder", "probe"]]
    sel = d.merge(best, on=["task", "encoder", "probe"], how="inner")
    agg = sel.groupby(["task", "encoder", "probe"]).score.agg(["mean", "std", "size"]).reset_index()
    agg = agg.rename(columns={"mean": "score", "std": "seed_sd", "size": "n_seeds"})
    agg.to_csv(AB / "scores_ab_selected_shuf.csv", index=False)
    print(agg.round(4).to_string(index=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--merge-only", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(min(a.threads, 6))
    if a.merge_only:
        return merge()
    PARTS.mkdir(parents=True, exist_ok=True)
    for task in a.tasks:
        f = PARTS / f"ab_shuf_long_{task}__{KEY}.csv"
        if f.exists() and (PRED / f"{task}__{KEY}.npz").exists():
            print(f"{task}/{KEY}: done already", flush=True)
            continue
        t0 = time.time()
        rows = ab_score.score_one(task, KEY)
        d = pd.DataFrame(rows)
        d.to_csv(f, index=False)
        pv = d.groupby("probe").val_loss.mean()
        sel = d[d.probe == pv.idxmin()].score.mean()
        print(f"  {task:26s} {KEY:16s} selected {pv.idxmin():6s} {sel:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        new = not TIMES.exists()
        with open(TIMES, "a") as fh:
            if new:
                fh.write("stage,what,start,end,seconds,note\n")
            fh.write(f"score,{KEY}/{task},{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t0))},"
                     f"{time.strftime('%Y-%m-%d %H:%M:%S')},{time.time() - t0:.0f},\n")
    merge()


if __name__ == "__main__":
    main()
