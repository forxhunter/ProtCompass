"""Every number of Appendix sec:appendix_uncertainty, recomputed from the released per-seed tables.

Probing: the nonlinear readout (mlp2, pair_mlp) is the seeded one; a cell is (source, probe) over the 33 encoders and
the ten released floors, on the paper's headline metric (Pearson, else circular cosine, else balanced accuracy).
Downstream: the selected probe of each of the 329 encoder cells (extract/scores2_selected.csv).

  python uncertainty_stats.py
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
sys.path.insert(0, str(ROOT / "scripts"))
from panel import panel                                                 # noqa: E402

FLOORS = ["ref_length", "ref_position", "ref_aa", "ref_seq_window", "ref_seq_expert", "ref_seq_global",
          "ref_coords", "ref_backbone", "ref_global", "ref_pairhist"]


def probing():
    d = pd.read_csv(ROOT / "probing_results" / "scores_long.csv")
    d = d[d.readout.isin(["mlp2", "pair_mlp"]) & d.source.isin(panel() + FLOORS)].copy()
    d["score"] = d["pearson"].fillna(d["circ_cos"]).fillna(d["balanced_accuracy"])
    g = d.groupby(["source", "probe"]).score
    sd = g.std()[g.count() == 3]
    per = sd.groupby(level="probe").median()
    top = sd.idxmax()
    return {"cells": int(len(sd)), "median": float(sd.median()), "mean": float(sd.mean()), "max": float(sd.max()),
            "max_cell": list(top), "per_probe_median": {k: float(v) for k, v in per.items()}}


def downstream():
    s = pd.read_csv(ROOT / "extract" / "scores2_selected.csv")
    s = s[s.encoder.isin(panel())]
    sd = s.seed_sd
    per = s.groupby("task").seed_sd.agg(["median", "max", "mean"])
    ppi = s[s.task == "ppi_affinity"].sort_values("seed_sd", ascending=False)
    return {"cells": int(len(s)), "median": float(sd.median()), "mean": float(sd.mean()), "min": float(sd.min()),
            "max": float(sd.max()), "below_0.05": int((sd < 0.05).sum()), "above_0.1": int((sd > 0.1).sum()),
            "above_0.1_tasks": sorted(s[sd > 0.1].task.unique().tolist()),
            "per_task": {t: {k: float(v) for k, v in r.items()} for t, r in per.iterrows()},
            "ppi_above_0.1": int((ppi.seed_sd > 0.1).sum()), "ppi_n": int(len(ppi)),
            "ppi_top5": [(r.encoder, float(r.seed_sd)) for r in ppi.head(5).itertuples()],
            "other_nine_mean": float(s[s.task != "ppi_affinity"].seed_sd.mean())}


if __name__ == "__main__":
    out = {"probing": probing(), "downstream": downstream()}
    (ROOT / "probing_results" / "uncertainty_stats.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
