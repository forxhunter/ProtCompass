"""Aggregate probing_results/architecture_all_floors.csv into the tables section 3 needs.

Aggregation rule is the project's own (scripts/floor_vs_encoders.py): mean over seeds within a readout family, then
the better family where a best-readout number is wanted. Section 3's table is matched-readout, so linear and mlp2 are
also reported separately. Nothing is written by this script except stdout.
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
OUT = ROOT / "probing_results"
FLOORS = ["ref_length", "ref_position", "ref_aa", "ref_seq_window", "ref_coords", "ref_backbone", "ref_global"]


def load():
    return pd.read_csv(OUT / "architecture_all_floors.csv")


def main():
    d = load()
    d["is_floor"] = d.source.str.startswith("ref_")
    m = d.groupby(["class_set", "split", "source", "readout", "is_floor"]).balanced_accuracy.mean().reset_index()

    pd.set_option("display.width", 250)
    for cs in ("common13", "full"):
        for readout in ("linear", "mlp2"):
            g = m[(m.class_set == cs) & (m.readout == readout)]
            print(f"\n=== {cs}, matched readout {readout}, balanced accuracy ===")
            for split in ("superfamily", "topology"):
                gs = g[g.split == split]
                fl = gs[gs.is_floor].set_index("source").balanced_accuracy
                en = gs[~gs.is_floor].set_index("source").balanced_accuracy
                if fl.empty or en.empty:
                    continue
                print(f"  {split:12s} best encoder {en.max():.3f} ({en.idxmax():20s}) "
                      f"best floor {fl.max():.3f} ({fl.idxmax():15s}) gap {en.max() - fl.max():+.3f}   "
                      f"| ref_global {fl.get('ref_global', np.nan):.3f} "
                      f"published-style gap {en.max() - fl.get('ref_global', np.nan):+.3f}")
                print("      floors: " + "  ".join(f"{f} {fl.get(f, np.nan):.3f}" for f in FLOORS if f in fl.index))
                print("      encoders above best floor: "
                      f"{int((en > fl.max()).sum())} of {len(en)}")

    # best-readout-family view, the rule used by floor_vs_encoders.py
    print("\n=== best readout family per source (project aggregation rule) ===")
    fam = d.copy()
    fam["family"] = np.where(fam.readout == "linear", "linear", "nonlinear")
    fm = fam.groupby(["class_set", "split", "source", "family", "is_floor"]).balanced_accuracy.mean().reset_index()
    best = fm.loc[fm.groupby(["class_set", "split", "source"]).balanced_accuracy.idxmax()]
    for cs in ("common13", "full"):
        for split in ("superfamily", "topology"):
            g = best[(best.class_set == cs) & (best.split == split)]
            if g.empty:
                continue
            fl = g[g.is_floor].set_index("source").balanced_accuracy
            en = g[~g.is_floor].set_index("source").balanced_accuracy
            if en.empty:
                print(f"  {cs:9s} {split:12s} floors only: " +
                      "  ".join(f"{f} {fl.get(f, np.nan):.3f}" for f in FLOORS if f in fl.index))
                continue
            print(f"  {cs:9s} {split:12s} best encoder {en.max():.3f} ({en.idxmax()}), "
                  f"best floor {fl.max():.3f} ({fl.idxmax()}), gap {en.max() - fl.max():+.3f}, "
                  f"encoders above floor {int((en > fl.max()).sum())} of {len(en)}")
            print("      floors: " + "  ".join(f"{f} {fl.get(f, np.nan):.3f}" for f in FLOORS if f in fl.index))

    # seed spread on the strictest cell
    print("\n=== seed spread, mlp2, three optimiser seeds (sd and range over seeds) ===")
    s = d[(d.readout == "mlp2")].groupby(["class_set", "split", "source"]).balanced_accuracy.agg(
        ["mean", "std", "min", "max", "count"])
    for cs in ("common13", "full"):
        for split in ("superfamily", "topology"):
            g = s.loc[(cs, split)].sort_values("mean", ascending=False)
            print(f"\n  -- {cs} / {split}: top 6 and all floors --")
            print(g.head(6).round(4).to_string())
            print(g.loc[[f for f in FLOORS if f in g.index]].round(4).to_string())
            print(f"  max seed sd over all sources in this cell: {g['std'].max():.4f} ({g['std'].idxmax()})")


if __name__ == "__main__":
    main()
