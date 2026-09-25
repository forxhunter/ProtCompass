"""Main-text Figure 2 of the benchmark manuscript, drawn from the symmetric rerun so that it follows the protocol the
text states: every readout fitted once on the training split, the readout family of every source (encoder or floor)
chosen on validation, the test score read once (symmetric/results_superfamily/by_source).

Layout and palette are those of make_figures_paper.fig2. Probes: the fifteen of the floor comparison, less contact
density and contact order, which a floor contains, grouped into the nine IIPs they belong to; sequence recovery (I10)
is not drawn, because the one-hot floor contains its label. Floor strip: the best of the six sequence floors, and the
best of all ten written floors on the full stored atoms, per IIP (IIP mean of each floor, then the maximum).

  python make_figure2_symmetric.py
Output: <manuscript>/figures/Figure2.png, and figure2_symmetric_values.csv next to this repository's symmetric/ data.
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import make_panels_original_style as P   # noqa: E402

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
SYM = ROOT / "symmetric" / "results_superfamily" / "by_source"
OUT = Path((PROTCOMPASS_ROOT + "/protcompass/paper/figures"))
TEAL, GROUP_COLOR = P.TEAL, P.GROUP_COLOR
plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"]})

IIP_OF = {"phi_circular": "I1", "psi_circular": "I1", "local_curvature": "I2", "dssp_3": "I3", "dssp_8": "I3",
          "rsa": "I4", "rsa_bins": "I4", "bfactor": "I5", "core_surface": "I6", "distance_bins": "I7",
          "radius_of_gyration_len_resid": "I8", "asphericity_len_resid": "I8", "architecture_label": "I9"}
IIPS = [f"I{i}" for i in range(1, 10)]
SEQ_FLOORS = ["ref_length", "ref_position", "ref_aa", "ref_seq_window", "ref_seq_expert", "ref_seq_global"]
ALL_FLOORS = SEQ_FLOORS + ["ref_coords", "ref_backbone", "ref_global", "ref_pairhist"]


def matrix():
    d = pd.concat([pd.read_csv(f) for f in glob.glob(str(SYM / "*.csv"))], ignore_index=True)
    d = d[d.probe.isin(IIP_OF) & ~d.source.str.endswith("_vframe")]
    d["family"] = np.where(d.readout == "linear", "linear", "nonlinear")
    m = d.groupby(["source", "probe", "family", "split"]).headline.mean().unstack("split").reset_index()
    pick = m.loc[m.groupby(["source", "probe"]).val.idxmax()]          # readout family chosen on validation
    pick = pick.assign(iip=pick.probe.map(IIP_OF))
    fam = pick.groupby(["source", "iip"]).test.mean().unstack("iip").reindex(columns=IIPS)
    return fam, pick


def main():
    fam, pick = matrix()
    uses = P.audit()
    encs = [s for s in fam.index if not s.startswith("ref_") and s in uses.index]
    assert len(encs) == 33, len(encs)
    groups = {k: sorted([e for e in encs if P.declared(e) == k], key=lambda e: -fam.loc[e].mean())
              for k in ("seq", "struct", "multi")}
    seq_floor = fam.loc[SEQ_FLOORS].max()
    all_floor = fam.loc[ALL_FLOORS].max()
    order = groups["seq"] + groups["struct"] + groups["multi"]
    fig = plt.figure(figsize=(6.6, 3.2))
    ax = fig.add_axes([0.14, 0.45, 0.76, 0.51])
    im = ax.imshow(fam.loc[order].values.T, cmap=TEAL, aspect="auto", vmin=0.05, vmax=0.95, interpolation="nearest")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([P.DISP.get(e, e) for e in order], rotation=60, fontsize=5.6, ha="right")
    for t, e in zip(ax.get_xticklabels(), order):
        t.set_color(GROUP_COLOR[P.declared(e)])
    ax.set_yticks(np.arange(len(IIPS)))
    ax.set_yticklabels([P.IIP_SHORT[i].replace("\n", " ") for i in IIPS], fontsize=6.5)
    ax.axvline(len(groups["seq"]) - 0.5, color="white", linewidth=2.2)
    ax.axvline(len(groups["seq"]) + len(groups["struct"]) - 0.5, color="white", linewidth=2.2)
    cax = fig.add_axes([0.915, 0.50, 0.012, 0.42])
    cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=6); cb.set_label("probe score", fontsize=6.5)
    ax.text(-0.17, 1.0, "A", transform=ax.transAxes, fontsize=11, fontweight="bold", ha="right", va="bottom")
    ax2 = fig.add_axes([0.14, 0.09, 0.76, 0.12])
    strip = np.vstack([seq_floor.values, all_floor.values])
    ax2.imshow(strip, cmap=TEAL, aspect="auto", vmin=0.05, vmax=0.95)
    ax2.set_yticks([0, 1]); ax2.set_yticklabels(["best sequence floor", "best of all ten floors"], fontsize=6.5)
    ax2.set_xticks(np.arange(len(IIPS)))
    ax2.set_xticklabels([P.IIP_SHORT[i].replace("\n", " ") for i in IIPS], fontsize=6, rotation=20, ha="right")
    for j in range(len(IIPS)):
        for i in range(2):
            v = strip[i, j]
            ax2.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5, color="white" if v > 0.6 else "#222")
    ax2.text(-0.17, 0.9, "B", transform=ax2.transAxes, fontsize=11, fontweight="bold", ha="right", va="bottom")
    fig.savefig(OUT / "Figure2.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    vals = fam.loc[order + SEQ_FLOORS + ["ref_coords", "ref_backbone", "ref_global", "ref_pairhist"]]
    vals.to_csv(ROOT / "symmetric" / "figure2_symmetric_values.csv")
    # the caption's statement, checked on the drawn values
    seq_above = all((fam.loc[groups["seq"]].min() > seq_floor).values)
    print("wrote Figure2.png; declared-sequence encoders above the best sequence floor on every IIP:", seq_above)
    print(pd.DataFrame({"seq_floor": seq_floor, "all_floor": all_floor,
                        "seq_min": fam.loc[groups["seq"]].min(), "crd_max": fam.loc[groups["struct"]].max()}).round(3))


if __name__ == "__main__":
    main()
