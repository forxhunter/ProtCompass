"""Regenerate the three main-text figures from the released CSVs.

The previous Figures 2, 3 and 4 were drawn from the predecessor pipeline: Figure 2 showed IIP profiles with no floor,
Figure 3 a 26x14 downstream matrix including four tasks later found unusable, and Figure 4 the withdrawn alignment
heatmap. A caption cannot be made consistent with a superseded image, so the images are rebuilt here.

Design choices that matter for reading them:
  - Encoders are grouped by the MEASURED input audit, never by a declared modality.
  - The downstream heatmap uses one sequential hue per column, because the ten tasks use four different metrics and
    a shared colour scale across them would be meaningless; each column is scaled within itself and the metric named.
  - The floor comparison is drawn as a paired bar per probe rather than a ranking, because the claim is about the
    sign of each pair, not about an ordering of encoders.

  python make_main_figures.py
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import glob
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
OUT = Path((PROTCOMPASS_ROOT + "/protcompass/paper/figures"))
RES = ROOT / "probing_results"

FLOOR = "#4C6EF5"      # non-learned floor
ENC = "#E8590C"        # encoder
SEQ = "#0B7285"        # sequence-only, by audit
CRD = "#C2255C"        # reads coordinates, by audit
GREY = "#868E96"

UND = {"rsa", "rsa_bins", "bfactor", "architecture_label"}
TAUT = {"contact_density", "contact_order_len_resid"}
LAB = {"phi_circular": "phi", "psi_circular": "psi", "local_curvature": "curvature", "dssp_3": "DSSP-3",
       "dssp_8": "DSSP-8", "core_surface": "core/surface", "contact_density": "contact density",
       "distance_bins": "CA distance bins", "contact_order_len_resid": "contact order",
       "radius_of_gyration_len_resid": "radius of gyration", "asphericity_len_resid": "asphericity",
       "rsa": "RSA", "rsa_bins": "RSA bins", "bfactor": "B-factor", "architecture_label": "fold architecture"}
TASK = {"ec_classification": "EC\n(acc)", "go_bp": "GO-BP\n(F-max)", "go_mf": "GO-MF\n(F-max)",
        "remote_homology": "homology\n(acc)", "subcellular_localization": "localisation\n(acc)",
        "solubility": "solubility\n(AUROC)", "stability": "stability\n(rho)",
        "fluorescence": "fluorescence\n(rho)", "ppi_affinity": "affinity\n(rho)",
        "membrane_soluble": "membrane\n(AUROC)"}


def audit_groups():
    a = pd.read_csv(ROOT / "extract" / "encoder_input_audit.csv").drop_duplicates("encoder").set_index("encoder")
    return a["uses_coordinates"].astype(bool)


def fig2():
    """Probe scores against the floors, and the split the input audit produces."""
    d = pd.read_csv(RES / "floor_vs_encoders.csv").set_index("probe")
    d = d.reindex(sorted(d.index, key=lambda p: d.loc[p, "gap"]))
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.35, 1]})

    y = np.arange(len(d))[::-1]
    ax[0].barh(y + 0.19, d["best floor"], height=0.36, color=FLOOR, label="best non-learned floor")
    ax[0].barh(y - 0.19, d["max encoder"], height=0.36, color=ENC, label="best of 33 encoders")
    for k, p in enumerate(d.index):
        style = "italic" if p in TAUT else "normal"
        weight = "bold" if p in UND else "normal"
        ax[0].text(-0.02, y[k], LAB.get(p, p), ha="right", va="center", fontsize=9,
                   style=style, fontweight=weight)
        ax[0].text(max(d.loc[p, "best floor"], d.loc[p, "max encoder"]) + 0.015, y[k],
                   f"{d.loc[p, 'gap']:+.3f}", va="center", fontsize=7.5,
                   color=ENC if d.loc[p, "gap"] > 0 else FLOOR)
    ax[0].set_yticks([]); ax[0].set_xlim(0, 1.16); ax[0].set_ylim(-1.0, len(d) - 0.1)
    ax[0].set_xlabel("probe score (metric follows the probe)")
    ax[0].legend(loc="lower right", frameon=False, fontsize=9)
    ax[0].spines[["top", "right", "left"]].set_visible(False)
    ax[0].set_title("A", loc="left", fontsize=12, fontweight="bold", pad=8)


    # panel B: per encoder, margin over the floor built from its own measured inputs
    import subprocess, sys, re
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "emit_conditional_floor_table.py")],
                         capture_output=True, text=True).stdout
    rows = [l for l in out.splitlines() if l.startswith("| ") and " of " in l]
    seq_below = crd_below = seq_tot = crd_tot = 0
    m = re.search(r"\*\*(\d+) of (\d+)\*\* sequence-only", out); seq_below, seq_tot = int(m.group(1)), int(m.group(2))
    m = re.search(r"\*\*(\d+) of (\d+)\*\* coordinate-encoder", out); crd_below, crd_tot = int(m.group(1)), int(m.group(2))

    bars = [("coordinate\nencoders", crd_tot - crd_below, crd_below, CRD),
            ("sequence-only\nencoders", seq_tot - seq_below, seq_below, SEQ)]
    for i, (name, above, below, col) in enumerate(bars):
        tot = above + below
        ax[1].barh(i, above / tot, color=col, height=0.42)
        ax[1].barh(i, below / tot, left=above / tot, color="#DEE2E6", height=0.42)
        ax[1].text(0.012, i + 0.30, f"{above} / {tot}", fontsize=10.5, color=col, fontweight="bold")
        if below:
            ax[1].text(0.988, i + 0.30, f"{below}", fontsize=10.5, color=GREY, ha="right")
    ax[1].set_yticks([0, 1]); ax[1].set_yticklabels([b[0] for b in bars], fontsize=9)
    ax[1].set_xlim(0, 1); ax[1].set_ylim(-0.6, 1.75)
    ax[1].set_xlabel("fraction of cells above own floor")
    ax[1].spines[["top", "right", "left"]].set_visible(False)
    ax[1].set_title("B", loc="left", fontsize=12, fontweight="bold", pad=8)


    fig.tight_layout()
    fig.savefig(OUT / "Figure2.png", dpi=200, bbox_inches="tight")
    print("wrote Figure2.png")


def fig3():
    """Downstream scores, grouped by the measured audit, each column scaled within its own metric."""
    w = pd.read_csv(ROOT / "extract" / "scores2_wide.csv", index_col=0)
    w = w.drop(index="ref_composition", errors="ignore")  # the non-learned reference is a table row, not an encoder
    sd = pd.read_csv(ROOT / "extract" / "scores2_selected.csv").pivot(index="encoder", columns="task",
                                                                     values="seed_sd")
    uses = audit_groups().reindex(w.index).fillna(True).astype(bool)
    order = [c for c in ["ec_classification", "go_mf", "go_bp", "subcellular_localization", "remote_homology",
                         "solubility", "membrane_soluble", "stability", "fluorescence", "ppi_affinity"]
             if c in w.columns]
    w, sd = w[order], sd.reindex(columns=order)
    rank = w.rank(ascending=False).mean(axis=1)
    idx = list(rank[~uses].sort_values().index) + list(rank[uses].sort_values().index)
    w, sd = w.loc[idx], sd.reindex(idx)

    norm = (w - w.min()) / (w.max() - w.min())
    # landscape: tasks as rows, encoders as columns, so the figure is about 2 inches tall at text width
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.imshow(norm.values.T, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    for i in range(len(w)):
        for j in range(len(order)):
            v = w.iloc[i, j]
            if np.isfinite(v):
                ax.text(i, j, f"{v:.2f}", ha="center", va="center", fontsize=5.4, rotation=90,
                        color="white" if norm.iloc[i, j] > 0.6 else "#212529")
            else:
                ax.text(i, j, "n/a", ha="center", va="center", fontsize=5, rotation=90, color=GREY)
    n_seq = int((~uses).sum())
    ax.axvline(n_seq - 0.5, color="black", lw=1.6)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([TASK.get(c, c).replace("\n", " ") for c in order], fontsize=8)
    ax.set_xticks(range(len(w)))
    ax.set_xticklabels(w.index, fontsize=7.2, rotation=60, ha="right")
    for t, e in zip(ax.get_xticklabels(), w.index):
        t.set_color(SEQ if not uses[e] else CRD)

    fig.tight_layout()
    fig.savefig(OUT / "Figure3.png", dpi=200, bbox_inches="tight")
    print("wrote Figure3.png")


def fig4():
    """Causal demand read against the detection bound."""
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from exclusion_bounds import load, mde, TASK_LABEL
    v2 = load("erasure_v2*.csv")
    ctrl = load("erasure_control*.csv")
    cond = v2[(v2.arm == "conditional") & v2.erasure_ok.fillna(False)]
    pc = ctrl[ctrl.check == "positive control"]
    reach = pc.pivot_table(index="task", columns="rank", values="effect", aggfunc="mean").max(axis=1)

    rows = []
    for (t, c), g in cond.groupby(["task", "concept"]):
        rows.append(dict(task=t, concept=c, n=len(g), eff=g.effect.mean(), sd=g.effect.std()))
    d = pd.DataFrame(rows)
    d["mde"] = [mde(r.sd, r.n) for r in d.itertuples()]
    keep = [t for t in d.task.unique() if t in reach.index and np.isfinite(reach[t]) and reach[t] > 0
            and np.isfinite(d[d.task == t].mde.max())]
    if not keep:
        print("fig4: not enough coverage yet"); return

    fig, ax = plt.subplots(figsize=(10, 0.48 * len(keep) + 1.4))  # compact rows for the page budget
    labels = []
    for i, t in enumerate(sorted(keep, key=lambda x: -reach[x])):
        g = d[d.task == t]
        bound, r = g.mde.max(), reach[t]
        ax.barh(i, r, height=0.62, color="#E9ECEF", zorder=1)
        ax.barh(i, bound, height=0.62, color="#CED4DA", zorder=2)
        for row in g.itertuples():
            col = ENC if row.eff > bound else GREY
            ax.scatter(row.eff, i, s=34, color=col, zorder=4, clip_on=False)
        top = g.loc[g.eff.idxmax()]
        if top.eff > bound:
            ax.annotate(top.concept, (top.eff, i), textcoords="offset points", xytext=(7, 4),
                        fontsize=7.5, color=ENC)
        ax.text(r + 0.008, i, f"{r:+.2f}", va="center", fontsize=8, color=GREY)
        labels.append(f"{TASK_LABEL.get(t, t)} (n={int(g.n.max())})")
    # task names as tick labels rather than text inside the axes: on the two tasks with negative effects the points
    # reach past x=0 and were drawn over the name.
    ax.set_yticks(range(len(keep))); ax.set_yticklabels(labels, fontsize=9)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.7, len(keep) - 0.3)
    ax.set_xlabel("conditional effect vs sham null")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(handles=[Patch(color="#CED4DA", label="below detection bound"),
                       Patch(color="#E9ECEF", label="reachable range"),
                       plt.Line2D([], [], marker="o", ls="", color=ENC, label="above bound")],
              loc="lower right", frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "Figure4.png", dpi=200, bbox_inches="tight")
    print("wrote Figure4.png")


# Figures 2 and 3 in the manuscript are assembled by hand from the original-style panels of
# make_panels_original_style.py. The versions this script draws are a different design and are written to
# figures/regenerated_v2/ so that a run of this script cannot overwrite the assembled ones again.
ALT = OUT / "regenerated_v2"

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="also redraw the alternative Figures 2 and 3")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ALT.mkdir(parents=True, exist_ok=True)
    if a.all:
        OUT = ALT          # the alternative designs never land on the assembled figures
        fig2(); fig3()
        OUT = Path((PROTCOMPASS_ROOT + "/protcompass/paper/figures"))
    fig4()
