"""Sub-panels for Figures 2 and 3 in the style of the original figure scripts, from the released data.

The original figures were assembled by hand from panels produced by ProtCompass/scripts/visualize_paper_v12.py
(probing: teal heatmap + one radar per group) and visualize_downstream_v1.py (downstream: rose heatmap + radars).
Those scripts read the predecessor pipeline and group encoders by declared modality. This script keeps their
colour maps, radar geometry and font sizes, and changes only what the results changed:

  - encoders are grouped by the measured input audit (uses_coordinates), two groups instead of three;
  - the probing radars carry the applicable non-learned floor as a grey filled polygon, which is the point of
    section 1;
  - the downstream heatmap covers the nine scored tasks, three-seed means.

No titles and no sentences are drawn; everything explanatory belongs in the caption.

  python make_panels_original_style.py
Writes manuscript figures/panels_v2/*.{png,pdf}.
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import glob
from math import pi
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
RES = ROOT / "probing_results"
OUT = Path((PROTCOMPASS_ROOT + "/protcompass/paper/figures/panels_v2"))
OUT.mkdir(parents=True, exist_ok=True)

# ---- style constants copied from the original scripts ---------------------------------------------------------
TEAL = LinearSegmentedColormap.from_list("paper_teal", ["#f2f9f9", "#aed9d9", "#5eb5b5", "#2b8c8c", "#144d4d"], N=256)
ROSE = LinearSegmentedColormap.from_list("paper_rose", ["#fdf0f0", "#f0c4c4", "#d4777a", "#a83a3e", "#6b1a1d"], N=256)
SEQ_PALETTE = ["#B0E0E6", "#87CEEB", "#5bacd4", "#4682B4", "#4169E1", "#0000CD", "#00008B", "#1a4a8a",
               "#123a6e", "#0d2c55"]
CRD_PALETTE = ["#FFF4D6", "#FFEAB3", "#FFE0A3", "#FFD580", "#FFCC66", "#FFC04D", "#FFB347", "#FFA500", "#F4A460",
               "#F39C12", "#E67E22", "#D2691E", "#CD7F32", "#B8621B", "#A0522D", "#8B2500", "#7a2000", "#6a1c00",
               "#5a1800", "#4a1400", "#3a1000", "#2a0c00", "#1a0800"]
MULTI_PALETTE = ["#c9a0e0", "#b07fd4", "#9966bb", "#7c4fa0", "#4d2475"]
GROUP_COLOR = {"seq": "#1a4a8a", "struct": "#D2691E", "multi": "#7c4fa0"}
# Declared modality, as in the original manuscript. This is the presentation grouping and it is never changed by the
# audit; the audit only decides which floor each encoder is held to (a multimodal encoder measured not to read
# coordinates, ProstT5, is held to the sequence floor).
DECLARED = {**{e: "seq" for e in ["esm1b", "prottrans_t5_xl_u50", "protbert_bfd", "esm2_8m", "esm2_35m", "esm2_150m",
                                  "esm2_650m", "esm2_3b", "ankh_base"]},
            **{e: "multi" for e in ["prostt5", "cdconv", "saprot", "esm3", "foldvision"]}}
def declared(e):
    return DECLARED.get(e, "struct")
FLOOR_COLOR = "#7f7f7f"
plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"]})

IIP_SHORT = {"I1": "Local\nConf.", "I2": "Local\nCurv.", "I3": "Sec.\nStruct.", "I4": "Solv.\nAccess.",
             "I5": "Flexib.", "I6": "Burial/\nPacking", "I7": "Contacts", "I8": "Global\nGeom.", "I9": "Fold\nArch.",
             "I10": "Sequence"}
IIPS = [f"I{i}" for i in range(1, 11)]
SEQ_FLOORS = ["ref_length", "ref_position", "ref_aa", "ref_seq_window"]
ALL_FLOORS = SEQ_FLOORS + ["ref_coords", "ref_backbone", "ref_global", "ref_pairhist"]
TASKS = ["ec_classification", "go_mf", "go_bp", "subcellular_localization", "remote_homology", "solubility",
         "stability", "fluorescence", "ppi_affinity"]
TASK_SHORT = {"ec_classification": "EC", "go_mf": "GO-MF", "go_bp": "GO-BP", "subcellular_localization": "Subcell.",
              "remote_homology": "Homology", "solubility": "Solubility", "stability": "Stability",
              "fluorescence": "Fluoresc.", "ppi_affinity": "PPI aff."}
DISP = {"esm2_650m": "ESM2-650M", "esm2_150m": "ESM2-150M", "esm2_35m": "ESM2-35M", "esm2_8m": "ESM2-8M",
        "esm2_3b": "ESM2-3B", "esm1b": "ESM-1b", "ankh_base": "Ankh", "prottrans_t5_xl_u50": "ProtT5-XL",
        "protbert_bfd": "ProtBert", "prostt5": "ProstT5", "esm_if1": "ESM-IF1", "esm3": "ESM3", "saprot": "SaProt",
        "foldseek_3di": "Foldseek", "foldvision": "FoldVision", "proteinmpnn": "ProteinMPNN",
        "pottsmpnn": "PottsMPNN", "pottsmpnn_pretrained": "PottsMPNN-pre", "pottsmpnn_ft": "PottsMPNN-ft",
        "pottsmpnn_soluble": "PottsMPNN-sol", "gearnet": "GearNet", "gearnet_pretrained": "GearNet-pre",
        "gcpnet_random": "GCPNet", "gcpnet_pretrained": "GCPNet-pre", "gvp": "GVP", "tfn": "TFN",
        "tfn_pretrained": "TFN-pre", "ipa": "IPA", "ipa_pretrained": "IPA-pre", "schnet": "SchNet",
        "dimenet": "DimeNet", "mace": "MACE", "cdconv": "CDConv"}


def audit():
    a = pd.read_csv(RES / "floor_assignment.csv").rename(columns={"Unnamed: 0": "source"}).set_index("source")
    return a.uses_coordinates.astype(bool)


def radar_chart(ax, values_list, labels, line_colors, mean_color, floors=(), ylim=1.0):
    """Copied from visualize_paper_v12.radar_chart, plus an optional floor polygon."""
    N = len(labels)
    angles = [n / float(N) * 2 * pi for n in range(N)]; angles += angles[:1]
    ax.set_theta_offset(pi / 2); ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=5)
    ax.set_ylim(0, ylim); ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=4.5, color="grey")
    ax.grid(color="grey", linestyle="--", linewidth=0.4, alpha=0.5); ax.spines["polar"].set_visible(False)
    for k, floor in enumerate(floors):
        fv = list(floor) + [floor[0]]
        ax.fill(angles, fv, alpha=0.18 if k == 0 else 0.12, color=FLOOR_COLOR, zorder=1)
        ax.plot(angles, fv, linewidth=1.0, color=FLOOR_COLOR, linestyle="-" if k == 0 else "--", zorder=2)
    for i, vals in enumerate(values_list):
        v = list(vals) + [vals[0]]
        ax.plot(angles, v, linewidth=0.6, alpha=0.4, color=line_colors[i % len(line_colors)])
        ax.fill(angles, v, alpha=0.05, color=line_colors[i % len(line_colors)])
    mu = np.nanmean(np.array(values_list, dtype=float), axis=0); mv = list(mu) + [mu[0]]
    ax.plot(angles, mv, linewidth=1.5, color=mean_color, marker="o", markersize=2.5,
            markeredgecolor="white", markeredgewidth=0.5, zorder=5)
    ax.fill(angles, mv, alpha=0.20, color=mean_color)
    for angle, lab, k in zip(angles[:-1], labels, range(N)):
        mx = max([v[k] for v in values_list] + [f[k] for f in floors])
        ax.text(angle, ylim * (1.28 if mx > 0.85 else 1.17), lab, ha="center", va="center", fontsize=5)
    ax.set_xticklabels([])


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig); print("wrote", name)


# ---- Figure 2: probing --------------------------------------------------------------------------------------
def figure2():
    d = pd.concat([pd.read_csv(f) for f in glob.glob(str(RES / "by_source" / "*.csv"))], ignore_index=True)
    d["headline"] = d["pearson"].fillna(d["circ_cos"] if "circ_cos" in d else np.nan).fillna(d["balanced_accuracy"])
    d["family"] = np.where(d.readout == "linear", "linear", "nonlinear")
    m = d.groupby(["source", "probe", "iip", "family"]).headline.mean().reset_index()
    best = m.loc[m.groupby(["source", "probe"]).headline.idxmax()]
    fam = best.groupby(["source", "iip"]).headline.mean().unstack("iip").reindex(columns=IIPS)
    uses = audit()
    encs = [s for s in fam.index if not s.startswith("ref_") and s in uses.index]
    groups = {k: sorted([e for e in encs if declared(e) == k], key=lambda e: -fam.loc[e].mean())
              for k in ("seq", "struct", "multi")}
    seq_floor = fam.loc[[f for f in SEQ_FLOORS if f in fam.index]].max()
    all_floor = fam.loc[[f for f in ALL_FLOORS if f in fam.index]].max()

    # heatmap: encoders in declared order Seq -> Struct -> Multi, white separators, tick colour by declared group
    order = groups["seq"] + groups["struct"] + groups["multi"]
    mat = fam.loc[order].values.T
    fig, ax = plt.subplots(figsize=(4.5, 4.5 * 10 / len(order) + 0.5))
    im = ax.imshow(mat, cmap=TEAL, aspect="auto", vmin=0.05, vmax=0.95, interpolation="nearest")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([DISP.get(e, e) for e in order], rotation=45, fontsize=5, ha="right")
    for t, e in zip(ax.get_xticklabels(), order):
        t.set_color(GROUP_COLOR[declared(e)])
    ax.set_yticks(np.arange(10)); ax.set_yticklabels([IIP_SHORT[i].replace("\n", " ") for i in IIPS], fontsize=5.5)
    ax.axvline(len(groups["seq"]) - 0.5, color="white", linewidth=2.5)
    ax.axvline(len(groups["seq"]) + len(groups["struct"]) - 0.5, color="white", linewidth=2.5)
    cb = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01); cb.ax.tick_params(labelsize=6)
    plt.tight_layout(); save(fig, "fig2_heatmap")

    fig, ax = plt.subplots(figsize=(4.5, 0.9))
    im = ax.imshow(np.vstack([seq_floor.values, all_floor.values]), cmap=TEAL, aspect="auto", vmin=0.05, vmax=0.95)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["sequence floor", "coordinate floor"], fontsize=5.5)
    ax.set_xticks(np.arange(10)); ax.set_xticklabels([IIP_SHORT[i].replace("\n", " ") for i in IIPS], fontsize=5, rotation=45, ha="right")
    for j in range(10):
        for i, v in enumerate([seq_floor.values[j], all_floor.values[j]]):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=4.5, color="white" if v > 0.6 else "#222")
    plt.tight_layout(); save(fig, "fig2_floor_strip")

    # radars: one per declared group, each against the floor built from the inputs that class receives: the
    # sequence floors for the sequence group, and the best of all eight descriptors (sequence and coordinate) for the
    # structure and multimodal groups, since both receive coordinates. This is the "mixed floor" for multimodal.
    pal = {"seq": SEQ_PALETTE, "struct": CRD_PALETTE, "multi": MULTI_PALETTE}
    floors = {"seq": [seq_floor], "struct": [all_floor], "multi": [all_floor]}
    for key, group in groups.items():
        vals = [fam.loc[e].fillna(0).values for e in group]
        fig, ax = plt.subplots(figsize=(2.25, 2.25), subplot_kw=dict(projection="polar"))
        radar_chart(ax, vals, [IIP_SHORT[i] for i in IIPS], pal[key], GROUP_COLOR[key],
                    floors=[f.fillna(0).values for f in floors[key]])
        plt.tight_layout(); save(fig, f"fig2_radar_{key}")
        fig, ax = plt.subplots(figsize=(2.0, 0.16 * len(group) + 0.5)); ax.axis("off")
        h = [mpatches.Patch(color=pal[key][i % len(pal[key])], label=DISP.get(e, e)) for i, e in enumerate(group)]
        h.append(mpatches.Patch(color=FLOOR_COLOR, alpha=0.4, label="non-learned floor"))
        ax.legend(handles=h, loc="center", frameon=False, fontsize=5, ncol=2)
        save(fig, f"fig2_legend_{key}")


# ---- Figure 3: downstream ----------------------------------------------------------------------------------
def figure3():
    w = pd.read_csv(ROOT / "extract" / "scores2_wide.csv", index_col=0).reindex(columns=TASKS)
    rank = w.rank(ascending=False).mean(axis=1)
    groups = {k: list(rank[[e for e in w.index if declared(e) == k]].sort_values().index)
              for k in ("seq", "struct", "multi")}
    order = groups["seq"] + groups["struct"] + groups["multi"]
    fig, ax = plt.subplots(figsize=(4.5, 4.5 * 9 / len(order) + 0.5))
    im = ax.imshow(w.loc[order].values.T, cmap=ROSE, aspect="auto", vmin=0.05, vmax=0.95, interpolation="nearest")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([DISP.get(e, e) for e in order], rotation=45, fontsize=5, ha="right")
    for t, e in zip(ax.get_xticklabels(), order):
        t.set_color(GROUP_COLOR[declared(e)])
    ax.set_yticks(np.arange(9)); ax.set_yticklabels([TASK_SHORT[t] for t in TASKS], fontsize=5.5)
    ax.axvline(len(groups["seq"]) - 0.5, color="white", linewidth=2.5)
    ax.axvline(len(groups["seq"]) + len(groups["struct"]) - 0.5, color="white", linewidth=2.5)
    for i in range(len(order)):
        for j in range(9):
            if not np.isfinite(w.loc[order[i], TASKS[j]]):
                ax.text(i, j, "n/a", ha="center", va="center", fontsize=4, color="#888")
    cb = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01); cb.ax.tick_params(labelsize=6)
    plt.tight_layout(); save(fig, "fig3_heatmap")

    pal = {"seq": SEQ_PALETTE, "struct": CRD_PALETTE, "multi": MULTI_PALETTE}
    for key, group in groups.items():
        vals = [w.loc[e].fillna(0).values for e in group]
        fig, ax = plt.subplots(figsize=(2.25, 2.25), subplot_kw=dict(projection="polar"))
        radar_chart(ax, vals, [TASK_SHORT[t] for t in TASKS], pal[key], GROUP_COLOR[key])
        plt.tight_layout(); save(fig, f"fig3_radar_{key}")
        fig, ax = plt.subplots(figsize=(2.0, 0.16 * len(group) + 0.4)); ax.axis("off")
        ax.legend(handles=[mpatches.Patch(color=pal[key][i % len(pal[key])], label=DISP.get(e, e)) for i, e in enumerate(group)],
                  loc="center", frameon=False, fontsize=5, ncol=2)
        save(fig, f"fig3_legend_{key}")


if __name__ == "__main__":
    figure2(); figure3()
