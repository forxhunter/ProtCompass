"""The four data figures in one style: Figure 2, Figure 3, Figure 4 and the residue-ranking figure.

Same data and values as the scripts of the earlier figure layout (make_figure2_symmetric.py, make_figures_paper.py,
paper/scripts/plot_residue_rankings.py). What is unified:
  - one canvas width (6.6 in) for every figure, cropped only vertically, so text has the same printed size in all four;
  - one set of font sizes (row labels 6, encoder names 5.6, numeric ticks 6, axis labels 6.5, legends 5.8, panel
    letters 10 bold) and one font family;
  - one colour style, taken from Figure 1: muted slate teal (probe scores), terracotta (downstream scores, repair
    detections), indigo blue / terracotta / purple (declared class as name colour; blue / terracotta also for the
    measured input group) and cool greys; the colours differ by role and share one saturation and lightness range;
  - one layout grammar: full box on heat maps, bottom spine only on dot plots, frameless legends below the panels,
    hatched cells for missing values.
No text is drawn inside a figure other than axis labels, tick labels, legends and panel letters.

  python scripts/make_figures_unified.py     # from protcompass/; writes Figure2, Figure3, Figure4 and
                                             # residue_rankings (.png and .pdf) to <repository>/figures/
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import make_panels_original_style as P    # noqa: E402  (palettes, audit(), DISP, declared())
import make_figure2_symmetric as S        # noqa: E402  (Figure 2 matrix from the symmetric rerun)

ROOT = Path(PROTCOMPASS_ROOT) / "protcompass"
MS = ROOT / "paper"                       # holds generated/contact_task.tex and generated/masif_task.tex
OUT = Path(PROTCOMPASS_ROOT) / "figures"  # overwritten on each run

# Composition-and-length reference scores (last row of the full downstream table); checked against scores2_wide.csv.
REF_TABLE = {"ec_classification": 0.086, "go_mf": 0.227, "go_bp": 0.274, "subcellular_localization": 0.470,
             "remote_homology": 0.071, "solubility": 0.588, "membrane_soluble": 0.779, "stability": 0.090,
             "fluorescence": 0.362, "ppi_affinity": 0.091}
# Follow-up outcome of the seven pairs that pass the repair rule (chain-disjoint PPI folds; mutation-feature controls).
RETAINED, CHAIN, FLOOR = "retained", "chain", "floor"
FOLLOW = {("stability", "amino-acid composition"): RETAINED,
          ("fluorescence", "amino-acid composition"): FLOOR,
          ("fluorescence", "burial and packing"): FLOOR,
          ("ppi_affinity", "secondary structure content"): CHAIN,
          ("ppi_affinity", "contact order"): CHAIN,
          ("ppi_affinity", "size"): CHAIN,
          ("ppi_affinity", "global shape"): CHAIN}
FOLLOW_MARK = {RETAINED: r"$\checkmark$", CHAIN: r"$\dagger$", FLOOR: r"$\ddagger$"}
GROUP_ORDER = [[("stability", "amino-acid composition")],
               [("fluorescence", "amino-acid composition"), ("fluorescence", "burial and packing")],
               [("ppi_affinity", "secondary structure content"), ("ppi_affinity", "contact order"),
                ("ppi_affinity", "size"), ("ppi_affinity", "global shape")]]

# ---- shared style --------------------------------------------------------------------------------------------------
W = 6.6                                   # canvas width in inches, identical for every figure
FS_ROW, FS_ENC, FS_NUM, FS_AX, FS_LEG, FS_LETTER = 6.0, 5.6, 6.0, 6.5, 6.5, 10   # subscripts at 6.5 pt are 4.55 pt
# Palette in the tone of Figure 1 (the hand-drawn schematic): four muted hue families at similar saturation --
# slate teal, terracotta, indigo blue and purple -- plus neutral greys. Hues keep their earlier roles: teal for probe
# scores, terracotta for downstream scores and for repair-rule detections, blue / terracotta / purple for the classes.
INK, MUTED = "#2a2e3a", "#5b5f6b"
TEAL = LinearSegmentedColormap.from_list("slate_teal", ["#f3f7f7", "#cfdfe1", "#9dbcc2", "#608490", "#2c4752"], N=256)
ROSE = LinearSegmentedColormap.from_list("terracotta", ["#fcf4f1", "#efd6ce", "#d6a59c", "#b06a66", "#6e3537"], N=256)
GROUP_COLOR = {"seq": "#48609c", "struct": "#a9605a", "multi": "#8662a3"}
DETECT, SIGN_ONLY, BELOW = "#9c4a48", "#dbaaa4", "#8c8f96"
MEASURED = {False: GROUP_COLOR["seq"], True: GROUP_COLOR["struct"]}   # sequence-only / coordinate-reading
MEASURED_LABEL = {False: "Sequence-only", True: "Coordinate-reading"}
DECLARED_LABEL = {"seq": "Sequence-based", "struct": "Structure-based", "multi": "Multimodal"}
CELL = 0.143                              # heat-map cell width in inches, shared by Figures 2 and 3
ROW = 0.165                               # row height in inches, shared by all figures
HEAT_RIGHT = 6.0                          # right edge of every heat map; the colour bar starts at 6.08
TASK_NAME = {"ec_classification": "EC", "go_mf": "GO-MF", "go_bp": "GO-BP",
             "subcellular_localization": "Localization", "remote_homology": "Homology", "solubility": "Solubility",
             "membrane_soluble": "Membrane", "stability": "Stability", "fluorescence": "Fluorescence",
             "ppi_affinity": "PPI affinity"}
TASKS = list(TASK_NAME)

plt.rcParams.update({"font.size": FS_NUM, "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                     "pdf.fonttype": 42, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "hatch.linewidth": 0.5, "hatch.color": "#9a9da6",
                     "legend.fontsize": FS_LEG, "legend.frameon": False})


def canvas(height):
    return plt.figure(figsize=(W, height))


def axes_in(fig, left, bottom, width, height, **kw):
    """Axes placed in inches from the lower-left corner of the canvas."""
    _, H = fig.get_size_inches()
    return fig.add_axes([left / W, bottom / H, width / W, height / H], **kw)


def letter(fig, ax, s, x_in=0.04):
    """Panel letter at a fixed distance from the left canvas edge, level with the top of the panel."""
    _, H = fig.get_size_inches()
    fig.text(x_in / W, ax.get_position().y1 + 0.04 / H, s, fontsize=FS_LETTER, fontweight="bold", ha="left",
             va="bottom")


def legend_in(fig, handles, x_in, y_in, ncol=1, title=None, **kw):
    """Frameless legend whose upper-left corner is at (x_in, y_in) inches."""
    _, H = fig.get_size_inches()
    opts = dict(loc="upper left", bbox_to_anchor=(x_in / W, y_in / H), ncol=ncol, title=title,
                title_fontsize=FS_LEG, handletextpad=0.5, columnspacing=1.4, labelspacing=0.35, borderaxespad=0,
                borderpad=0, alignment="left")
    opts.update(kw)
    return fig.legend(handles=handles, **opts)


def dot_axes(ax):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=FS_ROW)
    ax.tick_params(axis="x", labelsize=FS_NUM)


def heat_axes(ax):
    ax.tick_params(axis="y", labelsize=FS_ROW)
    ax.tick_params(axis="x", labelsize=FS_ENC)


def encoder_ticks(ax, encoders):
    ax.set_xticks(range(len(encoders)))
    ax.set_xticklabels([P.DISP.get(e, e) for e in encoders], rotation=60, ha="right", rotation_mode="anchor",
                       fontsize=FS_ENC)
    for t, e in zip(ax.get_xticklabels(), encoders):
        t.set_color(GROUP_COLOR[P.declared(e)])


def hatch_missing(ax, values):
    """Missing cells are hatched; values is laid out as drawn (rows on y, columns on x)."""
    for (j, i) in zip(*np.where(~np.isfinite(values))):
        ax.add_patch(mpatches.Rectangle((i - 0.5, j - 0.5), 1, 1, facecolor="white", edgecolor="#9a9da6",
                                        hatch="//////", linewidth=0))


def colorbar(fig, im, ax, label, ticks):
    pos = ax.get_position()
    cax = fig.add_axes([(HEAT_RIGHT + 0.08) / W, pos.y0, 0.075 / W, pos.height])
    cb = fig.colorbar(im, cax=cax, ticks=ticks)
    cb.ax.tick_params(labelsize=FS_NUM, length=2)
    cb.set_label(label, fontsize=FS_AX)
    cb.outline.set_linewidth(0.6)


H_DECLARED = [Line2D([], [], ls="", marker=r"$\mathrm{Aa}$", ms=7.5, mec=GROUP_COLOR[k], mfc=GROUP_COLOR[k], mew=0.1,
                     label=DECLARED_LABEL[k]) for k in ("seq", "struct", "multi")]
H_MISSING = mpatches.Patch(facecolor="white", edgecolor="#9a9da6", hatch="//////", linewidth=0.5,
                           label="Not evaluated")


def save(fig, stem):
    """Crop vertically to the drawn content; the width stays W so every figure prints at the same scale."""
    fig.canvas.draw()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    assert tb.x0 >= -0.005 and tb.x1 <= W + 0.005, f"{stem}: content leaves the canvas ({tb.x0:.2f}, {tb.x1:.2f})"
    box = Bbox([[0, tb.y0 - 0.03], [W, tb.y1 + 0.03]])
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=600, bbox_inches=box)
    plt.close(fig)
    print(f"wrote {stem}.png / .pdf  ({W:.1f} x {box.height:.2f} in)")


# ---- Figure 2 ------------------------------------------------------------------------------------------------------
def fig2():
    fam, _ = S.matrix()
    uses = P.audit()
    encs = [s for s in fam.index if not s.startswith("ref_") and s in uses.index]
    assert len(encs) == 33, len(encs)
    groups = {k: sorted([e for e in encs if P.declared(e) == k], key=lambda e: -fam.loc[e].mean())
              for k in ("seq", "struct", "multi")}
    order = groups["seq"] + groups["struct"] + groups["multi"]
    iips = S.IIPS
    rows = [{"I5": "B-factor"}.get(i, P.IIP_SHORT[i].replace("\n", " ").replace("/ ", "/")) for i in iips]
    left = HEAT_RIGHT - CELL * len(order)

    fig = canvas(3.9)
    ax = axes_in(fig, left, 2.25, CELL * len(order), ROW * len(iips))
    vals = fam.loc[order].values.T
    im = ax.imshow(vals, cmap=TEAL, aspect="auto", vmin=0.05, vmax=0.95, interpolation="nearest")
    hatch_missing(ax, vals)
    encoder_ticks(ax, order)
    ax.set_yticks(range(len(iips))); ax.set_yticklabels(rows)
    heat_axes(ax)
    for cut in (len(groups["seq"]), len(groups["seq"]) + len(groups["struct"])):
        ax.axvline(cut - 0.5, color="white", linewidth=2.5)
    colorbar(fig, im, ax, "Probe score", [0.2, 0.4, 0.6, 0.8])
    letter(fig, ax, "A")

    ax2 = axes_in(fig, left, 1.05, CELL * len(order), ROW * 2)
    strip = np.vstack([fam.loc[S.SEQ_FLOORS].max().values, fam.loc[S.ALL_FLOORS].max().values])
    ax2.imshow(strip, cmap=TEAL, aspect="auto", vmin=0.05, vmax=0.95, interpolation="nearest")
    ax2.set_yticks([0, 1]); ax2.set_yticklabels(["Best sequence floor", "Best of all ten floors"])
    ax2.set_xticks(range(len(iips)))
    ax2.set_xticklabels(rows, rotation=20, ha="right", rotation_mode="anchor")
    heat_axes(ax2); ax2.tick_params(axis="x", labelsize=FS_ROW)
    letter(fig, ax2, "B")
    legend_in(fig, H_DECLARED + [H_MISSING], 1.6, 0.52, ncol=4)
    save(fig, "Figure2")


# ---- Figure 3 ------------------------------------------------------------------------------------------------------
def fig3():
    wall = pd.read_csv(ROOT / "extract" / "scores2_wide.csv", index_col=0)
    ref = wall.loc["ref_composition", TASKS]
    w = wall.drop(index="ref_composition")[TASKS]
    assert len(w) == 33, len(w)
    for t in TASKS:
        assert round(float(ref[t]), 3) == REF_TABLE[t], (t, ref[t])
    uses = P.audit().reindex(w.index).fillna(True).astype(bool)
    rank = w.rank(ascending=False).mean(axis=1)
    idx = list(rank[~uses].sort_values().index) + list(rank[uses].sort_values().index)
    w = w.loc[idx]
    lo, hi = w.min(), w.max()
    norm = (w - lo) / (hi - lo)                                  # encoder colours scaled within task
    assert (ref >= lo).all() and (ref <= hi).all()               # the reference sits inside every encoder range
    ref_norm = (ref - lo) / (hi - lo)

    n = len(w)
    w_ref, gap = 2.0 * CELL, 0.9 * CELL
    left = HEAT_RIGHT - CELL * n
    left_ref = left - gap - w_ref
    bottom, height = 1.3, ROW * len(TASKS)

    fig = canvas(3.3)
    ax0 = axes_in(fig, left_ref, bottom, w_ref, height)
    ax0.imshow(ref_norm.values[:, None], cmap=ROSE, aspect="auto", vmin=0, vmax=1, interpolation="nearest")
    ax0.set_yticks(range(len(TASKS))); ax0.set_yticklabels([TASK_NAME[t] for t in TASKS])
    ax0.set_xticks([0]); ax0.set_xticklabels(["Composition + length"], rotation=60, ha="right",
                                            rotation_mode="anchor", color=MUTED, style="italic")
    heat_axes(ax0)
    for s in ax0.spines.values():
        s.set_linewidth(0.9)

    ax = axes_in(fig, left, bottom, CELL * n, height)
    vals = norm.values.T
    im = ax.imshow(vals, cmap=ROSE, aspect="auto", vmin=0, vmax=1, interpolation="nearest")
    hatch_missing(ax, w.values.T)
    n_seq = int((~uses.loc[idx]).sum())
    ax.axvline(n_seq - 0.5, color="white", linewidth=2.5)
    ax.set_yticks([])
    encoder_ticks(ax, idx)
    heat_axes(ax)
    colorbar(fig, im, ax, "Score, scaled within task", [0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # measured input group as a colour strip above the encoder columns, split at the same place as the heat map
    axs = axes_in(fig, left, bottom + height + 0.035, CELL * n, 0.05)
    axs.imshow(uses.loc[idx].values.astype(int)[None, :], cmap=ListedColormap([MEASURED[False], MEASURED[True]]),
               aspect="auto", vmin=0, vmax=1, interpolation="nearest")
    axs.axvline(n_seq - 0.5, color="white", linewidth=2.5)
    axs.set_xticks([]); axs.set_yticks([])
    for s in axs.spines.values():
        s.set_visible(False)

    h_meas = [Line2D([], [], color=MEASURED[k], lw=3, solid_capstyle="butt", label=MEASURED_LABEL[k])
              for k in (False, True)]
    legend_in(fig, h_meas + H_DECLARED + [H_MISSING], 1.6, 0.42, ncol=3)
    save(fig, "Figure3")
    med = w.median()
    print("fig3 check: reference within 0.02 of the encoder median on",
          [t for t in TASKS if abs(ref[t] - med[t]) <= 0.02])


# ---- Figure 4 ------------------------------------------------------------------------------------------------------
def _screen(ax):
    """Panel A: the erasure effect against the sham per task and property, with the MDE and the reachable range."""
    from exclusion_bounds import load, mde, reachable_at_property_rank
    v2 = load("erasure_v2*.csv"); ctrl = load("erasure_control*.csv")
    cond = v2[(v2.arm == "conditional") & v2.erasure_ok.fillna(False)]
    reach, _ = reachable_at_property_rank(ctrl, cond)
    rows = [dict(task=t, concept=c, n=len(g), eff=g.effect.mean(), sd=g.effect.std())
            for (t, c), g in cond.groupby(["task", "concept"])]
    d = pd.DataFrame(rows); d["mde"] = [mde(r.sd, r.n) for r in d.itertuples()]
    # every task with a finite MDE is drawn; a task whose positive control lost nothing (EC) has no reachable bar
    keep = [t for t in d.task.unique() if t in reach.index and np.isfinite(d[d.task == t].mde.max())]
    keep = sorted(keep, key=lambda x: -reach[x])
    assert len(keep) == 10, keep
    V = pd.read_csv(ROOT / "erasure" / "named_pair_verdicts_full.csv")
    ruled = set(zip(V[V.named_by_effect_rule].task, V[V.named_by_effect_rule].concept))
    assert len(ruled) == 8, ruled                                # the eight pairs of the sham-referenced rule
    labels = []
    for i, t in enumerate(keep):
        g = d[d.task == t]; bound, r = g.mde.max(), reach[t]
        if r > 0:
            ax.barh(i, r, height=0.62, color=TEAL(0.15), zorder=1)
        ax.barh(i, bound, height=0.62, color=TEAL(0.55), zorder=2)
        ax.plot([bound, bound], [i - 0.31, i + 0.31], color=TEAL(0.95), lw=0.9, zorder=3, solid_capstyle="butt")
        for row in g.itertuples():
            if (t, row.concept) in ruled:
                assert row.eff > bound, (t, row.concept)         # a ruled pair is always above the MDE here
                ax.scatter(row.eff, i, s=17, facecolor=DETECT, edgecolor=INK, linewidth=0.9, zorder=6, clip_on=False)
            elif row.eff > bound:
                ax.scatter(row.eff, i, s=14, facecolor=DETECT, edgecolor="white", linewidth=0.5, zorder=5,
                           clip_on=False)
            else:
                ax.scatter(row.eff, i, s=13, facecolor="white", edgecolor=BELOW, linewidth=0.8, zorder=4,
                           clip_on=False)
        labels.append(f"{TASK_NAME[t]} (n={int(g.n.max())})")
    ax.set_yticks(range(len(keep))); ax.set_yticklabels(labels)
    ax.set_ylim(-0.7, len(keep) - 0.3); ax.invert_yaxis()
    ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4])
    ax.set_xlabel(r"Sham-referenced erasure effect ($\Delta_{\rm sham}$)", fontsize=FS_AX)
    dot_axes(ax)
    return [mpatches.Patch(facecolor=TEAL(0.55), edgecolor=TEAL(0.95), linewidth=0.6,
                           label="Minimum detectable effect (MDE)"),
            mpatches.Patch(color=TEAL(0.15), label="Reachable range"),
            Line2D([], [], marker="o", ls="", ms=4, mfc=DETECT, mec=INK, mew=0.9, label="Passes erasure rule"),
            Line2D([], [], marker="o", ls="", ms=4, mfc=DETECT, mec=DETECT, label="Above MDE"),
            Line2D([], [], marker="o", ls="", ms=3.6, mfc="white", mec=BELOW, mew=0.8, label="Below MDE")]


# repair-rule outcome: colour and fill both change, so the three states survive greyscale printing
RULE_STYLE = {"pass": (DETECT, dict(mfc=DETECT, mec=DETECT, mew=0.8, fillstyle="full")),
              "sign": (DETECT, dict(mfc=DETECT, mec=DETECT, mew=0.8, fillstyle="left", markerfacecoloralt="white")),
              "fail": (BELOW, dict(mfc="#cdd0d6", mec=BELOW, mew=0.8, fillstyle="full"))}


def _attribution(ax):
    """Panel B: effect against each null and the repair contrast; follow-up outcome on a right-hand axis."""
    V = pd.read_csv(ROOT / "erasure" / "named_pair_verdicts_full.csv")
    C = pd.read_csv(ROOT / "erasure" / "named_pair_cells_full.csv")
    comp = V.concept == "amino-acid composition"
    V = V[(V.named_by_effect_rule) | (V.passes_R_rule) | (comp & V.passes_R_sign)].copy()
    passed = set(zip(V[V.passes_R_rule].task, V[V.passes_R_rule].concept))
    assert passed == set(FOLLOW), passed                      # exactly the seven candidates
    first = [p for grp in GROUP_ORDER for p in grp]
    rest = V[[(t, c) not in passed for t, c in zip(V.task, V.concept)]].sort_values("R_null", ascending=False)
    V = V.set_index(["task", "concept"]).loc[first + list(zip(rest.task, rest.concept))].reset_index()

    short = {"amino-acid composition": "composition", "burial and packing": "burial/packing",
             "secondary structure content": "sec. structure", "global shape": "global shape",
             "contact order": "contact order", "size": "size"}
    task = dict(TASK_NAME, ppi_affinity="PPI")
    ax.axhspan(-0.5, 0.5, color="#eef0f3", zorder=0, lw=0)       # locates the retained pair; encodes no value
    for b in np.cumsum([len(g) for g in GROUP_ORDER]):
        ax.axhline(b - 0.5, color="#b9bcc4", lw=0.5, ls=(0, (2, 2)), zorder=1)
    labels, marks = [], []
    for i, r in V.iterrows():
        g = C[(C.task == r.task) & (C.concept == r.concept)]
        se = g.R_null.std() / np.sqrt(len(g)) if len(g) > 1 else 0.0
        ax.scatter(r.eff_sham, i, s=16, facecolor="white", edgecolor=BELOW, linewidth=0.8, zorder=3)
        ax.scatter(r.eff_null, i, s=14, marker="D", color="#cdd0d6", edgecolor=BELOW, linewidth=0.5, zorder=3)
        col, style = RULE_STYLE["pass" if r.passes_R_rule else ("sign" if r.passes_R_sign else "fail")]
        ax.errorbar(r.R_null, i, xerr=1.96 * se, fmt="o", ms=4.2, ecolor=col, elinewidth=0.8, capsize=1.5,
                    zorder=4, **style)
        labels.append(f"{task[r.task]} / {short.get(r.concept, r.concept)}"
                      f"{'' if r.named_by_effect_rule else ' +'} ({int(r.pos_null)}/{int(r.n)})")
        marks.append(FOLLOW_MARK.get(FOLLOW.get((r.task, r.concept)), ""))
    ax.axvline(0, color=MUTED, lw=0.6, zorder=1)
    ax.set_yticks(range(len(V))); ax.set_yticklabels(labels)
    ax.set_ylim(-0.7, len(V) - 0.3); ax.invert_yaxis()
    ax.set_xticks([-0.05, 0.0, 0.05, 0.10])
    ax.set_xlabel("Change in score", fontsize=FS_AX)
    dot_axes(ax)
    axr = ax.secondary_yaxis("right")                             # follow-up outcome as right-hand tick labels
    axr.set_yticks(range(len(V))); axr.set_yticklabels(marks, fontsize=7.5)
    axr.tick_params(axis="y", length=0, pad=3)
    axr.spines["right"].set_visible(False)

    h_rule = [Line2D([], [], marker="o", ls="", ms=4, mfc="white", mec=BELOW, label=r"$\Delta_{\rm sham}$"),
              Line2D([], [], marker="D", ls="", ms=3.5, color="#cdd0d6", mec=BELOW, label=r"$\Delta_{\rm var}$"),
              Line2D([], [], marker="o", ls="", ms=4.2, label=r"$R_{\rm var}$: passes", **RULE_STYLE["pass"][1]),
              Line2D([], [], marker="o", ls="", ms=4.2, label=r"$R_{\rm var}$: sign only", **RULE_STYLE["sign"][1]),
              Line2D([], [], marker="o", ls="", ms=4.2, label=r"$R_{\rm var}$: fails", **RULE_STYLE["fail"][1])]
    follow = [(FOLLOW_MARK[RETAINED], "Retained"),
              (FOLLOW_MARK[CHAIN], "Fails chain-disjoint folds"),
              (FOLLOW_MARK[FLOOR], "Below 0.02 threshold"),
              (r"$+$", "Repair rule only")]
    h_follow = [Line2D([], [], ls="", marker=m, ms=5.5, mec=INK, mfc=INK, mew=0.2, label=l) for m, l in follow]
    return h_rule, h_follow


def fig4():
    fig = canvas(3.7)
    bottom, height = 1.45, ROW * 11
    axA = axes_in(fig, 1.0, bottom, 1.7, height)
    axB = axes_in(fig, 4.72, bottom, 1.62, height)
    hA = _screen(axA)
    h_rule, h_follow = _attribution(axB)
    letter(fig, axA, "A"); letter(fig, axB, "B", x_in=2.86)
    y = bottom - 0.47
    legend_in(fig, hA, 0.25, y)
    legend_in(fig, h_rule, 2.95, y, title="Initial screen")
    legend_in(fig, h_follow, 4.6, y, title="Follow-up controls")
    save(fig, "Figure4")


# ---- residue-level rankings ----------------------------------------------------------------------------------------
def _read_table(name):
    rows = {}
    for line in (MS / "generated" / name).read_text().splitlines():
        cells = [s.strip().replace("$", "") for s in line.split("&")]
        if len(cells) == 5 and cells[1] in ("yes", "no"):
            rows[cells[0]] = (cells[1], float(cells[2]), int(cells[3]), float(cells[4].replace("\\", "").strip()))
    return rows


def fig_residue():
    contact, interface = _read_table("contact_task.tex"), _read_table("masif_task.tex")
    names = sorted(contact.keys() & interface.keys(), key=lambda k: contact[k][3])
    assert len(names) == 12
    for k in names:
        assert contact[k][3] == interface[k][3]
    height = ROW * len(names)
    fig = canvas(3.2)
    lefts, width, bottom = (1.0, 3.9), 2.55, 0.95
    for left, data, what, s in zip(lefts, (contact, interface), ("contact", "interface"), "AB"):
        ax = axes_in(fig, left, bottom, width, height)
        for i, k in enumerate(names):
            coord, _, rank, pooled = data[k]
            color = MEASURED[coord == "yes"]
            ax.plot([pooled, rank], [i, i], color=color, lw=1.1, zorder=2)
            ax.scatter(pooled, i, s=16, marker="o", edgecolor=color, facecolor="white", linewidth=0.8, zorder=3)
            ax.scatter(rank, i, s=14, marker="D", color=color, zorder=4)
        ax.set_xlim(0.2, 33.8); ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 33])
        ax.set_xlabel(f"{what.capitalize()} prediction rank", fontsize=FS_AX)
        ax.set_ylim(len(names) - 0.4, -0.7)
        ax.set_yticks(range(len(names)))
        if s == "A":
            ax.set_yticklabels(names)
        else:
            ax.set_yticklabels([])
        dot_axes(ax)
        letter(fig, ax, s, x_in=left - (0.96 if s == "A" else 0.2))
    h = [Line2D([], [], marker="o", ls="", ms=4, mfc="white", mec=MUTED, label="Pooled mean rank"),
         Line2D([], [], marker="D", ls="", ms=3.5, color=MUTED, label="Residue-level rank"),
         Line2D([], [], color=MEASURED[False], lw=1.1, label=MEASURED_LABEL[False]),
         Line2D([], [], color=MEASURED[True], lw=1.1, label=MEASURED_LABEL[True])]
    legend_in(fig, h, 1.9, bottom - 0.47, ncol=2)
    save(fig, "residue_rankings")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig2(); fig3(); fig4(); fig_residue()
