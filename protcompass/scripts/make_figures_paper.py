"""Figures 2, 3 and 4 in the earlier layout, in one style: the palette and typography of make_panels_original_style
(Arial 8 pt, teal colour map for probing, rose colour map for downstream, navy / orange / purple for the declared
encoder classes, grey for non-learned floors), so the three read as one family.

  python make_figures_paper.py            # writes Figure2.png, Figure3.png, Figure4.png to paper/figures/
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import make_panels_original_style as P   # noqa: E402  (palettes, radar_chart, audit(), DISP, DECLARED, IIP labels)

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
RES = ROOT / "probing_results"
OUT = Path((PROTCOMPASS_ROOT + "/protcompass/paper/figures"))
TEAL, ROSE, GROUP_COLOR, FLOOR_COLOR = P.TEAL, P.ROSE, P.GROUP_COLOR, P.FLOOR_COLOR
DETECT = "#a83a3e"   # rose from the atlas label palette: a detection, distinct from the three class colours
BELOW = "#8a8a8a"
SIGN_ONLY = "#d99a9c"  # a lighter tint of DETECT: the repair holds in sign and falls under the magnitude floor
plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"]})
TASKS = ["ec_classification", "go_mf", "go_bp", "subcellular_localization", "remote_homology", "solubility",
         "membrane_soluble", "stability", "fluorescence", "ppi_affinity"]
TASK_SHORT = {**P.TASK_SHORT, "membrane_soluble": "Membrane", "subcellular_localization": "Localisation",
              "ppi_affinity": "Affinity", "fluorescence": "Fluorescence", "solubility": "Solubility",
              "stability": "Stability", "remote_homology": "Homology"}


def panel_letter(ax, letter, x=-0.02, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", ha="right", va="bottom")


def _probe_matrix():
    d = pd.concat([pd.read_csv(f) for f in glob.glob(str(RES / "by_source" / "*.csv"))], ignore_index=True)
    d["headline"] = d["pearson"].fillna(d["circ_cos"] if "circ_cos" in d else np.nan).fillna(d["balanced_accuracy"])
    d["family"] = np.where(d.readout == "linear", "linear", "nonlinear")
    m = d.groupby(["source", "probe", "iip", "family"]).headline.mean().reset_index()
    best = m.loc[m.groupby(["source", "probe"]).headline.idxmax()]
    fam = best.groupby(["source", "iip"]).headline.mean().unstack("iip").reindex(columns=P.IIPS)
    uses = P.audit()
    encs = [s for s in fam.index if not s.startswith("ref_") and s in uses.index]
    groups = {k: sorted([e for e in encs if P.declared(e) == k], key=lambda e: -fam.loc[e].mean())
              for k in ("seq", "struct", "multi")}
    seq_floor = fam.loc[[f for f in P.SEQ_FLOORS if f in fam.index]].max()
    all_floor = fam.loc[[f for f in P.ALL_FLOORS if f in fam.index]].max()
    return fam, groups, seq_floor, all_floor


def fig2():
    """Main-text Figure 2: probe scores per encoder (A) and the two-row floor strip (B), drawn at print width."""
    fam, groups, seq_floor, all_floor = _probe_matrix()
    order = groups["seq"] + groups["struct"] + groups["multi"]
    fig = plt.figure(figsize=(6.6, 3.3))
    ax = fig.add_axes([0.14, 0.44, 0.76, 0.52])
    im = ax.imshow(fam.loc[order].values.T, cmap=TEAL, aspect="auto", vmin=0.05, vmax=0.95, interpolation="nearest")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([P.DISP.get(e, e) for e in order], rotation=60, fontsize=5.6, ha="right")
    for t, e in zip(ax.get_xticklabels(), order):
        t.set_color(GROUP_COLOR[P.declared(e)])
    ax.set_yticks(np.arange(10)); ax.set_yticklabels([P.IIP_SHORT[i].replace("\n", " ") for i in P.IIPS], fontsize=6.5)
    ax.axvline(len(groups["seq"]) - 0.5, color="white", linewidth=2.2)
    ax.axvline(len(groups["seq"]) + len(groups["struct"]) - 0.5, color="white", linewidth=2.2)
    cax = fig.add_axes([0.915, 0.50, 0.012, 0.42])
    cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=6); cb.set_label("probe score", fontsize=6.5)
    panel_letter(ax, "A", x=-0.17, y=1.0)
    ax2 = fig.add_axes([0.14, 0.09, 0.76, 0.12])
    strip = np.vstack([seq_floor.values, all_floor.values])
    ax2.imshow(strip, cmap=TEAL, aspect="auto", vmin=0.05, vmax=0.95)
    ax2.set_yticks([0, 1]); ax2.set_yticklabels(["sequence floor", "coordinate floor"], fontsize=6.5)
    ax2.set_xticks(np.arange(10))
    ax2.set_xticklabels([P.IIP_SHORT[i].replace("\n", " ") for i in P.IIPS], fontsize=6, rotation=20, ha="right")
    for j in range(10):
        for i in range(2):
            v = strip[i, j]
            ax2.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5, color="white" if v > 0.6 else "#222")
    panel_letter(ax2, "B", x=-0.17, y=0.9)
    fig.savefig(OUT / "Figure2.png", dpi=300, bbox_inches="tight"); plt.close(fig); print("wrote Figure2.png")


def fig2_radars():
    """Appendix companion of Figure 2: per-class radar profiles against the floor built from the class's inputs."""
    fam, groups, seq_floor, all_floor = _probe_matrix()
    pal = {"seq": P.SEQ_PALETTE, "struct": P.CRD_PALETTE, "multi": P.MULTI_PALETTE}
    floors = {"seq": seq_floor, "struct": all_floor, "multi": all_floor}
    titles = {"seq": "sequence-based", "struct": "structure-based", "multi": "multimodal"}
    fig = plt.figure(figsize=(6.6, 2.9))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.55, left=0.06, right=0.97, top=0.82, bottom=0.2)
    for k, (key, group) in enumerate(groups.items()):
        axr = fig.add_subplot(gs[0, k], projection="polar")
        vals = [fam.loc[e].fillna(0).values for e in group]
        P.radar_chart(axr, vals, [P.IIP_SHORT[i] for i in P.IIPS], pal[key], GROUP_COLOR[key],
                      floors=[floors[key].fillna(0).values])
        axr.set_title(f"{titles[key]} ({len(group)})", fontsize=8, color=GROUP_COLOR[key], pad=24)
        panel_letter(axr, "ABC"[k], x=-0.2, y=1.2)
    h = [mpatches.Patch(color=GROUP_COLOR[k], label=f"{titles[k]}: encoders (thin), class mean (bold)") for k in groups]
    h.append(mpatches.Patch(color=FLOOR_COLOR, alpha=0.5, label="non-learned floor for the inputs the class receives"))
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(OUT / "Figure2_radars.png", dpi=300, bbox_inches="tight"); plt.close(fig); print("wrote Figure2_radars.png")


def fig3():
    w = pd.read_csv(ROOT / "extract" / "scores2_wide.csv", index_col=0).drop(index="ref_composition", errors="ignore")
    order_t = [t for t in TASKS if t in w.columns]
    w = w[order_t]
    uses = P.audit().reindex(w.index).fillna(True).astype(bool)
    rank = w.rank(ascending=False).mean(axis=1)
    idx = list(rank[~uses].sort_values().index) + list(rank[uses].sort_values().index)
    w = w.loc[idx]
    norm = (w - w.min()) / (w.max() - w.min())            # colour within each task, since the metrics differ
    fig, ax = plt.subplots(figsize=(6.6, 2.3))
    im = ax.imshow(norm.values.T, cmap=ROSE, aspect="auto", vmin=0, vmax=1, interpolation="nearest")
    for i in range(len(w)):
        for j in range(len(order_t)):
            if not np.isfinite(w.iloc[i, j]):
                ax.text(i, j, "n/a", ha="center", va="center", fontsize=4.5, rotation=90, color="#888")
    n_seq = int((~uses.loc[idx]).sum())
    ax.axvline(n_seq - 0.5, color="white", linewidth=3.0)
    ax.set_yticks(range(len(order_t))); ax.set_yticklabels([TASK_SHORT.get(t, t) for t in order_t], fontsize=6.5)
    ax.set_xticks(range(len(w))); ax.set_xticklabels([P.DISP.get(e, e) for e in w.index], fontsize=5.6, rotation=60, ha="right")
    for t, e in zip(ax.get_xticklabels(), w.index):
        t.set_color(GROUP_COLOR[P.declared(e)])
    ax.text(n_seq / 2 - 0.5, -0.8, "measured not to read coordinates", ha="center", va="bottom", fontsize=6.5, color="#333")
    ax.text(n_seq + (len(w) - n_seq) / 2 - 0.5, -0.8, "measured to read coordinates", ha="center", va="bottom", fontsize=6.5, color="#333")
    cb = fig.colorbar(im, ax=ax, fraction=0.012, pad=0.01); cb.ax.tick_params(labelsize=5.5); cb.set_label("score, scaled within task", fontsize=6)
    h = [mpatches.Patch(color=GROUP_COLOR[k], label=l) for k, l in
         (("seq", "declared sequence-based"), ("struct", "declared structure-based"), ("multi", "declared multimodal"))]
    fig.tight_layout(); fig.savefig(OUT / "Figure3.png", dpi=300, bbox_inches="tight"); plt.close(fig); print("wrote Figure3.png")


def _fig4_screen(ax):
    """Panel A: the erasure effect against the sham, per task, read against the detection bound. A screen: it shows
    where erasing a property costs anything beyond a rank-matched sham, before that cost is attributed."""
    from exclusion_bounds import load, mde, TASK_LABEL, reachable_at_property_rank
    v2 = load("erasure_v2*.csv"); ctrl = load("erasure_control*.csv")
    cond = v2[(v2.arm == "conditional") & v2.erasure_ok.fillna(False)]
    # the reference bar is read at the rank the property blocks occupy, not at the end of the extended ladder
    reach, reach_rank = reachable_at_property_rank(ctrl, cond)
    rows = []
    for (t, c), g in cond.groupby(["task", "concept"]):
        rows.append(dict(task=t, concept=c, n=len(g), eff=g.effect.mean(), sd=g.effect.std()))
    d = pd.DataFrame(rows); d["mde"] = [mde(r.sd, r.n) for r in d.itertuples()]
    keep = [t for t in d.task.unique() if t in reach.index and np.isfinite(reach[t]) and reach[t] > 0
            and np.isfinite(d[d.task == t].mde.max())]
    labels = []
    for i, t in enumerate(sorted(keep, key=lambda x: -reach[x])):
        g = d[d.task == t]; bound, r = g.mde.max(), reach[t]
        ax.barh(i, r, height=0.62, color=TEAL(0.18), zorder=1)
        ax.barh(i, bound, height=0.62, color=TEAL(0.45), zorder=2)
        for row in g.itertuples():
            ax.scatter(row.eff, i, s=14, color=DETECT if row.eff > bound else BELOW, zorder=4, clip_on=False,
                       edgecolor="white", linewidth=0.4)
        ax.text(r + 0.006, i, f"{r:+.2f}", va="center", fontsize=5.5, color="#555")
        labels.append(f"{dict(TASK_LABEL, ppi_affinity='PPI affinity').get(t, t)} (n={int(g.n.max())})")
    ax.set_yticks(range(len(keep))); ax.set_yticklabels(labels, fontsize=6); ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.7, len(keep) - 0.3); ax.invert_yaxis()
    ax.set_xlabel(r"erasure effect $\Delta_{\rm sham}$", fontsize=6.5)
    ax.tick_params(axis="x", labelsize=6)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(handles=[mpatches.Patch(color=TEAL(0.45), label="below the detection bound"),
                       mpatches.Patch(color=TEAL(0.18), label="reachable range"),
                       plt.Line2D([], [], marker="o", ls="", ms=4, color=DETECT, label="above the bound"),
                       plt.Line2D([], [], marker="o", ls="", ms=4, color=BELOW, label="below the bound")],
              loc="upper center", bbox_to_anchor=(0.45, -0.16), ncol=2, frameon=False, fontsize=5.5,
              columnspacing=1.0, handletextpad=0.4)


def _fig4_attribution(ax):
    """Panel B: for every pair either rule names over the full 60-pair grid, the effect against each null and the repair
    contrast against the variance-matched null. Reads the two CSVs written by scripts/named_pair_verdicts.py --tag full.
    Pairs on PPI affinity carry a dagger: they do not hold with chain-disjoint folds (scripts/ppi_repair_cv.py)."""
    V = pd.read_csv(ROOT / "erasure" / "named_pair_verdicts_full.csv")
    C = pd.read_csv(ROOT / "erasure" / "named_pair_cells_full.csv")
    comp = V.concept == "amino-acid composition"
    V = V[(V.named_by_effect_rule) | (V.passes_R_rule) | (comp & V.passes_R_sign)]
    V = V.sort_values("R_null", ascending=False).reset_index(drop=True)
    short = {"amino-acid composition": "composition", "burial and packing": "burial/packing",
             "secondary structure content": "sec. structure", "global shape": "global shape",
             "contact order": "contact order", "size": "size"}
    task = {"membrane_soluble": "membrane", "subcellular_localization": "localisation", "fluorescence": "fluorescence",
            "solubility": "solubility", "ppi_affinity": "PPI", "go_mf": "GO-MF", "go_bp": "GO-BP",
            "stability": "stability"}
    labels = []
    for i, r in V.iterrows():
        g = C[(C.task == r.task) & (C.concept == r.concept)]
        se = g.R_null.std() / np.sqrt(len(g)) if len(g) > 1 else 0.0
        ax.scatter(r.eff_sham, i, s=16, facecolor="white", edgecolor=BELOW, linewidth=0.8, zorder=3)
        ax.scatter(r.eff_null, i, s=14, marker="D", color="#c7c7c7", edgecolor="#8a8a8a", linewidth=0.4, zorder=3)
        col = DETECT if r.passes_R_rule else (SIGN_ONLY if r.passes_R_sign else BELOW)
        ax.errorbar(r.R_null, i, xerr=1.96 * se, fmt="o", ms=4, color=col, ecolor=col, elinewidth=0.8, capsize=1.5,
                    zorder=4)
        mark = ("" if r.named_by_effect_rule else " +") + (" \u2020" if r.task == "ppi_affinity" else "")
        labels.append(f"{task.get(r.task, r.task)} / {short.get(r.concept, r.concept)}{mark} "
                      f"({int(r.pos_null)}/{int(r.n)})")
    ax.axvline(0, color="#555", lw=0.6, zorder=1)
    ax.set_yticks(range(len(V))); ax.set_yticklabels(labels, fontsize=6); ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.7, len(V) - 0.3); ax.invert_yaxis()
    ax.set_xlabel("change in score, mean over encoders", fontsize=6.5)
    ax.set_xticks([-0.05, 0.0, 0.05, 0.10])
    ax.tick_params(axis="x", labelsize=6)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", ms=4, mfc="white", mec=BELOW, label=r"$\Delta_{\rm sham}$"),
                       plt.Line2D([], [], marker="D", ls="", ms=3.5, color="#c7c7c7", mec="#8a8a8a",
                                  label=r"$\Delta_{\rm var}$"),
                       plt.Line2D([], [], marker="o", ls="", ms=4, color=DETECT, label=r"$R_{\rm var}$: passes the rule"),
                       plt.Line2D([], [], marker="o", ls="", ms=4, color=SIGN_ONLY, label=r"$R_{\rm var}$: sign only"),
                       plt.Line2D([], [], marker="o", ls="", ms=4, color=BELOW, label=r"$R_{\rm var}$: fails")],
              loc="upper center", bbox_to_anchor=(0.45, -0.16), ncol=2, frameon=False, fontsize=5.5,
              columnspacing=1.0, handletextpad=0.4)


def fig4():
    fig = plt.figure(figsize=(6.9, 3.05))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.08], wspace=0.95)
    axA, axB = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    _fig4_screen(axA); _fig4_attribution(axB)
    panel_letter(axA, "A", x=-0.62, y=1.03); panel_letter(axB, "B", x=-0.9, y=1.03)
    fig.savefig(OUT / "Figure4.png", dpi=300, bbox_inches="tight"); plt.close(fig); print("wrote Figure4.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig2(); fig2_radars(); fig3(); fig4()
