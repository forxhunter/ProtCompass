"""Redraw the seed-variability panels of the uncertainty appendix from the current runs.

The eight panels there came from the predecessor pipeline: five seeds, a linear-only readout, ten task axes and
encoders grouped by a declared modality. The numbers in the text next to them were already the current ones, so the
figures contradicted their own caption. These are drawn from the released per-seed tables instead, with the panel
conventions of the rest of the paper: three declared classes in the paper's colours, the nonlinear readout the text
describes, and the probes and tasks the release actually has.

  python make_uncertainty_figures.py
Writes figures/uncertainty_appendix/{IIP,downstream}/*.png.
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
OUT = Path((PROTCOMPASS_ROOT + "/protcompass/paper/figures/uncertainty_appendix"))
sys.path.insert(0, str(ROOT / "scripts"))
from make_panels_original_style import DECLARED, DISP, IIP_SHORT, TASK_SHORT  # noqa: E402
from panel import panel as released_panel  # noqa: E402

CLASS_COLOR = {"seq": "#1a4a8a", "struct": "#D2691E", "multi": "#7c4fa0"}
CLASS_NAME = {"seq": "sequence", "struct": "structure", "multi": "multimodal"}
TEAL = LinearSegmentedColormap.from_list("paper_teal", ["#f2f9f9", "#aed9d9", "#5eb5b5", "#2b8c8c", "#144d4d"], N=256)
plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"]})
METRIC = ["accuracy", "spearman", "circ_cos", "r2"]


def probe_scores():
    """One score per (source, probe, seed) from the nonlinear readout, on whichever metric the probe reports."""
    d = pd.read_csv(ROOT / "probing_results" / "scores_long.csv")
    d = d[d.readout.isin(["mlp2", "pair_mlp"]) & d.source.isin(released_panel())].copy()
    # the paper's headline metric: Pearson, else circular cosine, else balanced accuracy
    d["score"] = d["pearson"].fillna(d["circ_cos"]).fillna(d["balanced_accuracy"])
    return d.dropna(subset=["score"])


def panel(ax, g, value, err, xlabels, colour, title):
    x = np.arange(len(xlabels))
    for enc, h in g:
        ax.errorbar(x, h[value], yerr=h[err], fmt="o-", ms=2.5, lw=0.8, capsize=1.5, alpha=0.65,
                    color=colour, elinewidth=0.7)
    ax.set_xticks(x); ax.set_xticklabels(xlabels, rotation=90, fontsize=6)
    ax.set_title(title, fontsize=8, color=colour)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=6)


def by_class(d, key, order, labels, folder, stem, ylabel, heatmap):
    OUT.joinpath(folder).mkdir(parents=True, exist_ok=True)
    m = d.groupby(["source", key]).score.agg(["mean", "std"]).reset_index()
    for cls in ("seq", "struct", "multi"):
        encs = [e for e in m.source.unique() if DECLARED.get(e, "struct") == cls]
        fig, ax = plt.subplots(figsize=(4.2, 2.8))
        groups = []
        for e in encs:
            h = m[m.source == e].set_index(key).reindex(order)
            groups.append((e, h))
        panel(ax, groups, "mean", "std", labels, CLASS_COLOR[cls], f"{CLASS_NAME[cls]} ({len(encs)})")
        ax.set_ylabel(ylabel, fontsize=7)
        fig.tight_layout()
        fig.savefig(OUT / folder / f"{stem}_{CLASS_NAME[cls]}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    # the spread itself, as a heatmap over every cell
    piv = m.pivot(index="source", columns=key, values="std").reindex(columns=order)
    piv = piv.reindex(sorted(piv.index, key=lambda e: ("seq", "struct", "multi").index(DECLARED.get(e, "struct"))))
    fig, ax = plt.subplots(figsize=(0.32 * len(order) + 3.2, 0.17 * len(piv) + 1.6))
    im = ax.imshow(piv.values, cmap=TEAL, aspect="auto")
    ax.set_xticks(range(len(order))); ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(piv))); ax.set_yticklabels([DISP.get(e, e) for e in piv.index], fontsize=5.5)
    for t, e in zip(ax.get_yticklabels(), piv.index):
        t.set_color(CLASS_COLOR[DECLARED.get(e, "struct")])
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.ax.tick_params(labelsize=6)
    cb.set_label("seed s.d.", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / folder / f"{heatmap}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {folder}: {len(piv)} sources x {len(order)} columns, median sd {np.nanmedian(piv.values):.4f}")


def main():
    d = probe_scores()
    probes = [p for p in pd.read_csv(ROOT / "probing_results" / "floor_vs_encoders.csv").probe if p in set(d.probe)]
    by_class(d[d.probe.isin(probes)], "probe", probes, probes, "IIP", "appendix_uncertainty", "probe score",
             "appendix_std_heatmap")

    s = pd.read_csv(ROOT / "extract" / "scores2_long.csv")
    s = s[s.encoder.isin(released_panel())].rename(columns={"encoder": "source"})
    pv = s.groupby(["task", "source", "probe"]).val_loss.mean().reset_index()
    best = pv.loc[pv.groupby(["task", "source"]).val_loss.idxmin(), ["task", "source", "probe"]]
    s = s.merge(best, on=["task", "source", "probe"])
    tasks = [t for t in TASK_SHORT if t in set(s.task)] + [t for t in sorted(set(s.task)) if t not in TASK_SHORT]
    by_class(s, "task", tasks, [TASK_SHORT.get(t, t) for t in tasks], "downstream",
             "appendix_uncertainty_downstream", "downstream score", "appendix_std_heatmap_downstream")


if __name__ == "__main__":
    main()
