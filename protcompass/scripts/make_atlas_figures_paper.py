"""Atlas figures for the SI in the paper's palette, so they read as one family with Figures 2 and 3.

Layout follows the ESM Atlas convention: a grid of small multiples on a white ground, one panel per encoder, the
same proteins in every panel after Procrustes alignment, and no text inside the panels beyond the encoder name.
Colours follow the main figures: encoder classes are navy (sequence), orange (structure) and purple (multimodal),
the CKA matrix uses the teal colour map of Figure 2, and protein labels use a muted categorical set from the same
family. Encoder class comes from the paper's declared taxonomy, not from the atlas metadata.

  python make_atlas_figures_paper.py
Writes manuscript figures/atlas/{maps,cka,encoder_map}_<layer>.{png,pdf} for every layer of atlas/data_v4 (one layer per
scored downstream task) and cka_grid.{png,pdf}, the ten CKA matrices side by side.
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import json, math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

ATLAS = Path((PROTCOMPASS_ROOT + "/protcompass/atlas/data_v4/atlas.json"))
OUT = Path((PROTCOMPASS_ROOT + "/protcompass/paper/figures/atlas"))
OUT.mkdir(parents=True, exist_ok=True)

TEAL = LinearSegmentedColormap.from_list("paper_teal", ["#f2f9f9", "#aed9d9", "#5eb5b5", "#2b8c8c", "#144d4d"], N=256)
CLASS_COLOR = {"seq": "#1a4a8a", "struct": "#D2691E", "multi": "#7c4fa0"}
# protein-label palette: the teal and rose families of the Figure 2 and 3 colour maps plus two neutrals, chosen so
# that no protein label shares a hue with the three encoder-class colours used for borders and names
LABEL_PALETTE = ["#144d4d", "#a83a3e", "#5eb5b5", "#d4777a", "#8a8a3a", "#5a5a5a", "#c9a227", "#2e7d32",
                 "#e0a3c8", "#7b3f00"]
SEQ = ["esm1b", "prottrans_t5_xl_u50", "prottrans", "protbert_bfd", "proteinbert_bfd", "esm2", "esm2_8m", "esm2_35m",
       "esm2_150m", "esm2_650m", "esm2_3b", "ankh_base", "ankh"]
# encoders that are not in the released 33-encoder panel (audit failures and degenerate caches) are not drawn
EXCLUDE_PREFIX = ("egnn", "geobpe", "dmasif")
def in_panel(e):
    return e.get("in_paper_panel", True) and not e["key"].startswith(EXCLUDE_PREFIX)
def disp(e):
    return "ESM2" if e["name"].startswith("ESM2 (cache key") else e["name"][:22]
MULTI = ["prostt5", "cdconv", "saprot", "esm3", "foldvision"]
def cls(key):
    k = key.replace("_pretrained", "").replace("_random", "")
    return "seq" if k in SEQ else ("multi" if k in MULTI else "struct")
ORDER = {"seq": 0, "struct": 1, "multi": 2}
LAYER_TITLE = {"ec": "EC classification", "go_mf": "GO-MF", "go_bp": "GO-BP", "localisation": "Localisation",
               "homology": "Remote homology", "solubility": "Solubility", "membrane": "Membrane vs soluble",
               "stability": "Stability", "fluorescence": "Fluorescence", "affinity": "PPI affinity"}
plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"]})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig); print("wrote", name)


def maps(layer, name):
    encs = [e for e in layer["encoders"] if in_panel(e)]
    y = np.array(layer["proteins"]["label"])
    # one common frame for every panel, so the boxes are the same size and the aligned maps are comparable
    allxy = np.vstack([np.array(e["umap_aligned"]).reshape(-1, 2) for e in encs])
    lim = np.percentile(np.abs(allxy), 99.5) * 1.05
    order = sorted(range(len(encs)), key=lambda i: (ORDER[cls(encs[i]["key"])], -encs[i]["metrics"]["knn_purity"]))
    ncol = 7 if len(encs) > 12 else 5
    nrow = math.ceil(len(encs) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.75, nrow * 1.85))
    draw = np.random.default_rng(0).permutation(len(y))
    for ax, i in zip(axes.ravel(), order):
        e = encs[i]
        xy = np.array(e["umap_aligned"]).reshape(-1, 2)
        ax.scatter(xy[draw, 0], xy[draw, 1], c=[LABEL_PALETTE[k % len(LABEL_PALETTE)] for k in y[draw]],
                   s=1.2, linewidths=0, alpha=0.7, rasterized=True)
        c = CLASS_COLOR[cls(e["key"])]
        ax.set_title(disp(e) + (" (rand.)" if e["weights"].startswith("random") else ""),
                     color=c, fontsize=6.5, pad=2)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        for s in ax.spines.values():
            s.set_linewidth(0.8); s.set_color(c)
    for ax in axes.ravel()[len(order):]:
        ax.axis("off")
    handles = [Line2D([], [], marker="o", ls="", color=LABEL_PALETTE[k % len(LABEL_PALETTE)], label=lab, markersize=5)
               for k, lab in enumerate(layer["label_names"])]
    handles += [Line2D([], [], marker="s", ls="", color=c, label=lab, markersize=5)
                for lab, c in (("sequence", CLASS_COLOR["seq"]), ("structure", CLASS_COLOR["struct"]),
                               ("multimodal", CLASS_COLOR["multi"]))]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 7), frameon=False, fontsize=6.5,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(); save(fig, f"maps_{name}")


def cka(layer, name):
    encs = layer["encoders"]; keep = [i for i, e in enumerate(encs) if in_panel(e)]
    C = np.array(layer["cka"])[np.ix_(keep, keep)]
    order = layer["encoder_map"].get("cluster_order")
    idx = [i for i in order if i in keep] if order else keep
    pos = {i: k for k, i in enumerate(keep)}
    sel = [pos[i] for i in idx]
    C = C[np.ix_(sel, sel)]; names = [disp(encs[i]) for i in idx]; classes = [cls(encs[i]["key"]) for i in idx]
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    im = ax.imshow(C, vmin=0, vmax=1, cmap=TEAL, interpolation="nearest")
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=5.5); ax.set_yticklabels(names, fontsize=5.5)
    for t, c in zip(ax.get_xticklabels(), classes): t.set_color(CLASS_COLOR[c])
    for t, c in zip(ax.get_yticklabels(), classes): t.set_color(CLASS_COLOR[c])
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02); cb.ax.tick_params(labelsize=6)
    cb.set_label("linear CKA", fontsize=7)
    fig.tight_layout(); save(fig, f"cka_{name}")


def repel_labels(pts, labels, iters=400):
    """Move overlapping labels apart so every encoder name is legible; a thin line ties a moved label to its point."""
    span = np.ptp(pts, axis=0)
    w = np.array([0.018 * len(t) for t in labels]) * span[0]
    h = np.full(len(labels), 0.04 * span[1])
    pos = pts + np.stack([0.02 * span[0] * np.ones(len(pts)), 0.02 * span[1] * np.ones(len(pts))], 1)
    rng = np.random.default_rng(0)
    for _ in range(iters):
        moved = False
        for a in range(len(pos)):
            for b in range(a + 1, len(pos)):
                dx, dy = pos[b] - pos[a]
                ox = (w[a] + w[b]) / 2 - abs(dx); oy = (h[a] + h[b]) / 2 - abs(dy)
                if ox > 0 and oy > 0:
                    moved = True
                    if oy < ox:
                        s = 0.5 * oy * (1 if dy >= 0 else -1) + 1e-3 * span[1] * rng.standard_normal()
                        pos[a, 1] -= s; pos[b, 1] += s
                    else:
                        s = 0.5 * ox * (1 if dx >= 0 else -1)
                        pos[a, 0] -= s; pos[b, 0] += s
        # keep a label near its own point: pull back gently
        pos += 0.02 * (pts + 0.02 * span - pos)
        if not moved:
            break
    return pos


def encoder_map(layer, name):
    encs = layer["encoders"]; keep = [i for i, e in enumerate(encs) if in_panel(e)]
    xy = np.array(layer["encoder_map"]["mds"]).reshape(-1, 2)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for i in keep:
        e = encs[i]; c = CLASS_COLOR[cls(e["key"])]; rnd = e["weights"].startswith("random")
        ax.scatter(*xy[i], s=44, color="white" if rnd else c, edgecolors=c, linewidths=1.3, zorder=3)
    for i, (tx, ty) in zip(keep, repel_labels(xy[keep], [disp(encs[i])[:18] for i in keep])):
        e = encs[i]; c = CLASS_COLOR[cls(e["key"])]
        ax.annotate(disp(e)[:18], xy[i], (tx, ty), textcoords="data", fontsize=5.5, color=c,
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.4, alpha=0.6) if np.hypot(tx - xy[i][0], ty - xy[i][1]) > 0.06 * np.ptp(xy[:, 0]) else None)
    handles = [Line2D([], [], marker="o", ls="", color=c, label=lab, markersize=6)
               for lab, c in (("sequence", CLASS_COLOR["seq"]), ("structure", CLASS_COLOR["struct"]),
                              ("multimodal", CLASS_COLOR["multi"]))]
    handles.append(Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor="#444", label="random weights", markersize=6))
    ax.legend(handles=handles, frameon=False, fontsize=6.5, loc="best")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("MDS on 1 - CKA", fontsize=7)
    for s in ax.spines.values(): s.set_linewidth(0.4); s.set_color("#999")
    fig.tight_layout(); save(fig, f"encoder_map_{name}")


def cka_grid(layers):
    """All layers' CKA matrices in one figure: each ordered by its own clustering, class strips on the edges."""
    names = [n for n in LAYER_TITLE if n in layers]
    ncol = 5; nrow = math.ceil(len(names) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.6, 1.65 * nrow + 0.3))
    for ax, name in zip(axes.ravel(), names):
        layer = layers[name]; encs = layer["encoders"]
        keep = [i for i, e in enumerate(encs) if in_panel(e)]
        order = layer["encoder_map"].get("cluster_order") or keep
        idx = [i for i in order if i in keep]
        C = np.array(layer["cka"])[np.ix_(idx, idx)]
        im = ax.imshow(C, vmin=0, vmax=1, cmap=TEAL, interpolation="nearest")
        n = len(idx)
        for k, i in enumerate(idx):
            c = CLASS_COLOR[cls(encs[i]["key"])]
            ax.add_patch(plt.Rectangle((k - 0.5, -2.3), 1, 1.4, color=c, clip_on=False, lw=0))
            ax.add_patch(plt.Rectangle((-2.3, k - 0.5), 1.4, 1, color=c, clip_on=False, lw=0))
        iu = np.triu_indices(n, 1)
        ax.set_title(f"{LAYER_TITLE[name]}\nmean CKA {C[iu].mean():.2f}", fontsize=6.2, pad=8, linespacing=1.1)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.subplots_adjust(left=0.03, right=0.90, top=0.90, bottom=0.03, wspace=0.28, hspace=0.45)
    cax = fig.add_axes([0.925, 0.2, 0.012, 0.6])
    cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=6); cb.set_label("linear CKA", fontsize=7)
    save(fig, "cka_grid")


if __name__ == "__main__":
    d = json.load(open(ATLAS))
    for name, layer in d["layers"].items():
        n = sum(in_panel(e) for e in layer["encoders"])
        print(f"{name}: {n} encoders drawn of {len(layer['encoders'])} in the atlas")
        maps(layer, name); cka(layer, name); encoder_map(layer, name)
    cka_grid(d["layers"])
