"""
analyze_ifamily_correlation.py — I-family × downstream task Spearman ρ matrix.

For each (I-family, task) pair, computes Spearman ρ across encoders between:
  - I-family MLP-2 composite score  (from InvariantsBench sweep_v4_full)
  - Best downstream task score mean  (from downstream_tasks/results/)

Outputs:
  results/ifamily_task_correlation.json   full matrix + pvals + bootstrap CI
  results/heatmap_all.png
  results/heatmap_pretrained.png
  results/heatmap_random_init.png
  results/top_correlations.txt

Usage:
  PYTHONPATH=. python downstream_tasks/analyze_ifamily_correlation.py
  PYTHONPATH=. python downstream_tasks/analyze_ifamily_correlation.py --n-bootstrap 5000
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import sys
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from downstream_tasks.config import (
    IFAMILY_PROBE_TASKS, PRETRAINED_SET, RANDOM_SET, RES_DIR, TASKS,
)

SWEEP_DIR = ROOT / "outputs" / "sweep_v4_full"

# I5 excluded from primary metric (sequence leakage)
IFAMILIES_PRIMARY = ["I1", "I2", "I3", "I4", "I6", "I7", "I8"]
IFAMILIES_ALL     = ["I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8"]

TASK_ORDER = [
    "secondary_structure", "contact_prediction", "remote_homology",
    "fluorescence", "stability",
    "ec_classification", "go_mf", "go_bp",
    "ppi_site", "metal_binding",
    "subcellular_localization", "membrane_soluble",
]
TASK_LABELS = {
    "secondary_structure":      "SS\n(T1)",
    "contact_prediction":       "Contact\n(T2)",
    "remote_homology":          "Homology\n(T3)",
    "fluorescence":             "Fluoresc\n(T7)",
    "stability":                "Stability\n(T8)",
    "ec_classification":        "EC\n(T4)",
    "go_mf":                    "GO-MF\n(T5)",
    "go_bp":                    "GO-BP\n(T6)",
    "ppi_site":                 "PPI-site\n(T9)",
    "metal_binding":            "Metal\n(T10)",
    "subcellular_localization": "SubcellLoc\n(T11)",
    "membrane_soluble":         "Membrane\n(T12)",
}
CATEGORY_SPANS = [
    ("Structure",    [0, 1, 2]),
    ("Fitness",      [3, 4]),
    ("Function",     [5, 6, 7]),
    ("Interaction",  [8, 9]),
    ("Localization", [10, 11]),
]


# ── Load I-family scores ──────────────────────────────────────────────────────

def load_ifamily_scores() -> dict:
    """Returns {encoder_name: {I1: float, ..., I8: float}}."""
    enc_scores = {}
    for enc_dir in SWEEP_DIR.iterdir():
        if not enc_dir.is_dir():
            continue
        agg = enc_dir / "aggregated_results.json"
        if not agg.exists():
            continue
        with open(agg) as f:
            data = json.load(f)
        mlp2 = data.get("mlp_2", {})
        if not mlp2:
            continue
        family_scores = {}
        for fam, probe_tasks in IFAMILY_PROBE_TASKS.items():
            vals = [mlp2.get(t, {}).get("mean") for t in probe_tasks]
            vals = [v for v in vals if v is not None]
            if vals:
                family_scores[fam] = float(np.mean(vals))
        enc_scores[enc_dir.name] = family_scores
    print(f"[I-family] {len(enc_scores)} encoders loaded")
    return enc_scores


# ── Load downstream scores ────────────────────────────────────────────────────

def load_downstream_scores() -> dict:
    """Returns {encoder_name: {task: best_mean_score}}."""
    enc_task = {}
    for enc_dir in RES_DIR.iterdir():
        if not enc_dir.is_dir():
            continue
        enc = enc_dir.name
        scores = {}
        for task_dir in enc_dir.iterdir():
            if not task_dir.is_dir():
                continue
            task = task_dir.name
            agg  = task_dir / "aggregated.json"
            if not agg.exists():
                continue
            with open(agg) as f:
                data = json.load(f)
            bm = data.get("best_mean")
            if bm is not None:
                scores[task] = float(bm)
        if scores:
            enc_task[enc] = scores
    print(f"[Downstream] {len(enc_task)} encoders with results")
    return enc_task


# ── Correlation computation with bootstrap CI ─────────────────────────────────

def spearman_with_bootstrap(x: np.ndarray, y: np.ndarray,
                             n_bootstrap: int = 2000,
                             ci: float = 0.95) -> dict:
    """Spearman ρ with bootstrap CI."""
    n = len(x)
    if n < 5:
        return {"rho": None, "pval": None, "ci_lo": None, "ci_hi": None, "n": n}

    rho, pval = spearmanr(x, y)

    rng      = np.random.RandomState(42)
    boot_rho = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        r, _ = spearmanr(x[idx], y[idx])
        if not np.isnan(r):
            boot_rho.append(r)

    alpha    = (1 - ci) / 2
    boot_rho = np.array(boot_rho)
    ci_lo    = float(np.percentile(boot_rho, 100 * alpha))
    ci_hi    = float(np.percentile(boot_rho, 100 * (1 - alpha)))

    return {
        "rho":   round(float(rho), 4),
        "pval":  round(float(pval), 5),
        "ci_lo": round(ci_lo, 4),
        "ci_hi": round(ci_hi, 4),
        "n":     int(n),
    }


def compute_matrix(ifam_scores: dict, ds_scores: dict,
                   encoder_subset=None, n_bootstrap: int = 2000) -> dict:
    encoders = sorted(set(ifam_scores) & set(ds_scores))
    if encoder_subset is not None:
        encoders = [e for e in encoders if e in encoder_subset]
    print(f"  n encoders = {len(encoders)}: {encoders}")

    matrix = {}
    for fam in IFAMILIES_ALL:
        matrix[fam] = {}
        x_all = np.array([ifam_scores[e].get(fam, np.nan) for e in encoders])
        valid_x = ~np.isnan(x_all)

        for task in TASK_ORDER:
            y_all   = np.array([ds_scores[e].get(task, np.nan) for e in encoders])
            valid   = valid_x & ~np.isnan(y_all)
            x_valid = x_all[valid]
            y_valid = y_all[valid]

            if valid.sum() < 5:
                matrix[fam][task] = {"rho": None, "pval": None,
                                      "ci_lo": None, "ci_hi": None, "n": int(valid.sum())}
            else:
                matrix[fam][task] = spearman_with_bootstrap(x_valid, y_valid, n_bootstrap)

    return matrix, encoders


# ── Heatmap ───────────────────────────────────────────────────────────────────

def save_heatmap(matrix: dict, encoders: list, title: str, out_path: Path,
                 families=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  [skip heatmap] matplotlib not available")
        return

    if families is None:
        families = IFAMILIES_PRIMARY

    tasks   = [t for t in TASK_ORDER if t in TASK_LABELS]
    xlabels = [TASK_LABELS[t] for t in tasks]
    ylabels = [f"{f}\n({','.join(IFAMILY_PROBE_TASKS[f])[:18]}...)" for f in families]

    data = np.full((len(families), len(tasks)), np.nan)
    for i, fam in enumerate(families):
        for j, task in enumerate(tasks):
            v = matrix.get(fam, {}).get(task, {})
            if isinstance(v, dict) and v.get("rho") is not None:
                data[i, j] = v["rho"]

    fig, ax = plt.subplots(figsize=(15, 5.5))
    cmap = plt.cm.RdBu_r
    im   = ax.imshow(data, cmap=cmap, vmin=-1.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(xlabels, rotation=0, ha="center", fontsize=8)
    ax.set_yticks(range(len(families)))
    ax.set_yticklabels(families, fontsize=9)

    # Annotate with ρ and significance stars
    for i in range(len(families)):
        for j in range(len(tasks)):
            val = data[i, j]
            if np.isnan(val):
                continue
            pval = matrix.get(families[i], {}).get(tasks[j], {}).get("pval")
            star = "***" if pval and pval < 0.001 else \
                   "**"  if pval and pval < 0.01  else \
                   "*"   if pval and pval < 0.05  else ""
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}{star}", ha="center", va="center",
                    color=color, fontsize=7, fontweight="bold" if star else "normal")

    plt.colorbar(im, ax=ax, label="Spearman ρ", shrink=0.85, pad=0.02)

    # Category dividers and labels
    spans = {cat: idxs for cat, idxs in CATEGORY_SPANS}
    for cat, idxs in CATEGORY_SPANS:
        ax.axvline(max(idxs) + 0.5, color="black", linewidth=1.2, alpha=0.6)
        mid = (min(idxs) + max(idxs)) / 2
        ax.text(mid, -0.8, cat, ha="center", va="top", fontsize=8,
                fontweight="bold", transform=ax.get_xaxis_transform())

    ax.set_title(f"{title}\n(n={len(encoders)} encoders; * p<0.05, ** p<0.01, *** p<0.001)",
                 fontsize=10, pad=14)
    ax.set_ylabel("Invariant family", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {out_path.name}")


# ── Summary report ────────────────────────────────────────────────────────────

def print_top_correlations(matrix: dict, label: str, min_abs_rho: float = 0.5):
    lines = [f"\n=== Top correlations — {label} (|ρ| ≥ {min_abs_rho}) ==="]
    rows  = []
    for fam in IFAMILIES_PRIMARY:
        for task in TASK_ORDER:
            entry = matrix.get(fam, {}).get(task, {})
            rho   = entry.get("rho") if isinstance(entry, dict) else None
            if rho is not None and abs(rho) >= min_abs_rho:
                pval = entry.get("pval", "?")
                ci   = f"[{entry.get('ci_lo', '?'):.2f}, {entry.get('ci_hi', '?'):.2f}]"
                n    = entry.get("n", "?")
                rows.append((abs(rho), f"  {fam} × {task:<28}  ρ={rho:+.3f}  p={pval:.4f}  CI={ci}  n={n}"))
    rows.sort(reverse=True)
    lines.extend(r for _, r in rows)
    txt = "\n".join(lines)
    print(txt)
    return txt


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--min-rho",     type=float, default=0.5)
    args = parser.parse_args()

    ifam_scores = load_ifamily_scores()
    ds_scores   = load_downstream_scores()

    if not ds_scores:
        print("\nNo downstream results yet. Run evaluate_tasks.py first.")
        print("I-family scores available for:", sorted(ifam_scores.keys()))
        return

    RES_DIR.mkdir(parents=True, exist_ok=True)
    output = {}

    for label, subset in [
        ("all",         None),
        ("pretrained",  PRETRAINED_SET),
        ("random_init", RANDOM_SET),
    ]:
        print(f"\n{'='*60}\n{label.upper()}\n{'='*60}")
        mat, encs = compute_matrix(ifam_scores, ds_scores,
                                   encoder_subset=subset,
                                   n_bootstrap=args.n_bootstrap)
        output[label] = {"matrix": mat, "encoders": encs}

        save_heatmap(mat, encs, f"I-family × Downstream Task ({label})",
                     RES_DIR / f"heatmap_{label}.png")

    # Save full JSON
    out_json = RES_DIR / "ifamily_task_correlation.json"
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_json}")

    # Print top correlations for pretrained regime
    txt_all = print_top_correlations(
        output["pretrained"]["matrix"], "pretrained", args.min_rho)
    txt_rnd = print_top_correlations(
        output["random_init"]["matrix"], "random_init", args.min_rho)

    with open(RES_DIR / "top_correlations.txt", "w") as f:
        f.write(txt_all + "\n" + txt_rnd)
    print(f"\nSaved: {RES_DIR/'top_correlations.txt'}")


if __name__ == "__main__":
    main()
