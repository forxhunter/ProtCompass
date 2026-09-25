"""Uncertainty over proteins for the held-out demand results, from the saved per-protein predictions.

Every interval in the released paper is a seed spread, which measures training instability conditional on the
proteins sampled. Review asked for the other source: resampling the evaluation units. This script provides it
without a second pass over the GPU, because scripts/addback_holdout.py wrote the per-protein prediction of every
arm and descriptor.

The resampling is paired. Within one replicate the same protein indices are used for every arm, every descriptor and
every encoder of a task, so the difference of differences that defines the repair contrast is computed on the same
proteins throughout and the encoder-to-encoder comparison is not disturbed by a different draw.

The unit is the protein. These cohorts are UniProt accessions and GFP variants without a homology grouping in the
manifest, so a homology-group bootstrap is not available here and is not claimed; what this measures is the
sampling of proteins, not of families.

Two null families are read when the prediction files carry them: the permuted sham of the released protocol and the
variance-matched null of scripts/matched_null.py. Every contrast is reported against each, on the same draws, so the
two differ only in which arm stands for "damaged but not in the property's directions".

  python holdout_bootstrap.py [--boot 2000] [--pred-dir ...] [--markdown | --latex]
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
CD = ROOT / "confirmation_downstream"
PRED = CD / "predictions"
KIND = {"membrane_soluble": "binary", "subcellular_localization": "clf",
        "fluorescence": "reg", "stability": "reg"}
LABEL = {"membrane_soluble": "membrane vs soluble", "subcellular_localization": "subcellular localisation",
         "fluorescence": "fluorescence", "stability": "stability"}


def metric(kind, y, p):
    if kind == "reg":
        r = spearmanr(y, p)[0]
        return float(r) if np.isfinite(r) else np.nan
    if kind == "binary":
        if len(np.unique(y)) < 2:
            return np.nan
        return float(roc_auc_score(y, p))
    return float((p == y).mean())


def load_task(task, pred_dir=None):
    """{encoder: (y, {arm|descriptor: prediction})} for one task, all files aligned on the same protein order."""
    out, ref_ids = {}, None
    for f in sorted((pred_dir or PRED).glob(f"{task}__*.npz")):
        enc = f.stem.split("__", 1)[1]
        z = np.load(f, allow_pickle=True)
        ids = np.asarray(z["ids"])
        if ref_ids is None:
            ref_ids = ids
        elif not np.array_equal(ids, ref_ids):
            raise SystemExit(f"{f.name}: protein order differs from the first file of this task")
        out[enc] = (z["y"], {k: z[k] for k in z.files if "|" in k})
    return out, ref_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--pred-dir", default=None)
    ap.add_argument("--out-prefix", default="holdout_bootstrap")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--latex", action="store_true")
    a = ap.parse_args()
    pred_dir = Path(a.pred_dir) if a.pred_dir else PRED
    rng = np.random.default_rng(0)
    rows = []

    for task, kind in KIND.items():
        data, ids = load_task(task, pred_dir)
        if not data:
            continue
        n = len(ids)
        keys = next(iter(data.values()))[1]
        fam = {p: sorted({k.split("|")[0] for k in keys if k.startswith(p)}) for p in ("sham", "null")}
        fam = {p: v for p, v in fam.items() if v}
        draws = rng.integers(0, n, size=(a.boot, n))          # one index set per replicate, shared everywhere

        for enc, (y, P) in data.items():
            def stat(idx):
                yy = y[idx]
                m = lambda key: metric(kind, yy, P[key][idx])
                out = {}
                for p, arms in fam.items():
                    suffix = "" if p == "sham" else f"_{p}"
                    out[f"effect{suffix}"] = float(np.mean([m(f"{s}|none") for s in arms])) - m("erased|none")
                    for q in ("named", "permuted", "other"):
                        if f"erased|{q}" not in P:
                            continue
                        real = m(f"erased|{q}") - m("erased|none")
                        ctrl = float(np.mean([m(f"{s}|{q}") - m(f"{s}|none") for s in arms]))
                        out[f"R_{q}{suffix}"] = real - ctrl
                return out

            point = stat(np.arange(n))
            boot = [stat(d) for d in draws]
            rec = dict(task=task, encoder=enc, n_proteins=n, n_boot=a.boot)
            for k, v in point.items():
                vals = np.array([b[k] for b in boot], dtype=float)
                vals = vals[np.isfinite(vals)]
                rec[k] = v
                rec[f"{k}_lo"] = float(np.percentile(vals, 2.5)) if len(vals) > 20 else np.nan
                rec[f"{k}_hi"] = float(np.percentile(vals, 97.5)) if len(vals) > 20 else np.nan
            rows.append(rec)
            msg = f"  {task:24s} {enc:18s}"
            for suffix in ("", "_null"):
                if f"effect{suffix}" not in rec:
                    continue
                tag = "sham" if suffix == "" else "null"
                msg += (f" | {tag} eff {rec[f'effect{suffix}']:+.4f}"
                        f"[{rec[f'effect{suffix}_lo']:+.4f},{rec[f'effect{suffix}_hi']:+.4f}]"
                        f" R {rec.get(f'R_named{suffix}', float('nan')):+.4f}"
                        f"[{rec.get(f'R_named{suffix}_lo', float('nan')):+.4f},"
                        f"{rec.get(f'R_named{suffix}_hi', float('nan')):+.4f}]")
            print(msg, flush=True)

    t = pd.DataFrame(rows)
    t.to_csv(CD / f"{a.out_prefix}.csv", index=False)

    # per task: the panel mean with an interval built from the same paired draws, encoder by encoder
    summ = []
    for task, g in t.groupby("task"):
        rec = dict(task=task, label=LABEL.get(task, task), n_encoders=len(g),
                   n_proteins=int(g.n_proteins.max()))
        for suffix in ("", "_null"):
            for base in ("effect", "R_named", "R_permuted", "R_other"):
                col = base + suffix
                if col not in g:
                    continue
                rec[col] = g[col].mean()
                lo = col + "_lo"
                rec[f"n_{col}_excl0"] = int((g[lo] > 0).sum()) if lo in g else 0
        summ.append(rec)
    S = pd.DataFrame(summ)
    S.to_csv(CD / f"{a.out_prefix}_summary.csv", index=False)

    if a.latex:
        print("% generated by scripts/holdout_bootstrap.py; do not edit by hand")
        print("\\begin{tabular}{lrrrrr}")
        print("\\toprule")
        print("task & proteins & effect & encoders whose interval excludes 0 & $R$ (named) & encoders with $R>0$ "
              "excluding 0 \\\\")
        print("\\midrule")
        for r in S.itertuples():
            print(f"{r.label} & {r.n_proteins} & ${r.effect:+.3f}$ & {r.n_effect_excl0} / {r.n_encoders} & "
                  f"${r.R_named:+.4f}$ & {r.n_R_named_excl0} / {r.n_encoders} \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")
        print(f"% paired protein bootstrap, {a.boot} replicates, the same draw across every arm, descriptor and "
              f"encoder of a task")
        return

    pd.set_option("display.width", 220)
    print("\n=== per task, paired protein bootstrap ===")
    print(S.round(4).to_string(index=False))
    print(f"\nwritten: {CD / (a.out_prefix + '.csv')} and {CD / (a.out_prefix + '_summary.csv')}")


if __name__ == "__main__":
    main()
