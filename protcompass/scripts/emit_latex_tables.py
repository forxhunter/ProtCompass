"""LaTeX bodies for the tables added in response to the round-1 review.

Each one is generated from the released files so that the paper cannot drift from them:

  downstream-reference  every task against a non-learned composition-and-length reference
  group-controls        the sequence-vs-coordinate comparison under each restriction that removes a confound
  prediction-test       whether the demanded property predicts downstream score beyond the erasure that defines it
  train-size            the group gap as the training budget is reduced

  python emit_latex_tables.py downstream-reference
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
PY = sys.executable


def _open(cols, header):
    print("% generated file: the whole tabular, so the caller never inputs rows into an open tabular")
    print("\\begin{tabular}{" + cols + "}")
    print("\\toprule")
    print(header + " \\\\")
    print("\\midrule")


def _close():
    print("\\bottomrule")
    print("\\end{tabular}")


def run(script, *args):
    return subprocess.run([PY, str(HERE / script), *args], capture_output=True, text=True, check=True).stdout


def downstream_reference():
    sys.path.insert(0, str(HERE))
    from downstream_reference_table import table
    t = table()
    _open("llrrrr", "task & metric & composition reference & median encoder & best encoder & encoders above")
    for r in t.itertuples():
        print(f"{r.task} & {r.metric} & {r.reference:.3f} & {r.median_encoder:.3f} & {r.best_encoder:.3f} & "
              f"{r.above} / {r.n} \\\\")
    _close()
    near = list(t.task[t.margin_median < 0.02])
    print(f"% median encoder within 0.02 of the reference on {len(near)} of {len(t)}: {', '.join(near)}")


def group_controls():
    sys.path.insert(0, str(HERE))
    from group_controls import summarize
    t = summarize()
    _open("lccccc", "restriction & encoders (seq / coord) & families & tasks where sequence leads & mean gap & "
                    "tasks $p<0.05$")
    for name, g in t.groupby("restriction", sort=False):
        print(f"{name} & {g.n_seq.iloc[0]} / {g.n_crd.iloc[0]} & {g.fam_seq.iloc[0]} / {g.fam_crd.iloc[0]} & "
              f"{int(g.seq_leads.sum())} / {len(g)} & {g.gap.mean():+.3f} & "
              f"{int((g.fam_p < 0.05).sum())} / {len(g)} \\\\")
    _close()


def prediction_test():
    d = pd.read_csv(HERE.parent / "erasure" / "demand_prediction_test.csv")
    st = pd.read_csv(HERE.parent / "erasure" / "causal_demand_v2.csv")
    lab = {"ec_classification": "EC", "go_mf": "GO-MF", "go_bp": "GO-BP",
           "subcellular_localization": "localisation", "remote_homology": "homology", "solubility": "solubility",
           "stability": "stability", "fluorescence": "fluorescence", "ppi_affinity": "affinity",
           "membrane_soluble": "membrane"}
    _open("lllll", "task & demanded property & partial $\\rho$, demanded & partial $\\rho$, the others & "
                   "demanded above all others")
    n_ok = 0
    for task, g in d.groupby("task"):
        if not g.demanded.any():
            continue
        dm, un = g[g.demanded], g[~g.demanded]
        fmt = lambda v: (f"{v.min():+.2f} to {v.max():+.2f}" if len(v) > 1 else f"{v.iloc[0]:+.2f}")
        ok = dm.rho_partial.min() > un.rho_partial.max()
        n_ok += bool(ok)
        print(f"{lab.get(task, task)} & {', '.join(sorted(dm.concept))} & {fmt(dm.rho_partial)} & "
              f"{fmt(un.rho_partial)} & {'yes' if ok else 'no'} \\\\")
    _close()
    print(f"% the demanded property outranks every undemanded one on {n_ok} tasks")


def recoverability():
    import numpy as np
    d = pd.read_csv(HERE.parent / "erasure" / "recoverability_check.csv")
    g = d.groupby("concept").agg(lin_b=("linear_before", "median"), lin_a=("linear_after", "median"),
                                 mlp_b=("mlp_before", "median"), mlp_a=("mlp_after", "median"), n=("task", "size"))
    _open("lrrrrr", "erased property & linear, before & linear, after & MLP, before & MLP, after & cells")
    for c, r in g.iterrows():
        print(f"{c} & {r.lin_b:.3f} & {r.lin_a:.3f} & {r.mlp_b:.3f} & {r.mlp_a:.3f} & {int(r.n)} \\\\")
    _close()
    keep_l = (d.linear_after.clip(lower=0) / d.linear_before.clip(lower=1e-6)).median()
    keep_m = (d.mlp_after.clip(lower=0) / d.mlp_before.clip(lower=1e-6)).median()
    print(f"% median fraction of the readout kept: linear {keep_l:.2f}, MLP {keep_m:.2f}")


def nonlinear_control():
    # gated means (encoders whose erasure verified), the set the rule is computed on, as in positive_control()
    d = pd.read_csv(HERE.parent / "erasure" / "positive_control_nonlinear.csv")
    v = pd.read_csv(HERE.parent / "erasure" / "positive_control_nonlinear_verdict.csv")
    ok = d[d.erasure_ok.fillna(False).astype(bool)]
    piv = ok.pivot_table(index="concept", columns="planted", values="effect", aggfunc="mean")
    cnt = ok.pivot_table(index="concept", columns="planted", values="effect", aggfunc="count")
    cols = [c for c in ["amino-acid composition", "global shape", "none"] if c in piv.columns]
    n_all = int(d.groupby(["concept", "planted"]).encoder.nunique().max())
    named = {c: set(v.concept[(v.planted == c) & v.named]) for c in cols}
    _open("l" + "r" * len(cols), "erased property & " + " & ".join(f"planted {c}, nonlinearly" if c != "none"
                                                                   else "planted nothing" for c in cols))
    for concept, r in piv[cols].iterrows():
        cells = []
        for c in cols:
            val = f"{r[c]:+.3f}"
            if int(cnt.loc[concept, c]) < n_all:
                val += f" ($n={int(cnt.loc[concept, c])}$)"
            cells.append(f"\\textbf{{{val}}}" if concept in named[c] else val)
        print(f"{concept} & " + " & ".join(cells) + " \\\\")
    _close()
    print(f"% bold marks what the rule names; cells are means over the encoders whose erasure verified, n = {n_all} "
          f"unless stated")


def probing_restricted():
    import numpy as np
    t = pd.read_csv(HERE.parent / "probing_results" / "probing_restricted.csv")
    _open("lrrrr", "probe & floor & all coordinate encoders below & pretrained $\\ge$100M below & best of those "
                   "minus floor")
    for r in t.itertuples():
        print(f"{r.probe} & {r.floor:.3f} & {r.below_all} / {r.n_all} & {r.below_big} / {r.n_big} & "
              f"{r.gap_big:+.3f} \\\\")
    _close()
    print(f"% {t.below_all.sum()} of {t.n_all.sum()} over all; {t.below_big.sum()} of {t.n_big.sum()} restricted")


def demand_restricted():
    t = pd.read_csv(HERE.parent / "erasure" / "demand_restricted.csv")
    _open("lllll", "task & encoders kept & named over all encoders & named over the kept ones & detection bound, "
                   "all $\\to$ kept")
    for r in t.itertuples():
        print(f"{r.task} & {r.n_kept} / {r.n_all} & {r.named_all} & {r.named_kept} & "
              f"{r.bound_all:+.3f} $\\to$ {r.bound_kept:+.3f} \\\\")
    _close()
    print(f"% named set unchanged on {(t.named_all == t.named_kept).sum()} of {len(t)} tasks")


def strong_floors():
    """Fold architecture against the two floors added after the round-1 review, next to the eight built before."""
    import glob
    FL = ["ref_length", "ref_position", "ref_aa", "ref_seq_window", "ref_coords", "ref_backbone", "ref_global",
          "ref_pairhist"]
    d = pd.read_csv(HERE.parent / "probing_results" / "architecture_all_floors.csv")
    d = d[d.class_set == "common11"]
    extra = pd.concat([pd.read_csv(f) for f in
                       sorted(glob.glob(str(HERE.parent / "probing_results" / "architecture_strong_floors_*.csv")))],
                      ignore_index=True)
    # balanced accuracy, averaged over the seeds of the best readout, which is the metric and the aggregation the
    # rest of the paper uses for this probe. An earlier version of this table took the max over seeds of raw
    # accuracy, which is a different quantity and did not reconcile with the 15-seed interval in the main text.
    _open("lrrrr", "split & best of the eight floors & alignment 1-NN & spectral floor & best encoder")
    for split in ("superfamily", "topology"):
        m = d[d.split == split].groupby(["source", "readout"], as_index=False).balanced_accuracy.mean()
        g = m.loc[m.groupby("source").balanced_accuracy.idxmax()].set_index("source").balanced_accuracy
        fl, enc = g[g.index.isin(FL)], g[~g.index.isin(FL)]
        e = extra[extra.split == split].groupby("source").balanced_accuracy.max()
        al = e.get("ref_align_1nn", float("nan"))
        sp = e.get("ref_spectral", float("nan"))
        cells = [f"{fl.max():.3f} ({fl.idxmax().replace('_', chr(92) + '_')})",
                 f"{al:.3f}" if al == al else "--", f"{sp:.3f}" if sp == sp else "--",
                 f"{enc.max():.3f} ({enc.idxmax().replace('_', chr(92) + '_')})"]
        print(f"{split}-disjoint & " + " & ".join(cells) + " \\\\")
    _close()


def positive_control():
    d = pd.read_csv(HERE.parent / "erasure" / "positive_control.csv")
    v = pd.read_csv(HERE.parent / "erasure" / "positive_control_verdict.csv")
    # cells are means over the encoders whose erasure verified, which is the set the paper's rule (and the verdict
    # file) is computed on; an earlier version averaged over all twelve and disagreed with the text on global shape
    ok = d[d.erasure_ok.fillna(False).astype(bool)]
    piv = ok.pivot_table(index="concept", columns="planted", values="effect", aggfunc="mean")
    cnt = ok.pivot_table(index="concept", columns="planted", values="effect", aggfunc="count")
    cols = [c for c in ["amino-acid composition", "secondary structure content", "global shape", "none"]
            if c in piv.columns]
    piv, cnt = piv[cols], cnt[cols]
    n_all = int(d.groupby(["concept", "planted"]).encoder.nunique().max())
    named = {c: set(v.concept[(v.planted == c) & v.named]) for c in cols}
    _open("l" + "r" * len(cols), "erased property & " + " & ".join(f"planted {c}" for c in cols))
    for concept, r in piv.iterrows():
        cells = []
        for c in cols:
            val = f"{r[c]:+.3f}"
            if int(cnt.loc[concept, c]) < n_all:
                val += f" ($n={int(cnt.loc[concept, c])}$)"
            cells.append(f"\\textbf{{{val}}}" if concept in named[c] else val)
        print(f"{concept} & " + " & ".join(cells) + " \\\\")
    _close()
    print(f"% bold marks the properties the paper's own rule names; cells are means over the encoders whose erasure "
          f"verified, n = {n_all} unless stated")


def discard():
    import numpy as np
    from scipy.stats import spearmanr
    d = pd.read_csv(HERE.parent / "probing_results" / "discard_score.csv", index_col=0)
    sys.path.insert(0, str(HERE))
    from make_panels_original_style import DISP
    _open("llrrr", "encoder & class & gap to its floor & width & parameters")
    for e, r in d.iterrows():
        par = f"{int(r.params) / 1e6:.0f}M" if r.params >= 1e6 else (f"{int(r.params) / 1e3:.0f}K" if r.params else "--")
        print(f"{DISP.get(e, e)} & {r['class']} & {r.discard:+.3f} & {int(r.dim)} & {par} \\\\")
    _close()
    c = d[(d["class"] != "seq") & (d.params > 0)]
    print(f"% coordinate side: vs log params rho {spearmanr(np.log(c.params), c.discard).statistic:+.2f}")


def capacity():
    sys.path.insert(0, str(HERE))
    import capacity_control as C
    d = C.scores(); uses = C.audit()
    import numpy as np
    probes = list(pd.read_csv(HERE.parent / "probing_results" / "capacity_control.csv").probe)
    t = pd.read_csv(HERE.parent / "probing_results" / "capacity_control.csv")
    lab = {"phi_circular": "backbone $\\phi$", "psi_circular": "backbone $\\psi$",
           "local_curvature": "local curvature", "dssp_3": "DSSP-3", "dssp_8": "DSSP-8",
           "core_surface": "core/surface", "contact_density": "contact density",
           "distance_bins": "CA distance bins", "contact_order_len_resid": "contact order",
           "radius_of_gyration_len_resid": "radius of gyration", "asphericity_len_resid": "asphericity",
           "rsa": "RSA", "rsa_bins": "RSA bins", "bfactor": "B-factor",
           "architecture_label": "fold architecture"}
    _open("lrrrc", "probe & score per log width & floor $-$ best coordinate encoder & after removing width & "
                   "width-matched pairs the floor leads")
    for r in t.itertuples():
        print(f"{lab.get(r.probe, r.probe)} & {r.slope:+.3f} & {r.gap:+.3f} & {r.gap_resid:+.3f} & "
              f"{r.pairs_floor_ahead} / {r.n_pairs} \\\\")
    _close()
    same = int((np.sign(t.gap) == np.sign(t.gap_resid)).sum())
    print(f"% sign unchanged on {same} of {len(t)}")


def train_size():
    sys.path.insert(0, str(HERE))
    from train_size_sweep import load, selected, LABEL
    from group_controls import meta
    grp = meta().group
    d = load()
    rows = []
    for frac, g in d.groupby("train_frac"):
        s = selected(g).reset_index()
        s["group"] = [grp.get(e, "coordinate-side") for e in s.encoder]
        for task, h in s.groupby("task"):
            a_, b_ = h.score[h.group == "sequence"], h.score[h.group == "coordinate-side"]
            if len(a_) >= 2 and len(b_) >= 2:
                rows.append(dict(frac=frac, task=LABEL.get(task, task), gap=a_.mean() - b_.mean()))
    piv = pd.DataFrame(rows).pivot(index="task", columns="frac", values="gap")
    _open("lrrr", "task & 25\\% of training rows & 50\\% & 100\\%")
    for task, r in piv.iterrows():
        print(f"{task} & " + " & ".join(f"{r[c]:+.3f}" for c in piv.columns) + " \\\\")
    _close()
    lo, hi = piv[piv.columns.min()], piv[piv.columns.max()]
    print(f"% mean |gap change| {float((hi - lo).abs().mean()):.3f}; sign preserved on "
          f"{int((np.sign(lo) == np.sign(hi)).sum())} of {len(piv)}")


if __name__ == "__main__":
    {"downstream-reference": downstream_reference, "group-controls": group_controls,
     "prediction-test": prediction_test, "train-size": train_size,
     "capacity": capacity, "discard": discard, "positive-control": positive_control,
     "strong-floors": strong_floors, "probing-restricted": probing_restricted,
     "demand-restricted": demand_restricted, "recoverability": recoverability,
     "nonlinear-control": nonlinear_control}[sys.argv[1]]()
