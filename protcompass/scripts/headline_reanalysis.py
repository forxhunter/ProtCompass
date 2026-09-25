"""Re-reading the headline comparisons on existing results (no model is fitted or scored here).

Four analyses of the symmetric probing cells and the downstream scores, answering the round-6 review:
  1. probing counts split by whether the label's best floor is saturated (>= 0.95) or contained in a floor, for all
     coordinate encoders, the pretrained ones and the pretrained ones of at least 100M parameters;
  2. the gap to the input-matched floor regressed on log parameters, pretraining and reading coordinates, with label
     fixed effects and standard errors clustered by architecture family;
  3. on the undetermined labels, which encoders beat every one of the ten written floors once ESM3 (pretrained with
     secondary-structure and accessibility tracks) and the side-chain readers are set aside;
  4. the large-pretrained downstream comparison: the smallest p a two-sided family-level Mann-Whitney test can reach
     with six and four families, and a bootstrap interval over families for the mean gap over the ten tasks.

  python headline_reanalysis.py
Outputs: symmetric/headline_reanalysis.json, symmetric/headline_counts_by_saturation.csv,
         symmetric/gap_regression.csv, symmetric/undetermined_without_esm3.csv
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import glob
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
sys.path.insert(0, str(ROOT / "scripts"))
from group_controls import FAMILY, meta  # noqa: E402

SAT = ["phi_circular", "psi_circular", "local_curvature", "radius_of_gyration_len_resid", "asphericity_len_resid"]
TAUT = ["contact_density", "contact_order_len_resid"]
OPEN = ["dssp_3", "dssp_8", "core_surface", "distance_bins"]
UNDET = ["rsa", "rsa_bins", "bfactor"]
TEN = ["ref_length", "ref_position", "ref_aa", "ref_seq_window", "ref_seq_expert", "ref_seq_global",
       "ref_coords", "ref_backbone", "ref_global", "ref_pairhist"]
SIDECHAIN = ["gcpnet_random", "gcpnet_pretrained", "foldvision"]


def counts(c, name):
    return dict(subset=name, cells=len(c), below=int((c.state == "below").sum()),
                unresolved=int((c.state == "unresolved").sum()), above=int((c.state == "above").sum()))


def cluster_ols(y, X, groups):
    """OLS with CR1 cluster-robust standard errors."""
    XtX = np.linalg.pinv(X.T @ X)
    b = XtX @ X.T @ y
    e = y - X @ b
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in np.unique(groups):
        k = groups == g
        s = X[k].T @ e[k]
        meat += np.outer(s, s)
    G, n, p = len(np.unique(groups)), len(y), X.shape[1]
    V = XtX @ meat @ XtX * (G / (G - 1)) * ((n - 1) / (n - p))
    return b, np.sqrt(np.diag(V)), G


def main():
    out = {}
    m = meta()
    cells = pd.read_csv(ROOT / "symmetric" / "cells_superfamily.csv")
    cells = cells[cells.rule == "admissible"].copy()
    cells["weights"] = cells.encoder.map(m.weights)
    cells["params"] = cells.encoder.map(m.params)
    crd = cells[cells.coordinates]
    seq = cells[~cells.coordinates]

    # 1. counts by saturation and subgroup
    rows = []
    for lab, probes in (("saturated", SAT), ("contained", TAUT), ("open", OPEN)):
        for sub, mask in (("all coordinate", crd.index == crd.index),
                          ("pretrained coordinate", crd.weights == "pretrained"),
                          ("pretrained coordinate, $\\geq$100M", (crd.weights == "pretrained") & (crd.params >= 100e6))):
            c = crd[mask & crd.probe.isin(probes)]
            rows.append(dict(labels=lab, **counts(c, sub)))
        rows.append(dict(labels=lab, **counts(seq[seq.probe.isin(probes)], "sequence-only")))
    C = pd.DataFrame(rows)
    C.to_csv(ROOT / "symmetric" / "headline_counts_by_saturation.csv", index=False)
    out["counts"] = C.to_dict("records")
    print(C.to_string(index=False))

    # 2. regression of the gap on the determined labels no floor contains
    d = cells[cells.determined & ~cells.probe.isin(TAUT)].dropna(subset=["params"]).copy()
    d = d[d.params > 0]
    d["log10_params"] = np.log10(d.params)
    d["pretrained"] = (d.weights == "pretrained").astype(float)
    d["reads_coordinates"] = d.coordinates.astype(float)
    labs = sorted(d.probe.unique())
    X = np.column_stack([np.ones(len(d)), d.log10_params, d.pretrained, d.reads_coordinates] +
                        [(d.probe == l).astype(float) for l in labs[1:]])
    fam = d.encoder.map(lambda e: FAMILY.get(e, e)).to_numpy()
    b, se, G = cluster_ols(d.gap.to_numpy(), X, fam)
    R = pd.DataFrame(dict(term=["intercept", "log10 parameters", "pretrained", "reads coordinates"],
                          coef=b[:4], se=se[:4]))
    R["lo"], R["hi"] = R.coef - 1.96 * R.se, R.coef + 1.96 * R.se
    R.to_csv(ROOT / "symmetric" / "gap_regression.csv", index=False)
    out["regression"] = dict(cells=len(d), families=int(G), labels=len(labs), rows=R.to_dict("records"))
    print(R.round(3).to_string(index=False), f"\n{len(d)} cells, {G} family clusters, label fixed effects")

    # 3. undetermined labels against the ten written floors, readout and floor chosen on validation
    sym = pd.concat([pd.read_csv(f) for f in glob.glob(str(ROOT / "symmetric" / "results_superfamily" / "by_source" / "*.csv"))])
    sym = sym[~sym.source.str.endswith("_vframe")]
    sym["family"] = np.where(sym.readout == "linear", "linear", "nonlinear")
    mm = sym.groupby(["source", "probe", "family", "split"]).headline.mean().unstack("split").reset_index()
    pick = mm.loc[mm.groupby(["source", "probe"]).val.idxmax()].set_index(["source", "probe"])
    rows = []
    for p in UNDET:
        x = pick.xs(p, level="probe")
        fl = x.loc[[f for f in TEN if f in x.index]]
        best = fl.val.idxmax()
        enc = x[~x.index.str.startswith("ref_")]
        g = (enc.test - fl.loc[best, "test"])
        for e, v in g.items():
            rows.append(dict(probe=p, encoder=e, margin=v, floor=best, esm3=e == "esm3", sidechain=e in SIDECHAIN))
    U = pd.DataFrame(rows)
    U.to_csv(ROOT / "symmetric" / "undetermined_without_esm3.csv", index=False)
    rest = U[~U.esm3 & ~U.sidechain]
    top = rest.loc[rest.groupby("probe").margin.idxmax()][["probe", "encoder", "margin"]]
    npos = rest[rest.margin > 0].groupby("probe").encoder.nunique()
    out["undetermined_without_esm3_sidechain"] = dict(top=top.to_dict("records"), n_above=npos.to_dict())
    print(top.round(3).to_string(index=False)); print("encoders above every written floor:", npos.to_dict())

    # 4. large-pretrained downstream: attainable p and a family bootstrap interval of the mean gap
    sc = pd.read_csv(ROOT / "extract" / "scores2_selected.csv").pivot(index="encoder", columns="task", values="score")
    mb = m[(m.weights == "pretrained") & (m.params >= 100e6)]
    fs = sc.loc[mb.index[mb.group == "sequence"]].groupby(mb.family[mb.group == "sequence"]).mean()
    fc = sc.loc[mb.index[mb.group == "coordinate-side"]].groupby(mb.family[mb.group == "coordinate-side"]).mean()
    ns, nc = len(fs), len(fc)
    n_assign = len(list(combinations(range(ns + nc), nc)))
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(10000):
        a = fs.iloc[rng.integers(0, ns, ns)].mean()
        c = fc.iloc[rng.integers(0, nc, nc)].mean()
        boots.append(float((a - c).mean()))
    gap = float((fs.mean() - fc.mean()).mean())
    out["downstream_large"] = dict(seq_families=ns, crd_families=nc, assignments=n_assign,
                                   min_two_sided_p=2 / n_assign, holm_first_threshold=0.05 / 10,
                                   mean_gap=gap, boot_lo=float(np.percentile(boots, 2.5)),
                                   boot_hi=float(np.percentile(boots, 97.5)))
    print(json.dumps(out["downstream_large"], indent=1))
    # 5. determinacy assigned from the atoms each encoder reads (encoder_atom_audit.csv), one rule for every label:
    #    determined when the label is an exact closed-form function of the atoms read; not established when it is
    #    recoverable only through ideal geometry (an ideal carbonyl O for DSSP, a virtual CB for core/surface);
    #    not determined when an atom the label needs is not read.
    atoms = pd.read_csv(ROOT / "extract" / "encoder_atom_audit.csv").set_index("encoder")
    def aset(e):
        r = atoms.loc[e]
        return {x for x in ("N", "CA", "C", "O", "CB") if bool(r[f"reads_{x}"])}
    EXACT = {"phi_circular": {"N", "CA", "C"}, "psi_circular": {"N", "CA", "C"}, "local_curvature": {"CA"},
             "dssp_3": {"N", "CA", "C", "O"}, "dssp_8": {"N", "CA", "C", "O"},
             "core_surface": {"N", "CA", "C", "CB"}, "distance_bins": {"CA"},
             "radius_of_gyration_len_resid": {"CA"}, "asphericity_len_resid": {"CA"}}
    APPROX = {"dssp_3": [{"N", "CA", "C"}], "dssp_8": [{"N", "CA", "C"}],
              "core_surface": [{"N", "CA", "C"}, {"N", "CA", "CB"}]}
    def status(e, p):
        a = aset(e)
        if EXACT[p] <= a:
            return "determined"
        if any(s <= a for s in APPROX.get(p, [])):
            return "not established"
        return "not determined"
    d9 = crd[crd.probe.isin(EXACT) & crd.encoder.isin(atoms.index)].copy()
    d9["status"] = [status(e, p) for e, p in zip(d9.encoder, d9.probe)]
    S = (d9.groupby(["status", "state"]).size().unstack(fill_value=0)
         .reindex(columns=["below", "unresolved", "above"], fill_value=0))
    S["cells"] = S.sum(axis=1)
    out["per_encoder_determinacy"] = {k: {c: int(v) for c, v in r.items()} for k, r in S.iterrows()}
    above = d9[d9.state == "above"][["encoder", "probe", "status"]]
    out["per_encoder_above"] = above.to_dict("records")
    print(S); print(above.to_string(index=False))

    # 6. a practical-equivalence margin of 0.01 in place of the sign of the interval
    M = 0.01
    rows = []
    for lab, probes in (("saturated", SAT), ("open", OPEN)):
        for grp, g in (("coordinate", crd), ("sequence-only", seq)):
            c = g[g.probe.isin(probes)]
            rows.append(dict(labels=lab, group=grp, cells=len(c), below_margin=int((c.hi < -M).sum()),
                             within_margin=int(((c.lo >= -M) & (c.hi <= M)).sum()),
                             above_margin=int((c.lo > M).sum())))
    out["margin"] = rows
    print(pd.DataFrame(rows).to_string(index=False))

    # 7. the composition reference against the large pretrained sequence-only encoders
    ref = sc.loc["ref_composition"]
    seq_big = mb.index[mb.group == "sequence"]
    med = sc.loc[seq_big].median()
    four = ["go_bp", "subcellular_localization", "solubility", "fluorescence"]
    out["composition_vs_large_seq"] = {t: dict(ref=float(ref[t]), median_large_seq=float(med[t]),
                                             diff=float(med[t] - ref[t])) for t in four}
    print(pd.DataFrame(out["composition_vs_large_seq"]).T.round(3))

    # 8. the pretrained-only comparison without the three checkpoints not checked against published results
    drop = ["gearnet_pretrained", "ipa_pretrained", "tfn_pretrained"]
    mp = m[(m.weights == "pretrained") & ~m.index.isin(drop)]
    fs2 = sc.loc[mp.index[mp.group == "sequence"]].groupby(mp.family[mp.group == "sequence"]).mean()
    fc2 = sc.loc[mp.index[mp.group == "coordinate-side"]].groupby(mp.family[mp.group == "coordinate-side"]).mean()
    gap2 = fs2.mean() - fc2.mean()
    out["pretrained_without_unchecked"] = dict(seq_families=len(fs2), crd_families=len(fc2),
                                               tasks_seq_leads=int((gap2 > 0).sum()), mean_gap=float(gap2.mean()))
    print(json.dumps(out["pretrained_without_unchecked"]))
    (ROOT / "symmetric" / "headline_reanalysis.json").write_text(json.dumps(out, indent=2, default=float))
    if "--latex-atoms" in sys.argv:
        dest2 = Path(sys.argv[sys.argv.index("--latex-atoms") + 1])
        L = ["% generated by scripts/headline_reanalysis.py --latex-atoms; do not edit by hand",
             "\\begin{tabular}{lrrrr}", "\\toprule",
             "status for the atoms the encoder reads & cells & below & unresolved & above \\\\", "\\midrule"]
        for st in ("determined", "not established", "not determined"):
            r = out["per_encoder_determinacy"].get(st, {})
            L.append(f"{st} & {r.get('cells', 0)} & {r.get('below', 0)} & {r.get('unresolved', 0)} & {r.get('above', 0)} \\\\")
        L += ["\\midrule", "\\multicolumn{5}{l}{with a practical-equivalence margin of $0.01$ (coordinate encoders)} \\\\",
              "labels & cells & below by $>0.01$ & within $\\pm 0.01$ & above by $>0.01$ \\\\", "\\midrule"]
        for r in out["margin"]:
            if r["group"] != "coordinate":
                continue
            lab = "best floor $\\geq 0.958$" if r["labels"] == "saturated" else "the other four"
            L.append(f"{lab} & {r['cells']} & {r['below_margin']} & {r['within_margin']} & {r['above_margin']} \\\\")
        L += ["\\bottomrule", "\\end{tabular}"]
        dest2.write_text("\n".join(L) + "\n")
        print("wrote", dest2)
    if "--latex" in sys.argv:
        dest = Path(sys.argv[sys.argv.index("--latex") + 1])
        lab = {"saturated": "best floor $\\geq 0.958$ (5)", "contained": "label in a floor (2)",
               "open": "the other four"}
        L = ["% generated by scripts/headline_reanalysis.py --latex; do not edit by hand",
             "\\begin{tabular}{llrrrr}", "\\toprule",
             "labels & encoders & cells & below & unresolved & above \\\\", "\\midrule"]
        for r in C.itertuples():
            L.append(f"{lab[r.labels]} & {r.subset} & {r.cells} & {r.below} & {r.unresolved} & {r.above} \\\\")
        L += ["\\midrule", "\\multicolumn{6}{l}{gap regressed on the encoder, " + f"{len(d)} cells, {G} family clusters, "
              "label fixed effects} \\\\"]
        for r in R.itertuples():
            if r.term == "intercept":
                continue
            L.append(f"\\multicolumn{{2}}{{l}}{{{r.term}}} & \\multicolumn{{4}}{{l}}{{${r.coef:+.3f}$ $[{r.lo:+.3f}, {r.hi:+.3f}]$}} \\\\")
        L += ["\\bottomrule", "\\end{tabular}"]
        dest.write_text("\n".join(L) + "\n")
        print("wrote", dest)


if __name__ == "__main__":
    main()
