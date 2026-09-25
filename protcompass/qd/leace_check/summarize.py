"""Collect the LEACE check into results.csv: one row per reported quantity, with its source file.

  python summarize.py
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

HERE = Path(__file__).parent
sys.path.insert(0, (PROTCOMPASS_ROOT + "/protcompass/scripts"))
from family_weighted_demand import family, holm                         # noqa: E402
from panel import only_panel                                            # noqa: E402

rows = []


def add(section, quantity, value, source, note=""):
    rows.append(dict(section=section, quantity=quantity, value=value, source=source, note=note))


S = pd.read_csv(HERE / "results_synthetic.csv")
for (impl, dt), g in S.groupby(["impl", "dtype"]):
    tag = f"{impl} {dt}"
    add("synthetic", f"max relative difference of erased representation vs project [{tag}]", g.rel_diff_vs_project.max(), "results_synthetic.csv")
    add("synthetic", f"max idempotence error ||P^2-P||/||P|| [{tag}]", g.idempotence.max(), "results_synthetic.csv")
    add("synthetic", f"max residual cross-covariance (relative) [{tag}]", g.residual_xcov.max(), "results_synthetic.csv")
    add("synthetic", f"max held-out R2 of concept, least-squares probe [{tag}]", g.heldout_r2_max.max(), "results_synthetic.csv")
    add("synthetic", f"max held-out R2 of concept, ridge probe [{tag}]", g.heldout_ridge_r2_max.max(), "results_synthetic.csv")
    add("synthetic", f"max |V_loss - D_edit| [{tag}]", (g.V_loss - g.D_edit).abs().max(), "results_synthetic.csv")
add("synthetic", "configurations", len(S) // 3, "results_synthetic.csv", "n x d x concept x conditioning x dtype")
P = S[S.impl == "project"]
add("synthetic", "median share of nonzero singular values of P equal to 1", P.sv_equal_one_share.median(), "results_synthetic.csv")

R = pd.read_csv(HERE / "results_real.csv")
for r in R[R.impl != "matched_null"].itertuples():
    add("real", f"{r.encoder} / {r.concept} [{r.impl}] rank, rel. diff vs project, V_loss, D_edit, residual xcov",
        f"{r.rank}, {r.rel_diff_vs_project:.4f}, {r.V_loss:.4f}, {r.D_edit:.4f}, {r.residual_xcov:.2e}", "results_real.csv")
for r in R[R.impl == "matched_null"].itertuples():
    add("real", f"{r.encoder} / {r.concept} [matched null] idempotence, symmetry, V_loss vs target, property share",
        f"{r.idempotence:.1e}, {r.symmetric:.1e}, {r.V_loss:.4f} vs {r.target_V_loss:.4f}, {r.property_share_removed:.4f}", "results_real.csv")
F = pd.read_csv(HERE / "results_floor.csv")
for r in F.itertuples():
    add("floor", f"{r.encoder} / {r.concept} [{r.floor}] V_loss, D_edit, rel. diff vs exact",
        f"{r.V_loss:.4f}, {r.D_edit:.4f}, {r.rel_diff_vs_official_exact:.4f}", "results_floor.csv",
        f"share of covariance eigenvalues below 1e-6: {r.eig_share_below_1e6:.2f}")
B = pd.read_csv(HERE / "results_basis.csv")
for (e, c), g in B.groupby(["encoder", "concept"]):
    add("basis", f"{e} / {c}: repair S(E+q)-S(E) over 7 bases (svd, principal axes, 5 random rotations)",
        f"{g.repair.min():+.4f} to {g.repair.max():+.4f} (svd {g[g.basis.str.startswith('svd')].repair.iloc[0]:+.4f})", "results_basis.csv")


def rule(df, col, label, src):
    df = only_panel(df)
    out = []
    for c, g in df.groupby("concept"):
        v = g[col].dropna()
        k = int((v > 0).sum())
        fam = g.assign(f=g.encoder.map(family)).groupby("f")[col].mean()
        out.append(dict(concept=c, variant=label, n=len(v), mean=float(v.mean()), pos=k, p=binomtest(k, len(v), 0.5).pvalue,
                        fam_pos=int((fam > 0).sum()), fam_n=len(fam), src=src))
    O = pd.DataFrame(out)
    O["p_holm"] = holm(O.p.to_numpy())
    O["passes"] = (O.p_holm < 0.05) & (O["mean"] >= 0.02)
    return O


V = []
if (HERE / "results_basis_panel.csv").exists():
    BP = pd.read_csv(HERE / "results_basis_panel.csv")
    for col, lab in (("R_released", "released map, released head, SVD basis"),
                     ("R_pc_basis", "released map, released head, principal-axes basis"),
                     ("R_invariant", "released map, rotation-invariant head")):
        V.append(rule(BP, col, lab, "results_basis_panel.csv"))
if (HERE / "results_exact_panel.csv").exists():
    EP = pd.read_csv(HERE / "results_exact_panel.csv")
    for col, lab in (("R_exact_released_head", "exact LEACE map, released head"),
                     ("R_exact_invariant_head", "exact LEACE map, rotation-invariant head")):
        V.append(rule(EP, col, lab, "results_exact_panel.csv"))
if V:
    V = pd.concat(V, ignore_index=True)
    V.to_csv(HERE / "results_panel_rule.csv", index=False)
    for r in V.itertuples():
        add("panel", f"fluorescence / {r.concept} [{r.variant}] mean R_var, encoders positive, Holm p, families positive, passes",
            f"{r.mean:+.4f}, {r.pos}/{r.n}, {r.p_holm:.3g}, {r.fam_pos}/{r.fam_n}, {r.passes}", r.src)
pd.DataFrame(rows).to_csv(HERE / "results.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False, max_colwidth=120))
