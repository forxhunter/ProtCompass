"""Does the fluorescence repair contrast depend on the basis in which the erased arm is expressed?

The released erased arm is ((x - mu) P^T) B with B an orthonormal basis of range(P) from an SVD of P. Most singular
values of P equal 1, so B is determined only up to a rotation inside that subspace, and the released head
(erasure.fit_score) standardises every column before RidgeCV, which is not invariant to that rotation. The
variance-matched null uses principal axes as its basis. This script recomputes R_var for the two fluorescence pairs
the repair rule names, over the panel, three ways:

  released     erased arm in the SVD basis, released head (as in scripts/addback_named_pairs.py)
  pc_basis     erased arm in its own principal axes (the null's convention), released head
  invariant    a rotation-invariant head for every arm: the representation block is scaled by one global factor
               (the root mean column variance of the intact training representation) instead of per column; the
               descriptor block is standardised per column as before; RidgeCV with the released alphas; Spearman

  venv/bin/python basis_panel.py            # writes results_basis_panel.csv (resumes per encoder)
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV

HERE = Path(__file__).parent
sys.path.insert(0, (PROTCOMPASS_ROOT + "/protcompass/scripts"))
import common_struct                                                    # noqa: E402
from addback_named_pairs import conditional                            # noqa: E402
from erasure import ALPHAS, ROOT, concepts_for, fit_score, leace_eraser, load_task   # noqa: E402
from matched_null import matched_null_projector, variance_removed      # noqa: E402
from panel import panel                                                 # noqa: E402

TASK, PAIRS = "fluorescence", ["amino-acid composition", "burial and packing"]
OUTF = HERE / "results_basis_panel.csv"


def invariant_score(R, y, Rva, yva, Rte, yte, scale, q=None):
    parts = [R / scale, Rva / scale, Rte / scale]
    if q is not None:
        mu, sd = q[0].mean(0), q[0].std(0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        parts = [np.hstack([p, (qq - mu) / sd]) for p, qq in zip(parts, q)]
    m = RidgeCV(alphas=ALPHAS).fit(parts[0], y)
    return float(spearmanr(yte, m.predict(parts[2]))[0])


def cell(enc, name):
    data = load_task(TASK, enc)
    if data is None:
        return None
    (Xtr, ytr, itr), (Xva, yva, iva), (Xte, yte, ite) = (data[s] for s in ("train", "valid", "test"))
    ok = ~np.isnan(ytr)
    conc = concepts_for({"train": itr, "valid": iva, "test": ite})
    if name not in conc:
        return None
    Z = conditional(conc, name, ok)
    if Z is None:
        return None
    X, y = Xtr[ok], ytr[ok]
    mu, P, r, B = leace_eraser(X, Z)
    if r == 0 or B.shape[1] < 8:
        return None
    Qtr, Qva, Qte, _ = conc[name]
    q = (Qtr[ok], Qva, Qte)
    scale = float(np.sqrt(X.var(0).mean()))
    er = lambda W, Bx: ((W - mu) @ P.T) @ Bx
    E = er(X, B)
    _, V = np.linalg.eigh(np.cov(E, rowvar=False))
    arms = {"erased_svd": B, "erased_pc": B @ V}
    target = variance_removed(X, lambda W: er(W, B))
    nulls = []
    for s in (0, 1, 2, 3):
        mun, Pn, _, Bn, _, _ = matched_null_projector(X, Z, r, target, seed=300 + s)
        nulls.append(lambda W, mun=mun, Pn=Pn, Bn=Bn: ((W - mun) @ Pn.T) @ Bn)

    def released(f):
        A, Bv, C = f(X), f(Xva), f(Xte)
        s0 = fit_score(A, y, Bv, yva, C, yte, "reg")
        s1 = fit_score(np.hstack([A, q[0]]), y, np.hstack([Bv, q[1]]), yva, np.hstack([C, q[2]]), yte, "reg")
        return s1 - s0

    def invariant(f):
        A, Bv, C = f(X), f(Xva), f(Xte)
        return (invariant_score(A, y, Bv, yva, C, yte, scale, q) - invariant_score(A, y, Bv, yva, C, yte, scale))

    null_rel = np.mean([released(f) for f in nulls])
    null_inv = np.mean([invariant(f) for f in nulls])
    rep_svd = released(lambda W: er(W, B))
    rep_pc = released(lambda W: er(W, B @ V))
    rep_inv = invariant(lambda W: er(W, B))
    return dict(encoder=enc, concept=name, dim=X.shape[1], rank=r, repair_erased_svd=rep_svd, repair_erased_pc=rep_pc,
                repair_null_released=null_rel, repair_erased_invariant=rep_inv, repair_null_invariant=null_inv,
                R_released=rep_svd - null_rel, R_pc_basis=rep_pc - null_rel, R_invariant=rep_inv - null_inv)


def main():
    common_struct.set_task(TASK)
    rows = pd.read_csv(OUTF).to_dict("records") if OUTF.exists() else []
    done = {(r["encoder"], r["concept"]) for r in rows}
    encs = [e for e in panel() if (ROOT / "extract" / "embeddings" / TASK / e / "meta.json").exists()]
    for e in encs:
        for name in PAIRS:
            if (e, name) in done:
                continue
            r = cell(e, name)
            if r:
                rows.append(r)
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
                pd.DataFrame(rows).to_csv(OUTF, index=False)


if __name__ == "__main__":
    main()
