"""The fluorescence repair contrast with the exact LEACE map.

The released eraser floors covariance eigenvalues at an absolute 1e-6 (erasure.leace_eraser, ridge=1e-6). Most
eigenvalues of these embeddings lie below that value, so the released map removes more of the representation than
exact LEACE does (results_floor.csv). Here the map is recomputed with a floor relative to the spectrum (1e-9 x the
largest eigenvalue), which reproduces the official exact map (results_floor.csv), the variance-matched nulls are
matched to that map's variance, and R_var is computed with the released head (SVD basis) and with the
rotation-invariant head of basis_panel.py.

  venv/bin/python exact_panel.py            # writes results_exact_panel.csv (resumes per encoder)
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, (PROTCOMPASS_ROOT + "/protcompass/scripts"))
sys.path.insert(0, str(HERE))
import common_struct                                                    # noqa: E402
from addback_named_pairs import conditional                            # noqa: E402
from basis_panel import PAIRS, TASK, invariant_score                    # noqa: E402
from erasure import ROOT, concepts_for, fit_score, leace_eraser, load_task   # noqa: E402
from matched_null import matched_null_projector, variance_removed      # noqa: E402
from panel import panel                                                 # noqa: E402

OUTF = HERE / "results_exact_panel.csv"


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
    ev_max = float(np.linalg.eigvalsh(np.cov(X, rowvar=False)).max())
    mu, P, r, B = leace_eraser(X, Z, ridge=1e-9 * ev_max)
    if r == 0 or B.shape[1] < 8:
        return None
    Qtr, Qva, Qte, _ = conc[name]
    q = (Qtr[ok], Qva, Qte)
    scale = float(np.sqrt(X.var(0).mean()))
    er = lambda W: ((W - mu) @ P.T) @ B
    target = variance_removed(X, er)
    nulls, got = [], []
    for s in (0, 1, 2, 3):
        mun, Pn, _, Bn, vn, _ = matched_null_projector(X, Z, r, target, seed=300 + s)
        nulls.append(lambda W, mun=mun, Pn=Pn, Bn=Bn: ((W - mun) @ Pn.T) @ Bn)
        got.append(vn)

    def released(f):
        A, Bv, C = f(X), f(Xva), f(Xte)
        return (fit_score(np.hstack([A, q[0]]), y, np.hstack([Bv, q[1]]), yva, np.hstack([C, q[2]]), yte, "reg")
                - fit_score(A, y, Bv, yva, C, yte, "reg"))

    def invariant(f):
        A, Bv, C = f(X), f(Xva), f(Xte)
        return invariant_score(A, y, Bv, yva, C, yte, scale, q) - invariant_score(A, y, Bv, yva, C, yte, scale)

    rep, rep_i = released(er), invariant(er)
    nr, ni = np.mean([released(f) for f in nulls]), np.mean([invariant(f) for f in nulls])
    return dict(encoder=enc, concept=name, dim=X.shape[1], rank=r, V_loss_exact=target, V_loss_nulls=float(np.mean(got)),
                R_exact_released_head=rep - nr, R_exact_invariant_head=rep_i - ni)


def main():
    common_struct.set_task(TASK)
    rows = pd.read_csv(OUTF).to_dict("records") if OUTF.exists() else []
    done = {(r["encoder"], r["concept"]) for r in rows}
    encs = [e for e in panel() if (ROOT / "extract" / "embeddings" / TASK / e / "meta.json").exists()]
    for e in encs:
        for name in PAIRS:
            if (e, name) in done:
                continue
            try:
                r = cell(e, name)
            except ValueError as err:                 # a null that cannot be matched fails loudly, as released
                print(f"[{e} / {name}] {err}", flush=True)
                continue
            if r:
                rows.append(r)
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
                pd.DataFrame(rows).to_csv(OUTF, index=False)


if __name__ == "__main__":
    main()
