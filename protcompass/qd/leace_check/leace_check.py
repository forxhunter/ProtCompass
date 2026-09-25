"""Check the project's LEACE eraser (protcompass/scripts/erasure.py: leace_eraser) against the official implementation
(EleutherAI concept-erasure 0.2.4, installed in ./venv only), and the variance-matched null (matched_null.py).

Part 1, synthetic data with a known concept: agreement of the erased representations, idempotence of the map,
residual cross-covariance, held-out linear recoverability of the concept, variance lost versus squared edit, rank.
The official eraser is run twice: with settings that match the project (no shrinkage, no trace constraint, the
same relative singular-value threshold) and with its defaults (optimal linear shrinkage, trace constraint,
absolute svd_tol 0.01).

Part 2, real embeddings (fluorescence, 3 encoders, 2 properties, the conditional residual the repair rule erases):
the same comparison, the null's algebra, and the sensitivity of the erased-arm score and of the repair term to the
choice of orthonormal basis of range(P). P has most singular values equal to 1, so the basis the SVD returns is
arbitrary, and the released head standardises each column, which is not invariant to a rotation of the basis.

  venv/bin/python leace_check.py [--part 1|2]
Outputs: results_synthetic.csv, results_real.csv, results_basis.csv (this directory).
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).parent
SCRIPTS = Path((PROTCOMPASS_ROOT + "/protcompass/scripts"))
sys.path.insert(0, str(SCRIPTS))
from concept_erasure import LeaceEraser, LeaceFitter                     # noqa: E402

from erasure import leace_eraser                                         # noqa: E402


def official(X, Z, matched, rel_tol=1e-6):
    Xt, Zt = torch.from_numpy(X), torch.from_numpy(Z)
    if matched:
        # the project's threshold is relative to the largest singular value of the whitened cross-covariance
        f0 = LeaceFitter.fit(Xt, Zt, shrinkage=False, constrain_cov_trace=False, svd_tol=1e-30)
        W = f0.eraser  # noqa: F841  (forces the statistics)
        L, V = torch.linalg.eigh(f0.sigma_xx)
        mask = L > (L[-1] * L.shape[-1] * torch.finfo(L.dtype).eps)
        Wm = V * torch.where(mask, L.clamp_min(0).rsqrt(), 0.0) @ V.mH
        s0 = torch.linalg.svdvals(Wm @ f0.sigma_xz)[0].item()
        f = LeaceFitter.fit(Xt, Zt, shrinkage=False, constrain_cov_trace=False, svd_tol=max(rel_tol * s0, 1e-30))
    else:
        f = LeaceFitter.fit(Xt, Zt)
    e = f.eraser
    P = e.P.numpy()
    rank = int(torch.linalg.matrix_rank(e.proj_left @ e.proj_right).item())
    return f.mean_x.numpy()[None], P, rank


def metrics(X, Z, Xte, Zte, mu, P):
    Xc = X - X.mean(0, keepdims=True)
    E = (X - mu) @ P.T
    Ete = (Xte - mu) @ P.T
    Zc = Z - Z.mean(0, keepdims=True)
    tot = (Xc ** 2).sum() / len(X)
    Ec = E - E.mean(0, keepdims=True)
    xcov0 = np.linalg.norm(Xc.T @ Zc / len(X))
    xcov = np.linalg.norm(Ec.T @ Zc / len(X)) / max(xcov0, 1e-300)
    # held-out linear recoverability: least-squares fit on train, R^2 on test, max over concept columns
    A = np.hstack([Ec, np.ones((len(E), 1))])
    coef = np.linalg.lstsq(A, Zc, rcond=None)[0]
    pred = np.hstack([Ete - E.mean(0, keepdims=True), np.ones((len(Ete), 1))]) @ coef
    Zt = Zte - Z.mean(0, keepdims=True)
    r2 = 1 - ((Zt - pred) ** 2).sum(0) / np.maximum(((Zt - Zt.mean(0)) ** 2).sum(0), 1e-300)
    # the same with a ridge probe on standardised columns, as the project's concept probe (erasure.probe_concept)
    from sklearn.linear_model import RidgeCV
    mu_, sd_ = E.mean(0), E.std(0)
    sd_ = np.where(sd_ < 1e-8, 1.0, sd_)
    rp = RidgeCV(alphas=np.logspace(-3, 3, 7)).fit((E - mu_) / sd_, Zc).predict((Ete - mu_) / sd_)
    rp = np.atleast_2d(rp.T).T
    r2r = 1 - ((Zt - rp) ** 2).sum(0) / np.maximum(((Zt - Zt.mean(0)) ** 2).sum(0), 1e-300)
    return dict(heldout_ridge_r2_max=float(np.max(r2r)), idempotence=float(np.linalg.norm(P @ P - P) / np.linalg.norm(P)),
                residual_xcov=float(xcov), heldout_r2_max=float(np.max(r2)),
                V_loss=float(1 - (Ec ** 2).sum() / len(X) / tot),
                D_edit=float(((Xc - Ec) ** 2).sum() / len(X) / tot))


def synth(n, d, k, kind, cond, seed):
    rng = np.random.default_rng(seed)
    lam = np.logspace(0, -cond, d)
    U = np.linalg.qr(rng.normal(size=(d, d)))[0]
    N = rng.normal(size=(2 * n, d)) * np.sqrt(lam) @ U.T
    if kind == "categorical":
        lab = rng.integers(0, k, 2 * n)
        Z = np.eye(k)[lab]
    else:
        Z = rng.normal(size=(2 * n, k))
    A = rng.normal(size=(Z.shape[1], d)) @ U.T * np.sqrt(lam[None])      # the concept lives in the covariance's own scale
    X = N + 0.7 * (Z - Z.mean(0)) @ A
    return X[:n], Z[:n], X[n:], Z[n:]


def part1():
    rows = []
    for n in (2000, 20000):
        for d in (32, 256):
            for kind, k in (("categorical", 2), ("categorical", 5), ("continuous", 1), ("continuous", 4)):
                for cond in (1, 8):
                    for dtype in ("fp64", "fp32"):
                        X, Z, Xte, Zte = synth(n, d, k, kind, cond, seed=n + d + k + cond)
                        if dtype == "fp32":
                            X, Z, Xte, Zte = (a.astype(np.float32) for a in (X, Z, Xte, Zte))
                        mu, P, r, B = leace_eraser(X, Z)
                        P = P.astype(np.float64)
                        out = {"project": (np.asarray(mu, float), P, r)}
                        for name, matched in (("official_matched", True), ("official_default", False)):
                            out[name] = official(X.astype(np.float64) if dtype == "fp64" else X,
                                                 Z.astype(X.dtype), matched)
                        Xd, Zd, Xted, Zted = (a.astype(np.float64) for a in (X, Z, Xte, Zte))
                        Eref = (Xd - out["project"][0]) @ out["project"][1].T
                        sv = np.linalg.svd(out["project"][1], compute_uv=False)
                        for name, (m_, P_, r_) in out.items():
                            P_ = np.asarray(P_, float)
                            E_ = (Xd - np.asarray(m_, float)) @ P_.T
                            rec = dict(n=n, d=d, concept=kind, k=k, cond=f"1e{cond}", dtype=dtype, impl=name, rank=r_,
                                       rel_diff_vs_project=float(np.linalg.norm(E_ - Eref) / np.linalg.norm(Xd - Xd.mean(0))),
                                       **metrics(Xd, Zd, Xted, Zted, np.asarray(m_, float), P_))
                            if name == "project":
                                rec["sv_equal_one_share"] = float((np.abs(sv[sv > 1e-8] - 1) < 1e-6).mean())
                            rows.append(rec)
                        print(n, d, kind, k, cond, dtype, flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(HERE / "results_synthetic.csv", index=False)
    return D


def part2():
    import common_struct
    from addback_named_pairs import conditional
    from erasure import concepts_for, fit_score, load_task
    from matched_null import matched_null_projector, variance_removed
    common_struct.set_task("fluorescence")
    real, basis = [], []
    rng = np.random.default_rng(0)
    for enc in ("esm2_650m", "proteinmpnn", "gcpnet_pretrained"):
        data = load_task("fluorescence", enc)
        (Xtr, ytr, itr), (Xva, yva, iva), (Xte, yte, ite) = (data[s] for s in ("train", "valid", "test"))
        ok = ~np.isnan(ytr)
        conc = concepts_for({"train": itr, "valid": iva, "test": ite})
        for name in ("amino-acid composition", "burial and packing"):
            Z = conditional(conc, name, ok)
            X, y = Xtr[ok], ytr[ok]
            # concept on the test split, residualised with the training coefficients' structure: use the raw block
            Zte_raw = conc[name][2]
            mu, P, r, B = leace_eraser(X, Z)
            out = {"project": (mu, P, r)}
            for impl, matched in (("official_matched", True), ("official_default", False)):
                out[impl] = official(X, Z, matched)
            Eref = (X - mu) @ P.T
            sv = np.linalg.svd(P, compute_uv=False)
            for impl, (m_, P_, r_) in out.items():
                E_ = (X - m_) @ P_.T
                met = metrics(X, Z, X, Z, m_, P_)                     # in-sample on the training split
                real.append(dict(encoder=enc, concept=name, dim=X.shape[1], impl=impl, rank=r_,
                                 rel_diff_vs_project=float(np.linalg.norm(E_ - Eref) / np.linalg.norm(X - X.mean(0))),
                                 sv_equal_one_share=float((np.abs(sv[sv > 1e-8] - 1) < 1e-6).mean()) if impl == "project" else np.nan,
                                 **{k: v for k, v in met.items() if k != "heldout_r2_max"}))
            # the variance-matched null: an orthogonal projection whose basis is a set of principal axes
            target = variance_removed(X, lambda W: ((W - mu) @ P.T) @ B)
            mun, Pn, _, Bn, vn, pn = matched_null_projector(X, Z, r, target, seed=300)
            real.append(dict(encoder=enc, concept=name, dim=X.shape[1], impl="matched_null", rank=r,
                             idempotence=float(np.linalg.norm(Pn @ Pn - Pn) / np.linalg.norm(Pn)),
                             symmetric=float(np.abs(Pn - Pn.T).max()), V_loss=vn, target_V_loss=target,
                             property_share_removed=pn))
            # basis sensitivity of the erased arm and of the repair term under the released head
            q = conc[name]
            Q3 = (q[0][ok], q[1], q[2])
            er = lambda W, Bx: ((W - mu) @ P.T) @ Bx
            Ec = er(X, B)
            evals, evecs = np.linalg.eigh(np.cov(Ec, rowvar=False))
            bases = {"svd (released)": B, "principal axes": B @ evecs}
            for j in range(5):
                bases[f"random rotation {j}"] = B @ np.linalg.qr(rng.normal(size=(B.shape[1], B.shape[1])))[0]
            for bname, Bx in bases.items():
                A_, V_, T_ = er(X, Bx), er(Xva, Bx), er(Xte, Bx)
                s0 = fit_score(A_, y, V_, yva, T_, yte, "reg")
                s1 = fit_score(np.hstack([A_, Q3[0]]), y, np.hstack([V_, Q3[1]]), yva, np.hstack([T_, Q3[2]]), yte, "reg")
                basis.append(dict(encoder=enc, concept=name, basis=bname, S_erased=s0, S_erased_plus_q=s1, repair=s1 - s0))
                print(enc, name, bname, round(s0, 4), round(s1 - s0, 4), flush=True)
            pd.DataFrame(real).to_csv(HERE / "results_real.csv", index=False)
            pd.DataFrame(basis).to_csv(HERE / "results_basis.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, default=0)
    a = ap.parse_args()
    if a.part in (0, 1):
        part1()
    if a.part in (0, 2):
        part2()
