"""Numerical check of the linear-risk identity used for the repair contrast.

For the optimal linear readout with intercept under squared loss, L(A) = Var(Y) - s_A^T S_A^+ s_A, and the gain from
adding a descriptor q is

    G(A, q) = L(A) - L(A (+) q) = c^T S_{q.A}^+ c,   q.A = q - Cov(q, A) S_A^+ (A - E A),   c = Cov(q.A, Y).

The left side is computed from two separate least-squares fits, the right side from the residualised descriptor, both
from the same sample moments; the population values (known covariance) are computed alongside, so the sample can be
compared with the population as n grows. Cases: several widths of A, a rank-deficient q, q correlated with A, and
the residual descriptor q.A added back instead of q (same span, so the same gain).

The pipeline case: a representation H carries a property block q and other property blocks O; the erasure removes
the linear information about q.O (q residualised on O, as the conditional arm does), and the repair adds back the
full q. The identity still holds for the pair (A, q) as algebra; what changes is which quantity it attributes the
gain to. Reported: G(A, q) with the full q, G(A, q.O) with the residual that was erased, and their difference.

  python identity_check.py      # writes identity_check.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent


def loss(A, y):
    """Optimal in-sample least-squares loss with intercept (the sample analogue of L)."""
    X = np.hstack([A, np.ones((len(A), 1))]) if A.shape[1] else np.ones((len(y), 1))
    w = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(np.mean((y - X @ w) ** 2))


def resid(q, A):
    Ac = A - A.mean(0)
    B = np.linalg.lstsq(Ac, q - q.mean(0), rcond=None)[0]
    return q - q.mean(0) - Ac @ B


def rhs(A, q, y):
    r = resid(q, A)
    c = r.T @ (y - y.mean()) / len(y)
    S = r.T @ r / len(y)
    return float(c @ np.linalg.pinv(S, rcond=1e-10) @ c)


def pop_gain(Sigma, ia, iq, iy):
    """Population G from the joint covariance of (A, q, Y)."""
    def L(idx):
        if not idx:
            return Sigma[iy, iy]
        s = Sigma[np.ix_(idx, [iy])][:, 0]
        return Sigma[iy, iy] - s @ np.linalg.pinv(Sigma[np.ix_(idx, idx)], rcond=1e-10) @ s
    return float(L(ia) - L(ia + iq))


rows = []
rng = np.random.default_rng(0)
for dA in (5, 50):
    for dq, rank_def in ((1, False), (4, False), (4, True)):
        for corr in (0.0, 0.8):
            # latent Gaussian: A, q = corr * A M + noise (q duplicated column when rank-deficient), Y linear + noise
            M = rng.normal(size=(dA, dq)) / np.sqrt(dA)
            beta, gamma = rng.normal(size=dA) / np.sqrt(dA), rng.normal(size=dq)
            SA = np.eye(dA)
            Sqq = corr ** 2 * M.T @ SA @ M + (1 - corr ** 2) * np.eye(dq)
            SAq = corr * SA @ M
            if rank_def:
                T = np.eye(dq); T[:, -1] = T[:, 0]                       # last column duplicates the first
                Sqq, SAq, gamma = T.T @ Sqq @ T, SAq @ T, gamma
            d = dA + dq
            Sxx = np.block([[SA, SAq], [SAq.T, Sqq]])
            w = np.concatenate([beta, gamma])
            Sigma = np.zeros((d + 1, d + 1))
            Sigma[:d, :d] = Sxx
            Sigma[:d, d] = Sigma[d, :d] = Sxx @ w
            Sigma[d, d] = w @ Sxx @ w + 1.0
            ia, iq, iy = list(range(dA)), list(range(dA, d)), d
            Gpop = pop_gain(Sigma, ia, iq, iy)
            L_, V_ = np.linalg.eigh(Sigma)
            root = V_ * np.sqrt(np.clip(L_, 0, None)) @ V_.T
            for n in (10_000, 1_000_000):
                D = rng.normal(size=(n, d + 1)) @ root
                A, q, y = D[:, :dA], D[:, dA:d], D[:, d]
                lhs = loss(A, y) - loss(np.hstack([A, q]), y)
                lhs_r = loss(A, y) - loss(np.hstack([A, resid(q, A)]), y)
                rows.append(dict(case="identity", dA=dA, dq=dq, q_rank_deficient=rank_def, corr_qA=corr, n=n,
                                 G_lhs_q=lhs, G_rhs=rhs(A, q, y), G_lhs_residual_q=lhs_r, G_population=Gpop,
                                 abs_err_identity=abs(lhs - rhs(A, q, y)), abs_err_residual=abs(lhs_r - lhs),
                                 abs_err_vs_population=abs(lhs - Gpop)))
            print(dA, dq, rank_def, corr, flush=True)

# pipeline case: erase the residual of q on the other blocks O, add back the full q
for O_noise, y_on_O in ((0.0, 0.5), (1.0, 0.5), (1.0, 0.0), (0.3, 0.5), (0.3, 0.0)):
    n, dO, dq, dH = 400_000, 6, 3, 40
    O = rng.normal(size=(n, dO))
    q = O @ (rng.normal(size=(dO, dq)) * 0.7) + rng.normal(size=(n, dq))           # q correlated with O
    qO = resid(q, O)                                                                 # the conditional target
    Y = qO @ rng.normal(size=dq) * 0.6 + y_on_O * O @ rng.normal(size=dO) / np.sqrt(dO) + rng.normal(size=n)
    # H carries qO exactly and O with noise of standard deviation O_noise (0 = O exactly recoverable from H)
    K = np.hstack([qO, O + O_noise * rng.normal(size=(n, dO)), rng.normal(size=(n, dH - dq - dO))])
    H = K @ np.linalg.qr(rng.normal(size=(dH, dH)))[0]
    # LEACE on the conditional target (exact, full-precision covariance)
    Hc, Zc = H - H.mean(0), qO - qO.mean(0)
    S = Hc.T @ Hc / n
    ev, V = np.linalg.eigh(S)
    W, Wi = V @ np.diag(ev ** -0.5) @ V.T, V @ np.diag(ev ** 0.5) @ V.T
    U = np.linalg.svd(W @ (Hc.T @ Zc / n), full_matrices=False)[0]
    P = np.eye(dH) - Wi @ U @ U.T @ W
    A = Hc @ P.T
    g_full, g_res = loss(A, Y) - loss(np.hstack([A, q]), Y), loss(A, Y) - loss(np.hstack([A, qO]), Y)
    rows.append(dict(case="erase residual, add full q", O_noise_in_H=O_noise, y_on_O=y_on_O, n=n, dA=dH, dq=dq,
                     G_full_q=g_full, G_residual_q=g_res, G_rhs_full_q=rhs(A, q, Y), G_rhs_residual_q=rhs(A, qO, Y),
                     excess_full_minus_residual=g_full - g_res, excess_relative=(g_full - g_res) / g_res,
                     abs_err_identity=abs(g_full - rhs(A, q, Y))))
    print("pipeline", O_noise, y_on_O, flush=True)

D = pd.DataFrame(rows)
D.to_csv(HERE / "identity_check.csv", index=False)
pd.set_option("display.width", 250)
print(D.round(6).to_string(index=False))
