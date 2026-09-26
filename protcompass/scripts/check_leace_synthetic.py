"""Does the LEACE eraser in erasure.py actually remove the concept? A synthetic check with a known answer.

This script checks the statement that on a synthetic example the concept goes from 0.9999 to 0.0000 and the
residual cross-covariance is 8.3e-15, so the statement is verified rather than asserted.

Three properties are checked, each with an answer known in advance:

1. A concept planted as a linear function of the representation must be recoverable before erasure and not after.
2. The residual cross-covariance between the erased representation and the concept must be at machine precision, which
   is what "linearly unpredictable" means for the closed-form projection.
3. Erasure must not destroy an independent direction. A second concept orthogonal to the first has to survive, or the
   eraser is removing more than it was asked to.

The third is the one that matters for the experiment, since an eraser that flattens everything would produce a large
apparent effect for any property.

  python check_leace_synthetic.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from erasure import leace_eraser


def r2(pred_from, target):
    """Best linear prediction of target from pred_from, reported as R^2 per column and averaged."""
    A = np.c_[pred_from, np.ones(len(pred_from))]
    coef = np.linalg.lstsq(A, target, rcond=None)[0]
    resid = target - A @ coef
    ss_res = (resid ** 2).sum(0)
    ss_tot = ((target - target.mean(0)) ** 2).sum(0)
    return float(np.mean(1 - ss_res / np.maximum(ss_tot, 1e-30)))


def main():
    rng = np.random.default_rng(0)
    n, d, k = 4000, 64, 3
    X = rng.normal(size=(n, d))

    # concept A lives in a known subspace of X; concept B lives in an orthogonal one
    W = np.linalg.qr(rng.normal(size=(d, 2 * k)))[0]
    A_dirs, B_dirs = W[:, :k], W[:, k:]
    Za = X @ A_dirs + 0.01 * rng.normal(size=(n, k))
    Zb = X @ B_dirs + 0.01 * rng.normal(size=(n, k))

    print(f"synthetic: n={n}, d={d}, concept rank {k}, two concepts in orthogonal subspaces\n")
    print(f"{'quantity':52s} {'before':>12s} {'after':>12s}")
    mu, P, r, B = leace_eraser(X, Za)
    Xe = ((X - mu) @ P.T) @ B

    a_before, a_after = r2(X, Za), r2(Xe, Za)
    b_before, b_after = r2(X, Zb), r2(Xe, Zb)
    print(f"{'R^2 recovering the erased concept A':52s} {a_before:12.4f} {a_after:12.4f}")
    print(f"{'R^2 recovering the untouched concept B':52s} {b_before:12.4f} {b_after:12.4f}")

    cov_before = np.abs((X - X.mean(0)).T @ (Za - Za.mean(0)) / n).max()
    cov_after = np.abs((Xe - Xe.mean(0)).T @ (Za - Za.mean(0)) / n).max()
    print(f"{'max |cross-covariance| with concept A':52s} {cov_before:12.3e} {cov_after:12.3e}")
    print(f"\nrank erased: {r} (expected {k}); retained dimensions: {B.shape[1]} of {d}")

    ok = a_after < 1e-6 and cov_after < 1e-10 and b_after > 0.9 * b_before
    print(f"\nerased concept removed to numerical precision: {a_after < 1e-6}")
    print(f"cross-covariance at machine precision:          {cov_after < 1e-10}")
    print(f"orthogonal concept preserved:                   {b_after > 0.9 * b_before} "
          f"({b_after / b_before:.4f} of its original R^2)")
    print(f"\n{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
