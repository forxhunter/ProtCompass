"""A null for the erasure that matches the rank, the variance and the re-basing of the real map.

The sham used in the released results erases a permuted copy of the property. It matches the real erasure in rank
and in the change of basis, but not in how much of the representation it removes: across the panel it takes 1.4% to
3.4% of the representation variance where the real erasure takes 6.5% to 24%, a factor of 3 to 17. Two things are
therefore mixed into the difference between the two arms. One is that the real map removes directions aligned with
the property. The other is that it removes far more of the representation, which on a head that is overparameterised
relative to its training split changes the fit for reasons unrelated to the property. The second term is what makes
the sham arm score above the unerased representation on the high-output-dimension tasks, and it inflates the
measured effect on every task.

This module supplies the missing control. The null removes a rank-r subspace spanned by principal directions of the
training representation, chosen so that

  (1) the removed variance equals that of the real erasure to within a tolerance, and
  (2) among the subsets meeting (1), the property content of the removed subspace is as small as the search finds.

Condition (1) makes the comparison a comparison about direction rather than about amount. Condition (2) makes the
null the least favourable member of its family for the paper's claim: if the real erasure still costs more than a
removal of the same size that avoids the property, the loss is specific to the property and not to the capacity.

Principal directions are used because the score of a principal direction is uncorrelated with that of every other,
so the property variance the subset removes is the sum of its members' and the search objective is exact rather
than an approximation. The erased representation is expressed in an orthonormal basis of the complement, the same
treatment the real erasure receives in scripts/erasure.py, so dimension and conditioning are matched by construction
and neither arm is standardised back onto near-zero directions.

The null is not a claim that principal directions are the right coordinates for a representation. It is a control
whose only requirement is that it removes the same amount in directions the property does not occupy.

One alternative was tried and rejected. Instead of matching the amount removed, match the fitting procedure: build a
surrogate target of the same width whose linear recoverability from the representation equals the property's, then
fit LEACE to it. Built out of the principal directions that carry the property least, such a surrogate matches the
recoverability to three decimals and removes essentially none of the property, but it removes 0.05% of the
representation variance against the real map's 13%, because the directions free of the property are the low-variance
tail. Matching the procedure therefore reproduces the same failure as the permuted sham. The subset search below
does not have that problem: a rank-19 subspace removing 13.3% of the variance while carrying 6% of the property does
exist, so the two requirements are compatible when the subspace is chosen directly rather than induced by a fit.
"""
import hashlib

import numpy as np


_SPECTRUM_CACHE = {}


def _fingerprint(*arrays):
    """A content hash of the inputs, so the cache below cannot return another cell's spectrum."""
    h = hashlib.blake2b(digest_size=16)
    for A in arrays:
        B = np.ascontiguousarray(A)
        h.update(str(B.shape).encode())
        h.update(str(B.dtype).encode())
        h.update(B.view(np.uint8) if B.dtype.kind in "fiu" else B.tobytes())
    return h.digest()


def _spectrum_cached(X, Z):
    """`_spectrum` memoised on the contents of its arguments.

    The four null seeds of a cell differ only in the search, not in the spectrum, so without this the eigendecom-
    position and the projection of the representation onto its eigenbasis run four times over identical inputs and
    dominate the cost of the arm. Hashing the two arrays costs milliseconds against seconds for the decomposition.
    Only the last cell is kept, which is all the seed loop needs and bounds the memory at one eigenbasis.
    """
    key = _fingerprint(X, Z)
    hit = _SPECTRUM_CACHE.get(key)
    if hit is None:
        hit = _spectrum(X, Z)
        _SPECTRUM_CACHE.clear()
        _SPECTRUM_CACHE[key] = hit
    return hit


def _spectrum(X, Z):
    """Per-principal-direction variance share and property share of the training representation.

    Returns (evals, evecs, prop) with evals descending. prop[j] is the fraction of the property block's total
    variance that direction j explains; the scores of distinct directions are uncorrelated, so the fraction a set
    of directions explains jointly is the sum of its members'.
    """
    mu = X.mean(0, keepdims=True)
    Xc = X - mu
    w, V = np.linalg.eigh(Xc.T @ Xc / len(Xc))
    order = np.argsort(w)[::-1]
    w, V = np.clip(w[order], 0.0, None), V[:, order]
    T = Xc @ V                                            # (n, d) principal scores, columns uncorrelated
    Zc = Z - Z.mean(0, keepdims=True)
    ztot = float((Zc ** 2).sum())
    tn = (T ** 2).sum(0)
    num = ((T.T @ Zc) ** 2).sum(1)                        # per direction, summed over the property's columns
    prop = num / np.where(tn < 1e-12, 1.0, tn) / max(ztot, 1e-12)
    return w, V, prop


def _select(w, prop, r, target_frac, seed, tol=0.01, restarts=8, sweeps=60):
    """Rank-r index set whose variance share matches target_frac, carrying as little property as the search finds.

    Randomised greedy construction followed by swap descent. The objective is the property share, with the variance
    mismatch entering as a penalty that dominates it whenever the mismatch exceeds the tolerance, so a feasible
    subset is always preferred to an infeasible one however little property it carries.
    """
    d = len(w)
    share = w / max(w.sum(), 1e-12)
    target = float(target_frac)
    rng = np.random.default_rng(seed)
    big = 1e6

    def cost(idx):
        v, p = share[idx].sum(), prop[idx].sum()
        gap = abs(v - target)
        return p + big * max(0.0, gap - tol * target), v, p

    best = None
    for _ in range(restarts):
        idx = list(rng.choice(d, size=r, replace=False))
        cur = cost(idx)
        for _ in range(sweeps):
            improved = False
            for pos in rng.permutation(r):
                out = rng.choice(d, size=min(64, d), replace=False)
                for cand in out:
                    if cand in idx:
                        continue
                    trial = list(idx)
                    trial[pos] = int(cand)
                    c = cost(trial)
                    if c[0] < cur[0] - 1e-12:
                        idx, cur, improved = trial, c, True
            if not improved:
                break
        if best is None or cur[0] < best[1][0]:
            best = (list(idx), cur)
    idx, (_, v, p) = best
    return np.array(sorted(idx)), v, p


def variance_matched_null(X, Z, r, target_frac, seed=0, tol=0.01):
    """(mu, B, achieved_variance_share, property_share) for one null arm.

    Apply it as `(x - mu) @ B`, which is the same form the real erasure takes once it is expressed in an orthonormal
    basis of its range: d - r coordinates, orthonormal, nothing rescaled.
    """
    w, V, prop = _spectrum_cached(X, Z)
    idx, v, p = _select(w, prop, r, target_frac, seed, tol=tol)
    keep = np.setdiff1d(np.arange(len(w)), idx)
    return X.mean(0, keepdims=True), V[:, keep], float(v), float(p)


def variance_removed(X, f):
    """Share of the representation's total variance the map f removes.

    f is the arm as it is actually applied, so for the real erasure it carries both the oblique LEACE projection and
    the change to an orthonormal basis. The distinction matters: on the panel the oblique projection accounts for
    most of what the real map removes, and a control built from the orthogonal projection onto the same subspace
    removes an order of magnitude less.
    """
    Xc = X - X.mean(0, keepdims=True)
    tot = float((Xc ** 2).sum())
    Y = f(X)
    Yc = Y - Y.mean(0, keepdims=True)
    return 1.0 - float((Yc ** 2).sum()) / max(tot, 1e-12)


def property_share_removed(X, Z, f):
    """Drop in the linear recoverability of the property block under the map f."""
    Zc = Z - Z.mean(0, keepdims=True)
    ztot = float((Zc ** 2).sum())

    def r2(F):
        Fc = F - F.mean(0, keepdims=True)
        coef = np.linalg.lstsq(Fc, Zc, rcond=None)[0]
        return 1.0 - float(((Zc - Fc @ coef) ** 2).sum()) / max(ztot, 1e-12)

    return r2(X) - r2(f(X))


def matched_null_projector(X, Z, r, target_frac, seed=0, tol=0.01, full_basis=False):
    """The null in the same algebraic shape as the LEACE map, so both arms are applied by identical code.

    Returns (mu, P, r, B, achieved_variance_share, property_share). The arm is `((x - mu) @ P.T) @ B`, with P the
    orthogonal projection that removes the selected principal directions. `full_basis` selects between the two
    conventions the two runners use for B: an orthonormal basis of the complement, which gives d - r coordinates,
    or a full orthogonal d x d rotation, which keeps d coordinates with r of them at near-zero variance. Each runner
    passes the convention its own erased arm uses, so the null never differs from the erased arm in width.
    """
    if not 0.0 < target_frac < 1.0:
        # A target outside the unit interval means the map it is meant to match does not restrict the
        # representation at all: an oblique projection that erases a numerically null direction expands it instead,
        # which shows up here as a negative share. Matching that is not defined, and clamping it would return a null
        # that removes nothing while looking like a null that removes the right amount, so the cell fails loudly.
        raise ValueError(f"variance-matched null needs a target in (0, 1), got {target_frac:.4g}; "
                         f"the arm it should match expands the representation rather than restricting it")
    w, V, prop = _spectrum_cached(X, Z)
    idx, v, p = _select(w, prop, r, target_frac, seed, tol=tol)
    U = V[:, idx]
    P = np.eye(X.shape[1]) - U @ U.T
    if full_basis:
        B = np.linalg.qr(P.T)[0]
    else:
        B = V[:, np.setdiff1d(np.arange(len(w)), idx)]
    return X.mean(0, keepdims=True), P, r, B, float(v), float(p)

