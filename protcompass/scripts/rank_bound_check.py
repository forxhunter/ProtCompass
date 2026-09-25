"""What a label-directed rank-r erasure bounds, and what it does not.

Section 4.3 reads the rank ladder -- the cost of erasing the subspace LEACE fits to the task label itself, rank by
rank -- as a reference point for the property effects. An earlier version of the text read it as an upper bound over
arbitrary subspaces of that rank: if removing the label's own rank-32 subspace costs nothing, then no property of
rank 32 can account for any of the score. That reading is true only under conditions the downstream scores do not
meet, and this script separates the two.

Part A verifies the case where the reading is valid. For a centred representation H with Cov(H) = I, a centred
multi-output label Y, the cross-covariance C = Cov(H, Y), an orthogonal rank-r projector Q and an optimal
unregularised linear readout under squared loss,

    risk((I - Q) H) - risk(H) = || Q C ||_F^2  <=  sum_{j=1..r} sigma_j(C)^2,

with equality when Q projects onto the top-r left singular subspace of C. The label-directed removal is then the
most damaging removal of that rank, so a flat ladder does bound every other subspace of the same rank.

Part B shows that the conclusion does not carry to a retrained nonlinear head, which is what the downstream tasks
use. The label is built as a function two directions of the representation determine jointly and neither determines
alone, so the cross-covariance carries no signal about it: the rank-1 subspace the label itself selects is
arbitrary, and erasing it costs nothing, while erasing one of the two directions that carry the label removes the
score entirely. Both erasures have rank 1 and the same head is retrained after each.

  python rank_bound_check.py [--json OUT]
"""
import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- Part A


def optimal_linear_risk(H, Y):
    """Residual squared loss of the optimal unregularised linear map from H to Y, per sample."""
    W, *_ = np.linalg.lstsq(H, Y, rcond=None)
    return float(((Y - H @ W) ** 2).sum() / len(Y))


def whiten(H):
    H = H - H.mean(0)
    C = np.cov(H, rowvar=False)
    w, V = np.linalg.eigh(C)
    return H @ (V / np.sqrt(w)) @ V.T


def part_a(d=12, k=4, n=200_000, ranks=(1, 2, 4, 8), n_random=200):
    """Check the identity, the bound, and that the top-r left singular subspace attains it."""
    H = whiten(RNG.normal(size=(n, d)))
    B = RNG.normal(size=(d, k))
    Y = H @ B + 0.5 * RNG.normal(size=(n, k))
    Y = Y - Y.mean(0)

    C = H.T @ Y / n
    U, sv, _ = np.linalg.svd(C, full_matrices=False)
    base = optimal_linear_risk(H, Y)

    out = []
    for r in ranks:
        # the subspace the label itself selects: the top-r left singular directions of the cross-covariance
        Ur = U[:, :r]
        Q = Ur @ Ur.T
        measured = optimal_linear_risk(H - H @ Q.T, Y) - base
        predicted = float(((Q @ C) ** 2).sum())
        bound = float((sv[:r] ** 2).sum())

        # every other rank-r subspace, sampled at random, must cost no more
        worst_random, id_err = -np.inf, 0.0
        for _ in range(n_random):
            A = RNG.normal(size=(d, r))
            Qr = A @ np.linalg.pinv(A)
            m = optimal_linear_risk(H - H @ Qr.T, Y) - base
            id_err = max(id_err, abs(m - ((Qr @ C) ** 2).sum()))
            worst_random = max(worst_random, m)
        err = float(max(abs(measured - predicted), id_err))
        out.append(dict(rank=r, label_directed=measured, identity_QC=predicted, bound_sigma=bound,
                        worst_of_random=float(worst_random), identity_max_abs_error=err,
                        identity_max_rel_error=err / max(predicted, 1e-12)))
    return out


# ---------------------------------------------------------------- Part B


def fit_score(Xtr, ytr, Xte, yte, seeds=(0, 1, 2)):
    """Test accuracy of a retrained two-layer head, averaged over seeds, with a linear head alongside."""
    mlp = float(np.mean([MLPClassifier(hidden_layer_sizes=(64,), max_iter=800, random_state=s,
                                       early_stopping=True, n_iter_no_change=20)
                         .fit(Xtr, ytr).score(Xte, yte) for s in seeds]))
    lin = float(LogisticRegression(max_iter=2000).fit(Xtr, ytr).score(Xte, yte))
    return mlp, lin


def leace_projector(H, y, rank):
    """The rank-r subspace the label selects: the top-r left singular directions of Cov(H, y), as LEACE removes."""
    Hc = H - H.mean(0)
    yc = (y - y.mean()).reshape(-1, 1)
    C = Hc.T @ yc / len(H)
    U, sv, _ = np.linalg.svd(C, full_matrices=False)
    Ur = U[:, :rank]
    return Ur @ Ur.T, sv[:rank]


def part_b(d=8, n=20_000, leak=0.0):
    """A label two directions determine jointly, optionally with a weak linear leak into a third direction."""
    H = RNG.normal(size=(n, d))
    logit = 3.0 * np.sign(H[:, 0] * H[:, 1]) + leak * H[:, 2]
    y = (RNG.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    ntr = n // 2
    Htr, Hte, ytr, yte = H[:ntr], H[ntr:], y[:ntr], y[ntr:]

    Q_label, sv = leace_projector(Htr, ytr, rank=1)
    e0 = np.zeros((d, 1))
    e0[0] = 1.0
    Q_driver = e0 @ e0.T

    base = fit_score(Htr, ytr, Hte, yte)
    after_label = fit_score(Htr - Htr @ Q_label.T, ytr, Hte - Hte @ Q_label.T, yte)
    after_driver = fit_score(Htr - Htr @ Q_driver.T, ytr, Hte - Hte @ Q_driver.T, yte)
    return dict(leak=leak, majority=float(max(yte.mean(), 1 - yte.mean())),
                top_singular_value=float(sv[0]),
                base_mlp=base[0], base_linear=base[1],
                label_directed_mlp=after_label[0], label_directed_linear=after_label[1],
                driver_mlp=after_driver[0], driver_linear=after_driver[1],
                effect_label_directed=base[0] - after_label[0], effect_driver=base[0] - after_driver[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    A = part_a()
    print("Part A: optimal linear readout, squared loss, whitened representation")
    print(f"{'rank':>5} {'label-directed':>15} {'||QC||_F^2':>12} {'sum sigma^2':>12} "
          f"{'worst random':>13} {'rel. id. err':>14}")
    for r in A:
        print(f"{r['rank']:>5} {r['label_directed']:>15.6f} {r['identity_QC']:>12.6f} {r['bound_sigma']:>12.6f} "
              f"{r['worst_of_random']:>13.6f} {r['identity_max_rel_error']:>14.2e}")
    # tolerance is relative: the risks are O(10) and lstsq on 200k samples carries float64 noise at O(1e-4)
    ok_a = all(r["identity_max_rel_error"] < 1e-4 and r["worst_of_random"] <= r["label_directed"] + 1e-3 for r in A)
    print(f"identity holds and the label-directed subspace is the most damaging of its rank: {ok_a}")

    print("\nPart B: retrained head, label jointly determined by two directions (rank-1 erasures)")
    B = [part_b(leak=l) for l in (0.0, 1.0)]
    for b in B:
        print(f"  linear leak {b['leak']:.1f}: majority {b['majority']:.3f}, "
              f"unerased MLP {b['base_mlp']:.3f} (linear {b['base_linear']:.3f})")
        print(f"    erase the rank-1 subspace the label selects: MLP {b['label_directed_mlp']:.3f} "
              f"(effect {b['effect_label_directed']:+.3f}), top singular value {b['top_singular_value']:.4f}")
        print(f"    erase one of the two directions that carry it:  MLP {b['driver_mlp']:.3f} "
              f"(effect {b['effect_driver']:+.3f})")
    ok_b = all(b["effect_driver"] > b["effect_label_directed"] + 0.1 for b in B)
    print(f"a rank-1 erasure the label does not select costs more than the one it does: {ok_b}")

    if a.json:
        with open(a.json, "w") as fh:
            json.dump(dict(part_a=A, part_b=B, part_a_ok=ok_a, part_b_ok=ok_b), fh, indent=2)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
