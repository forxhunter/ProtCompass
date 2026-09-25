"""The repair contrast after a nonlinear erasure: LEACE applied in a random Fourier feature space.

LEACE removes the linear subspace that predicts a property, and a two-layer MLP still recovers most of the property
afterwards (Appendix: what the erasure removes). To test whether the one demand that survives every control,
amino-acid composition on stability, also holds when the erasure and the head are nonlinear in the representation,
every representation is first mapped to random Fourier features of a Gaussian kernel,

    phi(x) = sqrt(2 / D) cos(W z + b),   z = the first 256 principal components of the standardised x,
    W ~ N(0, I / sigma^2), b ~ U[0, 2 pi], D = 2048, sigma = the median pairwise distance of z (training split)

and the released pipeline (scripts/addback_named_pairs.py) is run on phi(X) unchanged: LEACE removes what predicts the
property linearly from phi(X), which is what a kernel method with that kernel could use; the variance-matched null
and the sham are drawn in the same space; the head is the released linear head on phi(X), which is a nonlinear head
on X. Everything is fitted on the training split. A check of the erasure is written per cell: the share of the
property a two-layer MLP recovers from phi(X) before and after the erasure.

  python addback_kernel.py --encoders ... [--out erasure/named_pairs_kernel/stability.csv]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import addback_named_pairs as ab                                         # noqa: E402
from erasure import load_task                                             # noqa: E402

D, PCS = 2048, 256


def rff(data):
    Xtr = data["train"][0]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Z = (Xtr - mu) / sd
    k = min(PCS, Z.shape[1])
    U, S, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    proj = Vt[:k].T
    zc = Z.mean(0)
    ztr = (Z - zc) @ proj
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ztr), size=min(2000, len(ztr)), replace=False)
    dd = np.linalg.norm(ztr[idx, None] - ztr[None, idx], axis=-1)
    sigma = float(np.median(dd[dd > 0]))
    W = rng.normal(0, 1.0 / sigma, size=(k, D))
    b = rng.uniform(0, 2 * np.pi, size=D)
    f = lambda X: np.sqrt(2.0 / D) * np.cos(((((X - mu) / sd) - zc) @ proj) @ W + b)
    out = {}
    for s, v in data.items():
        out[s] = (f(v[0]),) + tuple(v[1:])
    return out


def kernel_load(task, enc):
    data = load_task(task, enc)
    return None if data is None else rff(data)


def recoverability(enc):
    """R^2 of the conditional property from phi(X) by a linear map and by a two-layer MLP, before and after the
    kernel-space erasure; fitted on 70% of the training proteins and scored on the other 30%."""
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.metrics import r2_score
    data = kernel_load("stability", enc)
    (Xtr, ytr, itr), (_, _, iva), (_, _, ite) = (data[s] for s in ("train", "valid", "test"))
    ok = ~np.isnan(ytr)
    conc = ab.concepts_for({"train": itr, "valid": iva, "test": ite})
    Z = ab.conditional(conc, "amino-acid composition", ok)
    X = Xtr[ok]
    mu, P, r, B = ab.leace_eraser(X, Z)
    E = ((X - mu) @ P.T) @ B
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X)); a_, b_ = perm[: int(0.7 * len(X))], perm[int(0.7 * len(X)):]
    out = dict(encoder=enc)
    for name, M in (("before", X), ("after", E)):
        lin = Ridge(alpha=1.0).fit(M[a_], Z[a_])
        mlp = MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=300, early_stopping=True, random_state=0).fit(M[a_], Z[a_])
        out[f"linear_{name}"] = r2_score(Z[b_], lin.predict(M[b_]))
        out[f"mlp_{name}"] = r2_score(Z[b_], mlp.predict(M[b_]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", required=True)
    ap.add_argument("--out", default="erasure/named_pairs_kernel/stability.csv")
    ap.add_argument("--check", action="store_true", help="recoverability of the property before and after the erasure")
    a = ap.parse_args()
    ab.load_task = kernel_load                     # run_cell reads load_task from its module namespace
    ab.common_struct.set_task("stability")
    if a.check:
        R = pd.DataFrame([recoverability(e) for e in a.encoders])
        R.to_csv(Path(a.out).with_name("recoverability.csv"), index=False)
        print(R.round(3).to_string(index=False))
        return
    dest = Path(a.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    rows, done = [], set()
    if dest.exists():
        prev = pd.read_csv(dest)
        rows, done = prev.to_dict("records"), set(prev.encoder)
    for enc in a.encoders:
        if enc in done:
            continue
        try:
            rows += ab.run_cell("stability", "amino-acid composition", enc)
        except Exception as e:                                            # noqa: BLE001
            print(f"  [failed] {enc}: {type(e).__name__}: {e}", flush=True)
        pd.DataFrame(rows).to_csv(dest, index=False)
    print(f"written {dest} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
