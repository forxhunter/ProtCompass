"""The repair contrast on PPI affinity under repeated cross-validation with chain-disjoint folds.

PPI affinity carries two naming claims from the effect rule and one from the repair rule, and it is the noisiest task
in the suite: 326 test pairs and a median seed standard deviation of 0.067 in the downstream benchmark. Its released
split also shares chains between training and test. Every one of the 326 test pairs has at least one chain that
appears in a training pair, and one chain appears in 283 pairs.

This script re-measures the two PPI candidates on all 2,872 pairs with folds that share no chain. The chains form 72
connected components when two chains are joined by every pair that contains both, and whole components are assigned
to folds, so no chain of a test fold appears in its training fold. Five folds are drawn three times with different
component-to-fold assignments. Inside each fold everything is fitted on the training part alone: the PCA that the
released pair loader applies when a pair representation is wider than half the training pairs, the conditional
target, the LEACE map, four shams, four variance-matched nulls and every head. The head is the released regression
head (scripts/erasure.fit_score).

  python ppi_repair_cv.py [--encoders ...] [--concepts ...] [--out ...]
  python ppi_repair_cv.py --random-folds --out erasure/ppi_cv_random/enc_<name>.csv

--random-folds is the control for the chain constraint: the same five folds and three repeats, with pairs assigned
to folds at random (balanced), so test pairs share chains with training pairs as in the released split. Everything
else is identical, so a difference between the two runs is the effect of keeping chains disjoint.
"""
import argparse
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common_struct                                                    # noqa: E402
from common_struct import INDEPENDENT_BLOCKS                            # noqa: E402
from erasure import EMB, OUT, ROOT, fit_score, leace_eraser            # noqa: E402
from matched_null import matched_null_projector, variance_removed      # noqa: E402
from panel import panel                                                 # noqa: E402

TASK = "ppi_affinity"
CONCEPTS = ["secondary structure content", "contact order"]
FOLDS, REPEATS = 5, 3
SHAM_SEEDS = (0, 1, 2, 3)
NULL_SEEDS = (0, 1, 2, 3)


def components(pairs):
    G = nx.Graph()
    G.add_edges_from(zip(pairs.chain_a, pairs.chain_b))
    cid = {ch: i for i, c in enumerate(nx.connected_components(G)) for ch in c}
    return pairs.chain_a.map(cid).to_numpy()


def fold_assignments(comp, seed):
    """Whole components to folds, largest first, each to the currently smallest fold after a random shuffle of
    ties, so folds are balanced in pairs and no chain crosses a fold boundary."""
    rng = np.random.default_rng(seed)
    sizes = pd.Series(comp).value_counts()
    order = sizes.index.to_numpy()
    rng.shuffle(order)
    order = sorted(order, key=lambda c: -sizes[c])
    load, where = np.zeros(FOLDS), {}
    for c in order:
        k = int(np.argmin(load + rng.random(FOLDS) * 1e-6))
        where[c] = k
        load[k] += sizes[c]
    return np.array([where[c] for c in comp])


def random_assignments(n, seed):
    """Pairs to folds at random, balanced, ignoring chains: the control for the chain constraint."""
    return np.random.default_rng(1000 + seed).permutation(n) % FOLDS


def pair_matrix(enc, pairs):
    d = EMB / TASK / enc
    if not (d / "chains.npy").exists():
        return None
    vec = np.load(d / "chains.npy").astype(np.float64)
    pos = {i: k for k, i in enumerate((d / "ids_chains.txt").read_text().split())}
    keep = pairs.chain_a.isin(pos) & pairs.chain_b.isin(pos)
    if not keep.all():
        return None
    return np.concatenate([vec[[pos[i] for i in pairs.chain_a]], vec[[pos[i] for i in pairs.chain_b]]], 1)


def pair_blocks(pairs):
    """Property blocks for every pair: the two chains' blocks side by side, as the released pair loader builds them."""
    a = common_struct.protein_descriptors({"train": list(pairs.chain_a), "valid": list(pairs.chain_a[:2]),
                                           "test": list(pairs.chain_a[:2])}, task=TASK)
    b = common_struct.protein_descriptors({"train": list(pairs.chain_b), "valid": list(pairs.chain_b[:2]),
                                           "test": list(pairs.chain_b[:2])}, task=TASK)
    return {n: np.hstack([a[n][0], b[n][0]]) for n in a if n in b}


def one_fold(X, y, blocks, name, tr, te):
    Xtr, Xte = X[tr], X[te]
    if len(tr) < Xtr.shape[1]:
        # the released pair loader's rule: project onto a train-fitted PCA of n/2 directions when the pair
        # representation is wider than the training set
        k = max(16, min(Xtr.shape[1], len(tr) // 2))
        mu = Xtr.mean(0, keepdims=True)
        V = np.linalg.svd(Xtr - mu, full_matrices=False)[2][:k].T
        Xtr, Xte = (Xtr - mu) @ V, (Xte - mu) @ V
    others = [c for c in INDEPENDENT_BLOCKS if c != name and c in blocks]
    Ztr = blocks[name][tr]
    O = np.concatenate([blocks[c][tr] for c in others], 1)
    Z = Ztr - O @ np.linalg.lstsq(O, Ztr, rcond=None)[0]
    mu, P, r, B = leace_eraser(Xtr, Z)
    if r == 0 or B.shape[1] < 8:
        return None
    er = lambda W: ((W - mu) @ P.T) @ B
    arms = {"erased": (er(Xtr), er(Xte))}
    for s in SHAM_SEEDS:
        g = np.random.default_rng(100 + s)
        mus, Ps, rs, Bs = leace_eraser(Xtr, Z[g.permutation(len(Z))])
        if rs and Bs.shape[1] >= 8:
            es = lambda W, mus=mus, Ps=Ps, Bs=Bs: ((W - mus) @ Ps.T) @ Bs
            arms[f"sham{s}"] = (es(Xtr), es(Xte))
    target = variance_removed(Xtr, er)
    for s in NULL_SEEDS:
        mun, Pn, _, Bn, _, _ = matched_null_projector(Xtr, Z, r, target, seed=300 + s)
        en = lambda W, mun=mun, Pn=Pn, Bn=Bn: ((W - mun) @ Pn.T) @ Bn
        arms[f"null{s}"] = (en(Xtr), en(Xte))
    q = {"named": (blocks[name][tr], blocks[name][te])}
    pr = np.random.default_rng(0)
    q["permuted"] = (blocks[name][tr][pr.permutation(len(tr))], blocks[name][te][pr.permutation(len(te))])
    S = {}
    for arm, (A, C) in arms.items():
        for key in (None, "named", "permuted"):
            A2, C2 = (A, C) if key is None else (np.hstack([A, q[key][0]]), np.hstack([C, q[key][1]]))
            S[(arm, key or "none")] = fit_score(A2, y[tr], C2, y[te], C2, y[te], "reg")
    sh = [a for a in arms if a.startswith("sham")]
    nu = [a for a in arms if a.startswith("null")]
    gain = lambda a, k: S[(a, k)] - S[(a, "none")]
    rec = dict(rank=r, n_train=len(tr), n_test=len(te), erased=S[("erased", "none")])
    for fam, arms_ in (("sham", sh), ("var", nu)):
        rec[f"delta_{fam}"] = np.mean([S[(a, "none")] for a in arms_]) - S[("erased", "none")]
        rec[f"R_{fam}"] = gain("erased", "named") - np.mean([gain(a, "named") for a in arms_])
        rec[f"Rperm_{fam}"] = gain("erased", "permuted") - np.mean([gain(a, "permuted") for a in arms_])
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", default=None)
    ap.add_argument("--concepts", nargs="+", default=CONCEPTS)
    ap.add_argument("--out", default=str(OUT / "ppi_repair_cv.csv"))
    ap.add_argument("--max-folds", type=int, default=None, help="for a smoke test")
    ap.add_argument("--random-folds", action="store_true", help="assign pairs to folds at random (control)")
    a = ap.parse_args()
    common_struct.set_task(TASK)
    pairs = pd.read_csv(ROOT / "extract" / "pairs" / f"{TASK}_pairs.csv")
    y = pairs.label.to_numpy(dtype=float)
    comp = components(pairs)
    blocks = pair_blocks(pairs)
    encoders = a.encoders or panel()
    dest = Path(a.out)
    rows = pd.read_csv(dest).to_dict("records") if dest.exists() else []
    done = {(r["encoder"], r["concept"], r["repeat"], r["fold"]) for r in rows}
    n = 0
    for enc in encoders:
        X = pair_matrix(enc, pairs)
        if X is None:
            print(f"  [skip] {enc}: no chain embeddings", flush=True)
            continue
        for rep in range(REPEATS):
            fold = random_assignments(len(pairs), rep) if a.random_folds else fold_assignments(comp, seed=rep)
            for k in range(FOLDS):
                te, tr = np.where(fold == k)[0], np.where(fold != k)[0]
                for name in a.concepts:
                    if (enc, name, rep, k) in done:
                        continue
                    rec = one_fold(X, y, blocks, name, tr, te)
                    if rec is None:
                        continue
                    rows.append(dict(encoder=enc, concept=name, repeat=rep, fold=k, **rec))
                    print(f"  {enc:20s} {name[:22]:22s} rep {rep} fold {k}  R_var {rec['R_var']:+.4f}  "
                          f"R_sham {rec['R_sham']:+.4f}", flush=True)
                    pd.DataFrame(rows).to_csv(dest, index=False)
                    n += 1
                    if a.max_folds and n >= a.max_folds:
                        return
    pd.DataFrame(rows).to_csv(dest, index=False)
    print(f"\nwritten: {dest} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
