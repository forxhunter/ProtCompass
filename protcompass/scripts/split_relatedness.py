"""How closely the test sequences of each downstream task are related to training sequences and to each other.

Supports the reading of the reference variants (scripts/downstream_reference_variants.py; SI Section S3.3). For two
sequences the measure is the number of distinct 3-mers they share divided by the number of distinct 3-mers of the
one that has fewer. For each test sequence the script takes the maximum over the training sequences and the maximum
over the other test sequences, and reports the share of test sequences for which that maximum is at least 0.5 and
at least 0.9. For PPI affinity, whose examples are chain pairs, it reports the share of test pairs that contain a
chain present in a training pair. For stability it also counts the parent proteins of the test split from the ids.

  python scripts/split_relatedness.py      # from protcompass/; writes extract/ref_variants/split_relatedness.csv

Needs the task manifests and `extract/embeddings/<task>/ref_composition/ids_*.txt`.
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

EXT = Path(PROTCOMPASS_ROOT) / "protcompass" / "extract"
AA, K = "ACDEFGHIKLMNPQRSTVWY", 3
INDEX = {"".join(p): i for i, p in enumerate(itertools.product(AA, repeat=K))}
TASKS = ["ec_classification", "go_mf", "go_bp", "subcellular_localization", "remote_homology", "solubility",
         "membrane_soluble", "stability", "fluorescence"]


def kmer_sets(seqs):
    r, c = [], []
    for i, s in enumerate(seqs):
        ks = {INDEX[s[j:j + K]] for j in range(len(s) - K + 1) if s[j:j + K] in INDEX}
        r += [i] * len(ks)
        c += list(ks)
    return sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(len(seqs), len(INDEX)))


def nearest(A, B, same=False):
    """For each row of A, the largest shared-3-mer fraction with a row of B (excluding itself when A is B)."""
    na, nb = np.asarray(A.sum(1)).ravel(), np.asarray(B.sum(1)).ravel()
    out = np.zeros(A.shape[0], np.float32)
    for s in range(0, A.shape[0], 500):
        inter = (A[s:s + 500] @ B.T).toarray()
        den = np.minimum(na[s:s + 500, None], nb[None, :])
        den[den == 0] = 1
        frac = inter / den
        if same:
            frac[np.arange(frac.shape[0]), np.arange(s, s + frac.shape[0])] = 0
        out[s:s + 500] = frac.max(1)
    return out


def main():
    rows = []
    for task in TASKS:
        d = EXT / "embeddings" / task / "ref_composition"
        seq = pd.read_csv(EXT / "manifest" / f"{task}.csv").drop_duplicates("id").set_index("id").sequence
        ids = {s: (d / f"ids_{s}.txt").read_text().split() for s in ("train", "test")}
        A, B = kmer_sets(seq.reindex(ids["test"]).tolist()), kmer_sets(seq.reindex(ids["train"]).tolist())
        to_train, to_test = nearest(A, B), nearest(A, A, same=True)
        row = dict(task=task, n_train=len(ids["train"]), n_test=len(ids["test"]),
                   test_with_train_seq_ge50=float((to_train >= 0.5).mean()),
                   test_with_train_seq_ge90=float((to_train >= 0.9).mean()),
                   test_with_other_test_seq_ge90=float((to_test >= 0.9).mean()))
        if task == "stability":
            parent = (pd.Series(ids["test"]).str.replace(r"__test_\d+$", "", regex=True)
                      .str.replace(r"(\.pdb)?_[A-Z]\d+[A-Z]$", "", regex=True).str.replace(r"\.pdb$", "", regex=True))
            row["n_test_parents"] = int(parent.nunique())
        rows.append(row)
    pairs = pd.read_csv(EXT / "pairs" / "ppi_affinity_pairs.csv")
    tr, te = pairs[pairs.split == "train"], pairs[pairs.split == "test"]
    seen = set(tr.chain_a) | set(tr.chain_b)
    rows.append(dict(task="ppi_affinity", n_train=len(tr), n_test=len(te),
                     test_pairs_with_a_training_chain=float((te.chain_a.isin(seen) | te.chain_b.isin(seen)).mean()),
                     test_pairs_with_both_chains_in_training=float(
                         (te.chain_a.isin(seen) & te.chain_b.isin(seen)).mean())))
    out = pd.DataFrame(rows).round(4)
    (EXT / "ref_variants").mkdir(exist_ok=True)
    out.to_csv(EXT / "ref_variants" / "split_relatedness.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
