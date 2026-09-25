"""Build one manifest per task: a single common ID set for every encoder.

A row enters the manifest only if
  - its sequence has 1..MAX_LEN standard residues (chain separators removed for PPI pairs),
  - a structure file exists AND the residue sequence parsed from that file equals the task sequence,
  - the same sequence does not occur in a later split (test > valid > train priority).
Rows without a verified structure are dropped for all encoders, so every encoder is scored on the same IDs.
Nothing is ever padded, truncated or given placeholder coordinates.

Output: manifest/<task>.csv with columns
  id, split, sequence, structure_path, length, label, mean_plddt, stage1
and manifest/coverage.csv with the counts at every filtering step.
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
import hashlib
import pickle
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
HERE = ROOT / "." / "extract"
STRUCT = HERE / "structures"
TAPE = ROOT / "downstream_tasks" / "data" / "tape"
PFM = ROOT / "downstream_tasks" / "data" / "ppi" / "PFMBench" / "dataset"
MAX_LEN = 600
STAGE1_N = {"train": 6000, "valid": 1500, "test": 3000}
AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
       "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
STD = set(AA3.values())


def pdb_sequence(path):
    """Sequence of the first chain and the mean B-factor column (pLDDT for predicted models)."""
    seq, seen, b = [], set(), []
    chain0 = None
    with open(path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            chain = line[21]
            if chain0 is None:
                chain0 = chain
            if chain != chain0:
                break
            key = line[22:27]
            if line[12:16].strip() == "CA":
                b.append(float(line[60:66]))
            if key not in seen:
                seen.add(key)
                seq.append(AA3.get(line[17:20].strip(), "X"))
    return "".join(seq), (float(np.mean(b)) if b else np.nan)


def lmdb_rows(path):
    env = lmdb.open(str(path), readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        n = pickle.loads(txn.get(b"num_examples"))
        for i in range(n):
            yield i, pickle.loads(txn.get(str(i).encode()))
    env.close()


def seq_index(pdb_dir, cache_name):
    """Map sequence -> (path, mean pLDDT) for a directory of predicted structures. Cached."""
    cache = HERE / "manifest" / f"_{cache_name}_seqindex.pkl"
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    from concurrent.futures import ThreadPoolExecutor
    index = {}
    files = sorted(Path(pdb_dir).glob("*.pdb"))
    with ThreadPoolExecutor(16) as pool:                       # file reads dominate; the disk is shared with other jobs
        for k, (f, (s, p)) in enumerate(zip(files, pool.map(pdb_sequence, files, chunksize=64))):
            index.setdefault(s, (str(f), p))
            if (k + 1) % 10000 == 0:
                print(f"    indexed {k + 1}/{len(files)} in {Path(pdb_dir).name}", flush=True)
    pickle.dump(index, open(cache, "wb"))
    return index


def rows_stability():
    dirs = [TAPE / "tape_structures" / "stability" / d for d in ("predictions_train", "predictions_test")]
    out = []
    for split in ("train", "valid", "test"):
        for i, item in lmdb_rows(TAPE / "stability" / f"stability_{split}.lmdb"):
            pid = item["id"].decode() if isinstance(item["id"], bytes) else str(item["id"])
            path = next((d / f"{pid}.pdb" for d in dirs if (d / f"{pid}.pdb").exists()), None)
            out.append({"id": f"{pid}__{split}_{i:06d}", "split": split, "sequence": item["primary"],
                        "label": float(item["stability_score"][0]), "candidate": str(path) if path else ""})
    return out, None


def rows_fluorescence():
    index = seq_index(STRUCT / "predictions_fluorescence", "fluorescence")
    out = []
    for split in ("train", "valid", "test"):
        for i, item in lmdb_rows(TAPE / "fluorescence" / f"fluorescence_{split}.lmdb"):
            out.append({"id": f"gfp__{split}_{i:06d}", "split": split, "sequence": item["primary"],
                        "label": float(item["log_fluorescence"][0]), "candidate": ""})
    return out, index


def rows_csv(task_dir, files, cache_name, pdb_dir):
    index = seq_index(STRUCT / pdb_dir, cache_name)
    out = []
    for split, fname in files.items():
        df = pd.read_csv(PFM / task_dir / fname)
        for _, r in df.iterrows():
            if pd.isna(r["label"]) or not isinstance(r["aa_seq"], str):
                continue
            out.append({"id": str(r["unique_id"]), "split": split, "sequence": r["aa_seq"], "label": float(r["label"]), "candidate": ""})
    return out, index


TASKS = {
    "stability": rows_stability,
    "fluorescence": rows_fluorescence,
    "solubility": lambda: rows_csv("DeepSol", {"train": "train.csv", "valid": "val.csv", "test": "test.csv"}, "solubility", "predictions_solubility"),
    "ppi_affinity": lambda: rows_csv("ppi_affinity", {"train": "ppi_affinity_train.csv", "valid": "ppi_affinity_valid.csv", "test": "ppi_affinity_test.csv"}, "ppi_affinity", "predictions_ppi_affinity"),
}


def build_ppi():
    """PPI complexes are 'chainA|chainB'. ProtEnv holds one predicted monomer per unique chain, so the manifest lists
    unique chains (split = 'chains') and pairs/ppi_affinity_pairs.csv maps each complex to its two chain ids."""
    rows, index = TASKS["ppi_affinity"]()
    df = pd.DataFrame(rows)
    steps = [("complexes in the official splits", len(df))]
    parts = df["sequence"].str.upper().str.split("|")
    df = df[parts.map(len) == 2].copy(); parts = parts[df.index]
    df["seq_a"], df["seq_b"] = parts.str[0], parts.str[1]
    ok = df.apply(lambda r: all(0 < len(x) <= MAX_LEN and set(x) <= STD for x in (r.seq_a, r.seq_b)), axis=1)
    df = df[ok].copy(); steps.append((f"two chains, each of length 1..{MAX_LEN}, standard residues only", len(df)))
    has = df.apply(lambda r: r.seq_a in index and r.seq_b in index, axis=1)
    df = df[has].copy(); steps.append(("both chains have a predicted structure with an identical sequence", len(df)))
    df["pair_key"] = df.seq_a + "|" + df.seq_b
    df = df.drop_duplicates(subset=["split", "pair_key"], keep="first"); steps.append(("duplicate complexes within a split removed", len(df)))
    test_k = set(df.loc[df.split == "test", "pair_key"]); valid_k = set(df.loc[df.split == "valid", "pair_key"])
    leak = ((df.split == "valid") & df.pair_key.isin(test_k)) | ((df.split == "train") & df.pair_key.isin(test_k | valid_k))
    df = df[~leak].copy(); steps.append(("complexes shared with a later split removed (test > valid > train)", len(df)))
    chain_id = lambda s: "chain_" + hashlib.sha256(s.encode()).hexdigest()[:12]
    df["chain_a"], df["chain_b"] = df.seq_a.map(chain_id), df.seq_b.map(chain_id)
    train_chains = set(df.loc[df.split == "train", "seq_a"]) | set(df.loc[df.split == "train", "seq_b"])
    seen_both = df[df.split == "test"].apply(lambda r: r.seq_a in train_chains and r.seq_b in train_chains, axis=1).mean()
    (HERE / "pairs").mkdir(exist_ok=True)
    df[["id", "split", "chain_a", "chain_b", "label"]].to_csv(HERE / "pairs" / "ppi_affinity_pairs.csv", index=False)
    seqs = sorted(set(df.seq_a) | set(df.seq_b))
    chains = pd.DataFrame({"id": [chain_id(s) for s in seqs], "split": "chains", "sequence": seqs,
                           "structure_path": [index[s][0] for s in seqs], "length": [len(s) for s in seqs], "label": np.nan,
                           "mean_plddt": [index[s][1] for s in seqs], "chain_break": -1, "stage1": 1})
    assert chains["id"].is_unique
    chains.to_csv(HERE / "manifest" / "ppi_affinity.csv", index=False)
    cov = {"task": "ppi_affinity", **dict(steps), "unique_chains": len(chains), "test_complexes_with_both_chains_seen_in_train": round(float(seen_both), 3)}
    for split in ("train", "valid", "test"):
        cov[f"{split}_final"] = cov[f"{split}_stage1"] = int((df.split == split).sum())
    cov["mean_plddt"] = round(float(chains["mean_plddt"].mean()), 1)
    cov["manifest_sha256"] = hashlib.sha256((HERE / "manifest" / "ppi_affinity.csv").read_bytes()).hexdigest()[:16]
    for k, v in steps:
        print(f"  ppi_affinity   {v:7d}  {k}")
    return cov


def build(task):
    if task == "ppi_affinity":
        return build_ppi()
    rows, index = TASKS[task]()
    df = pd.DataFrame(rows)
    steps = [("rows in the official splits", len(df))]
    # PPI pairs are stored as "chainA|chainB"; the predicted model is one chain of the concatenation.
    df["chain_break"] = df["sequence"].str.find("|")
    df["sequence"] = df["sequence"].str.replace("|", "", regex=False).str.upper()
    df["length"] = df["sequence"].str.len()
    ok = df["sequence"].map(lambda s: 0 < len(s) <= MAX_LEN and set(s) <= STD)
    df = df[ok].copy(); steps.append((f"length 1..{MAX_LEN}, standard residues only", len(df)))

    paths, plddt = [], []
    for seq, cand in zip(df["sequence"], df["candidate"]):
        if index is not None:
            hit = index.get(seq)
            paths.append(hit[0] if hit else ""); plddt.append(hit[1] if hit else np.nan)
        elif cand:
            s, p = pdb_sequence(cand)
            paths.append(cand if s == seq else ""); plddt.append(p if s == seq else np.nan)
        else:
            paths.append(""); plddt.append(np.nan)
    df["structure_path"], df["mean_plddt"] = paths, plddt
    steps.append(("structure file exists and its sequence equals the task sequence", int((df["structure_path"] != "").sum())))
    per_split_before = df.groupby("split").size().to_dict()
    df = df[df["structure_path"] != ""].copy()

    df = df.drop_duplicates(subset=["split", "sequence"], keep="first"); steps.append(("duplicate sequences within a split removed", len(df)))
    test_seqs = set(df.loc[df.split == "test", "sequence"]); valid_seqs = set(df.loc[df.split == "valid", "sequence"])
    leak = ((df.split == "valid") & df.sequence.isin(test_seqs)) | ((df.split == "train") & df.sequence.isin(test_seqs | valid_seqs))
    df = df[~leak].copy(); steps.append(("sequences shared with a later split removed (test > valid > train)", len(df)))

    rng = np.random.default_rng(0)
    valid_source = "official"
    if (df.split == "valid").sum() == 0:
        tr = df.index[df.split == "train"].to_numpy()
        df.loc[rng.choice(tr, max(1, len(tr) // 10), replace=False), "split"] = "valid"
        valid_source = "10% of train (the official validation split has no structure files)"
    df["stage1"] = 0
    for split, n in STAGE1_N.items():
        idx = df.index[df.split == split].to_numpy()
        keep = idx if len(idx) <= n else rng.choice(idx, n, replace=False)
        df.loc[keep, "stage1"] = 1
    df = df[["id", "split", "sequence", "structure_path", "length", "label", "mean_plddt", "chain_break", "stage1"]]
    assert df["id"].is_unique, "ids must be unique"
    df.to_csv(HERE / "manifest" / f"{task}.csv", index=False)

    cov = {"task": task, **{k: v for k, v in steps}}
    for split in ("train", "valid", "test"):
        cov[f"{split}_with_length_ok"] = per_split_before.get(split, 0)
        cov[f"{split}_final"] = int((df.split == split).sum())
        cov[f"{split}_stage1"] = int(((df.split == split) & (df.stage1 == 1)).sum())
    cov["valid_source"] = valid_source
    cov["mean_plddt"] = round(float(df["mean_plddt"].mean()), 1)
    cov["manifest_sha256"] = hashlib.sha256((HERE / "manifest" / f"{task}.csv").read_bytes()).hexdigest()[:16]
    for k, v in steps:
        print(f"  {task:14s} {v:7d}  {k}")
    return cov


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tasks", nargs="+", default=list(TASKS)); a = ap.parse_args()
    (HERE / "manifest").mkdir(exist_ok=True)
    cov_path = HERE / "manifest" / "coverage.csv"
    old = pd.read_csv(cov_path) if cov_path.exists() else pd.DataFrame()
    new = pd.DataFrame([build(t) for t in a.tasks])
    if len(old):
        old = old[~old.task.isin(new.task)]
    pd.concat([old, new]).to_csv(cov_path, index=False)
    print(pd.concat([old, new]).T.to_string())
