"""Stage-2 manifests: the downstream tasks whose structures were collected in protcompass/structures/.

Two rules fix what the audit found wrong in the original benchmark.

  One common ID set per task. Every encoder is evaluated on exactly the same proteins, so scores are comparable.
  Rows with no usable structure are dropped for every encoder, not only for the structure-based ones.

  No protein appears in more than one split. DeepLoc ships 2,744 proteins that sit in a training partition and in the
  HPA test set at once; they are removed from training. Any other duplicate is resolved the same way, test first.

Output: manifest/<task>.csv with the columns extract.py already reads (id, split, sequence, structure_path, length,
label, stage1) plus align_path, which points at the task-to-structure residue map when the two sequences differ.
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
HERE = ROOT / "." / "extract"
STRUCT = ROOT / "." / "structures"
DATA = ROOT / "structure_encoder_data"
SEED = 0
STAGE1_N = {"train": 6000, "valid": 1500, "test": 3000}   # subsample for the first pass; --full uses every row


def read_manifest(task):
    m = pd.read_csv(STRUCT / task / "manifest.csv", low_memory=False)
    return m[m.status == "ok"].copy()


AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H",
       "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
       "TYR": "Y", "VAL": "V"}


def seq_from_pdb(path):
    """One letter per residue with a CA atom, in file order."""
    out, seen = [], set()
    for l in open(path):
        if l.startswith("ATOM") and l[12:16].strip() == "CA":
            key = (l[21], l[22:27])
            if key in seen:
                continue
            seen.add(key)
            out.append(AA3.get(l[17:20].strip(), "X"))
    return "".join(out)


def attach_alignment(m, task):
    """align_path points at the task-to-structure residue map; sequence is the task sequence."""
    d = STRUCT / task / "alignment"
    m = m.rename(columns={"pdb_path": "structure_path"})
    m["structure_path"] = [str(STRUCT / q) for q in m.structure_path]   # paths in the manifest are relative to structures/
    m["align_path"] = [str(d / f"{i}.json") if (d / f"{i}.json").exists() else "" for i in m.id]
    # The sequence passed to every encoder is the one the structure actually contains. A task sequence may be the
    # full SEQRES while the file holds only observed residues (GO, DeepLoc); passing the task sequence would leave a
    # structure encoder with coordinates for a subset of the residues a sequence encoder sees, so the two would be
    # scored on different proteins. `task_coverage` records how much of the task sequence this keeps.
    # The sequence is read with the extractor's own parser, so the manifest and the extraction can never disagree:
    # a residue the extractor drops for an incomplete backbone is not in the manifest sequence either. A task sequence
    # may be the full SEQRES while the file holds only observed residues (GO, DeepLoc); passing the task sequence
    # would give a structure encoder coordinates for a subset of what a sequence encoder sees, so the two would be
    # scored on different proteins. `struct_coverage` records how much of the task sequence survives.
    import sys as _sys
    _sys.path.insert(0, str(HERE))
    from common import InputError, load_backbone
    from concurrent.futures import ThreadPoolExecutor

    def one(path):
        try:
            return load_backbone(path, None)[1]
        except (InputError, Exception):
            return ""

    with ThreadPoolExecutor(16) as pool:
        seqs = list(pool.map(one, m.structure_path, chunksize=64))
    task_len = []
    for a in m.align_path:
        task_len.append(len(json.loads(Path(a).read_text())["task_seq"]) if a and Path(a).exists() else 0)
    m["sequence"] = seqs
    m["struct_coverage"] = [len(q) / t if t else 1.0 for q, t in zip(seqs, task_len)]
    return m


MAX_LEN = 1022   # ESM-1b's learned positional embedding holds 1024 tokens including BOS and EOS


def finish(df, task, stratify=None):
    """One row per id, no id in two splits, a reproducible stage-1 subsample, written to manifest/<task>.csv.

    Proteins longer than MAX_LEN are dropped for every encoder, not only for the models that cannot take them, so
    that the panel is scored on one common set. The share dropped is printed and belongs in the task description."""
    df = df[(df.sequence.fillna("").str.len() > 0)].dropna(subset=["structure_path"]).drop_duplicates("id")
    order = {"test": 0, "valid": 1, "train": 2}
    df = df.sort_values("split", key=lambda s: s.map(order).fillna(3)).drop_duplicates("id", keep="first")
    df["length"] = df.sequence.str.len()
    n_before = len(df)
    df = df[df.length <= MAX_LEN]
    # A non-standard residue is written as X by the parser. The registry wrappers index a 20-letter alphabet and have
    # no entry for it, and dropping the residue would shift every index after it, so these proteins leave the panel.
    n_len = n_before - len(df)
    df = df[~df.sequence.str.contains("X")]
    dropped = n_before - len(df)
    rng = np.random.default_rng(SEED)
    df["stage1"] = 0
    for part, n in STAGE1_N.items():
        idx = df.index[df.split == part]
        take = idx if len(idx) <= n else rng.choice(idx, n, replace=False)
        df.loc[take, "stage1"] = 1
    cols = ["id", "split", "sequence", "structure_path", "length", "label", "align_path", "struct_coverage", "stage1"]
    df = df[[c for c in cols if c in df.columns]]
    (HERE / "manifest").mkdir(exist_ok=True)
    df.to_csv(HERE / "manifest" / f"{task}.csv", index=False)
    print(f"{task:26s} {len(df):6d} proteins  splits {dict(df.split.value_counts())}  stage1 {int(df.stage1.sum())}"
          f"  dropped: {n_len} over {MAX_LEN} residues, {dropped - n_len} with a non-standard residue "
          f"({dropped / max(n_before, 1):.2%} in total)")
    return df


# ----------------------------------------------------------------------------- per task
def ec_classification():
    m = attach_alignment(read_manifest("ec_classification"), "ec_classification")
    lab = {}
    for line in (DATA / "proteinworkshop" / "ECReaction" / "chain_functions.txt").read_text().splitlines():
        if "," in line:
            c, v = line.split(",", 1)
            lab[c.strip()] = int(float(v))
    # chain_functions.txt keys look like "4fae.B"; the manifest id is "4fae_B"
    m["label"] = m.id.str.replace("_", ".", regex=False).str.lower().map({k.lower(): v for k, v in lab.items()})
    m["split"] = m.split.map({"training": "train", "validation": "valid", "testing": "test"}).fillna(m.split)
    return finish(m.dropna(subset=["label"]), "ec_classification")


def gene_ontology(branch):
    task = f"go_{branch}"
    m = attach_alignment(read_manifest(task), task)
    col = {"mf": 1, "bp": 2}[branch]          # column of the per-chain table: MF, BP, CC
    block = {"mf": "molecular_function", "bp": "biological_process"}[branch]
    path = DATA / "proteinworkshop" / "GeneOntology" / "nrPDB-GO_annot.tsv"
    if not path.exists():
        print(f"{task}: no annotation file found, skipped"); return None
    lines = path.read_text().splitlines()
    terms = next(lines[i + 1].split("\t") for i, l in enumerate(lines) if l.startswith(f"### GO-terms ({block})"))
    start = next(i for i, l in enumerate(lines) if l.startswith("### PDB-chain")) + 1
    idx = {t: j for j, t in enumerate(terms)}
    rows = {}
    for l in lines[start:]:
        f = l.split("\t")
        if len(f) > col:
            rows[f[0].upper()] = f[col]
    # "11AS-A" in the annotation file, "11as_A" in the manifest
    key = m.id.str.replace("_", "-", regex=False).str.upper()
    m["label"] = [json.dumps(sorted(idx[t] for t in str(rows.get(k, "")).split(",") if t in idx)) for k in key]
    print(f"{task}: {len(terms)} GO terms, annotations for {(m.label != '[]').mean():.1%} of rows")
    m = m[m.label != "[]"]
    m["split"] = m.split.replace({"not_in_split_files": np.nan})
    return finish(m.dropna(subset=["split"]), task)


def deeploc(task):
    """Subcellular localization: partitions 0-3 train, partition 4 validation, HPA test; a protein in both is kept
    only in the test set.

    Membrane/soluble cannot use HPA as its test set: every one of the 2,744 HPA proteins is labelled soluble
    (`structure_encoder_data/deeploc/test.csv`, Membrane column), so the metric is undefined there. That task uses
    the homology partitions instead: 0-2 train, 3 validation, 4 test, with the HPA-only proteins dropped."""
    m = attach_alignment(read_manifest(task), task)
    in_hpa = m.split.str.contains("test_hpa")
    part = m.split.str.extract(r"train_partition_(\d)")[0].astype(float)
    if task == "membrane_soluble":
        m = m[part.notna()].copy()
        part = part[part.notna()]
        m["split"] = np.where(part == 4, "test", np.where(part == 3, "valid", "train"))
    else:
        m["split"] = np.where(in_hpa, "test", np.where(part == 4, "valid", "train"))
    tr = pd.read_csv(DATA / "deeploc" / "train_partitions.csv") if (DATA / "deeploc" / "train_partitions.csv").exists() else None
    te = pd.read_csv(DATA / "deeploc" / "test.csv") if (DATA / "deeploc" / "test.csv").exists() else None
    lab = {}
    LOC = ["Cytoplasm", "Nucleus", "Extracellular", "Cell membrane", "Mitochondrion", "Plastid",
           "Endoplasmic reticulum", "Lysosome/Vacuole", "Golgi apparatus", "Peroxisome"]
    for src in (tr, te):
        if src is None:
            continue
        acc = "ACC" if "ACC" in src.columns else src.columns[0]
        if task == "membrane_soluble":
            col = next((c for c in src.columns if c.lower() in ("membrane", "is_membrane", "membrane_soluble")), None)
            if col:
                v = pd.to_numeric(src[col], errors="coerce")
                lab.update({a: float(x) for a, x in zip(src[acc].astype(str), v) if pd.notna(x)})
        else:
            present = [c for c in LOC if c in src.columns]
            if present:
                lab.update({a: int(np.argmax(v)) for a, v in zip(src[acc].astype(str), src[present].values)})
    m["label"] = m.id.astype(str).map(lab)      # the manifest id is the UniProt accession
    got = m.label.notna().mean()
    print(f"{task}: labels matched for {got:.1%} of rows")
    return finish(m.dropna(subset=["label"]), task)


def remote_homology():
    import pickle
    m = attach_alignment(read_manifest("remote_homology"), "remote_homology")
    try:
        import lmdb
    except ImportError:
        print("remote_homology: lmdb not importable, labels not attached"); return None
    lab, seq = {}, {}
    for f, part in (("train", "train"), ("valid", "valid"), ("test_fold_holdout", "test")):
        p = ROOT / "downstream_tasks" / "data" / "tape" / "remote_homology" / f"remote_homology_{f}.lmdb"
        if not p.exists():
            continue
        env = lmdb.open(str(p), readonly=True, lock=False)
        with env.begin() as txn:
            n = pickle.loads(txn.get(b"num_examples"))
            for i in range(n):
                r = pickle.loads(txn.get(str(i).encode()))
                sid = r["id"].decode() if isinstance(r["id"], bytes) else str(r["id"])
                lab[sid] = int(r["fold_label"]); seq[sid] = r["primary"]
        env.close()
    key = m.cache_id.astype(str).str.replace(r"^b'|'$", "", regex=True) if "cache_id" in m.columns else m.id.astype(str)
    m["label"] = key.map(lab)
    # The label comes from the TAPE record; the sequence stays the structure-derived one, as for every other task,
    # so that the manifest and the extractor cannot disagree.
    m["task_seq_len"] = key.map(seq).str.len()
    m["struct_coverage"] = m.sequence.str.len() / m.task_seq_len.where(m.task_seq_len > 0)
    m["split"] = np.where(m.split.str.startswith("test"), "test", np.where(m.split == "valid", "valid", "train"))
    return finish(m.dropna(subset=["label"]), "remote_homology")


def main():
    print("stage-2 manifests: one common ID set per task, no protein in two splits\n")
    ec_classification()
    for b in ("mf", "bp"):
        gene_ontology(b)
    for t in ("subcellular_localization", "membrane_soluble"):
        deeploc(t)
    remote_homology()


if __name__ == "__main__":
    main()
