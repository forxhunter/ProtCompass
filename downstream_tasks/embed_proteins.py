"""
embed_proteins.py — extract and cache frozen encoder embeddings for all downstream tasks.

Embeddings are DETERMINISTIC (no seed needed) — run once, reused across all 5 probe seeds.

Saves to:
  embeddings/{task}/{encoder}/train_embeddings.npy   [N_train, D]
  embeddings/{task}/{encoder}/train_labels.npy       [N_train, ...]
  embeddings/{task}/{encoder}/test_embeddings.npy    [N_test, D]
  embeddings/{task}/{encoder}/test_labels.npy        [N_test, ...]
  embeddings/{task}/{encoder}/meta.json              {D, n_train, n_test, level, ...}

For residue-level tasks, embeddings are stored as ragged arrays (list of [Li, D] arrays)
serialized via numpy object arrays.

Usage:
  # Single encoder, single task (for testing):
  PYTHONPATH=. python downstream_tasks/embed_proteins.py --encoder schnet --task remote_homology

  # All encoders, all tasks (full run):
  PYTHONPATH=. python downstream_tasks/embed_proteins.py --encoder all --task all

  # Skip already-done (default: True):
  PYTHONPATH=. python downstream_tasks/embed_proteins.py --encoder all --task all --skip-done

  # Force re-run specific encoder:
  PYTHONPATH=. python downstream_tasks/embed_proteins.py --encoder gearnet --task all --force
"""

import argparse
import json
import os
import pickle
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import lmdb
import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from downstream_tasks.config import (
    ALL_ENCODERS, DATA_DIR, EMB_DIR, SEQUENCE_ONLY_ENCODERS,
    SEQUENCE_ONLY_TASKS, TASKS,
)

EMB_DIR.mkdir(parents=True, exist_ok=True)

AA_VOCAB = {
    "A":0,"C":1,"D":2,"E":3,"F":4,"G":5,"H":6,"I":7,"K":8,"L":9,
    "M":10,"N":11,"P":12,"Q":13,"R":14,"S":15,"T":16,"V":17,"W":18,"Y":19,
}
RES3_TO_1 = {
    "ALA":"A","CYS":"C","ASP":"D","GLU":"E","PHE":"F","GLY":"G","HIS":"H",
    "ILE":"I","LYS":"K","LEU":"L","MET":"M","ASN":"N","PRO":"P","GLN":"Q",
    "ARG":"R","SER":"S","THR":"T","VAL":"V","TRP":"W","TYR":"Y",
}


def load_pdb_coords(pdb_path: str, device) -> Optional[torch.Tensor]:
    """Load backbone coordinates from PDB file.

    Returns: [1, L, 5, 3] tensor (N, CA, C, O, CB) or None if failed.
    """
    try:
        from Bio.PDB import PDBParser
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_path)

        coords_list = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0] != ' ':  # Skip hetero atoms
                        continue

                    # Extract backbone atoms
                    atoms = {}
                    for atom in residue:
                        if atom.name in ['N', 'CA', 'C', 'O', 'CB']:
                            atoms[atom.name] = atom.coord

                    # Require at least N, CA, C
                    if 'N' not in atoms or 'CA' not in atoms or 'C' not in atoms:
                        continue

                    # Use CA for missing O/CB
                    o_coord = atoms.get('O', atoms['CA'])
                    cb_coord = atoms.get('CB', atoms['CA'])

                    coords_list.append([
                        atoms['N'], atoms['CA'], atoms['C'], o_coord, cb_coord
                    ])
                break  # Only first chain
            break  # Only first model

        if not coords_list:
            return None

        coords = np.array(coords_list, dtype=np.float32)  # [L, 5, 3]
        return torch.from_numpy(coords).unsqueeze(0).to(device)  # [1, L, 5, 3]

    except Exception as e:
        print(f"Warning: Failed to load PDB {pdb_path}: {e}")
        return None


# ── LMDB reader ───────────────────────────────────────────────────────────────

def lmdb_iter(path: Path):
    env = lmdb.open(str(path), readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        n = pickle.loads(txn.get(b"num_examples"))
        for i in range(n):
            yield pickle.loads(txn.get(str(i).encode()))
    env.close()


# ── Task data loaders (return train and test separately) ──────────────────────

def load_tape_secondary_structure():
    base = DATA_DIR / "tape" / "secondary_structure"
    def _load(split):
        seqs, labels, ids = [], [], []
        for item in lmdb_iter(base / f"secondary_structure_{split}.lmdb"):
            seqs.append(item["primary"])
            labels.append(np.array(item["ss3"], dtype=np.int64))
            ids.append(str(item.get("id", len(ids))))
        return seqs, labels, ids
    train = _load("train")
    test  = _load("cb513")
    return train, test


def load_tape_contact_prediction():
    base = DATA_DIR / "tape" / "proteinnet"
    def _load(split):
        seqs, tertiaries, valid_masks, ids = [], [], [], []
        for item in lmdb_iter(base / f"proteinnet_{split}.lmdb"):
            seqs.append(item["primary"])
            tertiaries.append(np.array(item["tertiary"], dtype=np.float32))
            valid_masks.append(np.array(item["valid_mask"], dtype=np.bool_))
            ids.append(str(item.get("id", len(ids))))
        return seqs, (tertiaries, valid_masks), ids
    return _load("train"), _load("test")


def load_tape_remote_homology():
    base = DATA_DIR / "tape" / "remote_homology"
    def _load(split):
        seqs, labels, ids = [], [], []
        for item in lmdb_iter(base / f"remote_homology_{split}.lmdb"):
            seqs.append(item["primary"])
            labels.append(int(item["fold_label"]))
            ids.append(str(item.get("id", len(ids))))
        return seqs, labels, ids
    train = _load("train")
    test  = _load("test_fold_holdout")
    return train, test


_PROTENV_DIR = DATA_DIR / "tape" / "ProtEnv"

def _protenv_seq_lookup(pdb_dir: Path, prefix: str) -> dict:
    """Return {sequence: pdb_path_str} built from ProtEnv PDB files, cached on disk."""
    cache = pdb_dir.parent / f"{prefix}_seq_lookup.pkl"
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    _aa = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
           'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
           'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
    lookup: dict = {}
    print(f"  [ProtEnv] Building seq→pdb cache for {prefix} ({pdb_dir}) ...")
    for pdb_file in sorted(pdb_dir.glob(f"{prefix}_*.pdb"),
                           key=lambda f: int(f.stem.split("_", 1)[1])):
        seen: set = set(); seq: list = []
        for line in open(pdb_file):
            if not line.startswith("ATOM"):
                continue
            chain = line[21]; resi = int(line[22:26]); res = line[17:20].strip()
            if (chain, resi) not in seen:
                seen.add((chain, resi)); seq.append(_aa.get(res, "X"))
        full_seq = "".join(seq)
        if full_seq not in lookup:
            lookup[full_seq] = str(pdb_file)
    pickle.dump(lookup, open(cache, "wb"))
    print(f"  [ProtEnv] Cached {len(lookup)} unique sequences.")
    return lookup


def load_tape_fluorescence():
    base = DATA_DIR / "tape" / "fluorescence"
    pdb_dir = _PROTENV_DIR / "fluroscence" / "predictions_fluorescence"
    lookup = _protenv_seq_lookup(pdb_dir, "fluor") if pdb_dir.exists() else {}
    def _load(split):
        seqs, labels, ids, structs = [], [], [], []
        for item in lmdb_iter(base / f"fluorescence_{split}.lmdb"):
            seq = item["primary"]
            seqs.append(seq)
            labels.append(float(item["log_fluorescence"][0]))
            ids.append(str(item.get("id", len(ids))))
            structs.append(lookup.get(seq))
        return seqs, labels, ids, structs
    return _load("train"), _load("test")


def load_tape_stability():
    base = DATA_DIR / "tape" / "stability"
    struct_base = DATA_DIR / "tape" / "tape_structures" / "stability"
    split_map = {"train": "predictions_train", "test": "predictions_test"}
    def _load(split):
        pdb_dir = struct_base / split_map[split]
        seqs, labels, ids, structs = [], [], [], []
        for item in lmdb_iter(base / f"stability_{split}.lmdb"):
            pid = item["id"].decode() if isinstance(item["id"], bytes) else item["id"]
            seqs.append(item["primary"])
            labels.append(float(item["stability_score"][0]))
            ids.append(pid)
            pdb_path = pdb_dir / (pid + ".pdb")
            structs.append(str(pdb_path) if pdb_path.exists() else None)
        return seqs, labels, ids, structs
    return _load("train"), _load("test")


def load_ec_classification():
    from downstream_tasks.tasks.load_proteinworkshop import load_ec_reaction
    train_ids, train_labels, train_seqs, train_structs = load_ec_reaction("training")
    test_ids,  test_labels,  test_seqs, test_structs  = load_ec_reaction("testing")
    return (train_seqs, train_labels, train_ids, train_structs), (test_seqs, test_labels, test_ids, test_structs)


def load_go(ontology: str):
    from downstream_tasks.tasks.load_proteinworkshop import load_go as _load_go
    tr_ids, tr_labels, tr_seqs, tr_structs = _load_go(ontology, "train")
    te_ids, te_labels, te_seqs, te_structs = _load_go(ontology, "test")
    return (tr_seqs, tr_labels, tr_ids, tr_structs), (te_seqs, te_labels, te_ids, te_structs)


def load_ppi_site():
    from downstream_tasks.tasks.load_proteinworkshop import load_masif_site
    tr_ids, tr_labels, tr_seqs, tr_structs = load_masif_site("training")
    te_ids, te_labels, te_seqs, te_structs = load_masif_site("testing")
    return (tr_seqs, tr_labels, tr_ids, tr_structs), (te_seqs, te_labels, te_ids, te_structs)


def load_metal_binding():
    from downstream_tasks.tasks.load_proteinworkshop import load_metal3d
    tr_ids, tr_labels, tr_seqs, tr_structs = load_metal3d("train")
    te_ids, te_labels, te_seqs, te_structs = load_metal3d("test")
    return (tr_seqs, tr_labels, tr_ids, tr_structs), (te_seqs, te_labels, te_ids, te_structs)


def load_deeploc():
    import pandas as pd
    LOC_COLS = ["Cytoplasm","Nucleus","Extracellular","Cell membrane",
                "Mitochondrion","Plastid","Endoplasmic reticulum",
                "Lysosome/Vacuole","Golgi apparatus","Peroxisome"]
    def _load_train():
        df = pd.read_csv(DATA_DIR / "deeploc" / "train_partitions.csv")
        seqs, loc_labels, mem_labels, ids = [], [], [], []
        for _, row in df.iterrows():
            seq = str(row.get("Sequence", ""))
            if not seq or seq == "nan": continue
            lc = [c for c in LOC_COLS if c in df.columns]
            vals = [float(row.get(c, 0)) for c in lc]
            loc_labels.append(int(np.argmax(vals)) if any(v > 0 for v in vals) else -1)
            mem_labels.append(int(float(row.get("Membrane", 0)) > 0.5))
            seqs.append(seq)
            ids.append(str(row.get("ACC", len(ids))))
        return seqs, (loc_labels, mem_labels), ids
    def _load_test():
        df = pd.read_csv(DATA_DIR / "deeploc" / "test.csv")
        lc = [c for c in LOC_COLS if c in df.columns]
        seqs, loc_labels, mem_labels, ids = [], [], [], []
        for _, row in df.iterrows():
            seq = str(row.get("fasta", ""))
            if not seq or seq == "nan": continue
            vals = [float(row.get(c, 0)) for c in lc]
            loc_labels.append(int(np.argmax(vals)) if any(v > 0 for v in vals) else -1)
            # HPA test: proxy membrane from "Cell membrane" column
            mem_labels.append(int(float(row.get("Cell membrane", 0)) > 0.5))
            seqs.append(seq)
            ids.append(str(row.get("sid", len(ids))))
        return seqs, (loc_labels, mem_labels), ids
    return _load_train(), _load_test()


def load_mutation_effect(subsample_size=None):
    """ProteinGym DMS: deep mutational scan benchmark — predict variant effect score.

    Returns (train/test)_seqs containing the mutated sequence for embedding,
    with labels = DMS score. Structure paths are included for structure encoders.

    Args:
        subsample_size: If set, randomly sample this many sequences from train/test
                       (maintains 80/20 split). Use for faster computation.

    Returns:
        train: (seqs, labels, ids, structure_paths)
        test:  (seqs, labels, ids, structure_paths)
    """
    import pandas as pd
    import json
    import random

    # Use unified structure_encoder_data location
    pg_dir = Path(os.environ.get("PROTCOMPASS_ROOT") or ROOT) / "structure_encoder_data" / "proteingym"
    dms_dir = pg_dir / "DMS_ProteinGym_substitutions"

    # Load structure mapping (created by map_proteingym_structures.py)
    with open(pg_dir / "structure_mapping.json") as f:
        structure_map = json.load(f)

    # Load split metadata
    with open(pg_dir / "split_metadata.json") as f:
        metadata = json.load(f)

    train_seqs, train_labels, train_ids, train_structures = [], [], [], []
    test_seqs, test_labels, test_ids, test_structures = [], [], [], []

    for csv_file in metadata['assay_files']:
        csv_path = dms_dir / csv_file
        if not csv_path.exists():
            print(f"Warning: {csv_file} not found, skipping")
            continue

        df = pd.read_csv(csv_path)

        # Get structure path (None if missing)
        structure_path = structure_map.get(csv_file, None)

        # Use mutated sequence directly (not wild-type)
        n = len(df)
        split = int(0.8 * n)

        for i in range(n):
            seq = df['mutated_sequence'].iloc[i]
            mut_str = df['mutant'].iloc[i]
            score = df['DMS_score'].iloc[i]
            entry_id = f"{csv_file.replace('.csv', '')}_{i}_{mut_str}"

            if i < split:
                train_seqs.append(seq)
                train_labels.append(float(score))
                train_ids.append(entry_id)
                train_structures.append(structure_path)
            else:
                test_seqs.append(seq)
                test_labels.append(float(score))
                test_ids.append(entry_id)
                test_structures.append(structure_path)

    # Apply subsampling if requested
    if subsample_size is not None:
        print(f"  [SUBSAMPLE] Original: {len(train_seqs)} train, {len(test_seqs)} test")

        # Calculate split sizes (maintain 80/20 ratio)
        train_size = int(subsample_size * 0.8)
        test_size = subsample_size - train_size

        # Randomly sample with fixed seed for reproducibility
        random.seed(42)

        if len(train_seqs) > train_size:
            indices = random.sample(range(len(train_seqs)), train_size)
            train_seqs = [train_seqs[i] for i in indices]
            train_labels = [train_labels[i] for i in indices]
            train_ids = [train_ids[i] for i in indices]
            train_structures = [train_structures[i] for i in indices]

        if len(test_seqs) > test_size:
            indices = random.sample(range(len(test_seqs)), test_size)
            test_seqs = [test_seqs[i] for i in indices]
            test_labels = [test_labels[i] for i in indices]
            test_ids = [test_ids[i] for i in indices]
            test_structures = [test_structures[i] for i in indices]

        print(f"  [SUBSAMPLE] Sampled: {len(train_seqs)} train, {len(test_seqs)} test")

    return (train_seqs, train_labels, train_ids, train_structures), \
           (test_seqs, test_labels, test_ids, test_structures)


def load_ppi_affinity():
    """PPI binding affinity from PFMBench (ppi_affinity_train/test.csv)."""
    import pandas as pd
    ppi_dir = DATA_DIR / "ppi" / "PFMBench" / "dataset" / "ppi_affinity"
    pdb_dir = _PROTENV_DIR / "ppi_affinity" / "predictions_ppi_affinity"
    lookup = _protenv_seq_lookup(pdb_dir, "ppi") if pdb_dir.exists() else {}

    def _load(split_file):
        seqs, labels, ids, structs = [], [], [], []
        df = pd.read_csv(ppi_dir / split_file)
        for _, row in df.iterrows():
            seq = str(row.get("aa_seq", ""))
            label = row.get("label", None)
            if seq and pd.notna(label):
                seqs.append(seq)
                labels.append(float(label))
                ids.append(str(row.get("unique_id", len(ids))))
                structs.append(lookup.get(seq))
        return seqs, labels, ids, structs

    return _load("ppi_affinity_train.csv"), _load("ppi_affinity_test.csv")


def load_solubility():
    """Protein solubility from PFMBench DeepSol (train/test.csv)."""
    import pandas as pd
    sol_dir = DATA_DIR / "ppi" / "PFMBench" / "dataset" / "DeepSol"
    pdb_dir = _PROTENV_DIR / "solubility" / "predictions_solubility"
    lookup = _protenv_seq_lookup(pdb_dir, "solub") if pdb_dir.exists() else {}

    def _load(split_file):
        seqs, labels, ids, structs = [], [], [], []
        df = pd.read_csv(sol_dir / split_file)
        for _, row in df.iterrows():
            seq = str(row.get("aa_seq", ""))
            label = row.get("label", None)
            if seq and pd.notna(label):
                seqs.append(seq)
                labels.append(float(label))
                ids.append(str(row.get("unique_id", len(ids))))
                structs.append(lookup.get(seq))
        return seqs, labels, ids, structs

    return _load("train.csv"), _load("test.csv")


def load_stability_flip():
    """FLIP stability: predict thermal stability change (ΔΔG)."""
    import pandas as pd
    flip_dir = DATA_DIR / "flip"
    df = pd.read_csv(flip_dir / "stability.csv")

    seqs, labels, ids = [], [], []
    for _, row in df.iterrows():
        seq = str(row.get("sequence", ""))
        label = row.get("stability_label", None)
        if seq and pd.notna(label):
            seqs.append(seq)
            labels.append(float(label))
            ids.append(str(row.name))

    n = len(seqs)
    split = int(0.8 * n)
    return (seqs[:split], labels[:split], ids[:split]), (seqs[split:], labels[split:], ids[split:])


TASK_LOADERS = {
    "secondary_structure":      load_tape_secondary_structure,
    "contact_prediction":       load_tape_contact_prediction,
    "remote_homology":          load_tape_remote_homology,
    "fluorescence":             load_tape_fluorescence,
    "stability":                load_tape_stability,
    "ec_classification":        load_ec_classification,
    "go_mf":                    lambda: load_go("mf"),
    "go_bp":                    lambda: load_go("bp"),
    "ppi_site":                 load_ppi_site,
    "metal_binding":            load_metal_binding,
    "subcellular_localization": load_deeploc,
    "membrane_soluble":         load_deeploc,
    "mutation_effect":          load_mutation_effect,
    "ppi_affinity":             load_ppi_affinity,
    "solubility":               load_solubility,
    "stability_flip":           load_stability_flip,
}

# Tasks that share the same raw loader (deeploc loads both labels at once)
SHARED_LOADER = {
    "subcellular_localization": "deeploc",
    "membrane_soluble":         "deeploc",
}


# ── Encoder wrapper ───────────────────────────────────────────────────────────

# ── Offline weight paths ──────────────────────────────────────────────────────
# All weights are shipped with the project; no internet required on the cluster.
# Paths are relative to ROOT so they work after rsync.
#
#   HuggingFace models  → ROOT/weights/huggingface/   (set via HF_HOME)
#   PyTorch hub cache   → ROOT/weights/torch_hub/      (set via TORCH_HOME)
#   ProteinMPNN         → ROOT/checkpoints/proteinmpnn_v48_020.pt
#   GearNet-pretrained  → ROOT/checkpoints/mc_gearnet_edge.pth
#   GeoBPE              → ROOT/Geo_BPE/GeoBPE/ckpts/<timestamp>/bpe_iter=0.pkl
#   ESM-3               → ROOT/esm3-sm-open-v1/   (hardcoded in esm3_encoder.py)

_CKPT_MAP = {
    # encoders that need load_pretrained() called explicitly after construction
    "proteinmpnn_pretrained": ROOT / "checkpoints" / "proteinmpnn_v48_020.pt",
    "gearnet_pretrained":     ROOT / "checkpoints" / "gearnet" / "mc_gearnet_edge.pth",
}

def _find_geobpe_ckpt() -> Optional[Path]:
    """Return the most recently created GeoBPE checkpoint pkl."""
    import glob
    # Try original location first
    pattern = str(ROOT / "Geo_BPE" / "GeoBPE" / "ckpts" / "*" / "bpe_iter=0.pkl")
    candidates = sorted(glob.glob(pattern))
    if candidates:
        return Path(candidates[-1])

    # Fallback: check checkpoints directory
    ckpt_path = ROOT / "checkpoints" / "bpe_iter=0.pkl"
    if ckpt_path.exists():
        return ckpt_path

    return None


def _configure_offline_weights():
    """
    Point HuggingFace and PyTorch hub to local weight directories so that
    no internet connection is needed on the cluster.
    """
    import os
    hf_cache = ROOT / "weights" / "huggingface"
    th_home   = ROOT / "weights" / "torch_hub"
    # Only override if the local cache exists (transfer.sh creates it)
    if hf_cache.exists():
        os.environ["HF_HOME"]          = str(hf_cache)
        os.environ["TRANSFORMERS_CACHE"]= str(hf_cache / "hub")
        os.environ["HF_DATASETS_CACHE"] = str(hf_cache / "datasets")
        # Disable HuggingFace auto-download attempts
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"]       = "1"
    if th_home.exists():
        os.environ["TORCH_HOME"] = str(th_home)


def load_encoder(encoder_name: str, num_workers: int = 1):
    _configure_offline_weights()

    from src.models import get_encoder

    # GeoBPE: pass checkpoint path in constructor kwargs
    if encoder_name == "geobpe":
        ckpt = _find_geobpe_ckpt()
        if ckpt is None:
            raise FileNotFoundError(
                "GeoBPE checkpoint not found under Geo_BPE/GeoBPE/ckpts/. "
                "Make sure it was transferred with transfer.sh."
            )
        enc = get_encoder(encoder_name, checkpoint_path=str(ckpt), num_workers=num_workers)
    else:
        enc = get_encoder(encoder_name)

    # Encoders that need explicit load_pretrained() after construction
    if encoder_name in _CKPT_MAP:
        ckpt = _CKPT_MAP[encoder_name]
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Pretrained weights for '{encoder_name}' not found at {ckpt}. "
                f"Run transfer.sh from the local machine first."
            )
        enc.load_pretrained(str(ckpt))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = enc.eval().to(device)
    return enc, device


def encode_sequences(encoder, encoder_name: str, sequences: List[str],
                     residue_level: bool = False,
                     batch_size: int = 32,
                     structure_paths: Optional[List[Optional[str]]] = None) -> List[Optional[np.ndarray]]:
    """
    Run encoder on list of sequences. Returns list of np.ndarray.
    For protein-level: each array is [D].
    For residue-level: each array is [Li, D].

    Args:
        structure_paths: Optional list of PDB file paths for structure encoders.
                        None entries use dummy coordinates.
    """
    from src.models.base_encoder import ProteinInput

    # Determine device: prefer CUDA if available, otherwise use encoder's device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        try:
            device = next(encoder.parameters()).device if hasattr(encoder, 'parameters') else torch.device("cpu")
        except StopIteration:
            device = torch.device("cpu")

    print(f"  Encoding on device: {device}")
    results = [None] * len(sequences)

    is_structure_encoder = encoder_name not in SEQUENCE_ONLY_ENCODERS

    # OPTIMIZATION: Preload all structures into memory before encoding loop
    preloaded_coords = {}
    if is_structure_encoder and structure_paths:
        print(f"  Preloading {len(structure_paths)} structures into memory...")
        for idx, struct_data in enumerate(structure_paths):
            if struct_data:
                # Handle both PDB paths (str) and coordinate dicts (dict)
                if isinstance(struct_data, str):
                    # Traditional PDB file path
                    coords = load_pdb_coords(struct_data, device)
                    preloaded_coords[idx] = coords
                elif isinstance(struct_data, dict) and "coords" in struct_data:
                    # ProteinWorkshop coordinate dict: {"coords": np.array([L, 37, 3])}
                    coords_np = struct_data["coords"]  # [L, 37, 3]
                    # Extract backbone atoms: N(0), CA(1), C(2), O(3), CB(4)
                    # ProteinWorkshop uses standard atom ordering
                    backbone_indices = [0, 1, 2, 3, 4]
                    coords_backbone = coords_np[:, backbone_indices, :]  # [L, 5, 3]
                    # Convert to torch tensor: [1, L, 5, 3]
                    coords = torch.from_numpy(coords_backbone).unsqueeze(0).float().to(device)
                    preloaded_coords[idx] = coords
            if (idx + 1) % 1000 == 0:
                print(f"    Loaded {idx + 1}/{len(structure_paths)} structures", flush=True)
        print(f"  Preloading complete. Starting encoding...")

    # Sort sequences by length (shorter first) for better memory management
    seq_indices = list(range(len(sequences)))
    seq_lengths = [len(seq) for seq in sequences]
    sorted_pairs = sorted(zip(seq_indices, seq_lengths), key=lambda x: x[1])
    sorted_indices = [p[0] for p in sorted_pairs]

    # Filter out extremely long sequences that exceed GPU memory (all encoders)
    # Use more aggressive filtering for gcpnet to prevent OOM crashes
    if encoder_name == "gcpnet":
        MAX_LENGTH = 400
        print(f"  [INFO] Using MAX_LENGTH={MAX_LENGTH} for gcpnet (OOM prevention)")
    else:
        MAX_LENGTH = 600
    filtered_indices = [idx for idx in sorted_indices if seq_lengths[idx] <= MAX_LENGTH]
    skipped_count = len(sorted_indices) - len(filtered_indices)
    if skipped_count > 0:
        print(f"  [INFO] Skipping {skipped_count} sequences > {MAX_LENGTH} residues")
    sorted_indices = filtered_indices

    with torch.no_grad():
        for batch_idx in range(0, len(sorted_indices), batch_size):
            batch_original_indices = sorted_indices[batch_idx:batch_idx+batch_size]
            for idx in batch_original_indices:
                seq = sequences[idx]
                try:
                    seq_upper = seq.upper()
                    L = len(seq)

                    seq_tensor = torch.tensor(
                        [AA_VOCAB.get(aa, 0) for aa in seq_upper],
                        dtype=torch.long
                    ).unsqueeze(0).to(device)
                    mask = torch.ones(1, L, dtype=torch.bool, device=device)

                    # Use preloaded structure coordinates
                    if is_structure_encoder and structure_paths and structure_paths[idx]:
                        coords = preloaded_coords.get(idx)
                        if coords is None:
                            # Fallback to dummy coords if PDB load fails
                            coords = torch.zeros(1, L, 5, 3, device=device)
                        else:
                            # Ensure coords match sequence length
                            L_pdb = coords.shape[1]
                            if L_pdb != L:
                                # Truncate or pad to match sequence length
                                if L_pdb > L:
                                    coords = coords[:, :L, :, :]
                                else:
                                    # Pad with zeros
                                    pad = torch.zeros(1, L - L_pdb, 5, 3, device=device)
                                    coords = torch.cat([coords, pad], dim=1)
                    else:
                        coords = torch.zeros(1, L, 5, 3, device=device)

                    out = encoder.encode(ProteinInput(coords=coords, sequence=seq_tensor, mask=mask))

                    if residue_level:
                        # Per-residue embeddings: [1, L, D] → [L, D]
                        if out.residue_embeddings is not None:
                            emb = out.residue_embeddings.squeeze(0).cpu().float().numpy()
                        elif out.global_embedding is not None:
                            # Fallback: broadcast global to residue-level
                            g = out.global_embedding.squeeze(0).cpu().float().numpy()
                            emb = np.tile(g, (L, 1))
                        else:
                            results[idx] = None
                            continue
                        results[idx] = emb  # [L, D]
                    else:
                        # Protein-level embedding: prefer global, fallback to mean-pool residue
                        if out.global_embedding is not None:
                            emb = out.global_embedding.squeeze(0).cpu().float().numpy()  # [D]
                        elif out.residue_embeddings is not None:
                            emb = out.residue_embeddings.squeeze(0).cpu().float().numpy().mean(axis=0)
                        else:
                            results[idx] = None
                            continue
                        results[idx] = emb  # [D]

                except Exception as e:
                    print(f"\n  [ERROR] Encoding sequence {idx}: {e}")
                    traceback.print_exc()
                    results[idx] = None
                    # Clear GPU cache on error to prevent memory leaks
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # Adaptive memory clearing based on sequence length
                if L > 500:
                    # Clear memory after long sequences to prevent fragmentation
                    torch.cuda.empty_cache()

            if (batch_idx // batch_size) % 10 == 0:
                done = min(batch_idx + batch_size, len(sequences))
                print(f"    [{done}/{len(sequences)}]", end="\r", flush=True)

    return results


def save_split(out_dir: Path, split: str, embeddings: list, labels,
               ids: List[str], residue_level: bool):
    """Save embeddings + labels for one split."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filter out None embeddings
    valid = [(i, e) for i, e in enumerate(embeddings) if e is not None]
    if not valid:
        return 0

    valid_idx = [v[0] for v in valid]
    valid_emb = [v[1] for v in valid]

    if residue_level:
        # Object array for ragged residue-level embeddings
        arr = np.empty(len(valid_emb), dtype=object)
        for j, e in enumerate(valid_emb):
            arr[j] = e
        np.save(out_dir / f"{split}_embeddings.npy", arr, allow_pickle=True)
    else:
        np.save(out_dir / f"{split}_embeddings.npy",
                np.stack(valid_emb).astype(np.float32))

    # Labels: handle tuples (contact_prediction and deeploc return multiple label arrays)
    if isinstance(labels, tuple):
        for li, lab in enumerate(labels):
            if lab is None:
                continue
            lab_sub = [lab[i] for i in valid_idx]
            # Use object array for ragged (variable-length) label arrays
            try:
                arr_lab = np.array(lab_sub)
                if arr_lab.dtype == object or arr_lab.ndim == 1:
                    np.save(out_dir / f"{split}_labels_{li}.npy", arr_lab, allow_pickle=True)
                else:
                    np.save(out_dir / f"{split}_labels_{li}.npy", arr_lab)
            except ValueError:
                arr_lab = np.empty(len(lab_sub), dtype=object)
                for j, v in enumerate(lab_sub):
                    arr_lab[j] = v
                np.save(out_dir / f"{split}_labels_{li}.npy", arr_lab, allow_pickle=True)
    elif isinstance(labels[0], np.ndarray) and labels[0].ndim > 0:
        # Variable-length per-residue labels or GO multi-hot
        arr = np.empty(len(valid_idx), dtype=object)
        for j, i in enumerate(valid_idx):
            arr[j] = labels[i]
        np.save(out_dir / f"{split}_labels.npy", arr, allow_pickle=True)
    else:
        np.save(out_dir / f"{split}_labels.npy",
                np.array([labels[i] for i in valid_idx]))

    with open(out_dir / f"{split}_ids.txt", "w") as f:
        f.write("\n".join(ids[i] for i in valid_idx))

    return len(valid_emb)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_encoder_task(encoder_name: str, task: str, force: bool = False, mutation_effect_subsample: int = None):
    out_dir = EMB_DIR / task / encoder_name

    # Skip check
    if not force and (out_dir / "meta.json").exists():
        with open(out_dir / "meta.json") as f:
            meta = json.load(f)
        print(f"  [skip] {encoder_name}/{task} — already done "
              f"(n_train={meta.get('n_train')}, n_test={meta.get('n_test')})")
        return True

    task_cfg     = TASKS[task]
    residue_lvl  = (task_cfg["level"] == "residue")

    # Load data
    print(f"  Loading data for {task}...")
    try:
        # Pass subsample parameter to mutation_effect loader
        if task == "mutation_effect" and mutation_effect_subsample is not None:
            result = TASK_LOADERS[task](subsample_size=mutation_effect_subsample)
        else:
            result = TASK_LOADERS[task]()
        # Handle loaders that return structure paths (4-tuple) vs those that don't (3-tuple)
        if len(result[0]) == 4:
            (tr_seqs, tr_labels, tr_ids, tr_structures), (te_seqs, te_labels, te_ids, te_structures) = result
        else:
            (tr_seqs, tr_labels, tr_ids), (te_seqs, te_labels, te_ids) = result
            tr_structures = te_structures = None

    except Exception as e:
        print(f"  [ERROR] Loading {task}: {e}")
        traceback.print_exc()
        return False

    # Skip structure encoders if no structure data available
    if encoder_name not in SEQUENCE_ONLY_ENCODERS and tr_structures is None:
        print(f"  [skip] {encoder_name}/{task} — structure encoder but no PDB files available")
        return True

    # Handle shared deeploc loader: pick the right label array
    if task == "subcellular_localization" and isinstance(tr_labels, tuple):
        tr_labels = tr_labels[0]
        te_labels = te_labels[0]
    elif task == "membrane_soluble" and isinstance(tr_labels, tuple):
        tr_labels = tr_labels[1]
        te_labels = te_labels[1]

    # Load encoder
    print(f"  Loading encoder {encoder_name}...")
    try:
        # For GeoBPE, use environment variable or default to 16 workers
        num_workers = 1
        if encoder_name == "geobpe":
            num_workers = int(os.environ.get("GEOBPE_NUM_WORKERS", "16"))
            print(f"  Using {num_workers} worker processes for GeoBPE")
        encoder, device = load_encoder(encoder_name, num_workers=num_workers)
        D = encoder.embedding_dim
    except Exception as e:
        print(f"  [ERROR] Loading encoder {encoder_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

    t0 = time.time()

    # Determine batch size based on encoder memory requirements and available VRAM
    # Get available VRAM (in GB)
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    else:
        vram_gb = 16  # fallback

    # Memory-intensive structure encoders need smaller batches
    MEMORY_INTENSIVE_ENCODERS = [
        'cdconv', 'cdconv_pretrained',
        'dmasif', 'dmasif_pretrained',
        'gearnet', 'gearnet_pretrained',
        'foldvision', 'mace'
    ]

    # GCPNet and ESM3 need batch_size=1 due to memory scaling
    # GCPNet: graph size scales with residues × k_neighbors × geometric features
    # ESM3: geometric attention memory scaling
    if encoder_name in ['gcpnet', 'gcpnet_pretrained', 'esm3']:
        batch_size = 1
        print(f"  Using batch_size=1 for {encoder_name} (graph/attention memory scaling)")
    elif encoder_name in MEMORY_INTENSIVE_ENCODERS:
        # Dynamic batch size for memory-intensive encoders based on VRAM
        if vram_gb >= 40:
            batch_size = 8  # 2x increase for 40GB+ VRAM
        elif vram_gb >= 24:
            batch_size = 6
        else:
            batch_size = 4
        print(f"  Using batch_size={batch_size} for {encoder_name} (VRAM: {vram_gb:.1f}GB)")
    else:
        # Standard encoders: scale batch size with VRAM
        if vram_gb >= 40:
            batch_size = 64  # 2x increase for 40GB+ VRAM
        elif vram_gb >= 24:
            batch_size = 48
        else:
            batch_size = 32
        print(f"  Using batch_size={batch_size} for {encoder_name} (VRAM: {vram_gb:.1f}GB)")

    # A100 optimization: allow batch size multiplier via environment variable
    # Skip multiplier for gcpnet (already optimized with k_neighbors reduction)
    multiplier = int(os.environ.get('BATCH_SIZE_MULTIPLIER', '1'))
    if multiplier > 1 and encoder_name not in ['gcpnet', 'gcpnet_pretrained']:
        batch_size = batch_size * multiplier
        print(f"  Applying {multiplier}x multiplier -> batch_size={batch_size}")

    # Encode train
    print(f"  Encoding {len(tr_seqs)} train proteins...")
    tr_embs = encode_sequences(encoder, encoder_name, tr_seqs,
                               residue_level=residue_lvl,
                               batch_size=batch_size,
                               structure_paths=tr_structures)
    n_train = save_split(out_dir, "train", tr_embs, tr_labels, tr_ids, residue_lvl)

    # Encode test
    print(f"\n  Encoding {len(te_seqs)} test proteins...")
    te_embs = encode_sequences(encoder, encoder_name, te_seqs,
                               residue_level=residue_lvl,
                               batch_size=batch_size,
                               structure_paths=te_structures)
    n_test  = save_split(out_dir, "test",  te_embs, te_labels, te_ids, residue_lvl)

    elapsed = time.time() - t0

    # Save metadata
    meta = {
        "encoder":      encoder_name,
        "task":         task,
        "embedding_dim": D,
        "n_train":      n_train,
        "n_test":       n_test,
        "residue_level": residue_lvl,
        "elapsed_sec":  round(elapsed, 1),
        "device":       str(device),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  [done] {encoder_name}/{task} — "
          f"n_train={n_train}, n_test={n_test}, D={D}, {elapsed:.0f}s")

    # Free VRAM after encoder completes
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default="schnet",
                        help="Encoder name or 'all'")
    parser.add_argument("--task",    default="remote_homology",
                        choices=list(TASKS.keys()) + ["all"])
    parser.add_argument("--force",   action="store_true",
                        help="Re-run even if already done")
    parser.add_argument("--skip-done", action="store_true", default=True)
    parser.add_argument("--mutation-effect-subsample", type=int, default=None,
                        help="Subsample mutation_effect to N sequences (default: use all)")
    args = parser.parse_args()

    encoders = ALL_ENCODERS if args.encoder == "all" else [args.encoder]
    tasks    = list(TASKS.keys()) if args.task == "all" else [args.task]

    total = len(encoders) * len(tasks)
    done  = 0
    for enc in encoders:
        for task in tasks:
            done += 1
            print(f"\n{'='*60}")
            print(f"[{done}/{total}] {enc} × {task}")
            print(f"{'='*60}")
            run_encoder_task(enc, task, force=args.force,
                           mutation_effect_subsample=args.mutation_effect_subsample)


if __name__ == "__main__":
    main()
