"""
load_proteinworkshop.py — loaders for ProteinWorkshop tasks (T4-T6, T9-T10).

Reads the downloaded split files + pre-processed .pt structure files.
Returns (embeddings_dir, labels, ids) ready for evaluate_tasks.py.

Tasks:
  T4  EC reaction class    — downstream_tasks/data/proteinworkshop/ECReaction/
  T5  GO molecular function— downstream_tasks/data/proteinworkshop/GeneOntology/
  T6  GO biological process— same GeneOntology folder, different annotation column
  T9  PPI interface site   — downstream_tasks/data/proteinworkshop/masif_site/
  T10 Metal binding site   — downstream_tasks/data/proteinworkshop/Metal3D/

For structure encoders: loads .pt files (pre-processed by ProteinWorkshop as graphs).
For PLMs: extracts sequences from the .pt files.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np

PW_DIR = Path(__file__).parent.parent.parent / "structure_encoder_data" / "proteinworkshop"


def _load_pt_structure(pt_path: Path) -> Optional[Dict]:
    """Load a ProteinWorkshop pre-processed protein graph (.pt file)."""
    import torch
    try:
        data = torch.load(pt_path, weights_only=False, map_location='cpu')
        return data
    except Exception as e:
        return None


def load_ec_reaction(split: str = "testing") -> Tuple[List, List, List, List]:
    """
    T4: EC reaction class prediction.
    split: training | validation | testing
    Returns (pdb_ids, ec_labels, sequences, structures)
    ec_labels: integer 0-383 (384 classes after label encoding)
    structures: list of dicts with 'coords' key containing [L, 37, 3] coordinate arrays
    """
    import torch
    base     = PW_DIR / "ECReaction"
    split_f  = base / f"{split}.txt"
    label_f  = base / "chain_functions.txt"
    proc_dir = base / "processed"

    def norm(s):
        """Normalize 'PDBX.ChainY' → 'pdbx_ChainY' (lowercase PDB, preserve chain case).
        Processed filenames: 2pmo_X.pt (4-char lowercase PDB + uppercase chain)."""
        s = s.strip()
        if "." in s:
            pdb, chain = s.split(".", 1)
            return f"{pdb.lower()}_{chain}"
        if "_" in s:
            pdb, chain = s.rsplit("_", 1)
            return f"{pdb.lower()}_{chain}"
        return s.lower()

    # Build label map: pdb_chain -> EC int label
    label_map = {}
    ec_to_int = {}
    with open(label_f) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                pdb_chain = norm(parts[0])
                ec_str    = parts[1].strip()
                if ec_str not in ec_to_int:
                    ec_to_int[ec_str] = len(ec_to_int)
                label_map[pdb_chain] = ec_to_int[ec_str]

    pdb_ids, labels, sequences, structures = [], [], [], []
    with open(split_f) as f:
        for line in f:
            pdb_chain = norm(line)
            if not pdb_chain:
                continue
            pt_file = proc_dir / f"{pdb_chain}.pt"
            label   = label_map.get(pdb_chain, -1)
            if label == -1:
                continue

            data = _load_pt_structure(pt_file)
            if data is None:
                continue

            seq = _extract_sequence(data)

            # Extract coordinates for structure encoders
            try:
                import torch
                ns = data.node_stores[0]
                coords = ns.get("coords", None)
                if coords is not None:
                    if isinstance(coords, torch.Tensor):
                        coords = coords.cpu().numpy()
                    structure = {"coords": coords}
                else:
                    structure = None
            except Exception:
                structure = None

            pdb_ids.append(pdb_chain)
            labels.append(label)
            sequences.append(seq)
            structures.append(structure)

    print(f"[ECReaction/{split}] {len(pdb_ids)} proteins, {len(ec_to_int)} EC classes")
    return pdb_ids, labels, sequences, structures


def load_go(ontology: str = "mf", split: str = "test") -> Tuple[List, List, List, List]:
    """
    T5/T6: GO multi-label prediction.
    ontology: mf | bp | cc
    Returns (pdb_ids, go_labels_binary_matrix, sequences, structures)
    """
    base     = PW_DIR / "GeneOntology"
    proc_dir = base / "processed"

    # Load annotations
    annot_file = base / "nrPDB-GO_annot.tsv"
    go_terms   = {}   # pdb_chain -> set of GO IDs per ontology
    all_go_ids = set()

    def norm_go(s):
        """Normalize GO pdb-chain IDs → lowercase PDB + underscore + original chain case.
        e.g. '11AS-A' → '11as_A', '2ZPA-A' → '2zpa_A'"""
        s = s.strip()
        for sep in ("-", ".", "_"):
            if sep in s:
                pdb, chain = s.split(sep, 1)
                return f"{pdb.lower()}_{chain}"
        return s.lower()

    import csv
    with open(annot_file) as f:
        reader = csv.reader(f, delimiter="\t")
        # Skip header lines (starting with ###) and GO term/name metadata lines
        for row in reader:
            if not row or row[0].startswith("###") or row[0].startswith("GO:"):
                continue
            # Check if looks like an actual data row (4-char PDB-chain ID)
            raw_id = row[0].strip()
            if len(raw_id) < 4 or len(row) < 2:
                continue
            pdb_chain = norm_go(raw_id)
            # Columns: PDB-chain, GO-terms (MF), GO-terms (BP), GO-terms (CC)
            col_map = {"mf": 1, "bp": 2, "cc": 3}
            col_idx = col_map.get(ontology, 1)
            if col_idx < len(row) and row[col_idx]:
                terms = [t.strip() for t in row[col_idx].split(",") if t.strip() and t.strip().startswith("GO:")]
                if terms:
                    go_terms[pdb_chain] = set(terms)
                    all_go_ids.update(terms)

    # Load split file
    if split == "test":
        split_file = base / "nrPDB-GO_test.txt"
    elif split == "valid":
        split_file = base / "nrPDB-GO_valid.txt"
    else:
        split_file = base / "nrPDB-GO_train.txt"

    go_id_list = sorted(all_go_ids)
    go_idx     = {g: i for i, g in enumerate(go_id_list)}

    pdb_ids, labels, sequences, structures = [], [], [], []
    with open(split_file) as f:
        for line in f:
            pdb_chain = norm_go(line.strip())
            if not pdb_chain:
                continue
            pt_file = proc_dir / f"{pdb_chain}.pt"
            terms   = go_terms.get(pdb_chain, set())
            if not terms:
                continue

            data = _load_pt_structure(pt_file)
            if data is None:
                continue

            # Binary label vector
            label_vec = np.zeros(len(go_id_list), dtype=np.float32)
            for t in terms:
                if t in go_idx:
                    label_vec[go_idx[t]] = 1.0

            seq = _extract_sequence(data)

            # Extract coordinates for structure encoders
            try:
                import torch
                ns = data.node_stores[0]
                coords = ns.get("coords", None)
                if coords is not None:
                    if isinstance(coords, torch.Tensor):
                        coords = coords.cpu().numpy()
                    structure = {"coords": coords}
                else:
                    structure = None
            except Exception:
                structure = None

            pdb_ids.append(pdb_chain)
            labels.append(label_vec)
            sequences.append(seq)
            structures.append(structure)

    print(f"[GO-{ontology.upper()}/{split}] {len(pdb_ids)} proteins, {len(go_id_list)} GO terms")
    return pdb_ids, labels, sequences, structures


def load_masif_site(split: str = "testing") -> Tuple[List, List, List, List]:
    """
    T9: PPI interface site prediction (residue-level binary).
    Returns (pdb_ids, per_residue_labels, sequences, structures)
    """
    base     = PW_DIR / "masif_site"
    split_f  = base / f"{split}.txt"
    proc_dir = base / "processed"

    pdb_ids, labels, sequences, structures = [], [], [], []
    with open(split_f) as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            # MaSIF-site format: "1A99_D" (uppercase, keep case for filename)
            pt_file = proc_dir / f"{raw}.pt"
            data    = _load_pt_structure(pt_file)
            if data is None:
                continue

            # Labels: compute from graph_y chain identifier
            # graph_y = chain ID of the interacting partner -> mark chain residues
            ns = data.node_stores[0]
            partner_chain = ns.get("graph_y", None)
            chains        = ns.get("chains", None)
            if partner_chain is None or chains is None:
                continue
            # Residues belonging to the partner chain are labeled 1 (interface)
            import torch
            chain_ids   = ns["chains"]   # integer chain IDs per residue
            unique_c    = chain_ids.unique()
            # Use the first chain as query (partner marked by graph_y = chain letter)
            res_labels = (chain_ids != chain_ids[0]).long().numpy()
            seq        = _extract_sequence_from_store(ns)

            # Extract structure coordinates
            try:
                coords = ns.get("coords", None)
                if coords is not None:
                    structure = {"coords": coords.numpy()}
                else:
                    structure = None
            except Exception:
                structure = None

            pdb_ids.append(raw)
            labels.append(res_labels)
            sequences.append(seq)
            structures.append(structure)

    print(f"[MaSIF-site/{split}] {len(pdb_ids)} proteins")
    return pdb_ids, labels, sequences, structures


def load_metal3d(split: str = "test") -> Tuple[List, List, List, List]:
    """
    T10: Metal binding site prediction (residue-level binary).
    Split file format: "1AOL" (4-char PDB ID only).
    Processed files: "1aol_all.pt"
    Returns (pdb_ids, labels_per_residue, sequences, structures)
    """
    base     = PW_DIR / "Metal3D"
    split_f  = base / f"{split}.txt"
    proc_dir = base / "processed"

    pdb_ids, labels, sequences, structures = [], [], [], []
    with open(split_f) as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            # Metal3D: "1AOL" → "1aol_all.pt"
            pdb_key = raw.lower()
            pt_file = proc_dir / f"{pdb_key}_all.pt"
            data    = _load_pt_structure(pt_file)
            if data is None:
                continue

            # Compute metal binding labels: residues within 8Å CA of any metal ion
            import torch
            ns     = data.node_stores[0]
            coords = ns.get("coords", None)   # [L, 37, 3]
            hetatms= ns.get("hetatms", [])
            if coords is None:
                continue

            ca_coords  = coords[:, 1, :]       # CA atoms [L, 3]
            L          = ca_coords.shape[0]
            res_labels = torch.zeros(L, dtype=torch.long)
            for het in hetatms:
                for metal_name, metal_coords_t in het.items():
                    for mc in metal_coords_t:
                        dists      = torch.norm(ca_coords - mc.unsqueeze(0), dim=-1)
                        res_labels = res_labels | (dists < 8.0).long()

            seq = _extract_sequence_from_store(ns)

            # Extract structure coordinates
            try:
                coords_np = coords.numpy() if hasattr(coords, 'numpy') else coords
                struct = {"coords": coords_np}
            except Exception:
                struct = None

            pdb_ids.append(pdb_key)
            labels.append(res_labels.numpy())
            sequences.append(seq)
            structures.append(struct)

    print(f"[Metal3D/{split}] {len(pdb_ids)} proteins")
    return pdb_ids, labels, sequences, structures


AA_TOKENS = list("ACDEFGHIKLMNPQRSTVWYX")
RES3_TO_1 = {
    "ALA":"A","CYS":"C","ASP":"D","GLU":"E","PHE":"F","GLY":"G","HIS":"H",
    "ILE":"I","LYS":"K","LEU":"L","MET":"M","ASN":"N","PRO":"P","GLN":"Q",
    "ARG":"R","SER":"S","THR":"T","VAL":"V","TRP":"W","TYR":"Y","UNK":"X",
}


def _extract_sequence_from_store(ns) -> str:
    """Extract sequence from a ProteinWorkshop node store dict."""
    try:
        if "residues" in ns:
            residues = ns["residues"]
            if isinstance(residues, list):
                return "".join(RES3_TO_1.get(r, "X") for r in residues)
        if "residue_type" in ns:
            tokens = ns["residue_type"].numpy()
            return "".join(AA_TOKENS[min(int(t), len(AA_TOKENS)-1)] for t in tokens)
    except Exception:
        pass
    return ""


def _extract_sequence(data) -> str:
    """Extract amino acid sequence from a ProteinWorkshop protein graph."""
    try:
        ns = data.node_stores[0]
        return _extract_sequence_from_store(ns)
    except Exception:
        return ""


def _extract_node_labels(data) -> Optional[np.ndarray]:
    """Extract per-residue binary labels from ProteinWorkshop graph."""
    import torch
    try:
        if hasattr(data, "y") and data.y is not None:
            return data.y.numpy().astype(np.int64)
        if hasattr(data, "node_y"):
            return data.node_y.numpy().astype(np.int64)
    except Exception:
        pass
    return None


def inspect_pt_file(pt_path: Path):
    """Utility: print all attributes of a .pt file for debugging."""
    data = _load_pt_structure(pt_path)
    if data is None:
        print("Failed to load")
        return
    for attr in dir(data):
        if attr.startswith("_"):
            continue
        try:
            val = getattr(data, attr)
            if hasattr(val, "shape"):
                print(f"  {attr}: shape={val.shape} dtype={val.dtype}")
            elif not callable(val):
                print(f"  {attr}: {val}")
        except Exception:
            pass


if __name__ == "__main__":
    # Quick inspection of one file from each task
    import sys
    print("=== ECReaction sample ===")
    proc = PW_DIR / "ECReaction" / "processed"
    if proc.exists():
        sample = next(proc.glob("*.pt"), None)
        if sample:
            inspect_pt_file(sample)

    print("\n=== Metal3D sample ===")
    proc = PW_DIR / "Metal3D" / "processed"
    if proc.exists():
        sample = next(proc.glob("*.pt"), None)
        if sample:
            inspect_pt_file(sample)
