"""
Structure loader for downstream task evaluation.

Provides functions to load protein structures from various sources:
1. Pre-computed structures (PDB files, HDF5 archives)
2. AlphaFold Database (if available)
3. On-the-fly prediction with ESMFold (if installed)

Returns standardized (coords, sequence, mask) tensors for encoder input.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import torch

from .preprocessing.pdb_parser import parse_pdb_structure


# Standard amino acid to index mapping
AA_TO_INDEX = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
    'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
    'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
    'X': 20  # Unknown
}

# Three-letter to one-letter code
AA3_TO_AA1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    'UNK': 'X'
}


class StructureLoader:
    """
    Flexible structure loader for downstream tasks.

    Supports multiple structure sources with fallback:
    1. Local PDB files
    2. HDF5 structure archives
    3. AlphaFold Database (if configured)
    4. ESMFold prediction (if installed)
    """

    def __init__(
        self,
        pdb_dir: Optional[Path] = None,
        h5_path: Optional[Path] = None,
        alphafold_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        use_esmfold: bool = False,
        device: str = "cuda",
    ):
        """
        Initialize structure loader.

        Args:
            pdb_dir: Directory containing PDB files
            h5_path: Path to HDF5 archive with pre-computed structures
            alphafold_dir: Directory with AlphaFold predicted structures
            cache_dir: Directory to cache predicted structures
            use_esmfold: Whether to use ESMFold for on-the-fly prediction
            device: Device for ESMFold prediction
        """
        self.pdb_dir = Path(pdb_dir) if pdb_dir else None
        self.h5_path = Path(h5_path) if h5_path else None
        self.alphafold_dir = Path(alphafold_dir) if alphafold_dir else None
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_esmfold = use_esmfold
        self.device = device

        # Initialize ESMFold if requested
        self.esmfold_model = None
        if use_esmfold:
            self._init_esmfold()

        # Create cache directory
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _init_esmfold(self):
        """Initialize ESMFold model for structure prediction."""
        try:
            import esm
            self.esmfold_model = esm.pretrained.esmfold_v1()
            self.esmfold_model = self.esmfold_model.eval().to(self.device)
            print("ESMFold initialized successfully")
        except ImportError:
            warnings.warn("ESMFold not available. Install with: pip install fair-esm")
            self.use_esmfold = False
        except Exception as e:
            warnings.warn(f"Failed to initialize ESMFold: {e}")
            self.use_esmfold = False

    def load_structure(
        self,
        sequence: str,
        protein_id: Optional[str] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Load or predict structure for a protein sequence.

        Tries sources in order:
        1. HDF5 archive (if protein_id provided)
        2. Local PDB file (if protein_id provided)
        3. AlphaFold DB (if protein_id provided)
        4. Cached prediction
        5. ESMFold prediction (if enabled)

        Args:
            sequence: Amino acid sequence (one-letter code)
            protein_id: Optional protein identifier (PDB ID, UniProt ID, etc.)

        Returns:
            Dict with:
                - coords: (L, 4, 3) or (L, 5, 3) tensor [N, CA, C, O] or [N, CA, C, O, CB]
                - sequence: (L,) tensor of amino acid indices
                - mask: (L,) boolean tensor
            Returns None if structure cannot be loaded/predicted
        """
        # Try HDF5 archive
        if self.h5_path and protein_id:
            result = self._load_from_h5(protein_id)
            if result is not None:
                return result

        # Try local PDB file
        if self.pdb_dir and protein_id:
            result = self._load_from_pdb(protein_id)
            if result is not None:
                return result

        # Try AlphaFold DB
        if self.alphafold_dir and protein_id:
            result = self._load_from_alphafold(protein_id)
            if result is not None:
                return result

        # Try cache
        if self.cache_dir and protein_id:
            result = self._load_from_cache(protein_id)
            if result is not None:
                return result

        # Predict with ESMFold
        if self.use_esmfold and self.esmfold_model is not None:
            result = self._predict_with_esmfold(sequence, protein_id)
            if result is not None:
                return result

        # All sources failed
        warnings.warn(f"Could not load structure for {protein_id or 'sequence'}")
        return None

    def _load_from_h5(self, protein_id: str) -> Optional[Dict[str, torch.Tensor]]:
        """Load structure from HDF5 archive."""
        try:
            import h5py
            with h5py.File(self.h5_path, 'r') as f:
                if protein_id in f:
                    grp = f[protein_id]
                    coords = torch.tensor(grp['coords'][:], dtype=torch.float32)
                    sequence = torch.tensor(grp['sequence'][:], dtype=torch.long)
                    mask = torch.ones(len(sequence), dtype=torch.bool)
                    return {'coords': coords, 'sequence': sequence, 'mask': mask}
        except Exception as e:
            pass
        return None

    def _load_from_pdb(self, protein_id: str) -> Optional[Dict[str, torch.Tensor]]:
        """Load structure from local PDB file."""
        # Try various PDB file naming conventions
        possible_names = [
            f"{protein_id}.pdb",
            f"{protein_id.lower()}.pdb",
            f"{protein_id.upper()}.pdb",
            f"{protein_id[:4]}.pdb",  # PDB ID only (first 4 chars)
        ]

        for name in possible_names:
            pdb_path = self.pdb_dir / name
            if pdb_path.exists():
                result = parse_pdb_structure(str(pdb_path))
                if result is not None:
                    return result

        return None

    def _load_from_alphafold(self, protein_id: str) -> Optional[Dict[str, torch.Tensor]]:
        """Load structure from AlphaFold Database."""
        # AlphaFold files are typically named: AF-{uniprot_id}-F1-model_v4.pdb
        possible_names = [
            f"AF-{protein_id}-F1-model_v4.pdb",
            f"AF-{protein_id}-F1-model_v3.pdb",
            f"AF-{protein_id}-F1-model_v2.pdb",
            f"{protein_id}.pdb",
        ]

        for name in possible_names:
            af_path = self.alphafold_dir / name
            if af_path.exists():
                result = parse_pdb_structure(str(af_path))
                if result is not None:
                    return result

        return None

    def _load_from_cache(self, protein_id: str) -> Optional[Dict[str, torch.Tensor]]:
        """Load structure from cache."""
        cache_path = self.cache_dir / f"{protein_id}.pdb"
        if cache_path.exists():
            result = parse_pdb_structure(str(cache_path))
            if result is not None:
                return result
        return None

    def _predict_with_esmfold(
        self,
        sequence: str,
        protein_id: Optional[str] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Predict structure using ESMFold."""
        if self.esmfold_model is None:
            return None

        try:
            with torch.no_grad():
                output = self.esmfold_model.infer(sequence)

            # Extract coordinates (N, CA, C, O)
            coords = output['positions'][-1, :, :4, :]  # (L, 4, 3)

            # Convert sequence to indices
            seq_indices = torch.tensor(
                [AA_TO_INDEX.get(aa, 20) for aa in sequence],
                dtype=torch.long
            )

            # Create mask (all valid for predicted structure)
            mask = torch.ones(len(sequence), dtype=torch.bool)

            result = {
                'coords': coords.cpu(),
                'sequence': seq_indices,
                'mask': mask,
            }

            # Cache if protein_id provided
            if self.cache_dir and protein_id:
                self._save_to_cache(result, protein_id)

            return result

        except Exception as e:
            warnings.warn(f"ESMFold prediction failed: {e}")
            return None

    def _save_to_cache(self, structure: Dict[str, torch.Tensor], protein_id: str):
        """Save predicted structure to cache as PDB."""
        try:
            cache_path = self.cache_dir / f"{protein_id}.pdb"
            self._write_pdb(structure, cache_path)
        except Exception as e:
            warnings.warn(f"Failed to cache structure: {e}")

    def _write_pdb(self, structure: Dict[str, torch.Tensor], output_path: Path):
        """Write structure to PDB file."""
        coords = structure['coords'].numpy()
        sequence = structure['sequence'].numpy()

        # Reverse mapping: index to one-letter code
        INDEX_TO_AA = {v: k for k, v in AA_TO_INDEX.items()}

        with open(output_path, 'w') as f:
            atom_num = 1
            for res_idx, (res_coords, seq_idx) in enumerate(zip(coords, sequence), 1):
                aa = INDEX_TO_AA.get(int(seq_idx), 'X')
                # Convert to 3-letter code
                aa3 = {v: k for k, v in AA3_TO_AA1.items()}.get(aa, 'UNK')

                atom_names = ['N', 'CA', 'C', 'O']
                if res_coords.shape[0] > 4:
                    atom_names.append('CB')

                for atom_idx, atom_name in enumerate(atom_names):
                    if atom_idx < len(res_coords):
                        x, y, z = res_coords[atom_idx]
                        f.write(
                            f"ATOM  {atom_num:5d}  {atom_name:3s} {aa3:3s} A{res_idx:4d}    "
                            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom_name[0]:>2s}\n"
                        )
                        atom_num += 1
            f.write("END\n")


def load_structures_batch(
    sequences: List[str],
    protein_ids: Optional[List[str]] = None,
    loader: Optional[StructureLoader] = None,
    **loader_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load structures for a batch of sequences.

    Args:
        sequences: List of amino acid sequences
        protein_ids: Optional list of protein identifiers
        loader: Pre-initialized StructureLoader (or create new one)
        **loader_kwargs: Arguments for StructureLoader if creating new one

    Returns:
        coords: (B, L_max, 4, 3) padded coordinates
        seq_tensor: (B, L_max) padded sequence indices
        mask: (B, L_max) padding mask
    """
    if loader is None:
        loader = StructureLoader(**loader_kwargs)

    if protein_ids is None:
        protein_ids = [None] * len(sequences)

    structures = []
    for seq, pid in zip(sequences, protein_ids):
        struct = loader.load_structure(seq, pid)
        if struct is not None:
            structures.append(struct)
        else:
            # Create dummy structure if loading failed
            warnings.warn(f"Using dummy structure for {pid or 'sequence'}")
            length = len(seq)
            structures.append({
                'coords': torch.randn(length, 4, 3),
                'sequence': torch.tensor([AA_TO_INDEX.get(aa, 20) for aa in seq]),
                'mask': torch.ones(length, dtype=torch.bool),
            })

    # Pad to same length
    max_len = max(s['coords'].shape[0] for s in structures)
    batch_size = len(structures)

    # Determine coordinate dimensions (4 or 5 atoms)
    n_atoms = structures[0]['coords'].shape[1]

    coords = torch.zeros(batch_size, max_len, n_atoms, 3)
    seq_tensor = torch.zeros(batch_size, max_len, dtype=torch.long)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, struct in enumerate(structures):
        length = struct['coords'].shape[0]
        coords[i, :length] = struct['coords']
        seq_tensor[i, :length] = struct['sequence']
        mask[i, :length] = struct['mask']

    return coords, seq_tensor, mask
