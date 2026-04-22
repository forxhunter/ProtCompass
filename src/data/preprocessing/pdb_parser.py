"""
PDB Parser for extracting protein backbones.

Extracts coordinates for N, CA, C, O, CB atoms and amino acid sequences.
Handles multi-chain PDBs by splitting them or merging (configurable).
"""

import numpy as np
import torch
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from typing import Dict, List, Optional, Tuple, Union
import warnings
from Bio import BiopythonWarning

# Suppress PDB construction warnings
warnings.simplefilter('ignore', BiopythonWarning)

# Standard amino acids mapping
AA_TO_index = {
    'ALA': 0, 'CYS': 1, 'ASP': 2, 'GLU': 3, 'PHE': 4,
    'GLY': 5, 'HIS': 6, 'ILE': 7, 'LYS': 8, 'LEU': 9,
    'MET': 10, 'ASN': 11, 'PRO': 12, 'GLN': 13, 'ARG': 14,
    'SER': 15, 'THR': 16, 'VAL': 17, 'TRP': 18, 'TYR': 19,
    'UNK': 20
}

def parse_pdb_structure(pdb_path: str, chain_id: Optional[str] = None) -> Optional[Dict[str, torch.Tensor]]:
    """
    Parse a PDB file and extract backbone coordinates + sequence.
    
    Args:
        pdb_path: Path to PDB file
        chain_id: Specific chain to extract. If None, extracts the first valid chain found.
        
    Returns:
        Dict with:
        - coords: (L, 5, 3) tensor [N, CA, C, O, CB]
        - sequence: (L,) tensor of indices
        - mask: (L,) boolean tensor (1 if valid CA, 0 otherwise)
    """
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("protein", pdb_path)
    except Exception as e:
        print(f"Error parsing {pdb_path}: {e}")
        return None
        
    model = structure[0] # Use first model
    
    # Select chain
    target_chain = None
    if chain_id:
        if chain_id in model:
            target_chain = model[chain_id]
        else:
            return None # Chain not found
    else:
        # Find first chain with protein residues
        for chain in model:
            if any(is_aa(res, standard=True) for res in chain):
                target_chain = chain
                break
    
    if target_chain is None:
        return None
        
    coords_list = []
    seq_list = []
    valid_mask = []
    
    # Iterate residues
    for residue in target_chain:
        if not is_aa(residue, standard=True):
            continue
            
        res_name = residue.get_resname()
        res_idx = AA_TO_index.get(res_name, 20)
        
        # Backbone atoms: N, CA, C, O, CB
        # Dimensions: (5, 3)
        res_coords = np.zeros((5, 3), dtype=np.float32)
        res_atoms_present = [False] * 5
        
        # Extract atoms
        mapping = {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4}
        
        has_ca = False
        
        for atom_name, idx in mapping.items():
            if atom_name in residue:
                res_coords[idx] = residue[atom_name].get_coord()
                res_atoms_present[idx] = True
                if atom_name == 'CA':
                    has_ca = True
                    
        # Special case for Glycine (no CB)
        if res_name == 'GLY' and has_ca:
            # Reconstruct virtual CB if needed (not strictly required for backbone-only models)
            # For now, leave as 0.0 and handle in downstream logic if critical
            # Ideally: construct geometric CB.
            pass
            
        if has_ca:
            coords_list.append(res_coords)
            seq_list.append(res_idx)
            valid_mask.append(1)
        else:
            # Gap handling? usually simply skip or insert masked placeholder
            # For InvariantsBench, we ignore fragments without CA
            pass
            
    if not coords_list:
        return None
        
    coordinates = torch.tensor(np.array(coords_list), dtype=torch.float32)
    sequence = torch.tensor(np.array(seq_list), dtype=torch.long)
    mask = torch.tensor(np.array(valid_mask), dtype=torch.bool)
    
    return {
        "coords": coordinates,
        "sequence": sequence,
        "mask": mask
    }
