"""
Dataset wrapper for downstream classification tasks.
"""
from typing import Dict, Any, List, Tuple
import torch
from src.data.protein_dataset import ProteinStructureDataset, collate_proteins

class DownstreamDataset(ProteinStructureDataset):
    """
    Dataset that adds downstream task labels to protein samples.
    """
    def __init__(
        self,
        *args,
        labels: Dict[str, Tuple[int, int, int, int]] = None,
        split_file: str = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.labels = labels or {}
        
        # Prepare label mapping (cath_id -> key)
        # H5 keys are like '1oai_A' (pdb_chain), labels are '1oaiA00' (cath_id)
        # We need to map pdb_chain -> label
        self.key_to_label = {}
        for cath_id, label in self.labels.items():
            # Conversion: 1oaiA00 -> 1oai_A
            if len(cath_id) >= 5:
                key = f"{cath_id[:4]}_{cath_id[4]}"
                # Register both formats (PDB_CHAIN and CATH_ID) to be robust
                if key not in self.key_to_label:
                    self.key_to_label[key] = label
                if cath_id not in self.key_to_label:
                    self.key_to_label[cath_id] = label
        
        # Load split if provided
        self.split_keys = None
        if split_file:
            with open(split_file, 'r') as f:
                # Convert split CATH IDs to keys
                split_keys = set()
                for line in f:
                    line = line.strip()
                    if line and len(line) >= 5:
                        key = f"{line[:4]}_{line[4]}"
                        split_keys.add(key)
            self.split_keys = split_keys
            
            # Intersect with existing ids
            self.split_keys = split_keys
            
            # Intersect with existing ids
            original_len = len(self.protein_ids)
            filtered_ids = [pid for pid in self.protein_ids if pid in self.split_keys]
            
            if len(filtered_ids) == 0:
                print(f"Warning: Split file {split_file} resulted in 0 proteins! Falling back to ALL available proteins in HDF5.")
                # Keep original self.protein_ids (which are all keys in HDF5 that have labels)
                # This allows running on small/debug datasets where split file doesn't match
            else:
                self.protein_ids = filtered_ids
                print(f"Loaded split from {split_file}: {len(self.protein_ids)} / {original_len} proteins found")
        
        # Filter protein_ids to those that have labels
        self.original_ids = self.protein_ids
        self.protein_ids = [pid for pid in self.protein_ids if pid in self.key_to_label]
        
        print(f"Filtered {len(self.original_ids)} -> {len(self.protein_ids)} proteins with labels")
        
        # Create label mappings (to 0-indexed integers)
        # Use values from key_to_label to ensure coverage across splits
        all_labels = list(self.key_to_label.values())
        
        self.c_map = {c: i for i, c in enumerate(sorted(list(set(l[0] for l in all_labels))))}
        self.a_map = {a: i for i, a in enumerate(sorted(list(set(l[1] for l in all_labels))))}
        self.t_map = {t: i for i, t in enumerate(sorted(list(set(l[2] for l in all_labels))))}
        
        
        self.num_c = len(self.c_map)
        self.num_a = len(self.a_map)
        self.num_t = len(self.t_map)
        
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = super().__getitem__(idx)
        pid = sample["protein_id"]
        
        if pid in self.key_to_label:
            c, a, t, h = self.key_to_label[pid]
            # Add labels to sample
            # Use -1 for unknown if somehow it happens (though we filtered)
            sample["label_c"] = torch.tensor(self.c_map.get(c, -1), dtype=torch.long)
            sample["label_a"] = torch.tensor(self.a_map.get(a, -1), dtype=torch.long)
            sample["label_t"] = torch.tensor(self.t_map.get(t, -1), dtype=torch.long)
            
        return sample

def collate_downstream(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate with labels."""
    # Base collate
    batch_dict = collate_proteins(batch)
    
    # Add labels
    if "label_c" in batch[0]:
        batch_dict["label_c"] = torch.stack([s["label_c"] for s in batch])
        batch_dict["label_a"] = torch.stack([s["label_a"] for s in batch])
        batch_dict["label_t"] = torch.stack([s["label_t"] for s in batch])
        
    return batch_dict
