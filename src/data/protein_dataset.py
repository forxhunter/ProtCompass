"""
Protein structure dataset for probing experiments.

Provides standardized loading of protein structures with precomputed
invariant labels for I1-I10 probing tasks.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class ProteinStructureDataset(Dataset):
    def __init__(
        self,
        data_path: Union[str, Path],
        split: str = "train",
        max_length: int = 512,
        transform: Optional[Callable] = None,
        load_invariants: bool = True,
        invariant_families: Optional[List[str]] = None,
        fold_level: str = "topology",
    ):
        self.data_path = Path(data_path)
        self.split = split
        self.max_length = max_length
        self.transform = transform
        self.load_invariants = load_invariants
        self.invariant_families = invariant_families or [
            "I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9", "I10"
        ]
        self.fold_level = fold_level
        if self.fold_level not in {"architecture", "topology"}:
            raise ValueError(f"fold_level must be 'architecture' or 'topology', got {self.fold_level}")
        self.protein_ids = self._load_protein_ids()

    def _load_protein_ids(self) -> List[str]:
        if self.data_path.is_file():
            with h5py.File(self.data_path, "r") as f:
                if self.split in f:
                    return list(f[self.split].keys())
                if "proteins" in f:
                    return list(f["proteins"].keys())
                return list(f.keys())
        if self.data_path.is_dir():
            split_file = self.data_path / f"{self.split}.txt"
            if split_file.exists():
                with open(split_file) as f:
                    return [line.strip() for line in f if line.strip()]
            raise FileNotFoundError(f"Split file not found: {split_file}")
        raise FileNotFoundError(f"Data path not found: {self.data_path}")

    def __len__(self) -> int:
        return len(self.protein_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        protein_id = self.protein_ids[idx]
        with h5py.File(self.data_path, "r") as f:
            if self.split in f:
                grp = f[self.split][protein_id]
            elif "proteins" in f:
                grp = f["proteins"][protein_id]
            else:
                grp = f[protein_id]

            coords = torch.tensor(grp["coords"][:], dtype=torch.float32)
            sequence = torch.tensor(grp["sequence"][:], dtype=torch.long)

            length = coords.shape[0]
            if length > self.max_length:
                coords = coords[: self.max_length]
                sequence = sequence[: self.max_length]
                length = self.max_length
            mask = torch.ones(length, dtype=torch.bool)

            invariants: Dict[str, torch.Tensor] = {}
            if self.load_invariants:
                if "invariants" in grp:
                    invariants = self._load_invariants(grp["invariants"], length)
                if "I5" in self.invariant_families:
                    for key in ["dssp_3", "dssp_8"]:
                        if key in grp:
                            data = grp[key][:]
                            if len(data) > length:
                                data = data[:length]
                            invariants[key] = torch.tensor(data)
                if "I2" in self.invariant_families and "contacts" not in invariants:
                    from src.data.preprocessing.geometry_calculator import compute_pairwise_invariants
                    pairwise = compute_pairwise_invariants(coords.unsqueeze(0), mask.unsqueeze(0))
                    for key, value in pairwise.items():
                        invariants.setdefault(key, value.squeeze(0))

                if "I10" in self.invariant_families:
                    # I10: Sequence recovery - use sequence as the label
                    invariants["sequence"] = sequence

        sample = {
            "protein_id": protein_id,
            "coords": coords,
            "sequence": sequence,
            "mask": mask,
            "length": length,
            "invariants": invariants,
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def _load_invariants(self, inv_grp: h5py.Group, length: int) -> Dict[str, torch.Tensor]:
        invariants: Dict[str, torch.Tensor] = {}

        if "I1" in self.invariant_families:
            for key in ["phi", "psi", "phi_bins", "psi_bins", "curvature"]:
                if key in inv_grp:
                    data = inv_grp[key][:]
                    invariants[key] = torch.tensor(data[:length] if len(data) > length else data)

        if "I2" in self.invariant_families:
            for key in ["contacts", "distance_bins"]:
                if key in inv_grp:
                    data = inv_grp[key][:]
                    if data.shape[0] > length:
                        data = data[:length, :length]
                    invariants[key] = torch.tensor(data)

        if "I3" in self.invariant_families:
            for key in ["contact_density", "core_surface_labels"]:
                if key in inv_grp:
                    data = inv_grp[key][:]
                    data = data[:length] if len(data) > length else data
                    # Cast to long for classification labels
                    invariants[key] = torch.tensor(data, dtype=torch.long)

        if "I4" in self.invariant_families and "contact_order" in inv_grp:
            dataset = inv_grp["contact_order"]
            invariants["contact_order"] = torch.tensor(dataset[()] if dataset.shape == () else dataset[:])

        if "I5" in self.invariant_families:
            for key in ["dssp_3", "dssp_8"]:
                if key in inv_grp:
                    data = inv_grp[key][:]
                    invariants[key] = torch.tensor(data[:length] if len(data) > length else data)

        if "I6" in self.invariant_families:
            for key in ["rsa", "rsa_bins"]:
                if key in inv_grp:
                    data = inv_grp[key][:]
                    invariants[key] = torch.tensor(data[:length] if len(data) > length else data)

        if "I7" in self.invariant_families:
            # Try architecture_label first (new), then fold_label (old)
            if "architecture_label" in inv_grp:
                dataset = inv_grp["architecture_label"]
                raw_label = dataset[()] if dataset.shape == () else dataset[:]
                # Remap sparse labels to dense 0-30 range
                # CATH architecture labels are sparse (0,1,2,4,5,...,42)
                # Map to contiguous indices for CrossEntropyLoss
                if not hasattr(self, '_arch_label_map'):
                    # Build mapping on first access (will be same for all samples)
                    self._arch_label_map = {0: 0, 1: 1, 2: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7,
                                           10: 8, 11: 9, 12: 10, 14: 11, 15: 12, 19: 13, 24: 14,
                                           26: 15, 27: 16, 28: 17, 29: 18, 30: 19, 31: 20, 32: 21,
                                           33: 22, 35: 23, 36: 24, 37: 25, 38: 26, 39: 27, 40: 28,
                                           41: 29, 42: 30}
                remapped_label = self._arch_label_map.get(int(raw_label), int(raw_label))
                invariants["architecture_label"] = torch.tensor(remapped_label)
            elif "fold_label" in inv_grp:
                dataset = inv_grp["fold_label"]
                invariants["fold_label"] = torch.tensor(dataset[()] if dataset.shape == () else dataset[:])

        if "I8" in self.invariant_families and "bfactor" in inv_grp:
            data = inv_grp["bfactor"][:]
            invariants["bfactor"] = torch.tensor(data[:length] if len(data) > length else data)

        if "I9" in self.invariant_families:
            for key in ["radius_of_gyration", "asphericity"]:
                if key in inv_grp:
                    dataset = inv_grp[key]
                    invariants[key] = torch.tensor(dataset[()] if dataset.shape == () else dataset[:])

        if "I10" in self.invariant_families:
            # I10: Sequence recovery - the sequence itself is the label
            # It's already loaded in __getitem__, so we just mark it here
            pass

        return invariants


def collate_proteins(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    max_len = max(sample["length"] for sample in batch)
    batch_size = len(batch)
    coords = torch.zeros(batch_size, max_len, 5, 3)
    sequence = torch.zeros(batch_size, max_len, dtype=torch.long)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    lengths = torch.zeros(batch_size, dtype=torch.long)
    protein_ids = []
    invariants: Dict[str, torch.Tensor] = {}

    for i, sample in enumerate(batch):
        length = sample["length"]
        coords[i, :length] = sample["coords"]
        sequence[i, :length] = sample["sequence"]
        mask[i, :length] = sample["mask"]
        lengths[i] = length
        protein_ids.append(sample["protein_id"])

        for key, value in sample["invariants"].items():
            if key not in invariants:
                if len(value.shape) == 0:
                    invariants[key] = torch.zeros(batch_size, dtype=value.dtype)
                else:
                    is_pairwise = len(value.shape) >= 2 and value.shape[0] == value.shape[1] == length
                    shape = (batch_size, max_len, max_len) + value.shape[2:] if is_pairwise else (batch_size, max_len) + value.shape[1:]
                    invariants[key] = torch.zeros(shape, dtype=value.dtype)
            if len(value.shape) == 0:
                invariants[key][i] = value
            else:
                is_pairwise = len(value.shape) >= 2 and value.shape[0] == value.shape[1] == length
                if is_pairwise:
                    invariants[key][i, :length, :length] = value
                else:
                    invariants[key][i, :length] = value

    return {
        "protein_id": protein_ids,
        "coords": coords,
        "sequence": sequence,
        "mask": mask,
        "length": lengths,
        "invariants": invariants,
    }


def get_dataloader(
    data_path: Union[str, Path],
    split: str = "train",
    batch_size: int = 32,
    shuffle: bool = True,
    max_length: int = 512,
    num_workers: int = 0,
    load_invariants: bool = True,
    invariant_families: Optional[List[str]] = None,
    fold_level: str = "topology",
) -> DataLoader:
    dataset = ProteinStructureDataset(
        data_path=data_path,
        split=split,
        max_length=max_length,
        load_invariants=load_invariants,
        invariant_families=invariant_families,
        fold_level=fold_level,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if split == "train" else False,
        num_workers=num_workers,
        collate_fn=collate_proteins,
        pin_memory=torch.cuda.is_available(),
    )


def create_dummy_data(n_proteins: int = 100, max_length: int = 256) -> List[Dict[str, Any]]:
    data = []
    for i in range(n_proteins):
        length = np.random.randint(50, max_length)
        data.append({
            "protein_id": f"dummy_{i}",
            "coords": torch.randn(length, 5, 3),
            "sequence": torch.randint(0, 20, (length,)),
            "mask": torch.ones(length, dtype=torch.bool),
            "length": length,
            "invariants": {
                "phi_bins": torch.randint(0, 36, (length,)),
                "psi_bins": torch.randint(0, 36, (length,)),
                "contact_density": torch.randint(0, 4, (length,)),
            },
        })
    return data
