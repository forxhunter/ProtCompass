"""Reader for labels.h5. Depends on h5py and numpy only.

    data = ProbingData()                                   # opens labels.h5 next to this file
    for i in data.indices("train", split="superfamily"):
        p = data.protein(i)                                # coords (L,5,3), sequence (L,), mask (L,), id, pdb_path
        lab = data.residue_labels(i)                       # dict of per-residue arrays, -1 or NaN where undefined
        pairs = data.pairs(i)                              # i, j, distance, distance_bin of the fixed sampled pairs

`coords` follows the project convention (N, CA, C, O, CB; virtual CB for glycine) and `sequence` uses the project
alphabet ACDEFGHIKLMNPQRSTVWY, so a protein can be passed to the encoders as
ProteinInput(coords=coords[None], sequence=sequence[None], mask=mask[None]).
"""
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent
SPLIT_CODE = {"train": 0, "val": 1, "test": 2}
RESIDUE_LABELS = ["phi", "psi", "phi_sin", "phi_cos", "psi_sin", "psi_cos", "phi_bin", "psi_bin", "curvature", "dssp3", "dssp8",
                  "sasa", "rsa", "rsa_bin", "bfactor_z", "bfactor_raw", "core", "cb_neighbours", "contact_count",
                  "contact_density_bin", "sequence_recovery", "segment"]
PROTEIN_LABELS = ["contact_order", "radius_of_gyration", "asphericity"]


class ProbingData:
    def __init__(self, path=ROOT / "labels.h5"):
        self.h = h5py.File(path, "r")
        p = self.h["protein"]
        self.ids = [x.decode() for x in p["id"][:]]
        self.length = p["length"][:]
        self.offset = p["offset"][:]
        self.pair_offset = self.h["pair/offset"][:]
        self.pair_count = self.h["pair/count"][:]

    def __len__(self):
        return len(self.ids)

    def indices(self, part, split="superfamily"):
        return np.where(self.h[f"protein/split_{split}"][:] == SPLIT_CODE[part])[0]

    def _rows(self, i):
        return slice(int(self.offset[i]), int(self.offset[i]) + int(self.length[i]))

    def protein(self, i):
        r = self._rows(i)
        return {"id": self.ids[i], "coords": self.h["residue/coords"][r], "sequence": self.h["residue/aa"][r].astype(np.int64),
                "mask": np.ones(int(self.length[i]), dtype=bool), "pdb_path": str(ROOT / "structures" / f"{self.ids[i]}.pdb")}

    def residue_labels(self, i, names=RESIDUE_LABELS):
        r = self._rows(i)
        return {k: self.h[f"residue/{k}"][r] for k in names}

    def pairs(self, i):
        s = slice(int(self.pair_offset[i]), int(self.pair_offset[i]) + int(self.pair_count[i]))
        return {k: self.h[f"pair/{k}"][s] for k in ("i", "j", "distance", "distance_bin")}

    def protein_labels(self, split="superfamily"):
        """Protein-level targets for all domains: raw, length-residualised (fit on the training split) and architecture."""
        p = self.h["protein"]
        out = {k: p[k][:] for k in PROTEIN_LABELS}
        out.update({f"{k}_len_resid": p[f"{k}_len_resid_{split}"][:] for k in PROTEIN_LABELS})
        out["architecture_label"] = p[f"architecture_label_{split}"][:]
        return out
