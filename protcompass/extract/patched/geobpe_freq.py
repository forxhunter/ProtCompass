"""GeoBPE token frequencies with the merged tokenizer of the last local checkpoint (bpe_iter=220).

Reason for the patch (protcompass/atlas/reextract/README.md): the project wrapper zero-filled a protein whenever tokenisation
raised ValueError (30% of rows), used the pre-merge checkpoint bpe_iter=0, discarded the merged tokenizer that BPE.tokenize
returns, and mean-pooled rows of a seeded random lookup table. Here the structure file is tokenised directly, an angle outside
the learned histogram bins is snapped to the nearest bin and counted, and the protein vector is the normalised frequency of the
quantised token stream. There is no residue-aligned output: the vector is returned for every residue row and the encoder is
excluded from residue-level probing. Any other failure raises. Runs in the `GeoBPE-6` conda env on the CPU.
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))
import bisect
import pickle
import shutil
import sys
import tempfile
from pathlib import Path

from patched.foldvision_global import write_canonical_copy

import numpy as np

GEO = (PROTCOMPASS_ROOT + "/protcompass/external_repos/GeoBPE")
CKPT = GEO + "/ckpts/1774555624.5673451/bpe_iter=220.pkl"


class GeoBPEFreq:
    residue_level = False

    def __init__(self, device):
        sys.path.insert(0, GEO); sys.path.insert(0, GEO + "/foldingdiff")
        from foldingdiff import bpe as bpe_mod
        self.snapped = 0

        def get_ind_nearest(v, values):
            left = [s for s, _ in values]
            ind = bisect.bisect_right(left, v) - 1
            if 0 <= ind < len(values) and (values[ind][0] <= v < values[ind][1] or (ind == len(values) - 1 and v == values[ind][1])):
                return ind
            self.snapped += 1
            return int(np.argmin([min(abs(v - s), abs(v - e)) for s, e in values]))

        bpe_mod.BPE.get_ind = staticmethod(get_ind_nearest)
        self.model = pickle.load(open(CKPT, "rb"))
        self.vocab = self.model.vocab_size() if callable(self.model.vocab_size) else self.model.vocab_size
        from common import file_sha256
        self.meta = {"checkpoint": CKPT, "checkpoint_sha256": file_sha256(CKPT), "vocab": int(self.vocab), "n_parameters": 0,
                     "inputs_passed": "the structure file (backbone geometry), in a principal-axis pose", "pose": "principal-axis pose with fixed axis signs", "patch": "patched/geobpe_freq.py", "residue_level": False}

    def residue(self, sequence, coords=None, pdb_path=None):
        """The tokenizer reads Cartesian coordinates, so the same fold in a different pose gives a different token
        stream (measured: 9.1e-2 relative change under a rigid motion). The structure is put into a principal-axis
        pose first, as for IPA and FoldVision."""
        from foldingdiff.angles_and_coords import (canonical_distances_and_dihedrals, extract_aa_seq, extract_backbone_coords,
                                                   extract_c_beta_coords, extract_side_chain_coords)
        from foldingdiff.tokenizer import Tokenizer
        work = Path(tempfile.mkdtemp(prefix="geobpe_"))
        try:
            p = str(work / "q.pdb")
            write_canonical_copy(pdb_path, p)
            return self._tokenise(p, sequence, extract_aa_seq, canonical_distances_and_dihedrals, extract_backbone_coords,
                                  extract_c_beta_coords, extract_side_chain_coords)
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def _tokenise(self, p, sequence, extract_aa_seq, canonical_distances_and_dihedrals, extract_backbone_coords,
                  extract_c_beta_coords, extract_side_chain_coords):
        from foldingdiff.tokenizer import Tokenizer
        aa = extract_aa_seq(p)
        s = {"angles": canonical_distances_and_dihedrals(p, distances=["0C:1N", "N:CA", "CA:C"], angles=["phi", "psi", "omega", "tau", "CA:C:1N", "C:1N:1CA"]),
             "coords": extract_backbone_coords(p, atoms=["CA"]), "c_beta": extract_c_beta_coords(p),
             "full_coords": extract_backbone_coords(p, atoms=["N", "CA", "C"]), "full_idxes": sum([[i, i, i] for i in range(1, len(aa) + 1)], []),
             "side_chain": extract_side_chain_coords(p), "aa": aa, "fname": p}
        tk, _ = self.model.tokenize(Tokenizer(s))
        q = np.asarray(self.model.quantize(tk.tokenize()), dtype=np.int64)
        f = (np.bincount(q, minlength=self.vocab) / len(q)).astype(np.float32)
        self.meta["angles_snapped_to_nearest_bin"] = self.snapped
        return np.tile(f, (len(sequence), 1))


def build(device):
    return GeoBPEFreq(device)
