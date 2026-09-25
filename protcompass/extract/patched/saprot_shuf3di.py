"""SaProt (westlake-repl/SaProt_650M_AF2) with the real 3Di string permuted within each protein: a control for
patched/saprot_seqonly.py (ab_modality/PRESPEC_shuf.md).

Same checkpoint, tokenizer, layer (last hidden state) and slicing as patched/saprot_hf.py, and the same 3Di string
(mini3di on the real backbone, masked states filled with state 0, no pLDDT masking). The only change: the 3Di string of
the protein is permuted over its positions before it is paired with the amino acids,

    seed = int.from_bytes(sha256(protein_id.encode("utf-8")).digest()[:8], "big")
    perm = numpy.random.default_rng(seed).permutation(L)
    token i = sequence[i] + 3Di[perm[i]]

so the protein keeps its 3Di alphabet and composition (the multiset of 3Di letters) and loses the residue-to-structure
correspondence. The protein id is not part of the extract.py call signature, so the caller sets `protein_id` on this
object before every call (ab_modality/ab_shuf_extract.py does this); a call without an id raises.
"""
import hashlib

import numpy as np
import torch
from transformers import EsmModel, EsmTokenizer

from patched.foldseek_3di import states_3di

HUB = "westlake-repl/SaProt_650M_AF2"
ALPHABET_3DI = "acdefghiklmnpqrstvwy"


def seed_of(protein_id):
    return int.from_bytes(hashlib.sha256(str(protein_id).encode("utf-8")).digest()[:8], "big")


def permutation_of(protein_id, n):
    return np.random.default_rng(seed_of(protein_id)).permutation(n)


class SaProtShuf3Di:
    residue_level = True

    def __init__(self, device):
        self.tok = EsmTokenizer.from_pretrained(HUB)
        self.model = EsmModel.from_pretrained(HUB, add_pooling_layer=False).eval().to(device)
        self.device = device
        self.unk_id = self.tok.unk_token_id
        self.protein_id = None
        self.stats = {"calls": 0, "positions": 0, "same_letter_after_shuffle": 0}
        self.meta = {"checkpoint": HUB, "checkpoint_revision": getattr(self.model.config, "_commit_hash", None),
                     "n_parameters": int(sum(p.numel() for p in self.model.parameters())),
                     "inputs_passed": "amino-acid track: real sequence; 3Di track: real mini3di tokens permuted within the "
                                      "protein (seed = first 8 bytes of sha256(protein id)); no pLDDT masking",
                     "patch": "patched/saprot_shuf3di.py (control for patched/saprot_seqonly.py)"}

    def shuffled_3di(self, coords, n):
        if self.protein_id is None:
            raise RuntimeError("saprot_shuf3di: protein_id must be set before every call")
        t, _ = states_3di(coords)
        if len(t) != n:
            raise RuntimeError(f"saprot_shuf3di: {len(t)} 3Di states for {n} residues")
        ts = t[permutation_of(self.protein_id, n)]
        assert np.array_equal(np.sort(ts), np.sort(t))            # same multiset of 3Di letters
        self.stats["calls"] += 1
        self.stats["positions"] += n
        self.stats["same_letter_after_shuffle"] += int((ts == t).sum())
        return ts

    @torch.no_grad()
    def residue(self, sequence, coords=None, pdb_path=None):
        ts = self.shuffled_3di(coords, len(sequence))
        sa = "".join(a + ALPHABET_3DI[k] for a, k in zip(sequence, ts))
        enc = self.tok(sa, return_tensors="pt").to(self.device)
        if enc["input_ids"].shape[1] != len(sequence) + 2:
            raise RuntimeError(f"SaProt: {enc['input_ids'].shape[1]} tokens for {len(sequence)} residues")
        if (enc["input_ids"] == self.unk_id).any():
            raise RuntimeError("SaProt shuf3di: an input token mapped to <unk>")
        h = self.model(**enc).last_hidden_state[0, 1:1 + len(sequence)]
        return h.float().cpu().numpy()


def build(device):
    return SaProtShuf3Di(device)
