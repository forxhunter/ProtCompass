"""SaProt (westlake-repl/SaProt_650M_AF2) with real 3Di tokens.

Reason for the patch: both SaProt wrappers obtain 3Di tokens from saprot_encoder.get_3di_tokens, whose mini3di call raises and
is swallowed, so every residue received the constant 3Di state 0. Here the 3Di string is computed from the real backbone and
the structure-aware tokens (amino acid upper case + 3Di lower case) are passed to the Hugging Face checkpoint. No pLDDT
masking is applied; that choice is recorded in the metadata.
"""
import numpy as np
import torch
from transformers import EsmModel, EsmTokenizer

from patched.foldseek_3di import states_3di

HUB = "westlake-repl/SaProt_650M_AF2"
ALPHABET_3DI = "acdefghiklmnpqrstvwy"


class SaProtHF:
    def __init__(self, device):
        self.tok = EsmTokenizer.from_pretrained(HUB)
        self.model = EsmModel.from_pretrained(HUB, add_pooling_layer=False).eval().to(device)
        self.device = device
        self.meta = {"checkpoint": HUB, "checkpoint_revision": getattr(self.model.config, "_commit_hash", None),
                     "n_parameters": int(sum(p.numel() for p in self.model.parameters())),
                     "inputs_passed": "amino-acid track: real sequence; 3Di track: real tokens from mini3di on the structure; no pLDDT masking",
                     "patch": "patched/saprot_hf.py"}

    @torch.no_grad()
    def residue(self, sequence, coords=None, pdb_path=None):
        t, _ = states_3di(coords)
        sa = "".join(a + ALPHABET_3DI[k] for a, k in zip(sequence, t))
        enc = self.tok(sa, return_tensors="pt").to(self.device)
        if enc["input_ids"].shape[1] != len(sequence) + 2:
            raise RuntimeError(f"SaProt: {enc['input_ids'].shape[1]} tokens for {len(sequence)} residues")
        h = self.model(**enc).last_hidden_state[0, 1:1 + len(sequence)]
        return h.float().cpu().numpy()


def build(device):
    return SaProtHF(device)
