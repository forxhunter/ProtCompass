"""SaProt (westlake-repl/SaProt_650M_AF2), sequence input only: the modality ablation of patched/saprot_hf.py.

Same checkpoint, tokenizer, layer (last hidden state) and slicing as patched/saprot_hf.py. Every 3Di letter is replaced
by '#', the symbol SaProt uses for a masked or unavailable structure token (the vocabulary holds the 20 tokens 'A#' ...
'Y#'; checked in ab_modality/PRESPEC.md). The structure-aware token of residue i is therefore sequence[i] + '#'.
`coords` and `pdb_path` are accepted for interface compatibility and ignored; the input audit measures that.
"""
import torch
from transformers import EsmModel, EsmTokenizer

HUB = "westlake-repl/SaProt_650M_AF2"
MASK_3DI = "#"


class SaProtSeqOnly:
    residue_level = True

    def __init__(self, device):
        self.tok = EsmTokenizer.from_pretrained(HUB)
        self.model = EsmModel.from_pretrained(HUB, add_pooling_layer=False).eval().to(device)
        self.device = device
        vocab = self.tok.get_vocab()
        missing = [a + MASK_3DI for a in "ACDEFGHIKLMNPQRSTVWY" if a + MASK_3DI not in vocab]
        if missing:
            raise RuntimeError(f"SaProt vocabulary lacks {missing}")
        self.unk_id = self.tok.unk_token_id
        self.meta = {"checkpoint": HUB, "checkpoint_revision": getattr(self.model.config, "_commit_hash", None),
                     "n_parameters": int(sum(p.numel() for p in self.model.parameters())),
                     "inputs_passed": "amino-acid track: real sequence; 3Di track: '#' (masked structure token) at every residue",
                     "patch": "patched/saprot_seqonly.py (modality ablation of patched/saprot_hf.py)"}

    @torch.no_grad()
    def residue(self, sequence, coords=None, pdb_path=None):
        sa = "".join(a + MASK_3DI for a in sequence)
        enc = self.tok(sa, return_tensors="pt").to(self.device)
        if enc["input_ids"].shape[1] != len(sequence) + 2:
            raise RuntimeError(f"SaProt: {enc['input_ids'].shape[1]} tokens for {len(sequence)} residues")
        if (enc["input_ids"] == self.unk_id).any():
            raise RuntimeError("SaProt seq-only: an input token mapped to <unk>")
        h = self.model(**enc).last_hidden_state[0, 1:1 + len(sequence)]
        return h.float().cpu().numpy()


def build(device):
    return SaProtSeqOnly(device)
