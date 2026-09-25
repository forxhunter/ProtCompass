"""ESM3 (esm3_sm_open_v1), sequence input only: the modality ablation of patched/esm3_inproc.py.

Same checkpoint, loader, dtype (bfloat16 autocast inside model.logits), layer (final trunk embeddings) and slicing
(special tokens removed) as patched/esm3_inproc.py. The only difference is the input: `ESMProtein(sequence=...)` with no
coordinates, so the structure track holds the structure mask token at every position (ESM3 defaults), the per-residue
pLDDT conditioning is None, and the geometric attention receives NaN coordinates (affine mask all False).
`coords` and `pdb_path` are accepted for interface compatibility and ignored; the input audit measures that.
"""
import importlib.util

import torch

from common import ROOT

_WORKER = ROOT / "src" / "models" / "pretrained" / "esm3_worker.py"
_LOCAL = ROOT / "esm3-sm-open-v1"


class ESM3SeqOnly:
    residue_level = True

    def __init__(self, device):
        spec = importlib.util.spec_from_file_location("esm3_worker_module", _WORKER)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        self.model, self.tokenizers = mod._build_esm3_local(_LOCAL, device)
        self.device = device
        self.meta = {"checkpoint": str(_LOCAL / "data" / "weights" / "esm3_sm_open_v1.pth"), "dtype": "bfloat16",
                     "n_parameters": int(sum(p.numel() for p in self.model.parameters())),
                     "inputs_passed": "sequence track: true residues; structure track: mask token at every position (no "
                                      "coordinates, so no VQ-VAE tokens and no geometric attention input); function, SASA "
                                      "and secondary-structure tracks at their defaults",
                     "patch": "patched/esm3_seqonly.py (modality ablation of patched/esm3_inproc.py)"}

    @torch.no_grad()
    def residue(self, sequence, coords=None, pdb_path=None):
        from esm.sdk.api import ESMProtein, LogitsConfig
        L = len(sequence)
        pt = self.model.encode(ESMProtein(sequence=sequence))
        if pt.coordinates is not None:
            raise RuntimeError("ESM3 seq-only: encode() produced coordinates from a sequence-only input")
        out = self.model.logits(pt, LogitsConfig(return_embeddings=True))
        emb = out.embeddings[0]
        if emb.shape[0] != L + 2:
            raise RuntimeError(f"ESM3: {emb.shape[0]} token states for {L} residues")
        return emb[1:1 + L].float().cpu().numpy()


def build(device):
    return ESM3SeqOnly(device)
