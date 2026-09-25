"""ESM3 (esm3_sm_open_v1) in process, with the true amino-acid sequence and real structure input.

Reason for the patch: src/models/pretrained/esm3_worker.py decodes the project's residue indices (A=0, C=1, ...) with the
ESM token order (L=0, A=1, G=2, ...), so no residue was decoded correctly and ESM3 embedded a substituted sequence. The worker
also passed coordinates without structure tokens and was started once per protein. Here the sequence string is tokenised
directly, `model.encode` derives the structure tokens from the real backbone, and the model is loaded once.
"""
import importlib.util

import numpy as np
import torch

from common import ROOT

_WORKER = ROOT / "src" / "models" / "pretrained" / "esm3_worker.py"
_LOCAL = ROOT / "esm3-sm-open-v1"
_BACKBONE_ATOM37 = {"N": 0, "CA": 1, "C": 2, "CB": 3, "O": 4}


class ESM3InProcess:
    def __init__(self, device):
        spec = importlib.util.spec_from_file_location("esm3_worker_module", _WORKER)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        self.model, self.tokenizers = mod._build_esm3_local(_LOCAL, device)
        self.device = device
        self.meta = {"checkpoint": str(_LOCAL / "data" / "weights" / "esm3_sm_open_v1.pth"), "dtype": "bfloat16",
                     "n_parameters": int(sum(p.numel() for p in self.model.parameters())),
                     "inputs_passed": "sequence track: true residues; structure track: VQ-VAE tokens from the real backbone (N, CA, C, O, CB) "
                                      "plus coordinates for geometric attention; function, SASA and secondary-structure tracks at their defaults",
                     "patch": "patched/esm3_inproc.py"}

    @torch.no_grad()
    def residue(self, sequence, coords=None, pdb_path=None):
        from esm.sdk.api import ESMProtein, LogitsConfig
        L = len(sequence)
        atom37 = torch.full((L, 37, 3), float("nan"))
        xyz = torch.from_numpy(coords)
        for k, name in enumerate(("N", "CA", "C", "O", "CB")):
            atom37[:, _BACKBONE_ATOM37[name]] = xyz[:, k]
        pt = self.model.encode(ESMProtein(sequence=sequence, coordinates=atom37))
        out = self.model.logits(pt, LogitsConfig(return_embeddings=True))
        emb = out.embeddings[0]
        if emb.shape[0] != L + 2:
            raise RuntimeError(f"ESM3: {emb.shape[0]} token states for {L} residues")
        return emb[1:1 + L].float().cpu().numpy()


def build(device):
    return ESM3InProcess(device)
