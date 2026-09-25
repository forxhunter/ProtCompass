"""GVP with random weights, built once per process.

Reason for the patch: src/models/class_c/gvp_encoder.py starts a new conda subprocess for every protein and the worker
builds a fresh unseeded model each time, so every protein was embedded by a different random network; the wrapper also
returns the 16 vector channels flattened to 48 numbers, which change under rotation. Here the worker's own model class is
imported unchanged, instantiated once with a fixed seed, and the output is the 100 scalar channels plus the 16 vector norms.
"""
import importlib.util
from pathlib import Path

import numpy as np
import torch

from common import AA_TO_IDX, ROOT, SEED, weights_sha256

_WORKER = ROOT / "src" / "models" / "class_c" / "gvp_worker.py"


class GVPInProcess:
    def __init__(self, device):
        spec = importlib.util.spec_from_file_location("gvp_worker_module", _WORKER)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        self.model = mod.GVPModel(node_s_dim=100, node_v_dim=16, edge_s_dim=32, edge_v_dim=1, num_layers=3, k_neighbors=30,
                                  dropout=0.1, seed=SEED).eval().to(device)
        self.device = device
        self.meta = {"weights_sha256": weights_sha256(self.model), "seed": SEED, "n_parameters": int(sum(p.numel() for p in self.model.parameters())),
                     "inputs_passed": "backbone coordinates and residue identity", "patch": "patched/gvp_inproc.py"}

    @torch.no_grad()
    def residue(self, sequence, coords=None, pdb_path=None):
        L = len(sequence)
        c = torch.from_numpy(coords)[None].to(self.device)
        q = torch.tensor([[AA_TO_IDX[a] for a in sequence]], device=self.device)
        m = torch.ones(1, L, dtype=torch.bool, device=self.device)
        emb, _ = self.model(c, q, m)
        emb = emb[0].float()
        s, v = emb[:, :100], emb[:, 100:].reshape(L, 16, 3)
        return torch.cat([s, v.norm(dim=-1)], dim=-1).cpu().numpy()


def build(device):
    return GVPInProcess(device)
