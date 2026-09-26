"""Official GCPNet (PSR checkpoint), restored with strict=True, loaded once, on the all-atom structure file.

Reason for the patch (found during the atlas re-extraction): the project's
gcpnet_worker.py defines its own 44-tensor network; none of the 399 checkpoint tensors match it and strict=False hides that,
so "gcpnet_pretrained" ran with random weights, rebuilt unseeded in a new subprocess for every protein. The model build below
follows protcompass/atlas/reextract/extract_gcpnet.py. Output per residue: mean over the residue's heavy atoms of the 100
scalar channels plus the 16 vector-channel norms after the last interaction layer (invariant, 116 dims).
Runs in the `gcpnet` conda env; the repository imports itself as `src`, so it must not be mixed with the project's `src`.
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))
import glob
import sys

import numpy as np
import pandas as pd
import torch

GCP = (PROTCOMPASS_ROOT + "/protcompass/external_repos/GCPNet")


def atom_frame(path):
    rows, chain0 = [], None
    for line in open(path):
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM"):
            continue
        chain0 = line[21] if chain0 is None else chain0
        if line[21] != chain0 or line[16] not in (" ", "A"):
            continue
        el = (line[76:78].strip() or line[12:16].strip()[0]).upper()
        rows.append((el, float(line[30:38]), float(line[38:46]), float(line[46:54]), line[22:27]))
    df = pd.DataFrame(rows, columns=["element", "x", "y", "z", "res"])
    return df[df.element != "H"].reset_index(drop=True)


class GCPNetOfficial:
    def __init__(self, device, random_weights=False):
        sys.path = [p for p in sys.path if not p.rstrip("/").endswith("2026_encoder/2026_encoder")]   # the project's own `src` would shadow GCPNet's
        sys.path.insert(0, GCP)
        for m in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
            del sys.modules[m]
        from omegaconf import open_dict
        from src.datamodules.components.atom3d_dataset import BaseTransform
        from src.models.components.gcpnet import GCPInteractions
        from src.models.gcpnet_psr_module import GCPNetPSRLitModule
        ckpt = glob.glob(GCP + "/checkpoints/PSR/*.ckpt")[0]
        raw = torch.load(ckpt, map_location="cpu", weights_only=False)
        hp = raw["hyper_parameters"]

        def fill(cfg, defaults):
            with open_dict(cfg):
                for k, v in defaults.items():
                    if k not in cfg:
                        cfg[k] = v
        fill(hp["module_cfg"], {"ablate_scalars": False, "ablate_vectors": False, "enable_e3_equivariance": False})
        self.model = None
        for residual_gcp in (True, False):
            hp_try = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in hp.items()}
            fill(hp_try["layer_cfg"]["mp_cfg"], {"use_residual_message_gcp": residual_gcp})
            if random_weights:
                from common import SEED
                torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
            cand = GCPNetPSRLitModule(layer_class=GCPInteractions, **hp_try)
            if random_weights:                       # same architecture and featurisation, no checkpoint
                self.model = cand.eval().to(device); break
            try:
                cand.load_state_dict(raw["state_dict"], strict=True)
                self.model = cand.eval().to(device); break
            except RuntimeError:
                continue
        if self.model is None:
            raise RuntimeError("the PSR checkpoint does not fit the repository model under strict loading")
        self.device, self.transform = device, BaseTransform(device="cpu")
        from common import file_sha256
        from common import weights_sha256
        self.meta = {"checkpoint": None if random_weights else ckpt, "checkpoint_sha256": None if random_weights else file_sha256(ckpt),
                     "checkpoint_loaded_by": "none (random weights, fixed seed)" if random_weights else "strict=True", "weights_sha256": weights_sha256(self.model),
                     "n_parameters": int(sum(p.numel() for p in self.model.parameters())),
                     "inputs_passed": "all heavy atoms of the structure file (elements and coordinates); residue identity enters through side-chain atoms",
                     "patch": "patched/gcpnet_official.py"}

    @torch.no_grad()
    def residue(self, sequence, coords=None, pdb_path=None):
        from torch_geometric.data import Batch
        if not pdb_path:
            raise RuntimeError("GCPNet needs the all-atom structure file")
        df = atom_frame(pdb_path)
        batch = Batch.from_data_list([self.transform(df)]).to(self.device)
        batch, _ = self.model(batch)
        feat = torch.cat([batch.h, batch.chi.norm(dim=-1)], dim=-1).float().cpu().numpy()
        res = pd.factorize(df.res)[0]
        return np.stack([feat[res == r].mean(0) for r in range(res.max() + 1)])


def build(device):
    return GCPNetOfficial(device)
