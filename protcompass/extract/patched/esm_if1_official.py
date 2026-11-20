"""ESM-IF1 encoder output through the official helper of the ESM repository.

Reason for the patch. The registry wrapper (src/models/pretrained/esm_if_encoder.py) passes the coordinates straight
to `model.encoder`. The official pipeline, `esm.inverse_folding.util.get_encoder_output`, first runs the coordinate
batch converter, which adds the start and end positions the model was trained with, and removes them from the output.
Without them every residue is shifted by one position in the encoder; on the same coordinates the output differs from
the official one by a relative 0.28 to 0.36. Here the official helper is called.

Input: N, CA, C coordinates of one chain. Output: (L, 512).
"""
import torch

from common import weights_sha256


def _stub_biotite():
    import sys
    import types
    if "biotite" in sys.modules:
        return
    try:
        import biotite.structure.io  # noqa: F401
        return
    except Exception:
        pass
    biotite, bs = types.ModuleType("biotite"), types.ModuleType("biotite.structure")
    bs.AtomArray = type("AtomArray", (), {})
    bs.filter_backbone = bs.get_chains = lambda *a, **k: None
    biotite.structure = bs
    mods = {"biotite": biotite, "biotite.structure": bs}
    for name in ("biotite.structure.io", "biotite.structure.io.pdbx", "biotite.structure.io.pdb", "biotite.structure.residues",
                 "biotite.sequence", "biotite.database", "biotite.database.rcsb"):
        mods[name] = types.ModuleType(name)
    mods["biotite.structure.residues"].get_residues = lambda *a, **k: None
    mods["biotite.sequence"].ProteinSequence = type("ProteinSequence", (), {})
    mods["biotite.structure.io"].pdbx = mods["biotite.structure.io.pdbx"]
    mods["biotite.structure.io"].pdb = mods["biotite.structure.io.pdb"]
    sys.modules.update(mods)


class ESMIF1Official:
    def __init__(self, device):
        _stub_biotite()                           # the ESM package imports biotite for file parsing, which is not used here
        self.model, self.alphabet = torch.hub.load("facebookresearch/esm:main", "esm_if1_gvp4_t16_142M_UR50")
        self.model = self.model.eval().to(device)
        import esm.inverse_folding
        self.util = esm.inverse_folding.util
        self.device = device
        self.meta = {"checkpoint": "esm_if1_gvp4_t16_142M_UR50 (torch.hub, facebookresearch/esm)",
                     "weights_sha256": weights_sha256(self.model),
                     "n_parameters": int(sum(p.numel() for p in self.model.parameters())),
                     "inputs_passed": "N, CA, C coordinates of the chain through the official coordinate batch converter; no sequence",
                     "patch": "patched/esm_if1_official.py"}

    @torch.no_grad()
    def residue(self, sequence, coords=None, pdb_path=None):
        out = self.util.get_encoder_output(self.model, self.alphabet, torch.from_numpy(coords[:, :3]).float())
        return out.float().cpu().numpy()


def build(device):
    return ESMIF1Official(device)
