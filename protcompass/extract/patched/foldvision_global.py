"""FoldVision global embedding (AlexanderKroll/foldvision-encoder), one structure file per call, failures raise.

Reason for the patch (protcompass/atlas/reextract/README.md): the project wrapper returned zeros because `protein.pdb_ids`
was never set, FoldVision's id handling breaks on file names that contain ".pdb" twice, and the worker's fallback raised a
TypeError that was caught as a skip. Here every structure is copied to a dot-free name, preprocessed and encoded; a missing
voxel file raises. FoldVision has no residue states: the 1024-d global vector is returned for every residue row, so only the
protein-level mean is meaningful and the encoder is excluded from residue-level probing. The voxel CNN is frame dependent
(a rotated copy changes the embedding by 11%, against 18% for a different protein of the same length), so every structure
is first rotated onto its principal axes with a fixed sign convention. Runs in the `foldvision` conda env.
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

FV_ROOT = (PROTCOMPASS_ROOT + "/protcompass/external_repos/FoldVision")


def write_canonical_copy(src, dst):
    """Copy a structure file with all atoms rotated onto the principal axes of its CA trace (fixed axis signs)."""
    lines = [l for l in open(src) if l.startswith(("ATOM", "HETATM"))]
    xyz = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in lines], dtype=np.float64)
    ca = xyz[[l[12:16].strip() == "CA" and l.startswith("ATOM") for l in lines]]
    centre = ca.mean(0)
    _, _, vt = np.linalg.svd(ca - centre, full_matrices=False)
    sign = np.sign((((ca - centre) @ vt.T) ** 3).sum(0)); sign[sign == 0] = 1.0
    rot = vt * sign[:, None]
    if np.linalg.det(rot) < 0:
        rot[2] *= -1.0
    new = (xyz - centre) @ rot.T
    with open(dst, "w") as fh:
        for l, (x, y, z) in zip(lines, new):
            fh.write(f"{l[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{l[54:]}")
        fh.write("END\n")


class FoldVisionGlobal:
    residue_level = False

    def __init__(self, device):
        sys.path.insert(0, FV_ROOT)
        from foldvision import FoldVisionEncoder
        self.model = FoldVisionEncoder.from_pretrained("AlexanderKroll/foldvision-encoder").to(device).eval()
        self.device = device
        self.meta = {"checkpoint": "AlexanderKroll/foldvision-encoder", "n_parameters": int(sum(p.numel() for p in self.model.parameters())),
                     "inputs_passed": "the structure file as given (all atoms present in it), voxelised by FoldVision's own preprocessing",
                     "patch": "patched/foldvision_global.py", "residue_level": False,
                     "pose": "principal-axis pose with fixed axis signs, applied to the structure file before voxelisation"}

    @torch.no_grad()
    def residue(self, sequence, coords=None, pdb_path=None):
        from foldvision.dataloader import get_input_matrix
        from foldvision.preprocess.pdb_to_points import preprocess_Structures
        if not pdb_path:
            raise RuntimeError("FoldVision needs a structure file")
        work = Path(tempfile.mkdtemp(prefix="fv_"))
        try:
            (work / "pdb").mkdir(); (work / "pre").mkdir()
            write_canonical_copy(pdb_path, work / "pdb" / "q.pdb")
            preprocess_Structures(save_dir=str(work / "pre"), PDB_dir=str(work / "pdb"), continue_run=False)
            npz = work / "pre" / "numpy_3D_point_lists" / "q.npz"
            if not npz.exists():
                raise RuntimeError(f"FoldVision produced no voxel file for {pdb_path}")
            boxes = np.load(work / "pre" / "bounding_boxes.npy", allow_pickle=True).item()
            box = [int(v) for v in str(boxes["q.npz"]).strip("[]").replace(",", " ").split()]
            x = torch.tensor(get_input_matrix(str(npz), box, augment=False), dtype=torch.float32).unsqueeze(0).to(self.device)
            g = self.model.encode(x).squeeze(0).float().cpu().numpy()
        finally:
            shutil.rmtree(work, ignore_errors=True)
        return np.tile(g, (len(sequence), 1))


def build(device):
    return FoldVisionGlobal(device)
