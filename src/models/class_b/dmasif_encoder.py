"""
dMaSIF encoder wrapper — molecular surface encoder.

dMaSIF (Deep Molecular Surface Interaction Fingerprinting) encodes proteins via
their molecular surface, computed on-the-fly from backbone atom coordinates.
Architecturally distinct from all other encoders: operates on surface point clouds
rather than residue graphs or sequences.

Reference:
    Sverrisson et al. (2021). "Fast end-to-end learning on protein surfaces."
    CVPR 2021. https://github.com/ntranoslab/dMaSIF

Pretrained checkpoint: dMaSIF_search_3layer_12A_16dim (PPI-site search task)
Architecture: 3-layer surface conv, radius=12Å, emb_dim=16
Embedding dim: 16D per surface point → mapped to residues via CA nearest-neighbor

Notes:
    - Uses backbone atoms only (N, CA, C, O, CB); original trained on all heavy atoms.
    - Surface is generated on-the-fly via atoms_to_points_normals (no precomputation).
    - Requires pykeops.
    - Processes one protein at a time (surface point count varies per protein).
"""

import sys
import os
from argparse import Namespace
from typing import Optional

# Configure PyKeops for CUDA before importing
os.environ['CUDA_PATH'] = os.environ.get('CUDA_HOME', '/usr/local/cuda')
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

import torch
import torch.nn as nn

# Try to configure pykeops for CUDA
try:
    import pykeops
    pykeops.clean_pykeops()  # Clear any cached CPU-only builds
except:
    pass

from src.models.base_encoder import (
    BaseProteinEncoder,
    BottleneckClass,
    EncoderOutput,
    ProteinInput,
    register_encoder,
)

# Backbone atom element types → 6D one-hot: [C, H, O, N, S, Se]
# Atom order in ProteinInput.coords: N(0), CA(1), C(2), O(3), CB(4)
_BACKBONE_ELEMENT_IDX = [3, 0, 0, 2, 0]  # N→3, CA/C/CB→C→0, O→2
_ATOM_DIMS = 6

# Args reconstructed from state dict + model name (dMaSIF_search_3layer_12A_16dim)
_DMASIF_ARGS = Namespace(
    atom_dims=6,
    emb_dims=16,
    in_channels=16,            # 10 curvatures + 6 atomnet output
    orientation_units=16,
    post_units=16,
    n_layers=3,
    radius=12.0,
    search=True,
    site=False,
    use_mesh=False,
    embedding_layer="dMaSIF",
    curvature_scales=[1.0, 2.0, 3.0, 5.0, 10.0],
    resolution=1.0,
    distance=1.05,
    variance=0.1,
    sup_sampling=20,
    dropout=0.0,
    no_chem=False,
    no_geom=False,
    k=40,                      # for DGCNN (unused but required by init)
)


def _build_atom_features(coords: torch.Tensor, mask: torch.Tensor):
    """
    Convert ProteinInput backbone coords to dMaSIF atom cloud.

    Args:
        coords: (L, 5, 3) backbone atom coords for one protein
        mask: (L,) valid residue mask

    Returns:
        atom_xyz: (N_atoms, 3) all valid backbone atom coords
        atomtypes: (N_atoms, 6) one-hot element type
        batch_atoms: (N_atoms,) zeros (single protein)
    """
    valid_coords = coords[mask]  # (L_valid, 5, 3)
    L_valid = valid_coords.shape[0]

    # Flatten to atom cloud
    atom_xyz = valid_coords.reshape(-1, 3)  # (L_valid * 5, 3)
    N_atoms = atom_xyz.shape[0]

    # Build 6D one-hot atom types
    atomtypes = torch.zeros(N_atoms, _ATOM_DIMS, dtype=atom_xyz.dtype, device=atom_xyz.device)
    for atom_idx, elem_idx in enumerate(_BACKBONE_ELEMENT_IDX):
        rows = torch.arange(atom_idx, N_atoms, 5, device=atom_xyz.device)
        atomtypes[rows, elem_idx] = 1.0

    batch_atoms = torch.zeros(N_atoms, dtype=torch.long, device=atom_xyz.device)
    return atom_xyz, atomtypes, batch_atoms


@register_encoder("dmasif")
class dMaSIFEncoder(BaseProteinEncoder):
    """
    dMaSIF molecular surface encoder.

    Encodes proteins via their solvent-excluded molecular surface computed
    on-the-fly from backbone atom coordinates. Returns 16D per-residue
    embeddings via nearest-neighbor mapping from surface points to CA atoms.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self._model = None
        self._dmasif_dir = os.path.join(
            os.path.dirname(__file__), "../../../external_repos/dMaSIF"
        )

    def _ensure_model(self):
        if self._model is not None:
            return
        # Add dMaSIF to path for imports
        dmasif_abs = os.path.abspath(self._dmasif_dir)
        if dmasif_abs not in sys.path:
            sys.path.insert(0, dmasif_abs)

        # Patch missing torch_geometric.nn classes removed in newer versions.
        # dMaSIF's benchmark_models.py imports PointConv/XConv which no longer exist.
        # We only use dMaSIFConv_seg (not DGCNN/PointNet++), so stubs are safe.
        import torch_geometric.nn as tgnn
        for _stub_name in ("PointConv", "XConv"):
            if not hasattr(tgnn, _stub_name):
                setattr(tgnn, _stub_name, type(_stub_name, (nn.Module,), {
                    "__init__": lambda self, *a, **kw: nn.Module.__init__(self),
                    "forward": lambda self, *a, **kw: None,
                }))

        from model import dMaSIF
        self._model = dMaSIF(_DMASIF_ARGS)

    @property
    def bottleneck_class(self) -> BottleneckClass:
        return BottleneckClass.B  # surface point neighborhood

    @property
    def embedding_dim(self) -> int:
        return _DMASIF_ARGS.emb_dims  # 16

    def load_pretrained(self, checkpoint_path: str) -> None:
        self._ensure_model()
        ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        self._model.load_state_dict(state)
        self._model.eval()

    def encode(self, protein: ProteinInput) -> EncoderOutput:
        self._ensure_model()
        device = protein.coords.device
        B, L = protein.coords.shape[:2]

        # pykeops CUDA JIT segfaults on WSL2; keep surface computation on CPU.
        # The model weights are moved to CPU for inference; embeddings are moved
        # back to the target device after computation.
        cpu = torch.device("cpu")
        self._model.to(cpu)

        all_residue_embs = []

        for b in range(B):
            coords_b = protein.coords[b].to(cpu)   # (L, 5, 3)
            mask_b = protein.mask[b].to(cpu)        # (L,)

            # Build atom cloud
            atom_xyz, atomtypes, batch_atoms = _build_atom_features(coords_b, mask_b)

            # Center atoms (dMaSIF expects centered coordinates)
            center = atom_xyz.mean(dim=0, keepdim=True)
            atom_xyz_c = atom_xyz - center

            # CA positions for residue mapping (centered)
            ca_valid = coords_b[mask_b, 1, :] - center  # (L_valid, 3)

            P = {
                "atoms": atom_xyz_c,
                "atom_xyz": atom_xyz_c,
                "atomtypes": atomtypes,
                "batch_atoms": batch_atoms,
                "triangles": None,
            }

            with torch.no_grad():
                try:
                    _, _ = self._model.embed(P)
                    surf_emb = P["embedding_1"]   # (M_surface, 16)
                    surf_xyz = P["xyz"]            # (M_surface, 3)
                except Exception as e:
                    emb = torch.zeros(L, _DMASIF_ARGS.emb_dims, device=device)
                    all_residue_embs.append(emb)
                    continue

            # Map surface embeddings to residues via nearest CA → surface point
            dists = torch.cdist(ca_valid.float(), surf_xyz.float())  # (L_valid, M)
            nn_idx = dists.argmin(dim=1)                              # (L_valid,)
            residue_emb_valid = surf_emb[nn_idx]                     # (L_valid, 16)

            # Scatter back to full-length tensor; move to target device
            emb = torch.zeros(L, _DMASIF_ARGS.emb_dims, dtype=residue_emb_valid.dtype, device=device)
            valid_positions = mask_b.nonzero(as_tuple=True)[0]
            emb[valid_positions] = residue_emb_valid
            all_residue_embs.append(emb)

        # Model already on device, no need to move again
        # self._model.to(device)

        residue_embeddings = torch.stack(all_residue_embs, dim=0)  # (B, L, 16)
        global_emb = residue_embeddings.mean(dim=1)                # (B, 16) mean pool

        return EncoderOutput(
            residue_embeddings=residue_embeddings,
            global_embedding=global_emb,
        )
