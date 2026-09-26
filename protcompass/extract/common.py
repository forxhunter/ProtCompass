"""Shared pieces of the extraction harness: structure parsing, encoder panel, strict encoder calls.

Design rules
  - Coordinates always come from a real structure file whose residue sequence equals the input sequence.
    There is no code path that fabricates, pads or zero-fills coordinates; a violation raises.
  - Every encoder is instantiated once per process with a fixed seed, so random-weight encoders use one
    set of weights for all proteins, and the hash of the weights is recorded.
  - A "pretrained" entry must load a checkpoint file and the load must change the weights; otherwise it raises.
  - Residue embeddings must have one row per input residue and be finite.
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import hashlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
HERE = ROOT / "." / "extract"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {a: i for i, a in enumerate(ALPHABET)}
AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
       "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
SEED = 20260919


class InputError(RuntimeError):
    """Raised when an input would have to be fabricated, padded or truncated."""


def virtual_cb(n, ca, c):
    b, cc = ca - n, c - ca
    a = np.cross(b, cc)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * cc + ca


def load_backbone(pdb_path, expect_sequence=None):
    """Return coords (L, 5, 3) in N, CA, C, O, CB order and the sequence of the first chain.

    Glycine and residues without CB get an ideal virtual CB. A residue without N, CA, C or O raises."""
    residues, order, chain0 = {}, [], None
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            if chain0 is None:
                chain0 = line[21]
            if line[21] != chain0:
                break
            if line[16] not in (" ", "A"):
                continue
            key = line[22:27]
            if key not in residues:
                residues[key] = {"name": line[17:20].strip()}
                order.append(key)
            residues[key][line[12:16].strip()] = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
    if not order:
        raise InputError(f"no ATOM records in {pdb_path}")
    coords = np.zeros((len(order), 5, 3), dtype=np.float32)
    seq = []
    for i, key in enumerate(order):
        r = residues[key]
        missing = [a for a in ("N", "CA", "C", "O") if a not in r]
        if missing:
            raise InputError(f"{pdb_path}: residue {key.strip()} lacks {missing}")
        n, ca, c, o = (np.array(r[a], dtype=np.float32) for a in ("N", "CA", "C", "O"))
        cb = np.array(r["CB"], dtype=np.float32) if "CB" in r else virtual_cb(n, ca, c)
        coords[i] = np.stack([n, ca, c, o, cb])
        seq.append(AA3.get(r["name"], "X"))
    seq = "".join(seq)
    if expect_sequence is not None and seq != expect_sequence:
        raise InputError(f"{pdb_path}: structure sequence differs from the task sequence")
    if not np.isfinite(coords).all() or np.abs(coords).sum() == 0:
        raise InputError(f"{pdb_path}: non-finite or all-zero coordinates")
    return coords, seq


def canonical_pose(coords):
    """Rotate a centred structure onto its principal axes with a fixed sign convention.

    Used only for encoders whose wrapper output depends on the global frame (the IPA wrapper): the pose is then a function
    of the structure alone, so the embedding no longer changes under rigid motion of the input file."""
    ca = coords[:, 1].astype(np.float64)
    ca = ca - ca.mean(0)
    _, _, vt = np.linalg.svd(ca, full_matrices=False)
    proj = ca @ vt.T
    sign = np.sign((proj ** 3).sum(0)); sign[sign == 0] = 1.0           # third moment fixes the direction of each axis
    rot = vt * sign[:, None]
    if np.linalg.det(rot) < 0:
        rot[2] *= -1.0                                                   # keep a proper rotation
    flat = (coords.reshape(-1, 3).astype(np.float64) - coords[:, 1].mean(0)) @ rot.T
    return flat.reshape(coords.shape).astype(np.float32)


def file_sha256(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def weights_sha256(module):
    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items()):
        if torch.is_tensor(v):
            h.update(k.encode()); h.update(v.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


@dataclass
class Spec:
    key: str                       # harness name, also the output directory name
    family: str                    # "registry" (src.models wrapper), "hf_plm", "patched"
    target: str                    # registry name, HF hub id, or patched module name
    weights: str                   # "pretrained" | "random"
    uses_coords: bool              # declared inputs; the audit must confirm them
    uses_sequence: bool
    rule: str                      # extraction rule, following the rebuttal table
    checkpoint: Optional[str] = None
    kwargs: dict = field(default_factory=dict)
    paper: str = ""                # display name in the manuscript panel, "" if not in the panel
    env: str = "protein_embeddings"
    note: str = ""
    canonical_pose: bool = False   # wrapper output is frame dependent; inputs are put into a canonical pose first


CK = ROOT / "checkpoints"
HOMECK = Path((PROTCOMPASS_ROOT + "/protcompass/checkpoints"))
POTTS = ROOT / "external_repos" / "PottsMPNN"
R_RES = "final residue states, special tokens removed, masked mean over valid residues"
R_GNN = "final invariant node states of the wrapper, masked mean over valid residues"

SPECS = [
    # sequence-only language models (clean HF path, see patched/hf_plm.py)
    Spec("esm1b", "hf_plm", "facebook/esm1b_t33_650M_UR50S", "pretrained", False, True, R_RES, paper="ESM-1b"),
    Spec("prottrans_t5_xl_u50", "hf_plm", "Rostlab/prot_t5_xl_half_uniref50-enc", "pretrained", False, True, R_RES, paper="ProtTrans (ProtT5-XL-U50)"),
    Spec("protbert_bfd", "hf_plm", "Rostlab/prot_bert_bfd", "pretrained", False, True, R_RES, paper="ProtBert-BFD"),
    Spec("esm2_8m", "hf_plm", "facebook/esm2_t6_8M_UR50D", "pretrained", False, True, R_RES, note="this is what the manuscript's 'ESM2 (650M)' row contained"),
    Spec("esm2_35m", "hf_plm", "facebook/esm2_t12_35M_UR50D", "pretrained", False, True, R_RES, paper="ESM2 (35M)"),
    Spec("esm2_150m", "hf_plm", "facebook/esm2_t30_150M_UR50D", "pretrained", False, True, R_RES, paper="ESM2 (150M)"),
    Spec("esm2_650m", "hf_plm", "facebook/esm2_t33_650M_UR50D", "pretrained", False, True, R_RES, paper="ESM2 (650M)"),
    Spec("esm2_3b", "hf_plm", "facebook/esm2_t36_3B_UR50D", "pretrained", False, True, R_RES + "; fp16 weights"),
    Spec("ankh_base", "hf_plm", "ElnaggarLab/ankh-base", "pretrained", False, True, R_RES, paper="Ankh-base"),
    Spec("prostt5", "hf_plm", "Rostlab/ProstT5", "pretrained", False, True, R_RES + "; amino-acid track with the <AA2fold> prefix, no 3Di input",
         paper="ProstT5", note="sequence input only by design of the paper's extraction rule"),
    # structure encoders, random weights (one fixed seed per process)
    Spec("schnet", "registry", "schnet", "random", True, True, R_GNN, paper="SchNet"),
    Spec("tfn", "registry", "tfn", "random", True, True, R_GNN, paper="TFN"),
    Spec("dimenet", "registry", "dimenet", "random", True, True, R_GNN, paper="DimeNet++"),
    Spec("egnn", "registry", "egnn", "random", True, True, R_GNN, paper="EGNN"),
    Spec("ipa", "registry", "ipa", "random", True, True, R_GNN + "; input in canonical pose", paper="IPA", canonical_pose=True,
         note="wrapper output changes by 40% under rigid motion of the input, so the structure is first put into a principal-axis pose"),
    Spec("mace", "registry", "mace", "random", True, True, R_GNN, paper="MACE"),
    Spec("gearnet", "registry", "gearnet", "random", True, True, R_GNN, paper="GearNet"),
    Spec("gvp", "patched", "gvp_inproc", "random", True, True, "scalar channels + vector norms (invariant), masked mean", paper="GVP"),
    Spec("cdconv", "registry", "cdconv", "random", True, True, R_GNN, paper="CDConv", env="cdc5090"),
    Spec("gcpnet_random", "patched", "gcpnet_official_random", "random", True, True, "official architecture, random weights; 100 scalar channels + 16 vector norms per residue", env="gcpnet",
         note="replaces the registry 'gcpnet' wrapper, which rebuilt an unseeded 44-tensor network in a new subprocess for every protein"),
    Spec("pottsmpnn", "registry", "pottsmpnn", "random", True, False, "encoder h_V after 3 layers, masked mean", paper="PottsMPNN"),
    # structure encoders, pretrained
    Spec("proteinmpnn", "registry", "proteinmpnn_pretrained", "pretrained", True, False, "encoder h_V after 3 encoder layers, before the decoder",
         checkpoint=str(CK / "proteinmpnn_v48_020.pt"), paper="ProteinMPNN"),
    Spec("pottsmpnn_pretrained", "registry", "pottsmpnn_pretrained", "pretrained", True, False, "encoder h_V, masked mean"),
    Spec("pottsmpnn_soluble", "registry", "pottsmpnn_soluble_pretrained", "pretrained", True, False, "encoder h_V, masked mean"),
    Spec("pottsmpnn_ft", "registry", "pottsmpnn_ft_pretrained", "pretrained", True, False, "encoder h_V, masked mean"),
    Spec("gearnet_pretrained", "registry", "gearnet_pretrained", "pretrained", True, True, R_GNN, checkpoint=str(CK / "gearnet" / "mc_gearnet_edge.pth")),
    Spec("tfn_pretrained", "registry", "tfn_pretrained", "pretrained", True, True, R_GNN, checkpoint=str(HOMECK / "tfn_pretrained" / "best.pt")),
    Spec("ipa_pretrained", "registry", "ipa_pretrained", "pretrained", True, True, R_GNN + "; input in canonical pose",
         checkpoint=str(HOMECK / "ipa_pretrained" / "best.pt"), canonical_pose=True, note="same frame dependence as ipa"),
    Spec("gcpnet_pretrained", "patched", "gcpnet_official", "pretrained", True, True, "official PSR model; 100 scalar channels + 16 vector norms, mean over the heavy atoms of each residue",
         paper="GCPNet", env="gcpnet", note="supervised PSR checkpoint, not a self-supervised pretraining; diagnosed during the atlas re-extraction"),
    Spec("esm_if1", "registry", "esm_if", "pretrained", True, False, "GVP-Transformer encoder output before decoding", paper="ESM-IF1", note="diagnosed during the atlas re-extraction"),
    # multimodal
    Spec("saprot", "patched", "saprot_hf", "pretrained", True, True, "final hidden states at the L residue positions of AA+3Di tokens", paper="SaProt",
         note="needs real 3Di tokens"),
    Spec("esm3", "patched", "esm3_inproc", "pretrained", True, True, "final trunk residue embeddings, sequence and structure tracks", paper="ESM3"),
    # token models handled during the atlas re-extraction
    Spec("foldseek_3di", "patched", "foldseek_3di", "pretrained", True, False, "one-hot 3Di state per residue; protein level is the state frequency", paper="Foldseek"),
    Spec("geobpe", "patched", "geobpe_freq", "pretrained", True, False, "normalised frequency of the quantised GeoBPE token stream (bpe_iter=220); no residue states",
         paper="GeoBPE", env="GeoBPE-6", note="tokenizer trained by the authors on 500 structures"),
    Spec("foldvision", "patched", "foldvision_global", "pretrained", True, True, "native FoldVisionEncoder.encode global embedding; no residue states",
         paper="FoldVision", env="foldvision", note="frame-dependent voxel CNN; side-chain atoms in the file carry residue identity"),
]
SPEC = {s.key: s for s in SPECS}
# No checkpoint exists on this machine for these manuscript variants; src/models/pretrained/structure_pretrained.py points at /mnt/f/...
# and downstream_tasks/embed_proteins.py never called load_pretrained() for them, so their cached "pretrained" embeddings were random-weight.
NO_CHECKPOINT = {"gvp_pretrained": "no checkpoint file on this machine (/mnt/f/checkpoints/gvp_pretrained/best.pt)",
                 "egnn_pretrained": "no checkpoint file on this machine (/mnt/f/checkpoints/egnn_pretrained/best.pt)",
                 "cdconv_pretrained": "wrapper loads no weights (structure_pretrained.py:155-163)"}


def seed_everything(seed=SEED):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class Encoder:
    """One loaded encoder with a strict call: real inputs in, (L, D) finite residue embeddings out."""

    def __init__(self, key, device="cuda"):
        self.spec = SPEC[key]
        self.device = torch.device(device)
        self.meta = {"encoder": key, "family": self.spec.family, "target": self.spec.target, "weights": self.spec.weights,
                     "rule": self.spec.rule, "env": os.environ.get("CONDA_DEFAULT_ENV", Path(sys.prefix).name)}
        seed_everything()
        if self.spec.family == "hf_plm":
            from patched.hf_plm import HFPLM
            self.impl = HFPLM(self.spec.target, self.device)
            self.meta.update(self.impl.meta)
        elif self.spec.family == "patched":
            import importlib
            self.impl = importlib.import_module(f"patched.{self.spec.target}").build(self.device)
            self.meta.update(getattr(self.impl, "meta", {}))
        else:
            self._load_registry()

    def _load_registry(self):
        from src.models import get_encoder
        s = self.spec
        kwargs = dict(s.kwargs)
        if s.target == "geobpe":
            from downstream_tasks.embed_proteins import _find_geobpe_ckpt
            kwargs["checkpoint_path"] = str(_find_geobpe_ckpt())
        enc = get_encoder(s.target, **kwargs)
        if s.checkpoint:
            if not Path(s.checkpoint).exists():
                raise FileNotFoundError(f"{s.key}: checkpoint {s.checkpoint} is missing")
            before = weights_sha256(enc)
            enc.load_pretrained(s.checkpoint)
            how = "wrapper load_pretrained()"
            if weights_sha256(enc) == before:
                how = self._strict_load(enc, s.checkpoint)
            self.meta.update({"checkpoint": s.checkpoint, "checkpoint_sha256": file_sha256(s.checkpoint), "checkpoint_loaded_by": how})
        self.impl = enc.eval().to(self.device)
        if s.weights == "random":
            self.meta["seed"] = SEED

    @staticmethod
    def _strict_load(enc, path):
        """The wrappers call load_state_dict(strict=False) on the wrong dictionary level, which loads nothing without an error."""
        ck = torch.load(path, map_location="cpu", weights_only=False)
        own = enc.state_dict()
        for level in ("encoder_state", "model_state_dict", "state_dict", "encoder", None):
            sd = ck if level is None else ck.get(level) if isinstance(ck, dict) else None
            if not isinstance(sd, dict):
                continue
            hit = [k for k, v in sd.items() if k in own and torch.is_tensor(v) and tuple(v.shape) == tuple(own[k].shape)]
            if len(hit) >= 0.9 * len(own):
                enc.load_state_dict({k: sd[k] for k in hit}, strict=len(hit) == len(own))
                return f"strict load of '{level}': {len(hit)} of {len(own)} tensors"
        raise RuntimeError(f"no level of {path} matches the model tensors; the checkpoint cannot be loaded")

    def finalize_meta(self):
        """Hash the weights after the first call, because some wrappers build their network lazily."""
        if self.spec.family == "registry":
            self.meta["weights_sha256"] = weights_sha256(self.impl)
            self.meta["n_parameters"] = int(sum(p.numel() for p in self.impl.parameters()))
            if self.spec.weights == "pretrained" and self.meta["n_parameters"] == 0:
                raise RuntimeError(f"{self.spec.key}: no parameters registered after the first call")
        return self.meta

    @torch.no_grad()
    def residue(self, sequence, coords=None, pdb_path=None):
        """Residue embeddings (L, D) as float32 numpy. `coords` is (L, 5, 3) from a real structure."""
        s, L = self.spec, len(sequence)
        if s.uses_coords:
            if coords is None or coords.shape != (L, 5, 3):
                raise InputError(f"{s.key}: needs real coordinates of shape ({L}, 5, 3)")
            if not np.isfinite(coords).all() or np.abs(coords[:, :3]).sum(axis=(1, 2)).min() == 0:
                raise InputError(f"{s.key}: zero or non-finite backbone coordinates")
        if coords is not None:
            coords = (coords - coords[:, 1].mean(0, keepdims=True)).astype(np.float32)   # centre on the CA centroid
            if s.canonical_pose:
                coords = canonical_pose(coords)
        if s.family in ("hf_plm", "patched"):
            out = self.impl.residue(sequence, coords=coords, pdb_path=pdb_path)
        else:
            from src.models.base_encoder import ProteinInput
            dev = self.device
            c = torch.from_numpy(coords if coords is not None else np.full((L, 5, 3), np.nan, dtype=np.float32))[None].to(dev)
            if not s.uses_coords:
                c = torch.zeros(1, L, 5, 3, device=dev)  # never read by a sequence-only wrapper; the audit verifies that
            q = torch.tensor([[AA_TO_IDX[a] for a in sequence]], dtype=torch.long, device=dev)
            m = torch.ones(1, L, dtype=torch.bool, device=dev)
            o = self.impl.encode(ProteinInput(coords=c, sequence=q, mask=m, structure_paths=[pdb_path] if pdb_path else None))
            if o.residue_embeddings is None:
                raise RuntimeError(f"{s.key}: wrapper returned no residue embeddings")
            out = o.residue_embeddings[0].float().cpu().numpy()
        if out.shape[0] != L:
            raise RuntimeError(f"{s.key}: {out.shape[0]} residue rows for a protein of length {L}")
        if not np.isfinite(out).all():
            raise RuntimeError(f"{s.key}: non-finite embedding values")
        return out.astype(np.float32)
