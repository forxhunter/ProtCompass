"""Official dMaSIF (FreyrS/dMaSIF, search checkpoint), restored with strict=True, on the protonated all-atom structure.

Reason for the patch: the project wrapper (src/models/class_b/dmasif_encoder.py) moved the model to the CPU, the official
code builds CUDA tensors (geometry_processing.py:163, model.py:400), and the wrapper caught the resulting error and stored
a zero vector for every protein. The official code also calls torch.solve, which PyTorch removed after version 1.9.
Here the official model runs on the GPU, torch.solve is mapped to torch.linalg.solve, and every failure raises.

Input, following the official preprocessing (data_preprocessing/download_pdb.py and convert_pdb2npy.py): hydrogens are
removed and added again with `reduce -Trim` and `reduce -HIS`, and every atom of the chain is passed with its element as
one of six types (C, H, O, N, S, Se). No residue labels are passed; residue identity enters through side-chain atoms. An atom of any other element (for
example the arsenic of a modified cysteine) has no input type in the model and is omitted; the number of omitted atoms
is recorded in meta.json.
The surface is sampled from random points and thinned on an axis-aligned grid, so the output depends on the random seed
and on the pose. The seed is fixed for every protein, and every structure is first rotated onto the principal axes of its
CA trace with a fixed sign convention, which makes the embedding a function of the structure alone.

Output per residue: mean of the 16-d surface descriptors (`embedding_1`) of the surface points whose nearest atom belongs
to the residue; a residue that owns no surface point takes the descriptor of the surface point nearest to its CA.

Optional per-protein cache: with DMASIF_CACHE set to a directory, the residue embedding of each structure is stored under
the SHA-1 of the seed and the parsed atom records (names, residues, elements, coordinates) and read back on a later call
with the same records. An interrupted extraction then resumes without recomputing finished proteins.
`residue_from_rows` takes the parsed atom records directly, so a second machine can fill the same cache from a
worklist of records (remote_dmasif/) without access to the structure files.
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import types
from argparse import Namespace
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = os.environ.get("PROTCOMPASS_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
# DMASIF_REPO is a clone of https://github.com/FreyrS/dMaSIF (commit 0dcc26c) and DMASIF_REDUCE the reduce binary.
REPO = os.environ.get("DMASIF_REPO", PROTCOMPASS_ROOT + "/protcompass/external_repos/dMaSIF")
CKPT = REPO + "/models/dMaSIF_search_3layer_12A_16dim"
REDUCE = os.environ.get("DMASIF_REDUCE", "reduce")
SEED = 20260919
ELEMENTS = {"C": 0, "H": 1, "O": 2, "N": 3, "S": 4, "SE": 5}      # convert_pdb2npy.py: ele2num
# Arguments of the released model dMaSIF_search_3layer_12A_16dim (benchmark_scripts/dMaSIF_search.sh and Arguments.py defaults).
ARGS = Namespace(atom_dims=6, emb_dims=16, in_channels=16, orientation_units=16, post_units=16, n_layers=3, radius=12.0,
                 search=True, site=False, use_mesh=False, embedding_layer="dMaSIF", curvature_scales=[1.0, 2.0, 3.0, 5.0, 10.0],
                 resolution=1.0, distance=1.05, variance=0.1, sup_sampling=20, dropout=0.0, no_chem=False, no_geom=False, k=40)


def read_chain(path, omitted=None):
    """ATOM records of the first chain (alternate location blank or A), hydrogens dropped: residue index, atom name, element, xyz.

    Atoms whose element is not one of the six dMaSIF types are omitted and counted in `omitted` (element -> count)."""
    rows, order, chain0 = [], {}, None
    for line in open(path):
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM"):
            continue
        chain0 = line[21] if chain0 is None else chain0
        if line[21] != chain0:
            break
        if line[16] not in (" ", "A"):
            continue
        name = line[12:16].strip()
        el = (line[76:78].strip() or name.lstrip("0123456789")[0]).upper()
        if el in ("H", "D"):
            continue
        res = order.setdefault(line[22:27], len(order))
        if el not in ELEMENTS:
            if omitted is not None:
                omitted[el] = omitted.get(el, 0) + 1
            continue
        rows.append((res, name, line[17:20], el, float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return rows, len(order)


def principal_axis_pose(xyz, ca):
    centre = ca.mean(0)
    _, _, vt = np.linalg.svd(ca - centre, full_matrices=False)
    sign = np.sign((((ca - centre) @ vt.T) ** 3).sum(0)); sign[sign == 0] = 1.0
    rot = vt * sign[:, None]
    if np.linalg.det(rot) < 0:
        rot[2] *= -1.0
    return (xyz - centre) @ rot.T


def protonate(rows, xyz, work):
    """Write the posed heavy atoms with residues renumbered from 1, run reduce as the official pipeline does, and read all atoms back."""
    src, trim, out = work / "in.pdb", work / "trim.pdb", work / "h.pdb"
    with open(src, "w") as fh:
        for k, ((res, name, resname, el, *_), (x, y, z)) in enumerate(zip(rows, xyz), 1):
            field = f" {name:<3s}" if len(name) < 4 and len(el) == 1 else f"{name:<4s}"
            fh.write(f"ATOM  {k % 100000:5d} {field} {resname} A{res + 1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {el:>2s}\n")
        fh.write("END\n")
    for args, dst in ((["-Trim", str(src)], trim), (["-HIS", str(trim)], out)):
        with open(dst, "w") as fh:
            subprocess.run([REDUCE, "-Quiet", *args], stdout=fh, stderr=subprocess.DEVNULL, check=False)   # reduce exits with 1 on success
    res, typ, pos, ca = [], [], [], {}
    for line in open(out):
        if not line.startswith("ATOM"):
            continue
        name = line[12:16].strip()
        el = (line[76:78].strip() or name.lstrip("0123456789")[0]).upper()
        if el not in ELEMENTS:
            raise RuntimeError(f"element {el} is not one of the six dMaSIF atom types")
        r = int(line[22:26]) - 1
        p = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        res.append(r); typ.append(ELEMENTS[el]); pos.append(p)
        if name == "CA":
            ca[r] = p
    if sum(t != ELEMENTS["H"] for t in typ) != len(rows):
        raise RuntimeError(f"reduce returned {sum(t != 1 for t in typ)} heavy atoms for {len(rows)} written")
    if ELEMENTS["H"] not in typ:
        raise RuntimeError("reduce added no hydrogens")
    return np.array(res), np.array(typ), np.array(pos, dtype=np.float32), ca


class DMaSIFOfficial:
    def __init__(self, device, pretrained=True):
        if torch.device(device).type != "cuda":
            raise RuntimeError("the official dMaSIF code needs a CUDA device")
        sys.path.insert(0, REPO)
        # torch_cluster and these torch_geometric classes are imported for the DGCNN and PointNet++ baselines only.
        import torch_geometric.nn as tgnn                     # before the stub: torch_geometric probes torch_cluster on import
        try:
            import torch_cluster  # noqa: F401
        except ImportError:
            sys.modules["torch_cluster"] = types.SimpleNamespace(knn=None)
        for name in ("PointConv", "XConv", "Reshape"):
            if not hasattr(tgnn, name):
                setattr(tgnn, name, type(name, (nn.Module,), {}))
        # torch.solve(B, A) -> (solution, LU) is called once (geometry_processing.py:544); current PyTorch keeps the name
        # only to raise an error, so it is replaced unconditionally.
        sol = namedtuple("solve", ["solution", "LU"])
        torch.solve = lambda B, A: sol(torch.linalg.solve(A, B), None)
        from model import dMaSIF
        self.model = dMaSIF(ARGS)
        loaded = "none (random initialization)"
        if pretrained:
            ck = torch.load(CKPT, map_location="cpu", weights_only=False)
            self.model.load_state_dict(ck["model_state_dict"], strict=True)
            loaded = f"strict load of 'model_state_dict': {len(ck['model_state_dict'])} tensors"
        self.model = self.model.to(device).eval()
        self.device = device
        commit = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        self.meta = {"checkpoint": CKPT, "repository": "https://github.com/FreyrS/dMaSIF", "repository_commit": commit,
                     "n_parameters": int(sum(p.numel() for p in self.model.parameters())),
                     "checkpoint_loaded_by": loaded,
                     "inputs_passed": "all atoms of the structure file with hydrogens added by reduce (elements and coordinates); "
                                      "residue identity enters through side-chain atoms",
                     "patch": "patched/dmasif_official.py", "seed": SEED,
                     "atoms_of_other_elements_omitted": {},      # filled during extraction: element -> number of atoms
                     "pose": "principal-axis pose of the CA trace with fixed axis signs, applied before hydrogens are added"}

    @torch.no_grad()
    def residue(self, sequence, coords=None, pdb_path=None):
        if not pdb_path:
            raise RuntimeError("dMaSIF needs a structure file")
        rows, n_res = read_chain(pdb_path, self.meta["atoms_of_other_elements_omitted"])
        if n_res != len(sequence):
            raise RuntimeError(f"{pdb_path}: {n_res} residues in the file for a sequence of length {len(sequence)}")
        xyz = np.array([r[4:] for r in rows], dtype=np.float64)
        ca_rows = [i for i, r in enumerate(rows) if r[1] == "CA"]
        if len(ca_rows) != n_res:
            raise RuntimeError(f"{pdb_path}: {len(ca_rows)} CA atoms for {n_res} residues")
        return self.residue_from_rows(rows, n_res)

    @torch.no_grad()
    def residue_from_rows(self, rows, n_res):
        """Residue embedding from the atom records of `read_chain`; the caller has checked residues and CA atoms."""
        xyz = np.array([r[4:] for r in rows], dtype=np.float64)
        ca_rows = [i for i, r in enumerate(rows) if r[1] == "CA"]
        cache_file = None
        if os.environ.get("DMASIF_CACHE"):
            key = hashlib.sha1(repr((SEED, rows)).encode()).hexdigest()
            cache_file = Path(os.environ["DMASIF_CACHE"]) / key[:2] / f"{key}.npz"
            if cache_file.exists():
                with np.load(cache_file) as z:
                    self.last = json.loads(str(z["last"]))
                    return z["emb"]
        work = Path(tempfile.mkdtemp(prefix="dmasif_", dir=os.environ.get("DMASIF_TMP")))
        try:
            res, typ, pos, ca = protonate(rows, principal_axis_pose(xyz, xyz[ca_rows]), work)
        finally:
            for f in work.iterdir():
                f.unlink()
            work.rmdir()
        dev = self.device
        atoms = torch.from_numpy(pos).to(dev)
        types_ = torch.zeros(len(typ), 6, device=dev)
        types_[torch.arange(len(typ)), torch.from_numpy(typ).to(dev)] = 1.0
        P = {"atoms": atoms, "atom_xyz": atoms, "atomtypes": types_, "triangles": None,
             "batch_atoms": torch.zeros(len(typ), dtype=torch.long, device=dev)}
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        self.model.embed(P)
        emb, surf = P["embedding_1"].float(), P["xyz"].float()
        owner = torch.from_numpy(res).to(dev)[torch.cdist(surf, atoms).argmin(1)]          # residue of the nearest atom
        out = torch.zeros(n_res, emb.shape[1], device=dev).index_add_(0, owner, emb)
        count = torch.zeros(n_res, device=dev).index_add_(0, owner, torch.ones(len(owner), device=dev))
        out = out / count.clamp(min=1)[:, None]
        empty = (count == 0).nonzero(as_tuple=True)[0]
        if len(empty):
            ca_xyz = torch.tensor([ca[int(r)] for r in empty], dtype=torch.float32, device=dev)
            out[empty] = emb[torch.cdist(ca_xyz, surf).argmin(1)]
        self.last = {"n_atoms": len(typ), "n_surface_points": int(len(surf)), "residues_without_surface_point": int(len(empty))}
        result = out.cpu().numpy()
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_file.with_suffix(f".{os.getpid()}.tmp")
            with open(tmp, "wb") as fh:
                np.savez(fh, emb=result, last=json.dumps(self.last))
            os.replace(tmp, cache_file)
        return result


def build(device):
    return DMaSIFOfficial(device)
