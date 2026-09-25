"""E5: re-extract the equivariant encoders with an invariant representation that keeps vector directions.

The released extraction (extract.py with patched/gvp_inproc.py and patched/gcpnet_official.py) stores each equivariant
vector channel as its norm, and the MACE wrapper drops its vector channels. Here every vector channel v_c(i) of the
encoder's final node state (a 3-vector in the global frame) is projected onto the backbone frame F_i of residue i:

    x = (C - CA) / |C - CA|,  z = x cross (N - CA) normalised,  y = z cross x,  F_i = rows (x, y, z)

(the convention of protcompass/scripts/prereg_floors.py `frame`). The 3 numbers F_i @ v_c(i) are invariant to a global
rotation and translation and keep the direction of the channel relative to the residue's backbone. The new residue
representation is [scalar channels (s), F_i @ v_1, ..., F_i @ v_v] with width s + 3v, channel-major (v1x, v1y, v1z, v2x, ...).

Nothing in src/, patched/, common.py or extract.py is edited. The released model is built exactly as extract.py builds it
(common.Encoder, same seed, same checkpoint), and the step where the norm is taken is replaced inside this script:
  gvp                               GVPInProcess.residue is swapped for a method that projects instead of taking the norm
  gcpnet_random, gcpnet_pretrained  GCPNetOfficial.residue is swapped; GCPNet's vectors live on atoms, so the residue's
                                    vector block is the CA atom's 16 vectors projected on F_i (primary output); the mean over
                                    the residue's heavy atoms of F_i @ chi_a is written alongside as residue_atommean_fp16.npy
  mace                              a forward hook on the last MACE block captures the 16 vector channels, which the
                                    wrapper discards; they are projected and appended to the wrapper's 128 output channels

Usage, in the env and with the PYTHONPATH of the released run (run_queue.sh):
  PYTHONPATH=protcompass/pylib  protein_embeddings/bin/python extract_vector_frame.py --encoder gvp
  PYTHONPATH=protcompass/pylib  protein_embeddings/bin/python extract_vector_frame.py --encoder mace
  (PYTHONPATH unset)             gcpnet/bin/python             extract_vector_frame.py --encoder gcpnet_random
  (PYTHONPATH unset)             gcpnet/bin/python             extract_vector_frame.py --encoder gcpnet_pretrained

Output: protcompass/probing_embeddings_vframe/<encoder>_vframe/{residue_fp16.npy, pooled_fp32.npy, meta.json}, rows as in
the released directories. The run also compares, on every protein, the new scalar columns and the norms of the new vector
block with the released embedding, and runs a rigid-motion test on the 32 proteins of the released input audit.
"""
import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import ALPHABET, ROOT, Encoder, weights_sha256

RELEASED = ROOT / "." / "probing_embeddings"
OUT_ROOT = ROOT / "." / "probing_embeddings_vframe"
FRAME_TEXT = ("F_i rows (x, y, z): x = (C - CA)/|C - CA|, z = x cross (N - CA) normalised, y = z cross x "
              "(protcompass/scripts/prereg_floors.py frame); vector block = F_i @ v_c(i) for c = 1..v, channel-major "
              "(v1x, v1y, v1z, v2x, ...)")
# s = scalar columns, v = vector channels of the released embedding layout; released_norm_cols = released columns that hold norms
LAYOUT = {"gvp": dict(s=100, v=16, released_norm_cols=True),
          "gcpnet_random": dict(s=100, v=16, released_norm_cols=True),
          "gcpnet_pretrained": dict(s=100, v=16, released_norm_cols=True),
          "mace": dict(s=128, v=16, released_norm_cols=False)}


# ----------------------------------------------------------------------------------------------------------- geometry
def backbone_frames(n, ca, c):
    """(L, 3, 3) float64 frames with rows x, y, z from (L, 3) N, CA, C positions."""
    n, ca, c = (t.double() for t in (n, ca, c))
    x = c - ca
    x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
    z = torch.cross(x, n - ca, dim=-1)
    z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
    y = torch.cross(z, x, dim=-1)
    return torch.stack([x, y, z], dim=-2)


def project(F, v):
    """F (L, 3, 3), v (L, C, 3) -> (L, C, 3) components of each channel on the frame axes, float64."""
    return torch.einsum("lij,lcj->lci", F, v.double())


def random_rigid(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q)), rng.normal(size=3) * 10.0      # proper rotation, ~10 A translation


# ---------------------------------------------------------------------------------------------------------------- hooks
def hook_gvp(enc, lay):
    """Replace patched/gvp_inproc.GVPInProcess.residue: same forward, projection instead of norm."""
    from common import AA_TO_IDX
    S, V = lay["s"], lay["v"]

    @torch.no_grad()
    def residue(self, sequence, coords=None, pdb_path=None):
        L = len(sequence)
        c = torch.from_numpy(coords)[None].to(self.device)
        q = torch.tensor([[AA_TO_IDX[a] for a in sequence]], device=self.device)
        m = torch.ones(1, L, dtype=torch.bool, device=self.device)
        emb, _ = self.model(c, q, m)
        emb = emb[0].float()
        s, v = emb[:, :S], emb[:, S:].reshape(L, V, 3)
        F = backbone_frames(c[0, :, 0], c[0, :, 1], c[0, :, 2])
        pv = project(F, v)
        self._stash = {"released_style": torch.cat([s, v.norm(dim=-1)], dim=-1).cpu().numpy(),   # the released rule, same pass
                       "vnorm_direct": v.norm(dim=-1).double().cpu().numpy()}
        return torch.cat([s, pv.float().reshape(L, 3 * V)], dim=-1).cpu().numpy()

    enc.impl.residue = types.MethodType(residue, enc.impl)
    return lambda: enc.impl._stash


def hook_mace(enc, lay):
    """Capture (s, v) of the last MACE block with a forward hook and append F_i @ v to the wrapper output."""
    from src.models.base_encoder import EncoderOutput
    S, V = lay["s"], lay["v"]
    cap, stash = {}, {}
    enc.impl.blocks[-1].register_forward_hook(lambda mod, inp, out: cap.__setitem__("v", out[1]))
    original = enc.impl.encode

    @torch.no_grad()
    def encode(protein_input):
        cap.clear()
        out = original(protein_input)
        h = out.residue_embeddings[0].float()                      # (L, 128), the released output
        v = cap["v"][0]                                            # (L, 16, 3), final vector channels
        L = h.shape[0]
        assert v.shape == (L, V, 3), v.shape
        xyz = protein_input.coords[0]
        F = backbone_frames(xyz[:, 0], xyz[:, 1], xyz[:, 2])
        pv = project(F, v)
        stash["released_style"] = h.cpu().numpy()
        stash["vnorm_direct"] = v.norm(dim=-1).double().cpu().numpy()
        new = torch.cat([h, pv.float().reshape(L, 3 * V)], dim=-1)
        return EncoderOutput(residue_embeddings=new[None], global_embedding=new.mean(0, keepdim=True))

    enc.impl.encode = encode
    return lambda: stash


def atom_table(path):
    """patched/gcpnet_official.atom_frame with the atom name kept (same line filter, same order)."""
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
        rows.append((el, float(line[30:38]), float(line[38:46]), float(line[46:54]), line[22:27], line[12:16].strip()))
    df = pd.DataFrame(rows, columns=["element", "x", "y", "z", "res", "name"])
    return df[df.element != "H"].reset_index(drop=True)


def hook_gcpnet(enc, lay):
    """Replace patched/gcpnet_official.GCPNetOfficial.residue: same forward on the all-atom file, chi projected per residue."""
    from patched.gcpnet_official import atom_frame
    from torch_geometric.data import Batch
    S, V = lay["s"], lay["v"]
    impl = enc.impl
    impl._rigid = None                                             # (R, t) applied to the atoms, for the invariance test only

    @torch.no_grad()
    def residue(self, sequence, coords=None, pdb_path=None):
        if not pdb_path:
            raise RuntimeError("GCPNet needs the all-atom structure file")
        df = atom_frame(pdb_path)                                  # the exact frame the released run passed to the model
        tab = atom_table(pdb_path)
        assert len(tab) == len(df) and (tab.res.values == df.res.values).all() and np.array_equal(tab[["x", "y", "z"]].values, df[["x", "y", "z"]].values)
        if self._rigid is not None:
            R, t = self._rigid
            df = df.copy()
            df[["x", "y", "z"]] = df[["x", "y", "z"]].values.astype(np.float64) @ R.T + t
        batch = Batch.from_data_list([self.transform(df)]).to(self.device)
        batch, _ = self.model(batch)
        h, chi = batch.h.float(), batch.chi.float()                # (A, 100), (A, 16, 3)
        A = len(df)
        assert h.shape == (A, S) and chi.shape == (A, V, 3), (h.shape, chi.shape)
        res = pd.factorize(df.res)[0]
        R_ = res.max() + 1
        # backbone atoms of each residue, from the coordinates the model saw
        pos = torch.tensor(df[["x", "y", "z"]].values.astype(np.float64), device=self.device)
        idx = {}
        for nm in ("N", "CA", "C"):
            sel = np.full(R_, -1)
            rows = np.where(tab.name.values == nm)[0]
            for r_ in rows[::-1]:                                  # first occurrence wins
                sel[res[r_]] = r_
            if (sel < 0).any():
                raise RuntimeError(f"{pdb_path}: residue without {nm}")
            idx[nm] = torch.as_tensor(sel, device=self.device)
        F = backbone_frames(pos[idx["N"]], pos[idx["CA"]], pos[idx["C"]])          # (R, 3, 3)
        res_t = torch.as_tensor(res, device=self.device)
        pv_atom = project(F[res_t], chi)                                          # (A, 16, 3): each atom's vectors on its residue frame
        pv_ca = pv_atom[idx["CA"]]                                                # (R, 16, 3)
        feat = torch.cat([h, chi.norm(dim=-1), pv_atom.float().reshape(A, 3 * V), pv_atom.norm(dim=-1).float()], dim=-1).cpu().numpy()
        pooled = np.stack([feat[res == r].mean(0) for r in range(R_)])            # the released per-residue mean, extra columns appended
        s_mean = pooled[:, :S]
        self._stash = {"released_style": pooled[:, :S + V],                         # [mean h, mean |chi|] as released
                       "atom_norm_mean": pooled[:, S + V + 3 * V:],                 # mean over atoms of |F chi_a|
                       "atommean": np.concatenate([s_mean, pooled[:, S + V:S + V + 3 * V]], axis=1),
                       "vnorm_direct": chi[idx["CA"]].norm(dim=-1).double().cpu().numpy()}   # |chi_CA| before projection
        return np.concatenate([s_mean, pv_ca.float().reshape(R_, 3 * V).cpu().numpy()], axis=1)

    impl.residue = types.MethodType(residue, impl)
    return lambda: impl._stash


HOOKS = {"gvp": hook_gvp, "mace": hook_mace, "gcpnet_random": hook_gcpnet, "gcpnet_pretrained": hook_gcpnet}


# --------------------------------------------------------------------------------------------------------------- driver
def with_retry(fn, *a, **k):
    """The GPU is shared: on out-of-memory, free the cache, wait and retry instead of failing."""
    for attempt in range(40):
        try:
            return fn(*a, **k)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            torch.cuda.empty_cache()
            print(f"  OOM (attempt {attempt + 1}), waiting 60 s", flush=True)
            time.sleep(60)
    raise RuntimeError("out of memory after 40 attempts")


def audit_indices(data, n_pairs=16, seed=0):
    """The 32 proteins of the released input audit (input_audit.audit_proteins('probing')), as row indices."""
    rng = np.random.default_rng(seed)
    lengths = pd.Series(data.length)
    pool = [list(idx[:2]) for L, idx in lengths[(lengths >= 60) & (lengths <= 250)].groupby(lengths).groups.items() if len(idx) >= 2]
    picks = [pool[i] for i in rng.choice(len(pool), n_pairs, replace=False)]
    return [int(k) for pair in picks for k in pair]


def invariance(key, enc, get_stash, data, seqs):
    rng = np.random.default_rng(1)
    rel = {"new": [], "released_rule": [], "atommean": []}
    for i in audit_indices(data):
        p = data.protein(i)
        xyz = p["coords"].astype(np.float32)
        R, t = random_rigid(rng)
        out = {}
        for tag in ("base", "moved"):
            x2 = xyz if tag == "base" else (xyz.astype(np.float64) @ R.T + t).astype(np.float32)
            if key.startswith("gcpnet"):
                enc.impl._rigid = None if tag == "base" else (R, t)
            h = with_retry(enc.residue, seqs[i], coords=x2, pdb_path=p["pdb_path"])
            st = get_stash()
            out[tag] = {"new": h.astype(np.float64).mean(0), "released_rule": st["released_style"].astype(np.float64).mean(0)}
            if "atommean" in st:
                out[tag]["atommean"] = st["atommean"].astype(np.float64).mean(0)
        if key.startswith("gcpnet"):
            enc.impl._rigid = None
        for k in out["base"]:
            b, m = out["base"][k], out["moved"][k]
            rel[k].append(float(np.linalg.norm(m - b) / (np.linalg.norm(b) + 1e-12)))
    return {f"median_rel_change_{k}": float(np.median(v)) for k, v in rel.items() if v} | \
           {f"max_rel_change_{k}": float(np.max(v)) for k, v in rel.items() if v} | {"n_proteins": len(rel["new"]),
            "motion": "random proper rotation (QR of a Gaussian matrix) and a Gaussian translation of scale 10 A, "
                      "applied to the input coordinates (GVP, MACE) or to every atom of the structure (GCPNet)",
            "proteins": "the 32 proteins of the released input audit (input_audit.audit_proteins('probing'))"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=sorted(HOOKS))
    ap.add_argument("--limit", type=int, default=0, help="debugging: first N proteins only, nothing final is written")
    a = ap.parse_args()
    key, lay = a.encoder, LAYOUT[a.encoder]
    S, V = lay["s"], lay["v"]
    t_start = time.time()
    sys.path.insert(0, str(ROOT / "." / "probing_data"))
    from probing_dataset import ProbingData
    from extract import health
    data = ProbingData()
    lengths, offsets = data.length.astype(np.int64), data.offset.astype(np.int64)
    n_res = int(lengths.sum())
    rel_dir = RELEASED / key
    rel_meta = json.loads((rel_dir / "meta.json").read_text())
    released = np.load(rel_dir / "residue_fp16.npy", mmap_mode="r")
    assert released.shape == (n_res, S + (V if lay["released_norm_cols"] else 0)), released.shape

    OUT_ROOT.mkdir(exist_ok=True)
    for name, arr in (("offsets.npy", offsets), ("lengths.npy", lengths)):
        if (OUT_ROOT / name).exists():
            assert np.array_equal(np.load(OUT_ROOT / name), arr)
        else:
            np.save(OUT_ROOT / name, arr)
    if not (OUT_ROOT / "ids.txt").exists():
        (OUT_ROOT / "ids.txt").write_text("\n".join(data.ids) + "\n")
    out_dir = OUT_ROOT / f"{key}_vframe"
    if (out_dir / "meta.json").exists() and not a.limit:
        print(f"[{key}_vframe] meta.json exists; left untouched"); return
    out_dir.mkdir(exist_ok=True)

    enc = Encoder(key)                                             # the released build: seed, checkpoint, strict load
    get_stash = HOOKS[key](enc, lay)
    aa = data.h["residue/aa"][:]
    seqs = {i: "".join(ALPHABET[x] for x in aa[offsets[i]:offsets[i] + lengths[i]]) for i in range(len(data))}

    # weights must be the released weights
    wh = weights_sha256(enc.impl.model) if key in ("gvp", "gcpnet_random", "gcpnet_pretrained") else weights_sha256(enc.impl)
    wh_released = rel_meta.get("weights_sha256")
    print(f"[{key}] weights {wh}, released {wh_released}", flush=True)
    if wh_released and wh != wh_released:
        raise SystemExit(f"[{key}] weights differ from the released run ({wh} vs {wh_released})")

    t0 = time.time()
    inv = invariance(key, enc, get_stash, data, seqs)
    print(f"[{key}] invariance: {json.dumps({k: v for k, v in inv.items() if 'rel' in k})} ({time.time() - t0:.0f}s)", flush=True)

    D = S + 3 * V
    n_prot = a.limit or len(data)
    tag = ".partial"
    emb = np.lib.format.open_memmap(out_dir / f"residue_fp16.npy{tag}", mode="w+", dtype=np.float16, shape=(n_res, D))
    pooled = np.zeros((len(data), D), dtype=np.float32)
    extra = key.startswith("gcpnet")
    if extra:
        emb2 = np.lib.format.open_memmap(out_dir / f"residue_atommean_fp16.npy{tag}", mode="w+", dtype=np.float16, shape=(n_res, D))
        pooled2 = np.zeros((len(data), D), dtype=np.float32)
    # verification accumulators (all proteins)
    acc = {"a_scalar_maxabs_fp16_vs_released": 0.0, "a_scalar_n_mismatch": 0, "a_scalar_max_diff_in_fp16_ulps": 0.0, "a_released_rule_maxabs_vs_released": 0.0,
           "b_norm_maxabs_fp32_vs_released": 0.0, "b_norm_maxabs_fp16stored_vs_released": 0.0, "b_norm_maxrel_fp32_vs_released": 0.0, "b_norm_max_diff_fp16stored_in_fp16_ulps": 0.0,
           "b_projection_norm_maxabs_vs_unprojected": 0.0, "n_values_scalar": 0, "n_values_norm": 0}
    t1 = time.time()
    for i in range(n_prot):
        p = data.protein(i)
        L = int(lengths[i])
        h = with_retry(enc.residue, seqs[i], coords=p["coords"].astype(np.float32), pdb_path=p["pdb_path"])
        st = get_stash()
        assert h.shape == (L, D) and np.isfinite(h).all() and np.abs(h).max() < 6.5e4, f"{data.ids[i]}: bad output {h.shape}"
        h16 = h.astype(np.float16)
        sl = slice(int(offsets[i]), int(offsets[i]) + L)
        emb[sl] = h16
        pooled[i] = h.mean(0)
        rel = np.asarray(released[sl], dtype=np.float32)
        # (a) scalar columns of the new stored embedding against the released stored embedding
        d = np.abs(h16[:, :S].astype(np.float32) - rel[:, :S])
        acc["a_scalar_maxabs_fp16_vs_released"] = max(acc["a_scalar_maxabs_fp16_vs_released"], float(d.max()))
        acc["a_scalar_n_mismatch"] += int((d > 0).sum()); acc["n_values_scalar"] += d.size
        ulp = np.spacing(np.abs(rel[:, :S]).astype(np.float16)).astype(np.float32)
        acc["a_scalar_max_diff_in_fp16_ulps"] = max(acc["a_scalar_max_diff_in_fp16_ulps"], float((d / ulp).max()))
        acc["a_released_rule_maxabs_vs_released"] = max(acc["a_released_rule_maxabs_vs_released"],
                                                         float(np.abs(st["released_style"].astype(np.float16).astype(np.float32) - rel).max()))
        # (b) norms of the new vector block
        pv = h[:, S:].astype(np.float64).reshape(L, V, 3)
        nrm = np.linalg.norm(pv, axis=-1)
        acc["b_projection_norm_maxabs_vs_unprojected"] = max(acc["b_projection_norm_maxabs_vs_unprojected"], float(np.abs(nrm - st["vnorm_direct"]).max()))
        if lay["released_norm_cols"]:
            rel_n = rel[:, S:].astype(np.float64)
            # GVP: the vector block's own norms; GCPNet: the released column is a mean over atoms of |chi_a|, so the
            # comparable quantity is the mean over atoms of |F_i chi_a| from the same hook
            cmp = nrm if key == "gvp" else st["atom_norm_mean"].astype(np.float64)
            cmp16 = np.linalg.norm(h16[:, S:].astype(np.float64).reshape(L, V, 3), axis=-1) if key == "gvp" else cmp.astype(np.float16).astype(np.float64)
            acc["b_norm_maxabs_fp32_vs_released"] = max(acc["b_norm_maxabs_fp32_vs_released"], float(np.abs(cmp - rel_n).max()))
            acc["b_norm_maxabs_fp16stored_vs_released"] = max(acc["b_norm_maxabs_fp16stored_vs_released"], float(np.abs(cmp16 - rel_n).max()))
            acc["b_norm_maxrel_fp32_vs_released"] = max(acc["b_norm_maxrel_fp32_vs_released"], float((np.abs(cmp - rel_n) / (np.abs(rel_n) + 1e-2)).max()))
            acc["n_values_norm"] += rel_n.size
            ulp_n = np.spacing(np.abs(rel[:, S:]).astype(np.float16)).astype(np.float64)
            acc["b_norm_max_diff_fp16stored_in_fp16_ulps"] = max(acc["b_norm_max_diff_fp16stored_in_fp16_ulps"], float((np.abs(cmp16 - rel_n) / ulp_n).max()))
        if extra:
            h2 = st["atommean"]
            emb2[sl] = h2.astype(np.float16); pooled2[i] = h2.mean(0)
        if (i + 1) % 1000 == 0:
            el = time.time() - t1
            print(f"  [{key}] {i + 1}/{n_prot} domains, {el:.0f}s, eta {el / (i + 1) * (n_prot - i - 1):.0f}s | "
                  f"a {acc['a_scalar_maxabs_fp16_vs_released']:.2e} b {acc['b_norm_maxabs_fp32_vs_released']:.2e}", flush=True)
    emb.flush()
    if extra:
        emb2.flush()
    print(f"[{key}] verification: {json.dumps(acc)}", flush=True)
    if a.limit:
        print(f"[{key}] --limit run finished; partial files kept, no meta.json"); return

    (out_dir / "residue_fp16.npy.partial").rename(out_dir / "residue_fp16.npy")
    np.save(out_dir / "pooled_fp32.npy", pooled)
    # pooled check against the released pooled file (scalar columns)
    rel_pooled = np.load(rel_dir / "pooled_fp32.npy")
    acc["a_pooled_scalar_maxabs_vs_released"] = float(np.abs(pooled[:, :S] - rel_pooled[:, :S]).max())
    if lay["released_norm_cols"] and key == "gvp":
        # mean over residues of the per-residue norms of the new block, against the released pooled norm columns
        emb_r = np.load(out_dir / "residue_fp16.npy", mmap_mode="r")
        acc["b_pooled_norm_note"] = "released pooled norm columns are the residue mean of norms; checked per residue above"
        del emb_r
    meta = {"encoder": f"{key}_vframe", "source_encoder": key, "source_dir": str(rel_dir.relative_to(ROOT)),
            "representation": "[scalar channels (s), frame-projected vector channels (3v)]", "s": S, "v": V, "dim": D,
            "frame": FRAME_TEXT,
            "vector_source": {"gvp": "node vector channels (16 x 3) of the last GVPConvLayer, the vector half of GVPModel's output",
                              "mace": "vector channels v (16 x 3) of the last MACEBlock output, captured by a forward hook; the "
                                      "released wrapper output (output_proj of the scalars, 128) does not contain them in any form",
                              "gcpnet_random": "chi (16 x 3 per atom) after the last GCP interaction layer; residue vector block = "
                                               "the CA atom's chi projected on F_i. residue_atommean_fp16.npy holds the mean over the "
                                               "residue's heavy atoms of F_i @ chi_a instead",
                              "gcpnet_pretrained": "chi (16 x 3 per atom) after the last GCP interaction layer; residue vector block = "
                                                   "the CA atom's chi projected on F_i. residue_atommean_fp16.npy holds the mean over the "
                                                   "residue's heavy atoms of F_i @ chi_a instead"}[key],
            "scalar_source": ("mean over the residue's heavy atoms of h, as released" if key.startswith("gcpnet")
                              else "the released scalar columns, recomputed in the same forward pass"),
            "released_rule": rel_meta.get("rule"), "weights": rel_meta.get("weights"), "seed": rel_meta.get("seed"),
            "checkpoint": rel_meta.get("checkpoint"), "checkpoint_sha256": rel_meta.get("checkpoint_sha256"),
            "weights_sha256": wh, "weights_sha256_released": wh_released,
            "frame_coordinates": ("N, CA, C atoms of the structure file as passed to GCPNet" if key.startswith("gcpnet")
                                  else "N, CA, C of the coordinates passed to the model (labels.h5 coords centred on the CA centroid, as released)"),
            "env": Path(sys.prefix).name, "script": "protcompass/extract/extract_vector_frame.py",
            "dataset": "protcompass/probing_data/labels.h5", "date": time.strftime("%Y-%m-%d %H:%M"),
            "seconds_extraction": round(time.time() - t1, 1), "seconds_total": round(time.time() - t_start, 1),
            "n_domains": int(len(data)), "row_convention": "row r of residue_fp16.npy is row r of residue/* in labels.h5; "
                                                           "pooled_fp32.npy row k is protein k (mean over its residues)",
            "health": health(pooled), "verification_vs_released": acc, "rigid_motion_invariance": inv}
    if extra:
        (out_dir / "residue_atommean_fp16.npy.partial").rename(out_dir / "residue_atommean_fp16.npy")
        np.save(out_dir / "pooled_atommean_fp32.npy", pooled2)
        meta["extra_files"] = {"residue_atommean_fp16.npy": "[mean_a h_a (100), mean_a F_i @ chi_a (48)] per residue, a = heavy atoms of residue i",
                               "pooled_atommean_fp32.npy": "mean over residues of residue_atommean_fp16.npy (from fp32)"}
        meta["health_atommean"] = health(pooled2)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(f"[{key}_vframe] done: {len(data)} domains, dim {D}, {meta['seconds_total']}s", flush=True)


if __name__ == "__main__":
    main()
