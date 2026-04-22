"""
ProteinAE worker — runs inside the `proteinae` conda env.

Called by proteinae_encoder.py via subprocess.
Reads a .pt file with a batch of coords/masks/sequences,
runs ProteinAE encoder, saves per-residue latent embeddings.

Usage:
    conda run -n proteinae python proteinae_worker.py \
        --input  /tmp/batch.pt \
        --output /tmp/embeddings.pt \
        --ckpt   /path/to/ae_r1_d8_v1.ckpt \
        --root   /mnt/f/2026_structuerembed/external_repos/ProteinAE_v1
"""

import argparse
import sys
from pathlib import Path

import torch
from einops import rearrange


def build_batch_nn(coords, mask, seq_idx, device):
    """
    Build the batch_nn dict expected by ProteinAE encoder from raw tensors.

    Args:
        coords:  (B, L, 5, 3) float — [N, CA, C, O, CB] in Angstrom (h5 format)
        mask:    (B, L) bool
        seq_idx: (B, L) long — amino acid index 0-19
        device:  torch.device

    Returns:
        batch_nn dict ready for encoder.forward()
    """
    B, L = mask.shape
    ANG_TO_NM = 0.1  # 1 Å = 0.1 nm

    # Backbone atoms: N=0, CA=1, C=2, O=3  (skip CB=4)
    BB_IDX = [0, 1, 2, 3]
    bb_coords = coords[:, :, BB_IDX, :].float()  # (B, L, 4, 3) in Å

    # Convert to nm, flatten backbone: (B, L*4, 3)
    x_1_bb = bb_coords * ANG_TO_NM                          # Å → nm
    x_1    = rearrange(x_1_bb, "b n c d -> b (n c) d")     # (B, 4L, 3)

    # coords_mask: per-atom validity based on residue mask, (B, 4L)
    atom_mask = mask.unsqueeze(-1).expand(-1, -1, 4)        # (B, L, 4)
    coords_mask = rearrange(atom_mask, "b n c -> b (n c)")  # (B, 4L)

    # residue_pdb_idx: simple 1-indexed positions
    res_idx = torch.arange(1, L + 1, device=device).unsqueeze(0).expand(B, -1).long()

    # chain_breaks_per_residue: all zeros (single chain assumption)
    chain_breaks = torch.zeros(B, L, device=device)

    return {
        "x_1":                      x_1.to(device),
        "mask":                     mask.bool().to(device),  # Must be bool for ~ operator
        "coords_mask":              coords_mask.bool().to(device),  # Must be bool
        "residue_type":             seq_idx.long().to(device),
        "residue_pdb_idx":          res_idx,
        "chain_breaks_per_residue": chain_breaks,
        "strict_feats":             False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="Path to input .pt batch file")
    parser.add_argument("--output", required=True, help="Path to output .pt embeddings file")
    parser.add_argument("--ckpt",   required=True, help="Path to ae_r1_d8_v1.ckpt")
    parser.add_argument("--root",   required=True, help="Path to ProteinAE_v1 root")
    args = parser.parse_args()

    sys.path.insert(0, args.root)

    from omegaconf import OmegaConf
    from proteinfoundation.proteinflow.proteinae import ProteinAE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[proteinae_worker] device={device}")

    # ── Load checkpoint ────────────────────────────────────────────────────────
    print(f"[proteinae_worker] Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg  = OmegaConf.create(ckpt["hyper_parameters"]["cfg_exp"])

    model = ProteinAE(cfg_exp=cfg)
    model.load_state_dict(ckpt["state_dict"])
    model = model.eval().to(device)
    encoder = model.encoder

    print("[proteinae_worker] Model loaded.")

    # ── Load input batch ───────────────────────────────────────────────────────
    data = torch.load(args.input, map_location="cpu", weights_only=False)
    coords  = data["coords"]   # (N_total, L, 5, 3)
    masks   = data["masks"]    # (N_total, L)
    seqs    = data["seqs"]     # (N_total, L)

    BATCH_SIZE = 8
    N = coords.shape[0]
    all_embeddings = []

    for start in range(0, N, BATCH_SIZE):
        end  = min(start + BATCH_SIZE, N)
        c_b  = coords[start:end]
        m_b  = masks[start:end].bool()
        s_b  = seqs[start:end]

        batch_nn = build_batch_nn(c_b, m_b, s_b, device)

        with torch.no_grad():
            out = encoder(batch_nn)

        single_repr = out.get("single_repr")  # (B, L, dim_latent=8)
        if single_repr is None:
            raise RuntimeError("Encoder did not return 'single_repr'")

        all_embeddings.append(single_repr.cpu())
        print(f"  [{end}/{N}] shape={single_repr.shape}")

    embeddings = torch.cat(all_embeddings, dim=0)  # (N_total, L, 8)
    torch.save({"embeddings": embeddings}, args.output)
    print(f"[proteinae_worker] Saved {embeddings.shape} → {args.output}")


if __name__ == "__main__":
    main()
