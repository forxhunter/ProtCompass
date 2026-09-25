#!/usr/bin/env python
"""
ESM3 worker script — runs in `esm3` conda env (Python 3.10, esm package).

Receives protein data via JSON, runs ESM3 encoding, saves result to .pt file.
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ESM3 local directory
_ESM3_LOCAL_DIR = Path((PROTCOMPASS_ROOT + "/protcompass/esm3-sm-open-v1"))
_BACKBONE_ATOM37_IDX = [0, 1, 2, 3, 4]


def _patch_data_root(local_dir: Path) -> None:
    """Patch esm.utils.constants.esm3.data_root to return local_dir."""
    import esm.utils.constants.esm3 as C

    def _local_data_root(model: str) -> Path:
        return local_dir

    C.data_root = _local_data_root


def _build_esm3_local(local_dir: Path, device: torch.device):
    """Instantiate ESM3-sm-open and load weights from local directory."""
    weights_path = local_dir / "data" / "weights" / "esm3_sm_open_v1.pth"
    if not weights_path.exists():
        raise FileNotFoundError(f"ESM3 weights not found at {weights_path}")

    _patch_data_root(local_dir)

    from esm.models.esm3 import ESM3
    from esm.pretrained import (
        ESM3_structure_encoder_v0,
        ESM3_structure_decoder_v0,
        ESM3_function_decoder_v0,
    )
    from esm.tokenization import get_esm3_model_tokenizers as get_model_tokenizers

    with torch.device(device):
        model = ESM3(
            d_model=1536,
            n_heads=24,
            v_heads=256,
            n_layers=48,
            structure_encoder_fn=ESM3_structure_encoder_v0,
            structure_decoder_fn=ESM3_structure_decoder_v0,
            function_decoder_fn=ESM3_function_decoder_v0,
            tokenizers=get_model_tokenizers("esm3_sm_open_v1"),
        ).eval()

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    if device.type != "cpu":
        model = model.to(torch.bfloat16)

    return model, get_model_tokenizers("esm3_sm_open_v1")


def coords_to_atom37(coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Convert (B, L, 5, 3) backbone coords to (B, L, 37, 3) atom37 format."""
    B, L = coords.shape[:2]
    atom37 = torch.full((B, L, 37, 3), float("nan"), dtype=coords.dtype, device=coords.device)
    for i, idx in enumerate(_BACKBONE_ATOM37_IDX):
        atom37[:, :, idx, :] = coords[:, :, i, :]
    pad = ~mask
    atom37[pad] = float("nan")
    return atom37


AA_VOCAB_INV = {
    0: 'L', 1: 'A', 2: 'G', 3: 'V', 4: 'E', 5: 'S', 6: 'I', 7: 'Y', 8: 'C', 9: 'H',
    10: 'N', 11: 'M', 12: 'F', 13: 'P', 14: 'D', 15: 'K', 16: 'R', 17: 'W', 18: 'Q', 19: 'T',
    20: 'X', 21: 'U', 22: 'O', 23: 'B', 24: 'Z'
}


def tokenize_sequence(sequences: torch.Tensor, mask: torch.Tensor, tokenizers):
    """Convert amino acid indices to ESM3 sequence tokens."""
    seq_tok = tokenizers.sequence
    batch_tokens = []
    for b in range(sequences.shape[0]):
        seq_len = int(mask[b].sum().item())
        aa_str = "".join(
            AA_VOCAB_INV.get(int(sequences[b, i].item()), "X")
            for i in range(seq_len)
        )
        ids = seq_tok.encode(aa_str)
        batch_tokens.append(torch.tensor(ids, dtype=torch.long))

    max_len = max(t.shape[0] for t in batch_tokens)
    pad_id = seq_tok.pad_token_id if hasattr(seq_tok, 'pad_token_id') and seq_tok.pad_token_id is not None else 1
    device = sequences.device
    padded = torch.full((len(batch_tokens), max_len), pad_id, dtype=torch.long, device=device)
    for b, t in enumerate(batch_tokens):
        padded[b, : t.shape[0]] = t.to(device)
    return padded


def encode_batch(model, coords, sequence, mask, tokenizers, use_structure=True):
    """Encode a batch of proteins with ESM3."""
    from esm.sdk.api import ESMProteinTensor, LogitsConfig

    device = coords.device
    B, L = sequence.shape

    # Sequence tokens
    seq_tokens = tokenize_sequence(sequence, mask, tokenizers)

    # Structure coordinates
    structure_coords = None
    if use_structure:
        structure_coords = coords_to_atom37(coords, mask).float()

    cfg = LogitsConfig(return_embeddings=True)

    all_embeddings = []
    for b in range(B):
        seq_len = int(mask[b].sum().item())
        tok_len = seq_len + 2
        s_tokens = seq_tokens[b, :tok_len]

        if use_structure:
            res_coords = structure_coords[b, :seq_len]
            nan_row = torch.full((1, 37, 3), float("nan"), dtype=res_coords.dtype, device=res_coords.device)
            s_coords = torch.cat([nan_row, res_coords, nan_row], dim=0)
        else:
            s_coords = None

        pt = ESMProteinTensor(sequence=s_tokens, coordinates=s_coords)

        logits_out = model.logits(pt, cfg)
        emb = logits_out.embeddings  # (1, tok_len, 1536)

        # Strip BOS/EOS
        residue_emb = emb[0, 1 : 1 + seq_len, :].float()

        # Pad to L
        full_emb = torch.zeros(L, 1536, device=device)
        full_emb[:seq_len] = residue_emb
        all_embeddings.append(full_emb)

    residue_embeddings = torch.stack(all_embeddings, dim=0)  # (B, L, D)

    # Global: mean-pool
    mask_f = mask.unsqueeze(-1).float()
    global_emb = (residue_embeddings * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)

    return residue_embeddings, global_emb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSON input file path")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    # Input coords shape: (B, L, 5, 3) - N, CA, C, O, CB
    coords = torch.tensor(data["coords"], dtype=torch.float32)  # (B, L, 5, 3)
    sequence = torch.tensor(data["sequence"], dtype=torch.long)
    mask = torch.tensor(data["mask"], dtype=torch.bool)
    output_path = data.get("output_path", "output.pt")
    use_structure = data.get("use_structure", True)

    # Load ESM3 model
    model, tokenizers = _build_esm3_local(_ESM3_LOCAL_DIR, torch.device("cpu"))
    model.eval()

    # Encode
    with torch.no_grad():
        embeddings, global_emb = encode_batch(model, coords, sequence, mask, tokenizers, use_structure)

    result = {
        "embeddings": embeddings.numpy(),
        "global_embedding": global_emb.numpy(),
    }
    torch.save(result, output_path)
    print(f"Saved embeddings to {output_path}")


if __name__ == "__main__":
    main()
