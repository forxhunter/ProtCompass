"""
Worker script for Starling encoder — runs inside the 'starling' conda env via subprocess.

Called by StarlingEncoder with a temp JSON of protein inputs.
Outputs per-protein embeddings as a JSON.

Input JSON format:
    {
        "proteins": [
            {"name": "xxx", "sequence": "ACDEF...", "ca_coords": [[x,y,z], ...]},
            ...
        ]
    }

Output JSON format:
    {
        "xxx": {"global_emb": [...], "emb_dim": N}
    }

Starling VAE approach:
    1. Compute Cα-Cα distance matrix from backbone coords
    2. Normalise distances (divide by MAX_DIST_NM, convert Å→nm)
    3. Encode with VAE.encode() → DiagonalGaussianDistribution
    4. Take .mode() → spatial latent (latent_dim, H', W')
    5. Global-average-pool → (latent_dim,) global embedding
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ── Locate and import starling ────────────────────────────────────────────────
_STARLING_ROOT = Path(__file__).parents[3] / "starling"
if str(_STARLING_ROOT) not in sys.path:
    sys.path.insert(0, str(_STARLING_ROOT))

try:
    from starling.inference.model_loading import ModelManager
except ImportError as e:
    print(f"ERROR: Cannot import starling: {e}", file=sys.stderr)
    sys.exit(1)

MAX_DIST_NM   = 5.0    # normalise pairwise distances to [0,1] at this value (nm)
MAX_SEQ_LEN   = 380    # matches STARLING MAX_SEQUENCE_LENGTH
ANGSTROM_TO_NM = 0.1


def compute_distance_matrix(ca_coords: np.ndarray) -> np.ndarray:
    """
    Compute Cα-Cα pairwise distance matrix (in nm).
    Crops to MAX_SEQ_LEN if necessary.
    Returns float32 array of shape (L, L) where L = min(len(ca_coords), MAX_SEQ_LEN).
    """
    coords = np.array(ca_coords, dtype=np.float32)[:MAX_SEQ_LEN] * ANGSTROM_TO_NM
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    return dist.astype(np.float32)


def encode_protein(vae, dist_map: np.ndarray, device: torch.device) -> np.ndarray:
    """
    Encode a distance map (L, L) with the Starling VAE.
    Returns global embedding of shape (latent_dim,).
    """
    d = torch.from_numpy(dist_map / MAX_DIST_NM).clamp(0.0, 1.0)
    d = d.unsqueeze(0).unsqueeze(0).to(device)          # (1, 1, L, L)

    with torch.no_grad():
        posterior = vae.encode(d)
        z = posterior.mode()                             # (1, C, H', W')
        # Global average pool over spatial dims
        emb = z.mean(dim=(-2, -1)).squeeze(0)            # (C,)
    return emb.cpu().float().numpy()


def main():
    if len(sys.argv) != 3:
        print("Usage: starling_worker.py <input_json> <output_json>", file=sys.stderr)
        sys.exit(1)

    input_path  = sys.argv[1]
    output_path = sys.argv[2]

    with open(input_path) as f:
        data = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Starling VAE on {device}...", flush=True)

    model_manager = ModelManager()
    try:
        vae, _ = model_manager.get_models(device=str(device))
    except Exception as e:
        print(f"ERROR loading Starling models: {e}", file=sys.stderr)
        sys.exit(1)

    vae.eval()

    results = {}
    proteins = data["proteins"]
    print(f"Encoding {len(proteins)} proteins...", flush=True)

    for i, prot in enumerate(proteins):
        name    = prot["name"]
        ca_coords = prot.get("ca_coords")

        if ca_coords is None or len(ca_coords) < 2:
            print(f"  [{i+1}/{len(proteins)}] {name}: skipped (no coords)", flush=True)
            continue

        try:
            dist_map = compute_distance_matrix(ca_coords)
            emb = encode_protein(vae, dist_map, device)
            results[name] = {
                "global_emb": emb.tolist(),
                "emb_dim": int(emb.shape[0]),
            }
            print(f"  [{i+1}/{len(proteins)}] {name}: OK, dim={emb.shape[0]}", flush=True)
        except Exception as e:
            print(f"  [{i+1}/{len(proteins)}] {name}: ERROR — {e}", file=sys.stderr, flush=True)

    with open(output_path, "w") as f:
        json.dump(results, f)

    print(f"Saved embeddings for {len(results)} proteins to {output_path}", flush=True)


if __name__ == "__main__":
    main()
