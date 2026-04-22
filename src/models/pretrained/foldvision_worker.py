"""
FoldVision worker — runs inside the `foldvision` conda env.

Called by foldvision_encoder.py via subprocess.
Reads a JSON list of PDB paths, runs preprocess + encode, saves results.

Usage:
    conda run -n foldvision python foldvision_worker.py \
        --input /tmp/paths.json \
        --output /tmp/embeddings.pt \
        --model_id AlexanderKroll/foldvision-encoder \
        --foldvision_root /mnt/f/2026_structuerembed/FoldVision
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_id", default="AlexanderKroll/foldvision-encoder")
    parser.add_argument("--foldvision_root", required=True)
    args = parser.parse_args()

    sys.path.insert(0, args.foldvision_root)
    from foldvision import FoldVisionEncoder
    from foldvision.preprocess.pdb_to_points import (
        preprocess_Structures,
        DEFAULT_BOUNDING_BOXES,
        DEFAULT_DIC,
    )

    pdb_paths = json.load(open(args.input))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load pretrained model
    print(f"[foldvision_worker] Loading {args.model_id}")
    encoder = FoldVisionEncoder.from_pretrained(args.model_id).to(device).eval()

    results = {}
    for pdb_path in pdb_paths:
        pdb_path = Path(pdb_path)
        if not pdb_path.exists():
            print(f"  SKIP: {pdb_path} not found")
            continue

        try:
            # Preprocess PDB → dict of NPZ arrays
            with tempfile.TemporaryDirectory() as tmp_dir:
                preprocess_Structures(
                    save_dir=tmp_dir,
                    PDB_dir=str(pdb_path.parent),
                    continue_run=False,
                )
                # Load the NPZ for this protein
                npz_name = pdb_path.stem + ".npz"
                npz_path = Path(tmp_dir) / npz_name
                if not npz_path.exists():
                    # Try lower/upper variants
                    candidates = list(Path(tmp_dir).glob("*.npz"))
                    if not candidates:
                        print(f"  SKIP: no NPZ produced for {pdb_path.name}")
                        continue
                    npz_path = candidates[0]

                data = np.load(npz_path, allow_pickle=True)
                # FoldVision expects a dense 3D voxel tensor
                # The NPZ has sparse point list — convert using preprocess logic
                # Shape: pick the bounding box that fits
                bb = None
                for box in DEFAULT_BOUNDING_BOXES:
                    key = f"{box[0]}_{box[1]}_{box[2]}"
                    if key in data:
                        bb = data[key]
                        break
                if bb is None:
                    # Fallback: use any array
                    bb = data[list(data.keys())[0]]

                x = torch.tensor(bb, dtype=torch.float32).unsqueeze(0).to(device)
                # x: (1, C, H, W, D)
                with torch.no_grad():
                    emb = encoder.encode(x)  # (1, 1024)
                results[str(pdb_path)] = emb.squeeze(0).cpu().numpy()
                print(f"  OK: {pdb_path.name} → {emb.shape}")

        except Exception as e:
            print(f"  ERROR {pdb_path.name}: {e}")
            continue

    torch.save(results, args.output)
    print(f"[foldvision_worker] Saved {len(results)} embeddings → {args.output}")


if __name__ == "__main__":
    main()
