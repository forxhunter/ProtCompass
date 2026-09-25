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
    )
    from foldvision.dataloader import get_input_matrix

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
            # Create a temp dir with only this PDB file to avoid processing the entire directory
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Copy PDB to temp dir to avoid scanning entire directory
                import shutil
                pdb_copy = Path(tmp_dir) / pdb_path.name
                shutil.copy2(pdb_path, pdb_copy)

                preprocess_Structures(
                    save_dir=tmp_dir,
                    PDB_dir=tmp_dir,
                    continue_run=False,
                )
                # Load the NPZ for this protein
                # preprocess_Structures outputs to numpy_3D_point_lists/ subdirectory
                npz_path = Path(tmp_dir) / "numpy_3D_point_lists" / f"{pdb_path.stem}.npz"
                if not npz_path.exists():
                    # Try lower/upper variants
                    candidates = list(Path(tmp_dir) / "numpy_3D_point_lists").glob("*.npz")
                    if not candidates:
                        print(f"  SKIP: no NPZ produced for {pdb_path.name}")
                        continue
                    npz_path = candidates[0]

                # Get the bounding box size for this protein from bounding_boxes.npy
                boxes_data = np.load(Path(tmp_dir) / "bounding_boxes.npy", allow_pickle=True).item()
                box_key = npz_path.name  # e.g., "9v4l.npz"
                box_str = boxes_data.get(box_key, "[64, 64, 64]")
                # Parse "[64 64 64]" -> [64, 64, 64]
                box_size = [int(x) for x in box_str.strip("[]").split()]

                # Convert sparse point list to dense 3D voxel tensor using FoldVision's own function
                # get_input_matrix returns (5, H, W, D) with augment=False for inference
                x = get_input_matrix(str(npz_path), box_size, augment=False)
                x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
                # x: (1, 5, H, W, D)
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
