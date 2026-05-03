#!/usr/bin/env python
"""
Run invariant probing battery on a pretrained encoder.

Usage:
    python scripts/run_probing.py --encoder dmasif_pretrained --seed 0
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import get_encoder
from src.data.protein_dataset import ProteinStructureDataset, collate_proteins
from src.probes.probe_battery import ProbeBattery, AVAILABLE_PROBES


def main():
    parser = argparse.ArgumentParser(description="Run invariant probing battery")
    parser.add_argument("--encoder", required=True, help="Encoder name (e.g., dmasif_pretrained)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--data-path", default="data/processed/cath_with_labels_updated.h5", help="Path to CATH dataset")
    parser.add_argument("--output-dir", default="outputs/probing_results", help="Output directory")
    parser.add_argument("--max-train-samples", type=int, default=1000, help="Max training samples")
    parser.add_argument("--max-val-samples", type=int, default=200, help="Max validation samples")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=10, help="Training epochs per probe")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fold-level", default="topology", choices=["architecture", "topology"], help="CATH hierarchy level for I7 task")
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    print(f"=== Invariant Probing: {args.encoder} (seed={args.seed}) ===")
    print(f"Device: {args.device}")
    print(f"Data: {args.data_path}")
    print()

    # Load encoder
    print(f"Loading encoder: {args.encoder}")
    encoder = get_encoder(args.encoder)

    # Try to move to device, but skip if model uses device_map or is already placed
    try:
        # Check if model has device_map (from accelerate)
        if not hasattr(encoder, 'hf_device_map'):
            encoder.to(args.device)
    except (NotImplementedError, RuntimeError) as e:
        print(f"  Warning: Could not move encoder to {args.device}: {e}")
        print(f"  Model may already be on correct device or use device_map")

    encoder.eval()
    print(f"  Embedding dim: {encoder.embedding_dim}")
    print()

    # Load datasets
    print("Loading datasets...")
    train_dataset = ProteinStructureDataset(
        args.data_path,
        split="train",
        max_length=500,
        fold_level=args.fold_level,
    )
    val_dataset = ProteinStructureDataset(
        args.data_path,
        split="val",
        max_length=500,
        fold_level=args.fold_level,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_proteins,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_proteins,
        num_workers=0,
    )

    print(f"  Train: {len(train_dataset)} proteins")
    print(f"  Val: {len(val_dataset)} proteins")
    print()

    # Create probe battery (use AVAILABLE_PROBES - only families with labels in dataset)
    # Flatten dict of probe families into a list of configs
    probe_configs = []
    for family_probes in AVAILABLE_PROBES.values():
        probe_configs.extend(family_probes)

    battery = ProbeBattery(
        encoder=encoder,
        probe_configs=probe_configs,
        device=args.device,
    )

    # Run probing
    print("Running probe battery...")
    fingerprint = battery.run(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        num_epochs=args.num_epochs,
        batch_size=256,
        lr=1e-3,
        verbose=True,
    )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{args.encoder}_seed_{args.seed}.pt"
    torch.save({
        "encoder": args.encoder,
        "seed": args.seed,
        "fingerprint": fingerprint,
    }, output_file)

    print(f"\n✓ Results saved to {output_file}")


if __name__ == "__main__":
    main()
