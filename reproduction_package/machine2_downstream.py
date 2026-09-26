#!/usr/bin/env python3
"""Machine 2: 95 jobs across 9 encoders."""
import subprocess
import sys
import time
from pathlib import Path

# repository root; set PROTCOMPASS_ROOT to use another copy
ROOT = __import__("os").environ.get("PROTCOMPASS_ROOT") or str(Path(__file__).resolve().parents[1])

ENCODERS = ["cdconv", "egnn", "esm2_150m", "gcpnet", "gvp",
            "pottsmpnn", "pottsmpnn_soluble_pretrained", "proteinmpnn_pretrained", "schnet"]

TASKS = [
    "secondary_structure", "contact_prediction", "remote_homology", "stability",
    "ec_classification", "go_mf", "go_bp", "ppi_site", "metal_binding",
    "subcellular_localization", "membrane_soluble", "mutation_effect",
    "fluorescence", "solubility", "ppi_affinity"
]

SEQUENCE_ONLY = {"fluorescence", "solubility", "ppi_affinity"}
SEQUENCE_ENCODERS = {"esm2_150m"}

def check_exists(encoder, task):
    """Check if meta.json exists."""
    path = Path(f"{ROOT}/downstream_tasks/embeddings/{task}/{encoder}/meta.json")
    return path.exists()

def run_job(encoder, task):
    """Run single encoder×task job."""
    if check_exists(encoder, task):
        print(f"[SKIP] {encoder} × {task} (already exists)")
        return True

    print(f"\n{'='*60}")
    print(f"[START] {encoder} × {task}")
    print(f"{'='*60}")

    cmd = [
        "conda", "run", "-n", "protein_embeddings", "--no-capture-output",
        "python", "downstream_tasks/embed_proteins.py",
        "--encoder", encoder,
        "--task", task
    ]

    try:
        result = subprocess.run(cmd, cwd=ROOT, timeout=7200)
        if result.returncode == 0:
            print(f"[SUCCESS] {encoder} × {task}")
            return True
        else:
            print(f"[FAILED] {encoder} × {task} (code={result.returncode})")
            return False
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {encoder} × {task} (>2h)")
        return False
    except Exception as e:
        print(f"[ERROR] {encoder} × {task}: {e}")
        return False

def main():
    start_time = time.time()
    completed = 0
    failed = 0
    skipped = 0

    for task in TASKS:
        # Skip sequence-only tasks for structure encoders
        if task in SEQUENCE_ONLY:
            valid_encoders = [e for e in ENCODERS if e in SEQUENCE_ENCODERS]
        else:
            valid_encoders = ENCODERS

        for encoder in valid_encoders:
            if check_exists(encoder, task):
                skipped += 1
                continue

            success = run_job(encoder, task)
            if success:
                completed += 1
            else:
                failed += 1

            # Progress report
            total_done = completed + failed + skipped
            elapsed = (time.time() - start_time) / 60
            print(f"\n[PROGRESS] {total_done}/95 | ✓{completed} ✗{failed} ⊘{skipped} | {elapsed:.1f}min elapsed\n")

    print(f"\n{'='*60}")
    print(f"MACHINE 2 COMPLETE")
    print(f"Completed: {completed} | Failed: {failed} | Skipped: {skipped}")
    print(f"Total time: {(time.time() - start_time)/3600:.2f}h")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
