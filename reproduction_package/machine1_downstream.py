#!/usr/bin/env python3
"""Run encoder×task embedding jobs for a specified encoder subset (split across GPUs)."""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


BASE_DIR = Path(os.environ.get("PROTCOMPASS_ROOT") or Path(__file__).resolve().parents[1])  # repository root

ALL_ENCODERS = ["ankh", "dmasif_pretrained", "esm2", "foldvision", "gearnet_pretrained",
                "mace", "pottsmpnn_pretrained", "proteinbert_bfd", "saprot_pretrained"]

TASKS = [
    "secondary_structure", "contact_prediction", "remote_homology", "stability",
    "ec_classification", "go_mf", "go_bp", "ppi_site", "metal_binding",
    "subcellular_localization", "membrane_soluble", "mutation_effect",
    "fluorescence", "solubility", "ppi_affinity",
]


def flush_vram():
    """Kill any lingering embed_proteins.py processes and release GPU memory."""
    try:
        result = subprocess.run(["pgrep", "-f", "embed_proteins.py"],
                                capture_output=True, text=True)
        if result.returncode == 0:
            my_pid = os.getpid()
            for pid_str in result.stdout.strip().split('\n'):
                pid_str = pid_str.strip()
                if not pid_str or int(pid_str) == my_pid:
                    continue
                try:
                    os.kill(int(pid_str), signal.SIGTERM)
                    print(f"[VRAM] Killed lingering embed_proteins.py process {pid_str}")
                except ProcessLookupError:
                    pass
            time.sleep(2)
    except Exception as e:
        print(f"[VRAM] Warning: cleanup failed: {e}")

    subprocess.run(
        ["conda", "run", "-n", "protein_embeddings", "--no-capture-output",
         "python", "-c",
         "import torch; torch.cuda.empty_cache(); import gc; gc.collect()"],
        cwd=str(BASE_DIR), timeout=30, capture_output=True,
    )


def check_exists(encoder, task):
    out = BASE_DIR / "downstream_tasks" / "embeddings" / task / encoder
    for name in ("meta.json", "train_embeddings.npy", "test_embeddings.npy"):
        p = out / name
        if not (p.exists() and p.stat().st_size > 0):
            return False
    return True


def run_job(encoder, task, mutation_effect_subsample=None):
    if check_exists(encoder, task):
        print(f"[SKIP] {encoder} × {task} (already exists)")
        return "skipped"

    print(f"\n{'='*60}\n[START] {encoder} × {task}\n{'='*60}")

    cmd = ["conda", "run", "-n", "protein_embeddings", "--no-capture-output",
           "python", "downstream_tasks/embed_proteins.py",
           "--encoder", encoder, "--task", task]

    # Add subsample parameter for mutation_effect
    if task == "mutation_effect" and mutation_effect_subsample is not None:
        cmd.extend(["--mutation-effect-subsample", str(mutation_effect_subsample)])

    try:
        r = subprocess.run(cmd, cwd=str(BASE_DIR), timeout=7200)
        if r.returncode == 0:
            print(f"[SUCCESS] {encoder} × {task}")
            return "ok"
        print(f"[FAILED] {encoder} × {task} (code={r.returncode})")
        return "fail"
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {encoder} × {task} (>2h)")
        return "fail"
    except Exception as e:
        print(f"[ERROR] {encoder} × {task}: {e}")
        return "fail"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", required=True,
                    help="Encoder names to run (subset of the 9 encoders).")
    ap.add_argument("--tasks", nargs="+", default=None,
                    help="Task names to run (default: all tasks).")
    ap.add_argument("--mutation-effect-subsample", type=int, default=None,
                    help="Subsample mutation_effect to N sequences (e.g., 100000 for faster runs)")
    args = ap.parse_args()

    encoders = args.encoders
    bad = [e for e in encoders if e not in ALL_ENCODERS]
    if bad:
        sys.exit(f"Unknown encoders: {bad}")

    tasks = args.tasks if args.tasks else TASKS
    bad_tasks = [t for t in tasks if t not in TASKS]
    if bad_tasks:
        sys.exit(f"Unknown tasks: {bad_tasks}")

    t0 = time.time()
    completed = failed = skipped = 0
    total = len(encoders) * len(tasks)

    for task in tasks:
        for enc in encoders:
            status = run_job(enc, task, mutation_effect_subsample=args.mutation_effect_subsample)
            flush_vram()
            if status == "ok":
                completed += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
            done = completed + failed + skipped
            elapsed = (time.time() - t0) / 60
            print(f"\n[PROGRESS] {done}/{total} | ✓{completed} ✗{failed} ⊘{skipped} | {elapsed:.1f}min elapsed\n")

    print(f"\n{'='*60}")
    print(f"MACHINE DONE — encoders={encoders}")
    print(f"Completed: {completed} | Failed: {failed} | Skipped: {skipped}")
    print(f"Total time: {(time.time() - t0)/3600:.2f}h")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
