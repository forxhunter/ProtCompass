"""
Download all benchmark datasets for the downstream task evaluation.

Sources:
  TAPE  (T1-T3, T7-T8): s3.amazonaws.com/songlabdata  (Rao et al. NeurIPS 2019)
  ProteinWorkshop (T4-T6, T9-T10): github.com/a-r-j/ProteinWorkshop  (Jamasb et al. ICLR 2024)
  DeepLoc (T11-T12): github.com/TviNet/DeepLoc-2.0  (Thumuluri et al. NAR 2022)
  ProteinGym DMS (T13): github.com/Oatz00/ProteinGym  (notna85, v1.2 2024)
  SKEMPI v2 (T14): zora.uvt.nl  (Jankovic et al. 2021)
  FLIP (T15): github.com/J-SNACKSTER/FLIP  (Scaffold-agnostic stability)

Usage:
  conda run -n GeoBPE-6 python downstream_tasks/download_data.py
  python downstream_tasks/download_data.py --tasks tape          # only TAPE
  python downstream_tasks/download_data.py --tasks workshop      # only ProteinWorkshop
  python downstream_tasks/download_data.py --tasks deeploc       # only DeepLoc
  python downstream_tasks/download_data.py --tasks proteingym    # only ProteinGym DMS
  python downstream_tasks/download_data.py --tasks skempi       # only SKEMPI v2
  python downstream_tasks/download_data.py --tasks flip         # only FLIP
"""

import argparse
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT     = Path(__file__).parent.parent
DATA_DIR = Path(__file__).parent / "data"

TAPE_BASE = "https://s3.amazonaws.com/songlabdata/proteindata/data_pytorch"
TAPE_FILES = {
    "secondary_structure": "secondary_structure.tar.gz",   # T1
    "proteinnet":          "proteinnet.tar.gz",            # T2
    "remote_homology":     "remote_homology.tar.gz",       # T3
    "fluorescence":        "fluorescence.tar.gz",          # T7
    "stability":           "stability.tar.gz",             # T8
}

DEEPLOC_URL = "https://raw.githubusercontent.com/TviNet/DeepLoc-2.0/main/data"
DEEPLOC_FILES = {
    "train.csv": f"{DEEPLOC_URL}/train.csv",
    "test.csv":  f"{DEEPLOC_URL}/test.csv",
}

# ── New tasks ─────────────────────────────────────────────────────────────────
PROTEINGYM_URL = "https://raw.githubusercontent.com/Oatz00/ProteinGym/dev/ProteinGym"
PROTEINGYM_FILES = {
    "DMS_scores.csv": f"{PROTEINGYM_URL}/DMS_scores.csv",
}

SKEMPI_URL = "https://github.com/cmzb/BindingDB_Affinity/raw/main"
SKEMPI_FILES = {
    "skempi_v2.csv": f"{SKEMPI_URL}/skempi_v2.csv",
}

FLIP_URL = "https://raw.githubusercontent.com/J-SNACKSTER/FLIP/main"
FLIP_FILES = {
    "scaffold_transformations.csv": f"{FLIP_URL}/data/scaffold_transformations.csv",
    "secondary_structure.csv":      f"{FLIP_URL}/data/secondary_structure.csv",
    "stability.csv":                f"{FLIP_URL}/data/stability.csv",
    "solubility.csv":               f"{FLIP_URL}/data/solubility.csv",
}


def progress_hook(count, block_size, total_size):
    pct = min(count * block_size * 100 // total_size, 100)
    mb  = count * block_size / 1e6
    sys.stdout.write(f"\r  {mb:.1f} MB / {total_size/1e6:.1f} MB ({pct}%)")
    sys.stdout.flush()


def download_file(url: str, dest: Path, desc: str = ""):
    if dest.exists():
        print(f"  [skip] {dest.name} already exists")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {desc or dest.name} ...")
    try:
        urllib.request.urlretrieve(url, dest, reporthook=progress_hook)
        print()
        return True
    except Exception as e:
        print(f"\n  FAILED: {e}")
        if dest.exists():
            dest.unlink()
        return False


def extract_tar(tar_path: Path, dest_dir: Path):
    if not tar_path.exists():
        return False
    print(f"  Extracting {tar_path.name} ...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(dest_dir)
    return True


def download_tape():
    tape_dir = DATA_DIR / "tape"
    tape_dir.mkdir(parents=True, exist_ok=True)
    print("\n=== TAPE datasets ===")
    for name, fname in TAPE_FILES.items():
        out_tar  = tape_dir / fname
        out_data = tape_dir / name
        if out_data.exists():
            print(f"  [skip] {name}/ already extracted")
            continue
        url = f"{TAPE_BASE}/{fname}"
        ok  = download_file(url, out_tar, name)
        if ok:
            extract_tar(out_tar, tape_dir)
            out_tar.unlink()   # remove tar after extraction
            print(f"  [ok] {name}/")


def download_deeploc():
    dl_dir = DATA_DIR / "deeploc"
    dl_dir.mkdir(parents=True, exist_ok=True)
    print("\n=== DeepLoc 2.0 ===")
    ok_count = 0
    for fname, url in DEEPLOC_FILES.items():
        dest = dl_dir / fname
        ok   = download_file(url, dest, fname)
        if ok:
            ok_count += 1
    if ok_count == len(DEEPLOC_FILES):
        print("  [ok] DeepLoc 2.0 data ready")
    else:
        print(f"  WARNING: only {ok_count}/{len(DEEPLOC_FILES)} DeepLoc files downloaded")


def download_proteinworkshop():
    """
    ProteinWorkshop data is downloaded via its own CLI tool.
    We call it programmatically here using the GeoBPE-6 env.
    Tasks: EC, GO (BP+MF), PPI-site, metal-binding.
    """
    try:
        import proteinworkshop
        from proteinworkshop import download_pdb_mmtf
    except ImportError:
        print("\n=== ProteinWorkshop ===")
        print("  NOT AVAILABLE in current env. Run in GeoBPE-6:")
        print("  conda run -n GeoBPE-6 python downstream_tasks/download_data.py --tasks workshop")
        return

    pw_dir = DATA_DIR / "proteinworkshop"
    pw_dir.mkdir(parents=True, exist_ok=True)
    print("\n=== ProteinWorkshop datasets ===")

    # ProteinWorkshop uses its own download system via hydra configs
    # We trigger downloads by instantiating datasets
    try:
        from proteinworkshop.datasets.ec_reaction import ECReactionDataModule
        from proteinworkshop.datasets.go_reaction import GeneOntologyDataModule
        from proteinworkshop.datasets.masif_site import MasifSiteDataModule
        from proteinworkshop.datasets.metal_3d import Metal3DDataModule

        for name, cls, kwargs in [
            ("ec_reaction",      ECReactionDataModule,    {"path": str(pw_dir / "ec")}),
            ("go_mf",            GeneOntologyDataModule,  {"path": str(pw_dir / "go"), "go_label": "mf"}),
            ("go_bp",            GeneOntologyDataModule,  {"path": str(pw_dir / "go"), "go_label": "bp"}),
            ("masif_site",       MasifSiteDataModule,     {"path": str(pw_dir / "masif_site")}),
            ("metal_3d",         Metal3DDataModule,       {"path": str(pw_dir / "metal_3d")}),
        ]:
            try:
                dm = cls(**kwargs)
                dm.prepare_data()
                print(f"  [ok] {name}")
            except Exception as e:
                print(f"  [warn] {name}: {e}")
    except ImportError as e:
        print(f"  Import error: {e}")
        print("  ProteinWorkshop dataset modules not accessible this way.")
        print("  Will use proteinworkshop CLI to download.")
        _download_workshop_via_cli(pw_dir)


def _download_workshop_via_cli(pw_dir: Path):
    """Fallback: use proteinworkshop's download CLI."""
    import subprocess
    tasks = ["ec", "fold", "go-mf", "go-bp", "ppi_site", "metal_3d"]
    for task in tasks:
        result = subprocess.run(
            ["python", "-m", "proteinworkshop.scripts.download_pdb_mmtf",
             "--task", task, "--out_dir", str(pw_dir)],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"  [ok] {task}")
        else:
            print(f"  [warn] {task}: {result.stderr[:120]}")


def download_proteingym():
    """ProteinGym DMS: deep mutational scan benchmarks (natural variant effects)."""
    pg_dir = DATA_DIR / "proteingym"
    pg_dir.mkdir(parents=True, exist_ok=True)
    print("\n=== ProteinGym DMS ===")
    # The main index file lists all DMS assays
    idx_url = "https://raw.githubusercontent.com/Oatz00/ProteinGym/dev/ProteinGym/DMS_scores.csv"
    ok = download_file(idx_url, pg_dir / "DMS_scores.csv", "DMS index")
    if ok:
        print("  [ok] ProteinGym DMS index downloaded")
        print("  (Individual assay files downloaded on first use by HuggingFace datasets)")


def download_skempi():
    """SKEMPI v2: binding affinity changes for missense mutations."""
    sk_dir = DATA_DIR / "skempi"
    sk_dir.mkdir(parents=True, exist_ok=True)
    print("\n=== SKEMPI v2 ===")
    ok = download_file(
        "https://github.com/cmzb/BindingDB_Affinity/raw/refs/heads/main/skempi_v2.csv",
        sk_dir / "skempi_v2.csv",
        "SKEMPI v2"
    )
    if ok:
        print("  [ok] SKEMPI v2 downloaded")


def download_flip():
    """FLIP: scaffold-agnostic stability and solubility benchmarks."""
    flip_dir = DATA_DIR / "flip"
    flip_dir.mkdir(parents=True, exist_ok=True)
    print("\n=== FLIP benchmarks ===")
    base = "https://raw.githubusercontent.com/J-SNACKSTER/FLIP/main/data"
    ok_count = 0
    for fname in ["scaffold_transformations.csv", "stability.csv", "solubility.csv"]:
        url = f"{base}/{fname}"
        dest = flip_dir / fname
        if download_file(url, dest, fname):
            ok_count += 1
    print(f"  {ok_count}/{len(['scaffold_transformations.csv','stability.csv','solubility.csv'])} FLIP files downloaded")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+",
                        choices=["tape", "workshop", "deeploc", "proteingym",
                                 "skempi", "flip", "all"],
                        default=["all"])
    args = parser.parse_args()

    tasks = set(args.tasks)
    if "all" in tasks:
        tasks = {"tape", "workshop", "deeploc", "proteingym", "skempi", "flip"}

    if "tape"       in tasks: download_tape()
    if "workshop"   in tasks: download_proteinworkshop()
    if "deeploc"    in tasks: download_deeploc()
    if "proteingym" in tasks: download_proteingym()
    if "skempi"     in tasks: download_skempi()
    if "flip"       in tasks: download_flip()

    print("\n=== Download complete. Data in:", DATA_DIR, "===")


if __name__ == "__main__":
    main()
