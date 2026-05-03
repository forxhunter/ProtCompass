"""
Download PDB files for ProteinWorkshop tasks (EC, GO, masif_site, Metal3D)
from RCSB to structure_encoder_data/pdb_cache/.

Usage: python download_pw_pdbs.py [--workers 50] [--out-dir ...]
"""
import argparse, sys, os, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request, urllib.error

sys.path.insert(0, str(Path(__file__).parents[1]))

ROOT    = Path(__file__).parents[1]
OUT_DIR = ROOT / "structure_encoder_data" / "pdb_cache"

TASK_FILES = [
    ROOT / "downstream_tasks/data/proteinworkshop/ECReaction/training.txt",
    ROOT / "downstream_tasks/data/proteinworkshop/ECReaction/testing.txt",
    ROOT / "downstream_tasks/data/proteinworkshop/ECReaction/validation.txt",
    ROOT / "downstream_tasks/data/proteinworkshop/GeneOntology/nrPDB-GO_train.txt",
    ROOT / "downstream_tasks/data/proteinworkshop/GeneOntology/nrPDB-GO_test.txt",
    ROOT / "downstream_tasks/data/proteinworkshop/GeneOntology/nrPDB-GO_valid.txt",
    ROOT / "downstream_tasks/data/proteinworkshop/masif_site/training.txt",
    ROOT / "downstream_tasks/data/proteinworkshop/masif_site/testing.txt",
    ROOT / "downstream_tasks/data/proteinworkshop/Metal3D/train.txt",
    ROOT / "downstream_tasks/data/proteinworkshop/Metal3D/test.txt",
]


def parse_pdb_id(s: str) -> str:
    s = s.strip()
    for sep in ['.', '-', '_']:
        if sep in s:
            return s.split(sep)[0].lower()
    return s[:4].lower()


def collect_pdb_ids() -> set:
    ids = set()
    for f in TASK_FILES:
        if Path(f).exists():
            for line in Path(f).read_text().splitlines():
                if line.strip():
                    ids.add(parse_pdb_id(line))
    return ids


def download_one(pdb_id: str, out_dir: Path) -> tuple[str, bool, str]:
    out_path = out_dir / f"{pdb_id}.pdb"
    if out_path.exists() and out_path.stat().st_size > 100:
        return pdb_id, True, "cached"
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    try:
        data = urllib.request.urlopen(url, timeout=30).read()
        if len(data) < 100:
            return pdb_id, False, "empty"
        out_path.write_bytes(data)
        return pdb_id, True, f"{len(data)//1024}KB"
    except urllib.error.HTTPError as e:
        return pdb_id, False, f"HTTP {e.code}"
    except Exception as e:
        return pdb_id, False, str(e)[:60]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=50)
    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdb_ids = collect_pdb_ids()
    # Filter already downloaded
    todo = [p for p in pdb_ids if not (out_dir / f"{p}.pdb").exists()
            or (out_dir / f"{p}.pdb").stat().st_size < 100]
    cached = len(pdb_ids) - len(todo)
    print(f"Total unique PDB IDs: {len(pdb_ids)}")
    print(f"Already cached: {cached}")
    print(f"To download: {len(todo)}")
    print(f"Workers: {args.workers}")
    print(f"Output: {out_dir}")
    sys.stdout.flush()

    if not todo:
        print("Nothing to download.")
        return

    ok = fail = 0
    failed_ids = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download_one, p, out_dir): p for p in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            pdb_id, success, msg = fut.result()
            if success:
                ok += 1
            else:
                fail += 1
                failed_ids.append((pdb_id, msg))
            if i % 500 == 0 or i == len(todo):
                elapsed = time.time() - t0
                rate = i / elapsed
                eta = (len(todo) - i) / rate if rate > 0 else 0
                print(f"  [{i}/{len(todo)}] ok={ok} fail={fail} "
                      f"{rate:.0f}/s ETA={eta/60:.1f}min", flush=True)

    elapsed = time.time() - t0
    total_ok = ok + cached
    print(f"\nDone in {elapsed/60:.1f}min")
    print(f"Successfully available: {total_ok}/{len(pdb_ids)} "
          f"({100*total_ok/len(pdb_ids):.1f}%)")
    if failed_ids:
        print(f"Failed ({len(failed_ids)}): {failed_ids[:10]}")
        fail_log = out_dir / "failed_ids.txt"
        fail_log.write_text("\n".join(f"{p}\t{m}" for p, m in failed_ids))
        print(f"Full fail list: {fail_log}")


if __name__ == "__main__":
    main()
