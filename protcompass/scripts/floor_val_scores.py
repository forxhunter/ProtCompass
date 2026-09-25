"""Score every floor descriptor on the validation split, so that the floor of each probe can be chosen without the
test set.

The released floor of a probe is the best of the descriptors and readouts on the test split, the same split the gap
is reported on. This run gives the validation scores that let the choice be made on validation instead. It reuses
scripts/run_probing_suite.py unchanged: the Data object is built as released and its test part is then pointed at
the validation proteins (residue rows, protein indices and sampled pairs), so every probe is trained on the training
split, stopped early on validation as released, and scored on validation. Nothing in the test split is read.

Output: probing_results_val/by_source/<floor>.csv, the released layout, with validation scores in place of test scores.

  python floor_val_scores.py [ref_aa ref_backbone ...]
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import os
import sys
from pathlib import Path

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
os.environ["PROBING_OUT"] = str(ROOT / "probing_results_val")
sys.path.insert(0, str(ROOT / "scripts"))

import run_probing_suite as rps                                          # noqa: E402


def run_with_retry(data, source):
    """rps.run_source with the released retry on a shared GPU: wait and try again rather than lose the run."""
    import time
    import torch
    for attempt in range(12):
        try:
            return rps.run_source(data, source)
        except (torch.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower():
                raise
            torch.cuda.empty_cache()
            print(f"[{source}] CUDA out of memory (attempt {attempt + 1}/12); waiting 5 min", flush=True)
            time.sleep(300)
    raise SystemExit(f"[{source}] still out of memory after 12 attempts")

FLOORS = ["ref_length", "ref_position", "ref_aa", "ref_seq_window", "ref_seq_expert", "ref_seq_global",
          "ref_coords", "ref_backbone", "ref_global", "ref_pairhist"]


def main():
    wanted = sys.argv[1:] or FLOORS
    (rps.OUT / "by_source").mkdir(parents=True, exist_ok=True)
    data = rps.Data()
    for k in ("rows", "chosen", "prot_idx", "pairs", "pair_y"):
        getattr(data, k)["test"] = getattr(data, k)["val"]
    for s in wanted:
        if (rps.OUT / "by_source" / f"{s}.csv").exists():
            continue
        run_with_retry(data, s)


if __name__ == "__main__":
    main()
