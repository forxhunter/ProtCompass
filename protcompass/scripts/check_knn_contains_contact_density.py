"""Does the kNN distance block of the coordinate floors still contain the contact-density label?

Section 2 of FINDINGS.md used to present the 8 A shell feature as a fault that was found and corrected by moving the
shells to 5, 7, 9, 11, 13 A. The concern is that the correction does not remove the label, because the
descriptor also stores the 16 smallest CA-CA distances, and

    contact_count = #{k : knn_k < 8}

whenever the count is at most 16, saturating at 16 above that, which is exactly the top bin of the four-class label.
This script settles it on the released data rather than by argument: it rebuilds the kNN block from the stored
coordinates and reports how often the raw count and the four-class bin are recovered from the kNN distances alone.

  python check_knn_contains_contact_density.py --proteins 400
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
from pathlib import Path

import h5py
import numpy as np

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
K = 16                 # the descriptor stores the 16 smallest CA-CA distances
CUTOFF = 8.0           # the label counts CA atoms within 8 A
BINS = (5, 10, 15)     # contact_density_bin: 0-5 / 6-10 / 11-15 / 16+


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proteins", type=int, default=400)
    a = ap.parse_args()
    h = h5py.File(ROOT / "probing_data" / "labels.h5", "r")
    offs, L = h["protein/offset"][:], h["protein/length"][:]
    coords = h["residue/coords"]
    count = h["residue/contact_count"][:]

    rng = np.random.default_rng(0)
    idx = rng.choice(len(L), size=min(a.proteins, len(L)), replace=False)
    n_res = exact_count = exact_bin = 0
    for i in idx:
        r = slice(int(offs[i]), int(offs[i]) + int(L[i]))
        ca = coords[r][:, 1].astype(np.float64)
        if len(ca) < 2:
            continue
        d = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        knn = np.sort(d, axis=1)[:, :K]                       # exactly what the descriptor stores
        recovered = (knn < CUTOFF).sum(1)                     # the count, saturating at K
        true = count[r].astype(int)
        n_res += len(true)
        exact_count += int((recovered == true).sum())
        exact_bin += int((np.digitize(recovered, BINS) == np.digitize(true, BINS)).sum())
    print(f"{len(idx)} proteins, {n_res} residues")
    print(f"raw contact_count recovered from the kNN distances alone: {exact_count / n_res:.4%}")
    print(f"four-class contact_density_bin recovered:                 {exact_bin / n_res:.4%}")
    print("\nThe kNN block is a sufficient statistic for a neighbour count at any radius up to the k-th neighbour, so\n"
          "moving the shell radii does not remove the label. A floor score on this probe is a probe-fitting artefact,\n"
          "not an information bound.")


if __name__ == "__main__":
    main()
