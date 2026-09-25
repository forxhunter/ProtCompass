"""Freeze the protocol, the panel and the hypotheses of the topology-disjoint probing test before any source is scored
on that split.

The CATH 4.3 topology split holds 1,924 test domains in 122 topologies, none of which occurs in its training split
(checked below). Apart from the fold-architecture label (Appendix: fold architecture on the topology split), no probe of
this paper has been scored on it. The development of every floor and rule used the superfamily split, and the symmetric
rerun on that split (scripts/symmetric_run.py, symmetric_decide.py) is the reference the thresholds below were set
from. This script writes symmetric/freeze_topology.json and its sha256; scripts/symmetric_run.py refuses to score the
topology split unless the file exists and is unchanged.

  python symmetric_freeze.py
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import datetime
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
sys.path.insert(0, str(ROOT / "scripts"))
from panel import panel                                                  # noqa: E402

SYM = ROOT / "symmetric"
OUT = SYM / "freeze_topology.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; a freeze is written once")
    if (SYM / "results_topology" / "by_source").exists() and any((SYM / "results_topology" / "by_source").glob("*.csv")):
        raise SystemExit("a source has already been scored on the topology split")
    h = h5py.File(ROOT / "probing_data" / "labels.h5", "r")["protein"]
    T = np.array([f"{c}.{a}.{t}" for c, a, t in zip(h["cath_C"][:], h["cath_A"][:], h["cath_T"][:])])
    s = h["split_topology"][:]
    shared = int(np.isin(T[s == 2], list(set(T[s == 0]))).sum())
    assert shared == 0, shared
    floors = (["ref_length", "ref_position", "ref_aa", "ref_seq_window", "ref_seq_expert", "ref_seq_global",
               "ref_coords", "ref_pairhist", "ref_backbone", "ref_global"]
              + [f"{k}@{a}" for a in ("CA", "N+CA+C", "N+CA+C+O", "N+CA+CB") for k in ("ref_backbone", "ref_global")]
              + [f"ref_nbr{k}@{a}" for k in (4, 8, 16) for a in ("CA", "N+CA+C", "N+CA+C+O", "N+CA+CB", "N+CA+C+O+CB")])
    freeze = {
        "written_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "split": "topology (labels.h5 split_topology): 15,378 training, 2,159 validation and 1,924 test domains; "
                 f"test domains whose CATH topology occurs in the training split: {shared}; residue and pair probes on "
                 "the seed-0 draw of 3,000 / 500 / 1,000 domains, protein probes on every domain of each part",
        "prior_use": "fold architecture was scored on this split before (the released topology-split architecture "
                     "check); no other probe and no other floor has been",
        "encoders": panel(),
        "descriptive_sources": ["gvp_vframe", "gcpnet_random_vframe", "gcpnet_pretrained_vframe", "mace_vframe"],
        "floors": floors,
        "protocol": "scripts/symmetric_run.py unchanged: linear readout and MLP-2 with seeds 0, 1, 2, fitted on the "
                    "training part with the ridge penalty and early stopping on validation; validation and test "
                    "predicted from one fit. scripts/symmetric_decide.py unchanged: readout family of every encoder "
                    "and every floor chosen on validation; floor = the (descriptor, family) with the best validation "
                    "score among the floors built only from inputs the encoder is measured to read (floor_rules.py "
                    "admissibility); 2,000 paired bootstrap draws over test proteins; a cell is below / above when "
                    "the 95% percentile interval of encoder minus floor lies below / above zero, else unresolved",
        "labels": {"determined": ["phi_circular", "psi_circular", "local_curvature", "dssp_3", "dssp_8",
                                  "core_surface", "contact_density", "distance_bins", "contact_order_len_resid",
                                  "radius_of_gyration_len_resid", "asphericity_len_resid"],
                   "undetermined": ["rsa", "rsa_bins", "bfactor"], "not_established": ["architecture_label"]},
        "hypotheses": {
            "T1": "Of the coordinate-encoder cells on the eleven determined labels, at least {T1:.0%} are below the "
                  "admissible floor (95% interval below zero).",
            "T2": "No sequence-only cell on the eleven determined labels is below its admissible floor (95% interval "
                  "below zero).",
            "T3": "On each of rsa, rsa_bins and bfactor, at least one encoder is above its admissible floor (95% "
                  "interval above zero).",
            "T4": "Over coordinate-encoder cells, the mean gap to the admissible floor is higher on the three "
                  "undetermined labels than on the eleven determined ones, with a 95% bootstrap interval over "
                  "architecture families (10,000 draws) above zero.",
        },
        "thresholds": {"T1": None},
        "interpretation": {
            "T1-T4 hold": "the input-matched reading of Section 4.1 holds on folds absent from training",
            "T1 fails, T4 holds": "on new folds fewer determined cells are resolved below the floor; the direction of "
                                   "the determined / undetermined separation holds and its size does not",
            "T2 fails": "a sequence-only encoder falls below a sequence floor on a new fold; the claim that "
                        "sequence-only encoders stay above their floors does not transfer",
            "T3 fails": "no encoder passes the floor on some undetermined label on new folds",
            "T4 fails": "on new folds determinacy does not separate the gaps; the reading of Section 4.1 is "
                        "specific to the development split",
        },
        "sha256": {str(p.relative_to(ROOT)): sha(p) for p in
                   [ROOT / "scripts" / f for f in ("symmetric_run.py", "symmetric_decide.py", "run_probing_suite.py",
                                                   "atom_matched_floors.py", "prereg_floors.py", "floor_rules.py",
                                                   "prereg_decide.py", "symmetric_freeze.py")]
                   + [ROOT / "probing_data" / "labels.h5", ROOT / "extract" / "encoder_input_audit.csv",
                      ROOT / "extract" / "encoder_atom_audit.csv"]},
    }
    t1 = float(sys.argv[sys.argv.index("--t1") + 1])
    freeze["thresholds"]["T1"] = t1
    freeze["hypotheses"]["T1"] = freeze["hypotheses"]["T1"].format(T1=t1)
    OUT.write_text(json.dumps(freeze, indent=2))
    (SYM / "freeze_topology.sha256.json").write_text(json.dumps({"file": OUT.name, "sha256": sha(OUT),
                                                                 "written_utc": freeze["written_utc"]}, indent=2))
    print(json.dumps(freeze["hypotheses"], indent=2))
    print("frozen", sha(OUT))


if __name__ == "__main__":
    main()
