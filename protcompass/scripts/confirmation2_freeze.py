"""Freeze a second confirmation cohort, B, before any part of it is read.

The first cohort (scripts/confirmation_freeze.py, 2,000 domains) was drawn from 3,324 CATH v4.4 S40 domains absent
from the v4.3 release. After it was scored, the floor rules changed twice: an atom-level audit made floor
eligibility per encoder (scripts/atom_matched_floors.py, scripts/floor_rules.py), and the floor descriptor is now
chosen on the validation split. Those changes were made after cohort A's scores existed, so cohort A cannot confirm
them. Cohort B is the 1,324 candidates cohort A did not draw, all of them, with no sampling. It is written here with
its hash before any structure of it is unpacked.

Hypotheses. H1 to H4 are cohort A's, unchanged, under the released floor rule. H1b and H2b restate H1 and H2 under
the current rule (floors admissible for the encoder's measured inputs, descriptor and readout family chosen on the
released validation split); both rules are computed on released data only. H5 applies H1 to H4 on the part of the
cohort whose CATH topology is absent from the released training and validation splits, with the same thresholds,
provided it holds at least 20 domains.

  python confirmation2_freeze.py
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import confirmation_freeze as cf                                        # noqa: E402

ROOT, PD = cf.ROOT, cf.PD
OUT = ROOT / "confirmation2"

HYPOTHESES = cf.HYPOTHESES + [
    {"id": "H1b", "claim": "H1 with each encoder held only to floors built from inputs it is measured to read (the "
                           "residue-identity audit and the atom-level audit), the floor descriptor and readout family "
                           "chosen on the released validation split.",
     "released": "the counts of scripts/floor_rules.py under rule 'both', computed on released data only",
     "decision": "confirmed if every coordinate-reading panel encoder falls below its floor on at least 7 of the 9 probes"},
    {"id": "H2b", "claim": "H2 with the sequence floor chosen on the released validation split.",
     "released": "the counts of scripts/floor_rules.py under rule 'both'",
     "decision": "confirmed if 0 of the 27 sequence-only cells falls below the floor"},
    {"id": "H5", "claim": "H1 to H4 hold on the topology-novel part of cohort B: domains whose CATH topology is absent "
                          "from the released training and validation splits.",
     "released": "not tested on the released data; on cohort A the topology-novel part has 51 domains",
     "decision": "each of H1 to H4 decided on that part with its own rule, if the part holds at least 20 domains; "
                 "otherwise reported as not testable"},
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fp = OUT / "freeze.json"
    if fp.exists():
        raise SystemExit(f"{fp} exists; a freeze is written once")
    _, candidates, overlap, d44 = cf.build_cohort(cf.N_TARGET)
    cohort_a = {l.split(",")[0] for l in open(ROOT / "confirmation" / "cohort_manifest.csv").read().split("\n")[1:] if l}
    cohort = sorted(d for d in candidates if d not in cohort_a)
    assert not set(cohort) & cohort_a and len(cohort) == len(candidates) - len(cohort_a), (len(cohort), len(candidates))
    man = OUT / "cohort_manifest.csv"
    with open(man, "w") as fh:
        fh.write("domain,class,architecture,topology,homologous_superfamily\n")
        for d in cohort:
            fh.write(f"{d},{','.join(d44[d])}\n")
    freeze = {
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": "confirmation of the current probing claims and floor rules on proteins no part of this work has "
                   "inspected, including cohort A's analysis",
        "cohort": {
            "rule": "in cath-dataset-nonredundant-S40-v4_4_0.list, absent from cath-domain-list-v4_3_0.txt, and not "
                    "drawn into cohort A (confirmation/cohort_manifest.csv)",
            "source_release": "CATH v4.4.0, file date 16.12.2024",
            "candidates": len(candidates), "cohort_a": len(cohort_a),
            "overlap_with_anything_inspected": len(overlap),
            "sampling": "none: every remaining candidate",
            "n": len(cohort),
            "architectures": len({".".join(d44[d][:2]) for d in cohort}),
            "manifest": str(man.relative_to(ROOT)), "manifest_sha256": cf.sha256(man),
            "cohort_a_manifest_sha256": cf.sha256(ROOT / "confirmation" / "cohort_manifest.csv"),
            "provenance": "experimental structures, the CATH domain PDB set",
            "not_claimed": "absence from any encoder's pretraining corpus",
        },
        "panel": cf.PANEL,
        "floors": {"sequence": cf.FLOORS_SEQ, "coordinate": cf.FLOORS_COORD,
                   "admissible": "extract/encoder_atom_audit.csv and extract/encoder_input_audit.csv, released proteins only; "
                                 "multi-atom floors rebuilt on the encoder's atom set by scripts/atom_matched_floors.py"},
        "probes": cf.PROBES,
        "protocol": dict(cf.PROTOCOL, evaluate_on="the cohort B manifest written by this script, never used in fitting "
                                                  "or selection"),
        "hypotheses": HYPOTHESES,
        "script_sha256": {s: cf.sha256(ROOT / "scripts" / s) for s in
                          ["confirmation2_freeze.py", "confirmation_freeze.py", "run_probing_suite.py",
                           "floor_rules.py", "atom_matched_floors.py", "confirmation_report.py"]
                          if (ROOT / "scripts" / s).exists()},
    }
    fp.write_text(json.dumps(freeze, indent=2) + "\n")
    print(f"cohort B: {len(cohort)} domains over {freeze['cohort']['architectures']} architectures; "
          f"sha256 {freeze['cohort']['manifest_sha256'][:16]}; overlap with anything inspected {len(overlap)}")


if __name__ == "__main__":
    main()
