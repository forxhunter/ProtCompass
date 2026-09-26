"""Freeze the confirmation analysis before the uninspected cohort is scored.

Every floor, descriptor, threshold and naming rule in this paper was developed against the same CATH 4.3 splits the
results are reported on. Appendix O.7 says so, and it is the central objection to the analysis: splitting an already
inspected test set afterwards measures one internal selection effect and does not remove the dependence introduced by
repeated descriptor design.

This script writes the contract for a confirmation run on proteins no part of this work has inspected. It must run,
and its output must be on disk, before any score is computed on the cohort. It records four things:

  cohort      the rule that defines it, the resulting manifest, and a hash of that manifest
  panel       the encoders, fixed here and not chosen after seeing a cohort score
  protocol    floors, floor assignment, probes, readout family and selection rule, all copied from the released run
  hypotheses  what the released run predicts for this cohort, with the direction and threshold stated in advance

The cohort rule: a domain in the CATH v4.4 non-redundant S40 set that is absent from the whole v4.3 release. The
S40 set is non-redundant at 40% sequence identity within itself, so such a domain is below that identity to every
domain the released probing dataset was built from, and it entered CATH after the snapshot every descriptor was
designed against. This is new evaluation exposure at the domain level. It is not a new-fold test: every topology
among the added domains already exists in v4.3, and it is not a statement about what any encoder saw in pretraining,
which we cannot observe.

Probes are fitted on the released v4.3 training split and its validation split, with the released embeddings, and
evaluated on this cohort. Nothing is refitted and no threshold is re-chosen.

  python confirmation_freeze.py [--n 2000] [--force]
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
PD = ROOT / "probing_data"
OUT = ROOT / "confirmation"

SEED = 20260922
N_TARGET = 2000

# Fixed here, before any cohort score exists. Six encoders spanning the audit groups: four measured sequence-only,
# one multimodal, two coordinate-reading.
PANEL = ["esm2_650m", "prottrans_t5_xl_u50", "ankh_base", "esm3", "esm_if1", "proteinmpnn"]

# The eight released floors. Assignment is the released rule: an encoder measured not to read coordinates is held to
# the sequence floors, one measured to read them to the best of all eight.
FLOORS_SEQ = ["ref_length", "ref_position", "ref_aa", "ref_seq_window", "ref_seq_expert", "ref_seq_global"]
FLOORS_COORD = ["ref_coords", "ref_backbone", "ref_global", "ref_pairhist"]

# The fifteen probes of the floor comparison, with the status the released run assigns.
PROBES = {
    "phi_circular": "determined", "psi_circular": "determined", "local_curvature": "determined",
    "dssp_3": "determined", "dssp_8": "determined", "core_surface": "determined",
    "contact_density": "tautological", "distance_bins": "determined",
    "contact_order_len_resid": "tautological", "radius_of_gyration_len_resid": "determined",
    "asphericity_len_resid": "determined", "rsa": "undetermined", "rsa_bins": "undetermined",
    "bfactor": "undetermined", "architecture_label": "not established",
}

PROTOCOL = {
    "fit_on": "probing_data/splits/superfamily_train.txt and superfamily_val.txt, v4.3, released embeddings",
    "evaluate_on": "the cohort manifest written by this script, never used in fitting or selection",
    "readouts": "closed-form ridge with the penalty chosen on the released validation split, and MLP-2 "
                "(width 128, at most 30 epochs, patience 3, seeds 0/1/2); the family is chosen on validation loss",
    "aggregation": "mean over seeds within a readout family, then the better family, as in floor_vs_encoders.py",
    "floor_assignment": "the measured input audit in extract/encoder_input_audit.csv, unchanged",
    "no_refitting": "no descriptor, threshold, probe or readout rule may be changed after this file is written",
}

# What the released run predicts, stated before the cohort is scored. Each is a direction, not a value.
HYPOTHESES = [
    {"id": "H1", "claim": "On the nine informative determined probes, every coordinate-reading encoder in the panel "
                          "scores below the floor built from the atoms it receives.",
     "released": "188 of 200 cells over 23 coordinate encoders",
     "decision": "confirmed if both coordinate encoders fall below their floor on at least 7 of the 9 probes"},
    {"id": "H2", "claim": "On the nine informative determined probes, no sequence-only encoder in the panel falls "
                          "below the sequence floor.",
     "released": "0 of 90 cells over 10 sequence-only encoders",
     "decision": "confirmed if 0 of the 27 sequence-only cells falls below the floor"},
    {"id": "H3", "claim": "On RSA, RSA bins and B-factor, some encoder in the panel beats every floor.",
     "released": "+0.071, +0.119 and +0.073 over the full panel",
     "decision": "confirmed if the best panel encoder exceeds the best floor on all three"},
    {"id": "H4", "claim": "On fold architecture, ESM-IF1, ESM3 and ProstT5 clear ref_pairhist and ESM2-650M does "
                          "not. ProstT5 is outside this panel, so the prediction is tested for ESM-IF1 and ESM3.",
     "released": "+0.055, +0.056 and +0.048 clear; ESM2-650M at -0.004 does not",
     "decision": "confirmed if ESM-IF1 and ESM3 exceed ref_pairhist and ESM2-650M does not"},
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_list(path):
    return {l.strip() for l in open(path) if l.strip() and not l.startswith("#")}


def domain_table(path):
    out = {}
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        out[f[0]] = tuple(f[1:5])
    return out


def build_cohort(n_target):
    """Domains in the v4.4 non-redundant S40 set and absent from the whole v4.3 release, stratified by architecture."""
    s40_44 = read_list(PD / "raw_v4_4" / "cath-dataset-nonredundant-S40-v4_4_0.list")
    d43 = domain_table(PD / "raw" / "cath-domain-list-v4_3_0.txt")
    d44 = domain_table(PD / "raw_v4_4" / "cath-domain-list-v4_4_0.txt")
    candidates = sorted(d for d in s40_44 if d not in d43)

    # every domain this project has ever touched, from the released dataset and its rejection log
    seen = set(l.strip() for l in open(ROOT / "probing_embeddings" / "ids.txt") if l.strip())
    for f in (PD / "splits").glob("*.txt"):
        seen |= read_list(f)
    import csv as _csv
    with open(PD / "exclusions.csv") as fh:
        seen |= {r["domain"] for r in _csv.DictReader(fh)}
    overlap = sorted(set(candidates) & seen)
    candidates = [d for d in candidates if d not in seen]

    by_arch = {}
    for d in candidates:
        by_arch.setdefault(".".join(d44[d][:2]), []).append(d)
    rng = np.random.default_rng(SEED)
    # proportional allocation with a floor of 10 per architecture, so small architectures are not lost
    order = sorted(by_arch)
    total = sum(len(by_arch[a]) for a in order)
    take = {a: min(len(by_arch[a]), max(10, round(n_target * len(by_arch[a]) / total))) for a in order}
    while sum(take.values()) > n_target:
        a = max(order, key=lambda x: take[x])
        if take[a] <= 10:
            break
        take[a] -= 1
    cohort = []
    for a in order:
        pool = sorted(by_arch[a])
        cohort += list(rng.choice(pool, size=take[a], replace=False)) if take[a] < len(pool) else pool
    cohort = sorted(cohort)
    return cohort, candidates, overlap, d44


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_TARGET)
    ap.add_argument("--force", action="store_true", help="overwrite an existing freeze (records the reason)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    freeze_path = OUT / "freeze.json"
    if freeze_path.exists() and not a.force:
        raise SystemExit(f"{freeze_path} exists. A freeze is written once; pass --force only to correct it before "
                         f"any cohort score exists.")

    cohort, candidates, overlap, d44 = build_cohort(a.n)
    man = OUT / "cohort_manifest.csv"
    with open(man, "w") as fh:
        fh.write("domain,class,architecture,topology,homologous_superfamily\n")
        for d in cohort:
            fh.write(f"{d},{','.join(d44[d])}\n")

    scripts = ["confirmation_freeze.py", "floor_vs_encoders.py", "run_probing_suite.py", "summarize_probing.py"]
    freeze = {
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": "confirmation of the released probing claims on proteins no part of this work has inspected",
        "cohort": {
            "rule": "in cath-dataset-nonredundant-S40-v4_4_0.list and absent from cath-domain-list-v4_3_0.txt",
            "source_release": "CATH v4.4.0, file date 16.12.2024",
            "candidates_before_sampling": len(candidates),
            "overlap_with_anything_inspected": len(overlap),
            "sampling": f"stratified by CATH architecture, floor of 10 per architecture, seed {SEED}",
            "n": len(cohort),
            "architectures": len({".".join(d44[d][:2]) for d in cohort}),
            "manifest": str(man.relative_to(ROOT)),
            "manifest_sha256": sha256(man),
            "provenance": "experimental structures, the CATH domain PDB set; no predicted model enters the cohort",
            "not_claimed": "topology-disjointness from the released training split, and absence from any encoder's "
                           "pretraining corpus, neither of which this construction establishes",
        },
        "panel": PANEL,
        "floors": {"sequence": FLOORS_SEQ, "coordinate": FLOORS_COORD},
        "probes": PROBES,
        "protocol": PROTOCOL,
        "hypotheses": HYPOTHESES,
        "script_sha256": {s: sha256(ROOT / "scripts" / s) for s in scripts if (ROOT / "scripts" / s).exists()},
        "input_sha256": {
            "cath_domain_list_v4_4_0": sha256(PD / "raw_v4_4" / "cath-domain-list-v4_4_0.txt"),
            "s40_v4_4_0": sha256(PD / "raw_v4_4" / "cath-dataset-nonredundant-S40-v4_4_0.list"),
            "cath_domain_list_v4_3_0": sha256(PD / "raw" / "cath-domain-list-v4_3_0.txt"),
        },
    }
    freeze_path.write_text(json.dumps(freeze, indent=2) + "\n")

    print(f"cohort candidates (in v4.4 S40, absent from v4.3): {len(candidates):,}")
    print(f"overlap with any domain this project has touched:  {len(overlap)}")
    print(f"cohort written: {len(cohort):,} domains over "
          f"{len({'.'.join(d44[d][:2]) for d in cohort})} architectures -> {man}")
    print(f"manifest sha256: {freeze['cohort']['manifest_sha256'][:16]}...")
    print(f"freeze written:  {freeze_path}")
    for h in HYPOTHESES:
        print(f"  {h['id']}: {h['decision']}")


if __name__ == "__main__":
    main()
