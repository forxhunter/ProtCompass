"""Write the freeze of the pre-registered determinacy test, with automated protocol assertions, before any encoder or
reference is scored on the new labels.

  python prereg_freeze.py --draft     # prereg/freeze_draft.json, for the synthetic pipeline check only
  python prereg_freeze.py             # prereg/freeze.json and prereg/freeze.sha256.json; refuses to overwrite

The freeze records the hypotheses and their interpretation, the labels and their hashes, the effective-input map and
the checks behind it, the model panel and the family grouping, the reference family and the admissibility rule, the
readout-selection rule, the statistics, and the sha256 of every script and data file the decision depends on.
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
sys.path.insert(0, str(ROOT / "scripts"))
from family_weighted_demand import family                                 # noqa: E402
from panel import panel                                                   # noqa: E402

P = ROOT / "prereg"
SEQ_ALWAYS = ["ref_length", "ref_position"]
SEQ_ID = ["ref_aa", "ref_seq_window", "ref_seq_expert", "ref_seq_global"]
FIVE = {"N", "CA", "C", "O", "CB"}
COORD_FLOORS = {  # floor -> atoms it is built from ("full" = every heavy atom of the domain file)
    "ref_coords": {"CA"}, "ref_pairhist": {"CA"},
    "ref_backbone@CA": {"CA"}, "ref_global@CA": {"CA"},
    "ref_backbone@N+CA+C": {"N", "CA", "C"}, "ref_global@N+CA+C": {"N", "CA", "C"},
    "ref_backbone@N+CA+C+O": {"N", "CA", "C", "O"}, "ref_global@N+CA+C+O": {"N", "CA", "C", "O"},
    "ref_backbone@N+CA+CB": {"N", "CA", "CB"}, "ref_global@N+CA+CB": {"N", "CA", "CB"},
    "ref_backbone": FIVE, "ref_global": FIVE,
    "ref_atom_neighborhood@CA": {"CA"}, "ref_atom_neighborhood@N+CA+C": {"N", "CA", "C"},
    "ref_atom_neighborhood@N+CA+C+O": {"N", "CA", "C", "O"}, "ref_atom_neighborhood@N+CA+CB": {"N", "CA", "CB"},
    "ref_atom_neighborhood@full": {"full"}, "ref_allatom_local": {"full", "identity"},
}
FLOORS = SEQ_ALWAYS + SEQ_ID + list(COORD_FLOORS)
COHORT_B = ["esm2_650m", "prottrans_t5_xl_u50", "ankh_base", "esm3", "esm_if1", "proteinmpnn",
            "gcpnet_pretrained", "gcpnet_random"]
CODE = ["prereg_labels.py", "prereg_chi2.py", "prereg_floors.py", "prereg_determinacy.py", "prereg_run.py",
        "prereg_decide.py", "prereg_freeze.py", "prereg_merge_cohortB.py", "run_probing_suite.py",
        "atom_matched_floors.py", "family_weighted_demand.py", "panel.py", "confirmation_merge.py"]


def sha(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def encoder_inputs():
    dm = pd.read_csv(P / "determinacy_map.csv")
    inp = dm.groupby("encoder").input.first().to_dict()
    ident = pd.read_csv(ROOT / "probing_results" / "floor_admissibility.csv").set_index("encoder").residue_identity
    return inp, {e: bool(ident.get(e, True)) for e in inp}


def admissible(inp, ident):
    out = {}
    for e, s in inp.items():
        atoms = set() if s == "sequence" else (FIVE | {"full"} if s == "full" else set(s.split("+")))
        if ident[e]:
            atoms |= {"identity"}
        fl = SEQ_ALWAYS + (SEQ_ID if ident[e] else [])
        fl += [f for f, need in COORD_FLOORS.items() if atoms and need <= atoms]
        out[e] = fl
    return out


def assertions(adm, inp, panel_res):
    """Automated checks of the protocol; every one must hold or nothing is frozen."""
    import prereg_decide as pdd
    A = {}
    # 1 floors are built only from inputs the encoder reads
    A["admissible_floors_within_audited_input"] = all(
        (COORD_FLOORS.get(f, set()) - {"identity"}) <= (set() if inp[e] == "sequence" else
                                                       (FIVE | {"full"} if inp[e] == "full" else set(inp[e].split("+"))))
        for e, fl in adm.items() for f in fl if f in COORD_FLOORS)
    # 2 and 3 selection uses validation only: permuting every test score leaves every choice unchanged
    rng = np.random.default_rng(0)
    rows = []
    for src in ["enc_a", "floor_a", "floor_b"]:
        for ro, seeds in (("linear", [0]), ("mlp2", [0, 1, 2])):
            for s in seeds:
                for split in ("val", "test"):
                    rows.append(dict(source=src, probe="hbond_count", readout=ro, seed=s, split=split,
                                     balanced_accuracy=rng.random()))
    R = pd.DataFrame(rows)
    fake = dict(panel_residue=["enc_a"], admissible_floors={"enc_a": ["floor_a", "floor_b"]})
    C1 = pdd.cells(R, fake, {("enc_a", "hbond_count"): "determined"})
    R2 = R.copy(); m = R2.split == "test"
    R2.loc[m, "balanced_accuracy"] = rng.permutation(R2.loc[m, "balanced_accuracy"].to_numpy())
    C2 = pdd.cells(R2, fake, {("enc_a", "hbond_count"): "determined"})
    A["encoder_readout_selected_on_validation_only"] = bool((C1.enc_family == C2.enc_family).all())
    A["floor_and_floor_readout_selected_on_validation_only"] = bool(
        (C1.floor == C2.floor).all() and (C1.floor_family == C2.floor_family).all())
    # 4 every primary exact-determined cell rests on a passed reconstruction, every undetermined one on a construction
    chk = pd.read_csv(P / "determinacy_checks.csv").set_index("check").passes
    A["all_determinacy_checks_used_pass_or_cell_not_established"] = bool(chk[[
        "hbond@N+CA+C+O", "ca_dihedral@CA", "chi1@full", "chi2@full", "hbond@CA peptide-plane rotation",
        "chi1@N+CA+C+O+CB side-chain rotation", "chi2@N+CA+C+O+CB side-chain rotation",
        "ca_dihedral@sequence crankshaft"]].all())
    # 5 every panel encoder has residue embeddings, every family is defined, every admissible list is non-empty
    A["panel_encoders_have_residue_embeddings"] = all((ROOT / "probing_embeddings" / e / "residue_fp16.npy").exists()
                                                      for e in panel_res)
    A["families_defined"] = all(family(e) for e in panel_res)
    A["admissible_lists_nonempty"] = all(len(adm[e]) > 0 for e in panel_res)
    return A


def main():
    draft = "--draft" in sys.argv
    dest = P / ("freeze_draft.json" if draft else "freeze.json")
    if not draft and dest.exists():
        raise SystemExit("prereg/freeze.json exists; a freeze is written once")
    inp, ident = encoder_inputs()
    panel_res = [e for e in panel() if (ROOT / "probing_embeddings" / e / "residue_fp16.npy").exists()]
    adm = admissible(inp, ident)
    A = assertions(adm, inp, panel_res)
    if not all(A.values()):
        raise SystemExit(f"protocol assertion failed: {A}")
    dm = pd.read_csv(P / "determinacy_map.csv")
    F = dict(
        written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        status="draft for the synthetic pipeline check" if draft else "frozen before any encoder or reference was scored on the new labels",
        claim=("Empirical prediction. Under the pre-specified reference family and evaluation protocol below, labels "
               "determined by an encoder's audited input are predicted to yield predominantly non-positive "
               "encoder-minus-reference gaps (G_ref <= 0), whereas labels not determined by that input are predicted "
               "to yield predominantly positive gaps. This is a prediction about this reference family, not a "
               "consequence of the definition: a label that is a function of the input need not be recovered by a "
               "fixed descriptor and a finite probe."),
        labels=dict(
            primary={"hbond_count": "backbone hydrogen bonds the residue takes part in (DSSP energy < -0.5 kcal/mol, amide H from the backbone), classes 0, 1, 2+",
                     "chi1_class": "chi1 rotamer class g- [0,120), t [120,240), g+ [240,360)",
                     "chi2_class": "chi2 rotamer class for R K Q E M I L W, same bins"},
            control={"ca_dihedral": "CA(i-1) CA(i) CA(i+1) CA(i+2) virtual dihedral as (sin, cos); positive control, excluded from the headline"},
            descriptive={"sc_contacts": "residues with a side-chain heavy atom within 4.5 A of this residue's side chain; reported only"},
            dropped={"hse_up": "duplicates the CB-neighbour count (Spearman 0.88)", "sc_rsa": "duplicates RSA (0.965)",
                     "sc_polar_hbond": "3.5 A N/O criterion without angles and with His/Asn/Gln flip ambiguity"},
            metric={"hbond_count": "balanced_accuracy", "chi1_class": "balanced_accuracy", "chi2_class": "balanced_accuracy",
                    "ca_dihedral": "circ_cos", "sc_contacts": "pearson"},
            sha256={f: sha(P / f) for f in ["labels_released.h5", "labels_cohortB.h5", "labels_cohortB_merged.h5"]},
        ),
        determinacy=dict(
            definition=("determined: the label is reconstructed exactly from the atoms the encoder is measured to read "
                        "(agreement >= 0.99999, angle error <= 1e-4); undetermined: an executable construction changes the "
                        "label with every read atom fixed; not_established otherwise. not_established cells do not enter "
                        "H1-H4 and are reported descriptively. No approximate category is used."),
            checks=pd.read_csv(P / "determinacy_checks.csv").to_dict("records"),
            map=dm.to_dict("records"),
            audits={f: sha(ROOT / "extract" / f) for f in ["encoder_atom_audit.csv", "encoder_sidechain_audit.csv"]},
        ),
        panel_residue=panel_res,
        panel_excluded={"foldvision": "no residue-level embeddings (one vector per protein), so no residue label can be probed"},
        families={e: family(e) for e in panel_res},
        floors=FLOORS,
        floor_atoms={f: sorted(v) for f, v in COORD_FLOORS.items()},
        admissible_floors={e: adm[e] for e in panel_res},
        admissibility_rule=("an encoder is held to every floor built only from inputs it is measured to read: length and "
                            "position always; residue-identity floors if it reads residue identity; a coordinate floor if "
                            "the floor's atoms are a subset of the encoder's audited atoms (full = side chains read, "
                            "encoder_sidechain_audit.csv)"),
        readout=dict(families=["linear (closed-form ridge for angle and count, softmax for classes)", "mlp2 (two hidden layers of 128, seeds 0 1 2, mean)"],
                     fitting="train part; ridge alpha and MLP early stopping on the validation part, as in the released suite",
                     selection=("for encoders and floors alike: the family with the higher validation headline score, "
                                "ties to linear; the reference is the admissible floor with the highest validation score; "
                                "test scores are read only after both choices"),
                     split="released superfamily split, seed-0 draw of 3,000 train, 500 validation and 1,000 test domains"),
        hypotheses=dict(
            unit="an (architecture family, label, status) triple; G_ref averaged over the family's members of that status",
            H1="among determined units on the primary labels, the share with G_ref <= 0 is at least 0.80",
            H2="among undetermined units on the primary labels, the share with G_ref > 0 is at least 0.80",
            H3=("Delta_G = mean G_ref over undetermined units minus mean over determined units is > 0, and the 95% "
                "interval of a bootstrap resampling architecture families (10,000 draws, seed 0) excludes 0"),
            H4=("on cohort B (proteins frozen before reading, confirmation2/freeze.json) with the cohort panel below: "
                "Delta_G^B > 0 and at least 80% of primary cells have the predicted sign"),
            threshold_note=("0.80 is a pre-specified tolerance for exceptions and measurement noise, not a fitted estimate "
                            "of the released rate; the shares at every threshold from 0.50 to 1.00 are reported descriptively"),
            cell_intervals="protein-level paired bootstrap, 1,000 draws, seed 0; encoder and floor on the same resampled proteins",
        ),
        cohort_B_panel=COHORT_B,
        cohort_B_note=("the six encoders frozen for cohort B plus GCPNet (pretrained and random), added before scoring so "
                       "that the side-chain labels have a determined arm on cohort B; the six-encoder subset is reported "
                       "alongside"),
        interpretation={
            "H1 H2 H3 H4 hold": "a prospective confirmation of the prediction",
            "H1 H2 H3 hold, H4 fails": "generalisation to new labels on the development proteins holds; external replication on new proteins does not",
            "H3 holds, an 80% sign rule fails": "determinacy predicts a graded shift in the gap, not a binary sign",
            "determined side fails, undetermined side holds": "input determinacy alone is insufficient to predict floor dominance",
            "undetermined side fails": "encoders do not generally infer missing label information beyond the fixed descriptors",
            "hbond_count alone fails": "reported as a frozen failure on that label, with an analysis of descriptor sufficiency; it is not removed from the headline",
            "chi1 or chi2 fails for the full-atom encoders": "even when the input determines side-chain geometry, the representation and readout may not keep it accessible",
            "the whole test fails": "the central claim that determinacy predicts the sign is dropped; the matched-reference measurement framework is kept",
        },
        development_context=("released labels under admissible floors chosen on validation: 231/245 determined cells of "
                             "coordinate encoders at or below the floor and 0/110 for sequence-only encoders; with the encoder "
                             "readout chosen either way the count lies between 231 and 237 of 245, and stays 0/110 "
                             "(probing_results/floor_rules_cells.csv)"),
        protocol_assertions=A,
        code_sha256={f: sha(ROOT / "scripts" / f) for f in CODE},
        floor_cache_sha256={p.name: sha(p) for p in sorted((P / "floor_cache").rglob("*.npy"))},
    )
    dest.write_text(json.dumps(F, indent=2, default=str))
    if not draft:
        (P / "freeze.sha256.json").write_text(json.dumps(dict(file="freeze.json", sha256=sha(dest),
                                                              written_utc=F["written_utc"]), indent=2))
    print(f"wrote {dest}; assertions {A}")


if __name__ == "__main__":
    main()
