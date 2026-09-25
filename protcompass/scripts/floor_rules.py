"""The determined-probe floor counts under four rules for which floor an encoder is held to, and the per-encoder
admissibility matrix the rules rest on.

  released     a sequence-only encoder is held to the best sequence floor and a coordinate encoder to the best of all
               floors, each chosen on the test split (scripts/emit_conditional_floor_table.py; 233 of 245 and 0 of 110)
  admissible   each encoder is held only to floors built from inputs it is measured to read: residue identity from
               the released input audit, the atom set from the atom-level audit (extract/encoder_atom_audit.csv), and
               the two multi-atom floors rebuilt from that atom set (scripts/atom_matched_floors.py)
  validation   the released eligibility, with the floor descriptor and its readout family chosen on the validation
               split (scripts/floor_val_scores.py) and the chosen floor's test score used
  both         admissible eligibility and validation choice

Length and position are treated as inputs of every encoder; the script reports whether either floor is ever the
binding one, which is what would make that assumption matter. Encoder scores are as released: the better readout
family on the test split, which favours the encoder and so makes every count of cells below the floor conservative.

  python floor_rules.py [--latex-counts] [--latex-matrix]
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
import glob
from pathlib import Path

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
RES = ROOT / "probing_results"
SEQ_FLOORS = ["ref_length", "ref_position", "ref_aa", "ref_seq_window", "ref_seq_expert", "ref_seq_global"]
CA_FLOORS = ["ref_coords", "ref_pairhist"]
MULTI = ["ref_backbone", "ref_global"]
ALL_FLOORS = SEQ_FLOORS + CA_FLOORS + MULTI
ATOMS = ["N", "CA", "C", "O", "CB"]
DETERMINED = ["phi_circular", "psi_circular", "local_curvature", "dssp_3", "dssp_8", "core_surface",
              "contact_density", "distance_bins", "contact_order_len_resid", "radius_of_gyration_len_resid",
              "asphericity_len_resid"]
TAUTOLOGICAL = ["contact_density", "contact_order_len_resid"]
NAME = {"esm1b": "ESM-1b", "prottrans_t5_xl_u50": "ProtT5-XL", "protbert_bfd": "ProtBert-BFD", "esm2_8m": "ESM-2 8M",
        "esm2_35m": "ESM-2 35M", "esm2_150m": "ESM-2 150M", "esm2_650m": "ESM-2 650M", "esm2_3b": "ESM-2 3B",
        "ankh_base": "Ankh", "prostt5": "ProstT5", "schnet": "SchNet", "tfn": "TFN", "dimenet": "DimeNet",
        "ipa": "IPA", "mace": "MACE", "gearnet": "GearNet", "gvp": "GVP", "cdconv": "CDConv",
        "gcpnet_random": "GCPNet", "pottsmpnn": "PottsMPNN", "proteinmpnn": "ProteinMPNN",
        "pottsmpnn_pretrained": "PottsMPNN (pt)", "pottsmpnn_soluble": "PottsMPNN (sol)",
        "pottsmpnn_ft": "PottsMPNN (ft)", "gearnet_pretrained": "GearNet (pt)", "tfn_pretrained": "TFN (pt)",
        "ipa_pretrained": "IPA (pt)", "gcpnet_pretrained": "GCPNet (pt)", "esm_if1": "ESM-IF1", "saprot": "SaProt",
        "esm3": "ESM3", "foldseek_3di": "Foldseek 3Di", "foldvision": "FoldVision"}


def scores(paths):
    d = pd.concat([pd.read_csv(f) for p in paths for f in glob.glob(str(p / "by_source" / "*.csv"))], ignore_index=True)
    d["headline"] = d["pearson"].fillna(d["circ_cos"]).fillna(d["balanced_accuracy"])
    d["family"] = np.where(d.readout == "linear", "linear", "nonlinear")
    return d.groupby(["source", "probe", "family"]).headline.mean().rename("score").reset_index()


def admissibility():
    # the probing-dataset row where there is one, else the stage-1 row (ESM2-3B was audited on stage 1 only)
    audit = pd.read_csv(ROOT / "extract" / "encoder_input_audit.csv")
    audit = audit.assign(o=audit.dataset.ne("probing")).sort_values("o").drop_duplicates("encoder").set_index("encoder")
    atom = pd.read_csv(ROOT / "extract" / "encoder_atom_audit.csv").set_index("encoder")
    rows = []
    from panel import panel
    for e, a in audit.iterrows():
        if e not in panel():
            continue
        coords = bool(a.uses_coordinates)
        atoms = [x for x in ATOMS if coords and e in atom.index and bool(atom.loc[e, f"reads_{x}"])]
        floors = ["ref_length", "ref_position"]
        if bool(a.uses_residue_identity):
            floors += SEQ_FLOORS[2:]
        if "CA" in atoms:
            floors += CA_FLOORS
            floors += MULTI if atoms == ATOMS else [f"{m}@{'+'.join(atoms)}" for m in MULTI]
        rows.append(dict(encoder=e, residue_identity=bool(a.uses_residue_identity), coordinates=coords,
                         atoms="+".join(atoms) if atoms else ("--" if not coords else "unmeasured"),
                         scale=bool(atom.loc[e, "reads_scale"]) if coords and e in atom.index else None,
                         scale_change=float(atom.loc[e, "rel_change_scale"]) if coords and e in atom.index else np.nan,
                         floors=floors))
    return pd.DataFrame(rows).set_index("encoder")


def floor_value(test, val, probe, eligible, choose_on):
    """The floor over the eligible descriptors: best on test, or chosen on validation and read on test."""
    t = test[(test.probe == probe) & test.source.isin(eligible)]
    if not len(t):
        return np.nan, None
    if choose_on == "test":
        r = t.loc[t.score.idxmax()]
        return float(r.score), (r.source, r.family)
    v = val[(val.probe == probe) & val.source.isin(eligible)]
    v = v[v.set_index(["source", "family"]).index.isin(t.set_index(["source", "family"]).index)]
    r = v.loc[v.score.idxmax()]
    return float(t[(t.source == r.source) & (t.family == r.family)].score.iloc[0]), (r.source, r.family)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex-counts", action="store_true")
    ap.add_argument("--latex-matrix", action="store_true")
    a = ap.parse_args()
    test = scores([RES, ROOT / "probing_results_atom"])
    val = scores([ROOT / "probing_results_val", ROOT / "probing_results_atom_val"])
    enc = test[~test.source.str.startswith("ref_")].groupby(["source", "probe"]).score.max().unstack()
    M = admissibility()
    panel = [e for e in enc.index if e in M.index]
    rows, binding = [], []
    for rule, elig, choose in (("released", "group", "test"), ("admissible", "own", "test"),
                               ("validation", "group", "val"), ("both", "own", "val")):
        for p in DETERMINED:
            cache = {}
            for e in panel:
                s = enc.loc[e, p] if p in enc.columns else np.nan
                if not np.isfinite(s):
                    continue
                crd = bool(M.loc[e, "coordinates"])
                fl = (ALL_FLOORS if crd else SEQ_FLOORS) if elig == "group" else M.loc[e, "floors"]
                key = tuple(fl)
                if key not in cache:
                    cache[key] = floor_value(test, val, p, fl, choose)
                f, which = cache[key]
                if not np.isfinite(f):
                    continue
                rows.append(dict(rule=rule, probe=p, encoder=e, coordinates=crd, score=s, floor=f,
                                 floor_source=which[0], floor_family=which[1], below=bool(s < f)))
                if which[0] in ("ref_length", "ref_position"):
                    binding.append((rule, p, e, which[0]))
    D = pd.DataFrame(rows)
    D.to_csv(RES / "floor_rules_cells.csv", index=False)
    S = (D.groupby(["rule", "coordinates"]).below.agg(["sum", "count"]).reset_index())
    Si = (D[~D.probe.isin(TAUTOLOGICAL)].groupby(["rule", "coordinates"]).below.agg(["sum", "count"]).reset_index())
    S = S.merge(Si, on=["rule", "coordinates"], suffixes=("", "_informative"))
    S.to_csv(RES / "floor_rules_counts.csv", index=False)
    M.assign(floors=M.floors.map(lambda f: " ".join(f))).to_csv(RES / "floor_admissibility.csv")
    rel = S[S.rule == "released"].set_index("coordinates")
    assert (int(rel.loc[True, "sum"]), int(rel.loc[True, "count"])) == (233, 245), rel
    assert (int(rel.loc[False, "sum"]), int(rel.loc[False, "count"])) == (0, 110), rel
    moved = D.pivot_table(index=["probe", "encoder"], columns="rule", values="below").astype(bool)
    if a.latex_counts:
        print("% generated by scripts/floor_rules.py; do not edit by hand")
        print("\\begin{tabular}{lcccc}")
        print("\\toprule")
        print("rule & eligibility & floor chosen on & coordinate encoders & sequence-only encoders \\\\")
        print("\\midrule")
        desc = {"released": ("input group", "test"), "admissible": ("measured inputs", "test"),
                "validation": ("input group", "validation"), "both": ("measured inputs", "validation")}
        for r in ("released", "admissible", "validation", "both"):
            x = S[S.rule == r].set_index("coordinates")
            print(f"{r} & {desc[r][0]} & {desc[r][1]} & {int(x.loc[True, 'sum'])} of {int(x.loc[True, 'count'])} & "
                  f"{int(x.loc[False, 'sum'])} of {int(x.loc[False, 'count'])} \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")
        return
    if a.latex_matrix:
        print("% generated by scripts/floor_rules.py; do not edit by hand")
        print("\\begin{tabular}{lccccl}")
        print("\\toprule")
        print("encoder & residue identity & coordinates & atoms read & scale & multi-atom floors \\\\")
        print("\\midrule")
        for e, r in M.sort_values(["coordinates", "atoms"]).iterrows():
            multi = ("--" if not r.coordinates else "released" if r.atoms == "+".join(ATOMS)
                     else f"rebuilt on {r.atoms.replace('+', ', ')}" if "CA" in r.atoms else "none")
            sc = "--" if r.scale is None or pd.isna(r.scale) else ("yes" if r.scale else "no")
            print(f"{NAME.get(e, e)} & {'yes' if r.residue_identity else 'no'} & {'yes' if r.coordinates else 'no'} & "
                  f"{r.atoms.replace('+', ', ')} & {sc} & {multi} \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")
        return
    pd.set_option("display.width", 200)
    print(S.to_string(index=False))
    print("\nbinding length/position floors:", len(binding), binding[:5])
    for r in ("admissible", "validation", "both"):
        ch = moved[moved[r] != moved["released"]]
        print(f"{r}: {len(ch)} cells change verdict relative to released", ch.index.tolist()[:12])
    print(M[["residue_identity", "coordinates", "atoms", "scale", "scale_change"]].to_string())


if __name__ == "__main__":
    main()
