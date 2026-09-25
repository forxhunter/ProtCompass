"""Turn probing_results/scores_long.csv into encoder-by-probe tables.

Headline metric: Pearson r (regression), mean cosine of the angular error (circular), balanced accuracy (classification).
Gain = encoder score minus the reference floor for that encoder's inputs, on the same probe with the same readout
family. Reference feature sets: chain length, residue position, amino-acid identity or composition, and a fixed
non-learned descriptor of the backbone (sorted kNN CA distances, chain distances, shell counts). An encoder that
receives coordinates is held to the coordinate floor as well, so that recovering a property that a hand-made
descriptor of its own input already exposes earns no credit. A probe carries information about an encoder only
through its gain.
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path((PROTCOMPASS_ROOT + "/protcompass/probing_results"))
ORDER = ["phi_circular", "psi_circular", "phi_angle_bins", "psi_angle_bins", "local_curvature", "dssp_3", "dssp_8", "rsa", "rsa_bins",
         "bfactor", "core_surface", "contact_density", "distance_bins", "contact_order_len_resid", "radius_of_gyration_len_resid",
         "asphericity_len_resid", "contact_order", "radius_of_gyration", "asphericity", "architecture_label", "sequence_recovery"]
# Probes that enter the IIP profile: one formulation per property, length-residualised global targets.
PROFILE = {"I1 local conformation": ["phi_circular", "psi_circular"], "I2 local curvature": ["local_curvature"],
           "I3 secondary structure": ["dssp_3", "dssp_8"], "I4 solvent accessibility": ["rsa", "rsa_bins"],
           "I5 flexibility": ["bfactor"], "I6 burial and packing": ["core_surface", "contact_density"],
           "I7 pairwise distance": ["distance_bins"],
           "I8 global geometry": ["contact_order_len_resid", "radius_of_gyration_len_resid", "asphericity_len_resid"],
           "I9 fold architecture": ["architecture_label"]}


def main():
    d = pd.concat([pd.read_csv(f) for f in sorted((OUT / "by_source").glob("*.csv"))], ignore_index=True)
    d["headline"] = d["pearson"].fillna(d["circ_cos"] if "circ_cos" in d else np.nan).fillna(d["balanced_accuracy"])
    d["family"] = np.where(d.readout == "linear", "linear", "nonlinear")
    m = d.groupby(["source", "probe", "family"]).headline.agg(["mean", "std"]).reset_index()
    best = m.loc[m.groupby(["source", "probe"])["mean"].idxmax()]           # best readout family per source and probe
    wide = best.pivot(index="source", columns="probe", values="mean").reindex(columns=[c for c in ORDER if c in best.probe.unique()])
    is_ref = wide.index.str.startswith("ref_")

    # The floor is conditional on what the encoder is handed. A model that receives residue identity gets no credit for
    # recovering it, and a model that receives coordinates gets no credit for recovering what a fixed descriptor of the
    # same coordinates already exposes. A sequence-only model is not held to the coordinate floor: inferring solvent
    # accessibility from sequence alone is a property of the representation, not input pass-through.
    seq_floor = wide.loc[[r for r in ("ref_length", "ref_position", "ref_aa", "ref_seq_window") if r in wide.index]].max(axis=0)
    all_floor = wide[is_ref].max(axis=0)   # includes ref_backbone, the input-matched coordinate floor
    audit = pd.read_csv(Path(OUT).parent / "extract" / "encoder_input_audit.csv")
    audit = audit[audit.dataset == "probing"].drop_duplicates("encoder").set_index("encoder")
    uses_coords = {k: bool(audit.loc[k, "uses_coordinates"]) if k in audit.index else False for k in wide.index[~is_ref]}
    floor = pd.DataFrame({k: (all_floor if uses_coords[k] else seq_floor) for k in wide.index[~is_ref]}).T
    gain = wide[~is_ref] - floor
    pd.Series(uses_coords, name="uses_coordinates").to_csv(OUT / "floor_assignment.csv")
    pd.DataFrame({"sequence-input floor": seq_floor, "coordinate-input floor": all_floor}).to_csv(OUT / "floors.csv")
    wide.round(4).to_csv(OUT / "headline_by_probe.csv")
    gain.round(4).to_csv(OUT / "gain_over_reference_by_probe.csv")
    profile = pd.DataFrame({iip: gain[[p for p in probes if p in gain]].mean(axis=1) for iip, probes in PROFILE.items()})
    profile.round(4).to_csv(OUT / "iip_profile_gain.csv")

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    short = {c: c.replace("_len_resid", "*").replace("radius_of_gyration", "Rg").replace("contact_order", "CO").replace("asphericity", "asph")
             .replace("architecture_label", "arch").replace("sequence_recovery", "seqrec").replace("contact_density", "cdens")
             .replace("core_surface", "core").replace("local_curvature", "curv").replace("distance_bins", "dist").replace("_angle_bins", "_bin")
             .replace("_circular", "_circ") for c in wide.columns}
    print("Headline score, best readout (* = length-residualised target)")
    print(wide.rename(columns=short).round(3).to_string())
    print("\nFloors per probe. An encoder that receives coordinates is held to the higher one.")
    print(pd.DataFrame({"sequence-input": seq_floor, "coordinate-input": all_floor}).rename(index=short).round(3).T.to_string())
    print(f"held to the coordinate floor: {sum(uses_coords.values())} of {len(uses_coords)} encoders")
    print("\nIIP profile as gain over the floor")
    print(profile.round(3).to_string())
    sd = d[d.family == "nonlinear"].groupby(["source", "probe"]).headline.std()
    print(f"\nMedian seed sd of nonlinear readouts: {sd.median():.4f}; max {sd.max():.4f}")


if __name__ == "__main__":
    main()
