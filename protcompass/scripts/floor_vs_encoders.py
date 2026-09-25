"""Best non-learned floor against every encoder, per probe, with one aggregation rule used everywhere.

Aggregation: for each source and probe, take the mean over seeds within a readout family (linear or nonlinear) and
then the better family. That is the same rule `summarize_probing.py` applies, so the numbers here and there agree.
An earlier version of this table took the maximum over all (readout, seed) runs for the floor and the mean for the
encoders, which flattered the floor.

Each probe is also labelled by whether its target is a closed-form function of the atoms the encoder receives. That
column is an analytic statement about the label, established by derivation and checked numerically in
`scripts/check_label_determinacy.py`, not an empirical result about any descriptor.
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path((PROTCOMPASS_ROOT + "/protcompass/probing_results"))
FLOORS = ["ref_length", "ref_position", "ref_aa", "ref_seq_window", "ref_seq_expert", "ref_seq_global", "ref_coords",
          "ref_backbone", "ref_global", "ref_pairhist"]
COORD_FLOORS = ["ref_coords", "ref_backbone", "ref_global", "ref_pairhist"]
ORDER = ["phi_circular", "psi_circular", "local_curvature", "dssp_3", "dssp_8", "rsa", "rsa_bins", "bfactor",
         "core_surface", "contact_density", "distance_bins", "contact_order_len_resid",
         "radius_of_gyration_len_resid", "asphericity_len_resid", "architecture_label"]

# Is the label a closed-form function of N, CA, C, O, CB?
# yes: derived in the text. no: needs information the encoder is not given.
DETERMINED = {
    "phi_circular": "yes, dihedral of C(i-1) N CA C",
    "psi_circular": "yes, dihedral of N CA C N(i+1)",
    "local_curvature": "yes, angle between consecutive CA-CA vectors",
    "dssp_3": "yes, DSSP energies use N, CA, C, O",
    "dssp_8": "yes, same",
    "rsa": "no, freesasa uses side-chain atoms beyond CB",
    "rsa_bins": "no, same",
    "bfactor": "no, experimental quantity",
    "core_surface": "yes, CB neighbour count",
    "contact_density": "yes, CA neighbour count",
    "distance_bins": "yes, the CA-CA distance itself",
    "contact_order_len_resid": "yes, function of the contact map",
    "radius_of_gyration_len_resid": "yes, second moment of the CA coordinates",
    "asphericity_len_resid": "yes, gyration-tensor invariant",
    "architecture_label": "no, CATH database label",
}


def main():
    d = pd.concat([pd.read_csv(f) for f in sorted((OUT / "by_source").glob("*.csv"))], ignore_index=True)
    d["headline"] = d["pearson"].fillna(d["circ_cos"] if "circ_cos" in d else np.nan).fillna(d["balanced_accuracy"])
    d["family"] = np.where(d.readout == "linear", "linear", "nonlinear")
    m = d.groupby(["source", "probe", "family"]).headline.mean().reset_index()
    best = m.loc[m.groupby(["source", "probe"]).headline.idxmax()]
    wide = best.pivot(index="source", columns="probe", values="headline")

    # A floor listed in FLOORS but absent from by_source/ used to be skipped silently, which produced a table that
    # looked like it covered every listed floor while a floor that beats the encoders was missing from it. The failure
    # is loud now: a floor is either scored or not listed.
    absent = [f for f in FLOORS if f not in wide.index]
    if absent:
        raise SystemExit(f"FLOORS lists {', '.join(absent)} but probing_results/by_source/ has no scores for them. "
                         f"Run run_probing_suite.py for those sources, or remove them from FLOORS. Refusing to write "
                         f"a table that silently omits a floor.")
    missing = {f: [c for c in ORDER if c in wide.columns and pd.isna(wide.loc[f, c])] for f in FLOORS}
    missing = {f: c for f, c in missing.items() if c}
    if missing:
        for f, c in missing.items():
            print(f"warning: {f} has no score on {len(c)} of the compared probes ({', '.join(c)}); "
                  f"the best-floor column on those probes is a maximum over the remaining floors")
    cols = [c for c in ORDER if c in wide.columns]
    enc = wide.drop(index=[i for i in wide.index if i.startswith("ref_")])[cols]

    rows = []
    for c in cols:
        all_f = wide.loc[[f for f in FLOORS if f in wide.index], c]
        generic = wide.loc[[f for f in COORD_FLOORS if f in wide.index and f != "ref_global"], c]
        rows.append({
            "probe": c,
            "label determined by the input": DETERMINED.get(c, "?"),
            "best floor": all_f.max(), "which floor": all_f.idxmax(),
            "floor without ref_global": max(all_f.drop(index="ref_global", errors="ignore").max(), generic.max()),
            "max encoder": enc[c].max(), "which encoder": enc[c].idxmax(),
            "median encoder": enc[c].median(),
            "encoders below best floor": int((enc[c] < all_f.max()).sum()), "n encoders": int(enc[c].notna().sum()),
        })
    t = pd.DataFrame(rows).set_index("probe")
    t["gap"] = t["max encoder"] - t["best floor"]
    t["gap without ref_global"] = t["max encoder"] - t["floor without ref_global"]
    t.round(4).to_csv(OUT / "floor_vs_encoders.csv")

    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 20)
    print(t.sort_values("gap").round(3).to_string())
    det = t["label determined by the input"].str.startswith("yes")
    print(f"\nprobes where no encoder beats the floor: {int((t.gap <= 0).sum())} of {len(t)}")
    print(f"   among labels determined by the input: {int(((t.gap <= 0) & det).sum())} of {int(det.sum())}")
    print(f"   among labels not determined:          {int(((t.gap <= 0) & ~det).sum())} of {int((~det).sum())}")
    print(f"without ref_global, no encoder beats the floor on {int((t['gap without ref_global'] <= 0).sum())} of {len(t)}")


if __name__ == "__main__":
    main()
