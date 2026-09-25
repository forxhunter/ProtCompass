"""Build the selection-corrected headline table: floor chosen on the `design` half, reported on the `score` half.

The current headline table (probing_results/floor_vs_encoders.csv) reads two maxima on the same data: the best of eight
descriptors and the best of 33 encoders, both on the full test split. The eight descriptors were designed in sequence,
each after the previous one lost, by looking at test scores, so the floor column carries an unknown amount of selection.
This script removes that by using disjoint proteins for the two steps:

  select   argmax over the eight floors on the `design` half
  report   that one floor's score on the `score` half, against the encoder maximum on the `score` half

Aggregation is the rule used by scripts/floor_vs_encoders.py, applied inside each evaluation half: headline metric is
Pearson r, else circular cosine, else balanced accuracy; mean over seeds within a readout family (linear, nonlinear);
then the better family. Two selection regimes are reported, differing in whether the readout family is also chosen on
the design half:

  floor_selection        floor identity chosen on design, readout family chosen within each half (matches the
                         existing table's rule, so the comparison isolates the floor-choice effect)
  floor_and_readout      floor identity and readout family both chosen on design, then read on score

  python design_score_table.py

Writes probing_results/design_score_split.csv and prints the table plus the summary counts.
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
RES = ROOT / "probing_results"
SPLITDIR = RES / "by_source_split"

FLOORS = ["ref_length", "ref_position", "ref_aa", "ref_seq_window", "ref_coords", "ref_backbone", "ref_global",
          "ref_pairhist"]
ORDER = ["phi_circular", "psi_circular", "local_curvature", "dssp_3", "dssp_8", "rsa", "rsa_bins", "bfactor",
         "core_surface", "contact_density", "distance_bins", "contact_order_len_resid",
         "radius_of_gyration_len_resid", "asphericity_len_resid", "architecture_label"]
UNDETERMINED = {"rsa", "rsa_bins", "bfactor", "architecture_label"}
TAUTOLOGICAL = {"contact_density", "contact_order_len_resid"}


def load():
    files = sorted(SPLITDIR.glob("*.csv"))
    if not files:
        raise SystemExit(f"no per-source files in {SPLITDIR}; run run_probing_suite_split.py first")
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    for col in ("pearson", "circ_cos", "balanced_accuracy"):
        if col not in d:
            d[col] = np.nan
    d["headline"] = d["pearson"].fillna(d["circ_cos"]).fillna(d["balanced_accuracy"])
    d["family"] = np.where(d.readout == "linear", "linear", "nonlinear")
    fam = d.groupby(["source", "probe", "eval_set", "family"], as_index=False).headline.mean()
    n = d.groupby(["probe", "eval_set"], as_index=False).n_eval.max()
    return fam, n


def best_family(fam, eval_set):
    """source x probe table of the better readout family, chosen inside `eval_set`."""
    m = fam[fam.eval_set == eval_set]
    b = m.loc[m.groupby(["source", "probe"]).headline.idxmax()]
    return (b.pivot(index="source", columns="probe", values="headline"),
            b.pivot(index="source", columns="probe", values="family"))


def family_fixed(fam, eval_set, chosen_family):
    """source x probe table read in `eval_set` at the readout family chosen elsewhere."""
    m = fam[fam.eval_set == eval_set].pivot_table(index=["source", "probe"], columns="family", values="headline")
    out = pd.DataFrame(index=chosen_family.index, columns=chosen_family.columns, dtype=float)
    for s in chosen_family.index:
        for p in chosen_family.columns:
            f = chosen_family.loc[s, p]
            if isinstance(f, str) and (s, p) in m.index:
                out.loc[s, p] = m.loc[(s, p), f]
    return out


def main():
    fam, n_eval = load()
    present_floors = [f for f in FLOORS if f in set(fam.source)]
    absent = [f for f in FLOORS if f not in present_floors]
    if absent:
        raise SystemExit(f"missing split scores for floors: {absent}. Refusing to write a table that omits a floor.")

    wide = {}
    chosen_fam = {}
    for es in ("test", "design", "score"):
        wide[es], chosen_fam[es] = best_family(fam, es)
    # readout family selected on design, read on score
    score_at_design_family = family_fixed(fam, "score", chosen_fam["design"])

    enc_names = sorted(i for i in wide["test"].index if not i.startswith("ref_"))
    rows = []
    for p in ORDER:
        if p not in wide["test"].columns:
            continue
        fl = {es: wide[es].loc[present_floors, p] for es in wide}
        en = {es: wide[es].loc[enc_names, p].dropna() for es in wide}

        sel_design = fl["design"].idxmax()                       # the floor chosen on held-out design proteins
        sel_test = fl["test"].idxmax()                           # the floor the current table reports
        sel_score_oracle = fl["score"].idxmax()

        floor_rep = fl["score"][sel_design]                      # reported number: selected floor, score half
        enc_max = en["score"].max()
        enc_arg = en["score"].idxmax()

        # variant: floor identity and readout family both fixed on design
        fl2 = score_at_design_family.loc[present_floors, p]
        en2 = score_at_design_family.loc[enc_names, p].dropna()
        floor_rep2 = fl2[sel_design]

        # Half-to-half noise. The two halves are disjoint protein samples of the same size, so for one source the
        # difference (design - score) has variance 2 sigma^2 where sigma is the sampling standard error of a score on
        # one half; the RMS of that difference over sources estimates sigma. The same construction applied to the
        # quantity that is actually reported, the encoder-minus-selected-floor difference, estimates the standard error
        # of the gap, which is smaller than sqrt(2) sigma because both terms are read on the same proteins.
        both = pd.concat([wide["design"][p], wide["score"][p]], axis=1).dropna()
        sigma = float(np.sqrt(((both.iloc[:, 0] - both.iloc[:, 1]) ** 2).mean() / 2)) if len(both) else np.nan
        gd = (wide["design"].loc[enc_names, p] - fl["design"][sel_design]).dropna()
        gs = (wide["score"].loc[enc_names, p] - floor_rep).dropna()
        common = gd.index.intersection(gs.index)
        sigma_gap = float(np.sqrt(((gd[common] - gs[common]) ** 2).mean() / 2)) if len(common) else np.nan

        rows.append({
            "probe": p,
            "determined": "no" if p in UNDETERMINED else ("yes, floor contains label" if p in TAUTOLOGICAL else "yes"),
            "level": fam[(fam.probe == p)].iloc[0].name if False else "",
            "n_eval_design": int(n_eval[(n_eval.probe == p) & (n_eval.eval_set == "design")].n_eval.iloc[0]),
            "n_eval_score": int(n_eval[(n_eval.probe == p) & (n_eval.eval_set == "score")].n_eval.iloc[0]),
            # selection-corrected
            "selected_floor_design": sel_design,
            "selected_floor_on_design": fl["design"][sel_design],
            "selected_floor_on_score": floor_rep,
            "max_encoder_score_half": enc_max,
            "which_encoder_score_half": enc_arg,
            "gap_corrected": enc_max - floor_rep,
            "encoders_below_selected_floor": int((en["score"] < floor_rep).sum()),
            "n_encoders": int(en["score"].notna().sum()),
            # the current, selection-on-test table recomputed from the same runs
            "best_floor_test": fl["test"].max(),
            "which_floor_test": sel_test,
            "max_encoder_test": en["test"].max(),
            "which_encoder_test": en["test"].idxmax(),
            "gap_test": en["test"].max() - fl["test"].max(),
            "encoders_below_best_floor_test": int((en["test"] < fl["test"].max()).sum()),
            # selection diagnostics
            "argmax_floor_differs_design_vs_test": sel_design != sel_test,
            "argmax_floor_score_half": sel_score_oracle,
            "argmax_floor_differs_design_vs_score": sel_design != sel_score_oracle,
            "best_floor_on_score_half_oracle": fl["score"].max(),
            "selection_optimism_floor": fl["design"].max() - floor_rep,
            "gap_corrected_minus_gap_test": (enc_max - floor_rep) - (en["test"].max() - fl["test"].max()),
            # variant: readout family also fixed on design
            "selected_floor_on_score_designfam": floor_rep2,
            "max_encoder_score_designfam": en2.max(),
            "gap_corrected_designfam": en2.max() - floor_rep2,
            "encoders_below_designfam": int((en2 < floor_rep2).sum()),
            "half_noise_sigma": sigma,
            "gap_noise_sigma": sigma_gap,
        })
    t = pd.DataFrame(rows).drop(columns=["level"])
    t = t.sort_values("gap_corrected").reset_index(drop=True)
    t.round(4).to_csv(RES / "design_score_split.csv", index=False)

    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
    show = ["probe", "determined", "selected_floor_design", "selected_floor_on_design", "selected_floor_on_score",
            "max_encoder_score_half", "which_encoder_score_half", "gap_corrected",
            "encoders_below_selected_floor", "n_encoders", "gap_test", "gap_corrected_minus_gap_test",
            "argmax_floor_differs_design_vs_test", "half_noise_sigma", "gap_noise_sigma"]
    print(t[show].round(4).to_string(index=False))

    # The full-test column of this rerun should reproduce the published table. If it does not, the two are not
    # measuring the same thing and nothing below can be compared to FINDINGS.md section 1.
    pub = RES / "floor_vs_encoders.csv"
    if pub.exists():
        q = pd.read_csv(pub).set_index("probe")
        j = t.set_index("probe")
        shared = [p for p in j.index if p in q.index]
        d_gap = (j.loc[shared, "gap_test"] - q.loc[shared, "gap"]).abs()
        d_fl = (j.loc[shared, "best_floor_test"] - q.loc[shared, "best floor"]).abs()
        same_floor = (j.loc[shared, "which_floor_test"] == q.loc[shared, "which floor"]).all()
        print(f"\ncross-check against published floor_vs_encoders.csv over {len(shared)} probes: "
              f"max |gap difference| {d_gap.max():.2e}, max |best-floor difference| {d_fl.max():.2e}, "
              f"same best floor on every probe: {same_floor}")

    det = ~t.probe.isin(UNDETERMINED)
    inf = det & ~t.probe.isin(TAUTOLOGICAL)
    print(f"\nfloor wins (gap < 0), selection-corrected: {int((t.gap_corrected < 0).sum())} of {len(t)} probes")
    print(f"   determined labels:            {int((t.gap_corrected < 0)[det].sum())} of {int(det.sum())}")
    print(f"   determined and informative:   {int((t.gap_corrected < 0)[inf].sum())} of {int(inf.sum())}")
    print(f"floor wins on the current selection-on-test table: {int((t.gap_test < 0).sum())} of {len(t)}")
    print(f"   determined and informative:   {int((t.gap_test < 0)[inf].sum())} of {int(inf.sum())}")
    flip = t[(t.gap_test < 0) != (t.gap_corrected < 0)]
    print(f"\nrows where the conclusion flips: {len(flip)}")
    if len(flip):
        print(flip[["probe", "gap_test", "gap_corrected", "selected_floor_design", "which_floor_test"]].round(4).to_string(index=False))
    print(f"\ndesign-half argmax floor differs from test-half argmax floor on "
          f"{int(t.argmax_floor_differs_design_vs_test.sum())} of {len(t)} probes: "
          f"{', '.join(t[t.argmax_floor_differs_design_vs_test].probe)}")
    print(f"design-half argmax differs from score-half argmax on "
          f"{int(t.argmax_floor_differs_design_vs_score.sum())} of {len(t)} probes")
    print(f"mean selection optimism of the floor (design max minus its score-half value): "
          f"{t.selection_optimism_floor.mean():+.4f}; max {t.selection_optimism_floor.max():+.4f}")
    print(f"\nvariant with the readout family also fixed on design: floor wins "
          f"{int((t.gap_corrected_designfam < 0).sum())} of {len(t)}, determined and informative "
          f"{int((t.gap_corrected_designfam < 0)[inf].sum())} of {int(inf.sum())}")
    print(f"\nwrote {RES / 'design_score_split.csv'}")


if __name__ == "__main__":
    main()
