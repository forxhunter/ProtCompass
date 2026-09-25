"""Read the held-out erasure results against both nulls and report what changes.

Two nulls are scored for every cell. The permuted sham is the one the released results use: it fits LEACE to a
permuted copy of the property, which matches the real erasure in rank and in the change of basis. The
variance-matched null of scripts/matched_null.py additionally matches the share of representation variance the real
map removes, in principal directions chosen to carry as little of the property as the search finds.

Two contrasts are reported against each null.

  effect   the difference in level, null minus erased. This asks whether the erased representation scores below a
           representation damaged the same way but not in the property's directions.
  R        the difference in repair, [erased+q - erased] minus [null+q - null], with q the property block. This
           asks whether returning the property helps the arm the property was taken from more than it helps an
           equally damaged arm that never carried it.

The second is the contrast the naming claims rest on. It is the more robust of the two because a capacity loss
common to both arms cancels in the difference of differences, while it does not cancel in a difference of levels.

  python matched_null_report.py [--latex]
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
CD = ROOT / "confirmation_downstream"
LABEL = {"membrane_soluble": "membrane vs soluble", "subcellular_localization": "subcellular localisation",
         "fluorescence": "fluorescence", "stability": "stability", "ec_classification": "EC classification",
         "go_mf": "GO-MF", "go_bp": "GO-BP", "remote_homology": "remote homology"}


def load():
    """The four CPU tasks come from one file per encoder shard, the four multi-output tasks from the GPU runner."""
    files = sorted((CD / "matched_parts").glob("*.csv")) + [CD / "bighead_matched" / "remote_holdout.csv"]
    frames = []
    for f in files:
        if f.exists():
            d = pd.read_csv(f)
            if len(d):
                frames.append(d)
    if not frames:
        raise SystemExit("no matched-null results yet")
    d = pd.concat(frames, ignore_index=True)
    return d.drop_duplicates(subset=["task", "encoder", "arm", "descriptor"], keep="last")


def per_cell(d):
    rows = []
    for (task, enc), g in d.groupby(["task", "encoder"]):
        p = g.pivot_table(index="arm", columns="descriptor", values="score")
        fam = {"sham": [a for a in p.index if a.startswith("sham")],
               "null": [a for a in p.index if a.startswith("null")]}
        if not fam["null"] or "erased" not in p.index or "intact" not in p.index:
            continue
        rec = dict(task=task, encoder=enc, intact=p.loc["intact", "none"], erased=p.loc["erased", "none"])
        vr = g.loc[g.arm == "erased", "var_removed"]
        rec["var_removed_erased"] = float(vr.iloc[0]) if len(vr) else np.nan
        for name, arms in fam.items():
            lvl = p.loc[arms, "none"].mean()
            rec[f"{name}_level"] = lvl
            rec[f"effect_{name}"] = lvl - rec["erased"]
            rec[f"{name}_above_intact"] = bool(lvl > rec["intact"])
            v = g.loc[g.arm.isin(arms), "var_removed"]
            rec[f"var_removed_{name}"] = float(v.mean()) if len(v) else np.nan
            for q in ("named", "permuted", "other"):
                if q not in p.columns or p.loc[arms, q].isna().all():
                    continue
                rec[f"R_{q}_{name}"] = ((p.loc["erased", q] - p.loc["erased", "none"])
                                        - (p.loc[arms, q] - p.loc[arms, "none"]).mean())
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    a = ap.parse_args()
    C = per_cell(load())
    C.to_csv(CD / "matched_null_cells.csv", index=False)

    summ = []
    for task, g in C.groupby("task"):
        summ.append(dict(
            task=task, label=LABEL.get(task, task), n=len(g),
            var_erased=g.var_removed_erased.mean(), var_sham=g.var_removed_sham.mean(),
            var_null=g.var_removed_null.mean(),
            sham_above_intact=int(g.sham_above_intact.sum()), null_above_intact=int(g.null_above_intact.sum()),
            effect_sham=g.effect_sham.mean(), n_effect_sham_pos=int((g.effect_sham > 0).sum()),
            effect_null=g.effect_null.mean(), n_effect_null_pos=int((g.effect_null > 0).sum()),
            R_sham=g.get("R_named_sham", pd.Series(dtype=float)).mean(),
            n_R_sham_pos=int((g.get("R_named_sham", pd.Series(dtype=float)) > 0).sum()),
            R_null=g.get("R_named_null", pd.Series(dtype=float)).mean(),
            n_R_null_pos=int((g.get("R_named_null", pd.Series(dtype=float)) > 0).sum())))
    S = pd.DataFrame(summ).sort_values("task").reset_index(drop=True)
    S.to_csv(CD / "matched_null_summary.csv", index=False)

    if a.latex:
        print("% generated by scripts/matched_null_report.py; do not edit by hand")
        print("\\begin{tabular}{lrrrrrrr}")
        print("\\toprule")
        print(" & \\multicolumn{3}{c}{variance removed} & \\multicolumn{2}{c}{effect} & "
              "\\multicolumn{2}{c}{repair contrast $R$} \\\\")
        print("\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}")
        print("task & erased & sham & null & vs.\\ sham & vs.\\ null & vs.\\ sham & vs.\\ null \\\\")
        print("\\midrule")
        for r in S.itertuples():
            rn = "" if np.isnan(r.R_null) else f"${r.R_null:+.4f}$ ({r.n_R_null_pos}/{r.n})"
            rs = "" if np.isnan(r.R_sham) else f"${r.R_sham:+.4f}$ ({r.n_R_sham_pos}/{r.n})"
            print(f"{r.label} & {r.var_erased:.3f} & {r.var_sham:.3f} & {r.var_null:.3f} & "
                  f"${r.effect_sham:+.4f}$ ({r.n_effect_sham_pos}/{r.n}) & "
                  f"${r.effect_null:+.4f}$ ({r.n_effect_null_pos}/{r.n}) & {rs} & {rn} \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")
        return

    pd.set_option("display.width", 250)
    print("=== variance removed and the sham-above-intact pathology ===")
    print(S[["label", "n", "var_erased", "var_sham", "var_null",
             "sham_above_intact", "null_above_intact"]].round(4).to_string(index=False))
    print("\n=== level contrast (effect) and repair contrast (R), against each null ===")
    print(S[["label", "n", "effect_sham", "n_effect_sham_pos", "effect_null", "n_effect_null_pos",
             "R_sham", "n_R_sham_pos", "R_null", "n_R_null_pos"]].round(4).to_string(index=False))
    print(f"\nwritten: {CD / 'matched_null_cells.csv'} and {CD / 'matched_null_summary.csv'}")


if __name__ == "__main__":
    main()
