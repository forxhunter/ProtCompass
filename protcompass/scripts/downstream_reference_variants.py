"""Sensitivity of the downstream sequence reference to its feature set.

The downstream reference of the manuscript (`ref_composition`, scripts/make_downstream_reference.py) contains the
20 amino-acid frequencies and the sequence length. This script asks how far the reference moves when it is given
more: explicit physicochemical descriptors, and the 400 dipeptide frequencies, which add residue-order information
at lag 1. Every variant is a fixed function of the amino-acid sequence with no learned parameter.

Variants (columns):
  ref_composition              20 frequencies, log L, L/600, fraction outside the 20-letter alphabet (23); a
                               recomputation of the released reference, used as a check on this script
  ref_physchem                 physicochemical descriptors, log L, L/600 (22); no amino-acid frequencies
  ref_comp_physchem            ref_composition + the 20 descriptors (43)
  ref_comp_dipeptide           ref_composition + 400 dipeptide frequencies (423)
  ref_comp_physchem_dipeptide  all of the above (443)

Physicochemical descriptors, 20 per chain: GRAVY (Kyte-Doolittle), aromaticity, aliphatic index, isoelectric point,
net charge at pH 7 per residue, instability index, log molecular weight, mean residue mass, molar extinction
coefficient per residue (reduced cysteines), helix, turn and sheet propensity fractions, mean flexibility
(Vihinen scale, window 9), and the fractions of positive (KR), negative (DE), polar uncharged (STNQ), aliphatic
(AVILM), aromatic-or-histidine (FWYH), glycine-or-proline, and cysteine residues. The instability index and the
flexibility depend on residue order; the others are functions of composition and length.

Scoring uses extract/score2.py unchanged: the script points that module at `extract/ref_variants/`, so the features,
the per-task parts and the merged tables of this analysis stay in that folder and the released score tables
(`extract/scores2_*.csv`) are not rewritten.

  python scripts/downstream_reference_variants.py            # from protcompass/: build, score, summarise
  python scripts/downstream_reference_variants.py --summary  # rebuild summary tables from the existing scores
  python scripts/downstream_reference_variants.py --latex    # print the rows of SI Table S10, panel (b)

Needs `extract/embeddings/<task>/ref_composition/ids_*.txt` (the id lists of the released reference), the task
manifests and Biopython.
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_downstream_reference import AA, featurize as composition_features  # noqa: E402

ROOT = Path(PROTCOMPASS_ROOT) / "protcompass"
EXT = ROOT / "extract"
VAR = EXT / "ref_variants"
VARIANTS = ["ref_composition", "ref_physchem", "ref_comp_physchem", "ref_comp_dipeptide",
            "ref_comp_physchem_dipeptide"]
PHYS_NAMES = ["gravy", "aromaticity", "aliphatic_index", "isoelectric_point", "charge_pH7_per_residue",
              "instability_index", "log_molecular_weight", "mean_residue_mass", "extinction_per_residue",
              "helix_fraction", "turn_fraction", "sheet_fraction", "mean_flexibility", "frac_positive_KR",
              "frac_negative_DE", "frac_polar_STNQ", "frac_aliphatic_AVILM", "frac_aromatic_FWYH", "frac_GP",
              "frac_C"]
DIPEP = {a + b: i for i, (a, b) in enumerate((a, b) for a in AA for b in AA)}
STANDARD = set(AA)


def physchem_features(seqs):
    X = np.zeros((len(seqs), len(PHYS_NAMES)), dtype=np.float32)
    for r, s in enumerate(seqs):
        s = "".join(ch for ch in str(s) if ch in STANDARD)
        n = max(len(s), 1)
        if not s:
            continue
        pa = ProteinAnalysis(s)
        f = {a: s.count(a) / n for a in AA}
        mw = pa.molecular_weight()
        flex = pa.flexibility() if len(s) >= 9 else []
        helix, turn, sheet = pa.secondary_structure_fraction()
        X[r] = [pa.gravy(), pa.aromaticity(), 100.0 * (f["A"] + 2.9 * f["V"] + 3.9 * (f["I"] + f["L"])),
                pa.isoelectric_point(), pa.charge_at_pH(7.0) / n, pa.instability_index(), np.log(mw), mw / n,
                pa.molar_extinction_coefficient()[0] / n, helix, turn, sheet,
                float(np.mean(flex)) if len(flex) else 0.0,
                f["K"] + f["R"], f["D"] + f["E"], f["S"] + f["T"] + f["N"] + f["Q"],
                f["A"] + f["V"] + f["I"] + f["L"] + f["M"], f["F"] + f["W"] + f["Y"] + f["H"], f["G"] + f["P"],
                f["C"]]
    return X


def dipeptide_features(seqs):
    X = np.zeros((len(seqs), 400), dtype=np.float32)
    for r, s in enumerate(seqs):
        s = str(s)
        for a, b in zip(s[:-1], s[1:]):
            i = DIPEP.get(a + b)
            if i is not None:
                X[r, i] += 1.0
        X[r] /= max(len(s) - 1, 1)
    return X


def variant_matrices(seqs):
    comp = composition_features(seqs)                    # 23 columns; columns 20 and 21 are log L and L/600
    phys, dip = physchem_features(seqs), dipeptide_features(seqs)
    return {"ref_composition": comp,
            "ref_physchem": np.concatenate([phys, comp[:, 20:22]], 1),
            "ref_comp_physchem": np.concatenate([comp, phys], 1),
            "ref_comp_dipeptide": np.concatenate([comp, dip], 1),
            "ref_comp_physchem_dipeptide": np.concatenate([comp, phys, dip], 1)}


def build(task):
    src = EXT / "embeddings" / task / "ref_composition"
    man = pd.read_csv(EXT / "manifest" / f"{task}.csv")
    seq = man.drop_duplicates("id").set_index("id").sequence
    splits = ["chains"] if (src / "ids_chains.txt").exists() else ["train", "valid", "test"]
    dims = {}
    for split in splits:
        ids = (src / f"ids_{split}.txt").read_text().split()
        v = seq.reindex(ids)
        if v.isna().any():
            raise SystemExit(f"{task}: {int(v.isna().sum())} of {len(ids)} ids have no sequence in the manifest")
        for name, X in variant_matrices(v.tolist()).items():
            out = VAR / "embeddings" / task / name
            out.mkdir(parents=True, exist_ok=True)
            np.save(out / f"{split}.npy", X)
            (out / f"ids_{split}.txt").write_text("\n".join(ids) + "\n")
            dims[name] = X.shape[1]
    for name, d in dims.items():
        (VAR / "embeddings" / task / name / "meta.json").write_text(json.dumps({
            "encoder": name, "family": "non-learned reference", "weights": "none", "n_parameters": 0,
            "inputs_passed": "amino-acid sequence only", "hidden_size": d, "task": task,
            "physchem_columns": PHYS_NAMES if "physchem" in name else [],
            "date": time.strftime("%Y-%m-%d %H:%M")}, indent=1))
    print(f"built {task:26s} " + " ".join(f"{k}={v}" for k, v in dims.items()), flush=True)


def score():
    """Run extract/score2.py on the variants with its paths redirected to extract/ref_variants/."""
    for name in ("manifest", "pairs"):
        link = VAR / name
        if not link.exists():
            link.symlink_to(Path("..") / name)
    spec = importlib.util.spec_from_file_location("score2", EXT / "score2.py")
    s2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s2)
    s2.HERE, s2.PARTS = VAR, VAR / "score_parts"
    argv, sys.argv = sys.argv, ["score2.py", "--encoders", *VARIANTS]
    try:
        s2.main()
    finally:
        sys.argv = argv
    return s2.TASKS


def summarise():
    sel = pd.read_csv(VAR / "scores2_selected.csv")
    long = pd.read_csv(VAR / "scores2_long.csv")
    probe = long.groupby(["task", "encoder", "probe"]).val_loss.mean().reset_index()
    probe = probe.loc[probe.groupby(["task", "encoder"]).val_loss.idxmin()].set_index(["task", "encoder"]).probe
    metric = long.drop_duplicates("task").set_index("task").metric
    wide = pd.read_csv(EXT / "scores2_wide.csv", index_col=0)
    released = wide.loc["ref_composition"]
    enc = wide.drop(index=[i for i in wide.index if i.startswith("ref_")])
    rows = []
    for (task, name), g in sel.groupby(["task", "encoder"]):
        e = enc[task].dropna()
        s = float(g.score.iloc[0])
        rows.append(dict(task=task, metric=metric[task], reference=name, score=s, seed_sd=float(g.seed_sd.iloc[0]),
                         probe=probe[(task, name)], n_encoders=len(e), median_encoder=round(float(e.median()), 4),
                         best_encoder=round(float(e.max()), 4), encoders_above=int((e > s).sum()),
                         median_minus_reference=round(float(e.median()) - s, 4),
                         best_minus_reference=round(float(e.max()) - s, 4),
                         released_ref_composition=float(released[task])))
    out = pd.DataFrame(rows)
    out["reference"] = pd.Categorical(out.reference, VARIANTS, ordered=True)
    out = out.sort_values(["task", "reference"])
    out.to_csv(VAR / "summary.csv", index=False)
    w = out.pivot(index="task", columns="reference", values="score")[VARIANTS]
    w["median_encoder"], w["best_encoder"] = enc.median().reindex(w.index), enc.max().reindex(w.index)
    w.round(4).to_csv(VAR / "summary_wide.csv")
    chk = out[out.reference == "ref_composition"]
    print("\nrecomputed ref_composition minus released score, per task:")
    print((chk.set_index("task").score - chk.set_index("task").released_ref_composition).round(4).to_string())
    print("\nscores:")
    print(w.round(3).to_string())
    above = out.pivot(index="task", columns="reference", values="encoders_above")[VARIANTS]
    print("\nencoders above each reference:")
    print(above.to_string())


LATEX_TASKS = [("ec_classification", "EC classification"), ("go_mf", "GO-MF"), ("go_bp", "GO-BP"),
               ("subcellular_localization", "Localisation"), ("remote_homology", "Remote homology"),
               ("solubility", "Solubility"), ("membrane_soluble", "Membrane/soluble"), ("stability", "Stability"),
               ("fluorescence", "Fluorescence"), ("ppi_affinity", "PPI affinity")]


def latex():
    """Rows of the reference-variant panel: five scores, then encoders above composition + dipeptide."""
    s = pd.read_csv(VAR / "summary.csv").set_index(["task", "reference"])
    for task, label in LATEX_TASKS:
        cells = [f"{s.loc[(task, v), 'score']:.3f}".replace("-", "$-$") for v in VARIANTS]
        r = s.loc[(task, "ref_comp_dipeptide")]
        print(f"{label} & " + " & ".join(cells) + f" & {int(r.encoders_above)}/{int(r.n_encoders)} \\\\")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true", help="only rebuild the summary tables")
    ap.add_argument("--latex", action="store_true", help="print the LaTeX rows from the existing summary")
    a = ap.parse_args()
    if a.latex:
        return latex()
    if not a.summary:
        tasks = sorted(d.name for d in (EXT / "embeddings").iterdir()
                       if (d / "ref_composition" / "meta.json").exists())
        for t in tasks:
            build(t)
        score()
    summarise()


if __name__ == "__main__":
    main()
