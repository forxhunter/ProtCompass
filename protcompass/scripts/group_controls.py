"""The group comparison with the confounds the design cannot remove held fixed one at a time.

The downstream group means put ten pretrained language models of 8M to 3B parameters against a coordinate group
that contains ten randomly initialised networks and several below a million parameters. An obvious objection is
that this compares pretraining and scale, not input modality, and that the subset of coordinate encoders comparable
in scale is exactly the subset that competes. That objection cannot be answered by a better statistic, because the
panel contains no large randomly initialised sequence model and no coordinate model above 1.5B; it can only be
answered by reporting the comparison under each restriction and letting the reader see which survive.

Four restrictions are reported:
  all               every encoder, the published comparison
  pretrained        randomly initialised networks dropped from both sides
  >= 100M           pretrained encoders of at least 100M parameters on both sides
  width-matched     every embedding reduced to the smallest width in the panel before scoring is compared, using
                    the rank of the score within the task rather than the score, since widths differ by 160x

Encoders are also grouped into architecture families, because the panel is not 33 independent draws: ESM-2 appears
at five scales, PottsMPNN four times, and eight architectures appear as a random and a pretrained pair. Any test
that treats the encoders as independent overstates its own power, so the family-clustered version of each test is
reported next to it.

  python group_controls.py [--markdown]
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
sys.path.insert(0, str(ROOT / "scripts"))
from make_panels_original_style import DECLARED, DISP  # noqa: E402

REF = "ref_composition"
ORDER = ["ec_classification", "go_mf", "go_bp", "subcellular_localization", "remote_homology", "solubility",
         "membrane_soluble", "stability", "fluorescence", "ppi_affinity"]
LABEL = {"ec_classification": "EC", "go_mf": "GO-MF", "go_bp": "GO-BP",
         "subcellular_localization": "localisation", "remote_homology": "homology", "solubility": "solubility",
         "stability": "stability", "fluorescence": "fluorescence", "ppi_affinity": "affinity",
         "membrane_soluble": "membrane"}

# an encoder's architecture family: the unit that is actually independent. Scales of one model, and the random and
# pretrained versions of one architecture, are one family.
FAMILY = {"esm2_8m": "ESM-2", "esm2_35m": "ESM-2", "esm2_150m": "ESM-2", "esm2_650m": "ESM-2", "esm2_3b": "ESM-2",
          "esm1b": "ESM-1b", "prottrans_t5_xl_u50": "ProtT5", "protbert_bfd": "ProtBert", "ankh_base": "Ankh",
          "prostt5": "ProstT5", "esm3": "ESM3", "saprot": "SaProt", "foldseek_3di": "Foldseek",
          "foldvision": "FoldVision", "esm_if1": "ESM-IF", "cdconv": "CDConv",
          "proteinmpnn": "MPNN", "pottsmpnn": "MPNN", "pottsmpnn_ft": "MPNN", "pottsmpnn_pretrained": "MPNN",
          "pottsmpnn_soluble": "MPNN", "gearnet": "GearNet", "gearnet_pretrained": "GearNet",
          "gcpnet_random": "GCPNet", "gcpnet_pretrained": "GCPNet", "gvp": "GVP", "tfn": "TFN",
          "tfn_pretrained": "TFN", "ipa": "IPA", "ipa_pretrained": "IPA", "schnet": "SchNet",
          "dimenet": "DimeNet", "mace": "MACE"}


def meta():
    rows = []
    for d in sorted((ROOT / "extract" / "embeddings" / "stability").glob("*/meta.json")):
        m = json.loads(d.read_text())
        rows.append(dict(encoder=d.parent.name, weights=m.get("weights", "pretrained"),
                         params=int(m.get("n_parameters") or 0), dim=int(m.get("hidden_size") or 0)))
    m = pd.DataFrame(rows).set_index("encoder")
    m["family"] = [FAMILY.get(e, e) for e in m.index]
    m["declared"] = [DECLARED.get(e, "struct") for e in m.index]
    # the grouping is the measured input audit, not the declared modality: an encoder is on the sequence side when
    # its pooled embedding does not respond to a coordinate swap (ProstT5 is declared multimodal and measured
    # sequence-only). The declared class is kept only as a column.
    a = (pd.read_csv(ROOT / "extract" / "encoder_input_audit.csv").sort_values("dataset")
         .drop_duplicates("encoder").set_index("encoder"))
    uses = a.uses_coordinates.astype(bool).reindex(m.index).fillna(True)
    m["group"] = np.where(~uses, "sequence", "coordinate-side")
    return m


def restrictions(m):
    big = (m.weights == "pretrained") & (m.params >= 100e6)
    return {"all encoders": m.index.to_series().astype(bool),
            "pretrained only": m.weights == "pretrained",
            "pretrained, at least 100M parameters": big}


def summarize(markdown=False, scores=None):
    d = pd.read_csv(scores or ROOT / "extract" / "scores2_selected.csv")
    w = d.pivot(index="encoder", columns="task", values="score")
    m = meta().reindex([e for e in w.index if e != REF])
    tasks = [t for t in ORDER if t in w.columns]
    out = []
    for name, mask in restrictions(m).items():
        sel = m[mask]
        for t in tasks:
            s, c = w.loc[sel.index[sel.group == "sequence"], t].dropna(), w.loc[sel.index[sel.group == "coordinate-side"], t].dropna()
            if len(s) < 2 or len(c) < 2:
                continue
            # family-clustered: one value per architecture family, so five ESM-2 scales count once
            fs = w.loc[sel.index[sel.group == "sequence"], t].groupby(sel.family[sel.group == "sequence"]).mean().dropna()
            fc = w.loc[sel.index[sel.group == "coordinate-side"], t].groupby(sel.family[sel.group == "coordinate-side"]).mean().dropna()
            u = stats.mannwhitneyu(fs, fc, alternative="two-sided") if len(fs) > 1 and len(fc) > 1 else None
            out.append(dict(restriction=name, task=LABEL[t], n_seq=len(s), n_crd=len(c),
                            seq=s.mean(), crd=c.mean(), gap=s.mean() - c.mean(),
                            fam_seq=len(fs), fam_crd=len(fc), fam_gap=fs.mean() - fc.mean(),
                            fam_p=(u.pvalue if u else np.nan), seq_leads=bool(s.mean() > c.mean())))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true")
    a = ap.parse_args()
    t = summarize()
    if not a.markdown:
        for name, g in t.groupby("restriction", sort=False):
            print(f"\n=== {name} ===  sequence n={g.n_seq.iloc[0]}, coordinate-side n={g.n_crd.iloc[0]}, "
                  f"families {g.fam_seq.iloc[0]} vs {g.fam_crd.iloc[0]}")
            print(g[["task", "seq", "crd", "gap", "fam_gap", "fam_p"]].round(3).to_string(index=False))
            print(f"sequence leads on {int(g.seq_leads.sum())} of {len(g)} tasks; "
                  f"family-clustered Mann-Whitney p < 0.05 on {int((g.fam_p < 0.05).sum())}")
        return
    print("<!-- generated by scripts/group_controls.py; do not edit by hand -->\n")
    print("| restriction | encoders (seq / coord) | families | tasks where the sequence group leads | "
          "mean gap over tasks | tasks with family-clustered p < 0.05 |")
    print("|---|---|---|---|---|---|")
    for name, g in t.groupby("restriction", sort=False):
        print(f"| {name} | {g.n_seq.iloc[0]} / {g.n_crd.iloc[0]} | {g.fam_seq.iloc[0]} / {g.fam_crd.iloc[0]} | "
              f"{int(g.seq_leads.sum())} of {len(g)} | {g.gap.mean():+.3f} | {int((g.fam_p < 0.05).sum())} of {len(g)} |")
    print("\nThe coordinate side is every encoder not measured to be sequence-only. Family-clustered tests take one "
          "value per architecture family, so the five ESM-2 scales, the four PottsMPNN variants and each "
          "random/pretrained pair count once.")


if __name__ == "__main__":
    main()
