"""The naming rule with one vote per architecture family instead of one per encoder.

The panel is not 33 independent draws. Ten of the 23 coordinate encoders are randomly initialised counterparts of
pretrained networks that are also in the panel, ESM-2 appears at five scales, and PottsMPNN at four. The sign test
behind every named demand treats these as equally weighted observations, which overstates the
evidence for whichever effect the repeated architectures happen to carry.

This script recomputes the rule with the encoders first averaged inside an architecture family and the test then run
over families. The family is the architecture, so a random and a pretrained run of one network count once, as do the
five ESM-2 scales. The rule itself is unchanged: a sign test over the units at Holm-corrected p < 0.05 within a task,
a mean effect of at least MIN_EFFECT, and the property must vary on the task.

  python family_weighted_demand.py [--markdown]
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
sys.path.insert(0, str(ROOT / "scripts"))
from exclusion_bounds import TASK_LABEL, load                            # noqa: E402

MIN_EFFECT = 0.02
ALPHA = 0.05

# architecture family: the unit we are willing to treat as one observation. Random and pretrained runs of the same
# network, and the scales of one language model, are one family.
def family(enc):
    e = enc.lower()
    for pre, fam in (("esm2_", "ESM-2"), ("pottsmpnn", "PottsMPNN"), ("gcpnet", "GCPNet"), ("gearnet", "GearNet"),
                     ("ipa", "IPA"), ("tfn", "TFN"), ("prottrans", "ProtTrans"), ("protbert", "ProtTrans")):
        if e.startswith(pre):
            return fam
    return {"esm1b": "ESM-1b", "esm3": "ESM3", "saprot": "SaProt", "esm_if1": "ESM-IF", "prostt5": "ProstT5",
            "ankh_base": "Ankh", "foldseek_3di": "Foldseek", "foldvision": "FoldVision",
            "proteinmpnn": "ProteinMPNN", "gvp": "GVP", "schnet": "SchNet", "dimenet": "DimeNet++",
            "mace": "MACE", "cdconv": "CDConv"}.get(e, e)


def holm(pvals):
    """Holm-corrected p-values, returned in the input order."""
    order = np.argsort(pvals)
    m, out, running = len(pvals), np.empty(len(pvals)), 0.0
    for k, i in enumerate(order):
        running = max(running, (m - k) * pvals[i])
        out[i] = min(running, 1.0)
    return out


def sign_p(v):
    """Two-sided sign test that the effects are centred at zero."""
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v) & (v != 0)]
    if len(v) < 3:
        return np.nan, len(v), np.nan
    k = int((v > 0).sum())
    return float(binomtest(k, len(v), 0.5).pvalue), len(v), k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true")
    a = ap.parse_args()

    v2 = load("erasure_v2*.csv")
    cond = v2[(v2.arm == "conditional") & v2.erasure_ok.fillna(False)].copy()
    cond["family"] = [family(e) for e in cond.encoder]

    rows = []
    for task, g in cond.groupby("task"):
        per_enc, per_fam, names = {}, {}, []
        for concept, h in g.groupby("concept"):
            per_enc[concept] = h.effect.to_numpy()
            per_fam[concept] = h.groupby("family").effect.mean().to_numpy()
            names.append(concept)
        for unit, store in (("encoder", per_enc), ("family", per_fam)):
            ps, eff, ns, ks = [], [], [], []
            for c in names:
                p, n, k = sign_p(store[c])
                ps.append(p if np.isfinite(p) else 1.0)
                eff.append(float(np.mean(store[c])))
                ns.append(n)
                ks.append(k)
            hp = holm(np.array(ps))
            for c, p, h_, e, n, k in zip(names, ps, hp, eff, ns, ks):
                rows.append(dict(task=task, concept=c, unit=unit, n=n, n_pos=k, mean_effect=e,
                                 p=p, p_holm=h_, named=bool(h_ < ALPHA and e >= MIN_EFFECT)))
    t = pd.DataFrame(rows)
    t.to_csv(ROOT / "erasure" / "family_weighted_demand.csv", index=False)

    enc = t[t.unit == "encoder"].set_index(["task", "concept"])
    fam = t[t.unit == "family"].set_index(["task", "concept"])
    both = enc[["named", "mean_effect", "n", "p_holm"]].join(
        fam[["named", "mean_effect", "n", "p_holm"]], lsuffix="_enc", rsuffix="_fam")

    if not a.markdown:
        pd.set_option("display.width", 220)
        print(both[both.named_enc | both.named_fam].round(4).to_string())
        print(f"\nnamed over encoders: {int(both.named_enc.sum())} pairs; "
              f"over families: {int(both.named_fam.sum())} pairs")
        print(f"families: {cond.family.nunique()} over {cond.encoder.nunique()} encoders")
        lost = both[both.named_enc & ~both.named_fam]
        gained = both[~both.named_enc & both.named_fam]
        print(f"lost under family weighting: {list(lost.index)}")
        print(f"gained under family weighting: {list(gained.index)}")
        return

    print("% generated by scripts/family_weighted_demand.py; do not edit by hand")
    print("\\begin{tabular}{llrrrr}")
    print("\\toprule")
    print("task & property & effect (enc.) & Holm $p$ (enc.) & effect (fam.) & Holm $p$ (fam.) \\\\")
    print("\\midrule")
    for (task, concept), r in both[both.named_enc | both.named_fam].iterrows():
        print(f"{TASK_LABEL.get(task, task)} & {concept} & ${r.mean_effect_enc:+.3f}$ & "
              f"${r.p_holm_enc:.1e}$ & ${r.mean_effect_fam:+.3f}$ & ${r.p_holm_fam:.1e}$ \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print(f"% {cond.encoder.nunique()} encoders collapse to {cond.family.nunique()} architecture families")


if __name__ == "__main__":
    main()
