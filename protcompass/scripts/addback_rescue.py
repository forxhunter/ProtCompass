"""Does the named property selectively repair its own erasure, or does it repair any erasure equally?

Section 4.3 names amino-acid composition as the demand of membrane/soluble, solubility and subcellular localisation,
on the evidence that erasing its linearly accessible subspace costs the retrained head more than a sham erasure of
the same property permuted across proteins. That comparison shows the erasure did something the sham did not. It
does not show that what it did was specific to composition: the LEACE map also removes variance the property does
not carry, and a head deprived of that variance loses score for reasons that have nothing to do with the named
property.

The test that separates the two is repair. Concatenate an independently computed descriptor of the named property to
the erased representation and ask whether it recovers more of the score after the real erasure than after the sham:

    R = [ S(E_p X + q) - S(E_p X) ] - mean_pi [ S(E_pi X + q) - S(E_pi X) ]

with `+` denoting concatenation, every head retrained under one protocol, and every map and standardiser fitted on
the training split alone. R is positive when the descriptor repairs the real erasure more than it repairs sham damage
of the same rank. Three descriptors are compared at every arm: the named property, the same property permuted across
proteins (the same columns and marginal, no relation to the protein), and the other structural blocks. A named demand
that is real should show a positive R for the property and not for the permuted or alternative descriptors.

This is run on the same proteins and encoders as the rest of the paper, so it tests selectivity, not generalisation.
It is exploratory evidence about the naming rule and is not a confirmation experiment.

  python addback_rescue.py --tasks membrane_soluble solubility --encoders esm2_650m esm_if1
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, str(Path(__file__).parent))
import common_struct                                                    # noqa: E402
from common_struct import INDEPENDENT_BLOCKS                            # noqa: E402
from erasure import EMB, OUT, leace_eraser, load_task                   # noqa: E402
from matched_null import matched_null_projector, property_share_removed, variance_removed

TASKS = ["membrane_soluble", "solubility", "subcellular_localization"]
KIND = {"membrane_soluble": "binary", "solubility": "binary", "subcellular_localization": "clf"}
PROPERTY = "amino-acid composition"
ENCODERS = ["esm2_650m", "esm1b", "prottrans_t5_xl_u50", "saprot", "esm_if1", "proteinmpnn"]
SHAM_SEEDS = (0, 1, 2, 3)
NULL_SEEDS = (0, 1, 2, 3)
WITH_NULLS = False
C_GRID = (1e-2, 1e-1, 1.0)


# ---------------------------------------------------------------- heads


def _standardise(tr, *rest):
    mu, sd = tr.mean(0), tr.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return [(Z - mu) / sd for Z in (tr, *rest)]


def score_linear(Xtr, ytr, Xva, yva, Xte, yte, kind):
    """Logistic head, penalty chosen on validation; AUROC for binary, accuracy for multiclass."""
    A, B, C = _standardise(Xtr, Xva, Xte)
    best = None
    for c in C_GRID:
        m = LogisticRegression(C=c, max_iter=2000).fit(A, ytr)
        v = (roc_auc_score(yva, m.predict_proba(B)[:, 1]) if kind == "binary" else m.score(B, yva))
        if best is None or v > best[0]:
            best = (v, m)
    m = best[1]
    return float(roc_auc_score(yte, m.predict_proba(C)[:, 1]) if kind == "binary" else m.score(C, yte))


def score_mlp(Xtr, ytr, Xva, yva, Xte, yte, kind, seed=0):
    """Two-layer head, the family the downstream protocol selects in most cells."""
    A, B, C = _standardise(Xtr, Xva, Xte)
    m = MLPClassifier(hidden_layer_sizes=(128,), max_iter=400, random_state=seed,
                      early_stopping=True, n_iter_no_change=15).fit(A, ytr)
    return float(roc_auc_score(yte, m.predict_proba(C)[:, 1]) if kind == "binary" else m.score(C, yte))


# ---------------------------------------------------------------- descriptors


def descriptors(conc, ok, rng):
    """The descriptor blocks concatenated to a representation, all computed from the protein, never from the label.

    named       the property the erasure names, as the released descriptor block
    permuted    the same columns with the proteins shuffled inside each split: same width, same marginal, no protein
    other       every other independent structural block, as a prespecified alternative
    """
    Ztr, Zva, Zte, _ = conc[PROPERTY]
    Ztr = Ztr[ok]
    out = {"named": (Ztr, Zva, Zte)}
    out["permuted"] = (Ztr[rng.permutation(len(Ztr))], Zva[rng.permutation(len(Zva))], Zte[rng.permutation(len(Zte))])
    others = [c for c in INDEPENDENT_BLOCKS if c != PROPERTY and c in conc]
    if others:
        out["other"] = tuple(np.concatenate([conc[c][i] for c in others], 1) for i in (0, 1, 2))
        out["other"] = (out["other"][0][ok], out["other"][1], out["other"][2])
    return out


def conditional_target(conc, ok):
    """Composition residualised against the other independent blocks: exactly what the conditional arm erases."""
    Ztr, _, Zte, _ = conc[PROPERTY]
    others = [c for c in INDEPENDENT_BLOCKS if c != PROPERTY and c in conc]
    O = np.concatenate([conc[c][0] for c in others], 1)[ok]
    coef = np.linalg.lstsq(O, Ztr[ok], rcond=None)[0]
    Ote = np.concatenate([conc[c][2] for c in others], 1)
    return Ztr[ok] - O @ coef, Zte - Ote @ coef


# ---------------------------------------------------------------- one cell


def run_cell(task, enc, use_mlp):
    kind = KIND[task]
    data = load_task(task, enc)
    if data is None:
        return []
    (Xtr, ytr, itr), (Xva, yva, iva), (Xte, yte, ite) = (data[s] for s in ("train", "valid", "test"))
    ok = ~np.isnan(ytr)
    if ok.sum() < 200:
        return []
    Xtr, ytr = Xtr[ok], ytr[ok]
    conc = common_struct.protein_descriptors({"train": itr, "valid": iva, "test": ite}, task=task)
    if PROPERTY not in conc:
        return []
    Ztr, _ = conditional_target(conc, ok)
    rng = np.random.default_rng(0)
    desc = descriptors(conc, ok, rng)

    def fit(A, B, C):
        r = {"linear": score_linear(A, ytr, B, yva, C, yte, kind)}
        if use_mlp:
            r["mlp"] = score_mlp(A, ytr, B, yva, C, yte, kind)
        return r

    def with_desc(A, B, C, key):
        if key is None:
            return A, B, C
        d = desc[key]
        return (np.hstack([A, d[0]]), np.hstack([B, d[1]]), np.hstack([C, d[2]]))

    arms = {}
    mu, P, r, Bb = leace_eraser(Xtr, Ztr)
    if r == 0 or Bb.shape[1] < 8:
        return []
    er = lambda Z: ((Z - mu) @ P.T) @ Bb
    arms["intact"] = (Xtr, Xva, Xte)
    arms["erased"] = (er(Xtr), er(Xva), er(Xte))
    for s in SHAM_SEEDS:
        rs_ = np.random.default_rng(100 + s)
        Zs = Ztr[rs_.permutation(len(Ztr))]
        mus, Ps, rr, Bs = leace_eraser(Xtr, Zs)
        if rr == 0 or Bs.shape[1] < 8:
            continue
        es = lambda Z, mus=mus, Ps=Ps, Bs=Bs: ((Z - mus) @ Ps.T) @ Bs
        arms[f"sham{s}"] = (es(Xtr), es(Xva), es(Xte))
    diag = {"intact": (0.0, 0.0), "erased": (variance_removed(Xtr, er), property_share_removed(Xtr, Ztr, er))}
    if WITH_NULLS:
        # the null of scripts/matched_null.py: same rank and same share of variance as the erased arm, in
        # principal directions carrying as little of the property as the search finds
        target = diag["erased"][0]
        for s_ in NULL_SEEDS:
            mun, Pn, _, Bn, _, _ = matched_null_projector(Xtr, Ztr, r, target, seed=300 + s_)
            en = lambda Z, mun=mun, Pn=Pn, Bn=Bn: ((Z - mun) @ Pn.T) @ Bn
            arms[f"null{s_}"] = (en(Xtr), en(Xva), en(Xte))
            diag[f"null{s_}"] = (variance_removed(Xtr, en), property_share_removed(Xtr, Ztr, en))

    rows = []
    for arm, (A, B, C) in arms.items():
        for key in [None] + [k for k in ("named", "permuted", "other") if k in desc]:
            A2, B2, C2 = with_desc(A, B, C, key)
            sc = fit(A2, B2, C2)
            rows.append(dict(task=task, encoder=enc, arm=arm, descriptor=key or "none", rank=r,
                             dim=Xtr.shape[1], desc_dim=0 if key is None else desc[key][0].shape[1],
                             n_train=int(len(ytr)), n_test=int(len(yte)),
                             var_removed=diag.get(arm, (np.nan, np.nan))[0],
                             prop_removed=diag.get(arm, (np.nan, np.nan))[1], **sc))
            print(f"  {task:24s} {enc:20s} {arm:7s} {(key or 'none'):9s} "
                  + " ".join(f"{k} {v:.4f}" for k, v in sc.items()), flush=True)
    return rows


# ---------------------------------------------------------------- contrast


def contrasts(d, head="linear"):
    """R per (task, encoder, descriptor): repair of the real erasure minus mean repair of the sham erasures."""
    out = []
    for (task, enc), g in d.groupby(["task", "encoder"]):
        piv = g.pivot_table(index="arm", columns="descriptor", values=head)
        if "erased" not in piv.index or "none" not in piv.columns:
            continue
        shams = [a for a in piv.index if a.startswith("sham")]
        for q in [c for c in piv.columns if c != "none"]:
            real = piv.loc["erased", q] - piv.loc["erased", "none"]
            sham = np.array([piv.loc[a, q] - piv.loc[a, "none"] for a in shams], dtype=float)
            out.append(dict(task=task, encoder=enc, descriptor=q, head=head,
                            repair_real=real, repair_sham_mean=float(np.mean(sham)),
                            repair_sham_sd=float(np.std(sham, ddof=1)) if len(shams) > 1 else np.nan,
                            n_sham=len(shams), R=real - float(np.mean(sham)),
                            intact=piv.loc["intact", "none"] if "intact" in piv.index else np.nan,
                            erased=piv.loc["erased", "none"],
                            sham_none=float(np.mean([piv.loc[a, "none"] for a in shams]))))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--nulls", action="store_true",
                    help="also score the variance-matched null arms of scripts/matched_null.py")
    ap.add_argument("--encoders", nargs="+", default=ENCODERS)
    ap.add_argument("--mlp", action="store_true", help="also fit the two-layer head")
    ap.add_argument("--out", default=str(OUT / "addback_rescue.csv"))
    a = ap.parse_args()
    global WITH_NULLS
    WITH_NULLS = a.nulls

    rows = []
    for task in a.tasks:
        common_struct.set_task(task)
        for enc in a.encoders:
            if not (EMB / task / enc / "meta.json").exists():
                print(f"  [skip] {task}/{enc}: no extraction", flush=True)
                continue
            try:
                rows += run_cell(task, enc, a.mlp)
            except Exception as e:                                       # noqa: BLE001
                print(f"  [failed] {task}/{enc}: {type(e).__name__}: {e}", flush=True)
            pd.DataFrame(rows).to_csv(a.out, index=False)
    d = pd.DataFrame(rows)
    d.to_csv(a.out, index=False)
    print(f"\nwritten: {a.out}  ({len(d)} rows)")

    for head in (["linear", "mlp"] if a.mlp else ["linear"]):
        if head not in d:
            continue
        c = contrasts(d, head)
        c.to_csv(str(a.out).replace(".csv", f"_contrasts_{head}.csv"), index=False)
        print(f"\n=== selective repair, {head} head: R = real repair - mean sham repair ===")
        print(c.groupby(["task", "descriptor"]).R.agg(["mean", "std", "count"]).round(4).to_string())
        print("\nper encoder, named descriptor:")
        print(c[c.descriptor == "named"][["task", "encoder", "intact", "erased", "sham_none",
                                          "repair_real", "repair_sham_mean", "R"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
