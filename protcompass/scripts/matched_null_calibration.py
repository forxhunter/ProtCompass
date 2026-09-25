"""Calibrate the variance-matched null and the repair contrast on tasks whose answer is set in advance.

The positive control of Appendix sec:appendix_positive_control shows that the erasure effect against the sham
recovers a planted demand and stays quiet when nothing is planted. The naming rule now rests on a different
instrument, the repair contrast against the variance-matched null, and that instrument has not been checked the same
way. Two questions need an answer before its verdicts can be read:

  power and size    when a property is planted, does the repair contrast find it, at the sizes the real tasks show,
                    and does it stay near zero for the properties that were not planted?
  the null's reach  the variance-matched null removes high-variance directions that avoid the property. A task that
                    also uses such directions loses score to the null for reasons unrelated to the property. How far
                    does that move the effect against the null, and does the repair contrast survive it?

Labels are built on the stability proteins and twelve encoders, as in the released positive control, from the same
conditional residual the protocol erases. Thirteen labels are scored at once:

  strong / weak / faint <P>   y = sqrt(s) f(P) + sqrt(1 - s) noise, with s = 0.5, 0.05 and 0.01
  mixed <P>                   y = sqrt(0.05) f(P) + sqrt(0.45) g + sqrt(0.5) noise
  none                        y = sqrt(0.5) g + sqrt(0.5) noise

where f(P) is a fixed linear function of the conditional residual of property P, the same direction for every encoder,
and g is a direction of the representation with every property block projected out. A random direction of the
representation is dominated by its high-variance components, so g lives where the variance-matched null cuts: `none`
is the condition in which the null is harshest, and `mixed` puts a small demand for P inside a task that also uses g.

Every erasure map, sham and null depends on the representation and the property and not on the label, so each map is
built once per encoder and property and every label is scored against it in one multi-target ridge fit, with the
penalty chosen per target. That is the released regression head, fitted once instead of thirteen times, and the
script checks the equivalence on the first cell it scores.

  python matched_null_calibration.py [--encoders ...] [--out ...]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import common_struct                                                    # noqa: E402
from common_struct import INDEPENDENT_BLOCKS                            # noqa: E402
from erasure import ALPHAS, OUT, concepts_for, fit_score, leace_eraser, load_task, probe_concept   # noqa: E402
from matched_null import matched_null_projector, property_share_removed, variance_removed           # noqa: E402

TASK = "stability"
DEFAULT = ["esm2_650m", "esm2_150m", "esm1b", "ankh_base", "prottrans_t5_xl_u50", "prostt5",
           "esm_if1", "esm3", "saprot", "proteinmpnn", "gvp", "gearnet_pretrained"]
PLANTED = ["amino-acid composition", "secondary structure content", "global shape"]
SHAM_SEEDS = (0, 1, 2)
NULL_SEEDS = (0, 1, 2)
LEVELS = {"strong": 0.5, "weak": 0.05, "faint": 0.01}
MIXED_P, MIXED_G = 0.05, 0.45


def conditional_target(name, conc, split):
    """The property with the other independent blocks projected out, on one split (0 train, 1 valid, 2 test).

    The projection is fitted on the training split and applied to every split, as in the released positive control
    (scripts/erasure_positive_control.py), so a planted label is one fixed function of the protein. An earlier
    version of this script fitted the projection inside each split; on a property collinear with the others, global
    shape with size above all, the test label then differed from the function the head learned on the training
    split, and the planted demand became nearly unpredictable before anything was erased."""
    others = [c for c in INDEPENDENT_BLOCKS if c != name and c in conc]
    Z = conc[name][split]
    if not others:
        return Z
    coef = np.linalg.lstsq(np.concatenate([conc[c][0] for c in others], 1), conc[name][0], rcond=None)[0]
    O = np.concatenate([conc[c][split] for c in others], 1)
    return Z - O @ coef


def std(v):
    v = v - v.mean()
    return v / (v.std() + 1e-12)


def labels(X, conc):
    """The thirteen labels for one encoder, as (names, train, valid, test) with one column per label."""
    names, cols = [], {0: [], 1: [], 2: []}
    f = {}
    for P in PLANTED:
        r = np.random.default_rng(0)                  # the same planted direction for every encoder
        w = r.standard_normal(conc[P][0].shape[1])
        f[P] = {s: std(conditional_target(P, conc, s) @ w) for s in (0, 1, 2)}
    rg = np.random.default_rng(7)
    d = rg.standard_normal(X[0].shape[1])
    g = {}
    O0 = np.concatenate([conc[c][0] for c in conc], 1)
    cg = np.linalg.lstsq(O0, X[0] @ d, rcond=None)[0]         # fitted on train, applied to every split
    for s in (0, 1, 2):
        O = np.concatenate([conc[c][s] for c in conc], 1)
        g[s] = std(X[s] @ d - O @ cg)

    def noise(k, s):
        return np.random.default_rng(1000 * k + s).standard_normal(len(g[s]))

    k = 0
    for P in PLANTED:
        for lvl, a in LEVELS.items():
            names.append(f"{lvl} | {P}")
            for s in (0, 1, 2):
                cols[s].append(np.sqrt(a) * f[P][s] + np.sqrt(1 - a) * noise(k, s))
            k += 1
        names.append(f"mixed | {P}")
        for s in (0, 1, 2):
            cols[s].append(np.sqrt(MIXED_P) * f[P][s] + np.sqrt(MIXED_G) * g[s]
                           + np.sqrt(1 - MIXED_P - MIXED_G) * noise(k, s))
        k += 1
    names.append("none")
    for s in (0, 1, 2):
        cols[s].append(np.sqrt(0.5) * g[s] + np.sqrt(0.5) * noise(k, s))
    return names, *(np.stack(cols[s], 1) for s in (0, 1, 2))


def fit_multi(Xtr, Ytr, Xte, Yte):
    """The released regression head (standardise on train, RidgeCV over the released grid, Spearman on test), with
    one penalty per target, so each column is scored exactly as a separate fit would score it."""
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    A, C = (Xtr - mu) / sd, (Xte - mu) / sd
    try:
        m = RidgeCV(alphas=ALPHAS, alpha_per_target=True).fit(A, Ytr)
    except np.linalg.LinAlgError:
        A = A + 1e-6 * np.random.default_rng(0).standard_normal(A.shape)
        m = RidgeCV(alphas=ALPHAS, alpha_per_target=True).fit(A, Ytr)
    P = m.predict(C)
    return np.array([spearmanr(Yte[:, j], P[:, j])[0] for j in range(Yte.shape[1])])


CHECKED = False


def one_encoder(enc):
    global CHECKED
    data = load_task(TASK, enc)
    if data is None:
        return []
    (Xtr, _, itr), (Xva, _, iva), (Xte, _, ite) = (data[s] for s in ("train", "valid", "test"))
    conc = concepts_for({"train": itr, "valid": iva, "test": ite})
    names, Ytr, Yva, Yte = labels((Xtr, Xva, Xte), conc)
    rows = []
    for name in conc:
        Zt, Zt_te = conditional_target(name, conc, 0), conditional_target(name, conc, 2)
        mu, P, r, B = leace_eraser(Xtr, Zt)
        if r == 0 or B.shape[1] < 8:
            continue
        er = lambda W: ((W - mu) @ P.T) @ B
        cb = probe_concept(Xtr, Zt, Xte, Zt_te, conc[name][3])
        ca = probe_concept(er(Xtr), Zt, er(Xte), Zt_te, conc[name][3])
        arms = {"erased": (er(Xtr), er(Xte))}
        vr = {"erased": variance_removed(Xtr, er)}
        for s in SHAM_SEEDS:
            gsh = np.random.default_rng(s)
            mus, Ps, rs, Bs = leace_eraser(Xtr, Zt[gsh.permutation(len(Zt))])
            if rs == 0 or Bs.shape[1] < 8:
                continue
            es = lambda W, mus=mus, Ps=Ps, Bs=Bs: ((W - mus) @ Ps.T) @ Bs
            arms[f"sham{s}"] = (es(Xtr), es(Xte))
            vr[f"sham{s}"] = variance_removed(Xtr, es)
        for s in NULL_SEEDS:
            mun, Pn, _, Bn, _, _ = matched_null_projector(Xtr, Zt, r, vr["erased"], seed=300 + s)
            en = lambda W, mun=mun, Pn=Pn, Bn=Bn: ((W - mun) @ Pn.T) @ Bn
            arms[f"null{s}"] = (en(Xtr), en(Xte))
            vr[f"null{s}"] = variance_removed(Xtr, en)
        q = {"named": (conc[name][0], conc[name][2])}
        pr = np.random.default_rng(0)
        q["permuted"] = (conc[name][0][pr.permutation(len(conc[name][0]))],
                         conc[name][2][pr.permutation(len(conc[name][2]))])
        for arm, (A, C) in arms.items():
            for key in (None, "named", "permuted"):
                A2, C2 = (A, C) if key is None else (np.hstack([A, q[key][0]]), np.hstack([C, q[key][1]]))
                sc = fit_multi(A2, Ytr, C2, Yte)
                if not CHECKED:
                    # one separate fit through the released head must reproduce the matching column
                    j = names.index("strong | amino-acid composition")
                    ref = fit_score(A2, Ytr[:, j], C2, Yte[:, j], C2, Yte[:, j], "reg")   # the regression branch ignores validation
                    if abs(ref - sc[j]) > 1e-6:
                        raise SystemExit(f"multi-target fit disagrees with fit_score: {sc[j]:.6f} vs {ref:.6f}")
                    print(f"  checked: multi-target column equals the released head ({ref:.6f})", flush=True)
                    CHECKED = True
                for lab, v in zip(names, sc):
                    rows.append(dict(encoder=enc, concept=name, label=lab, arm=arm, descriptor=key or "none",
                                     score=float(v), rank=r, var_removed=vr[arm],
                                     concept_before=cb, concept_after=ca, erasure_ok=bool(ca < 0.5 * cb)))
        print(f"  {enc:20s} {name:28s} rank {r:3d}  var erased {vr['erased']:.3f}  "
              f"null {np.mean([vr[k] for k in vr if k.startswith('null')]):.3f}  "
              f"erasure {'ok' if ca < 0.5 * cb else 'not verified'}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", default=DEFAULT)
    ap.add_argument("--out", default=str(OUT / "matched_null_calibration.csv"))
    a = ap.parse_args()
    common_struct.set_task(TASK)
    rows = []
    for enc in a.encoders:
        rows += one_encoder(enc)
        pd.DataFrame(rows).to_csv(a.out, index=False)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"\nwritten: {a.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
