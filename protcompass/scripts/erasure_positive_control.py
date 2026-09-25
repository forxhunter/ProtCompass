"""Does the interventional protocol recover a demand that was planted, and stay quiet when none was planted?

The protocol of section 4.3 reports which property a task demands. Nothing in the paper showed that it can find a
demand whose answer is known, or that it does not invent one. This builds tasks whose answer is known and runs them
through the identical path: the same LEACE erasure, the same sham null over permutation seeds, the same conditional
arm, the same sign test over encoders, the same magnitude floor.

Four synthetic labels are built on the stability proteins, using the same eight concept blocks the real sweep uses:

  planted <property>   y = the label's own projection of that property block, plus noise at a fixed signal ratio.
                       The protocol should name that property and no other.
  planted none         y = a direction of the representation that is orthogonal to every concept block, plus noise.
                       Nothing should be named. This gives the false-positive rate of the rule.

Because the label is a function of the property and not of the encoder, every encoder sees the same label, and the
sign test over encoders means the same thing it means in the real sweep.

  python erasure_positive_control.py --encoders esm2_650m esm_if1 ...      (default: 12 encoders, mixed classes)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import common_struct
from common_struct import INDEPENDENT_BLOCKS
from causal_demand_v2 import MIN_EFFECT, holm
from erasure import EMB, OUT, concepts_for, fit_score, leace_eraser, load_task, probe_concept
from erasure_v2 import SHAM_SEEDS

TASK = "stability"          # the task whose proteins and encoders carry the synthetic labels
SNR = 0.5                   # share of the label's variance that is the planted property
NONLINEAR = False
DEFAULT = ["esm2_650m", "esm2_150m", "esm1b", "ankh_base", "prottrans_t5_xl_u50", "prostt5",
           "esm_if1", "esm3", "saprot", "proteinmpnn", "gvp", "gearnet_pretrained"]


def planted_label(Z, rng, snr=SNR):
    """A label that is a known linear function of the property block, at a fixed signal-to-noise ratio."""
    w = rng.standard_normal(Z.shape[1])
    s = Z @ w
    s = (s - s.mean()) / (s.std() + 1e-12)
    e = rng.standard_normal(len(s))
    return np.sqrt(snr) * s + np.sqrt(1 - snr) * e


def orthogonal_label(X, conc, split_rows, rng, snr=SNR):
    """A label built from a representation direction with every concept block projected out: nothing to find."""
    O = np.concatenate([conc[c][split_rows] for c in conc], 1)
    d = rng.standard_normal(X.shape[1])
    s = X @ d
    coef = np.linalg.lstsq(O, s, rcond=None)[0]
    s = s - O @ coef
    s = (s - s.mean()) / (s.std() + 1e-12)
    e = rng.standard_normal(len(s))
    return np.sqrt(snr) * s + np.sqrt(1 - snr) * e


def conditional_target(name, conc, ok):
    """The conditional arm of the real sweep: the property with the other independent blocks projected out."""
    others = [c for c in INDEPENDENT_BLOCKS if c != name and c in conc]
    Ztr = conc[name][0][ok]
    if not others:
        return Ztr, conc[name][2]
    O = np.concatenate([conc[c][0] for c in others], 1)[ok]
    coef = np.linalg.lstsq(O, Ztr, rcond=None)[0]
    Ote = np.concatenate([conc[c][2] for c in others], 1)
    return Ztr - O @ coef, conc[name][2] - Ote @ coef


def conditional_residual_valid(name, conc):
    """The validation-split twin of `conditional_target`, fitted on the same split as its own block."""
    others = [c for c in INDEPENDENT_BLOCKS if c != name and c in conc]
    Z = conc[name][1]
    if not others:
        return Z
    O = np.concatenate([conc[c][1] for c in others], 1)
    return Z - O @ np.linalg.lstsq(O, Z, rcond=None)[0]


def one_encoder(enc, planted, rng):
    data = load_task(TASK, enc)
    if data is None:
        return []
    (Xtr, _, itr), (Xva, _, iva), (Xte, _, ite) = (data[s] for s in ("train", "valid", "test"))
    conc = concepts_for({"train": itr, "valid": iva, "test": ite})
    ok = np.ones(len(Xtr), bool)
    if planted == "none":
        ytr = orthogonal_label(Xtr, {c: conc[c][0] for c in conc}, slice(None), rng)
        yva = orthogonal_label(Xva, {c: conc[c][1] for c in conc}, slice(None), rng)
        yte = orthogonal_label(Xte, {c: conc[c][2] for c in conc}, slice(None), rng)
    else:
        # The label is planted on the same quantity the protocol erases: the property with the other independent
        # blocks projected out. Planting it on the raw block instead tests something else, because the raw blocks
        # are correlated with one another and the conditional arm is defined on the residual. That version was run
        # first and named burial and packing when global shape was planted, which is the correct answer to the
        # question it asks and not to the one the paper asks.
        r = np.random.default_rng(0)                      # the same planted direction for every encoder
        Ztr_c, Zte_c = conditional_target(planted, conc, ok)
        Zva_c = conditional_residual_valid(planted, conc)
        w = r.standard_normal(Ztr_c.shape[1])
        v = r.standard_normal(Ztr_c.shape[1])

        def lab(Z, seed):
            if NONLINEAR:
                # a product of two directions and a squared term: the label is a function of the same property
                # block, but no linear readout of that block predicts it
                a1, a2 = Z @ w, Z @ v
                s = a1 * a2 + a1 ** 2 - (a1 ** 2).mean()
            else:
                s = Z @ w
            s = (s - s.mean()) / (s.std() + 1e-12)
            e = np.random.default_rng(seed).standard_normal(len(s))
            return np.sqrt(SNR) * s + np.sqrt(1 - SNR) * e
        ytr, yva, yte = lab(Ztr_c, 1), lab(Zva_c, 2), lab(Zte_c, 3)

    base = fit_score(Xtr, ytr, Xva, yva, Xte, yte, "reg")
    rows = []
    for name in conc:
        Zt, Zt_te = conditional_target(name, conc, ok)
        mu, P, r, B = leace_eraser(Xtr, Zt)
        if r == 0 or B.shape[1] < 8:
            continue
        er = lambda Z: ((Z - mu) @ P.T) @ B
        erased = fit_score(er(Xtr), ytr, er(Xva), yva, er(Xte), yte, "reg")
        sham = []
        for s in SHAM_SEEDS:
            g = np.random.default_rng(s)
            mus, Ps, rs, Bs = leace_eraser(Xtr, Zt[g.permutation(len(Zt))])
            if rs == 0 or Bs.shape[1] < 8:
                continue
            es = lambda Z: ((Z - mus) @ Ps.T) @ Bs
            sham.append(fit_score(es(Xtr), ytr, es(Xva), yva, es(Xte), yte, "reg"))
        cb = probe_concept(Xtr, Zt, Xte, Zt_te, conc[name][3])
        ca = probe_concept(er(Xtr), Zt, er(Xte), Zt_te, conc[name][3])
        rows.append(dict(planted=planted, encoder=enc, concept=name, base=base, erased=erased,
                         concept_before=cb, concept_after=ca,
                         sham=float(np.mean(sham)) if sham else np.nan,
                         effect=(float(np.mean(sham)) if sham else np.nan) - erased,
                         erasure_ok=bool(ca < 0.5 * cb)))
        print(f"  [{planted[:18]:18s}] {enc:20s} {name:28s} effect {rows[-1]['effect']:+.3f}"
              f"{'' if rows[-1]['erasure_ok'] else '  (erasure not verified)'}", flush=True)
    return rows


def verdict(d, gate=True):
    """The same rule the paper applies: sign test over encoders, Holm correction, magnitude floor.

    `gate` applies the erasure check the real sweep applies. It is reported both ways because the check is strict
    enough to discard a true positive: on the planted runs the erasure of the planted property cuts its readout by
    just under half, which the check reads as a failure although the label's score falls by 0.7."""
    out = []
    for planted, g in d.groupby("planted"):
        g = g[g.erasure_ok] if gate else g
        ps, meta = [], []
        for concept, gg in g.groupby("concept"):
            k, n = int((gg.effect > 0).sum()), len(gg)
            ps.append(binomtest(k, n, 0.5).pvalue if n else 1.0)
            meta.append((concept, gg.effect.mean(), n))
        for (concept, mean, n), a in zip(meta, holm(ps)):
            out.append(dict(planted=planted, concept=concept, mean_effect=mean, n=n, p_holm=a,
                            named=bool(a < 0.05 and mean >= MIN_EFFECT)))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", default=DEFAULT)
    ap.add_argument("--planted", nargs="+",
                    default=["amino-acid composition", "secondary structure content", "global shape", "none"])
    ap.add_argument("--nonlinear", action="store_true",
                    help="plant a nonlinear function of the property instead of a linear one. The linear control "
                         "shows the protocol recovers a linearly carried demand; it says nothing about a demand a "
                         "task uses nonlinearly, which is what the exclusions need.")
    a = ap.parse_args()
    global NONLINEAR
    NONLINEAR = a.nonlinear
    common_struct.set_task(TASK)
    rng = np.random.default_rng(0)
    rows = []
    for planted in a.planted:
        print(f"\n=== planted: {planted} ===", flush=True)
        for enc in a.encoders:
            rows += one_encoder(enc, planted, rng)
        pd.DataFrame(rows).to_csv(OUT / f"positive_control{'_nonlinear' if a.nonlinear else ''}.csv", index=False)
    d = pd.DataFrame(rows)
    for gate in (True, False):
        v = verdict(d, gate)
        v["erasure_gate"] = gate
        v.to_csv(OUT / f"positive_control{'_nonlinear' if a.nonlinear else ''}_verdict"
                       f"{'' if gate else '_ungated'}.csv", index=False)
        print(f"\n=== what the rule names, erasure gate {'on' if gate else 'off'} ===")
        for planted, g in v.groupby("planted"):
            named = sorted(g.concept[g.named])
            print(f"  planted {planted:28s} -> named: {', '.join(named) if named else 'nothing'}")


if __name__ == "__main__":
    main()
