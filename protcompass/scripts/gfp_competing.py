"""The fluorescence repair contrast with simple competing explanations in every arm.

Pre-specification: manuscript/ICLR/QuestionDrivenVersion/gfp_competing/PRESPEC.md (written before any result).
The arms, eraser, variance-matched nulls, head and descriptor are those of scripts/addback_named_pairs.py; the only
change is a block of competing covariates C appended to every arm, so the repair of property p is measured as

    R|C = [S(E_p X + C + q) - S(E_p X + C)] - mean_null [S(E_c X + C + q) - S(E_c X + C)].

C (per variant, from the sequence and the manifest): mutation burden (edits to the avGFP parent, its square, one-hot
bins), substitution-class counts, composition change (burial and packing only; for composition it is the descriptor
up to an affine map), and the mean pLDDT of the predicted structure.

  python gfp_competing.py --concept "burial and packing" [--encoders ...] [--recompute-nocov]
  python gfp_competing.py --report
Outputs under erasure/gfp_competing/.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common_struct                                                    # noqa: E402
from addback_named_pairs import conditional                            # noqa: E402
from erasure import OUT, ROOT, concepts_for, fit_score, leace_eraser, load_task   # noqa: E402
from matched_null import matched_null_projector, property_share_removed, variance_removed   # noqa: E402
from panel import panel                                                 # noqa: E402

TASK = "fluorescence"
DIR = OUT / "gfp_competing"
NULL_SEEDS = (0, 1, 2, 3)
AA = "ACDEFGHIKLMNPQRSTVWY"
HYD, AROM = set("AVILMFWC"), set("FWY")
CHARGE = {**{a: 1 for a in "KR"}, **{a: -1 for a in "DE"}}
BINS = [0, 1, 2, 3, 4, 5, 6, 8]          # bins {0,1,2,3,4,5,6-7,8+}


def parent():
    m = pd.read_csv(ROOT / "extract" / "manifest" / f"{TASK}.csv")
    m = m[(m.stage1 == 1) & (m.split == "train")]
    L = int(m.length.mode()[0])
    S = [s for s in m.sequence if len(s) == L]
    return "".join(max(set(c), key=c.count) for c in zip(*S))


def covariates():
    f = DIR / "covariates.csv"
    if f.exists():
        return pd.read_csv(f, index_col=0)
    from Bio.Align import PairwiseAligner
    al = PairwiseAligner(mode="global", match_score=0, mismatch_score=-1, open_gap_score=-1, extend_gap_score=-1)
    par = parent()
    m = pd.read_csv(ROOT / "extract" / "manifest" / f"{TASK}.csv")
    m = m[m.stage1 == 1].set_index("id")
    rows = {}
    pc = np.array([par.count(a) for a in AA], float)
    for i, r in m.iterrows():
        s = r.sequence
        a = al.align(par, s)[0]
        subs, gaps = [], 0
        for (p0, p1), (s0, s1) in zip(*a.aligned):
            subs += [(par[p0 + k], s[s0 + k]) for k in range(p1 - p0) if par[p0 + k] != s[s0 + k]]
        gaps = len(par) - sum(p1 - p0 for p0, p1 in a.aligned[0]) + len(s) - sum(s1 - s0 for s0, s1 in a.aligned[1])
        n = len(subs) + gaps
        rec = {"n_mut": n, "n_mut_sq": n * n}
        for k, lo in enumerate(BINS):
            hi = BINS[k + 1] if k + 1 < len(BINS) else 10 ** 6
            rec[f"bin_{lo}"] = float(lo <= n < hi)
        rec.update(hyd_to_non=sum(x in HYD and y not in HYD for x, y in subs),
                   non_to_hyd=sum(x not in HYD and y in HYD for x, y in subs),
                   to_pro=sum(y == "P" for _, y in subs), from_pro=sum(x == "P" for x, _ in subs),
                   to_gly=sum(y == "G" for _, y in subs), from_gly=sum(x == "G" for x, _ in subs),
                   charge_altering=sum(CHARGE.get(x, 0) != CHARGE.get(y, 0) for x, y in subs),
                   net_charge_change=sum(CHARGE.get(y, 0) - CHARGE.get(x, 0) for x, y in subs),
                   arom_gained=sum(x not in AROM and y in AROM for x, y in subs),
                   arom_lost=sum(x in AROM and y not in AROM for x, y in subs), n_gaps=gaps)
        for j, aa in enumerate(AA):
            rec[f"dcomp_{aa}"] = s.count(aa) - pc[j]
        rec["mean_plddt"] = r.mean_plddt
        rows[i] = rec
    C = pd.DataFrame.from_dict(rows, orient="index")
    DIR.mkdir(parents=True, exist_ok=True)
    C.to_csv(f)
    return C


def competing_set(C, concept):
    cols = ([c for c in C.columns if c.startswith(("n_mut", "bin_"))] +
            ["hyd_to_non", "non_to_hyd", "to_pro", "from_pro", "to_gly", "from_gly", "charge_altering",
             "net_charge_change", "arom_gained", "arom_lost", "n_gaps"] + ["mean_plddt"])
    if concept == "burial and packing":
        cols += [c for c in C.columns if c.startswith("dcomp_")]
    return cols


def run_cell(name, enc, C, recompute_nocov=False):
    data = load_task(TASK, enc)
    if data is None:
        return []
    (Xtr, ytr, itr), (Xva, yva, iva), (Xte, yte, ite) = (data[s] for s in ("train", "valid", "test"))
    ok = ~np.isnan(ytr)
    conc = concepts_for({"train": itr, "valid": iva, "test": ite})
    if name not in conc:
        return []
    Z = conditional(conc, name, ok)
    if Z is None:
        return []
    X, y = Xtr[ok], ytr[ok]
    mu, P, r, B = leace_eraser(X, Z)
    if r == 0 or B.shape[1] < 8:
        return []
    er = lambda W: ((W - mu) @ P.T) @ B
    arms = {"intact": (X, Xva, Xte), "erased": (er(X), er(Xva), er(Xte))}
    diag = {"intact": (0.0, 0.0), "erased": (variance_removed(X, er), property_share_removed(X, Z, er))}
    target = diag["erased"][0]
    for s in NULL_SEEDS:
        mun, Pn, _, Bn, _, _ = matched_null_projector(X, Z, r, target, seed=300 + s)
        en = lambda W, mun=mun, Pn=Pn, Bn=Bn: ((W - mun) @ Pn.T) @ Bn
        arms[f"null{s}"] = (en(X), en(Xva), en(Xte))
        diag[f"null{s}"] = (variance_removed(X, en), property_share_removed(X, Z, en))
    rng = np.random.default_rng(0)
    Qtr, Qva, Qte, _ = conc[name]
    Qtr = Qtr[ok]
    desc = {"named": (Qtr, Qva, Qte),
            "permuted": (Qtr[rng.permutation(len(Qtr))], Qva[rng.permutation(len(Qva))], Qte[rng.permutation(len(Qte))])}
    cols = competing_set(C, name)
    cov = (C.loc[np.asarray(itr)[ok], cols].to_numpy(float), C.loc[iva, cols].to_numpy(float),
           C.loc[ite, cols].to_numpy(float))
    rows = []
    for use_c in ([True, False] if recompute_nocov else [True]):
        for arm, (A, Bv, Cv) in arms.items():
            for key in [None, "named", "permuted"]:
                if arm == "intact" and key == "permuted":
                    continue
                if arm == "intact" and not use_c and key is not None:
                    continue
                parts = [(A, Bv, Cv)] + ([cov] if use_c else []) + ([desc[key]] if key else [])
                A2, B2, C2 = (np.hstack([p[j] for p in parts]) for j in range(3))
                sc = fit_score(A2, y, B2, yva, C2, yte, "reg")
                rows.append(dict(task=TASK, concept=name, encoder=enc, covariates="C" if use_c else "none", arm=arm,
                                 descriptor=key or "none", rank=r, dim=X.shape[1], score=sc,
                                 var_removed=diag[arm][0], prop_removed=diag[arm][1]))
                print(f"  {name[:22]:22s} {enc:20s} {'C' if use_c else '-'} {arm:7s} {(key or 'none'):8s} {sc:.4f}", flush=True)
    return rows


def no_representation(C, name):
    """S(C) and S(C + q): the covariates and the descriptor with no representation at all."""
    data = load_task(TASK, "esm2_8m")
    (_, ytr, itr), (_, yva, iva), (_, yte, ite) = (data[s] for s in ("train", "valid", "test"))
    ok = ~np.isnan(ytr)
    conc = concepts_for({"train": itr, "valid": iva, "test": ite})
    Qtr, Qva, Qte, _ = conc[name]
    cols = competing_set(C, name)
    c = (C.loc[np.asarray(itr)[ok], cols].to_numpy(float), C.loc[iva, cols].to_numpy(float), C.loc[ite, cols].to_numpy(float))
    s0 = fit_score(c[0], ytr[ok], c[1], yva, c[2], yte, "reg")
    s1 = fit_score(np.hstack([c[0], Qtr[ok]]), ytr[ok], np.hstack([c[1], Qva]), yva, np.hstack([c[2], Qte]), yte, "reg")
    sq = fit_score(Qtr[ok], ytr[ok], Qva, yva, Qte, yte, "reg")
    return dict(concept=name, S_C=s0, S_C_plus_q=s1, S_q=sq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept")
    ap.add_argument("--encoders", nargs="+", default=None)
    ap.add_argument("--recompute-nocov", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        import gfp_competing_report
        return gfp_competing_report.main()
    common_struct.set_task(TASK)
    C = covariates()
    tag = {"amino-acid composition": "comp", "burial and packing": "burial"}[a.concept]
    dest = DIR / (f"cells_{tag}" + ("_repro" if a.recompute_nocov else "") + ".csv")
    rows = pd.read_csv(dest).to_dict("records") if dest.exists() else []
    done = {r["encoder"] for r in rows}
    encs = a.encoders or [e for e in panel() if (ROOT / "extract" / "embeddings" / TASK / e / "meta.json").exists()]
    for e in encs:
        if e in done:
            continue
        rows += run_cell(a.concept, e, C, a.recompute_nocov)
        pd.DataFrame(rows).to_csv(dest, index=False)
    nr = DIR / f"no_representation_{tag}.csv"
    if not nr.exists():
        pd.DataFrame([no_representation(C, a.concept)]).to_csv(nr, index=False)
    print("written", dest)


if __name__ == "__main__":
    main()
