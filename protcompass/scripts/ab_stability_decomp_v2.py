"""Conditional rows and per-parent scores for the stability composition decomposition.

Pre-specification: erasure/ab_stability_decomp/PRESPEC.md and PRESPEC_addendum.md. Same data, erasure, nulls,
permutations and head as scripts/ab_stability_decomp.py. The head is stability_competing.head, copied here so that
it also returns the Spearman inside every test parent.

  python ab_stability_decomp_v2.py --pass 1 --encoders ...     # rows (A) and (B): C1 then C0
  python ab_stability_decomp_v2.py --pass 2 --encoders ...     # the other PRESPEC fits
  python ab_stability_decomp_v2.py --report
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr
from sklearn.linear_model import RidgeCV

sys.path.insert(0, str(Path(__file__).parent))
import common_struct                                                    # noqa: E402
from addback_named_pairs import conditional                            # noqa: E402
from erasure import ALPHAS, concepts_for, leace_eraser, load_task       # noqa: E402
from matched_null import matched_null_projector, variance_removed       # noqa: E402
from panel import panel                                                 # noqa: E402
import ab_stability_decomp as base                                      # noqa: E402

TASK, NAME, DIR = base.TASK, base.NAME, base.DIR
NULL_SEEDS = base.NULL_SEEDS
COMP_ORDER, DESCS = base.COMP_ORDER, base.DESCS


def head(Xtr, ytr, Xva, yva, Xte, yte, parents):
    """stability_competing.head, line for line, plus the per-parent Spearman values it averages."""
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    A, C = (Xtr - mu) / sd, (Xte - mu) / sd
    try:
        m = RidgeCV(alphas=ALPHAS).fit(A, ytr)
    except np.linalg.LinAlgError:
        rng = np.random.default_rng(0)
        A = A + 1e-6 * rng.standard_normal(A.shape)
        m = RidgeCV(alphas=ALPHAS).fit(A, ytr)
    p = m.predict(C)
    pooled = float(spearmanr(yte, p)[0])
    within, per = [], {}
    for g in np.unique(parents):
        k = parents == g
        if k.sum() >= 20 and np.std(p[k]) > 0:
            v = spearmanr(yte[k], p[k])[0]
            within.append(v)
            per[g] = float(v)
    return pooled, float(np.mean(within)), per


def fits_for(pass_no, arm):
    """(label, [block names]) for one arm; blocks are appended after [arm representation, competitor block]."""
    if pass_no == 1:
        if arm == "intact":
            return []
        return [("cparent", ["cparent"]), ("dc", ["dc"]), ("cparent+dc", ["cparent", "dc"]),
                ("cparent+dc_perm", ["cparent", "dc_perm"]), ("dc+cparent_perm", ["dc", "cparent_perm"])]
    if arm == "intact":
        return [("none", []), ("full", ["full"])]
    keys = ["none", "full", "full_perm", "cparent_perm", "dc_perm", "cparent_seq", "cparent_seq_perm", "dc_seq",
            "dc_seq_perm"]
    return [(k, [] if k == "none" else [k]) for k in keys]


def run_cell(enc, pass_no, C, C0, C1, D):
    data = load_task(TASK, enc)
    (Xtr, ytr, itr), (Xva, yva, iva), (Xte, yte, ite) = (data[s] for s in ("train", "valid", "test"))
    ok = ~np.isnan(ytr)
    conc = concepts_for({"train": itr, "valid": iva, "test": ite})
    Z = conditional(conc, NAME, ok)
    X, y = Xtr[ok], ytr[ok]
    mu, P, r, B = leace_eraser(X, Z)
    assert r > 0 and B.shape[1] >= 8
    er = lambda W: ((W - mu) @ P.T) @ B
    arms = {"intact": (X, Xva, Xte), "erased": (er(X), er(Xva), er(Xte))}
    diag = {"intact": 0.0, "erased": variance_removed(X, er)}
    target = diag["erased"]
    for s in NULL_SEEDS:
        mun, Pn, _, Bn, _, _ = matched_null_projector(X, Z, r, target, seed=300 + s)
        en = lambda W, mun=mun, Pn=Pn, Bn=Bn: ((W - mun) @ Pn.T) @ Bn
        arms[f"null{s}"] = (en(X), en(Xva), en(Xte))
        diag[f"null{s}"] = variance_removed(X, en)
    ids = (np.asarray(itr)[ok], np.asarray(iva), np.asarray(ite))
    rng = np.random.default_rng(0)
    Qtr, Qva, Qte, _ = conc[NAME]
    Qtr = Qtr[ok]
    perm = (rng.permutation(len(Qtr)), rng.permutation(len(Qva)), rng.permutation(len(Qte)))
    cp = [f"cparent_{a}" for a in COMP_ORDER]
    cps = [f"cparent_seq_{a}" for a in COMP_ORDER]
    Q = (Qtr, Qva, Qte)
    blocks = {"full": Q,
              "cparent": tuple(D.loc[i, cp].to_numpy(float) for i in ids),
              "cparent_seq": tuple(D.loc[i, cps].to_numpy(float) for i in ids)}
    blocks["dc"] = tuple(q.astype(np.float64) - c for q, c in zip(Q, blocks["cparent"]))
    blocks["dc_seq"] = tuple(q.astype(np.float64) - c for q, c in zip(Q, blocks["cparent_seq"]))
    for k in DESCS:
        blocks[k + "_perm"] = tuple(v[p] for v, p in zip(blocks[k], perm))
    parents = C.loc[ids[2], "parent"].to_numpy()
    rows, prow = [], []
    order = (("C1", C1), ("C0", C0)) if pass_no == 1 else (("C0", C0), ("C1", C1))
    for tag, T in order:
        cov = tuple(T.loc[i].to_numpy(float) for i in ids)
        for arm, (A, Bv, Cv) in arms.items():
            for label, names in fits_for(pass_no, arm):
                parts = [(A, Bv, Cv), cov] + [blocks[n] for n in names]
                A2, B2, C2 = (np.hstack([p[j] for p in parts]) for j in range(3))
                pooled, within, per = head(A2, y, B2, yva, C2, yte, parents)
                rows.append(dict(encoder=enc, competitors=tag, arm=arm, descriptor=label, rank=r, dim=X.shape[1],
                                 score=pooled, score_within=within, var_removed=diag[arm]))
                prow += [dict(encoder=enc, competitors=tag, arm=arm, descriptor=label, parent=g, spearman=v)
                         for g, v in per.items()]
                print(f"  {enc:20s} {tag} {arm:7s} {label:16s} {pooled:.4f} {within:.4f}", flush=True)
    return rows, prow


# ----------------------------------------------------------------------------- report
ROWS = [(d, d, "none", d + "_perm") for d in DESCS] + [
    ("dc|cparent", "cparent+dc", "cparent", "cparent+dc_perm"),
    ("cparent|dc", "cparent+dc", "dc", "dc+cparent_perm")]


def load_v2(pattern):
    fs = sorted(DIR.glob(pattern))
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    return d


def contrast(p, named, none, arms_null):
    return (p.loc["erased", named] - p.loc["erased", none]) - (p.loc[arms_null, named] - p.loc[arms_null, none]).mean()


def report():
    from family_weighted_demand import family
    enc_panel = set(panel())
    d = load_v2("v2cells_*.csv").drop_duplicates(["encoder", "competitors", "arm", "descriptor"], keep="last")
    d = d[d.encoder.isin(enc_panel)]
    pp = load_v2("v2parent_*.csv").drop_duplicates(["encoder", "competitors", "arm", "descriptor", "parent"], keep="last")
    pp = pp[pp.encoder.isin(enc_panel)]

    # consistency with the main run where both exist
    main = base.cells()
    main = main[main.descriptor.isin(["none"] + DESCS + [k + "_perm" for k in DESCS])]
    j = d.merge(main, on=["encoder", "competitors", "arm", "descriptor"], suffixes=("", "_main"))
    cons = dict(n_overlap=len(j), max_d_score=float((j.score - j.score_main).abs().max()) if len(j) else np.nan,
                max_d_within=float((j.score_within - j.score_within_main).abs().max()) if len(j) else np.nan)
    print("v2 vs main run:", cons)

    E, P = [], []
    for (enc, cs), g in d.groupby(["encoder", "competitors"]):
        pv = {m: g.pivot_table(index="arm", columns="descriptor", values=m) for m in ("score", "score_within")}
        gp = pp[(pp.encoder == enc) & (pp.competitors == cs)]
        nulls = [a for a in pv["score"].index if a.startswith("null")]
        for row, named, none, permd in ROWS:
            cols = pv["score"].columns
            if not {named, none, permd} <= set(cols):
                continue
            rec = dict(encoder=enc, competitors=cs, row=row)
            for m, sfx in (("score", ""), ("score_within", "_within")):
                rec["R_var" + sfx] = contrast(pv[m], named, none, nulls)
                rec["Rperm" + sfx] = contrast(pv[m], permd, none, nulls)
            E.append(rec)
            for par, h in gp.groupby("parent"):
                q = h.pivot_table(index="arm", columns="descriptor", values="spearman")
                if {named, none} <= set(q.columns):
                    P.append(dict(encoder=enc, competitors=cs, row=row, parent=par,
                                  R_var=contrast(q, named, none, nulls)))
    E, P = pd.DataFrame(E), pd.DataFrame(P)
    E["family"] = E.encoder.map(family)
    E.to_csv(DIR / "v2_cells_summary.csv", index=False)
    P.to_csv(DIR / "parent_contrasts.csv", index=False)

    rng = np.random.default_rng(0)
    out = []
    for (row, cs), g in E.groupby(["row", "competitors"], sort=False):
        for m, sfx in (("pooled (primary)", ""), ("within-parent (secondary)", "_within")):
            v, vp = g["R_var" + sfx].to_numpy(), g["Rperm" + sfx].to_numpy()
            k, kp = int((v > 0).sum()), int((vp > 0).sum())
            fam = g.groupby("family")["R_var" + sfx].mean()
            kf = int((fam > 0).sum())
            rec = dict(metric=m, row=row, competitors=cs, n=len(v), R_var_mean=float(v.mean()), R_var_pos=k,
                       R_var_p=binomtest(k, len(v), 0.5).pvalue, fam_mean=float(fam.mean()), n_fam=len(fam),
                       fam_pos=kf, fam_p=binomtest(kf, len(fam), 0.5).pvalue, Rperm_mean=float(vp.mean()),
                       Rperm_pos=kp, Rperm_p=binomtest(kp, len(vp), 0.5).pvalue)
            if sfx:
                h = P[(P.row == row) & (P.competitors == cs)]
                per_parent = h.groupby("parent").R_var.mean()             # mean over encoders, per parent
                vals = per_parent.to_numpy()
                boots = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(2000)]
                rec.update(n_parents=len(vals), parent_mean=float(vals.mean()),
                           parent_ci_lo=float(np.percentile(boots, 2.5)), parent_ci_hi=float(np.percentile(boots, 97.5)),
                           parents_pos=int((vals > 0).sum()))
            out.append(rec)
    V = pd.DataFrame(out)
    V["family_set"] = np.where(V.row.str.contains(r"\|"), "addendum", "original")
    V["holm_own"] = np.nan
    V["holm_all14"] = np.nan
    for (m, fs), g in V.groupby(["metric", "family_set"]):
        V.loc[g.index, "holm_own"] = base.holm(g.R_var_p.to_numpy())
    for m, g in V.groupby("metric"):
        V.loc[g.index, "holm_all14"] = base.holm(g.R_var_p.to_numpy())

    def verdict(r):
        if r.holm_own >= 0.05:
            return "not detected"
        if r.R_var_mean < 0:
            return "negative"
        return "carries repair" if r.R_var_mean >= 0.02 else "attenuated"
    V["verdict"] = V.apply(verdict, axis=1)
    V.to_csv(DIR / "v2_verdicts.csv", index=False)
    pd.Series(cons).to_csv(DIR / "v2_consistency_with_main.csv", header=["value"])
    pd.set_option("display.width", 300)
    print(V.round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="pass_no", type=int, choices=[1, 2])
    ap.add_argument("--encoders", nargs="+", default=None)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        return report()
    common_struct.set_task(TASK)
    C, C0, C1, D = base.load_tables()
    dest, pdest = DIR / f"v2cells_p{a.pass_no}.csv", DIR / f"v2parent_p{a.pass_no}.csv"
    rows = pd.read_csv(dest).to_dict("records") if dest.exists() else []
    prow = pd.read_csv(pdest).to_dict("records") if pdest.exists() else []
    done = {r["encoder"] for r in rows}
    for e in a.encoders:
        if e in done:
            continue
        r_, p_ = run_cell(e, a.pass_no, C, C0, C1, D)
        rows += r_
        prow += p_
        pd.DataFrame(rows).to_csv(dest, index=False)
        pd.DataFrame(prow).to_csv(pdest, index=False)
    print("written", dest)


if __name__ == "__main__":
    main()
