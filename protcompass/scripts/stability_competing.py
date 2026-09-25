"""The stability repair contrast for amino-acid composition with simple competing explanations in every arm.

Pre-specification: manuscript/ICLR/QuestionDrivenVersion/stability_competing/PRESPEC.md (written before any result).
Arms, eraser, shams, variance-matched nulls, head and descriptor are those of scripts/addback_named_pairs.py; the
changes are a block of competing covariates C appended to every arm (computed in the same run as the arms without
C) and a second test metric, the within-parent Spearman, read from the same fitted heads.

  python stability_competing.py [--encoders ...] [--out cells_part.csv]
  python stability_competing.py --report
Outputs under erasure/stability_competing/.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr
from sklearn.linear_model import RidgeCV

sys.path.insert(0, str(Path(__file__).parent))
import common_struct                                                    # noqa: E402
from addback_named_pairs import conditional                            # noqa: E402
from erasure import ALPHAS, OUT, ROOT, concepts_for, fit_score, leace_eraser, load_task   # noqa: E402
from matched_null import matched_null_projector, property_share_removed, variance_removed   # noqa: E402
from panel import panel                                                 # noqa: E402

TASK, NAME = "stability", "amino-acid composition"
DIR = OUT / "stability_competing"
SHAM_SEEDS, NULL_SEEDS = (0, 1, 2, 3), (0, 1, 2, 3)
AA = "ACDEFGHIKLMNPQRSTVWY"
HYD, AROM = set("AVILMFWC"), set("FWY")
CHARGE = {**{a: 1 for a in "KR"}, **{a: -1 for a in "DE"}}
EDIT_BINS = [0, 1, 2, 3, 6, 11, 21]            # {0, 1, 2, 3-5, 6-10, 11-20, 21+}
TOPO, ROUNDS = ["EEHEE", "EHEE", "HEEH", "HHH"], ["rd1", "rd2", "rd3", "rd4"]
KINDS = ["original", "point", "hp", "random", "PG_hp", "buryD", "other"]


def parse(n):
    mm = re.match(r"^(.+)_([ACDEFGHIKLMNPQRSTVWY]\d+[ACDEFGHIKLMNPQRSTVWY])$", n)
    if mm:
        return mm.group(1), "point"
    mm = re.match(r"^(.*?\.pdb)(?:_(.+))?$", n)
    if mm:
        return mm.group(1), (mm.group(2) or "original")
    mm = re.match(r"^(.+?)_(PG_hp.*|hp|random.*|buryD.*)$", n)
    if mm:
        return mm.group(1), mm.group(2)
    return n, "original"


def kind_class(k):
    for c in ("original", "point", "hp", "buryD"):
        if k == c:
            return c
    if k.startswith("random"):
        return "random"
    if k.startswith("PG_hp"):
        return "PG_hp"
    return "other"


def covariates():
    f = DIR / "covariates.csv"
    if f.exists():
        return pd.read_csv(f, index_col=0)
    from Bio.Align import PairwiseAligner
    al = PairwiseAligner(mode="global", match_score=0, mismatch_score=-1, open_gap_score=-1, extend_gap_score=-1)
    m = pd.read_csv(ROOT / "extract" / "manifest" / f"{TASK}.csv")
    m["name"] = m.id.str.split("__").str[0]
    P = m.name.apply(parse)
    m["parent"], m["kind"] = P.str[0], P.str[1]
    seq_of = m.drop_duplicates("name").set_index("name").sequence
    rows = {}
    for r in m[m.stage1 == 1].itertuples():
        s = r.sequence
        par = seq_of.get(r.parent)
        missing = par is None or not isinstance(par, str)
        par = s if missing else par
        a = al.align(par, s)[0]
        subs = []
        for (p0, p1), (s0, s1) in zip(*a.aligned):
            subs += [(par[p0 + k], s[s0 + k]) for k in range(p1 - p0) if par[p0 + k] != s[s0 + k]]
        gaps = len(par) - sum(p1 - p0 for p0, p1 in a.aligned[0]) + len(s) - sum(s1 - s0 for s0, s1 in a.aligned[1])
        n = len(subs) + gaps
        rec = {"n_edit": n, "n_edit_sq": n * n}
        for k, lo in enumerate(EDIT_BINS):
            hi = EDIT_BINS[k + 1] if k + 1 < len(EDIT_BINS) else 10 ** 6
            rec[f"bin_{lo}"] = float(lo <= n < hi)
        rec.update(hyd_to_non=sum(x in HYD and y not in HYD for x, y in subs),
                   non_to_hyd=sum(x not in HYD and y in HYD for x, y in subs),
                   to_pro=sum(y == "P" for _, y in subs), from_pro=sum(x == "P" for x, _ in subs),
                   to_gly=sum(y == "G" for _, y in subs), from_gly=sum(x == "G" for x, _ in subs),
                   charge_altering=sum(CHARGE.get(x, 0) != CHARGE.get(y, 0) for x, y in subs),
                   net_charge_change=sum(CHARGE.get(y, 0) - CHARGE.get(x, 0) for x, y in subs),
                   arom_gained=sum(x not in AROM and y in AROM for x, y in subs),
                   arom_lost=sum(x in AROM and y not in AROM for x, y in subs), n_gaps=gaps)
        rec["mean_plddt"] = r.mean_plddt
        topo = re.match(r"^([A-Za-z]*)", r.parent).group(1)
        for t in TOPO:
            rec[f"topo_{t}"] = float(topo == t)
        rec["topo_other"] = float(topo not in TOPO)
        rd = re.search(r"_(rd\d)_", r.parent)
        for x in ROUNDS:
            rec[f"round_{x}"] = float(bool(rd) and rd.group(1) == x)
        rec["round_none"] = float(not rd)
        kc = kind_class(r.kind)
        for k in KINDS:
            rec[f"kind_{k}"] = float(kc == k)
        rec["parent_missing"] = float(missing)
        rec["parent"] = r.parent
        rows[r.id] = rec
    C = pd.DataFrame.from_dict(rows, orient="index")
    DIR.mkdir(parents=True, exist_ok=True)
    C.to_csv(f)
    return C


def cov_cols(C):
    return [c for c in C.columns if c != "parent"]


def head(Xtr, ytr, Xva, yva, Xte, yte, parents):
    """The released regression head (erasure.fit_score, kind 'reg'), returning both test metrics."""
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
    within = []
    for g in np.unique(parents):
        k = parents == g
        if k.sum() >= 20 and np.std(p[k]) > 0:
            within.append(spearmanr(yte[k], p[k])[0])
    return pooled, float(np.mean(within))


def run_cell(enc, C):
    data = load_task(TASK, enc)
    if data is None:
        return []
    (Xtr, ytr, itr), (Xva, yva, iva), (Xte, yte, ite) = (data[s] for s in ("train", "valid", "test"))
    ok = ~np.isnan(ytr)
    conc = concepts_for({"train": itr, "valid": iva, "test": ite})
    if NAME not in conc:
        return []
    Z = conditional(conc, NAME, ok)
    if Z is None:
        return []
    X, y = Xtr[ok], ytr[ok]
    mu, P, r, B = leace_eraser(X, Z)
    if r == 0 or B.shape[1] < 8:
        return []
    er = lambda W: ((W - mu) @ P.T) @ B
    arms = {"intact": (X, Xva, Xte), "erased": (er(X), er(Xva), er(Xte))}
    diag = {"intact": (0.0, 0.0), "erased": (variance_removed(X, er), property_share_removed(X, Z, er))}
    for s in SHAM_SEEDS:
        rs = np.random.default_rng(100 + s)
        mus, Ps, rr, Bs = leace_eraser(X, Z[rs.permutation(len(Z))])
        if rr == 0 or Bs.shape[1] < 8:
            continue
        es = lambda W, mus=mus, Ps=Ps, Bs=Bs: ((W - mus) @ Ps.T) @ Bs
        arms[f"sham{s}"] = (es(X), es(Xva), es(Xte))
        diag[f"sham{s}"] = (variance_removed(X, es), property_share_removed(X, Z, es))
    target = diag["erased"][0]
    for s in NULL_SEEDS:
        mun, Pn, _, Bn, _, _ = matched_null_projector(X, Z, r, target, seed=300 + s)
        en = lambda W, mun=mun, Pn=Pn, Bn=Bn: ((W - mun) @ Pn.T) @ Bn
        arms[f"null{s}"] = (en(X), en(Xva), en(Xte))
        diag[f"null{s}"] = (variance_removed(X, en), property_share_removed(X, Z, en))
    rng = np.random.default_rng(0)
    Qtr, Qva, Qte, _ = conc[NAME]
    Qtr = Qtr[ok]
    desc = {"named": (Qtr, Qva, Qte),
            "permuted": (Qtr[rng.permutation(len(Qtr))], Qva[rng.permutation(len(Qva))], Qte[rng.permutation(len(Qte))])}
    cols = cov_cols(C)
    cov = (C.loc[np.asarray(itr)[ok], cols].to_numpy(float), C.loc[iva, cols].to_numpy(float),
           C.loc[ite, cols].to_numpy(float))
    parents = C.loc[ite, "parent"].to_numpy()
    rows = []
    for use_c in (True, False):
        for arm, (A, Bv, Cv) in arms.items():
            for key in [None, "named", "permuted"]:
                if arm == "intact" and key == "permuted":
                    continue
                parts = [(A, Bv, Cv)] + ([cov] if use_c else []) + ([desc[key]] if key else [])
                A2, B2, C2 = (np.hstack([p[j] for p in parts]) for j in range(3))
                pooled, within = head(A2, y, B2, yva, C2, yte, parents)
                rows.append(dict(task=TASK, concept=NAME, encoder=enc, covariates="C" if use_c else "none", arm=arm,
                                 descriptor=key or "none", rank=r, dim=X.shape[1], score=pooled, score_within=within,
                                 var_removed=diag[arm][0], prop_removed=diag[arm][1]))
                print(f"  {enc:20s} {'C' if use_c else '-'} {arm:7s} {(key or 'none'):8s} {pooled:.4f} {within:.4f}", flush=True)
    # the head reproduces the released one on one arm
    ref = fit_score(arms["erased"][0], y, arms["erased"][1], yva, arms["erased"][2], yte, "reg")
    got = [x["score"] for x in rows if x["covariates"] == "none" and x["arm"] == "erased" and x["descriptor"] == "none"][0]
    assert abs(ref - got) < 1e-9, (enc, ref, got)
    return rows


def no_representation(C):
    data = load_task(TASK, "esm2_8m")
    (_, ytr, itr), (_, yva, iva), (_, yte, ite) = (data[s] for s in ("train", "valid", "test"))
    ok = ~np.isnan(ytr)
    conc = concepts_for({"train": itr, "valid": iva, "test": ite})
    Qtr, Qva, Qte, _ = conc[NAME]
    cols = cov_cols(C)
    c = (C.loc[np.asarray(itr)[ok], cols].to_numpy(float), C.loc[iva, cols].to_numpy(float), C.loc[ite, cols].to_numpy(float))
    par = C.loc[ite, "parent"].to_numpy()
    out = {}
    for tag, parts in (("C", [c]), ("C_plus_q", [c, (Qtr[ok], Qva, Qte)]), ("q", [(Qtr[ok], Qva, Qte)])):
        A2, B2, C2 = (np.hstack([p[j] for p in parts]) for j in range(3))
        out[f"S_{tag}"], out[f"S_{tag}_within"] = head(A2, ytr[ok], B2, yva, C2, yte, par)
    return out


def per_encoder(d):
    rows = []
    for enc, g in d.groupby("encoder"):
        rec = dict(encoder=enc)
        for cv, sfx in (("none", ""), ("C", "_C")):
            for metric, msfx in (("score", ""), ("score_within", "_within")):
                p = g[g.covariates == cv].pivot_table(index="arm", columns="descriptor", values=metric)
                for fam in ("null", "sham"):
                    arms = [a for a in p.index if a.startswith(fam)]
                    rec[f"R_{'var' if fam == 'null' else 'sham'}{sfx}{msfx}"] = \
                        (p.loc["erased", "named"] - p.loc["erased", "none"]) - (p.loc[arms, "named"] - p.loc[arms, "none"]).mean()
                nulls = [a for a in p.index if a.startswith("null")]
                rec[f"Rperm{sfx}{msfx}"] = (p.loc["erased", "permuted"] - p.loc["erased", "none"]) - \
                    (p.loc[nulls, "permuted"] - p.loc[nulls, "none"]).mean()
                rec[f"inc{sfx}{msfx}"] = p.loc["intact", "named"] - p.loc["intact", "none"]
                rec[f"S_intact{sfx}{msfx}"] = p.loc["intact", "none"]
        rows.append(rec)
    return pd.DataFrame(rows)


def report():
    from family_weighted_demand import family
    from panel import only_panel
    d = pd.concat([pd.read_csv(f) for f in sorted(DIR.glob("cells*.csv"))], ignore_index=True)
    d = d.drop_duplicates(["encoder", "covariates", "arm", "descriptor"], keep="last")
    E = only_panel(per_encoder(d))
    E["family"] = E.encoder.map(family)
    E.to_csv(DIR / "cells_summary.csv", index=False)
    out = []
    for metric in ("", "_within"):
        rec = dict(metric="pooled Spearman (primary)" if not metric else "within-parent Spearman (secondary)", n=len(E))
        for col in ("R_var", "R_var_C", "R_sham", "R_sham_C", "Rperm_C", "inc_C"):
            v = E[col + metric]
            k = int((v > 0).sum())
            rec[f"{col}_mean"], rec[f"{col}_pos"], rec[f"{col}_p"] = float(v.mean()), k, binomtest(k, len(v), 0.5).pvalue
        fam = E.groupby("family")[["R_var" + metric, "R_var_C" + metric]].mean()
        for col in ("R_var", "R_var_C"):
            k = int((fam[col + metric] > 0).sum())
            rec[f"{col}_fam_pos"], rec["n_fam"], rec[f"{col}_fam_p"] = k, len(fam), binomtest(k, len(fam), 0.5).pvalue
        rec["ratio_C_to_none"] = rec["R_var_C_mean"] / rec["R_var_mean"] if rec["R_var_mean"] else np.nan
        rec["verdict"] = ("retained" if rec["R_var_C_p"] < 0.05 and rec["R_var_C_mean"] >= 0.02 else
                          "attenuated" if rec["R_var_C_p"] < 0.05 else "explained by the competitors")
        out.append(rec)
    V = pd.DataFrame(out)
    nr = DIR / "no_representation.csv"
    if nr.exists():
        for k, v in pd.read_csv(nr).iloc[0].items():
            V[k] = v
    V.to_csv(DIR / "verdicts.csv", index=False)
    pd.set_option("display.width", 250)
    print(V.round(4).T.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", default=None)
    ap.add_argument("--out", default="cells.csv")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        return report()
    common_struct.set_task(TASK)
    C = covariates()
    dest = DIR / a.out
    rows = pd.read_csv(dest).to_dict("records") if dest.exists() else []
    done = {r["encoder"] for r in rows}
    encs = a.encoders or [e for e in panel() if (ROOT / "extract" / "embeddings" / TASK / e / "meta.json").exists()]
    for e in encs:
        if e in done:
            continue
        rows += run_cell(e, C)
        pd.DataFrame(rows).to_csv(dest, index=False)
    nr = DIR / "no_representation.csv"
    if not nr.exists():
        pd.DataFrame([no_representation(C)]).to_csv(nr, index=False)
    print("written", dest)


if __name__ == "__main__":
    main()
