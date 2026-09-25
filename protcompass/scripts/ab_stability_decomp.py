"""Which part of amino-acid composition carries the stability repair: parent background or substitution?

Pre-specification: erasure/ab_stability_decomp/PRESPEC.md (sha256 in PRESPEC.sha256, written before any score).
The erasure, variance-matched nulls, head, encoders, splits and C0 are those of scripts/stability_competing.py.
Only the descriptor added back and the competitor block present in every arm change.

  python ab_stability_decomp.py --features                 # substitutions, descriptors, C1 site features
  python ab_stability_decomp.py --encoders ... --out cells_x.csv
  python ab_stability_decomp.py --check                    # reproduction gate against the released rows
  python ab_stability_decomp.py --alone                    # competitors and descriptors with no representation
  python ab_stability_decomp.py --report
Outputs under erasure/ab_stability_decomp/.
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
_HOME = _os.path.expanduser("~")
import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).parent))
import common_struct                                                    # noqa: E402
from addback_named_pairs import conditional                            # noqa: E402
from erasure import OUT, ROOT, concepts_for, leace_eraser, load_task    # noqa: E402
from matched_null import matched_null_projector, variance_removed       # noqa: E402
from panel import panel                                                 # noqa: E402
from stability_competing import head, parse                             # noqa: E402

TASK, NAME = "stability", "amino-acid composition"
DIR = OUT / "ab_stability_decomp"
REL = OUT / "stability_competing"
NULL_SEEDS = (0, 1, 2, 3)
AA1 = "ACDEFGHIKLMNPQRSTVWY"                         # order of the C1 wild-type / mutant counts
COMP_ORDER = "ARNDCQEGHILKMFPSTWYV"                  # order of the released composition block (common_struct.AA3)
DESCS = ["full", "cparent", "dc", "cparent_seq", "dc_seq"]
MKDSSP = (_HOME + "/miniconda3/envs/GeoBPE-6/bin/mkdssp")
MAX_ASA = {"ALA": 129, "ARG": 274, "ASN": 195, "ASP": 193, "CYS": 167, "GLN": 225, "GLU": 223, "GLY": 104,
           "HIS": 224, "ILE": 197, "LEU": 201, "LYS": 236, "MET": 224, "PHE": 240, "PRO": 159, "SER": 155,
           "THR": 172, "TRP": 285, "TYR": 263, "VAL": 174}
THREE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H",
         "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
         "TYR": "Y", "VAL": "V"}
DSSP8 = {"G": 0, "H": 1, "I": 2, "B": 3, "E": 4, "T": 5, "S": 6, " ": 7, "C": 7, "P": 7}
DSSP8_TO_3 = np.array([0, 0, 0, 1, 1, 2, 2, 2], dtype=np.int8)
TMP = Path((PROTCOMPASS_ROOT + "/protcompass/cache/scratch/dssp"))


def comp(s):
    v = np.zeros(20)
    for ch in s:
        if ch in COMP_ORDER:                     # non-standard letters are skipped, as common_struct skips them
            v[COMP_ORDER.index(ch)] += 1
    return v / max(v.sum(), 1)


# ----------------------------------------------------------------------------- substitutions and descriptors
def substitutions(par, s, kind, name, al):
    """[(site index in the variant, wild-type, mutant)] per PRESPEC: released alignment, point-variant fallback."""
    a = al.align(par, s)[0]
    subs = []
    for (p0, p1), (s0, s1) in zip(*a.aligned):
        subs += [(s0 + k, par[p0 + k], s[s0 + k]) for k in range(p1 - p0) if par[p0 + k] != s[s0 + k]]
    fixed = False
    if kind == "point":
        mm = re.match(r"^(.+)_([A-Z])(\d+)([A-Z])$", name)
        X, Y = mm.group(2), mm.group(4)
        if not (len(subs) == 1 and subs[0][1:] == (X, Y)):
            n = len(s)
            for seg in (par[:n], par[-n:] if n <= len(par) else ""):
                if len(seg) != n:
                    continue
                mis = [(i, seg[i], s[i]) for i in range(n) if seg[i] != s[i]]
                if len(mis) == 1 and mis[0][1:] == (X, Y):
                    subs, fixed = mis, True
                    break
    return subs, fixed


def dssp3(path, L):
    TMP.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".dssp", delete=False, dir=str(TMP)) as tmp:
        out = tmp.name
    try:
        rc = subprocess.run([MKDSSP, "-i", str(path), "-o", out], capture_output=True, timeout=120).returncode
        if rc != 0 or not os.path.getsize(out):
            return None
        ss = np.full(L, -1, dtype=np.int8)
        started = False
        for line in open(out):
            if line.startswith("  #  RESIDUE"):
                started = True
                continue
            if not started or len(line) < 17 or line[13] == "!":
                continue
            idx = int(line[5:10]) - 1
            if 0 <= idx < L:
                ss[idx] = DSSP8.get(line[16], 7)
        return np.where(ss >= 0, DSSP8_TO_3[np.clip(ss, 0, 7)], -1)
    except Exception:                                                   # noqa: BLE001
        return None
    finally:
        if os.path.exists(out):
            os.unlink(out)


def rsa(path, L):
    """Relative solvent accessibility per residue number and the structure's residue letter there."""
    import freesasa
    freesasa.setVerbosity(freesasa.silent)
    try:
        st = freesasa.Structure(str(path), options={"skip-unknown": True})
        res = freesasa.calc(st).residueAreas()
    except Exception:                                                   # noqa: BLE001
        return None, None
    out, name = np.full(L, np.nan), np.array([""] * L, dtype=object)
    chain = res.get("A") or (next(iter(res.values())) if res else {})
    for num, area in chain.items():
        idx = int(num.strip()) - 1
        if 0 <= idx < L and area.residueType in MAX_ASA:
            out[idx] = min(max(area.total / MAX_ASA[area.residueType], 0.0), 1.0)
            name[idx] = THREE[area.residueType]
    return out, name


def env_one(args):
    path, L, sites = args
    r, nm = rsa(path, L)
    ss = dssp3(path, L)
    vals = [r[i] for i in sites if r is not None and np.isfinite(r[i])]
    cls = [int(ss[i]) for i in sites if ss is not None and ss[i] >= 0]
    name_match = [nm[i] for i in sites] if nm is not None else None
    return (float(np.mean(vals)) if vals else np.nan, len(vals), [cls.count(k) for k in range(3)], len(cls), name_match)


def features():
    from Bio.Align import PairwiseAligner
    from multiprocessing import Pool
    al = PairwiseAligner(mode="global", match_score=0, mismatch_score=-1, open_gap_score=-1, extend_gap_score=-1)
    M = pd.read_csv(ROOT / "extract" / "manifest" / f"{TASK}.csv")
    M["name"] = M.id.str.split("__").str[0]
    seq_of = M.drop_duplicates("name").set_index("name").sequence
    m = M[M.stage1 == 1].set_index("id")
    C = pd.read_csv(REL / "covariates.csv", index_col=0)
    common_struct.set_task(TASK)
    tab = common_struct.build(TASK)
    rows, desc, jobs = {}, {}, []
    n_fixed = 0
    for i in C.index:
        s = m.loc[i, "sequence"]
        name = i.split("__")[0]
        par_name, kind = parse(name)
        assert par_name == C.loc[i, "parent"], i
        q = tab[i][10:30].astype(np.float64)
        L = len(s)
        if C.loc[i, "parent_missing"] == 1:
            subs, cp_seq = [], q.copy()
        else:
            par = seq_of[par_name]
            subs, fixed = substitutions(par, s, kind, name, al)
            n_fixed += fixed
            cp_seq = comp(par)
        rev = list(s)
        for k, x, y in subs:
            assert rev[k] == y
            rev[k] = x
        cp = comp("".join(rev)) if subs else q.copy()
        desc[i] = np.concatenate([cp, cp_seq])
        rec = {f"wt_{a}": 0.0 for a in AA1} | {f"mut_{a}": 0.0 for a in AA1}
        for _, x, y in subs:
            rec[f"wt_{x}"] += 1
            rec[f"mut_{y}"] += 1
        rec["relpos_mean"] = float(np.mean([(k + 1) / L for k, _, _ in subs])) if subs else 0.0
        for b in range(5):
            rec[f"pos_bin{b}"] = float(sum(min(int(5 * k / L), 4) == b for k, _, _ in subs))
        rec.update(rsa_mean=0.0, dssp_H=0.0, dssp_E=0.0, dssp_C=0.0, no_sub=float(not subs), env_missing=0.0)
        rows[i] = rec
        if subs:
            jobs.append((i, (m.loc[i, "structure_path"], L, [k for k, _, _ in subs]), [y for _, _, y in subs]))
    print(f"variants {len(rows)}, with substitutions {len(jobs)}, point fallback used {n_fixed}", flush=True)
    with Pool(4) as pool:
        res = pool.map(env_one, [j[1] for j in jobs], chunksize=32)
    name_mismatch = 0
    for (i, _, muts), (r, nr, cls, nc, nm) in zip(jobs, res):
        rows[i]["rsa_mean"] = r if nr else 0.0
        rows[i]["dssp_H"], rows[i]["dssp_E"], rows[i]["dssp_C"] = map(float, cls)
        rows[i]["env_missing"] = float(nr == 0 and nc == 0)
        if nm is None or list(nm) != muts:
            name_mismatch += 1
    F = pd.DataFrame.from_dict(rows, orient="index")
    F.index.name = "id"
    F.to_csv(DIR / "site_features.csv")
    D = pd.DataFrame.from_dict(desc, orient="index",
                              columns=[f"cparent_{a}" for a in COMP_ORDER] + [f"cparent_seq_{a}" for a in COMP_ORDER])
    D.index.name = "id"
    D.to_csv(DIR / "descriptors.csv")
    S = C.join(m.split)
    t = S.split == "test"
    info = dict(n_variants=len(rows), n_with_subs=len(jobs), point_fallback=n_fixed,
                structure_residue_mismatch=name_mismatch, env_missing=int(F.env_missing.sum()),
                sites_rsa_nan=int(sum(len(j[1][2]) - x[1] for j, x in zip(jobs, res))),
                sites_dssp_missing=int(sum(len(j[1][2]) - x[3] for j, x in zip(jobs, res))),
                test_one_sub=int((F.loc[t[t].index, [f"wt_{a}" for a in AA1]].sum(1) == 1).sum()),
                test_no_sub=int(F.loc[t[t].index, "no_sub"].sum()))
    pd.Series(info).to_csv(DIR / "site_features_info.csv", header=["value"])
    print(info)


def load_tables():
    C = pd.read_csv(REL / "covariates.csv", index_col=0)
    F = pd.read_csv(DIR / "site_features.csv", index_col=0).loc[C.index]
    D = pd.read_csv(DIR / "descriptors.csv", index_col=0).loc[C.index]
    c0 = [c for c in C.columns if c != "parent"]
    C1 = pd.concat([C[c0], F], axis=1)
    return C, C[c0], C1, D


# ----------------------------------------------------------------------------- cells
def run_cell(enc, C, C0, C1, D):
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
    base = {"full": Q,
            "cparent": tuple(D.loc[i, cp].to_numpy(float) for i in ids),
            "cparent_seq": tuple(D.loc[i, cps].to_numpy(float) for i in ids)}
    base["dc"] = tuple(q.astype(np.float64) - c for q, c in zip(Q, base["cparent"]))
    base["dc_seq"] = tuple(q.astype(np.float64) - c for q, c in zip(Q, base["cparent_seq"]))
    desc = {}
    for k in DESCS:
        desc[k] = base[k]
        desc[k + "_perm"] = tuple(v[p] for v, p in zip(base[k], perm))
    parents = C.loc[ids[2], "parent"].to_numpy()
    rows = []
    for tag, T in (("C0", C0), ("C1", C1)):
        cov = tuple(T.loc[i].to_numpy(float) for i in ids)
        for arm, (A, Bv, Cv) in arms.items():
            keys = [None, "full"] if arm == "intact" else [None] + [k for d in DESCS for k in (d, d + "_perm")]
            for key in keys:
                parts = [(A, Bv, Cv), cov] + ([desc[key]] if key else [])
                A2, B2, C2 = (np.hstack([p[j] for p in parts]) for j in range(3))
                pooled, within = head(A2, y, B2, yva, C2, yte, parents)
                rows.append(dict(encoder=enc, competitors=tag, arm=arm, descriptor=key or "none", rank=r,
                                 dim=X.shape[1], score=pooled, score_within=within, var_removed=diag[arm]))
                print(f"  {enc:20s} {tag} {arm:7s} {(key or 'none'):16s} {pooled:.4f} {within:.4f}", flush=True)
    return rows


def released():
    d = pd.concat([pd.read_csv(REL / f) for f in ("cells_a.csv", "cells_b.csv")], ignore_index=True)
    d = d.drop_duplicates(["encoder", "covariates", "arm", "descriptor"], keep="last")
    d = d[(d.covariates == "C") & ~d.arm.str.startswith("sham")]
    d["descriptor"] = d.descriptor.map({"none": "none", "named": "full", "permuted": "full_perm"})
    return d.set_index(["encoder", "arm", "descriptor"])[["score", "score_within"]]


def cells():
    fs = sorted(DIR.glob("cells_*.csv"))
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    return d.drop_duplicates(["encoder", "competitors", "arm", "descriptor"], keep="last")


def check(encs=None):
    d = cells()
    d = d[(d.competitors == "C0") & d.descriptor.isin(["none", "full", "full_perm"])]
    if encs:
        d = d[d.encoder.isin(encs)]
    ref = released()
    j = d.set_index(["encoder", "arm", "descriptor"])[["score", "score_within"]].join(ref, rsuffix="_rel", how="left")
    j["d_score"] = (j.score - j.score_rel).abs()
    j["d_within"] = (j.score_within - j.score_within_rel).abs()
    per = j.groupby(level=0).agg(n=("d_score", "size"), missing=("score_rel", lambda v: int(v.isna().sum())),
                                 max_d_score=("d_score", "max"), max_d_within=("d_within", "max"))
    per["pass"] = (per.missing == 0) & (per.max_d_score <= 1e-6) & (per.max_d_within <= 1e-6)
    per.to_csv(DIR / "reproduction_check.csv")
    print(per.to_string())
    print("ALL PASS" if per["pass"].all() else "GATE FAILED")
    return bool(per["pass"].all())


# ----------------------------------------------------------------------------- no representation
def alone():
    C, C0, C1, D = load_tables()
    common_struct.set_task(TASK)
    data = load_task(TASK, "esm2_8m")
    (_, ytr, itr), (_, yva, iva), (_, yte, ite) = (data[s] for s in ("train", "valid", "test"))
    ok = ~np.isnan(ytr)
    conc = concepts_for({"train": itr, "valid": iva, "test": ite})
    Qtr, Qva, Qte, _ = conc[NAME]
    ids = (np.asarray(itr)[ok], np.asarray(iva), np.asarray(ite))
    Q = (Qtr[ok], Qva, Qte)
    cp = tuple(D.loc[i, [f"cparent_{a}" for a in COMP_ORDER]].to_numpy(float) for i in ids)
    cps = tuple(D.loc[i, [f"cparent_seq_{a}" for a in COMP_ORDER]].to_numpy(float) for i in ids)
    base = {"full": Q, "cparent": cp, "cparent_seq": cps,
            "dc": tuple(q.astype(np.float64) - c for q, c in zip(Q, cp)),
            "dc_seq": tuple(q.astype(np.float64) - c for q, c in zip(Q, cps))}
    par = C.loc[ids[2], "parent"].to_numpy()
    out = []
    for tag, T in (("none", None), ("C0", C0), ("C1", C1)):
        cov = [tuple(T.loc[i].to_numpy(float) for i in ids)] if T is not None else []
        for key in [None] + DESCS:
            parts = cov + ([base[key]] if key else [])
            if not parts:
                continue
            A2, B2, C2 = (np.hstack([p[j] for p in parts]) for j in range(3))
            s, w = head(A2, ytr[ok], B2, yva, C2, yte, par)
            out.append(dict(competitors=tag, descriptor=key or "none", S=s, S_within=w))
            print(f"  {tag:4s} {(key or 'none'):12s} {s:.4f} {w:.4f}", flush=True)
    pd.DataFrame(out).to_csv(DIR / "no_representation.csv", index=False)


# ----------------------------------------------------------------------------- report
def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    m, out, run = len(p), np.empty(len(p)), 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * p[i])
        out[i] = min(run, 1.0)
    return out


def contrasts(d):
    rows = []
    for (enc, cs), g in d.groupby(["encoder", "competitors"]):
        for metric in ("score", "score_within"):
            p = g.pivot_table(index="arm", columns="descriptor", values=metric)
            nulls = [a for a in p.index if a.startswith("null")]
            for k in DESCS:
                rv = (p.loc["erased", k] - p.loc["erased", "none"]) - (p.loc[nulls, k] - p.loc[nulls, "none"]).mean()
                rp = (p.loc["erased", k + "_perm"] - p.loc["erased", "none"]) - \
                    (p.loc[nulls, k + "_perm"] - p.loc[nulls, "none"]).mean()
                rows.append(dict(encoder=enc, competitors=cs, descriptor=k, metric=metric, R_var=rv, Rperm=rp))
    return pd.DataFrame(rows)


def report():
    from family_weighted_demand import family
    d = cells()
    d = d[d.encoder.isin(panel())]
    E = contrasts(d)
    E["family"] = E.encoder.map(family)
    E.to_csv(DIR / "cells_summary.csv", index=False)
    out = []
    for (metric, desc, cs), g in E.groupby(["metric", "descriptor", "competitors"], sort=False):
        v, vp = g.R_var.to_numpy(), g.Rperm.to_numpy()
        k, kp = int((v > 0).sum()), int((vp > 0).sum())
        fam = g.groupby("family").R_var.mean()
        kf = int((fam > 0).sum())
        out.append(dict(metric=metric, descriptor=desc, competitors=cs, n=len(v), R_var_mean=float(v.mean()),
                        R_var_pos=k, R_var_p=binomtest(k, len(v), 0.5).pvalue, n_fam=len(fam), fam_pos=kf,
                        fam_p=binomtest(kf, len(fam), 0.5).pvalue, Rperm_mean=float(vp.mean()), Rperm_pos=kp,
                        Rperm_p=binomtest(kp, len(vp), 0.5).pvalue))
    V = pd.DataFrame(out)
    V["R_var_p_holm"] = np.nan
    for metric, g in V.groupby("metric"):
        V.loc[g.index, "R_var_p_holm"] = holm(g.R_var_p.to_numpy())

    def verdict(r):
        if r.R_var_p_holm >= 0.05:
            return "not detected"
        if r.R_var_mean < 0:
            return "negative"
        return "carries repair" if r.R_var_mean >= 0.02 else "attenuated"
    V["verdict"] = V.apply(verdict, axis=1)
    V["metric"] = V.metric.map({"score": "pooled (primary)", "score_within": "within-parent (secondary)"})
    order = {k: i for i, k in enumerate(DESCS)}
    V = V.sort_values(["metric", "competitors", "descriptor"], key=lambda s: s.map(order) if s.name == "descriptor" else s)
    V.to_csv(DIR / "verdicts.csv", index=False)
    pd.set_option("display.width", 250)
    print(V.round(4).to_string(index=False))
    nr = DIR / "no_representation.csv"
    if nr.exists():
        print(pd.read_csv(nr).round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--alone", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--encoders", nargs="+", default=None)
    ap.add_argument("--out", default="cells_all.csv")
    a = ap.parse_args()
    DIR.mkdir(parents=True, exist_ok=True)
    if a.features:
        return features()
    if a.check:
        return check(a.encoders)
    if a.alone:
        return alone()
    if a.report:
        return report()
    common_struct.set_task(TASK)
    C, C0, C1, D = load_tables()
    dest = DIR / a.out
    rows = pd.read_csv(dest).to_dict("records") if dest.exists() else []
    done = {r["encoder"] for r in rows}
    encs = a.encoders or [e for e in panel() if (ROOT / "extract" / "embeddings" / TASK / e / "meta.json").exists()]
    for e in encs:
        if e in done:
            continue
        rows += run_cell(e, C, C0, C1, D)
        pd.DataFrame(rows).to_csv(dest, index=False)
    print("written", dest)


if __name__ == "__main__":
    main()
