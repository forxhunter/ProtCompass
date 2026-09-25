"""Decisions of the pre-registered determinacy test, exactly as frozen in prereg/freeze.json.

Selection (validation only). For every source (encoder or floor) and label, the readout family is the one with the
higher validation headline score (linear, or the MLP-2 mean over its three seeds; ties go to linear), and its test
score is used (the MLP-2 test score is the mean over seeds). For every encoder and label, the reference is the
admissible floor with the highest validation score after its own family choice, and G_ref is the encoder's test score
minus that floor's test score. Headline metrics: balanced accuracy (classes), circular cosine (angle), Pearson (count).

Units. Primary labels are hbond_count, chi1_class and chi2_class; ca_dihedral is the positive control and
sc_contacts is descriptive. Cells with status not_established do not enter H1-H4. A unit is an (architecture family,
label, status) triple, with G_ref averaged over the family's members of that status.

  H1  among determined units, the share with G_ref <= 0 is at least 0.80
  H2  among undetermined units, the share with G_ref > 0 is at least 0.80
  H3  Delta_G = mean G_ref over undetermined units - mean over determined units > 0, and the 95% interval of a
      bootstrap that resamples architecture families (10,000 draws, seed 0) excludes 0
  H4  on cohort B, with the frozen cohort panel, Delta_G^B > 0 and at least 80% of cells have the predicted sign
      (G_ref <= 0 when determined, > 0 when undetermined)

Descriptive outputs: the H1/H2 shares at every threshold from 0.50 to 1.00, cell-level counts, the control and
descriptive labels, and a protein-level paired bootstrap interval for every cell (1,000 draws, seed 0; the encoder
and its floor are scored on the same resampled test proteins).

  python prereg_decide.py [--cohort B] [--synthetic]
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
from scipy.stats import pearsonr

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
sys.path.insert(0, str(ROOT / "scripts"))
from family_weighted_demand import family                                 # noqa: E402

PRIMARY = ["hbond_count", "chi1_class", "chi2_class"]
CONTROL, DESCRIPTIVE = ["ca_dihedral"], ["sc_contacts"]
KIND = {"hbond_count": "clf", "chi1_class": "clf", "chi2_class": "clf", "ca_dihedral": "circ", "sc_contacts": "reg"}
THRESH = 0.80
SYNTH = "--synthetic" in sys.argv


def headline(d):
    h = pd.Series(np.nan, index=d.index)
    for c in ("balanced_accuracy", "circ_cos", "pearson"):
        if c in d:
            h = h.fillna(d[c])
    return h


def selected(res):
    """(source, probe) -> chosen family, validation score, test score."""
    res = res.assign(h=headline(res), family=np.where(res.readout == "linear", "linear", "mlp2"))
    s = res.groupby(["source", "probe", "family", "split"]).h.mean().unstack("split").reset_index()
    rows = []
    for (src, pr), g in s.groupby(["source", "probe"]):
        g = g.set_index("family")
        fam = "linear" if g.loc["linear", "val"] >= g.loc["mlp2", "val"] else "mlp2"
        rows.append(dict(source=src, probe=pr, family=fam, val=g.loc[fam, "val"], test=g.loc[fam, "test"]))
    return pd.DataFrame(rows).set_index(["source", "probe"])


def admissible_floors(freeze):
    return {e: v for e, v in freeze["admissible_floors"].items()}


def cells(res, freeze, dmap):
    S = selected(res)
    adm = admissible_floors(freeze)
    rows = []
    for e in freeze["panel_residue"]:
        for lab in PRIMARY + CONTROL + DESCRIPTIVE:
            if (e, lab) not in S.index:
                continue
            fl = [f for f in adm[e] if (f, lab) in S.index]
            if not fl:
                continue
            best = max(fl, key=lambda f: S.loc[(f, lab), "val"])
            rows.append(dict(encoder=e, family=family(e), label=lab, status=dmap.get((e, lab), "not_established"),
                             enc_family=S.loc[(e, lab), "family"], enc_test=S.loc[(e, lab), "test"],
                             floor=best, floor_family=S.loc[(best, lab), "family"], floor_test=S.loc[(best, lab), "test"],
                             G_ref=S.loc[(e, lab), "test"] - S.loc[(best, lab), "test"]))
    return pd.DataFrame(rows)


def units(C):
    P = C[C.label.isin(PRIMARY) & C.status.isin(["determined", "undetermined"])]
    return P.groupby(["family", "label", "status"]).G_ref.mean().reset_index()


def delta(U):
    return U[U.status == "undetermined"].G_ref.mean() - U[U.status == "determined"].G_ref.mean()


def family_bootstrap(U, n=10_000, seed=0):
    rng = np.random.default_rng(seed)
    fams = U.family.unique()
    by = {f: U[U.family == f] for f in fams}
    out = []
    for _ in range(n):
        pick = rng.choice(fams, len(fams))
        V = pd.concat([by[f] for f in pick])
        if (V.status == "determined").any() and (V.status == "undetermined").any():
            out.append(delta(V))
    return np.percentile(out, [2.5, 97.5]) if out else np.array([np.nan, np.nan])


def metric(kind, y, p):
    if kind == "clf":
        cls = np.unique(y)
        return float(np.mean([(p[y == c] == c).mean() for c in cls]))
    if kind == "circ":
        return float(np.cos(np.arctan2(y[:, 0], y[:, 1]) - np.arctan2(p[:, 0], p[:, 1])).mean())
    p = p.reshape(-1)
    return float(pearsonr(y, p)[0]) if np.std(p) > 0 else 0.0


def protein_stats(kind, y, p, prot, ups):
    """Per-protein sufficient statistics of the headline metric, so that a bootstrap draw is a weighted sum."""
    j = np.searchsorted(ups, prot)
    n = len(ups)
    if kind == "clf":
        k = int(max(y.max(), p.max())) + 1
        tot, hit = np.zeros((n, k)), np.zeros((n, k))
        np.add.at(tot, (j, y), 1.0); np.add.at(hit, (j, y), (p == y).astype(float))
        return tot, hit
    if kind == "circ":
        c = np.cos(np.arctan2(y[:, 0], y[:, 1]) - np.arctan2(p[:, 0], p[:, 1]))
        return np.bincount(j, c, n)[:, None], np.bincount(j, None, n)[:, None]
    p = p.reshape(-1)
    cols = [np.ones_like(y), y, p, y * y, p * p, y * p]
    return np.stack([np.bincount(j, c, n) for c in cols], 1), None


def metric_from(kind, W, a, b):
    """Headline metric for every row of the weight matrix W (draws x proteins)."""
    if kind == "clf":
        tot, hit = W @ a, W @ b
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.nanmean(np.where(tot > 0, hit / tot, np.nan), 1)
    if kind == "circ":
        return (W @ a)[:, 0] / (W @ b)[:, 0]
    S = W @ a
    n, sy, sp, syy, spp, syp = S.T
    cov = syp / n - sy * sp / n ** 2
    return cov / np.sqrt(np.maximum(syy / n - (sy / n) ** 2, 1e-12) * np.maximum(spp / n - (sp / n) ** 2, 1e-12))


def cell_bootstrap(C, pred_dir, n=1000, seed=0):
    """Protein-level paired bootstrap of G_ref for every cell; encoder and floor see the same resampled proteins
    (the same weight matrix)."""
    cache, lo, hi = {}, [], []

    def load(src):
        if src not in cache:
            cache[src] = dict(np.load(pred_dir / f"{src}__test_predictions.npz", allow_pickle=False))
        return cache[src]

    keys = lambda fam: ["linear_0"] if fam == "linear" else ["mlp2_0", "mlp2_1", "mlp2_2"]

    def score(kind, W, y, pred, prot, ups):
        if kind == "clf":
            pred = pred.astype(np.int64)
        return metric_from(kind, W, *protein_stats(kind, y, pred, prot, ups))

    for r in C.itertuples():
        kind = KIND[r.label]
        E, F = load(r.encoder), load(r.floor)
        y, prot = E[f"{r.label}|y"], E[f"{r.label}|protein"]
        assert np.array_equal(prot, F[f"{r.label}|protein"]), (r.encoder, r.floor, r.label)
        assert SYNTH or np.array_equal(y, F[f"{r.label}|y"]), (r.encoder, r.floor, r.label)
        if kind == "clf":
            y = y.astype(np.int64)
        ups = np.unique(prot)
        rng = np.random.default_rng(seed)
        W = np.stack([np.bincount(rng.integers(0, len(ups), len(ups)), minlength=len(ups)) for _ in range(n)]).astype(float)
        me = np.mean([score(kind, W, y, E[f"{r.label}|{k}"], prot, ups) for k in keys(r.enc_family)], 0)
        mf = np.mean([score(kind, W, y, F[f"{r.label}|{k}"], prot, ups) for k in keys(r.floor_family)], 0)
        g = me - mf
        lo.append(float(np.percentile(g, 2.5))); hi.append(float(np.percentile(g, 97.5)))
    return C.assign(G_lo=lo, G_hi=hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--no-cell-bootstrap", action="store_true")
    a = ap.parse_args()
    freeze = json.loads((ROOT / "prereg" / "freeze.json").read_text()) if not a.synthetic else \
        json.loads((ROOT / "prereg" / "freeze_draft.json").read_text())
    base = ROOT / "prereg" / ("results_cohortB" if a.cohort == "B" else "results")
    sub = base / ("synthetic" if a.synthetic else "by_source")
    res = pd.concat([pd.read_csv(f) for f in sorted(sub.glob("*.csv"))], ignore_index=True)
    dm = pd.read_csv(ROOT / "prereg" / "determinacy_map.csv")
    dmap = {(r.encoder, r.label): r.status for r in dm.itertuples()}
    if a.cohort == "B":
        freeze = dict(freeze, panel_residue=freeze["cohort_B_panel"])
    C = cells(res, freeze, dmap)
    if not a.no_cell_bootstrap:
        C = cell_bootstrap(C, sub)
    tag = ("_cohortB" if a.cohort == "B" else "") + ("_synthetic" if a.synthetic else "")
    C.to_csv(ROOT / "prereg" / f"cells{tag}.csv", index=False)
    out = {}
    if a.cohort == "B":
        P = C[C.label.isin(PRIMARY) & C.status.isin(["determined", "undetermined"])]
        dG = P[P.status == "undetermined"].G_ref.mean() - P[P.status == "determined"].G_ref.mean()
        right = ((P.status == "determined") & (P.G_ref <= 0)) | ((P.status == "undetermined") & (P.G_ref > 0))
        out["H4"] = dict(delta_G=float(dG), direction_accuracy=float(right.mean()), n_cells=int(len(P)),
                         n_determined=int((P.status == "determined").sum()),
                         passes=bool(dG > 0 and right.mean() >= THRESH))
    else:
        U = units(C)
        U.to_csv(ROOT / "prereg" / f"units{tag}.csv", index=False)
        det, und = U[U.status == "determined"], U[U.status == "undetermined"]
        s1, s2 = float((det.G_ref <= 0).mean()), float((und.G_ref > 0).mean())
        lo, hi = family_bootstrap(U)
        out["H1"] = dict(share=s1, n_units=int(len(det)), passes=bool(s1 >= THRESH))
        out["H2"] = dict(share=s2, n_units=int(len(und)), passes=bool(s2 >= THRESH))
        out["H3"] = dict(delta_G=float(delta(U)), ci=[float(lo), float(hi)], passes=bool(delta(U) > 0 and lo > 0))
        out["threshold_sweep"] = {f"{t:.2f}": dict(H1=bool(s1 >= t), H2=bool(s2 >= t)) for t in np.arange(0.5, 1.001, 0.05)}
        P = C[C.label.isin(PRIMARY)]
        out["cells"] = {st: dict(n=int((P.status == st).sum()), below_or_at_floor=int(((P.status == st) & (P.G_ref <= 0)).sum()))
                        for st in ("determined", "undetermined", "not_established")}
        for lab in CONTROL + DESCRIPTIVE:
            Q = C[C.label == lab]
            out[lab] = {st: dict(n=int((Q.status == st).sum()), below_or_at_floor=int(((Q.status == st) & (Q.G_ref <= 0)).sum()))
                        for st in Q.status.unique()}
    (ROOT / "prereg" / f"decision{tag}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
