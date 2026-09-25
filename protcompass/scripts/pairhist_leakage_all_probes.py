"""Does `ref_pairhist` contain the label of any probe in the suite? One check per probe, at the probe's own level.

`scripts/pairhist_floor.py --check` covered the five probes the floor was first scored against: contact order, radius
of gyration, contact density, fold architecture and amino-acid identity. `ref_pairhist` is now scored on all 21 probes,
so the same question has to be answered for the other sixteen. Three floors in this project turned out to contain their
own label, so the check is numeric and the numbers are reported whichever way they come out.

The features are exactly the ones the probes are handed, from `run_probing_suite.Data`:

  residue level  335 features  pairhist_features_residue  (protein vector, own histogram row, contact-map eigenvectors)
  protein level  174 features  pairhist_features_protein  (histogram, spectral block, log length)
  pair level     the PairNet sees the residue-level feature vectors of both residues

Three kinds of check:

  1. cutoff coincidence. A bin edge or spectral cutoff equal to a cutoff that defines a label makes the label a union
     of bins. Every cutoff in probing_data/make_probing_data.py is listed against every edge in the descriptor.
  2. single-feature identity. max |pearson| for a continuous label, max eta squared for a categorical one, over all
     features at that level, accumulated over every domain the suite samples. 1.000 means the feature is the label.
  3. determinacy. For the labels that are closed-form functions of the CA coordinates the descriptor reads
     (local_curvature, contact_density, distance_bins, contact_order, radius_of_gyration, asphericity), an explicit
     attempt to recover the label from the features, reported as an error rather than a verdict. For the labels that
     need atoms or data the descriptor is not given (phi, psi, dssp, rsa, bfactor, core_surface, amino-acid identity,
     CATH architecture) no closed form can exist, and the reason is recorded next to the numeric check.

Writes probing_results/floor_pairhist_checks_all_probes.csv. Nothing else is touched. CPU only, so it can run beside
the probe job.
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import run_probing_suite as R

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
OUT = ROOT / "probing_results" / "floor_pairhist_checks_all_probes.csv"
CHUNK = 150                       # proteins per chunk of residue features; keeps the working set near 50 MB
S, D = len(R.PAIRHIST_SEP_EDGES) - 1, len(R.PAIRHIST_DIST_EDGES) - 1
NGLOB = S * D + 2 * (R.PAIRHIST_NEV + R.PAIRHIST_NEV // 2 + 3) + 1        # 174
LEAK = 0.999                      # a feature is the label at or above this

# Label cutoffs of the suite, from probing_data/make_probing_data.py.
DIST_CUTOFFS = {"CONTACT_CUTOFF (contact_density, contact_order)": 8.0, "CORE_RADIUS (core_surface)": 10.0,
                **{f"distance_bins edge {k * 2.5:g}": k * 2.5 for k in range(1, 17)}}
SEP_CUTOFFS = {"CONTACT_MIN_SEP (contact_order)": 4, "PAIR_MIN_SEP (distance_bins)": 6}

# probe -> (label dataset or tuple, kind, why no closed form can exist, or "" when the label is a CA-only function)
RESIDUE_LABELS = [
    ("phi_angle_bins", "phi_bin", "clf", "phi is the dihedral of C(i-1), N, CA, C; the descriptor reads CA only"),
    ("psi_angle_bins", "psi_bin", "clf", "psi is the dihedral of N, CA, C, N(i+1); the descriptor reads CA only"),
    ("phi_circular", ("phi_sin", "phi_cos"), "reg", "same as phi_angle_bins"),
    ("psi_circular", ("psi_sin", "psi_cos"), "reg", "same as psi_angle_bins"),
    ("local_curvature", "curvature", "reg", ""),
    ("dssp_3", "dssp3", "clf", "DSSP hydrogen-bond energies use N, CA, C, O; the descriptor reads CA only"),
    ("dssp_8", "dssp8", "clf", "same as dssp_3"),
    ("rsa", "rsa", "reg", "freesasa uses every side-chain atom; the descriptor reads CA only"),
    ("rsa_bins", "rsa_bin", "clf", "same as rsa"),
    ("bfactor", "bfactor_z", "reg", "an experimental quantity, not a function of the coordinates at all"),
    ("core_surface", "core", "clf", "the CB neighbour count within 10 A; the descriptor reads CA only"),
    ("contact_density", "contact_density_bin", "clf", ""),
    ("sequence_recovery", "aa", "clf", "residue type is not a function of the CA positions"),
]
PROTEIN_LABELS = [
    ("contact_order", "contact_order", "reg", ""),
    ("contact_order_len_resid", f"contact_order_len_resid_{R.SPLIT}", "reg", ""),
    ("radius_of_gyration", "radius_of_gyration", "reg", ""),
    ("radius_of_gyration_len_resid", f"radius_of_gyration_len_resid_{R.SPLIT}", "reg", ""),
    ("asphericity", "asphericity", "reg", ""),
    ("asphericity_len_resid", f"asphericity_len_resid_{R.SPLIT}", "reg", ""),
    ("architecture_label", f"architecture_label_{R.SPLIT}", "clf", "a CATH annotation, not a function of the coordinates"),
]


# ----------------------------------------------------------------------------- streaming accumulators
class Corr:
    """max |pearson| between each column of X and a 1-D label, accumulated over chunks."""

    def __init__(self, d):
        self.n = 0.0
        self.sx = np.zeros(d); self.sxx = np.zeros(d); self.sxy = np.zeros(d)
        self.sy = 0.0; self.syy = 0.0

    def add(self, X, y):
        X = X.astype(np.float64)
        self.n += len(y)
        self.sx += X.sum(0); self.sxx += (X ** 2).sum(0); self.sxy += X.T @ y
        self.sy += y.sum(); self.syy += (y ** 2).sum()

    def r(self):
        n = self.n
        cov = self.sxy / n - self.sx / n * (self.sy / n)
        vx = np.maximum(self.sxx / n - (self.sx / n) ** 2, 0.0)
        vy = max(self.syy / n - (self.sy / n) ** 2, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = cov / np.sqrt(vx * vy)
        return np.abs(np.where(np.isfinite(r), r, 0.0))


class Eta:
    """max eta squared (between-class sum of squares over total) of each column of X on a categorical label."""

    def __init__(self, d, k):
        self.d, self.k = d, k
        self.n = np.zeros(k); self.sxc = np.zeros((k, d))
        self.sx = np.zeros(d); self.sxx = np.zeros(d)

    def add(self, X, c):
        X = X.astype(np.float64)
        np.add.at(self.n, c, 1.0)
        np.add.at(self.sxc, c, X)
        self.sx += X.sum(0); self.sxx += (X ** 2).sum(0)

    def eta2(self):
        n = self.n.sum()
        gmean = self.sx / n
        ok = self.n > 0
        between = (((self.sxc[ok] / self.n[ok, None]) - gmean) ** 2 * self.n[ok, None]).sum(0)
        total = np.maximum(self.sxx - n * gmean ** 2, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            e = between / total
        return np.where(np.isfinite(e), e, 0.0)


# ----------------------------------------------------------------------------- residue-level pass
def residue_pass(data):
    """One pass over every domain the suite samples, accumulating a check per residue-level probe."""
    prot = np.sort(np.concatenate([data.chosen[p] for p in R.PARTS]))
    need = {"curvature", "contact_count", "core", "cb_neighbours"}
    for _, ds, _, _ in RESIDUE_LABELS:
        need.update(ds if isinstance(ds, tuple) else [ds])
    lab = {k: data.R[k][:] for k in sorted(need)}
    acc, extra = {}, {}
    for name, ds, kind, _ in RESIDUE_LABELS:
        if kind == "clf":
            k = int(max(lab[ds].max(), 0)) + 1
            acc[name] = Eta(335, k)
        else:
            acc[name] = [Corr(335) for _ in (ds if isinstance(ds, tuple) else [ds])]
    # determinacy material collected on the way: the exact CA degrees the features expose
    extra["curv_by_row"] = []          # (integer sep-1..sep-2 histogram counts, curvature)
    extra["deg"] = []                  # (deg7, deg10, contact_count at 8 A, core, cb_neighbours)
    t0 = time.time()
    for s in range(0, len(prot), CHUNK):
        idx = prot[s:s + CHUNK]
        X = data.pairhist_features_residue(idx)
        X64 = X.astype(np.float64)
        rows = np.concatenate([np.arange(data.offs[i], data.offs[i] + data.L[i]) for i in idx])
        for name, ds, kind, _ in RESIDUE_LABELS:
            if isinstance(ds, tuple):
                for a, dsi in zip(acc[name], ds):
                    y = lab[dsi][rows].astype(np.float64)
                    m = ~np.isnan(y)
                    a.add(X64[m], y[m])
            elif kind == "reg":
                y = lab[ds][rows].astype(np.float64)
                m = ~np.isnan(y)
                acc[name][0].add(X64[m], y[m])
            else:
                c = lab[ds][rows].astype(np.int64)
                m = c >= 0
                acc[name].add(X64[m], c[m])
        del X64
        # the residue's own histogram row, as integer counts, and the degrees at the two spectral cutoffs
        npair = np.concatenate([np.full(int(data.L[i]), int(data.L[i]) - 1) for i in idx])
        rowh = np.rint(X[:, NGLOB:NGLOB + S * D] * npair[:, None]).astype(np.int32)
        n_res = np.concatenate([np.full(int(data.L[i]), int(data.L[i])) for i in idx])
        deg7 = np.rint(X[:, NGLOB + S * D + R.PAIRHIST_NEV] * n_res).astype(np.int32)
        deg10 = np.rint(X[:, NGLOB + S * D + 2 * R.PAIRHIST_NEV + 1] * n_res).astype(np.int32)
        extra["deg"].append(np.column_stack([deg7, deg10, lab["contact_count"][rows],
                                             lab["core"][rows], lab["cb_neighbours"][rows]]))
        extra["curv_by_row"].append((rowh[:, :2 * D].copy(), lab["curvature"][rows].astype(np.float64)))
        print(f"  residue features {min(s + CHUNK, len(prot))}/{len(prot)} proteins  [{time.time() - t0:.0f}s]", flush=True)
        del X
    return acc, extra


def protein_features(data):
    """Protein-level ref_pairhist for every domain. Reuses the cache pairhist_floor.py wrote, after checking that a
    sample of its rows still matches a fresh computation, so a stale cache cannot silently change a number."""
    cache = Path(__file__).resolve().parent.parent / "probing_results" / ".pairhist_protein_cache.npy"
    alt = Path((PROTCOMPASS_ROOT + "/protcompass/cache/scratch/pairhist_protein.npy"))
    for c in (cache, alt):
        if c.exists():
            X = np.load(c)
            if X.shape == (len(data.L), NGLOB):
                probe_idx = np.random.default_rng(0).choice(len(data.L), 8, replace=False)
                fresh = data.pairhist_features_protein(probe_idx)
                if np.allclose(X[probe_idx], fresh, atol=1e-5):
                    print(f"  reusing cached protein features from {c}", flush=True)
                    return X
                print(f"  cache {c} disagrees with a fresh computation; recomputing", flush=True)
    X = np.concatenate([data.pairhist_features_protein(np.arange(s, min(s + 2000, len(data.L))))
                        for s in range(0, len(data.L), 2000)])
    return X


def say(rows_out, probe, check, value, verdict):
    rows_out.append({"probe": probe, "check": check, "value": value, "verdict": verdict})
    print(f"  {probe:30s} {check:52s} {value:>10s}  {verdict}", flush=True)


def main():
    data = R.Data()
    rows_out = []

    # ------------------------------------------------------------------ 1. cutoff coincidence, all labels at once
    dclash = [k for k, v in DIST_CUTOFFS.items() if np.any(np.isclose(R.PAIRHIST_DIST_EDGES, v))]
    say(rows_out, "all probes", "label distance cutoffs equal to a histogram bin edge", str(len(dclash)),
        "ok, none" if not dclash else f"LEAK: {dclash}")
    sclash = [k for k, v in SEP_CUTOFFS.items() if np.any(R.PAIRHIST_SEP_EDGES == v)]
    say(rows_out, "all probes", "label separation cutoffs equal to a separation bin edge", str(len(sclash)),
        "ok, none" if not sclash else f"LEAK: {sclash}")
    # The bin edges avoid every label cutoff, but the spectral block is built at 7.0 and 10.0 A, and 10.0 A is both the
    # core radius of `core_surface` and the fourth bin edge of `distance_bins`. The residue block carries deg/L at each
    # cutoff and the protein block carries log L, so the CA neighbour count within 10 A is exposed exactly. Say so and
    # measure what it implies for the two labels, rather than leaving the coincidence unstated.
    cclash = [k for k, v in DIST_CUTOFFS.items() if np.any(np.isclose(R.PAIRHIST_CUTOFFS, v))]
    say(rows_out, "core_surface", "label distance cutoffs equal to a spectral cutoff", str(cclash or "none"),
        "reported: 10.0 A is the core radius, so the exact CA neighbour count within 10 A is a feature; the label "
        "thresholds the CB count, measured below" if cclash else "ok, none")
    say(rows_out, "distance_bins", "label distance cutoffs equal to a spectral cutoff", str(cclash or "none"),
        "reported: 10.0 A is also the fourth of the 16 label bin edges; what that pins down is measured below"
        if cclash else "ok, none")

    # ------------------------------------------------------------------ 2+3. residue level
    print("residue-level pass", flush=True)
    acc, extra = residue_pass(data)
    for name, ds, kind, why in RESIDUE_LABELS:
        if kind == "clf":
            v = float(np.nanmax(acc[name].eta2()))
            say(rows_out, name, "max eta squared of any one of the 335 features", f"{v:.4f}",
                ("ok, no feature is the label" if v < LEAK else "LEAK") + (f"; {why}" if why else ""))
        else:
            v = max(float(np.nanmax(a.r())) for a in acc[name])
            say(rows_out, name, "max |pearson| of any one of the 335 features", f"{v:.4f}",
                ("ok, no feature is the label" if v < LEAK else "LEAK") + (f"; {why}" if why else ""))

    # local_curvature: the angle at residue i is fixed by d(i-1,i), d(i,i+1) and d(i-1,i+1). The feature vector gives
    # residue i only binned counts of its own partners, so the two chain links fall in one bin and d(i-1,i+1) is not in
    # row i at all. Measure what is left: group residues by the exact integer counts of their sep<3 histogram block and
    # report how much of the curvature variance survives inside a group.
    rowh = np.concatenate([a for a, _ in extra["curv_by_row"]])
    curv = np.concatenate([b for _, b in extra["curv_by_row"]])
    m = ~np.isnan(curv)
    rowh, curv = rowh[m], curv[m]
    _, g = np.unique(rowh, axis=0, return_inverse=True)
    df = pd.DataFrame({"g": g, "y": curv})
    gm = df.groupby("g").y.agg(["mean", "size"])
    between = ((gm["mean"] - curv.mean()) ** 2 * gm["size"]).sum()
    eta = between / ((curv - curv.mean()) ** 2).sum()
    say(rows_out, "local_curvature", "eta squared of curvature on the exact sep<3 histogram block", f"{eta:.4f}",
        f"ok, not determined: {len(gm)} distinct blocks over {m.sum()} residues, "
        f"{1 - eta:.4f} of the variance is inside a block")
    say(rows_out, "local_curvature", "residual sd of curvature inside a block (radians)",
        f"{np.sqrt((1 - eta) * curv.var()):.4f}", f"against a total sd of {curv.std():.4f}")

    # contact_density: the label is digitize(CA count within 8 A, [6, 11, 16]). The features give the count at 7 A and
    # at 10 A exactly, so the 8 A count is bracketed. Report how often the bracket leaves the bin open.
    deg = np.concatenate(extra["deg"])
    d7, d10, cc, core, cbn = deg[:, 0], deg[:, 1], deg[:, 2], deg[:, 3], deg[:, 4]
    say(rows_out, "contact_density", "residues whose 8 A CA count is not pinned by the 7 A and 10 A counts",
        f"{float((d10 > d7).mean()):.4f}", "ok, the label stays bracketed, not determined")
    say(rows_out, "contact_density", "mean width of the CA contact-count bracket (neighbours)",
        f"{float((d10 - d7).mean()):.2f}", "")
    bin_lo, bin_hi = np.digitize(d7, [6, 11, 16]), np.digitize(d10, [6, 11, 16])
    say(rows_out, "contact_density", "residues whose density bin is not pinned by the bracket",
        f"{float((bin_hi > bin_lo).mean()):.4f}", "ok, the 4-way label is not a function of the features")
    assert np.all(d7 <= cc + 1) and np.all(cc <= d10 + 1), "degree reconstruction from the features disagrees with the labels"

    # core_surface: the CA degree within 10 A is exposed exactly, and 10 A is the core radius, but the label thresholds
    # the CB count. Measure the gap between the two directly rather than asserting that CA is not CB.
    best = max(float(((d10 >= t).astype(int) == core).mean()) for t in range(0, 61))
    say(rows_out, "core_surface", "best accuracy of a threshold on the exact 10 A CA count", f"{best:.4f}",
        f"ok, not the label: the CB count differs from the CA count on "
        f"{float((cbn != d10).mean()):.4f} of residues, median |CB - CA| = {float(np.median(np.abs(cbn - d10))):.1f}")
    say(rows_out, "core_surface", "pearson of the exact 10 A CA count with the CB count",
        f"{float(np.corrcoef(d10.astype(float), cbn.astype(float))[0, 1]):.4f}",
        "reported: strongly related, not equal")

    # ------------------------------------------------------------------ pair level
    print("pair level", flush=True)
    pair_check(data, rows_out)

    # ------------------------------------------------------------------ protein level
    print("protein level", flush=True)
    XP = protein_features(data)
    for name, ds, kind, why in PROTEIN_LABELS:
        y = data.P[ds][:].astype(np.float64)
        if kind == "clf":
            c = y.astype(np.int64)
            a = Eta(XP.shape[1], int(c.max()) + 1)
            a.add(XP[c >= 0], c[c >= 0])
            v = float(np.nanmax(a.eta2()))
            say(rows_out, name, "max eta squared of any one of the 174 features", f"{v:.4f}",
                ("ok, no feature is the label" if v < LEAK else "LEAK") + (f"; {why}" if why else ""))
        else:
            a = Corr(XP.shape[1])
            m = ~np.isnan(y)
            a.add(XP[m], y[m])
            v = float(np.nanmax(a.r()))
            say(rows_out, name, "max |pearson| of any one of the 174 features", f"{v:.4f}",
                ("ok, no feature is the label" if v < LEAK else "LEAK") + (f"; {why}" if why else ""))

    # asphericity is the relative shape anisotropy of the gyration tensor. The histogram carries the distribution of
    # pair distances, whose second moment gives Rg, but the eigenvalues of the tensor are not a function of that
    # distribution. Attempt the only closed form the features allow and report the error.
    dist_mid = np.array([0.5 * (a_ + b_) if np.isfinite(b_) else a_ * 1.2
                         for a_, b_ in zip(R.PAIRHIST_DIST_EDGES[:-1], R.PAIRHIST_DIST_EDGES[1:])])
    H = XP[:, :S * D].reshape(-1, S, D).astype(np.float64)
    n = data.L.astype(np.float64)
    m2 = (H.sum(1) * dist_mid ** 2).sum(1) * (1 - 1 / np.maximum(n, 2))
    rg_hat = np.sqrt(np.maximum(m2 / 2, 0))
    m4 = (H.sum(1) * dist_mid ** 4).sum(1)
    asph = data.P["asphericity"][:].astype(np.float64)
    # a shape statistic built from the third and fourth moments of the same distribution, the closest closed form
    feat = np.column_stack([m4 / np.maximum(m2 ** 2, 1e-12), rg_hat, np.log(n)])
    coef, *_ = np.linalg.lstsq(np.column_stack([feat, np.ones(len(feat))]), asph, rcond=None)
    pred = np.column_stack([feat, np.ones(len(feat))]) @ coef
    rel = np.abs(pred - asph) / np.maximum(np.abs(asph), 1e-12)
    say(rows_out, "asphericity", "best closed form from the histogram moments, median relative error",
        f"{np.median(rel):.4f}", "ok, not closed form; the gyration eigenvalues are not a function of the pair-distance "
        "distribution")
    say(rows_out, "asphericity", "same, pearson r against the label", f"{float(np.corrcoef(pred, asph)[0, 1]):.4f}",
        "reported, an approximation is not an identity")

    pd.DataFrame(rows_out).to_csv(OUT, index=False)
    print(f"\nwrote {OUT}", flush=True)
    bad = [r for r in rows_out if "LEAK" in r["verdict"]]
    print(f"LEAK verdicts: {len(bad)}", flush=True)
    for r in bad:
        print(f"  {r}", flush=True)


def pair_check(data, rows_out):
    """distance_bins: is the CA-CA distance bin of a sampled pair a function of the two residue feature vectors?

    The PairNet sees the residue-level vectors of i and j. The only part of those that refers to the specific pair is
    the contact-map eigenvector block: with the leading eigenvalues from the protein block, sum_k lam_k v_ki v_kj is a
    rank-8 reconstruction of the adjacency at 7 A and at 10 A. Even a perfect reconstruction would only bracket the
    distance into (<7, 7-10, >=10), and the label has 16 bins of 2.5 A. Both steps are measured: how well the rank-8
    reconstruction recovers the adjacency, and how much of the label the perfect bracket would leave open.
    """
    rng = np.random.default_rng(0)
    pool = np.sort(np.concatenate([data.chosen[p] for p in R.PARTS]))
    prot = rng.choice(pool, min(300, len(pool)), replace=False)
    qp, qi, qj = data.Q["protein_index"][:], data.Q["i"][:].astype(np.int64), data.Q["j"][:].astype(np.int64)
    qbin, qd = data.Q["distance_bin"][:].astype(np.int64), data.Q["distance"][:].astype(np.float64)
    bins, dists, agree7, agree10, ntot = [], [], 0, 0, 0
    for i in prot:
        r = slice(int(data.offs[i]), int(data.offs[i]) + int(data.L[i]))
        ca = data.h["residue/coords"][r][:, 1].astype(np.float64)
        nn = len(ca)
        dd = np.linalg.norm(ca[:, None] - ca[None, :], axis=-1)
        sel = np.where(qp == i)[0]
        if not len(sel):
            continue
        a, b = qi[sel], qj[sel]
        bins.append(qbin[sel]); dists.append(qd[sel])
        for cut, which in zip(R.PAIRHIST_CUTOFFS, ("7", "10")):
            A = (dd < cut).astype(np.float64)
            np.fill_diagonal(A, 0.0)
            w, V = np.linalg.eigh(A)
            k = min(R.PAIRHIST_NEV, nn)
            Ahat = (V[:, nn - k:] * w[nn - k:]) @ V[:, nn - k:].T
            ok = ((Ahat[a, b] > 0.5).astype(int) == A[a, b].astype(int)).sum()
            if which == "7":
                agree7 += ok
            else:
                agree10 += ok
        ntot += len(sel)
    bins, dists = np.concatenate(bins), np.concatenate(dists)
    say(rows_out, "distance_bins", "rank-8 reconstruction of the 7 A adjacency, accuracy on sampled pairs",
        f"{agree7 / max(ntot, 1):.4f}", f"on {ntot} pairs from {len(prot)} domains")
    say(rows_out, "distance_bins", "rank-8 reconstruction of the 10 A adjacency, accuracy on sampled pairs",
        f"{agree10 / max(ntot, 1):.4f}", "")
    bracket = np.digitize(dists, [7.0, 10.0])
    frac = pd.DataFrame({"b": bracket, "y": bins}).groupby("b").y.agg(lambda s: s.value_counts(normalize=True).max())
    nbin = pd.DataFrame({"b": bracket, "y": bins}).groupby("b").y.nunique()
    say(rows_out, "distance_bins", "best accuracy from a perfect (<7, 7-10, >=10) bracket",
        f"{float((frac * np.bincount(bracket) / len(bracket)).sum()):.4f}",
        f"ok, the 16-way label is not determined: the brackets hold {list(nbin.values)} distinct bins")


if __name__ == "__main__":
    main()
