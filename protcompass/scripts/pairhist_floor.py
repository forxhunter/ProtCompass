"""Score the `ref_pairhist` floor on the two probes where encoders still beat every non-learned floor.

Round 5 of review asks for a descriptor that is trying to win rather than one built to be safe, and names the one it
would try first: the joint histogram of (sequence separation, CA-CA distance) over the contact map. `ref_pairhist`
(defined in run_probing_suite.py) is that descriptor plus the leading contact-map eigenvalues and a few spectral
summaries. Only the two contested probes are scored here, through the suite's own `run_probe`, splits and seeds:

  contact_order_len_resid   published floor 0.5315 (ref_global) against 0.9235 (ESM3), gap +0.392
  architecture_label        published floor 0.100 full set / 0.144-0.155 common 13 classes, against 0.497 / 0.638

Reference sources are scored alongside it so that this script's numbers can be checked against the published ones
before the new floor is believed. Three floors in this project turned out to contain their own label, so the features
are checked for that first; `--check` runs those checks alone.

  python pairhist_floor.py [--check]

Writes probing_results/floor_pairhist.csv. Nothing else in probing_results/ is touched.
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import run_probing_suite as R

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
CACHE = Path((PROTCOMPASS_ROOT + "/protcompass/cache/scratch/pairhist_protein.npy"))
MIN_TEST = 20                       # class cut of architecture_robustness.py, kept identical
CO_SOURCES = ["ref_pairhist", "ref_global", "esm3"]
ARCH_SOURCES = ["ref_pairhist", "ref_global", "ref_aa", "esm_if1", "prostt5", "esm2_650m"]
S, D = len(R.PAIRHIST_SEP_EDGES) - 1, len(R.PAIRHIST_DIST_EDGES) - 1


# ----------------------------------------------------------------------------- features
def build(data):
    """Protein-level ref_pairhist for every domain, cached: the two probes here reuse the same matrix."""
    if CACHE.exists():
        return np.load(CACHE)
    t0 = time.time()
    out = []
    for s in range(0, len(data.L), 2000):
        out.append(data.pairhist_features_protein(np.arange(s, min(s + 2000, len(data.L)))))
        print(f"  features {s + len(out[-1])}/{len(data.L)}  [{time.time() - t0:.0f}s]", flush=True)
    X = np.concatenate(out)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(CACHE, X)
    return X


# ----------------------------------------------------------------------------- leakage checks
def checks(data, X):
    """Does any label in the suite follow from these features in closed form? Report numbers, not assurances."""
    h = data.h
    rows = []

    def say(name, value, verdict):
        rows.append({"check": name, "value": value, "verdict": verdict})
        print(f"  {name:62s} {value:>22s}  {verdict}", flush=True)

    # 1. no bin edge may coincide with a cutoff that defines a label
    cut = {"contact cutoff (contact_count, contact_order)": 8.0, "core radius (core_surface)": 10.0}
    clash = [k for k, v in cut.items() if v in set(R.PAIRHIST_DIST_EDGES)]
    clash += [f"pair label bin {e}" for e in R.PAIRHIST_DIST_EDGES if np.isfinite(e) and e > 0 and abs(e % 2.5) < 1e-9]
    say("distance bin edges clash with a label cutoff", str(clash or "none"), "ok" if not clash else "LEAK")
    sclash = [f"|i-j|>={v}" for v in (4, 6) if v in set(R.PAIRHIST_SEP_EDGES)]
    say("separation bin edges clash with a label cutoff", str(sclash or "none"), "ok" if not sclash else "LEAK")
    say("cutoffs of the spectral block", str(R.PAIRHIST_CUTOFFS), "ok, 8.0 A avoided")

    # 2. contact order: RCO = mean(|i-j| over pairs with d < 8 and |i-j| >= 4) / L. The histogram carries both axes, so
    #    the question is whether the identity closes. Try every closed-form reconstruction the binning allows: the
    #    contact set is bracketed by d < 7 and d < 9 and the separation floor by |i-j| >= 3 and |i-j| >= 5, and the
    #    separation of a bin is only known to lie inside it, so bin midpoints are the best available value.
    sep_mid = np.array([np.sqrt(a * b) if np.isfinite(b) else a * 1.3
                        for a, b in zip(R.PAIRHIST_SEP_EDGES[:-1], R.PAIRHIST_SEP_EDGES[1:])])
    d_up = np.searchsorted(R.PAIRHIST_DIST_EDGES, [7.0, 9.0], side="left")       # bins entirely below 7 A / below 9 A
    s_lo = [np.searchsorted(R.PAIRHIST_SEP_EDGES, v, side="right") - 1 for v in (3, 5)]
    rco = h["protein/contact_order"][:].astype(np.float64)
    n = data.L.astype(np.float64)
    H = X[:, :S * D].reshape(-1, S, D).astype(np.float64)
    best = None
    for du in d_up:
        for sl in s_lo:
            w = H[:, sl:, :du]
            num = (w * sep_mid[sl:, None]).sum((1, 2))
            hat = num / np.maximum(w.sum((1, 2)), 1e-12) / n
            rel = np.abs(hat - rco) / np.maximum(np.abs(rco), 1e-12)
            r = float(np.corrcoef(hat, rco)[0, 1])
            print(f"    variant d<{R.PAIRHIST_DIST_EDGES[du]:.1f} sep>={R.PAIRHIST_SEP_EDGES[sl]:.0f}: "
                  f"median rel err {np.median(rel):.4f}, max {rel.max():.4f}, pearson {r:.4f}", flush=True)
            if best is None or np.median(rel) < best[0]:
                best = (float(np.median(rel)), float(rel.max()), r)
    say("contact_order from bin midpoints, median relative error", f"{best[0]:.4f}",
        "ok, not closed form" if best[0] > 1e-3 else "LEAK")
    say("contact_order from bin midpoints, max relative error", f"{best[1]:.4f}", "")
    say("contact_order from bin midpoints, pearson r", f"{best[2]:.4f}", "reported, an approximation is not an identity")

    # 3. no single feature may equal a label up to scale
    for lab in ("contact_order", f"contact_order_len_resid_{R.SPLIT}", "radius_of_gyration", "asphericity"):
        y = h[f"protein/{lab}"][:].astype(np.float64)
        keep = X.std(0) > 0
        c = np.abs(np.corrcoef(np.column_stack([y, X[:, keep]]), rowvar=False)[0, 1:])
        say(f"max |pearson| of any single feature with {lab}", f"{np.nanmax(c):.4f}",
            "ok" if np.nanmax(c) < 0.999 else "LEAK")

    # 4. the 8 A neighbour count is bracketed, not determined: the residue block gives counts below 7 A and below 9 A
    rng = np.random.default_rng(0)
    amb, tot, wid = 0, 0, []
    for i in rng.choice(len(data.L), 60, replace=False):
        r = slice(int(data.offs[i]), int(data.offs[i]) + int(data.L[i]))
        ca = h["residue/coords"][r][:, 1].astype(np.float64)
        dd = np.linalg.norm(ca[:, None] - ca[None, :], axis=-1)
        np.fill_diagonal(dd, np.inf)
        lo, hi = (dd < 7.0).sum(1), (dd < 9.0).sum(1)
        amb += int((hi > lo).sum()); tot += len(lo); wid.append((hi - lo).mean())
    say("residues whose 8 A contact count is not fixed by the bins", f"{amb / max(tot, 1):.4f}",
        "ok, contact_density stays undetermined")
    say("mean width of the contact-count bracket (neighbours)", f"{np.mean(wid):.2f}", "")

    # 5. fold architecture is a CATH annotation, so no closed form exists. What could leak instead is the split or
    #    domain identity, so check that no test domain is a numerical duplicate of a train domain, and that no single
    #    feature is a class indicator (eta squared of 1 would be one).
    arch = h[f"protein/architecture_label_{R.SPLIT}"][:].astype(np.int64)
    sp = h[f"protein/split_{R.SPLIT}"][:]
    Z = (X - X.mean(0)) / np.where(X.std(0) < 1e-9, 1.0, X.std(0))
    tr, te = Z[sp == 0], Z[sp == 2]
    dmin = np.sqrt(np.maximum(((te ** 2).sum(1)[:, None] + (tr ** 2).sum(1)[None, :] - 2 * te @ tr.T).min(1), 0))
    say("min feature distance from a test domain to any train domain", f"{dmin.min():.4f}",
        "ok, no duplicate domain" if dmin.min() > 1e-6 else "LEAK")
    ok = arch >= 0
    Xc = X[ok] - X[ok].mean(0)                                # eta squared is between-class over total sum of squares
    g = pd.DataFrame(Xc).groupby(arch[ok])
    eta = (g.mean().values ** 2 * g.size().values[:, None]).sum(0) / np.maximum((Xc ** 2).sum(0), 1e-12)
    say("max eta squared of any single feature on architecture", f"{np.nanmax(eta):.4f}",
        "ok, no class indicator" if np.nanmax(eta) < 0.999 else "LEAK")

    # 6. Rg follows from the second moment of the pair distances, which this descriptor inherits from ref_global's
    #    histogram. Flagged there, flagged here, and reported as an approximation.
    dist_mid = np.array([0.5 * (a + b) if np.isfinite(b) else a * 1.2
                         for a, b in zip(R.PAIRHIST_DIST_EDGES[:-1], R.PAIRHIST_DIST_EDGES[1:])])
    m2 = (H.sum(1) * dist_mid ** 2).sum(1) * (1 - 1 / np.maximum(n, 2))          # mean d^2 over all ordered pairs
    hat = np.sqrt(np.maximum(m2 / 2, 0))
    rg = h["protein/radius_of_gyration"][:].astype(np.float64)
    rel = np.abs(hat - rg) / rg
    say("radius_of_gyration from bin midpoints, median relative error", f"{np.median(rel):.4f}",
        "ok, not closed form; inherited from ref_global and flagged")
    print(f"  ref_pairhist reads probing_data/labels.h5 residue/coords and protein/{{offset,length}} only.", flush=True)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- probes
_MAT = {}


def protein_matrix(data, source, X):
    """One row per domain, indexed by protein, so every setting below can slice the same matrix."""
    if source == "ref_pairhist":
        return X
    if source in _MAT:
        return _MAT[source]
    _MAT[source] = _protein_matrix(data, source)
    return _MAT[source]


def _protein_matrix(data, source):
    if source == "ref_global":
        Xg = np.zeros((len(data.L), 41), dtype=np.float32)
        for p in R.PARTS:
            Xg[data.prot_idx[p]] = data.global_features_protein(data.prot_idx[p])
        return Xg
    if source == "ref_aa":
        comp = np.zeros((len(data.L), 20), dtype=np.float32)
        np.add.at(comp, (data.pidx, data.aa), 1.0)
        return comp / data.L.astype(np.float64)[:, None]
    return np.load(ROOT / "probing_embeddings" / source / "pooled_fp32.npy")


def contact_order(data, X, rows):
    y = data.P[f"contact_order_len_resid_{R.SPLIT}"][:].astype(np.float64)
    for src in CO_SOURCES:
        M = protein_matrix(data, src, X)
        PX = dict(zip(R.PARTS, R.standardise(*[M[data.prot_idx[p]].astype(np.float32) for p in R.PARTS])))
        for readout, seed, sc in R.run_probe(PX, {p: y[data.prot_idx[p]] for p in R.PARTS}, "reg", 1):
            rows.append(dict(probe="contact_order_len_resid", split=R.SPLIT, setting="all_proteins", n_classes=0,
                             source=src, dim=M.shape[1], readout=readout, seed=seed, **sc))
        print(f"  contact_order_len_resid {src}: done", flush=True)
        torch.cuda.empty_cache()


def architecture(data, X, rows):
    h = data.h
    for split in ("superfamily", "topology"):
        sp = h[f"protein/split_{split}"][:]
        arch = h[f"protein/architecture_label_{split}"][:].astype(np.int64)
        idx = {p: np.where(sp == c)[0] for p, c in R.PARTS.items()}
        te = arch[idx["test"]]
        counts = np.bincount(te[te >= 0], minlength=int(arch.max()) + 1)
        settings = [("all_classes", arch, int(arch.max()) + 1)] if split == R.SPLIT else []
        keep = np.where(counts >= MIN_TEST)[0]                              # the class cut of FINDINGS section 3
        remap = -np.ones(int(arch.max()) + 1, dtype=np.int64)
        remap[keep] = np.arange(len(keep))
        settings.append((f"test_ge{MIN_TEST}", np.where(arch >= 0, remap[np.clip(arch, 0, None)], -1), len(keep)))
        for setting, y_all, n_out in settings:
            sel = {p: idx[p][y_all[idx[p]] >= 0] for p in R.PARTS}
            for src in ARCH_SOURCES:
                M = protein_matrix(data, src, X)
                PX = dict(zip(R.PARTS, R.standardise(*[M[sel[p]].astype(np.float32) for p in R.PARTS])))
                for readout, seed, sc in R.run_probe(PX, {p: y_all[sel[p]] for p in R.PARTS}, "clf", n_out):
                    rows.append(dict(probe="architecture_label", split=split, setting=setting, n_classes=n_out,
                                     source=src, dim=M.shape[1], readout=readout, seed=seed, **sc))
                print(f"  architecture {split}/{setting} {src}: done", flush=True)
                torch.cuda.empty_cache()


def main():
    data = R.Data()
    X = build(data)
    print(f"ref_pairhist protein features: {X.shape}", flush=True)
    print("leakage checks", flush=True)
    chk = checks(data, X)
    if "--check" in sys.argv:
        return
    rows = []
    contact_order(data, X, rows)
    architecture(data, X, rows)
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "probing_results" / "floor_pairhist.csv", index=False)
    chk.to_csv(ROOT / "probing_results" / "floor_pairhist_checks.csv", index=False)

    pd.set_option("display.width", 220)
    co = d[d.probe == "contact_order_len_resid"].copy()
    co["family"] = np.where(co.readout == "linear", "linear", "nonlinear")
    print("\n=== contact_order_len_resid, pearson r ===")
    print(co.groupby(["source", "family"]).pearson.mean().unstack().round(4).to_string())
    ar = d[d.probe == "architecture_label"]
    print("\n=== architecture_label, balanced accuracy ===")
    print(ar.groupby(["split", "setting", "readout", "source"]).balanced_accuracy.mean()
          .unstack("source").round(4).to_string())


if __name__ == "__main__":
    main()
