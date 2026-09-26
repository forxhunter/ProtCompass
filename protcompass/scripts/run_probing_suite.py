"""Run the rebuilt IIP probing suite on every encoder in protcompass/probing_embeddings/ and on reference feature sets.

Same protocol as protcompass/probe_fix/fixed_probes.py (standardised features and targets, ridge or softmax linear
readout, MLP-2 with 128 hidden units, AdamW lr 1e-3, weight decay 1e-4, batch 2048, at most 30 epochs, early stopping
on validation loss with patience 3, best state restored, scores on a held-out test split), run on the GPU when present.
Proteins sampled for the residue- and pair-level probes are the ones used for the reference rows in
probing_data/DIAGNOSTICS.md (seed 0; 3,000 train, 500 validation, 1,000 test domains, superfamily-disjoint split).

Reference feature sets (length, position, amino-acid one-hot / composition) go through the same code, so the floor and
the encoder scores are comparable line by line.

  python run_probing_suite.py                 # everything not yet scored
  python run_probing_suite.py esm2_650m ref_aa
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import balanced_accuracy_score, f1_score, r2_score

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
# The released run uses the defaults. The confirmation run of scripts/confirmation_merge.py points all three at the
# merged dataset instead, so the same scoring code is used for both and nothing here is special-cased.
EMB = Path(os.environ.get("PROBING_EMB") or ROOT / "probing_embeddings")
OUT = Path(os.environ.get("PROBING_OUT") or ROOT / "probing_results")
LABELS = Path(os.environ.get("PROBING_LABELS") or ROOT / "probing_data" / "labels.h5")
SPLIT, N_PROT = "superfamily", {"train": 3000, "val": 500, "test": 1000}
PARTS = {"train": 0, "val": 1, "test": 2}
HIDDEN, MAX_EPOCHS, PATIENCE, BATCH, LR, WD = 128, 30, 3, 2048, 1e-3, 1e-4
RIDGE_GRID = (1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3, 1e4)
MLP_SEEDS = (0, 1, 2)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------------- ref_pairhist bin edges
# The joint (sequence separation x CA-CA distance) histogram of the contact map. Bin edges are chosen so that no edge
# coincides with a cutoff used by any label in the suite: the contact cutoff is 8.0 A (contact_count, contact_order),
# the core radius is 10.0 A, the pair label bins are multiples of DIST_MAX/DIST_BINS = 2.5 A, the contact-order
# separation floor is |i-j| >= 4 and the sampled-pair floor is |i-j| >= 6. The 8 A contact set is therefore not a
# union of these bins, so no label is a closed-form function of the histogram. This is the same rule that moved the
# `ref_coords` shells off 8 A, and it removes only the exact coincidence: the histogram remains a near-sufficient
# statistic for anything defined by the pair distribution, which is the point of the floor, not a defect in it.
PAIRHIST_SEP_EDGES = np.array([1, 2, 3, 5, 7, 10, 14, 20, 29, 42, 61, 89, 130, np.inf])         # 13 bins
PAIRHIST_DIST_EDGES = np.array([0.0, 4.5, 6.0, 7.0, 9.0, 11.0, 13.5, 17.0, 21.0, 29.0, 44.0, np.inf])  # 11 bins
PAIRHIST_CUTOFFS = (7.0, 10.0)     # contact maps for the spectral block; 8.0 A is avoided for the reason above
PAIRHIST_NEV = 8                   # leading eigenvalues / eigenvectors kept per cutoff

RESIDUE_PROBES = [  # name, label dataset(s), kind, n_out, IIP
    ("phi_angle_bins", "phi_bin", "clf", 36, "I1"), ("psi_angle_bins", "psi_bin", "clf", 36, "I1"),
    ("phi_circular", ("phi_sin", "phi_cos"), "circ", 2, "I1"), ("psi_circular", ("psi_sin", "psi_cos"), "circ", 2, "I1"),
    ("local_curvature", "curvature", "reg", 1, "I2"),
    ("dssp_3", "dssp3", "clf", 3, "I3"), ("dssp_8", "dssp8", "clf", 8, "I3"),
    ("rsa", "rsa", "reg", 1, "I4"), ("rsa_bins", "rsa_bin", "clf", 3, "I4"),
    ("bfactor", "bfactor_z", "reg", 1, "I5"),
    ("core_surface", "core", "clf", 2, "I6"), ("contact_density", "contact_density_bin", "clf", 4, "I6"),
    ("sequence_recovery", "aa", "clf", 20, "I10"),
]
PROTEIN_REG = ["contact_order", "radius_of_gyration", "asphericity"]

# Reference sets whose residue features contain the residue's own amino-acid one-hot. For those the
# `sequence_recovery` label is a column of the feature matrix, so the probe reads its input back and scores 1.000 at
# every readout and seed; that is not a floor, it is an identity. Stated once here and applied in `run_source`, rather
# than as a per-source special case. `ref_aa` is the one-hot itself; `ref_seq_window` covers offsets -8..+8, which
# includes offset 0. The coordinate floors are not listed: N, CA, C, O, CB do not determine the residue type, and
# `ref_pairhist` is a function of CA positions only, so none of them contains the label.
SELF_IDENTITY_SOURCES = {"ref_aa", "ref_seq_window", "ref_seq_expert", "ref_seq_global"}


# ----------------------------------------------------------------------------- training
def standardise(train, *others):
    mu, sd = train.mean(0, keepdims=True), train.std(0, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return [torch.from_numpy(((a - mu) / sd).astype(np.float16)).to(DEV) for a in (train, *others)]


def ridge(Xtr, ytr, Xva, yva, Xte):
    """Closed-form ridge, float64 Gram matrix built in chunks; alpha chosen on validation MSE. y may have several columns."""
    y = torch.from_numpy(np.atleast_2d(ytr.T).T.astype(np.float64)).to(DEV)
    mu, sd = y.mean(0, keepdim=True), y.std(0, keepdim=True) + 1e-8
    z = (y - mu) / sd
    d = Xtr.shape[1]
    chunk = max(4096, 32_000_000 // d)        # about 256 MB per float64 chunk, whatever the embedding dimension
    G, b = torch.zeros(d, d, dtype=torch.float64, device=DEV), torch.zeros(d, z.shape[1], dtype=torch.float64, device=DEV)
    for s in range(0, len(Xtr), chunk):
        x = Xtr[s:s + chunk].double()
        G += x.T @ x; b += x.T @ z[s:s + chunk]
        del x
    yv = torch.from_numpy(np.atleast_2d(yva.T).T.astype(np.float64)).to(DEV)
    best = None
    for a in RIDGE_GRID:
        w = torch.linalg.solve(G + a * len(Xtr) / 1000.0 * torch.eye(d, dtype=torch.float64, device=DEV), b)
        mse = float((((Xva.double() @ w) * sd + mu - yv) ** 2).mean())
        if best is None or mse < best[0]:
            best = (mse, w)
    return ((Xte.double() @ best[1]) * sd + mu).cpu().numpy()



# Fixed per-residue property scales. None is fitted: these are the published tables a pre-deep-learning method
# would have used, and they are what makes `ref_seq_expert` the sequence-side analogue of `ref_global` rather than
# another one-hot window. Chou-Fasman helix and sheet propensities, Kyte-Doolittle hydropathy, Grantham volume,
# Zimmerman polarity, charge at pH 7, and Vihinen flexibility.
AA_SCALES = {
    #        P(a)  P(b)   KD    vol   pol   chg   flex
    "A": (1.42, 0.83,  1.8,  31.0,  0.0,  0.0, 0.984),
    "C": (0.70, 1.19,  2.5,  55.0,  1.48, 0.0, 0.906),
    "D": (1.01, 0.54, -3.5,  54.0, 49.7, -1.0, 1.068),
    "E": (1.51, 0.37, -3.5,  83.0, 49.9, -1.0, 1.094),
    "F": (1.13, 1.38,  2.8, 132.0,  0.35, 0.0, 0.915),
    "G": (0.57, 0.75, -0.4,   3.0,  0.0,  0.0, 1.031),
    "H": (1.00, 0.87, -3.2,  96.0, 51.6,  0.1, 0.950),
    "I": (1.08, 1.60,  4.5, 111.0,  0.13, 0.0, 0.927),
    "K": (1.16, 0.74, -3.9, 119.0, 49.5,  1.0, 1.102),
    "L": (1.21, 1.30,  3.8, 111.0,  0.13, 0.0, 0.935),
    "M": (1.45, 1.05,  1.9, 105.0,  1.43, 0.0, 0.952),
    "N": (0.67, 0.89, -3.5,  56.0,  3.38, 0.0, 1.048),
    "P": (0.57, 0.55, -1.6,  32.5,  1.58, 0.0, 1.049),
    "Q": (1.11, 1.10, -3.5,  85.0,  3.53, 0.0, 1.037),
    "R": (0.98, 0.93, -4.5, 124.0, 52.0,  1.0, 1.008),
    "S": (0.77, 0.75, -0.8,  32.0,  1.67, 0.0, 1.046),
    "T": (0.83, 1.19, -0.7,  61.0,  1.66, 0.0, 0.997),
    "V": (1.06, 1.70,  4.2,  84.0,  0.13, 0.0, 0.931),
    "W": (1.08, 1.37, -0.9, 170.0,  2.10, 0.0, 0.904),
    "Y": (0.69, 1.47, -1.3, 136.0,  1.61, 0.0, 0.929),
}
SCALES = np.array([AA_SCALES[c] for c in "ACDEFGHIKLMNPQRSTVWY"], dtype=np.float32)



class PairNet(nn.Module):
    """Shared projection of both residues, then an MLP on [p_i, p_j, p_i * p_j, |p_i - p_j|]."""

    def __init__(self, d_in, n_out, width=HIDDEN):
        super().__init__()
        self.proj = nn.Linear(d_in, width)
        self.head = nn.Sequential(nn.Linear(4 * width, width), nn.ReLU(), nn.Linear(width, n_out))

    def forward(self, xi, xj):
        a, b = self.proj(xi), self.proj(xj)
        return self.head(torch.cat([a, b, a * b, (a - b).abs()], -1))


def fit(X, y, kind, n_out, hidden, seed, pairs=None):
    """X: dict part -> standardised fp16 tensor on DEV. y: dict part -> numpy. pairs: dict part -> (i, j) local row tensors."""
    torch.manual_seed(seed)
    d = X["train"].shape[1]
    if pairs is not None:
        net = PairNet(d, n_out)
    elif hidden:
        net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_out))
    else:
        net = nn.Linear(d, n_out)
    net = net.to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    if kind == "clf":
        t = {p: torch.from_numpy(y[p].astype(np.int64)).to(DEV) for p in ("train", "val")}
        loss_fn, mu, sd = nn.functional.cross_entropy, None, None
    else:
        ytr = np.atleast_2d(y["train"].T).T.astype(np.float32)
        mu, sd = ytr.mean(0, keepdims=True), ytr.std(0, keepdims=True) + 1e-8
        t = {p: torch.from_numpy(((np.atleast_2d(y[p].T).T - mu) / sd).astype(np.float32)).to(DEV) for p in ("train", "val")}
        loss_fn = lambda out, tgt: ((out - tgt) ** 2).mean()

    def forward(part, idx):
        if pairs is None:
            return net(X[part][idx].float())
        i, j = pairs[part]
        return net(X[part][i[idx]].float(), X[part][j[idx]].float())

    def predict(part, n):
        out = [forward(part, torch.arange(s, min(s + 65536, n), device=DEV)) for s in range(0, n, 65536)]
        return torch.cat(out)

    n_tr, n_va, n_te = (len(y[p]) for p in ("train", "val", "test"))
    batch = min(BATCH, max(32, n_tr // 8))
    best, state, bad = float("inf"), None, 0
    for _ in range(MAX_EPOCHS):
        net.train()
        perm = torch.randperm(n_tr, device=DEV)
        for s in range(0, n_tr, batch):
            idx = perm[s:s + batch]
            loss = loss_fn(forward("train", idx), t["train"][idx])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            val = float(loss_fn(predict("val", n_va), t["val"]))
        if val < best - 1e-5:
            best, bad, state = val, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    net.load_state_dict(state); net.eval()
    with torch.no_grad():
        out = predict("test", n_te)
    return out.argmax(-1).cpu().numpy() if kind == "clf" else out.cpu().numpy() * sd + mu


def score(kind, y, pred, y_train):
    if kind == "reg":
        pred = pred.reshape(-1)
        ok = np.std(pred) > 0
        return {"pearson": float(pearsonr(y, pred)[0]) if ok else 0.0, "spearman": float(spearmanr(y, pred)[0]) if ok else 0.0,
                "r2": float(r2_score(y, pred))}
    if kind == "circ":  # y and pred are (n, 2) = (sin, cos); 1 means exact, 0 means no angular information
        ang_t, ang_p = np.arctan2(y[:, 0], y[:, 1]), np.arctan2(pred[:, 0], pred[:, 1])
        return {"circ_cos": float(np.cos(ang_t - ang_p).mean()), "circ_mae_deg": float(np.degrees(np.abs(np.angle(np.exp(1j * (ang_t - ang_p))))).mean())}
    majority = np.bincount(y_train).argmax()
    return {"accuracy": float((pred == y).mean()), "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)), "majority_rate": float((y == majority).mean())}


def run_probe(X, y, kind, n_out, pairs=None):
    """Returns list of (readout, seed, score dict)."""
    out = []
    if pairs is None:
        if kind == "clf":
            out.append(("linear", 0, score(kind, y["test"], fit(X, y, kind, n_out, 0, 0), y["train"])))
        else:
            out.append(("linear", 0, score(kind, y["test"], ridge(X["train"], y["train"], X["val"], y["val"], X["test"]), y["train"])))
    for seed in MLP_SEEDS:
        out.append(("pair_mlp" if pairs is not None else "mlp2", seed, score(kind, y["test"], fit(X, y, kind, n_out, HIDDEN, seed, pairs), y["train"])))
    return out


# ----------------------------------------------------------------------------- data
class Data:
    def __init__(self):
        h = h5py.File(LABELS, "r")
        self.P, self.R, self.Q = h["protein"], h["residue"], h["pair"]
        self.L = self.P["length"][:].astype(np.int64); self.offs = self.P["offset"][:].astype(np.int64)
        self.split = self.P[f"split_{SPLIT}"][:]
        rng = np.random.default_rng(0)   # same draw as probing_data/diagnose_probing_data.py
        self.chosen = {p: np.sort(rng.choice(np.where(self.split == c)[0], size=min(N_PROT[p], int((self.split == c).sum())), replace=False))
                       for p, c in PARTS.items()}
        self.rows = {p: np.concatenate([np.arange(self.offs[i], self.offs[i] + self.L[i]) for i in self.chosen[p]]) for p in PARTS}
        self.prot_idx = {p: np.where(self.split == c)[0] for p, c in PARTS.items()}
        self.h = h
        self.aa, self.pos, self.pidx = self.R["aa"][:], self.R["position"][:].astype(np.float64), self.R["protein_index"][:]
        n_res = len(self.aa)
        qp, qi, qj = self.Q["protein_index"][:], self.Q["i"][:].astype(np.int64), self.Q["j"][:].astype(np.int64)
        qbin = self.Q["distance_bin"][:].astype(np.int64)
        self.pairs, self.pair_y = {}, {}
        for p in PARTS:
            local = -np.ones(n_res, dtype=np.int64); local[self.rows[p]] = np.arange(len(self.rows[p]))
            sel = np.where(np.isin(qp, self.chosen[p]))[0]
            self.pairs[p] = (torch.from_numpy(local[self.offs[qp[sel]] + qi[sel]]).to(DEV), torch.from_numpy(local[self.offs[qp[sel]] + qj[sel]]).to(DEV))
            self.pair_y[p] = qbin[sel]

    def coord_features_residue(self, prot_idx):
        """A generic, non-learned descriptor of the backbone around each residue.

        The point of this reference is symmetry with `ref_aa`: an encoder that is handed residue identity gets no
        credit for recovering it, and an encoder that is handed coordinates should get no credit for recovering
        what a fixed descriptor of those same coordinates already exposes. The descriptor is deliberately generic,
        built only from CA-CA distances, so it is not tailored to any one probed property: sorted distances to the
        16 nearest CA atoms, distances along the chain to i+-1..4, and neighbour counts in five shells.
        """
        # No shell radius may coincide with a label cutoff: the 8 A shell was bit-identical to the `contact_count`
        # label on every residue tested, which made the contact-density probe circular. Note that this only removes
        # an exact coincidence. Sorted kNN distances remain a sufficient statistic for the neighbour count at any
        # radius, so a probe that is a deterministic function of the backbone cannot be separated from a backbone
        # descriptor by any choice of features; that is a property of the probe, argued in the text, not a result.
        K, SHELLS = 16, (5.0, 7.0, 9.0, 11.0, 13.0)
        out = []
        for i in prot_idx:
            r = slice(int(self.offs[i]), int(self.offs[i]) + int(self.L[i]))
            ca = self.h["residue/coords"][r][:, 1].astype(np.float64)
            n = len(ca)
            d = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
            np.fill_diagonal(d, np.inf)
            knn = np.sort(d, axis=1)[:, :K]
            knn[~np.isfinite(knn)] = 99.0
            chain = np.full((n, 8), 99.0)
            for c, off in enumerate((-4, -3, -2, -1, 1, 2, 3, 4)):
                j = np.arange(n) + off
                ok = (j >= 0) & (j < n)
                chain[ok, c] = d[np.arange(n)[ok], j[ok]]
            shells = np.stack([(d < s).sum(1) for s in SHELLS], 1).astype(np.float64)
            out.append(np.concatenate([knn, chain, shells], 1))
        return np.concatenate(out).astype(np.float32)

    def backbone_features_residue(self, prot_idx, W=2, K=16):
        """The input-matched floor: a generic non-learned function of the atoms an encoder is actually handed.

        `ref_coords` used CA only, which understates what a structure encoder can compute: phi is the dihedral of
        C(i-1), N(i), CA(i), C(i), so any model given N, CA and C recovers it exactly. This descriptor uses all five
        stored atoms (N, CA, C, O, CB) over a +-W residue window, as pairwise distances plus one chirality sign per
        residue offset, and keeps the kNN CA distances for the non-local part. Distances and a chirality sign
        determine the local geometry up to a rigid motion, so this is the correct upper bound for a probe whose label
        is a function of the backbone.
        """
        out = []
        for i in prot_idx:
            r = slice(int(self.offs[i]), int(self.offs[i]) + int(self.L[i]))
            c = self.h["residue/coords"][r].astype(np.float64)      # (L, 5, 3)
            n = len(c)
            idx = np.clip(np.arange(n)[:, None] + np.arange(-W, W + 1)[None, :], 0, n - 1)
            win = c[idx].reshape(n, (2 * W + 1) * 5, 3)             # atoms of the window, centred below
            win = win - c[:, 1][:, None, :]
            d = np.linalg.norm(win[:, :, None, :] - win[:, None, :, :], axis=-1)
            iu = np.triu_indices(win.shape[1], 1)
            dist = d[:, iu[0], iu[1]]
            # one chirality sign per offset: the sign of the volume spanned by N, CA, C of that residue
            sign = []
            for k in range(2 * W + 1):
                a, b, e = win[:, 5 * k], win[:, 5 * k + 1], win[:, 5 * k + 2]
                sign.append(np.sign(np.einsum("ij,ij->i", np.cross(a - b, e - b), win[:, 5 * W + 1] - b + 1e-9)))
            ca = c[:, 1]
            dd = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
            np.fill_diagonal(dd, np.inf)
            knn = np.sort(dd, axis=1)[:, :K]
            knn[~np.isfinite(knn)] = 99.0
            shells = np.stack([(dd < s).sum(1) for s in (5.0, 7.0, 9.0, 11.0, 13.0)], 1).astype(np.float64)
            out.append(np.concatenate([dist, np.stack(sign, 1), knn, shells], 1))
        return np.concatenate(out).astype(np.float32)

    def global_features_residue(self, prot_idx, W=2, K=16):
        """`ref_backbone` plus non-local context, with every label-equivalent feature removed.

        Rule applied mechanically: a floor feature may not be a closed-form function of any label in the suite. Two
        earlier versions of this descriptor broke it. A principal-axis coordinate block made the pairwise CA distance
        exact (7e-14), which is the `distance_bins` label; a gyration-eigenvalue block made radius of gyration and
        asphericity exact (2e-6, 9e-8). Both are gone. What remains is the residue's distance and rank relative to the
        centroid, which fix where a residue sits in the fold without fixing any pair, and the leading eigenvector
        components of the contact map, which describe the fold without reconstructing coordinates.
        """
        out = []
        for i in prot_idx:
            base = self.backbone_features_residue([i], W=W, K=K)
            r = slice(int(self.offs[i]), int(self.offs[i]) + int(self.L[i]))
            ca = self.h["residue/coords"][r][:, 1].astype(np.float64)
            c = ca - ca.mean(0)
            dist = np.linalg.norm(c, axis=1, keepdims=True)
            rank = (np.argsort(np.argsort(dist.ravel())) / max(len(c) - 1, 1))[:, None]
            d = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
            A = (d < 8.0).astype(np.float64)
            np.fill_diagonal(A, 0.0)
            k = min(8, len(A) - 1)
            if k > 0:
                vecs = np.linalg.eigh(A)[1][:, -k:]
                spec = np.pad(vecs * np.sign(vecs.sum(0)), ((0, 0), (0, 8 - k)))
            else:
                spec = np.zeros((len(A), 8))
            out.append(np.concatenate([base, dist, rank, spec], 1))
        return np.concatenate(out).astype(np.float32)

    def global_features_protein(self, prot_idx):
        """Protein-level twin. The gyration eigenvalues and the contact-order statistic are removed for the same
        reason. The pairwise-distance histogram is kept and flagged: radius of gyration is a function of the second
        moment of that distribution, so Rg cannot be separated from any faithful descriptor of the input."""
        BINS = np.linspace(0, 60, 25)
        out = []
        for i in prot_idx:
            r = slice(int(self.offs[i]), int(self.offs[i]) + int(self.L[i]))
            ca = self.h["residue/coords"][r][:, 1].astype(np.float64)
            n = len(ca)
            d = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
            iu = np.triu_indices(n, 1)
            hist = np.histogram(d[iu], bins=BINS)[0] / max(len(iu[0]), 1)
            A = (d < 8.0).astype(np.float64)
            np.fill_diagonal(A, 0.0)
            ev = np.sort(np.linalg.eigvalsh(A))[::-1][:16]
            ev = np.pad(ev, (0, max(0, 16 - len(ev)))) / max(n, 1)
            out.append(np.concatenate([hist, ev, [np.log(n)]]))
        return np.asarray(out, dtype=np.float32)

    def _pairhist_one(self, i):
        """Everything `ref_pairhist` needs for one protein, from its CA coordinates alone.

        A descriptor built to win rather than to be safe: the joint histogram of
        (sequence separation, CA-CA distance) over all residue pairs, which is the smallest generic extension of the
        pairwise-distance histogram `ref_global` already keeps, plus the leading eigenvalues of the contact map at two
        cutoffs and a few global spectral summaries. Nothing here is specific to any probed label and nothing is fitted.

        Returns (glob, rows): glob is the protein-level vector, rows is (L, .) with the protein-level vector repeated,
        the residue's own row of the same histogram, and the residue's components of the leading contact-map
        eigenvectors. Contact order is a length-residualised global statistic, so the residue block exposes the
        residue's own separation-by-distance profile, which is the residue-level analogue of the label.
        """
        S, D = len(PAIRHIST_SEP_EDGES) - 1, len(PAIRHIST_DIST_EDGES) - 1
        r = slice(int(self.offs[i]), int(self.offs[i]) + int(self.L[i]))
        ca = self.h["residue/coords"][r][:, 1].astype(np.float64)
        n = len(ca)
        d = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
        sep = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
        sb = np.digitize(sep, PAIRHIST_SEP_EDGES) - 1            # -1 on the diagonal, which is dropped below
        db = np.digitize(d, PAIRHIST_DIST_EDGES) - 1
        flat, ok = sb * D + db, sb >= 0
        iu = np.triu(ok, k=1)
        H = np.bincount(flat[iu], minlength=S * D).astype(np.float64) / max(int(iu.sum()), 1)
        cnt = np.bincount((np.arange(n)[:, None] * (S * D) + flat)[ok], minlength=n * S * D).reshape(n, S * D)
        rows_h = cnt / np.maximum(ok.sum(1), 1)[:, None]         # the residue's own separation-by-distance profile
        spec, rowspec = [], []
        for cut in PAIRHIST_CUTOFFS:
            A = (d < cut).astype(np.float64)
            np.fill_diagonal(A, 0.0)
            deg = A.sum(1)
            w, V = np.linalg.eigh(A)
            p = np.abs(w) / max(np.abs(w).sum(), 1e-12)
            ent = float(-(p * np.log(p + 1e-300)).sum() / np.log(max(n, 2)))
            top = np.pad((w[::-1])[:PAIRHIST_NEV], (0, max(0, PAIRHIST_NEV - n))) / max(n, 1)
            bot = np.pad(w[:PAIRHIST_NEV // 2], (0, max(0, PAIRHIST_NEV // 2 - n))) / max(n, 1)
            spec.append(np.concatenate([top, bot, [deg.mean(), deg.std(), ent]]))
            k = min(PAIRHIST_NEV, n)
            v = V[:, n - k:]
            v = v * np.sign(v.sum(0) + 1e-12)                    # eigenvector signs are arbitrary; fix them
            rowspec.append(np.concatenate([np.pad(v, ((0, 0), (0, PAIRHIST_NEV - k))), (deg / max(n, 1))[:, None]], 1))
        glob = np.concatenate([H, *spec, [np.log(n)]])
        rows = np.concatenate([np.repeat(glob[None], n, 0), rows_h, *rowspec], 1)
        return glob, rows

    def pairhist_features_protein(self, prot_idx):
        """Protein-level `ref_pairhist`: the (separation x distance) histogram plus the spectral block."""
        return np.asarray([self._pairhist_one(i)[0] for i in prot_idx], dtype=np.float32)

    def pairhist_features_residue(self, prot_idx):
        """Residue-level `ref_pairhist`: the protein-level vector plus the residue's own row of the histogram."""
        return np.concatenate([self._pairhist_one(i)[1] for i in prot_idx]).astype(np.float32)

    def coord_features_protein(self, prot_idx):
        """Protein-level twin: the normalised histogram of pairwise CA distances, plus log length."""
        BINS = np.linspace(0, 60, 25)
        out = []
        for i in prot_idx:
            r = slice(int(self.offs[i]), int(self.offs[i]) + int(self.L[i]))
            ca = self.h["residue/coords"][r][:, 1].astype(np.float64)
            d = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
            iu = np.triu_indices(len(ca), 1)
            hist = np.histogram(d[iu], bins=BINS)[0] / max(len(iu[0]), 1)
            out.append(np.concatenate([hist, [np.log(len(ca))]]))
        return np.asarray(out, dtype=np.float32)

    def seq_window_features(self, rows):
        """Amino-acid one-hot over a +-8 window, the sequence twin of ref_coords.

        `ref_aa` sees one residue and `ref_coords` sees a neighbourhood, so comparing an encoder against them is not
        a matched comparison: local context is available to any sequence model for free. This reference gives the
        same context to the floor. Positions past the chain ends are zero, with a flag column per offset.
        """
        W = 8
        pidx, pos = self.pidx[rows], self.pos[rows].astype(np.int64)
        L = self.L[pidx]
        out = np.zeros((len(rows), (2 * W + 1) * 21), dtype=np.float32)
        for k, off in enumerate(range(-W, W + 1)):
            j = pos + off
            ok = (j >= 0) & (j < L)
            base = k * 21
            out[ok, base + self.aa[self.offs[pidx[ok]] + j[ok]]] = 1.0
            out[~ok, base + 20] = 1.0                      # out-of-chain flag
        return out

    def seq_expert_features(self, rows, W=8):
        """The sequence-side floor built with the effort the coordinate side received.

        `ref_seq_window` is a one-hot window and nothing more, while the coordinate side has `ref_global` and
        `ref_pairhist`. One objection was that the two sides therefore received different amounts of
        descriptor-design effort and that the headline separation could be that difference. This is the strongest
        fixed sequence descriptor we can write without fitting anything: the one-hot window, the same window under
        seven published per-residue property scales, smoothed averages of those scales over three window widths,
        which is what a Chou-Fasman or a Kyte-Doolittle profile is, the composition of the whole chain, and the
        residue's position. No parameter here is fitted on any label.
        """
        base = self.seq_window_features(rows)
        pidx, pos = self.pidx[rows], self.pos[rows].astype(np.int64)
        L = self.L[pidx]
        aa = self.aa
        n_scale = SCALES.shape[1]
        win = np.zeros((len(rows), (2 * W + 1) * n_scale), dtype=np.float32)
        for k, off in enumerate(range(-W, W + 1)):
            j = np.clip(pos + off, 0, L - 1)
            win[:, k * n_scale:(k + 1) * n_scale] = SCALES[np.clip(aa[self.offs[pidx] + j], 0, 19)]
        smooth = []
        for w in (3, 7, 15):
            acc = np.zeros((len(rows), n_scale), dtype=np.float32)
            for off in range(-(w // 2), w // 2 + 1):
                j = np.clip(pos + off, 0, L - 1)
                acc += SCALES[np.clip(aa[self.offs[pidx] + j], 0, 19)]
            smooth.append(acc / w)
        comp = np.zeros((len(self.L), 20), dtype=np.float32)
        np.add.at(comp, (self.pidx, self.aa), 1.0)
        comp = comp / self.L[:, None]
        dterm = np.minimum(pos, L - 1 - pos).astype(np.float32)
        geom = np.stack([pos / np.maximum(L - 1, 1), dterm, np.log1p(dterm), dterm / L, np.log(L)], 1).astype(np.float32)
        return np.concatenate([base, win, *smooth, comp[pidx], geom], 1).astype(np.float32)

    def seq_global_features(self, rows):
        """The sequence-side analogue of `ref_global`: `ref_seq_expert` plus non-local context.

        `ref_global` was added because `ref_backbone` is local and the non-local probes still favoured the
        encoders. The sequence side needs the same addition to be a fair comparison, so this adds what a fixed
        function of the sequence can say about the chain as a whole: the dipeptide composition of the chain (400
        features, the standard non-learned sequence descriptor), the composition of the residue's own half of the
        chain, and the chain-level mean of each property scale in the residue's neighbourhood at two long scales.
        Nothing is fitted.
        """
        base = self.seq_expert_features(rows)
        pidx, pos = self.pidx[rows], self.pos[rows].astype(np.int64)
        L = self.L[pidx]
        aa = self.aa
        # dipeptide composition per chain, computed once
        if not hasattr(self, "_dipep"):
            d = np.zeros((len(self.L), 400), dtype=np.float32)
            for i in range(len(self.L)):
                r = slice(int(self.offs[i]), int(self.offs[i]) + int(self.L[i]))
                a = np.clip(aa[r], 0, 19).astype(np.int64)
                if len(a) > 1:
                    np.add.at(d[i], a[:-1] * 20 + a[1:], 1.0)
                    d[i] /= max(len(a) - 1, 1)
            self._dipep = d
        half = (pos > (L // 2)).astype(np.float32)[:, None]
        long_ctx = []
        for w in (31, 61):
            acc = np.zeros((len(rows), SCALES.shape[1]), dtype=np.float32)
            for off in range(-(w // 2), w // 2 + 1):
                j = np.clip(pos + off, 0, L - 1)
                acc += SCALES[np.clip(aa[self.offs[pidx] + j], 0, 19)]
            long_ctx.append(acc / w)
        return np.concatenate([base, self._dipep[pidx], half, *long_ctx], 1).astype(np.float32)

    def residue_features(self, source):
        if source == "ref_seq_global":
            return {p: self.seq_global_features(self.rows[p]) for p in PARTS}
        if source == "ref_backbone_w0":
            # an ablation: the same descriptor with no window, so the only atoms it sees are
            # the residue's own five. phi needs C(i-1) and psi needs N(i+1), so if the floor were carrying a
            # precomputed angle rather than recovering it from the window, this would still score high.
            return {p: self.backbone_features_residue(self.chosen[p], W=0) for p in PARTS}
        if source == "ref_seq_expert":
            return {p: self.seq_expert_features(self.rows[p]) for p in PARTS}
        if source == "ref_coords":
            return {p: self.coord_features_residue(self.chosen[p]) for p in PARTS}
        if source == "ref_backbone":
            return {p: self.backbone_features_residue(self.chosen[p]) for p in PARTS}
        if source == "ref_global":
            return {p: self.global_features_residue(self.chosen[p]) for p in PARTS}
        if source == "ref_pairhist":
            return {p: self.pairhist_features_residue(self.chosen[p]) for p in PARTS}
        if source == "ref_seq_window":
            return {p: self.seq_window_features(self.rows[p]) for p in PARTS}
        if source.startswith("ref_"):
            out = {}
            for p in PARTS:
                r = self.rows[p]; l, i = self.L[self.pidx[r]].astype(np.float64), self.pos[r]
                dterm = np.minimum(i, l - 1 - i)
                out[p] = {"ref_length": np.stack([np.log(l), l / 600.0], 1),
                          "ref_position": np.stack([i / np.maximum(l - 1, 1), dterm, np.log1p(dterm), dterm / l, np.log(l)], 1),
                          "ref_aa": np.eye(20, dtype=np.float32)[self.aa[r]]}[source].astype(np.float32)
            return out
        emb = np.load(EMB / source / "residue_fp16.npy", mmap_mode="r")
        return {p: np.asarray(emb[self.rows[p]], dtype=np.float32) for p in PARTS}

    def protein_features(self, source):
        L = self.L.astype(np.float64)
        if source == "ref_global":
            return {p: self.global_features_protein(self.prot_idx[p]) for p in PARTS}
        if source == "ref_pairhist":
            return {p: self.pairhist_features_protein(self.prot_idx[p]) for p in PARTS}
        if source in ("ref_coords", "ref_backbone", "ref_backbone_w0"):
            return {p: self.coord_features_protein(self.prot_idx[p]) for p in PARTS}
        if source in ("ref_seq_expert", "ref_seq_global"):
            # protein level: composition, the same seven scales averaged over the chain and their standard
            # deviations, plus length
            comp = np.zeros((len(L), 20), dtype=np.float32); np.add.at(comp, (self.pidx, self.aa), 1.0)
            comp = comp / L[:, None]
            sc = SCALES[np.clip(self.aa, 0, 19)]
            mean = np.zeros((len(L), SCALES.shape[1]), dtype=np.float32)
            np.add.at(mean, self.pidx, sc)
            mean /= L[:, None]
            sq = np.zeros_like(mean); np.add.at(sq, self.pidx, sc ** 2)
            sd = np.sqrt(np.maximum(sq / L[:, None] - mean ** 2, 0))
            X = np.concatenate([comp, mean, sd, np.stack([np.log(L), L / 600.0], 1)], 1).astype(np.float32)
            return {p: X[self.prot_idx[p]] for p in PARTS}
        if source in ("ref_aa", "ref_seq_window"):   # at protein level both reduce to composition
            comp = np.zeros((len(L), 20), dtype=np.float32); np.add.at(comp, (self.pidx, self.aa), 1.0)
            X = comp / L[:, None]
        elif source in ("ref_length", "ref_position"):
            X = np.stack([np.log(L), np.log(L) ** 2, L / 600.0], 1)
        else:
            X = np.load(EMB / source / "pooled_fp32.npy")
        return {p: X[self.prot_idx[p]].astype(np.float32) for p in PARTS}


PROJECT_K = int(os.environ.get("PROJECT_K", "0"))


def project(X, k, seed=0):
    """Every source reduced to the same width, so that a probe is handed the same number of features whatever it
    is reading. The floors run from 2 to 357 features and the embeddings from 20 to 2560, and a probe with a fixed
    head reads a wider and better-conditioned input more easily, so the released comparison confounds capacity
    with content. The projection is PCA fitted on the training split only and applied to all three, and a source
    narrower than k is padded with zeros rather than dropped, which costs it nothing."""
    from sklearn.decomposition import PCA
    tr = X["train"]
    if tr.shape[1] <= k:
        pad = lambda a: np.pad(a, ((0, 0), (0, k - a.shape[1])))
        return {p: pad(X[p]) for p in X}
    pca = PCA(n_components=k, random_state=seed).fit(tr)
    return {p: pca.transform(X[p]).astype(np.float32) for p in X}


def run_source(data, source):
    t0, rows = time.time(), []
    def add(probe, iip, level, results):
        for readout, seed, sc in results:
            rows.append(dict(source=source, probe=probe, iip=iip, level=level, readout=readout, seed=seed, **sc))
        print(f"  [{time.time() - t0:5.0f}s] {source}: {probe}", flush=True)

    # Encoders with one vector per protein (GeoBPE, FoldVision) have no residue states: protein-level probes only.
    pooled_only = not source.startswith("ref_") and not (EMB / source / "residue_fp16.npy").exists()
    raw = None if pooled_only else data.residue_features(source)
    dim = data.protein_features(source)["train"].shape[1] if pooled_only else raw["train"].shape[1]
    X = {} if pooled_only else dict(zip(PARTS, standardise(raw["train"], raw["val"], raw["test"]))); del raw
    if PROJECT_K and X:
        X = project({p: (v.cpu().numpy() if hasattr(v, "cpu") else v) for p, v in X.items()}, PROJECT_K)
        X = {p: torch.as_tensor(v, dtype=torch.float32, device=DEV) for p, v in X.items()}
    for name, ds, kind, n_out, iip in ([] if pooled_only else RESIDUE_PROBES):
        if name == "sequence_recovery" and source in SELF_IDENTITY_SOURCES:
            continue
        lab = {p: (np.stack([data.R[d][:][data.rows[p]] for d in ds], 1) if isinstance(ds, tuple) else data.R[ds][:][data.rows[p]]) for p in PARTS}
        ok = {p: (~np.isnan(lab[p]).reshape(len(lab[p]), -1).any(1) if kind != "clf" else lab[p] >= 0) for p in PARTS}
        Xp = {p: X[p][torch.from_numpy(np.where(ok[p])[0]).to(DEV)] for p in PARTS}
        y = {p: (lab[p][ok[p]].astype(np.int64) if kind == "clf" else lab[p][ok[p]].astype(np.float64)) for p in PARTS}
        add(name, iip, "residue", run_probe(Xp, y, kind, n_out)); del Xp
    if not pooled_only:
        add("distance_bins", "I7", "pair", run_probe(X, data.pair_y, "clf", 16, pairs=data.pairs))
    del X; torch.cuda.empty_cache()

    praw = data.protein_features(source)
    PX = dict(zip(PARTS, standardise(praw["train"], praw["val"], praw["test"])))
    if PROJECT_K:
        PX = project({p: (v.cpu().numpy() if hasattr(v, "cpu") else v) for p, v in PX.items()}, PROJECT_K)
        PX = {p: torch.as_tensor(v, dtype=torch.float32, device=DEV) for p, v in PX.items()}
    for k in PROTEIN_REG:
        for suffix, label in (("", k), (f"_len_resid_{SPLIT}", f"{k}_len_resid")):
            y_all = data.P[k + suffix][:].astype(np.float64)
            add(label, "I8", "protein", run_probe(PX, {p: y_all[data.prot_idx[p]] for p in PARTS}, "reg", 1))
    arch = data.P[f"architecture_label_{SPLIT}"][:].astype(np.int64)
    ok = {p: arch[data.prot_idx[p]] >= 0 for p in PARTS}
    PXa = {p: PX[p][torch.from_numpy(np.where(ok[p])[0]).to(DEV)] for p in PARTS}
    add("architecture_label", "I9", "protein", run_probe(PXa, {p: arch[data.prot_idx[p]][ok[p]] for p in PARTS}, "clf", int(arch.max()) + 1))
    df = pd.DataFrame(rows); df.insert(1, "dim", dim)
    sub = "by_source" if not PROJECT_K else f"by_source_k{PROJECT_K}"
    (OUT / sub).mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / sub / f"{source}.csv", index=False)
    print(f"[{source}] dim {dim}, {len(df)} rows, {time.time() - t0:.0f}s", flush=True)


def main():
    (OUT / "by_source").mkdir(parents=True, exist_ok=True)
    wanted = sys.argv[1:] or (["ref_length", "ref_position", "ref_aa", "ref_seq_window", "ref_coords", "ref_backbone", "ref_global",
                               "ref_pairhist"] + sorted(d.name for d in EMB.iterdir() if (d / "meta.json").exists()))
    sub = "by_source" if not PROJECT_K else f"by_source_k{PROJECT_K}"
    todo = [s for s in wanted if not (OUT / sub / f"{s}.csv").exists()]
    print(f"device {DEV}; to score: {todo}", flush=True)
    data = Data()
    for s in todo:
        for attempt in range(6):           # the GPU is shared with extraction jobs; wait and retry rather than lose the run
            try:
                run_source(data, s)
                break
            except (torch.OutOfMemoryError, torch.AcceleratorError, RuntimeError) as e:
                if "out of memory" not in str(e).lower():
                    raise
                torch.cuda.empty_cache()
                print(f"[{s}] CUDA out of memory (attempt {attempt + 1}/6); waiting 5 min for the GPU", flush=True)
                time.sleep(300)
        else:
            print(f"[{s}] skipped: still out of memory after 6 attempts", flush=True)
    frames = [pd.read_csv(f) for f in sorted((OUT / "by_source").glob("*.csv"))]
    if frames and not PROJECT_K:
        pd.concat(frames, ignore_index=True).to_csv(OUT / "scores_long.csv", index=False)
    json.dump({"split": SPLIT, "n_proteins_residue_level": N_PROT, "hidden": HIDDEN, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
               "batch": BATCH, "lr": LR, "weight_decay": WD, "ridge_grid": RIDGE_GRID, "mlp_seeds": MLP_SEEDS}, open(OUT / "protocol.json", "w"), indent=2)


if __name__ == "__main__":
    main()
