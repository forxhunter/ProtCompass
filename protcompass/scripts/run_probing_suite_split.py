"""Re-score every probe on the `design` and `score` halves of the test split, changing nothing about training.

The pipeline stores aggregate scores, not per-residue predictions, so the probes have to be rerun to get scores on a
partition of the test set. Nothing about the fit is touched: the trainer (`run_probing_suite.fit`, `.ridge`), the
train/val/test protein sets, the readouts, the seeds, the standardisation and the probe list are imported from
`run_probing_suite` and used as they are. The only change is that the test predictions are scored three times, on the
full test set, on the design half and on the score half. The full-test column is a reproduction check against
probing_results/by_source/<source>.csv: it should come back equal to the published number up to GPU nondeterminism.

  python run_probing_suite_split.py                    # everything not yet done
  python run_probing_suite_split.py esm2_650m ref_pairhist

Writes probing_results/by_source_split/<source>.csv with one row per (probe, readout, seed, eval_set).
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import run_probing_suite as R

OUTDIR = R.OUT / "by_source_split"
ASSIGN = R.OUT / "design_score_assignment.csv"
EVAL_SETS = ("test", "design", "score")
# The reference descriptors cost a pairwise distance matrix and an eigendecomposition per protein, and the GPU here is
# shared, so a run can be interrupted and restarted. Cache the descriptor matrices so a restart is cheap. Encoder
# features are already on disk as .npy and are not cached.
CACHE = Path((PROTCOMPASS_ROOT + "/protcompass/cache/scratch/split_feature_cache"))


def cached_features(data, source, level):
    """Descriptor matrices for a reference source, computed once and reused. Encoders go straight to the source."""
    fn = data.residue_features if level == "residue" else data.protein_features
    if not source.startswith("ref_"):
        return fn(source)
    CACHE.mkdir(parents=True, exist_ok=True)
    paths = {p: CACHE / f"{source}_{level}_{p}.npy" for p in R.PARTS}
    if all(p.exists() for p in paths.values()):
        return {p: np.load(paths[p]) for p in R.PARTS}
    out = fn(source)
    for p in R.PARTS:
        np.save(paths[p], out[p])
    return out


def load_masks(data):
    """Boolean masks over the test rows of each probe level, from the protein-level assignment."""
    a = pd.read_csv(ASSIGN)
    half = np.full(len(data.L), "", dtype=object)
    half[a.protein_index.to_numpy()] = a.half.to_numpy()

    res_half = half[data.pidx[data.rows["test"]]]                      # one entry per test residue row
    qp = data.Q["protein_index"][:]
    sel = np.where(np.isin(qp, data.chosen["test"]))[0]                # same selection Data.__init__ makes
    pair_half = half[qp[sel]]
    prot_half = half[data.prot_idx["test"]]
    for name, h in (("residue", res_half), ("pair", pair_half), ("protein", prot_half)):
        assert (h != "").all(), f"{name}: some test rows have no half assigned"
    return {"residue": res_half, "pair": pair_half, "protein": prot_half}


def standardise_cpu(train, *others):
    """R.standardise with the result left on the host.

    Identical arithmetic: R.standardise computes `((a - mu) / sd).astype(np.float16)` in numpy and only then moves the
    fp16 result to the device, so keeping the fp16 array on the host and uploading the rows a probe needs gives the same
    bits. The reason to do it is memory, not numerics: the GPU here has about 3 GB free (another user's AlphaFold job
    holds 25 GB), and the original code holds the whole residue feature matrix on the device and then makes a second,
    label-filtered copy of it per probe. Uploading only the filtered rows removes that duplicate.
    """
    mu, sd = train.mean(0, keepdims=True), train.std(0, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return [((a - mu) / sd).astype(np.float16) for a in (train, *others)]


def masked(kind, y, pred, y_train, half, keep=None):
    """R.score on the full test set and on each half. `keep` is the label-validity filter already applied to y."""
    h = half if keep is None else half[keep]
    assert len(h) == len(y["test"]), f"mask length {len(h)} != test rows {len(y['test'])}"
    out = {}
    for name in EVAL_SETS:
        m = np.ones(len(h), bool) if name == "test" else (h == name)
        out[name] = (int(m.sum()), R.score(kind, y["test"][m], pred[m], y["train"]))
    return out


def run_probe_split(X, y, kind, n_out, half, keep=None, pairs=None):
    """Mirror of R.run_probe: identical calls to R.ridge / R.fit, scored on the full test set and on both halves."""
    out = []
    if pairs is None:
        if kind == "clf":
            pred = R.fit(X, y, kind, n_out, 0, 0)
            out.append(("linear", 0, masked(kind, y, pred, y["train"], half, keep)))
        else:
            pred = R.ridge(X["train"], y["train"], X["val"], y["val"], X["test"])
            out.append(("linear", 0, masked(kind, y, pred, y["train"], half, keep)))
    for seed in R.MLP_SEEDS:
        pred = R.fit(X, y, kind, n_out, R.HIDDEN, seed, pairs)
        out.append(("pair_mlp" if pairs is not None else "mlp2", seed, masked(kind, y, pred, y["train"], half, keep)))
    return out


def run_source(data, masks, source):
    t0, rows = time.time(), []

    def add(probe, iip, level, results):
        for readout, seed, per_set in results:
            for eval_set, (n, sc) in per_set.items():
                rows.append(dict(source=source, probe=probe, iip=iip, level=level, readout=readout, seed=seed,
                                 eval_set=eval_set, n_eval=n, **sc))
        print(f"  [{time.time() - t0:5.0f}s] {source}: {probe}", flush=True)

    pooled_only = not source.startswith("ref_") and not (R.EMB / source / "residue_fp16.npy").exists()
    raw = None if pooled_only else cached_features(data, source, "residue")
    dim = cached_features(data, source, "protein")["train"].shape[1] if pooled_only else raw["train"].shape[1]
    X = {} if pooled_only else dict(zip(R.PARTS, standardise_cpu(raw["train"], raw["val"], raw["test"]))); del raw

    for name, ds, kind, n_out, iip in ([] if pooled_only else R.RESIDUE_PROBES):
        if name == "sequence_recovery" and source in R.SELF_IDENTITY_SOURCES:
            continue
        lab = {p: (np.stack([data.R[d][:][data.rows[p]] for d in ds], 1) if isinstance(ds, tuple)
                   else data.R[ds][:][data.rows[p]]) for p in R.PARTS}
        ok = {p: (~np.isnan(lab[p]).reshape(len(lab[p]), -1).any(1) if kind != "clf" else lab[p] >= 0) for p in R.PARTS}
        Xp = {p: torch.from_numpy(X[p][ok[p]]).to(R.DEV) for p in R.PARTS}
        y = {p: (lab[p][ok[p]].astype(np.int64) if kind == "clf" else lab[p][ok[p]].astype(np.float64)) for p in R.PARTS}
        add(name, iip, "residue", run_probe_split(Xp, y, kind, n_out, masks["residue"], keep=ok["test"]))
        del Xp
        torch.cuda.empty_cache()
    if not pooled_only:
        Xg = {p: torch.from_numpy(X[p]).to(R.DEV) for p in R.PARTS}
        add("distance_bins", "I7", "pair",
            run_probe_split(Xg, data.pair_y, "clf", 16, masks["pair"], pairs=data.pairs))
        del Xg
    del X
    torch.cuda.empty_cache()

    praw = cached_features(data, source, "protein")
    PX = dict(zip(R.PARTS, R.standardise(praw["train"], praw["val"], praw["test"])))
    for k in R.PROTEIN_REG:
        for suffix, label in (("", k), (f"_len_resid_{R.SPLIT}", f"{k}_len_resid")):
            y_all = data.P[k + suffix][:].astype(np.float64)
            add(label, "I8", "protein",
                run_probe_split(PX, {p: y_all[data.prot_idx[p]] for p in R.PARTS}, "reg", 1, masks["protein"]))
    arch = data.P[f"architecture_label_{R.SPLIT}"][:].astype(np.int64)
    ok = {p: arch[data.prot_idx[p]] >= 0 for p in R.PARTS}
    PXa = {p: PX[p][torch.from_numpy(np.where(ok[p])[0]).to(R.DEV)] for p in R.PARTS}
    add("architecture_label", "I9", "protein",
        run_probe_split(PXa, {p: arch[data.prot_idx[p]][ok[p]] for p in R.PARTS}, "clf", int(arch.max()) + 1,
                        masks["protein"], keep=ok["test"]))

    df = pd.DataFrame(rows)
    df.insert(1, "dim", dim)
    df.to_csv(OUTDIR / f"{source}.csv", index=False)
    print(f"[{source}] dim {dim}, {len(df)} rows, {time.time() - t0:.0f}s", flush=True)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    args = sys.argv[1:]
    # --features-only builds the descriptor cache without scoring anything. Run it with CUDA_VISIBLE_DEVICES= so the
    # process never opens a CUDA context; several can then run in parallel without competing for GPU memory.
    features_only = "--features-only" in args
    args = [a for a in args if a != "--features-only"]
    wanted = args or (["ref_length", "ref_position", "ref_aa", "ref_seq_window", "ref_coords", "ref_backbone",
                       "ref_global", "ref_pairhist"]
                      + sorted(d.name for d in R.EMB.iterdir() if (d / "meta.json").exists()))
    if features_only:
        data = R.Data()
        for s in wanted:
            for level in ("residue", "protein"):
                t = time.time()
                x = cached_features(data, s, level)
                print(f"[{s}] {level} features {x['train'].shape[1]}d cached, {time.time() - t:.0f}s", flush=True)
        return
    todo = [s for s in wanted if not (OUTDIR / f"{s}.csv").exists()]
    print(f"device {R.DEV}; to score: {todo}", flush=True)
    data = R.Data()
    masks = load_masks(data)
    for s in todo:
        for attempt in range(6):
            try:
                run_source(data, masks, s)
                break
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"[{s}] CUDA out of memory (attempt {attempt + 1}/6); waiting 5 min for the GPU", flush=True)
                time.sleep(300)
        else:
            print(f"[{s}] skipped: still out of memory after 6 attempts", flush=True)


if __name__ == "__main__":
    main()
