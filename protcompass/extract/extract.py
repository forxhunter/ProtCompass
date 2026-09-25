"""Extract embeddings for one encoder on one dataset, with the input audit as a gate.

  python extract.py --encoder esm2_650m --task stability            # protein-level, stage-1 rows of the manifest
  python extract.py --encoder schnet --task probing                 # residue-level on the rebuilt IIP probing dataset
  python extract.py --encoder esm2_35m --task probing --layers all  # every layer on the fixed probing subset (PLMs only)

Downstream output: embeddings/<task>/<encoder>/{train,valid,test}.npy (or chains.npy), ids_<split>.txt, meta.json
Probing output:    ../probing_embeddings/<encoder>/residue_fp16.npy (rows = residue rows of probing_data/labels.h5),
                   pooled_fp32.npy, meta.json; the shared ids.txt, offsets.npy, lengths.npy live at the root.

The script never substitutes inputs. A protein that cannot be encoded stops the run; nothing is written for that encoder.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import ALPHABET, HERE, ROOT, SPEC, Encoder, load_backbone
from input_audit import AUDIT_CSV, run_audit

PROBE_OUT = ROOT / "." / "probing_embeddings"
CACHE = HERE / "cache"


def coords_cache(task, man):
    """Backbone coordinates of all manifest rows, parsed once per task and shared by every encoder.

    Every file is parsed with the expected sequence, so a structure that does not match its manifest row raises here."""
    import fcntl
    from concurrent.futures import ThreadPoolExecutor
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"{task}_{hashlib.sha256(''.join(man.id).encode()).hexdigest()[:10]}.npz"
    with open(str(f) + ".lock", "w") as lock:                       # one builder; the other queues wait and then read
        fcntl.flock(lock, fcntl.LOCK_EX)
        if f.exists():
            z = np.load(f)
            return z["xyz"], z["offsets"]
        with ThreadPoolExecutor(16) as pool:                        # file reads dominate on the shared disk
            parsed = list(pool.map(lambda r: load_backbone(r[0], r[1])[0], zip(man.structure_path, man.sequence), chunksize=32))
        offsets = np.concatenate([[0], np.cumsum([len(c) for c in parsed])])
        xyz = np.concatenate(parsed)
        np.savez(f, xyz=xyz, offsets=offsets)
    return xyz, offsets


def health(X):
    """Distinct rows, numerical rank, NaN and all-zero rows of a pooled embedding matrix."""
    n = len(X)
    sub = X[np.random.default_rng(0).choice(n, min(n, 3000), replace=False)].astype(np.float64)
    s = np.linalg.svd(sub - sub.mean(0), compute_uv=False)
    return {"n_rows": int(n), "dim": int(X.shape[1]), "nan_values": int(np.isnan(X).sum()), "all_zero_rows": int((np.abs(X).sum(1) == 0).sum()),
            "distinct_rows": int(len(np.unique(np.round(X, 5), axis=0))), "numerical_rank": int((s > s[0] * 1e-6).sum()) if s[0] > 0 else 0,
            "max_abs": float(np.abs(X).max())}


def gate(key, dataset, enc):
    row, enc = run_audit(key, dataset, enc=enc)
    if row["verdict"] != "pass":
        raise SystemExit(f"[{key}] input audit failed on {dataset}: {row['problems']}. Nothing was extracted.")
    return row, enc


def pooled_many(enc, seqs, xyz, offsets, paths):
    """Masked mean of residue embeddings for every protein. PLMs are batched by token budget; the rest run one protein per call."""
    n, out = len(seqs), None
    if enc.spec.family == "hf_plm":
        from patched.hf_plm import token_budget_batches
        for idx in token_budget_batches([len(s) for s in seqs]):
            for k, h in zip(idx, enc.impl.residue_batch([seqs[i] for i in idx])):
                if out is None:
                    out = np.zeros((n, h.shape[1]), dtype=np.float32)
                assert h.shape[0] == len(seqs[k]) and np.isfinite(h).all()
                out[k] = h.mean(0)
        return out
    for k in range(n):
        h = enc.residue(seqs[k], coords=xyz[offsets[k]:offsets[k + 1]], pdb_path=paths[k])
        if out is None:
            out = np.zeros((n, h.shape[1]), dtype=np.float32)
        out[k] = h.mean(0)
        if (k + 1) % 2000 == 0:
            print(f"  [{enc.spec.key}] {k + 1}/{n}", flush=True)
    return out


def run_downstream(key, task, full, limit, ids_file=None, out_root=None):
    """`ids_file` and `out_root` are for a held-out cohort: restrict the manifest to those ids and write elsewhere.
    With neither, this is the released stage-1 extraction, unchanged."""
    root = Path(out_root) if out_root else HERE / "embeddings"
    out_dir = root / task / key
    if (out_dir / "meta.json").exists():
        print(f"[{key}/{task}] already extracted"); return
    man = pd.read_csv(HERE / "manifest" / f"{task}.csv")
    if ids_file:
        keep = [l.strip() for l in open(ids_file) if l.strip()]
        man = man[man.id.isin(set(keep))]
        if len(man) != len(set(keep)):
            raise SystemExit(f"[{key}/{task}] {len(man)} manifest rows for {len(set(keep))} requested ids")
    else:
        man = man if full else man[man.stage1 == 1]
    man = man.reset_index(drop=True)
    if limit:
        man = man.groupby("split").head(limit).reset_index(drop=True)
    xyz, offsets = coords_cache(task + ("_full" if full else "") + (f"_lim{limit}" if limit else "")
                                + ("_holdout" if ids_file else ""), man)
    t0 = time.time()
    enc = Encoder(key)
    audit, enc = gate(key, "stage1", enc)
    X = pooled_many(enc, list(man.sequence), xyz, offsets, list(man.structure_path))
    h = health(X)
    if h["nan_values"] or h["all_zero_rows"] or h["distinct_rows"] < 0.5 * len(np.unique(man.sequence)) and enc.spec.uses_sequence:
        raise SystemExit(f"[{key}/{task}] embedding health check failed: {h}. Nothing was written.")
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, g in man.groupby("split"):
        np.save(out_dir / f"{split}.npy", X[g.index.to_numpy()])
        (out_dir / f"ids_{split}.txt").write_text("\n".join(g.id) + "\n")
    meta = dict(enc.finalize_meta(), task=task, date=time.strftime("%Y-%m-%d %H:%M"), seconds=round(time.time() - t0, 1),
                pooling="masked mean over all residues of the chain", manifest=f"manifest/{task}.csv",
                rows=("held-out cohort: " + str(ids_file)) if ids_file else ("all rows" if full else "stage1 == 1"),
                structure_coverage="every row has a structure file whose sequence equals the task sequence (see manifest/coverage.csv)",
                n_per_split=man.groupby("split").size().to_dict(), health=h,
                input_audit={k: audit[k] for k in ("rel_change_jitter_1A", "rel_change_coord_swap", "rel_change_seq_shuffle", "rel_change_rigid_motion", "verdict")})
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(f"[{key}/{task}] {len(man)} proteins, dim {h['dim']}, rank {h['numerical_rank']}, distinct {h['distinct_rows']}, {meta['seconds']}s", flush=True)


def probing_subset(data, n_train=2000, n_test=1000, seed=0):
    rng = np.random.default_rng(seed)
    tr, te = data.indices("train"), data.indices("test")
    return np.sort(np.concatenate([rng.choice(tr, n_train, replace=False), rng.choice(te, n_test, replace=False)]))


def run_probing(key, layers, limit, labels=None, out_root=None):
    """Extract on the probing dataset. `labels` and `out_root` default to the released pair, so passing neither
    reproduces the released run exactly; the confirmation cohort passes its own labels file and output root."""
    sys.path.insert(0, str(ROOT / "." / "probing_data"))
    from probing_dataset import ProbingData
    data = ProbingData(Path(labels)) if labels else ProbingData()
    PROBE_OUT = Path(out_root) if out_root else globals()["PROBE_OUT"]
    dataset_name = labels or "protcompass/probing_data/labels.h5"
    lengths, offsets = data.length.astype(np.int64), data.offset.astype(np.int64)
    n_res = int(lengths.sum())
    PROBE_OUT.mkdir(exist_ok=True)
    for name, arr in (("offsets.npy", offsets), ("lengths.npy", lengths)):
        if (PROBE_OUT / name).exists():
            assert np.array_equal(np.load(PROBE_OUT / name), arr), f"{name} at the output root does not match labels.h5"
        else:
            np.save(PROBE_OUT / name, arr)
    if not (PROBE_OUT / "ids.txt").exists():
        (PROBE_OUT / "ids.txt").write_text("\n".join(data.ids) + "\n")
    out_dir = PROBE_OUT / (f"{key}_layers" if layers == "all" else key)   # a last-layer directory with meta.json is never touched
    tag = "layers_meta.json" if layers == "all" else "meta.json"
    if (out_dir / tag).exists():
        print(f"[{key}/probing] {tag} exists; directory left untouched"); return
    t0 = time.time()
    enc = Encoder(key)
    # Encoders without residue states (a global embedding or a token-frequency vector) still carry the protein-level
    # properties I8 and I9, so they are extracted pooled-only and the probe runner skips their residue and pair probes.
    pooled_only = getattr(enc.impl, "residue_level", True) is False
    if pooled_only and layers == "all":
        raise SystemExit(f"[{key}] has no residue states; --layers all does not apply")
    if layers == "all":
        if enc.spec.family != "hf_plm":
            raise SystemExit("--layers all is implemented for the Hugging Face language models only")
        from patched.hf_plm import HFPLM
        enc.impl = HFPLM(enc.spec.target, enc.device, all_layers=True)
    else:
        audit, enc = gate(key, "probing", enc)
    idx = probing_subset(data) if layers == "all" else np.arange(len(data))
    if limit:
        idx = idx[:limit]
    aa = data.h["residue/aa"][:]
    seqs = {int(i): "".join(ALPHABET[a] for a in aa[offsets[i]:offsets[i] + lengths[i]]) for i in idx}
    out_dir.mkdir(parents=True, exist_ok=True)
    emb = pooled = None

    def store(i, h):
        nonlocal emb, pooled
        L = int(lengths[i])
        assert np.isfinite(h).all() and np.abs(h).max() < 6.5e4, f"{data.ids[i]}: non-finite or float16 overflow"
        if pooled_only:   # the wrapper tiles one global vector over the L residues; keep a single row
            if pooled is None:
                pooled = np.zeros((len(data), h.shape[-1]), dtype=np.float32)
            pooled[i] = h[0]
            return
        assert h.shape[-2] == L, f"{data.ids[i]}: {h.shape[-2]} residue rows for {L} labelled residues"
        if emb is None:
            if layers == "all":
                sub_off = np.concatenate([[0], np.cumsum(lengths[idx])])
                np.save(out_dir / "layers_subset_index.npy", idx); np.save(out_dir / "layers_subset_offsets.npy", sub_off)
                emb = np.lib.format.open_memmap(out_dir / "layers_fp16.npy", mode="w+", dtype=np.float16, shape=(h.shape[0], int(sub_off[-1]), h.shape[-1]))
                store.sub = dict(zip(idx.tolist(), sub_off[:-1].tolist()))
            else:
                emb = np.lib.format.open_memmap(out_dir / "residue_fp16.npy.partial", mode="w+", dtype=np.float16, shape=(n_res, h.shape[-1]))
                pooled = np.zeros((len(data), h.shape[-1]), dtype=np.float32)
        if layers == "all":
            o = store.sub[int(i)]; emb[:, o:o + L] = h.astype(np.float16)
        else:
            emb[offsets[i]:offsets[i] + L] = h.astype(np.float16); pooled[i] = h.mean(0)

    if enc.spec.family == "hf_plm":
        from patched.hf_plm import token_budget_batches
        order = [int(i) for i in idx]
        for b in token_budget_batches([lengths[i] for i in order], budget=6000 if layers == "all" else 24000):
            for k, h in zip(b, enc.impl.residue_batch([seqs[order[j]] for j in b])):
                store(order[k], h)
    else:
        for n, i in enumerate(idx):
            p = data.protein(int(i))
            store(int(i), enc.residue(seqs[int(i)], coords=p["coords"].astype(np.float32), pdb_path=p["pdb_path"]))
            if (n + 1) % 2000 == 0:
                print(f"  [{key}] {n + 1}/{len(idx)} domains, {time.time() - t0:.0f}s", flush=True)
    if pooled_only:
        np.save(out_dir / "pooled_fp32.npy", pooled)
        meta = dict(enc.finalize_meta(), dataset=dataset_name, date=time.strftime("%Y-%m-%d %H:%M"),
                    seconds=round(time.time() - t0, 1), n_domains=int(len(idx)), dim=int(pooled.shape[1]), pooled_only=True,
                    row_convention="row r of pooled_fp32.npy is row r of protein/* in labels.h5; no residue states, so only the protein-level probes apply",
                    health=health(pooled), input_audit={k: audit[k] for k in ("rel_change_jitter_1A", "rel_change_coord_swap", "rel_change_seq_shuffle", "rel_change_rigid_motion", "verdict")})
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
        print(f"[{key}/probing] done, pooled only: {meta['n_domains']} domains, dim {meta['dim']}, {meta['seconds']}s", flush=True)
        return
    emb.flush()
    meta = dict(enc.finalize_meta(), dataset=dataset_name, date=time.strftime("%Y-%m-%d %H:%M"), seconds=round(time.time() - t0, 1),
                n_domains=int(len(idx)), dim=int(emb.shape[-1]), row_convention="row r of residue_fp16.npy is row r of residue/* in labels.h5")
    if layers == "all":
        meta.update(layers=int(emb.shape[0]), layer_0="embedding output", subset="2,000 train + 1,000 test domains of the superfamily split, seed 0")
        (out_dir / "layers_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    else:
        if limit:
            raise SystemExit("--limit run finished; partial file kept for inspection, no meta.json written")
        (out_dir / "residue_fp16.npy.partial").rename(out_dir / "residue_fp16.npy")
        np.save(out_dir / "pooled_fp32.npy", pooled)
        meta.update(health=health(pooled), input_audit={k: audit[k] for k in ("rel_change_jitter_1A", "rel_change_coord_swap", "rel_change_seq_shuffle", "rel_change_rigid_motion", "verdict")})
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(f"[{key}/probing] done: {meta['n_domains']} domains, dim {meta['dim']}, {meta['seconds']}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=list(SPEC))
    ap.add_argument("--task", required=True, choices=sorted([f.stem for f in (HERE / "manifest").glob("*.csv") if f.stem != "coverage"] + ["probing"]))
    ap.add_argument("--full", action="store_true", help="all manifest rows instead of the stage-1 subsample")
    ap.add_argument("--layers", default="last", choices=["last", "all"])
    ap.add_argument("--limit", type=int, default=0, help="debugging only")
    ap.add_argument("--probing-labels", default=None,
                    help="an alternative labels.h5 for --task probing; the default is the released one")
    ap.add_argument("--ids-file", default=None,
                    help="restrict a downstream task to these ids, one per line: the held-out cohort")
    ap.add_argument("--out-root", default=None, help="alternative embeddings root for a held-out extraction")
    ap.add_argument("--probing-out", default=None,
                    help="an alternative output root for --task probing; the default is protcompass/probing_embeddings")
    a = ap.parse_args()
    if a.task == "probing":
        run_probing(a.encoder, a.layers, a.limit, a.probing_labels, a.probing_out)
    else:
        run_downstream(a.encoder, a.task, a.full, a.limit, a.ids_file, a.out_root)
