"""Residue-level embeddings of sequence-only protein language models on the rebuilt IIP probing dataset.

Rows of `residue_fp16.npy` are the rows of `probing_data/labels.h5` (`residue/*`), so labels and embeddings are aligned
by construction. The final hidden layer is stored; special tokens are removed.

Run with the `protein_embeddings` env and PYTHONPATH=protcompass/pylib (sentencepiece).
  python extract_plm_probing.py esm2_8m esm2_35m ...        # default: all models
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import json
import sys
import time
from datetime import date
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
LABELS = ROOT / "probing_data" / "labels.h5"
OUT = ROOT / "probing_embeddings"
ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

# name: (hub id, family, token budget per batch)
MODELS = {
    "esm2_8m": ("facebook/esm2_t6_8M_UR50D", "esm", 40000),
    "esm2_35m": ("facebook/esm2_t12_35M_UR50D", "esm", 30000),
    "esm2_150m": ("facebook/esm2_t30_150M_UR50D", "esm", 20000),
    "esm2_650m": ("facebook/esm2_t33_650M_UR50D", "esm", 10000),
    "esm1b": ("facebook/esm1b_t33_650M_UR50S", "esm", 10000),
    "protbert_bfd": ("Rostlab/prot_bert_bfd", "bert", 10000),
    "ankh_base": ("ElnaggarLab/ankh-base", "ankh", 8000),
    "prottrans_t5_xl_u50": ("Rostlab/prot_t5_xl_half_uniref50-enc", "t5", 5000),
    "esm2_3b": ("facebook/esm2_t36_3B_UR50D", "esm", 4000),
}


def load(hub, family):
    from transformers import AutoTokenizer, BertModel, EsmModel, T5EncoderModel, T5Tokenizer
    if family == "esm":
        return AutoTokenizer.from_pretrained(hub), EsmModel.from_pretrained(hub, torch_dtype=torch.float16)
    if family == "bert":
        return AutoTokenizer.from_pretrained(hub, do_lower_case=False), BertModel.from_pretrained(hub, torch_dtype=torch.float16)
    if family == "t5":
        return T5Tokenizer.from_pretrained(hub, do_lower_case=False), T5EncoderModel.from_pretrained(hub, torch_dtype=torch.float16)
    if family == "ankh":
        # Ankh is numerically unstable in fp16; run it in bf16.
        return AutoTokenizer.from_pretrained(hub), T5EncoderModel.from_pretrained(hub, torch_dtype=torch.bfloat16)
    raise ValueError(family)


def encode(tok, model, family, seqs, device):
    """Return a list of (L, D) float32 arrays, one per sequence, special tokens removed."""
    if family == "esm":
        batch, start = tok(seqs, return_tensors="pt", padding=True), 1                     # <cls> x1..xL <eos>
    elif family == "bert":
        batch, start = tok([" ".join(s) for s in seqs], return_tensors="pt", padding=True), 1   # [CLS] x1..xL [SEP]
    elif family == "t5":
        batch, start = tok([" ".join(s) for s in seqs], return_tensors="pt", padding=True), 0   # x1..xL </s>
    else:
        batch, start = tok([list(s) for s in seqs], is_split_into_words=True, return_tensors="pt", padding=True), 0
    batch = {k: v.to(device) for k, v in batch.items() if k in ("input_ids", "attention_mask")}
    with torch.inference_mode():
        h = model(**batch).last_hidden_state.float()
    out = []
    for i, s in enumerate(seqs):
        n_tok = int(batch["attention_mask"][i].sum())
        expected = len(s) + (2 if family in ("esm", "bert") else 1)
        assert n_tok == expected, f"token count {n_tok} != {expected} for a sequence of length {len(s)}"
        out.append(h[i, start:start + len(s)].cpu().numpy())
    return out


def batches(lengths, budget):
    order = np.argsort(lengths)[::-1]
    cur, width = [], 0
    for i in order:
        w = max(width, int(lengths[i]) + 2)
        if cur and w * (len(cur) + 1) > budget:
            yield cur
            cur, w = [], int(lengths[i]) + 2
        cur.append(int(i)); width = w
    if cur:
        yield cur


def run(name, seqs, lengths, offsets, n_res, device):
    hub, family, budget = MODELS[name]
    out_dir = OUT / name
    if (out_dir / "meta.json").exists():
        print(f"[{name}] already done, skipping"); return
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tok, model = load(hub, family)
    model = model.to(device).eval()
    dim = model.config.hidden_size if hasattr(model.config, "hidden_size") else model.config.d_model
    emb = np.lib.format.open_memmap(out_dir / "residue_fp16.npy", mode="w+", dtype=np.float16, shape=(n_res, dim))
    pooled = np.zeros((len(seqs), dim), dtype=np.float32)
    max_abs, done = 0.0, 0
    for idx in batches(lengths, budget):
        for k, h in zip(idx, encode(tok, model, family, [seqs[k] for k in idx], device)):
            assert h.shape[0] == lengths[k]
            max_abs = max(max_abs, float(np.abs(h).max()))
            emb[offsets[k]:offsets[k] + lengths[k]] = h.astype(np.float16)
            pooled[k] = h.mean(0)
        done += len(idx)
        if done % 2000 < len(idx):
            print(f"[{name}] {done}/{len(seqs)} domains, {time.time() - t0:.0f}s", flush=True)
    emb.flush()

    # Health checks: finite values, fp16 range, distinct pooled rows, sensitivity to a shuffled sequence.
    rng = np.random.default_rng(0)
    probe_ids = rng.choice(len(seqs), 16, replace=False)
    shuffled = ["".join(rng.permutation(list(seqs[k]))) for k in probe_ids]
    h_shuf = encode(tok, model, family, shuffled, device)
    rel = [float(np.linalg.norm(h.mean(0) - pooled[k]) / (np.linalg.norm(pooled[k]) + 1e-9)) for k, h in zip(probe_ids, h_shuf)]
    sample = np.asarray(emb[:200000], dtype=np.float32)
    meta = {
        "encoder": name, "hub_id": hub, "revision": getattr(model.config, "_commit_hash", None), "layer": "final hidden state",
        "special_tokens": "removed", "dim": int(dim), "n_domains": len(seqs), "n_residues": int(n_res),
        "dtype_compute": str(next(model.parameters()).dtype), "dtype_stored": "float16", "max_abs_value": max_abs,
        "fp16_overflow": bool(max_abs > 65000), "nonfinite_in_first_200k_rows": int((~np.isfinite(sample)).sum()),
        "distinct_pooled_rows": int(np.unique(np.round(pooled, 4), axis=0).shape[0]),
        "median_rel_change_under_sequence_shuffle": float(np.median(rel)),
        "input": "sequence of the residues kept in labels.h5; discontinuous CATH domains are passed as one concatenated sequence",
        "rows": "identical to residue/* rows of probing_data/labels.h5", "seconds": round(time.time() - t0, 1), "date": str(date.today()),
    }
    np.save(out_dir / "pooled_fp32.npy", pooled)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[{name}] done in {meta['seconds']}s: dim {dim}, max|x| {max_abs:.1f}, distinct pooled rows {meta['distinct_pooled_rows']}, "
          f"shuffle change {meta['median_rel_change_under_sequence_shuffle']:.2f}", flush=True)
    del model
    torch.cuda.empty_cache()


def main():
    names = sys.argv[1:] or list(MODELS)
    with h5py.File(LABELS, "r") as h:
        ids = [x.decode() for x in h["protein/id"][:]]
        lengths, offsets = h["protein/length"][:].astype(int), h["protein/offset"][:].astype(int)
        aa = h["residue/aa"][:]
    assert aa.min() >= 0 and aa.max() < 20
    letters = np.array(list(ALPHABET))
    seqs = ["".join(letters[aa[o:o + n]]) for o, n in zip(offsets, lengths)]
    OUT.mkdir(exist_ok=True)
    (OUT / "ids.txt").write_text("\n".join(ids) + "\n")
    np.save(OUT / "offsets.npy", offsets); np.save(OUT / "lengths.npy", lengths)
    print(f"{len(seqs)} domains, {len(aa)} residues, length {lengths.min()}-{lengths.max()}")
    for name in names:
        run(name, seqs, lengths, offsets, len(aa), torch.device("cuda"))


if __name__ == "__main__":
    main()
