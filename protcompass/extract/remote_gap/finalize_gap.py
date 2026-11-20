"""Check the arrays of gap_worker.py against a local computation and write them in the layout of extract.py.

Run on the first machine in the encoder's own environment, on the CPU:

  python remote_gap/finalize_gap.py --encoder saprot --set masif
  python remote_gap/finalize_gap.py --encoder foldvision --set remote_homology
  python remote_gap/finalize_gap.py --encoder proteinmpnn --set ds_stability --replace
  python remote_gap/finalize_gap.py --encoder proteinmpnn --set probing --replace

A fixed sample of the proteins is recomputed here with extract/common.py:Encoder on the CPU. The run stops if the
largest relative difference to the remote embedding exceeds --tol. The metadata is the one extract.py writes, with the
input audit taken from the audit table (same wrapper, same dataset family) and three more fields: `computed_on`,
`seconds` (remote) and `remote_check`.

With --replace an existing output folder (embeddings of an earlier wrapper) is first moved to the sibling root
`<root>_old_wrappers/`, so nothing is deleted.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXT = HERE.parent
RES = EXT.parent
sys.path.insert(0, str(EXT))
from common import Encoder  # noqa: E402
from extract import health  # noqa: E402

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AUDIT_KEYS = ("rel_change_jitter_1A", "rel_change_coord_swap", "rel_change_seq_shuffle", "rel_change_rigid_motion", "verdict")


def audit_row(key, dataset):
    a = pd.read_csv(EXT / "encoder_input_audit.csv")
    r = a[(a.encoder == key) & (a.dataset == dataset)]
    if len(r) != 1 or r.iloc[0].verdict != "pass":
        raise SystemExit(f"[{key}] no passing {dataset} row in encoder_input_audit.csv")
    return {k: (r.iloc[0][k] if k == "verdict" else float(r.iloc[0][k])) for k in AUDIT_KEYS}


def retire(out, root):
    """Move an existing output folder to <root>_old_wrappers/<same relative path>."""
    dst = root.parent / (root.name + "_old_wrappers") / out.relative_to(root)
    if dst.exists():
        raise SystemExit(f"{dst} exists; nothing was moved")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(out), str(dst))
    print(f"  moved the earlier folder to {dst}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--set", required=True, help="masif, probing, remote_homology or ds_<task>")
    ap.add_argument("--n-check", type=int, default=12)
    ap.add_argument("--tol", type=float, default=2e-3)
    ap.add_argument("--replace", action="store_true")
    ap.add_argument("--device", default="cpu", help="device of the local recomputation")
    a = ap.parse_args()
    src = HERE / "out" / a.set / a.encoder
    done = json.loads((src / "done.json").read_text())
    rng = np.random.default_rng(0)
    enc = Encoder(a.encoder, device=a.device)
    diffs = []
    residue_set = a.set in ("masif", "probing", "cohortA", "cohortB")
    if residue_set:
        import h5py
        if a.set == "masif":
            base, root, labels = RES / "masif_site", RES / "masif_site" / "embeddings", RES / "masif_site" / "labels.h5"
            dataset_name, struct = str(labels), RES / "masif_site" / "structures"
        elif a.set in ("cohortA", "cohortB"):
            base = RES / ("confirmation" if a.set == "cohortA" else "confirmation2")
            root, labels = base / "embeddings", base / "labels_v44.h5"
            dataset_name, struct = str(labels), RES / "probing_data" / "structures"
        else:
            base, root, labels = RES, RES / "probing_embeddings", RES / "probing_data" / "labels.h5"
            dataset_name, struct = "resubmission/probing_data/labels.h5", RES / "probing_data" / "structures"
        out = root / a.encoder
        with h5py.File(labels, "r") as h:
            ids = [x.decode() for x in h["protein/id"][:]]
            lengths, offsets = h["protein/length"][:].astype(np.int64), h["protein/offset"][:].astype(np.int64)
            coords, aa = h["residue/coords"][:], h["residue/aa"][:]
        res = np.load(src / "residue_fp16.npy", mmap_mode="r")
        pooled = np.load(src / "pooled_fp32.npy")
        assert res.shape[0] == int(lengths.sum()) and pooled.shape[0] == len(ids)
        for name in ("offsets.npy", "lengths.npy"):
            assert np.array_equal(np.load(root / name), offsets if name == "offsets.npy" else lengths)
        short = np.where(lengths <= 250)[0]                       # the CPU check uses chains of at most 250 residues
        for i in rng.choice(short, a.n_check, replace=False):
            o, L = int(offsets[i]), int(lengths[i])
            seq = "".join(ALPHABET[k] for k in aa[o:o + L])
            h = enc.residue(seq, coords=coords[o:o + L].astype(np.float32), pdb_path=str(struct / f"{ids[i]}.pdb"))
            r = np.asarray(res[o:o + L], dtype=np.float32)
            diffs.append((float(np.abs(h - r).max()), float(np.linalg.norm(h - r) / np.linalg.norm(h))))
        dim = int(res.shape[-1])
        del res
    else:
        holdout = a.set.startswith("ho_")
        task = a.set[3:] if a.set.startswith(("ds_", "ho_")) else a.set
        root = RES / "confirmation_downstream" / "embeddings" if holdout else EXT / "embeddings"
        man = pd.read_csv(EXT / "manifest" / f"{task}.csv")
        ids_file = RES / "confirmation_downstream" / f"ids_{task}.txt"
        if holdout:
            keep = {l.strip() for l in open(ids_file) if l.strip()}
            man = man[man.id.isin(keep)].reset_index(drop=True)
        else:
            man = man[man.stage1 == 1].reset_index(drop=True)
        out = root / task / a.encoder
        X = np.load(src / "pooled.npy")
        bundle = HERE / ("bundle3" if a.set.startswith(("ds_", "ho_")) else "bundle") / "data" / f"{a.set}.npz"
        z = np.load(bundle)
        assert list(z["ids"]) == list(man.id) and X.shape[0] == len(man)
        for k in rng.choice(len(man), a.n_check, replace=False):
            h = enc.residue(man.sequence[k], coords=z["xyz"][z["offsets"][k]:z["offsets"][k + 1]], pdb_path=man.structure_path[k]).mean(0)
            diffs.append((float(np.abs(h - X[k]).max()), float(np.linalg.norm(h - X[k]) / np.linalg.norm(h))))
    worst_abs, worst_rel = max(d[0] for d in diffs), max(d[1] for d in diffs)
    print(f"[{a.encoder}/{a.set}] {len(diffs)} proteins recomputed here: largest absolute difference {worst_abs:.2e}, "
          f"largest relative difference {worst_rel:.2e}", flush=True)
    if worst_rel > a.tol:
        raise SystemExit(f"relative difference above {a.tol}; nothing was written")
    local = enc.finalize_meta()
    for k in ("weights_sha256", "n_parameters", "checkpoint_sha256"):
        if k in local and k in done and local[k] != done[k]:
            raise SystemExit(f"{k} differs between the machines: {local[k]} here, {done[k]} there")
    if out.exists():
        if not a.replace:
            raise SystemExit(f"{out} exists; nothing was written (use --replace to retire it)")
        retire(out, root)
    extra = dict(computed_on=f"second GPU machine ({done['gpu']}, PyTorch {done['torch']}) with extract/remote_gap/gap_worker.py; "
                             "same wrapper code and checkpoint",
                 remote_check=dict(n_proteins=len(diffs), max_abs_difference=worst_abs, max_relative_difference=worst_rel,
                                   recomputed="on the first machine, " + ("CPU" if a.device == "cpu" else "GPU")))
    out.mkdir(parents=True, exist_ok=True)
    if residue_set:
        shutil.copyfile(src / "residue_fp16.npy", out / "residue_fp16.npy")
        np.save(out / "pooled_fp32.npy", pooled)
        meta = dict(local, dataset=dataset_name, date=time.strftime("%Y-%m-%d %H:%M"), seconds=done["seconds"],
                    n_domains=len(ids), dim=dim,
                    row_convention="row r of residue_fp16.npy is row r of residue/* in labels.h5",
                    health=health(pooled), input_audit=audit_row(a.encoder, "probing"), **extra)
    else:
        hlt = health(X)
        if hlt["nan_values"] or hlt["all_zero_rows"]:
            raise SystemExit(f"embedding health check failed: {hlt}")
        for split, g in man.groupby("split"):
            np.save(out / f"{split}.npy", X[g.index.to_numpy()])
            (out / f"ids_{split}.txt").write_text("\n".join(g.id) + "\n")
        meta = dict(local, task=task, date=time.strftime("%Y-%m-%d %H:%M"), seconds=done["seconds"],
                    pooling="masked mean over all residues of the chain", manifest=f"manifest/{task}.csv",
                    rows=("held-out cohort: " + str(ids_file)) if holdout else "stage1 == 1",
                    structure_coverage="every row has a structure file whose sequence equals the task sequence (see manifest/coverage.csv)",
                    n_per_split=man.groupby("split").size().to_dict(), health=hlt,
                    input_audit=audit_row(a.encoder, "stage1"), **extra)
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(f"[{a.encoder}/{a.set}] written to {out}", flush=True)


if __name__ == "__main__":
    main()
