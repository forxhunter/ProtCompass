"""Build the non-learned downstream reference: amino-acid composition and length, no encoder.

The probing side of this work compares every encoder against fixed functions of its own input. The downstream side
did not have the same control, and a reviewer pointed out that it needs one: four tasks were removed from the
benchmark because a trivial feature solved them, and the same test was never applied to the nine that stayed. It is
also directly implied by the interventional result, where amino-acid composition is the property named on five of
the six tasks that name one. If a task's scores are largely a readout of composition, that is a fact about the task
and not about the encoders ranked on it.

The reference is written as if it were an encoder, into `extract/embeddings/<task>/ref_composition/`, with the same
ids and splits as a real encoder on that task. `score2.py` then scores it through the identical protocol: same
standardisation, same linear and MLP-2 probes, same validation-based selection, same three seeds. Nothing in the
scoring path knows it is not an encoder.

Features, 23 per chain: the 20 amino-acid frequencies, log length, length/600, and the fraction of residues not in
the 20-letter alphabet. Protein-level only, which is all the downstream tasks use.

  python make_downstream_reference.py            build for every task
  python make_downstream_reference.py --tasks stability
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
EMB = ROOT / "extract" / "embeddings"
MAN = ROOT / "extract" / "manifest"
NAME = "ref_composition"
AA = "ACDEFGHIKLMNPQRSTVWY"
IDX = {c: i for i, c in enumerate(AA)}


def featurize(seqs):
    X = np.zeros((len(seqs), 23), dtype=np.float32)
    for r, s in enumerate(seqs):
        s = str(s)
        n = max(len(s), 1)
        for ch in s:
            i = IDX.get(ch)
            if i is not None:
                X[r, i] += 1.0
        X[r, :20] /= n
        X[r, 20] = np.log(n)
        X[r, 21] = n / 600.0
        X[r, 22] = 1.0 - X[r, :20].sum()
    return X


def reference_encoder(task):
    """An encoder already extracted on this task, whose id lists the reference copies exactly."""
    cands = sorted(d.name for d in (EMB / task).iterdir() if (d / "meta.json").exists() and d.name != NAME)
    for pref in ("esm2_150m", "esm2_650m", "esm1b"):
        if pref in cands:
            return pref
    return cands[0]


def build(task):
    src = EMB / task / reference_encoder(task)
    out = EMB / task / NAME
    out.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(MAN / f"{task}.csv")
    seq = man.drop_duplicates("id").set_index("id").sequence
    splits = ["chains"] if (src / "chains.npy").exists() else ["train", "valid", "test"]
    n = {}
    for split in splits:
        ids = (src / f"ids_{split}.txt").read_text().split()
        v = seq.reindex(ids)
        if v.isna().any():
            raise SystemExit(f"{task}: {int(v.isna().sum())} of {len(ids)} ids have no sequence in the manifest")
        np.save(out / f"{split}.npy", featurize(v.tolist()))
        (out / f"ids_{split}.txt").write_text("\n".join(ids) + "\n")
        n[split] = len(ids)
    (out / "meta.json").write_text(json.dumps({
        "encoder": NAME, "family": "non-learned reference", "weights": "none", "n_parameters": 0,
        "rule": "20 amino-acid frequencies, log length, length/600, fraction outside the 20-letter alphabet",
        "inputs_passed": "amino-acid sequence only", "hidden_size": 23, "task": task,
        "ids_copied_from": reference_encoder(task), "n_per_split": n,
        "purpose": "downstream floor: what the task scores without any learned representation",
        "date": time.strftime("%Y-%m-%d %H:%M")}, indent=1))
    print(f"{task:26s} {sum(n.values()):6d} chains from {reference_encoder(task)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=sorted(d.name for d in EMB.iterdir() if d.is_dir()))
    for t in ap.parse_args().tasks:
        build(t)


if __name__ == "__main__":
    main()
