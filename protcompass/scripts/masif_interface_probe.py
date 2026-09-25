"""Score the residue-level interface-site task (masif_site/labels.h5) with the released probing readouts.

The readouts, the standardisation, the protein sampling and every reference descriptor are imported from
scripts/run_probing_suite.py with its three paths pointed at the MaSIF-site files, so an encoder and a floor on this
task are scored by the same code as on the probing suite. The one change is the output of the classifier: the
released `fit` returns the arg-max class, and this task is scored by AUROC and AUPRC, so the class-1 probability is
returned instead. The change is made on the released source text and asserted, so the training loop is the released one.

Readouts: linear (one linear layer, softmax) and a two-layer MLP with 128 hidden units, seeds 0-2; AdamW lr 1e-3,
weight decay 1e-4, early stopping on validation loss. Train, validation and test are the chains of the MaSIF-site
training list (90% / 10%) and its testing list.

  python masif_interface_probe.py                 # every reference and every extracted encoder not yet scored
  python masif_interface_probe.py esm2_650m ref_global
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import inspect
import os
import sys
import time
from pathlib import Path

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
MS = ROOT / "masif_site"
os.environ["PROBING_LABELS"] = str(MS / "labels.h5")
os.environ["PROBING_EMB"] = str(MS / "embeddings")
os.environ["PROBING_OUT"] = str(MS / "results")
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

import run_probing_suite as rps

_src = inspect.getsource(rps.fit)
_old = 'return out.argmax(-1).cpu().numpy() if kind == "clf" else'
assert _old in _src, "released fit() changed; the probability patch no longer applies"
exec(_src.replace("def fit(", "def fit_proba(").replace(_old, 'return out.softmax(-1)[:, 1].cpu().numpy() if kind == "clf" else'),
     rps.__dict__)
fit_proba = rps.fit_proba

REFS = ["ref_length", "ref_position", "ref_aa", "ref_seq_window", "ref_seq_expert", "ref_seq_global",
        "ref_coords", "ref_backbone", "ref_global", "ref_pairhist"]
OUT = MS / "results" / "by_source"


def metrics(y, p):
    return dict(auroc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)),
                balanced_accuracy=float(balanced_accuracy_score(y, (p >= 0.5).astype(int))), base_rate=float(y.mean()))


def run(data, source):
    t0 = time.time()
    raw = data.residue_features(source)
    X = dict(zip(rps.PARTS, rps.standardise(raw["train"], raw["val"], raw["test"])))
    dim = raw["train"].shape[1]
    del raw
    lab = data.R["interface"][:]
    y = {p: lab[data.rows[p]].astype(np.int64) for p in rps.PARTS}
    rows = [dict(source=source, dim=dim, readout="linear", seed=0, **metrics(y["test"], fit_proba(X, y, "clf", 2, 0, 0)))]
    for seed in rps.MLP_SEEDS:
        rows.append(dict(source=source, dim=dim, readout="mlp2", seed=seed,
                         **metrics(y["test"], fit_proba(X, y, "clf", 2, rps.HIDDEN, seed))))
    del X; torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"{source}.csv", index=False)
    print(f"[{source}] dim {dim}  linear AUPRC {rows[0]['auprc']:.3f}  mlp2 AUPRC {df[df.readout == 'mlp2'].auprc.mean():.3f}  "
          f"{time.time() - t0:.0f}s", flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    emb = MS / "embeddings"
    encs = sorted(d.name for d in emb.iterdir() if (d / "meta.json").exists()) if emb.exists() else []
    wanted = sys.argv[1:] or REFS + encs
    todo = [s for s in wanted if not (OUT / f"{s}.csv").exists()]
    print(f"device {rps.DEV}; to score: {todo}", flush=True)
    data = rps.Data()
    for s in todo:
        for attempt in range(6):
            try:
                run(data, s)
                break
            except (torch.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" not in str(e).lower():
                    raise
                torch.cuda.empty_cache()
                print(f"[{s}] CUDA out of memory (attempt {attempt + 1}/6); waiting 5 min", flush=True)
                time.sleep(300)
    frames = [pd.read_csv(f) for f in sorted(OUT.glob("*.csv"))]
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(MS / "results" / "scores_long.csv", index=False)


if __name__ == "__main__":
    main()
