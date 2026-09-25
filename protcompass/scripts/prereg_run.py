"""Score encoders and reference floors on the pre-registered labels, with validation and test scores from one fit.

Every readout is fitted once on the training part (ridge alpha and MLP early stopping on the validation part, as in
the released suite) and scored on both the validation and the test part, so the readout family and the reference
floor can be chosen on validation alone. Test predictions of every readout are saved for the protein-level paired
bootstrap. Nothing here selects anything; selection and the decisions are in prereg_decide.py, as frozen in
prereg/freeze.json.

  python prereg_run.py SOURCE [SOURCE ...]          # released split; output prereg/results/
  PREREG_COHORT=B python prereg_run.py SOURCE ...   # cohort B merged file; output prereg/results_cohortB/

This script refuses to run unless prereg/freeze.json exists and its recorded hash matches (the freeze comes first),
except with --synthetic, which replaces every label by a synthetic one for the pre-freeze pipeline check.
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
COHORT = os.environ.get("PREREG_COHORT", "")
if COHORT == "B":
    C = ROOT / "confirmation2"
    os.environ.setdefault("PROBING_LABELS", str(C / "merged" / "labels.h5"))
    os.environ.setdefault("PROBING_EMB", str(C / "merged"))      # prereg_floors sets every cohort domain as test
sys.path.insert(0, str(ROOT / "scripts"))

import h5py                                                              # noqa: E402
import pandas as pd                                                      # noqa: E402
import torch                                                             # noqa: E402

import run_probing_suite as rps                                          # noqa: E402
from prereg_floors import PreregData                                     # noqa: E402

PROBES = [  # name, prereg dataset, kind, n_out, role
    ("hbond_count", "hbond_count", "clf", 3, "primary"),
    ("chi1_class", "chi1_class", "clf", 3, "primary"),
    ("chi2_class", "chi2_class", "clf", 3, "primary"),
    ("ca_dihedral", "ca_dihedral", "circ", 2, "control"),
    ("sc_contacts", "sc_contacts", "reg", 1, "descriptive"),
]
OUT = ROOT / "prereg" / ("results_cohortB" if COHORT == "B" else "results")
FREEZE = ROOT / "prereg" / "freeze.json"
SYNTHETIC = "--synthetic" in sys.argv


def prereg_rows():
    """Prereg label arrays aligned with the rows of rps.LABELS."""
    if COHORT == "B":
        return h5py.File(ROOT / "prereg" / "labels_cohortB_merged.h5", "r")["residue"]
    return h5py.File(ROOT / "prereg" / "labels_released.h5", "r")["residue"]


def check_freeze():
    if SYNTHETIC:
        return
    if not FREEZE.exists():
        raise SystemExit("prereg/freeze.json does not exist: freeze before scoring")
    rec = json.loads((ROOT / "prereg" / "freeze.sha256.json").read_text())
    h = hashlib.sha256(FREEZE.read_bytes()).hexdigest()
    if h != rec["sha256"]:
        raise SystemExit(f"freeze.json changed after it was frozen ({h} != {rec['sha256']})")


def fit_both(X, y, kind, n_out):
    """(readout, seed, split, score) for the linear readout and the three MLP seeds, val and test from one fit."""
    nv = len(y["val"])
    Xc = {"train": X["train"], "val": X["val"], "test": torch.cat([X["val"], X["test"]])}
    yc = {"train": y["train"], "val": y["val"], "test": np.concatenate([y["val"], y["test"]])}
    out, preds = [], {}
    if kind == "clf":
        pl = rps.fit(Xc, yc, kind, n_out, 0, 0)
    else:
        pl = rps.ridge(X["train"], y["train"], X["val"], y["val"], Xc["test"])
    runs = [("linear", 0, pl)] + [("mlp2", s, rps.fit(Xc, yc, kind, n_out, rps.HIDDEN, s)) for s in rps.MLP_SEEDS]
    for readout, seed, p in runs:
        for split, part, yy in (("val", p[:nv], y["val"]), ("test", p[nv:], y["test"])):
            out.append((readout, seed, split, rps.score(kind, yy, part, y["train"])))
        preds[f"{readout}_{seed}"] = p[nv:]
    return out, preds


def run_source(data, lab, source):
    t0, rows, saved = time.time(), [], {}
    raw = data.residue_features(source)
    X = dict(zip(rps.PARTS, rps.standardise(raw["train"], raw["val"], raw["test"]))); del raw
    rng = np.random.default_rng(0)
    for name, ds, kind, n_out, role in PROBES:
        y_all = {p: lab[ds][:][data.rows[p]] for p in rps.PARTS}
        if SYNTHETIC:          # a label that is a fixed function of the first input column: checks the plumbing only
            base = {p: X[p][:, 0].float().cpu().numpy() for p in rps.PARTS}
            y_all = {p: (np.digitize(base[p], [-0.5, 0.5]).astype(np.int8) if kind == "clf" else
                         np.stack([np.sin(base[p]), np.cos(base[p])], 1) if kind == "circ" else base[p])
                     for p in rps.PARTS}
        ok = {p: (y_all[p] >= 0) if kind == "clf" else ~np.isnan(np.asarray(y_all[p], float).reshape(len(y_all[p]), -1)).any(1)
              for p in rps.PARTS}
        Xp = {p: X[p][torch.from_numpy(np.where(ok[p])[0]).to(rps.DEV)] for p in rps.PARTS}
        y = {p: (y_all[p][ok[p]].astype(np.int64) if kind == "clf" else y_all[p][ok[p]].astype(np.float64)) for p in rps.PARTS}
        res, preds = fit_both(Xp, y, kind, n_out)
        for readout, seed, split, sc in res:
            rows.append(dict(source=source, probe=name, role=role, readout=readout, seed=seed, split=split,
                             n=len(y[split]), **sc))
        prot = data.pidx[data.rows["test"]][ok["test"]]
        saved[name] = dict(y=y["test"], protein=prot, **preds)
        print(f"  [{time.time() - t0:5.0f}s] {source}: {name}", flush=True)
        del Xp
    del X; torch.cuda.empty_cache()
    sub = OUT / ("synthetic" if SYNTHETIC else "by_source")
    sub.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(sub / f"{source}.csv", index=False)
    np.savez_compressed(sub / f"{source}__test_predictions.npz",
                        **{f"{k}|{kk}": v for k, d in saved.items() for kk, v in d.items()})
    print(f"[{source}] {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)


def main():
    check_freeze()
    sources = [a for a in sys.argv[1:] if not a.startswith("--")]
    sub = OUT / ("synthetic" if SYNTHETIC else "by_source")
    todo = [s for s in sources if not (sub / f"{s}.csv").exists()]
    print(f"device {rps.DEV}; labels {rps.LABELS}; to score: {todo}", flush=True)
    data, lab = PreregData(), prereg_rows()
    for s in todo:
        for attempt in range(12):
            try:
                run_source(data, lab, s)
                break
            except (torch.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" not in str(e).lower():
                    raise
                torch.cuda.empty_cache()
                print(f"[{s}] CUDA out of memory (attempt {attempt + 1}/12); waiting 5 min", flush=True)
                time.sleep(300)


if __name__ == "__main__":
    main()
