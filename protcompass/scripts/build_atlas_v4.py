"""Latent-space atlas, one layer per scored downstream task.

Every layer holds a stratified sample of the proteins the task was scored on (train and test splits of the released
extraction, the same rows the downstream heads were trained and tested on), labelled by the task's own target:
the class for classification tasks, the SCOP class for remote homology, a GO term for the two GO tasks, and target
quartiles for the three regression tasks. PPI affinity is a property of a chain pair, so its layer holds the chains
and labels each chain by the quartile of the mean affinity of the pairs it takes part in. The layer builder, metrics
and Procrustes alignment are those of atlas/scripts/build_atlas.py, unchanged.

  python build_atlas_v4.py --tasks ec go_mf ...      # writes atlas/data_v4/<layer>.json per task
  python build_atlas_v4.py --merge                   # assembles atlas/data_v4/atlas.json from the per-layer files
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
import json
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
RES = ROOT / "."
EMB = RES / "extract" / "embeddings"
DATA = ROOT / "structure_encoder_data"
OUT = RES / "atlas" / "data_v4"
sys.path.insert(0, str(RES / "atlas" / "scripts"))
sys.path.insert(0, str(RES / "scripts"))
import build_atlas as ba                                     # noqa: E402
from make_panels_original_style import DECLARED, DISP        # noqa: E402

REFERENCE = "esm2_150m"
N_LAYER, FLOOR = 2000, 150
SEED = ba.SEED
MODALITY = {"seq": "sequence", "struct": "structure", "multi": "multimodal"}
QUARTERS = ["lowest quarter", "second quarter", "third quarter", "highest quarter"]
LOC = ["Cytoplasm", "Nucleus", "Extracellular", "Cell membrane", "Mitochondrion", "Plastid",
       "Endoplasmic reticulum", "Lysosome/Vacuole", "Golgi apparatus", "Peroxisome"]
SCOP = {0: "a: all alpha", 1: "b: all beta", 2: "c: alpha/beta", 3: "d: alpha+beta", 4: "e: multi-domain",
        5: "f: membrane and cell surface", 6: "g: small proteins"}
# eight frequent, non-nested terms per GO branch; a protein takes the least frequent one it carries, else "other"
GO_TERMS = {
    "mf": ["GO:0005524", "GO:0140096", "GO:0046914", "GO:0003723", "GO:0003677", "GO:0005198", "GO:0005215", "GO:0008233"],
    "bp": ["GO:0006810", "GO:0009056", "GO:0016070", "GO:0006950", "GO:0007154", "GO:0043412", "GO:0006082", "GO:0016043"],
}
TASKS = {  # layer name -> (task directory, description)
    "ec": ("ec_classification", "EC classification: ProteinWorkshop ECReaction chains, labelled by top-level EC class"),
    "go_mf": ("go_mf", "GO molecular function: nrPDB-GO chains, labelled by the least frequent of eight frequent MF terms"),
    "go_bp": ("go_bp", "GO biological process: nrPDB-GO chains, labelled by the least frequent of eight frequent BP terms"),
    "localisation": ("subcellular_localization", "DeepLoc 2 subcellular localisation, labelled by compartment"),
    "homology": ("remote_homology", "TAPE remote homology (SCOP domains), labelled by SCOP class"),
    "solubility": ("solubility", "DeepSol solubility, labelled soluble or insoluble"),
    "membrane": ("membrane_soluble", "DeepLoc 2 membrane versus soluble, labelled membrane or soluble"),
    "stability": ("stability", "Rocklin mini-protein stability, labelled by quartile of the stability score"),
    "fluorescence": ("fluorescence", "GFP fluorescence, labelled by quartile of log fluorescence"),
    "affinity": ("ppi_affinity", "PPI affinity chains, labelled by quartile of the mean affinity of the chain's pairs"),
}


def registry():
    reg = {}
    for d in sorted(EMB.glob("stability/*/meta.json")):
        key = d.parent.name
        if key.startswith("ref_"):
            continue
        m = json.loads(d.read_text())
        p = m.get("n_parameters")
        params = "0" if not p else (f"{p / 1e9:.1f}B" if p >= 1e9 else f"{p / 1e6:.2g}M" if p >= 1e6 else f"{p / 1e3:.0f}K")
        reg[key] = (DISP.get(key, key), MODALITY[DECLARED.get(key, "struct")], m.get("weights", "pretrained"),
                    params, True, m.get("inputs_passed", ""))
    return reg


def ids(task, key, split):
    return (EMB / task / key / f"ids_{split}.txt").read_text().split("\n")[:-1]


def pool_ids(task, splits):
    return sum((ids(task, REFERENCE, s) for s in splits), [])


def make_loader(task, ref_ids, rows_idx, splits):
    def load_rows(key):
        got = sum((ids(task, key, s) for s in splits), [])
        assert got == ref_ids, f"{key}: {task} ids differ from {REFERENCE}"
        x = np.concatenate([np.load(EMB / task / key / f"{s}.npy", mmap_mode="r") for s in splits])
        return np.asarray(x[rows_idx])
    return load_rows


def quartiles(v):
    q = np.quantile(v, [0.25, 0.5, 0.75])
    return np.searchsorted(q, v, side="right"), list(QUARTERS)


def manifest(task):
    return pd.read_csv(RES / "extract" / "manifest" / f"{task}.csv").set_index("id")


def build(name, keys, task, ref_ids, groups, class_names, splits, extra=None, extra_fields=None, cand=None):
    """Stratified sample of the pool by `groups`, then the shared layer builder. `cand` restricts the pool rows."""
    rng = np.random.default_rng(SEED)
    cand = np.arange(len(ref_ids)) if cand is None else np.asarray(cand)
    g = np.asarray(groups)[cand]
    sel_local = ba.sample_stratified(g, N_LAYER, rng, floor=FLOOR)
    rows_idx = cand[sel_local]
    classes = sorted(set(g[sel_local].tolist()))
    y = np.array([classes.index(t) for t in g[sel_local]])
    label_names = [class_names[c] for c in classes]
    man = manifest(task)
    chosen = [ref_ids[i] for i in rows_idx]
    proteins = {"id": chosen, "label": y.tolist(), "label_name": [class_names[c] for c in g[sel_local]]}
    if "split" in man.columns and all(c in man.index for c in chosen):
        proteins["split"] = man.loc[chosen, "split"].tolist()
        proteins["length"] = [int(v) if pd.notna(v) else None for v in man.loc[chosen, "length"]]
    for k, v in (extra_fields or {}).items():
        proteins[k] = [v[i] for i in rows_idx]
    counts = dict(zip(*np.unique(proteins["label_name"], return_counts=True)))
    print(f"[{name}] pool {len(cand)} -> sample {len(chosen)}; classes: {counts}", flush=True)
    ex = None if extra is None else np.asarray(extra, dtype=np.float64)[rows_idx]
    layer, cka, ov = ba.build_layer(name, keys, make_loader(task, ref_ids, rows_idx, splits), y, label_names,
                                    np.ones(len(y), bool), proteins, extra=ex)
    if ex is not None:
        for e in layer["encoders"]:
            e["metrics"]["target_spearman_cv5"] = e["metrics"].pop("stability_spearman_cv5")
    layer["summary"] = ba.summarize(layer, cka, ov, name)
    layer["task"] = task
    layer["description"] = TASKS[name][1]
    return layer


# ---------------------------------------------------------------- per-task labels
def layer_ec(keys):
    task, splits = "ec_classification", ("train", "test")
    names = json.loads((RES / "atlas" / "data" / "ec_class_names.json").read_text())
    chain_cls = {}
    for line in open(DATA / "proteinworkshop/ECReaction/chain_functions.txt"):
        p = line.strip().split(",")
        if len(p) >= 2:
            pdb, chain = p[0].split(".", 1)
            chain_cls[f"{pdb.lower()}_{chain}"] = p[1].strip()
    ref_ids = pool_ids(task, splits)
    ref_x = np.concatenate([np.load(EMB / task / REFERENCE / f"{s}.npy", mmap_mode="r") for s in splits])
    _, first = np.unique(np.round(np.asarray(ref_x, dtype=np.float32), 4), axis=0, return_index=True)
    cand = [i for i in sorted(first) if chain_cls.get(ref_ids[i]) in names
            and names[chain_cls[ref_ids[i]]]["top_level_agreement"] >= 0.8]
    top = np.zeros(len(ref_ids), int)
    for i in cand:
        top[i] = names[chain_cls[ref_ids[i]]]["top_level"]
    ec_number = [names[chain_cls[c]]["ec"] if chain_cls.get(c) in names else "" for c in ref_ids]
    return build("ec", keys, task, ref_ids, top, {t: f"EC {t} {ba.EC_TOP[t]}" for t in ba.EC_TOP}, splits,
                 extra_fields={"ec_number": ec_number}, cand=cand)


def layer_go(keys, branch):
    task, splits = f"go_{branch}", ("train", "test")
    block = {"mf": "molecular_function", "bp": "biological_process"}[branch]
    lines = (DATA / "proteinworkshop/GeneOntology/nrPDB-GO_annot.tsv").read_text().splitlines()
    terms = next(lines[i + 1].split("\t") for i, l in enumerate(lines) if l.startswith(f"### GO-terms ({block})"))
    tnames = next(lines[i + 1].split("\t") for i, l in enumerate(lines) if l.startswith(f"### GO-names ({block})"))
    ref_ids = pool_ids(task, splits)
    man = manifest(task)
    labs = [json.loads(man.loc[i, "label"]) for i in ref_ids]
    idx = [terms.index(t) for t in GO_TERMS[branch]]
    freq = {j: sum(j in l for l in labs) for j in idx}
    # the least frequent (most specific) of the chosen terms the protein carries; -1 = none of them
    groups = np.array([min((j for j in idx if j in set(l)), key=lambda j: freq[j], default=-1) for l in labs])
    names = {j: tnames[j] for j in idx}
    names[-1] = "other"
    return build(f"go_{branch}", keys, task, ref_ids, groups, names, splits,
                 extra_fields={"n_terms": [len(l) for l in labs]})


def layer_localisation(keys):
    task, splits = "subcellular_localization", ("train", "test")
    ref_ids = pool_ids(task, splits)
    lab = manifest(task).loc[ref_ids, "label"].to_numpy(int)
    return build("localisation", keys, task, ref_ids, lab, dict(enumerate(LOC)), splits)


def layer_homology(keys):
    import lmdb
    task, splits = "remote_homology", ("train", "test")
    cls = {}
    for f in ("train", "valid", "test_fold_holdout", "test_family_holdout", "test_superfamily_holdout"):
        p = ROOT / "downstream_tasks/data/tape/remote_homology" / f"remote_homology_{f}.lmdb"
        if not p.exists():
            continue
        env = lmdb.open(str(p), readonly=True, lock=False)
        with env.begin() as txn:
            n = pickle.loads(txn.get(b"num_examples"))
            for i in range(n):
                r = pickle.loads(txn.get(str(i).encode()))
                sid = r["id"].decode() if isinstance(r["id"], bytes) else str(r["id"])
                cls[sid] = int(r["class_label"])
    ref_ids = pool_ids(task, splits)
    groups = np.array([cls[i] for i in ref_ids])
    fold = manifest(task).loc[ref_ids, "label"].to_numpy(float)
    keys = [k for k in keys if (EMB / task / k / "ids_train.txt").exists()]
    return build("homology", keys, task, ref_ids, groups, SCOP, splits,
                 extra_fields={"fold_label": [int(f) for f in fold]})


def layer_binary(keys, name, task, names):
    splits = ("train", "test")
    ref_ids = pool_ids(task, splits)
    lab = manifest(task).loc[ref_ids, "label"].to_numpy(float).astype(int)
    return build(name, keys, task, ref_ids, lab, names, splits)


def membrane_names():
    """Which value means membrane: checked against DeepLoc 2's own Membrane column."""
    names = {0: "soluble", 1: "membrane"}
    p = DATA / "deeploc2/Swissprot_Train_Validation_dataset.csv"
    try:
        d = pd.read_csv(p)
        acc = next(c for c in d.columns if c.lower() in ("acc", "accession", "id"))
        man = manifest("membrane_soluble")
        common = man.index.intersection(d[acc])
        agree = (man.loc[common, "label"].to_numpy(float) == d.set_index(acc).loc[common, "Membrane"].to_numpy(float)).mean()
        print(f"[membrane] manifest label == DeepLoc2 Membrane column on {len(common)} proteins: {agree:.3f}", flush=True)
        if agree < 0.5:
            names = {0: "membrane", 1: "soluble"}
    except Exception as e:  # noqa: BLE001
        print("[membrane] could not check the DeepLoc2 column:", e, flush=True)
    return names


def layer_regression(keys, name, task):
    splits = ("train", "test")
    ref_ids = pool_ids(task, splits)
    score = manifest(task).loc[ref_ids, "label"].to_numpy(float)
    groups, qnames = quartiles(score)
    fields = {"target": [round(float(s), 4) for s in score]}
    if task == "stability":
        pat = re.compile(r"^(EEHEE|EHEE|HEEH|HHH)_")
        fields["topology"] = [(pat.match(i).group(1) if pat.match(i) else "other") for i in ref_ids]
    return build(name, keys, task, ref_ids, groups, dict(enumerate(qnames)), splits, extra=score, extra_fields=fields)


def layer_affinity(keys):
    task, splits = "ppi_affinity", ("chains",)
    ref_ids = pool_ids(task, splits)
    pairs = pd.read_csv(RES / "extract" / "pairs" / "ppi_affinity_pairs.csv")
    long = pd.concat([pairs[["chain_a", "label"]].rename(columns={"chain_a": "chain"}),
                      pairs[["chain_b", "label"]].rename(columns={"chain_b": "chain"})])
    agg = long.groupby("chain").label.agg(["mean", "count"])
    cand = [i for i, c in enumerate(ref_ids) if c in agg.index]
    mean_aff = np.full(len(ref_ids), np.nan); n_pairs = np.zeros(len(ref_ids), int)
    for i in cand:
        mean_aff[i] = agg.loc[ref_ids[i], "mean"]; n_pairs[i] = agg.loc[ref_ids[i], "count"]
    groups = np.zeros(len(ref_ids), int)
    groups[cand], qnames = quartiles(mean_aff[cand])
    return build("affinity", keys, task, ref_ids, groups, dict(enumerate(qnames)), splits, extra=mean_aff,
                 extra_fields={"mean_affinity": [None if np.isnan(v) else round(float(v), 4) for v in mean_aff],
                               "n_pairs": n_pairs.tolist()}, cand=cand)


BUILDERS = {
    "ec": layer_ec, "go_mf": lambda k: layer_go(k, "mf"), "go_bp": lambda k: layer_go(k, "bp"),
    "localisation": layer_localisation, "homology": layer_homology,
    "solubility": lambda k: layer_binary(k, "solubility", "solubility", {0: "insoluble", 1: "soluble"}),
    "membrane": lambda k: layer_binary(k, "membrane", "membrane_soluble", membrane_names()),
    "stability": lambda k: layer_regression(k, "stability", "stability"),
    "fluorescence": lambda k: layer_regression(k, "fluorescence", "fluorescence"),
    "affinity": layer_affinity,
}


def merge():
    layers = {}
    for name in TASKS:
        p = OUT / f"{name}.json"
        if p.exists():
            layers[name] = json.loads(p.read_text())
        else:
            print("missing layer:", name)
    atlas = {"meta": {"created": time.strftime("%Y-%m-%d"), "k_neighbours": ba.K, "seed": SEED,
                      "reference_encoder_for_alignment": REFERENCE,
                      "coords_format": "flat [x0, y0, x1, y1, ...], 3 decimals; umap_aligned and pca are centred and scaled to unit RMS radius",
                      "source": "protcompass/extract/embeddings (audited inputs, protein-level mean pooling); one layer per scored downstream task",
                      "modality": "declared class of the paper's taxonomy; the measured input audit is in extract/encoder_input_audit.csv",
                      "sampling": f"stratified by label, {N_LAYER} target with a floor of {FLOOR} per class, seed {SEED}",
                      "layers": {k: v[1] for k, v in TASKS.items()}},
             "layers": layers}
    path = OUT / "atlas.json"
    path.write_text(json.dumps(atlas, separators=(",", ":")))
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB) with {len(layers)} layers", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=list(TASKS))
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.merge:
        merge(); return
    reg = registry()
    keys = list(reg)
    assert REFERENCE in keys and len(keys) == 33, (len(keys), keys)
    ba.ENCODERS, ba.REFERENCE, ba.OUT = reg, REFERENCE, OUT
    ba.IIP_CSV = OUT / "no_iip_means.csv"
    for name in a.tasks:
        t0 = time.time()
        layer = BUILDERS[name](keys)
        (OUT / f"{name}.json").write_text(json.dumps(layer, separators=(",", ":")))
        print(f"[{name}] written ({time.time() - t0:.0f}s)\n", flush=True)


if __name__ == "__main__":
    main()
