"""Build the probe labels for the frozen confirmation cohort, with nothing refitted on it.

The cohort is 2,000 CATH v4.4 domains absent from the whole v4.3 release, fixed by scripts/confirmation_freeze.py
before anything was computed on them. Labels are produced by the released per-domain routine, imported from
probing_data/make_probing_data.py rather than reimplemented, so the definitions, the cleaning rules, the DSSP binary
and the SASA settings are the ones the released dataset used.

Two quantities in the released file are fitted on training data, and both are carried over rather than refitted:

  length residualisation   contact order, radius of gyration and asphericity are reported as the residual of a
                           quadratic in log length. The coefficients come from the released training split and are
                           read out of labels.h5, so a cohort residual is on the released scale.
  architecture classes     the released run keeps the architectures with at least 20 training domains and indexes
                           them. That index is read out of labels.h5 and applied here; a cohort domain in an
                           architecture the released run dropped is labelled -1 and is not scored.

Every cohort domain is marked test. There is no cohort training split, because the probes are fitted on the released
v4.3 training and validation splits and only evaluated here.

  python confirmation_build_labels.py [--workers 7]
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
import importlib.util
import json
import os
import pickle
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
PD = ROOT / "probing_data"
CONF = Path(os.environ.get("CONFIRMATION_DIR") or ROOT / "confirmation")   # cohort B: confirmation2


def load_builder():
    """Import make_probing_data with its paths pointed at the confirmation cohort."""
    spec = importlib.util.spec_from_file_location("mpd", PD / "make_probing_data.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["mpd"] = m
    spec.loader.exec_module(m)
    m.RAW = CONF                      # process_domain reads RAW/"dompdb"/<dom>
    m.STRUCT = CONF / "structures"
    m.WORK = CONF / "work"
    m.STRUCT.mkdir(parents=True, exist_ok=True)
    m.WORK.mkdir(parents=True, exist_ok=True)
    return m


def released_fits():
    """The training-fitted quantities carried over from the released dataset."""
    with h5py.File(PD / "labels.h5", "r") as h:
        p = h["protein"]
        coef = {k: np.array(p[f"{k}_len_resid_superfamily"].attrs["coef_a_b_c"], dtype=np.float64)
                for k in ("contact_order", "radius_of_gyration", "asphericity")}
        classes = [tuple(int(x) for x in a.split("."))
                   for a in str(p["architecture_label_superfamily"].attrs["classes"]).split(",")]
        attrs = dict(h.attrs)
    return coef, {a: i for i, a in enumerate(classes)}, attrs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    freeze = json.loads((CONF / "freeze.json").read_text())
    cohort = [l.split(",")[0] for l in (CONF / "cohort_manifest.csv").read_text().splitlines()[1:] if l.strip()]
    print(f"cohort: {len(cohort)} domains, frozen {freeze['written_utc']}")

    m = load_builder()
    meta44 = m.read_domain_list(PD / "raw_v4_4" / "cath-domain-list-v4_4_0.txt")

    shard = CONF / "work" / "labels.pkl"
    if shard.exists():
        out = pickle.load(open(shard, "rb"))
        print(f"reusing {shard}")
    else:
        with Pool(a.workers) as pool:
            out = pool.map(m.process_domain, cohort, chunksize=8)
        pickle.dump(out, open(shard, "wb"), protocol=4)

    records = [r for _, reason, _, r in out if reason is None]
    excl = [(d, reason, detail) for d, reason, detail, _ in out if reason]
    records.sort(key=lambda r: r["dom"])
    print(f"kept {len(records)} of {len(cohort)}; {len(excl)} excluded")
    print("  reasons:", dict(Counter(r[1] for r in excl)))
    with open(CONF / "exclusions.csv", "w") as fh:
        fh.write("domain,reason,detail\n")
        for d, reason, detail in excl:
            fh.write(f"{d},{reason},{detail}\n")

    coef, arch_index, released_attrs = released_fits()
    n = len(records)
    L = np.array([r["L"] for r in records], dtype=np.int32)
    offsets = np.concatenate([[0], np.cumsum(L)]).astype(np.int64)
    logL = np.log(L.astype(np.float64))
    X = np.stack([np.ones_like(logL), logL, logL ** 2], axis=1)
    ids = np.array([r["dom"] for r in records], dtype="S7")

    out_h5 = CONF / "labels_v44.h5"
    if out_h5.exists():
        out_h5.unlink()
    with h5py.File(out_h5, "w") as h:
        h.attrs.update(released_attrs)
        h.attrs["cath_release"] = "v4_4_0"
        h.attrs["domain_set"] = ("confirmation cohort: in the v4.4 non-redundant S40 set and absent from the whole "
                                 "v4.3 release; every domain is test")
        h.attrs["frozen_utc"] = freeze["written_utc"]
        h.attrs["cohort_manifest_sha256"] = freeze["cohort"]["manifest_sha256"]
        h.attrs["fits_carried_over"] = ("length-residualisation coefficients and architecture class index are read "
                                        "from the released labels.h5 and are not refitted here")
        gp, gr, gq = h.create_group("protein"), h.create_group("residue"), h.create_group("pair")
        gp.create_dataset("id", data=ids)
        gp.create_dataset("length", data=L)
        gp.create_dataset("offset", data=offsets[:-1])
        for k in "CATH":
            gp.create_dataset(f"cath_{k}", data=np.array([meta44[d.decode()][k] for d in ids], dtype=np.int32))
        gp.create_dataset("resolution", data=np.array([meta44[d.decode()]["resolution"] for d in ids],
                                                      dtype=np.float32))
        for k in ("n_segments", "n_virtual_cb", "n_incomplete", "n_nonstd"):
            gp.create_dataset(k, data=np.array([r[k] for r in records], dtype=np.int32))
        gp.create_dataset("bfactor_ok", data=np.array([r["bfactor_ok"] for r in records]))
        for name in ("superfamily", "topology"):
            ds = gp.create_dataset(f"split_{name}", data=np.full(n, 2, dtype=np.int8))
            ds.attrs["codes"] = "0=train,1=val,2=test; the whole cohort is test"

        for k in ("contact_order", "radius_of_gyration", "asphericity", "asphericity_legacy"):
            y = np.array([r[k] for r in records], dtype=np.float64)
            gp.create_dataset(k, data=y.astype(np.float32))
            if k == "asphericity_legacy":
                continue
            for name in ("superfamily", "topology"):
                ds = gp.create_dataset(f"{k}_len_resid_{name}", data=(y - X @ coef[k]).astype(np.float32))
                ds.attrs["fit"] = "released training-split coefficients, applied not refitted"
                ds.attrs["coef_a_b_c"] = coef[k]

        arch = [(meta44[d.decode()]["C"], meta44[d.decode()]["A"]) for d in ids]
        for name in ("superfamily", "topology"):
            lab = np.array([arch_index.get(t, -1) for t in arch], dtype=np.int16)
            ds = gp.create_dataset(f"architecture_label_{name}", data=lab)
            ds.attrs["classes"] = ",".join(f"{c}.{aa}" for c, aa in sorted(arch_index, key=arch_index.get))
            ds.attrs["source"] = "released architecture class index, applied to the cohort"
        print(f"architecture labels: {int((np.array([arch_index.get(t, -1) for t in arch]) >= 0).sum())} of {n} "
              f"fall in a released class")

        def cat(key, dtype=None):
            v = np.concatenate([r[key] for r in records])
            return v.astype(dtype) if dtype else v

        gr.create_dataset("protein_index", data=np.repeat(np.arange(n, dtype=np.int32), L), compression="gzip")
        gr.create_dataset("position", data=np.concatenate([np.arange(l, dtype=np.int16) for l in L]),
                          compression="gzip")
        for key in ("coords", "aa", "resseq", "icode", "segment", "cb_virtual", "phi", "psi", "phi_bin", "psi_bin",
                    "curvature", "dssp8", "dssp3", "sasa", "rsa", "rsa_bin", "bfactor_raw", "bfactor_z",
                    "contact_count", "contact_count_sep4", "contact_density_bin", "cb_neighbours", "core"):
            gr.create_dataset(key, data=cat(key), compression="gzip")
        for ang in ("phi", "psi"):
            v = gr[ang][:]
            gr.create_dataset(f"{ang}_sin", data=np.sin(v), compression="gzip")
            gr.create_dataset(f"{ang}_cos", data=np.cos(v), compression="gzip")
        gr.create_dataset("sequence_recovery", data=cat("aa"), compression="gzip")

        n_pairs = np.array([len(r["pair_i"]) for r in records], dtype=np.int64)
        gq.create_dataset("protein_index", data=np.repeat(np.arange(n, dtype=np.int32), n_pairs), compression="gzip")
        gq.create_dataset("offset", data=np.concatenate([[0], np.cumsum(n_pairs)])[:-1])
        gq.create_dataset("count", data=n_pairs.astype(np.int32))
        for src, dst in (("pair_i", "i"), ("pair_j", "j"), ("pair_dist", "distance"), ("pair_bin", "distance_bin")):
            gq.create_dataset(dst, data=cat(src), compression="gzip")

    (CONF / "ids.txt").write_text("\n".join(d.decode() for d in ids) + "\n")
    print(f"wrote {out_h5}: {n} domains, {int(L.sum()):,} residues, {int(n_pairs.sum()):,} pairs")
    print(f"wrote {CONF / 'ids.txt'}")


if __name__ == "__main__":
    main()
