"""The frozen probing hypotheses on the parts of the confirmation cohort whose topology or superfamily the fitting
data never contained.

The cohort (scripts/confirmation_freeze.py) is 2,000 CATH v4.4 domains absent from the whole v4.3 release. Absence of
the domain does not make its fold new: every cohort domain sits in a v4.3 superfamily. This script therefore
restricts the evaluation to the domains whose CATH topology (C.A.T) or homologous superfamily (C.A.T.H) is absent
from the released training and validation splits, and, as a second reference, from the 3,500 domains the probes were
actually fitted on (the seed-0 draw of run_probing_suite.Data).

The fitting is the released one and is done once: every probe is fitted on the merged file of
scripts/confirmation_merge.py (released train and validation draw, cohort as test), with the released readouts, and
its predictions on every cohort domain are kept. Each subset is then scored from those predictions with the released
metrics, and the frozen decision rules of confirmation_report.py are applied per subset. Nothing is chosen after
looking at a subset: the subsets are fixed by CATH codes before any score is read.

  python confirmation_novelty.py            # fit, predict and score every source
  python confirmation_novelty.py --report   # apply the frozen rules per subset
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
import os
import sys
from pathlib import Path

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
CONF = Path(os.environ.get("CONFIRMATION_DIR") or ROOT / "confirmation")   # cohort B: confirmation2
NOV = CONF / "novelty"
os.environ["PROBING_LABELS"] = str(CONF / "merged" / "labels.h5")
os.environ["PROBING_EMB"] = str(CONF / "merged")
os.environ["PROBING_OUT"] = str(NOV)
sys.path.insert(0, str(ROOT / "scripts"))

import h5py                                                              # noqa: E402
import numpy as np                                                       # noqa: E402
import pandas as pd                                                      # noqa: E402
import torch                                                             # noqa: E402

import run_probing_suite as rps                                          # noqa: E402
from confirmation_merge import released_fit_draw                         # noqa: E402

SOURCES = ["ref_length", "ref_position", "ref_aa", "ref_seq_window", "ref_seq_expert", "ref_seq_global",
           "ref_coords", "ref_backbone", "ref_global", "ref_pairhist",
           "esm2_650m", "prottrans_t5_xl_u50", "ankh_base", "esm3", "esm_if1", "proteinmpnn"]


def cath_codes(frame):
    t = frame["C"].astype(str) + "." + frame["A"].astype(str) + "." + frame["T"].astype(str)
    return t, t + "." + frame["H"].astype(str)


def subsets(data):
    """Protein-index sets of the merged file's test part: the whole cohort and four novelty subsets."""
    with h5py.File(ROOT / "probing_data" / "labels.h5", "r") as h:
        P = h["protein"]
        rel = pd.DataFrame({k: P[f"cath_{k}"][:] for k in "CATH"})
        split = P["split_superfamily"][:]
        tr, va = released_fit_draw(h)
    rel_t, rel_h = cath_codes(rel)
    seen = {"split": (set(rel_t[split <= 1]), set(rel_h[split <= 1])),
            "draw": (set(rel_t.iloc[np.concatenate([tr, va])]), set(rel_h.iloc[np.concatenate([tr, va])]))}
    ids = [x.decode() for x in data.P["id"][:]]
    man = pd.read_csv(CONF / "cohort_manifest.csv").rename(
        columns={"class": "C", "architecture": "A", "topology": "T", "homologous_superfamily": "H"}).set_index("domain")
    test = data.prot_idx["test"]
    coh = man.loc[[ids[i] for i in test]]
    ct, chh = cath_codes(coh)
    out = {"cohort": set(test.tolist())}
    for ref, (st, sh) in seen.items():
        out[f"topology_novel_{ref}"] = {i for i, c in zip(test, ct) if c not in st}
        out[f"superfamily_novel_{ref}"] = {i for i, c in zip(test, chh) if c not in sh}
    return out


def fit_predict(X, y, kind, n_out, pairs=None):
    """The released run_probe, returning test predictions instead of scores."""
    out = []
    if pairs is None:
        if kind == "clf":
            out.append(("linear", 0, rps.fit(X, y, kind, n_out, 0, 0)))
        else:
            out.append(("linear", 0, rps.ridge(X["train"], y["train"], X["val"], y["val"], X["test"])))
    for seed in rps.MLP_SEEDS:
        out.append(("pair_mlp" if pairs is not None else "mlp2", seed, rps.fit(X, y, kind, n_out, rps.HIDDEN, seed, pairs)))
    return out


def run(data, source, S):
    rows = []

    def add(probe, level, preds, y, owner):
        for name, members in S.items():
            m = np.isin(owner, list(members))
            if m.sum() < 5:
                continue
            for readout, seed, pr in preds:
                sc = rps.score(level[1], y["test"][m], pr[m], y["train"])
                rows.append(dict(subset=name, n_proteins=int(len(set(owner[m].tolist()))), source=source, probe=probe,
                                 level=level[0], readout=readout, seed=seed, **sc))

    raw = data.residue_features(source)
    X = dict(zip(rps.PARTS, rps.standardise(raw["train"], raw["val"], raw["test"])))
    del raw
    owner_rows = data.pidx[data.rows["test"]]
    for name, ds, kind, n_out, iip in rps.RESIDUE_PROBES:
        if name == "sequence_recovery":
            continue
        lab = {p: (np.stack([data.R[d][:][data.rows[p]] for d in ds], 1) if isinstance(ds, tuple)
                   else data.R[ds][:][data.rows[p]]) for p in rps.PARTS}
        ok = {p: (~np.isnan(lab[p]).reshape(len(lab[p]), -1).any(1) if kind != "clf" else lab[p] >= 0) for p in rps.PARTS}
        Xp = {p: X[p][torch.from_numpy(np.where(ok[p])[0]).to(rps.DEV)] for p in rps.PARTS}
        y = {p: (lab[p][ok[p]].astype(np.int64) if kind == "clf" else lab[p][ok[p]].astype(np.float64)) for p in rps.PARTS}
        add(name, ("residue", kind), fit_predict(Xp, y, kind, n_out), y, owner_rows[ok["test"]])
        del Xp
    qp = data.Q["protein_index"][:]
    owner_pairs = qp[np.where(np.isin(qp, data.chosen["test"]))[0]]
    y = {p: data.pair_y[p] for p in rps.PARTS}
    add("distance_bins", ("pair", "clf"), fit_predict(X, y, "clf", 16, pairs=data.pairs), y, owner_pairs)
    del X; torch.cuda.empty_cache()

    praw = data.protein_features(source)
    PX = dict(zip(rps.PARTS, rps.standardise(praw["train"], praw["val"], praw["test"])))
    owner_prot = data.prot_idx["test"]
    for k in ("radius_of_gyration", "asphericity", "contact_order"):
        y_all = data.P[f"{k}_len_resid_{rps.SPLIT}"][:].astype(np.float64)
        y = {p: y_all[data.prot_idx[p]] for p in rps.PARTS}
        add(f"{k}_len_resid", ("protein", "reg"), fit_predict(PX, y, "reg", 1), y, owner_prot)
    arch = data.P[f"architecture_label_{rps.SPLIT}"][:].astype(np.int64)
    ok = {p: arch[data.prot_idx[p]] >= 0 for p in rps.PARTS}
    PXa = {p: PX[p][torch.from_numpy(np.where(ok[p])[0]).to(rps.DEV)] for p in rps.PARTS}
    y = {p: arch[data.prot_idx[p]][ok[p]] for p in rps.PARTS}
    add("architecture_label", ("protein", "clf"), fit_predict(PXa, y, "clf", int(arch.max()) + 1), y, owner_prot[ok["test"]])
    return pd.DataFrame(rows)


def report():
    import confirmation_report as cr
    d = pd.concat([pd.read_csv(f) for f in sorted((NOV / "by_source").glob("*.csv"))], ignore_index=True)
    d["headline"] = cr.headline(d)
    d["family"] = np.where(d.readout == "linear", "linear", "nonlinear")
    audit = pd.read_csv(ROOT / "extract" / "encoder_input_audit.csv").query("dataset == 'probing'").set_index("encoder")
    rows = []
    for sub, g in d.groupby("subset"):
        S = g.groupby(["source", "probe", "family"]).headline.mean().groupby(["source", "probe"]).max().unstack()
        n = int(g.n_proteins[g.level == "protein"].max())
        for enc in [s for s in S.index if not s.startswith("ref_")]:
            crd = bool(audit.loc[enc, "uses_coordinates"])
            pool = cr.FLOORS_SEQ + (cr.FLOORS_COORD if crd else [])
            for p in cr.INFORMATIVE_DETERMINED + cr.UNDETERMINED:
                if p not in S.columns:
                    continue
                fl = S.loc[[f for f in pool if f in S.index], p].max()
                rows.append(dict(subset=sub, n=n, encoder=enc, coordinates=crd, probe=p, score=S.loc[enc, p],
                                 floor=fl, below=bool(S.loc[enc, p] < fl), determined=p in cr.INFORMATIVE_DETERMINED))
    C = pd.DataFrame(rows)
    C.to_csv(NOV / "novelty_cells.csv", index=False)
    out = []
    for sub, g in C.groupby("subset"):
        det = g[g.determined]
        crd, seq = det[det.coordinates], det[~det.coordinates]
        per = crd.groupby("encoder").below.sum()
        und = g[~g.determined]
        best = und.groupby("probe").apply(lambda x: (x.score - x.floor).max())
        arch = {}
        Sg = d[d.subset == sub].groupby(["source", "probe", "family"]).headline.mean().groupby(["source", "probe"]).max()
        for e in ("esm_if1", "esm3", "esm2_650m"):
            if (e, "architecture_label") in Sg.index and ("ref_pairhist", "architecture_label") in Sg.index:
                arch[e] = float(Sg[(e, "architecture_label")] - Sg[("ref_pairhist", "architecture_label")])
        h4 = bool(arch.get("esm_if1", -1) > 0 and arch.get("esm3", -1) > 0 and arch.get("esm2_650m", 1) <= 0)
        out.append(dict(subset=sub, n=int(g.n.iloc[0]), crd_below=int(crd.below.sum()), crd_cells=len(crd),
                        H4=h4, arch_vs_pairhist=" ".join(f"{e}:{v:+.3f}" for e, v in arch.items()),
                        H1=bool((per >= 7).all()), per_encoder=" ".join(f"{e}:{int(v)}" for e, v in per.items()),
                        seq_below=int(seq.below.sum()), seq_cells=len(seq), H2=bool(seq.below.sum() == 0),
                        und_best_gap=" ".join(f"{p}:{v:+.3f}" for p, v in best.items()), H3=bool((best > 0).all())))
    R = pd.DataFrame(out)
    R.to_csv(NOV / "novelty_verdicts.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("sources", nargs="*")
    a = ap.parse_args()
    if a.report:
        return report()
    rps.N_PROT["test"] = 10 ** 6                     # every cohort domain; train and validation are the released draw
    (NOV / "by_source").mkdir(parents=True, exist_ok=True)
    data = rps.Data()
    S = subsets(data)
    print({k: len(v) for k, v in S.items()}, flush=True)
    for s in a.sources or SOURCES:
        f = NOV / "by_source" / f"{s}.csv"
        if f.exists():
            continue
        for attempt in range(6):
            try:
                run(data, s, S).to_csv(f, index=False)
                print(f"[{s}] done", flush=True)
                break
            except (torch.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" not in str(e).lower():
                    raise
                torch.cuda.empty_cache()
                print(f"[{s}] CUDA out of memory (attempt {attempt + 1}/6); waiting 5 min", flush=True)
                import time; time.sleep(300)


if __name__ == "__main__":
    main()
