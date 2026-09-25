"""Decisions of the symmetric probing rerun (scripts/symmetric_run.py): readout family and floor chosen on validation
for encoders and floors alike, and a protein-level paired bootstrap interval on every encoder-minus-floor gap.

A cell (encoder, probe) is `below` when the upper end of the 95% interval of the gap is below zero, `above` when the
lower end is above zero, and `unresolved` otherwise; the same three states are also reported at 99.9%. The encoder's
readout family (linear, or the mean of the three MLP seeds) is the one with the higher validation score; the floor is
the (descriptor, family) with the highest validation score among the floors the rule allows the encoder; both test
scores are then read once. Encoder and floor are resampled together (the same proteins in every draw).

Floor rules:
  admissible   the floors built only from inputs the encoder is measured to read (scripts/floor_rules.py): residue
               identity from the input audit, the atom set from the atom-level audit, multi-atom floors rebuilt on it
  group        the released eligibility: a coordinate encoder is held to all ten released floors, a sequence-only
               encoder to the six sequence floors
  nbrK         the label-agnostic family only: length, position, and for a residue-identity reader the one-hot residue
               and the +-8 one-hot window, plus ref_nbr{K}@S for the encoder's atom set S (residue and pair probes)

  SYM_SPLIT=superfamily python symmetric_decide.py [--draws 2000]
Output: symmetric/cells_<split>.csv, symmetric/counts_<split>.csv, symmetric/selectivity_<split>.csv
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
sys.path.insert(0, str(ROOT / "scripts"))
from floor_rules import ALL_FLOORS, SEQ_FLOORS, admissibility            # noqa: E402
from panel import panel                                                  # noqa: E402
from prereg_decide import metric_from, protein_stats                     # noqa: E402

SPLIT = os.environ.get("SYM_SPLIT", "superfamily")
SYM = ROOT / "symmetric"
RES = SYM / f"results_{SPLIT}" / "by_source"
DETERMINED = ["phi_circular", "psi_circular", "local_curvature", "dssp_3", "dssp_8", "core_surface",
              "contact_density", "distance_bins", "contact_order_len_resid", "radius_of_gyration_len_resid",
              "asphericity_len_resid"]
TAUTOLOGICAL = ["contact_density", "contact_order_len_resid"]
UNDETERMINED = ["rsa", "rsa_bins", "bfactor", "architecture_label"]
RESIDUE_LEVEL = ["phi_circular", "psi_circular", "local_curvature", "dssp_3", "dssp_8", "core_surface",
                 "contact_density", "distance_bins"]
KIND = {p: "circ" for p in ("phi_circular", "psi_circular")}
KIND.update({p: "reg" for p in ("local_curvature", "rsa", "bfactor", "contact_order_len_resid",
                                "radius_of_gyration_len_resid", "asphericity_len_resid")})
KIND.update({p: "clf" for p in ("dssp_3", "dssp_8", "core_surface", "contact_density", "distance_bins", "rsa_bins",
                                "architecture_label", "dssp3_control")})
NBR_K = (4, 8, 16)


def load_scores():
    d = pd.concat([pd.read_csv(f) for f in sorted(RES.glob("*.csv"))], ignore_index=True)
    d["family"] = np.where(d.readout == "linear", "linear", "nonlinear")
    return d.groupby(["source", "probe", "family", "split"]).headline.mean().unstack("split").reset_index()


def eligible(M, e, rule, k=None):
    r = M.loc[e]
    if rule == "admissible":
        return list(r.floors)
    if rule == "group":
        return ALL_FLOORS if r.coordinates else SEQ_FLOORS
    fl = ["ref_length", "ref_position"] + (["ref_aa", "ref_seq_window"] if r.residue_identity else [])
    atoms = r.atoms
    if r.coordinates and "CA" in atoms.split("+"):
        fl.append(f"ref_nbr{k}@{atoms}")
    return fl


class Boot:
    def __init__(self, draws):
        self.draws, self.cache, self.W = draws, {}, {}

    def pred(self, src):
        if src not in self.cache:
            self.cache[src] = dict(np.load(RES / f"{src}__test_predictions.npz", allow_pickle=False))
        return self.cache[src]

    def weights(self, probe, ups):
        if probe not in self.W:
            rng = np.random.default_rng(sum(map(ord, probe)))       # one fixed draw per probe
            self.W[probe] = np.stack([np.bincount(rng.integers(0, len(ups), len(ups)), minlength=len(ups))
                                      for _ in range(self.draws)]).astype(float)
        return self.W[probe]

    def score(self, probe, src, fam):
        P = self.pred(src)
        y, prot = P[f"{probe}|y"], P[f"{probe}|protein"]
        kind = KIND[probe]
        ups = np.unique(prot)
        W = self.weights(probe, ups)
        keys = [k.split("|")[1] for k in P if k.startswith(f"{probe}|") and k.split("|")[1] not in ("y", "protein")]
        keys = [k for k in keys if (k == "linear_0") == (fam == "linear")]
        yy = y.astype(np.int64) if kind == "clf" else y
        out = []
        for k in sorted(keys):
            p = P[f"{probe}|{k}"]
            p = p.astype(np.int64) if kind == "clf" else p
            out.append(metric_from(kind, W, *protein_stats(kind, yy, p, prot, ups)))
        return np.mean(out, 0), prot, y

    def gap(self, probe, enc, efam, flo, ffam):
        me, pe, ye = self.score(probe, enc, efam)
        mf, pf_, yf = self.score(probe, flo, ffam)
        assert np.array_equal(pe, pf_) and np.array_equal(ye, yf), (probe, enc, flo)
        return me - mf


def hypotheses(C, t1=None, draws=10_000):
    """T1-T4 of symmetric/freeze_topology.json, computed the same way on any split."""
    from group_controls import FAMILY
    A = C[C.rule == "admissible"]
    det = A[A.determined & A.coordinates]
    seq = A[A.determined & ~A.coordinates]
    und = A[A.probe.isin(["rsa", "rsa_bins", "bfactor"])]
    out = {"T1_share_below": float((det.state == "below").mean()), "T1_below": int((det.state == "below").sum()),
           "T1_cells": len(det), "T2_seq_below": int((seq.state == "below").sum()), "T2_cells": len(seq),
           "T3_labels_with_encoder_above": {p: sorted(g.encoder[g.state == "above"]) for p, g in und.groupby("probe")}}
    crd = A[A.coordinates & A.probe.isin(DETERMINED + ["rsa", "rsa_bins", "bfactor"])].copy()
    crd["family"] = crd.encoder.map(lambda e: FAMILY.get(e, e))
    fams = sorted(crd.family.unique())
    rng = np.random.default_rng(0)
    by = {f: crd[crd.family == f] for f in fams}

    def sep(sample):
        d = pd.concat([by[f] for f in sample])
        return d.loc[~d.determined, "gap"].mean() - d.loc[d.determined, "gap"].mean()
    obs = sep(fams)
    boot = np.array([sep(rng.choice(fams, len(fams))) for _ in range(draws)])
    out.update(T4_delta=float(obs), T4_lo=float(np.nanpercentile(boot, 2.5)), T4_hi=float(np.nanpercentile(boot, 97.5)),
               T4_families=len(fams))
    out["T1"] = None if t1 is None else bool(out["T1_share_below"] >= t1)
    out["T2"] = out["T2_seq_below"] == 0
    out["T3"] = all(len(out["T3_labels_with_encoder_above"].get(p, [])) > 0 for p in ("rsa", "rsa_bins", "bfactor"))
    out["T4"] = bool(out["T4_lo"] > 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=2000)
    a = ap.parse_args()
    S = load_scores()
    M = admissibility()
    enc = [e for e in panel() if (RES / f"{e}.csv").exists()]
    val = S.set_index(["source", "probe", "family"])["val"]
    test = S.set_index(["source", "probe", "family"])["test"]
    B = Boot(a.draws)

    def choose(src_list, probe):
        best = None
        for s in src_list:
            for fam in ("linear", "nonlinear"):
                if (s, probe, fam) in val.index and np.isfinite(val[(s, probe, fam)]):
                    if best is None or val[(s, probe, fam)] > best[2] + 1e-12:
                        best = (s, fam, val[(s, probe, fam)])
        return best

    rules = [("admissible", None), ("group", None)] + [(f"nbr{k}", k) for k in NBR_K]
    rows = []
    for rule, k in rules:
        probes = DETERMINED + UNDETERMINED if k is None else RESIDUE_LEVEL
        for p in probes:
            for e in enc:
                ch_e = choose([e], p)
                if ch_e is None:
                    continue
                fl = [f for f in eligible(M, e, "nbr" if k else rule, k) if (RES / f"{f}.csv").exists()]
                ch_f = choose(fl, p)
                if ch_f is None:
                    continue
                g_test = test[(e, p, ch_e[1])] - test[(ch_f[0], p, ch_f[1])]
                g = B.gap(p, e, ch_e[1], ch_f[0], ch_f[1])
                lo, hi = np.percentile(g, [2.5, 97.5])
                lo3, hi3 = np.percentile(g, [0.05, 99.95])
                state = "below" if hi < 0 else "above" if lo > 0 else "unresolved"
                state3 = "below" if hi3 < 0 else "above" if lo3 > 0 else "unresolved"
                # the released reading of the same files: better family on test for the encoder
                enc_test_best = max(test.get((e, p, f), -np.inf) for f in ("linear", "nonlinear"))
                rows.append(dict(split=SPLIT, rule=rule, probe=p, encoder=e, coordinates=bool(M.loc[e, "coordinates"]),
                                 determined=p in DETERMINED, tautological=p in TAUTOLOGICAL,
                                 enc_family=ch_e[1], floor=ch_f[0], floor_family=ch_f[1],
                                 enc_test=test[(e, p, ch_e[1])], floor_test=test[(ch_f[0], p, ch_f[1])],
                                 gap=g_test, lo=lo, hi=hi, lo999=lo3, hi999=hi3, state=state, state999=state3,
                                 gap_enc_test_best=enc_test_best - test[(ch_f[0], p, ch_f[1])]))
            print(f"{rule:10s} {p:30s} done", flush=True)
    C = pd.DataFrame(rows)
    C.to_csv(SYM / f"cells_{SPLIT}.csv", index=False)
    cnt = []
    for (rule, crd, det), g in C.groupby(["rule", "coordinates", "determined"]):
        for sub, gg in (("all", g), ("informative", g[~g.tautological])):
            cnt.append(dict(split=SPLIT, rule=rule, coordinates=crd, determined=det, subset=sub, cells=len(gg),
                            point_below=int((gg.gap < 0).sum()),
                            below=int((gg.state == "below").sum()), unresolved=int((gg.state == "unresolved").sum()),
                            above=int((gg.state == "above").sum()),
                            below999=int((gg.state999 == "below").sum()), above999=int((gg.state999 == "above").sum()),
                            point_below_enc_test_best=int((gg.gap_enc_test_best < 0).sum())))
    K = pd.DataFrame(cnt)
    K.to_csv(SYM / f"counts_{SPLIT}.csv", index=False)
    pd.set_option("display.width", 220)
    print(K.to_string(index=False))
    # selectivity (Hewitt and Liang, 2019) on DSSP-3 against the floor verdict on DSSP-3
    sel = []
    for e in enc:
        a_ = choose([e], "dssp_3"); c_ = choose([e], "dssp3_control")
        if a_ is None or c_ is None:
            continue
        v = C[(C.rule == "admissible") & (C.probe == "dssp_3") & (C.encoder == e)]
        sel.append(dict(split=SPLIT, encoder=e, coordinates=bool(M.loc[e, "coordinates"]),
                        residue_identity=bool(M.loc[e, "residue_identity"]),
                        dssp3=test[(e, "dssp_3", a_[1])], control=test[(e, "dssp3_control", c_[1])],
                        selectivity=test[(e, "dssp_3", a_[1])] - test[(e, "dssp3_control", c_[1])],
                        gap=float(v.gap.iloc[0]) if len(v) else np.nan, state=v.state.iloc[0] if len(v) else ""))
    import json
    t1 = None
    fz = SYM / f"freeze_{SPLIT}.json"
    if fz.exists():
        t1 = json.loads(fz.read_text())["thresholds"]["T1"]
    H = hypotheses(C, t1)
    (SYM / f"hypotheses_{SPLIT}.json").write_text(json.dumps(H, indent=2))
    print(json.dumps(H, indent=2))
    Sel = pd.DataFrame(sel)
    Sel.to_csv(SYM / f"selectivity_{SPLIT}.csv", index=False)
    print(Sel.sort_values("selectivity").to_string(index=False))


if __name__ == "__main__":
    main()
