"""Modality ablation, probing contrasts (PRESPEC.md, step 3), with the machinery of scripts/symmetric_decide.py.

Reads the new sources from ab_modality/symmetric/results_superfamily/by_source/ and the released esm3, saprot and
floor results from symmetric/results_superfamily/by_source/ (read only). Readout family of every source and the floor
(descriptor, family) are chosen on validation; intervals are the protein-level paired bootstrap of symmetric_decide.Boot.
Writes ab_modality/probing_deltas.csv and ab_modality/probing_floor_cells.csv.
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
AB = ROOT / "extract" / "ab_modality"
sys.path.insert(0, str(ROOT / "scripts"))
import symmetric_decide as sd                                            # noqa: E402
from floor_rules import SEQ_FLOORS                                       # noqa: E402

assert sd.SPLIT == "superfamily"
NEW = AB / "symmetric" / "results_superfamily" / "by_source"
REL = sd.RES
PAIRS = [("esm3", "esm3_seqonly"), ("saprot", "saprot_seqonly")]
PROBES = sd.DETERMINED + sd.UNDETERMINED
DRAWS = 2000


def where(src):
    return NEW if (NEW / f"{src}.csv").exists() else REL


class Boot2(sd.Boot):
    def pred(self, src):
        if src not in self.cache:
            self.cache[src] = dict(np.load(where(src) / f"{src}__test_predictions.npz", allow_pickle=False))
        return self.cache[src]


def load_scores(sources):
    d = pd.concat([pd.read_csv(where(s) / f"{s}.csv") for s in sources], ignore_index=True)
    d["family"] = np.where(d.readout == "linear", "linear", "nonlinear")
    return d.groupby(["source", "probe", "family", "split"]).headline.mean().unstack("split").reset_index()


def state(lo, hi, pos, neg):
    return pos if lo > 0 else neg if hi < 0 else "unresolved"


def main():
    for s in ("esm3_seqonly", "saprot_seqonly"):
        assert (NEW / f"{s}.csv").exists(), s
    S = load_scores([s for p in PAIRS for s in p] + SEQ_FLOORS)
    val = S.set_index(["source", "probe", "family"])["val"]
    test = S.set_index(["source", "probe", "family"])["test"]
    B = Boot2(DRAWS)

    def choose(src_list, probe):                                        # symmetric_decide.main.choose
        best = None
        for s in src_list:
            for fam in ("linear", "nonlinear"):
                if (s, probe, fam) in val.index and np.isfinite(val[(s, probe, fam)]):
                    if best is None or val[(s, probe, fam)] > best[2] + 1e-12:
                        best = (s, fam, val[(s, probe, fam)])
        return best

    released = pd.read_csv(ROOT / "symmetric" / "cells_superfamily.csv")
    released = released[released.rule == "admissible"].set_index(["encoder", "probe"])
    drows, frows = [], []
    for p in PROBES:
        for full, seq in PAIRS:
            cf, cs = choose([full], p), choose([seq], p)
            g = B.gap(p, full, cf[1], seq, cs[1])
            lo, hi = np.percentile(g, [2.5, 97.5])
            lo3, hi3 = np.percentile(g, [0.05, 99.95])
            drows.append(dict(probe=p, determined=p in sd.DETERMINED, model=full, fam_full=cf[1], fam_seqonly=cs[1],
                              S_full=test[(full, p, cf[1])], S_seqonly=test[(seq, p, cs[1])],
                              delta=test[(full, p, cf[1])] - test[(seq, p, cs[1])], lo=lo, hi=hi, lo999=lo3, hi999=hi3,
                              state=state(lo, hi, "structure adds", "structure costs"),
                              state999=state(lo3, hi3, "structure adds", "structure costs")))
        for enc in [s for _, s in PAIRS] + [f for f, _ in PAIRS]:
            ce = choose([enc], p)
            cfl = choose(SEQ_FLOORS, p)
            if enc in ("esm3", "saprot"):                                  # released cell, for reference only
                r = released.loc[(enc, p)] if (enc, p) in released.index else None
                if r is not None:
                    frows.append(dict(probe=p, determined=p in sd.DETERMINED, source=enc, rule="admissible (released cell)",
                                      fam=r.enc_family, S=r.enc_test, floor=r.floor, floor_family=r.floor_family,
                                      floor_S=r.floor_test, gap=r.gap, lo=r.lo, hi=r.hi, state=r.state,
                                      state999=r.state999))
                continue
            g = B.gap(p, enc, ce[1], cfl[0], cfl[1])
            lo, hi = np.percentile(g, [2.5, 97.5])
            lo3, hi3 = np.percentile(g, [0.05, 99.95])
            frows.append(dict(probe=p, determined=p in sd.DETERMINED, source=enc, rule="admissible = sequence floors",
                              fam=ce[1], S=test[(enc, p, ce[1])], floor=cfl[0], floor_family=cfl[1],
                              floor_S=test[(cfl[0], p, cfl[1])], gap=test[(enc, p, ce[1])] - test[(cfl[0], p, cfl[1])],
                              lo=lo, hi=hi, state=state(lo, hi, "above", "below"),
                              state999=state(lo3, hi3, "above", "below")))
        print(f"{p} done", flush=True)
    D, F = pd.DataFrame(drows), pd.DataFrame(frows)
    D.to_csv(AB / "probing_deltas.csv", index=False)
    F.to_csv(AB / "probing_floor_cells.csv", index=False)
    pd.set_option("display.width", 250)
    print(D.round(4).to_string(index=False))
    print(F.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
