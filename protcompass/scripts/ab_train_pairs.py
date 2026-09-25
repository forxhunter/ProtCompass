"""Pretrained versus randomly initialised pairs of the same architecture and input (symmetric/ab_train_pairs/PRESPEC.md).

Probing: for each pair and each of the fifteen admissible-cell probes, delta_train = S_test(pretrained) - S_test(random),
each source with its own validation-chosen readout family, and a protein-level paired bootstrap interval computed with
the unchanged `Boot` class of scripts/symmetric_decide.py (2,000 draws, one fixed weight matrix per probe shared by
both sources). The admissible floor of each version is attached from symmetric/cells_superfamily.csv.

Downstream: delta_train from extract/scores2_selected.csv with the seed standard deviations (no per-protein downstream
predictions exist for the random encoders).

  python ab_train_pairs.py
Output: symmetric/ab_train_pairs/{probing_cells,probing_summary,downstream,downstream_summary}.csv
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
sys.path.insert(0, str(ROOT / "scripts"))
import symmetric_decide as sd                                            # noqa: E402

OUT = ROOT / "symmetric" / "ab_train_pairs"
SYM = ROOT / "symmetric"
DRAWS = 2000

PAIRS = [  # name, random, pretrained, primary
    ("GCPNet", "gcpnet_random", "gcpnet_pretrained", True),
    ("GearNet", "gearnet", "gearnet_pretrained", True),
    ("IPA", "ipa", "ipa_pretrained", True),
    ("TFN", "tfn", "tfn_pretrained", True),
    ("PottsMPNN", "pottsmpnn", "pottsmpnn_pretrained", True),
    ("PottsMPNN-soluble", "pottsmpnn", "pottsmpnn_soluble", False),
    ("PottsMPNN-ft", "pottsmpnn", "pottsmpnn_ft", False),
    ("GCPNet-vframe", "gcpnet_random_vframe", "gcpnet_pretrained_vframe", False),
]
PROBES = sd.DETERMINED + sd.UNDETERMINED


def check_prespec():
    rec = (OUT / "PRESPEC.sha256").read_text().split()[0]
    now = hashlib.sha256((OUT / "PRESPEC.md").read_bytes()).hexdigest()
    if rec != now:
        raise SystemExit("PRESPEC.md changed after its hash was stored")
    return now


def floors():
    """Admissible floor of each encoder and probe: cells_superfamily for the panel, vframe_superfamily for vframe."""
    C = pd.read_csv(SYM / "cells_superfamily.csv")
    A = C[C.rule == "admissible"][["encoder", "probe", "floor", "floor_family", "floor_test", "enc_test", "enc_family",
                                   "gap", "lo", "hi", "state"]].copy()
    V = pd.read_csv(SYM / "vframe_superfamily.csv")
    V = V.assign(encoder=V.encoder + "_vframe", floor_family="", floor_test=V.score_frame - V.gap_frame,
                 enc_test=V.score_frame, enc_family="", gap=V.gap_frame, lo=V.lo_frame, hi=V.hi_frame,
                 state=V.state_frame)[A.columns]
    return pd.concat([A, V], ignore_index=True).set_index(["encoder", "probe"])


def main():
    h = check_prespec()
    print(f"PRESPEC sha256 {h}", flush=True)
    S = sd.load_scores()
    val = S.set_index(["source", "probe", "family"])["val"]
    test = S.set_index(["source", "probe", "family"])["test"]
    B = sd.Boot(DRAWS)
    F = floors()

    def choose(src_list, probe):        # identical to the closure in symmetric_decide.main
        best = None
        for s in src_list:
            for fam in ("linear", "nonlinear"):
                if (s, probe, fam) in val.index and np.isfinite(val[(s, probe, fam)]):
                    if best is None or val[(s, probe, fam)] > best[2] + 1e-12:
                        best = (s, fam, val[(s, probe, fam)])
        return best

    rows = []
    for name, r, p, primary in PAIRS:
        for probe in PROBES:
            cr, cp = choose([r], probe), choose([p], probe)
            if cr is None or cp is None:
                print(f"  {name} {probe}: missing source score", flush=True)
                continue
            sr, sp = test[(r, probe, cr[1])], test[(p, probe, cp[1])]
            row = dict(pair=name, primary=primary, probe=probe, determined=probe in sd.DETERMINED,
                       tautological=probe in sd.TAUTOLOGICAL, random=r, pretrained=p,
                       random_family=cr[1], pretrained_family=cp[1], random_test=sr, pretrained_test=sp,
                       delta=sp - sr)
            try:
                mr, pr_, yr = B.score(probe, r, cr[1])
                mp, pp_, yp = B.score(probe, p, cp[1])
                if np.array_equal(pr_, pp_) and np.array_equal(yr, yp):
                    g = mp - mr
                    lo, hi = np.percentile(g, [2.5, 97.5])
                    lo3, hi3 = np.percentile(g, [0.05, 99.95])
                    row.update(lo=lo, hi=hi, lo999=lo3, hi999=hi3, boot_mean=float(np.mean(g)),
                               n_test_proteins=len(np.unique(pr_)), n_test_rows=len(yr),
                               state="increase" if lo > 0 else "decrease" if hi < 0 else "unresolved",
                               state999="increase" if lo3 > 0 else "decrease" if hi3 < 0 else "unresolved")
                else:
                    row.update(state="no_interval", note="test rows differ between the two sources")
            except KeyError as e:
                row.update(state="no_interval", note=f"prediction missing: {e}")
            for tag, e in (("random", r), ("pretrained", p)):
                if (e, probe) in F.index:
                    f = F.loc[(e, probe)]
                    row.update({f"{tag}_floor": f.floor, f"{tag}_floor_test": f.floor_test,
                                f"{tag}_floor_gap": f.gap, f"{tag}_floor_lo": f.lo, f"{tag}_floor_hi": f.hi,
                                f"{tag}_floor_state": f.state,
                                f"{tag}_check": abs(f.enc_test - (sr if tag == "random" else sp))})
            rows.append(row)
        print(f"{name:18s} done", flush=True)
    D = pd.DataFrame(rows)
    # the recomputed encoder test scores must equal the published cells
    worst = np.nanmax(D[["random_check", "pretrained_check"]].values)
    print(f"max |enc_test(cells) - recomputed| = {worst:.2e}", flush=True)
    assert worst < 1e-9, worst
    D = D.drop(columns=["random_check", "pretrained_check"])
    D.to_csv(OUT / "probing_cells.csv", index=False)

    summ = []
    for name, g in D.groupby("pair", sort=False):
        s = dict(pair=name, primary=bool(g.primary.iloc[0]), random=g.random.iloc[0], pretrained=g.pretrained.iloc[0])
        for sub, gg in (("all", g), ("det", g[g.determined]), ("undet", g[~g.determined])):
            for st in ("increase", "decrease", "unresolved", "no_interval"):
                s[f"{sub}_{st}"] = int((gg.state == st).sum())
            s[f"{sub}_increase999"] = int((gg.state999 == "increase").sum())
            s[f"{sub}_decrease999"] = int((gg.state999 == "decrease").sum())
        det = g[g.determined]
        s["det_cells"] = len(det)
        s["random_below_floor"] = int((det.random_floor_state == "below").sum())
        s["pretrained_below_floor"] = int((det.pretrained_floor_state == "below").sum())
        s["random_above_floor"] = int((det.random_floor_state == "above").sum())
        s["pretrained_above_floor"] = int((det.pretrained_floor_state == "above").sum())
        s["det_floor_state_changed"] = int((det.random_floor_state != det.pretrained_floor_state).sum())
        s["median_delta_all"] = float(g.delta.median())
        s["median_delta_det"] = float(det.delta.median())
        summ.append(s)
    P = pd.DataFrame(summ)
    P.to_csv(OUT / "probing_summary.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(P[["pair", "all_increase", "all_decrease", "all_unresolved", "all_no_interval", "det_increase",
             "det_decrease", "undet_increase", "undet_decrease", "random_below_floor", "pretrained_below_floor",
             "det_floor_state_changed", "median_delta_det"]].to_string(index=False))

    # ------------------------------------------------------------------ downstream
    Ssel = pd.read_csv(ROOT / "extract" / "scores2_selected.csv")
    L = pd.read_csv(ROOT / "extract" / "scores2_long.csv")
    L = L[L.train_frac.fillna(1.0) >= 1.0] if "train_frac" in L else L
    pv = L.groupby(["task", "encoder", "probe"]).val_loss.mean().reset_index()
    best = pv.loc[pv.groupby(["task", "encoder"]).val_loss.idxmin(), ["task", "encoder", "probe"]]
    metric = L.groupby("task").metric.first()
    sel = Ssel.merge(best, on=["task", "encoder"], how="left").set_index(["task", "encoder"])
    drows = []
    for name, r, p, primary in PAIRS:
        if name.endswith("vframe"):
            continue
        for task in sorted(Ssel.task.unique()):
            if (task, r) not in sel.index or (task, p) not in sel.index:
                drows.append(dict(pair=name, primary=primary, task=task, note="missing score"))
                continue
            a, b = sel.loc[(task, r)], sel.loc[(task, p)]
            se = float(np.sqrt(a.seed_sd ** 2 / a.n_seeds + b.seed_sd ** 2 / b.n_seeds))
            d = float(b.score - a.score)
            drows.append(dict(pair=name, primary=primary, task=task, metric=metric[task], random=r, pretrained=p,
                              random_readout=a.probe, pretrained_readout=b.probe,
                              random_score=a.score, random_seed_sd=a.seed_sd, pretrained_score=b.score,
                              pretrained_seed_sd=b.seed_sd, delta=d, se_seed=se,
                              seed_flag="higher" if d > 2 * se else "lower" if d < -2 * se else "within"))
    DD = pd.DataFrame(drows)
    DD.to_csv(OUT / "downstream.csv", index=False)
    ds = DD.groupby("pair", sort=False).agg(primary=("primary", "first"),
                                            higher=("seed_flag", lambda x: int((x == "higher").sum())),
                                            lower=("seed_flag", lambda x: int((x == "lower").sum())),
                                            within=("seed_flag", lambda x: int((x == "within").sum())),
                                            median_delta=("delta", "median")).reset_index()
    ds.to_csv(OUT / "downstream_summary.csv", index=False)
    print(DD.pivot(index="task", columns="pair", values="delta").round(3).to_string())
    print(ds.to_string(index=False))


if __name__ == "__main__":
    main()
