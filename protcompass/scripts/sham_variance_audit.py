"""How much of the representation does each arm of the erasure actually remove?

The permuted sham is built to match the real erasure in rank. This script measures the other quantity that decides
what a head sees: the share of the representation's variance the map removes. It fits no head, so it runs over the
whole released panel cheaply, and it is the evidence behind the statement that the two arms are not matched in
amount.

For every released stage-1 cell it reports, on the conditional target the naming claims use:

  var_erased   share of representation variance the LEACE map removes
  var_sham     the same for the permuted sham, averaged over its seeds
  var_null     the same for the variance-matched null of scripts/matched_null.py
  prop_*       share of the property's linear recoverability each map removes

  python sham_variance_audit.py [--tasks ...] [--concepts ...]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common_struct                                                    # noqa: E402
from common_struct import DEPENDS_ON, INDEPENDENT_BLOCKS                # noqa: E402
from erasure import EMB, OUT, TASK_KIND, concepts_for, leace_eraser, load_task   # noqa: E402
from matched_null import matched_null_projector, property_share_removed, variance_removed   # noqa: E402

SHAM_SEEDS = (0, 1, 2)
NULL_SEEDS = (0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=list(TASK_KIND))
    ap.add_argument("--concepts", nargs="+", default=None)
    ap.add_argument("--out", default=str(OUT / "sham_variance_audit.csv"))
    a = ap.parse_args()
    encoders = sorted(d.name for d in (EMB / "stability").iterdir() if (d / "meta.json").exists())
    rows = []
    for task in a.tasks:
        common_struct.set_task(task)
        for enc in encoders:
            data = load_task(task, enc)
            if data is None:
                continue
            (Xtr, ytr, itr), (_, _, iva), (_, _, ite) = (data[s] for s in ("train", "valid", "test"))
            ok = np.ones(len(ytr), bool) if ytr.ndim > 1 else ~np.isnan(ytr)
            if ok.sum() < 200:
                continue
            conc = concepts_for({"train": itr, "valid": iva, "test": ite})
            X = Xtr[ok]
            for name in [c for c in conc if a.concepts is None or c in a.concepts]:
              try:
                Ztr = conc[name][0][ok]
                others = ([c for c in conc if c != name and c != DEPENDS_ON[name] and c not in DEPENDS_ON]
                          if name in DEPENDS_ON else
                          [c for c in INDEPENDENT_BLOCKS if c != name and c in conc])
                if not others:
                    continue
                O = np.concatenate([conc[c][0] for c in others], 1)[ok]
                Z = Ztr - O @ np.linalg.lstsq(O, Ztr, rcond=None)[0]
                if np.linalg.norm(Z) / max(np.linalg.norm(Ztr - Ztr.mean(0)), 1e-12) < 1e-3:
                    continue
                try:
                    mu, P, r, B = leace_eraser(X, Z)
                except np.linalg.LinAlgError as e:
                    print(f"  [failed] {task}/{enc}/{name}: {e}", flush=True)
                    continue
                if r == 0 or B.shape[1] < 8:
                    continue
                er = lambda W: ((W - mu) @ P.T) @ B
                rec = dict(task=task, encoder=enc, concept=name, rank=r, dim=X.shape[1],
                           var_erased=variance_removed(X, er), prop_erased=property_share_removed(X, Z, er))
                sv, sp = [], []
                for s in SHAM_SEEDS:
                    rs = np.random.default_rng(s)
                    mus, Ps, rr, Bs = leace_eraser(X, Z[rs.permutation(len(Z))])
                    if rr == 0 or Bs.shape[1] < 8:
                        continue
                    es = lambda W, mus=mus, Ps=Ps, Bs=Bs: ((W - mus) @ Ps.T) @ Bs
                    sv.append(variance_removed(X, es))
                    sp.append(property_share_removed(X, Z, es))
                rec["var_sham"] = float(np.mean(sv)) if sv else np.nan
                rec["prop_sham"] = float(np.mean(sp)) if sp else np.nan
                nv, npr = [], []
                for s in NULL_SEEDS:
                    mun, Pn, _, Bn, _, _ = matched_null_projector(X, Z, r, rec["var_erased"], seed=300 + s)
                    en = lambda W, mun=mun, Pn=Pn, Bn=Bn: ((W - mun) @ Pn.T) @ Bn
                    nv.append(variance_removed(X, en))
                    npr.append(property_share_removed(X, Z, en))
                rec["var_null"] = float(np.mean(nv))
                rec["prop_null"] = float(np.mean(npr))
                rec["var_matched"] = bool(abs(rec["var_null"] - rec["var_erased"])
                                          <= 0.01 * max(rec["var_erased"], 1e-12))
                rows.append(rec)
                print(f"  {task:24s} {enc:20s} {name:28s} r{r:3d}  var {rec['var_erased']:.4f}/"
                      f"{rec['var_sham']:.4f}/{rec['var_null']:.4f}  prop {rec['prop_erased']:.3f}/"
                      f"{rec['prop_sham']:.3f}/{rec['prop_null']:.3f}", flush=True)
                pd.DataFrame(rows).to_csv(a.out, index=False)
              except Exception as e:                                    # noqa: BLE001
                print(f"  [failed] {task}/{enc}/{name}: {type(e).__name__}: {e}", flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"\nwritten: {a.out} ({len(rows)} cells)")


if __name__ == "__main__":
    main()
