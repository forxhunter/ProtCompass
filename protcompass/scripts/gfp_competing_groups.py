"""Descriptive (not pre-specified): the score of each competing group alone and with the descriptor, no representation.
Writes erasure/gfp_competing/groups.csv."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common_struct                                                    # noqa: E402
from erasure import concepts_for, fit_score, load_task                  # noqa: E402
from gfp_competing import DIR, TASK, covariates                          # noqa: E402

common_struct.set_task(TASK)
C = covariates()
data = load_task(TASK, "esm2_8m")
(_, ytr, itr), (_, yva, iva), (_, yte, ite) = (data[s] for s in ("train", "valid", "test"))
ok = ~np.isnan(ytr)
conc = concepts_for({"train": itr, "valid": iva, "test": ite})
G = {"C1 mutation burden": [c for c in C.columns if c.startswith(("n_mut", "bin_"))],
     "C2 substitution classes": ["hyd_to_non", "non_to_hyd", "to_pro", "from_pro", "to_gly", "from_gly", "charge_altering",
                                 "net_charge_change", "arom_gained", "arom_lost", "n_gaps"],
     "C3 composition change": [c for c in C.columns if c.startswith("dcomp_")], "C4 mean pLDDT": ["mean_plddt"]}
G["C1+C2+C4"] = G["C1 mutation burden"] + G["C2 substitution classes"] + G["C4 mean pLDDT"]
rows = []
for g, cols in G.items():
    parts = [C.loc[np.asarray(itr)[ok], cols].to_numpy(float), C.loc[iva, cols].to_numpy(float), C.loc[ite, cols].to_numpy(float)]
    s0 = fit_score(parts[0], ytr[ok], parts[1], yva, parts[2], yte, "reg")
    rec = dict(group=g, n_cols=len(cols), S=s0)
    for name, tag in (("amino-acid composition", "comp"), ("burial and packing", "burial")):
        q = conc[name]
        rec[f"S_plus_{tag}"] = fit_score(np.hstack([parts[0], q[0][ok]]), ytr[ok], np.hstack([parts[1], q[1]]), yva,
                                         np.hstack([parts[2], q[2]]), yte, "reg")
    rows.append(rec)
pd.DataFrame(rows).to_csv(DIR / "groups.csv", index=False)
print(pd.DataFrame(rows).round(4).to_string(index=False))
