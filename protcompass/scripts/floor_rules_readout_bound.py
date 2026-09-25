"""Bound on the floor counts under any choice of encoder readout family.

The released sweep scored encoders on the test split only, so the encoder's readout family (linear or MLP-2) is taken
as the better one on test. A validation choice picks one of the same two families, so its count lies between the
count with the better family everywhere and the count with the worse family everywhere. Both extremes are computed
here from the released test scores, under the released rule and under the admissible, validation-selected floors.

  python floor_rules_readout_bound.py   # writes probing_results/floor_rules_readout_bound.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import floor_rules as fr                                                 # noqa: E402

S = fr.scores([fr.RES]).pivot_table(index=["source", "probe"], columns="family", values="score")
S["worst"], S["best"] = S.min(1), S.max(1)
c = pd.read_csv(fr.RES / "floor_rules_cells.csv")
rows = []
for rule in ["released", "both"]:
    d = c[c.rule == rule].join(S, on=["encoder", "probe"])
    for coord, g in d.groupby("coordinates"):
        rows.append(dict(rule=rule, coordinates=bool(coord), cells=len(g), below_best=int((g.best <= g.floor).sum()),
                         below_worst=int((g.worst <= g.floor).sum()),
                         min_margin_worst=float((g.worst - g.floor).min()) if not coord else float("nan")))
out = pd.DataFrame(rows)
out.to_csv(fr.RES / "floor_rules_readout_bound.csv", index=False)
print(out.to_string(index=False))
