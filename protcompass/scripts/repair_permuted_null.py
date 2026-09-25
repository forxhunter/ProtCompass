"""How large a repair contrast does a descriptor with no relation to the protein produce?

Every candidate cell of scripts/addback_named_pairs.py also scores the property's descriptor with its rows permuted
across proteins: the same width and marginal, no protein information. Its repair contrast against the
variance-matched null is the noise level of R_var. This script reports it per cell, per pair, and as the largest
panel mean that resampling the encoders produces, which is the scale the 0.02 floor of the naming rule has to clear.

  python repair_permuted_null.py [--tag full]     # --tag full: the 60 pairs of named_pair_cells_full.csv
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from panel import only_panel                                            # noqa: E402

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
DRAWS = 500


def main():
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else ""
    sfx = f"_{tag}" if tag else ""
    C = only_panel(pd.read_csv(ROOT / "erasure" / f"named_pair_cells{sfx}.csv"))
    rng = np.random.default_rng(0)
    pair_mean = C.groupby(["task", "concept"]).Rperm_null.mean()
    resampled, per_pair = [], []
    for (t, c), g in C.groupby(["task", "concept"]):
        v = g.Rperm_null.dropna().to_numpy()
        resampled.append(max(abs(rng.choice(v, len(v)).mean()) for _ in range(DRAWS)))
        per_pair.append(dict(task=t, concept=c, n=len(v), panel_mean=float(v.mean()), resampled_absmax=resampled[-1]))
    pd.DataFrame(per_pair).to_csv(ROOT / "erasure" / f"repair_permuted_null_pairs{sfx}.csv", index=False)
    out = dict(cells=len(C), pairs=int(C.groupby(["task", "concept"]).ngroups),
               pair_mean_absmax=float(pair_mean.abs().max()),
               pair_mean_max=float(pair_mean.max()),
               pair_absmax_pair=" / ".join(pair_mean.abs().idxmax()),
               cell_sd=float(C.Rperm_null.std()),
               cell_abs_q99=float(C.Rperm_null.abs().quantile(0.99)),
               resampled_panel_mean_absmax=float(max(resampled)))
    pd.DataFrame([out]).to_csv(ROOT / "erasure" / f"repair_permuted_null{sfx}.csv", index=False)
    for k, v in out.items():
        print(f"  {k:30s} {v}")


if __name__ == "__main__":
    main()
