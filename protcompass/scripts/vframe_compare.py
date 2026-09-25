"""The equivariant encoders re-extracted with the direction of every vector channel kept (probing_embeddings_vframe/,
extract/extract_vector_frame.py), held to the input-matched floors of the released encoder under the protocol of
scripts/symmetric_decide.py, against the released (norm) extraction.

  SYM_SPLIT=superfamily python vframe_compare.py
Output: symmetric/vframe_<split>.csv (one row per encoder and probe, both extractions)
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
sys.path.insert(0, str(ROOT / "scripts"))
import symmetric_decide as sd                                            # noqa: E402
from floor_rules import admissibility                                    # noqa: E402

BASE = {"gvp_vframe": "gvp", "gcpnet_random_vframe": "gcpnet_random", "gcpnet_pretrained_vframe": "gcpnet_pretrained",
        "mace_vframe": "mace"}


def main():
    S = sd.load_scores()
    val = S.set_index(["source", "probe", "family"])["val"]
    test = S.set_index(["source", "probe", "family"])["test"]
    M = admissibility()
    B = sd.Boot(2000)
    C = pd.read_csv(sd.SYM / f"cells_{sd.SPLIT}.csv")
    C = C[C.rule == "admissible"].set_index(["encoder", "probe"])
    rows = []
    for v, e in BASE.items():
        if not (sd.RES / f"{v}.csv").exists():
            continue
        for p in sd.DETERMINED + sd.UNDETERMINED:
            fams = [f for f in ("linear", "nonlinear") if (v, p, f) in val.index]
            if not fams or (e, p) not in C.index:
                continue
            fam = max(fams, key=lambda f: val[(v, p, f)])
            c = C.loc[(e, p)]
            g = B.gap(p, v, fam, c.floor, c.floor_family)
            lo, hi = np.percentile(g, [2.5, 97.5])
            rows.append(dict(split=sd.SPLIT, encoder=e, probe=p, determined=p in sd.DETERMINED,
                             floor=c.floor, gap_norm=c.gap, state_norm=c.state,
                             score_norm=c.enc_test, score_frame=test[(v, p, fam)],
                             gap_frame=test[(v, p, fam)] - c.floor_test, lo_frame=lo, hi_frame=hi,
                             state_frame="below" if hi < 0 else "above" if lo > 0 else "unresolved"))
    D = pd.DataFrame(rows)
    D.to_csv(sd.SYM / f"vframe_{sd.SPLIT}.csv", index=False)
    pd.set_option("display.width", 200)
    d = D[D.determined]
    print("determined cells:", len(d), "below norm", int((d.state_norm == "below").sum()),
          "below frame", int((d.state_frame == "below").sum()), "changed", int((d.state_norm != d.state_frame).sum()))
    print("score change (frame - norm) on determined labels, median by encoder:")
    print(d.assign(delta=d.score_frame - d.score_norm).groupby("encoder").delta.agg(["median", "max"]).round(4))
    print(D[D.state_norm != D.state_frame][["encoder", "probe", "gap_norm", "gap_frame", "state_norm", "state_frame"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
