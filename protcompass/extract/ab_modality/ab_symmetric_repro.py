"""Reproduction check (not a contrast): rerun the released esm3 and saprot through scripts/symmetric_run.py today,
into ab_modality/symmetric_repro/, and compare with the released by_source files. The contrasts use the released files;
this measures how much of a probing delta could come from refitting the readouts on a different day.

  PROBING_EMB unset (released probing_embeddings); python ab_symmetric_repro.py esm3 saprot
  python ab_symmetric_repro.py --compare
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
AB = ROOT / "extract" / "ab_modality"
OUT = AB / "symmetric_repro" / "results_superfamily"
REL = ROOT / "symmetric" / "results_superfamily" / "by_source"

if "--compare" in sys.argv:
    rows = []
    for s in ("esm3", "saprot"):
        a = pd.read_csv(REL / f"{s}.csv"); b = pd.read_csv(OUT / "by_source" / f"{s}.csv")
        m = a.merge(b, on=["source", "probe", "readout", "seed", "split"], suffixes=("_rel", "_rep"))
        m["absdiff"] = (m.headline_rel - m.headline_rep).abs()
        rows.append(m[["source", "probe", "readout", "seed", "split", "headline_rel", "headline_rep", "absdiff"]])
    d = pd.concat(rows)
    d.to_csv(AB / "symmetric_repro_check.csv", index=False)
    print(d.groupby(["source", "readout"]).absdiff.agg(["max", "median"]).round(5).to_string())
    sys.exit()

assert not os.environ.get("PROBING_EMB"), "the released embeddings are read"
sys.path.insert(0, str(ROOT / "scripts"))
import symmetric_run as sr                                               # noqa: E402

sr.OUT = OUT
for s in [a for a in sys.argv[1:] if not a.startswith("--")]:
    assert s in ("esm3", "saprot"), s
if __name__ == "__main__":
    sr.main()
