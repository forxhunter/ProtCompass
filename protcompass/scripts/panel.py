"""The released encoder panel, from the scored downstream table, as the one definition every summary filters on.

Directory listings of the embedding cache also contain `ref_composition`, the non-learned composition and length
reference, which has an embedding file and a meta.json like an encoder. It is a reference, not an encoder, and a
summary that lists encoders by directory counts it as a 34th encoder and a 22nd architecture family unless it is
filtered out here.
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
from pathlib import Path

import pandas as pd

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
REFERENCES = {"ref_composition"}


def panel():
    enc = set(pd.read_csv(ROOT / "extract" / "scores2_selected.csv").encoder.unique())
    return sorted(enc - REFERENCES)


def only_panel(d, col="encoder"):
    """Rows of d whose encoder is in the released panel."""
    return d[d[col].isin(set(panel()))]


if __name__ == "__main__":
    p = panel()
    print(len(p), "encoders:", ", ".join(p))
