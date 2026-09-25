"""Map the protein index of remotely scored prediction files back to the released labels file.

The remote bundle (prereg_export_subset.py) holds the 4,500 drawn domains in their original order, so bundle protein k
is released protein idx[k], where idx concatenates the train, validation and test draws. Only the protein-index arrays
are rewritten; predictions and labels are untouched, and after the rewrite every label's protein array and label array
must equal those of a locally scored source. The unmapped files are kept in prereg/results/remote_raw/.

  python prereg_relabel_remote.py
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
sys.path.insert(0, str(ROOT / "scripts"))
import run_probing_suite as rps                                          # noqa: E402

data = rps.Data()
idx = np.concatenate([data.chosen[p] for p in rps.PARTS])
res, raw = ROOT / "prereg" / "results" / "by_source", ROOT / "prereg" / "results" / "remote_raw"
raw.mkdir(exist_ok=True)
ref = dict(np.load(res / "ref_length__test_predictions.npz"))
for e in (ROOT / "prereg" / "sources_remote.txt").read_text().split():
    f = res / f"{e}__test_predictions.npz"
    if not (raw / f.name).exists():
        shutil.copy(f, raw / f.name)
    d = dict(np.load(raw / f.name))
    for k in [k for k in d if k.endswith("|protein")]:
        d[k] = idx[d[k]]
        lab = k.split("|")[0]
        assert np.array_equal(d[k], ref[f"{lab}|protein"]), (e, lab)
        assert np.array_equal(d[f"{lab}|y"], ref[f"{lab}|y"]), (e, lab)
    np.savez_compressed(f, **d)
    print("relabelled", e)
