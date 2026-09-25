"""Pre-registered labels aligned with the rows of confirmation2/merged/labels.h5 (released train and validation
draw, then the cohort B domains), built with the same row gathering as scripts/confirmation_merge.py.

  CONFIRMATION_DIR=confirmation2 python prereg_merge_cohortB.py
"""
import h5py
import numpy as np

from confirmation_merge import CONF, PD, gather, released_fit_draw

ROOT = PD.parent
rel, coh = h5py.File(PD / "labels.h5", "r"), h5py.File(CONF / "labels_v44.h5", "r")
tr, va = released_fit_draw(rel)
g_rel, g_coh = gather(rel, np.concatenate([tr, va])), gather(coh, np.arange(len(coh["protein"]["id"])))
merged = h5py.File(CONF / "merged" / "labels.h5", "r")
n = len(merged["residue"]["protein_index"])
pr, pc = h5py.File(ROOT / "prereg" / "labels_released.h5", "r"), h5py.File(ROOT / "prereg" / "labels_cohortB.h5", "r")
with h5py.File(ROOT / "prereg" / "labels_cohortB_merged.h5", "w") as out:
    for k in pr["residue"]:
        v = np.concatenate([pr["residue"][k][:][g_rel["rows"]], pc["residue"][k][:][g_coh["rows"]]])
        assert len(v) == n, (k, len(v), n)
        out[f"residue/{k}"] = v
    # alignment check: the stored coordinates of the merged file are the source rows, row for row
    c = np.concatenate([rel["residue"]["coords"][:][g_rel["rows"]][:5000], coh["residue"]["coords"][:][g_coh["rows"]][:5000]])
    m = np.concatenate([merged["residue"]["coords"][:5000], merged["residue"]["coords"][len(g_rel["rows"]):len(g_rel["rows"]) + 5000]])
    assert np.array_equal(c, m), "row alignment with the merged file failed"
    out.attrs["merged_labels"] = str(CONF / "merged" / "labels.h5")
print("wrote prereg/labels_cohortB_merged.h5", n, "rows")
