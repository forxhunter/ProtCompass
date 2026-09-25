"""Extract a full-atom encoder (GCPNet) on cohort B for the pre-registered test.

The released ProbingData points every domain at probing_data/structures/<id>.pdb. Cohort B domains live in
confirmation2/structures, so this wrapper points the domain-file path there and otherwise calls the released
extract.run_probing unchanged, writing to confirmation2/embeddings like the other cohort encoders.

  python prereg_extract_cohortB.py gcpnet_pretrained gcpnet_random
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import sys
from pathlib import Path

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
C = ROOT / "." / "confirmation2"
sys.path.insert(0, str(ROOT / "." / "probing_data"))
import probing_dataset                                                    # noqa: E402

_protein = probing_dataset.ProbingData.protein


def protein(self, i):
    p = _protein(self, i)
    q = C / "structures" / Path(p["pdb_path"]).name
    if not Path(p["pdb_path"]).exists() and q.exists():     # cohort domains; the released audit proteins keep their path
        p["pdb_path"] = str(q)
    assert Path(p["pdb_path"]).exists(), p["pdb_path"]
    return p


probing_dataset.ProbingData.protein = protein

import extract                                                            # noqa: E402

for key in sys.argv[1:]:
    extract.run_probing(key, "last", 0, str(C / "labels_v44.h5"), str(C / "embeddings"))
