"""Shuffled-3Di control, probing (PRESPEC_shuf.md step 3): scripts/symmetric_run.py unchanged, output redirected.

Copy of ab_symmetric_run.py (not edited) that admits the source saprot_shuf3di only and writes its stage time to
stage_times_shuf.csv. Output: ab_modality/symmetric/results_superfamily/by_source/saprot_shuf3di{.csv,__test_predictions.npz}.

  PROBING_EMB=.../ab_modality/probing_embeddings SYM_SPLIT=superfamily python ab_shuf_symmetric_run.py saprot_shuf3di
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))
import os
import sys
import time
from pathlib import Path

AB = Path((PROTCOMPASS_ROOT + "/protcompass/extract/ab_modality"))
assert os.environ.get("PROBING_EMB") == str(AB / "probing_embeddings"), "set PROBING_EMB to ab_modality/probing_embeddings"
assert os.environ.get("SYM_SPLIT", "superfamily") == "superfamily"
sys.path.insert(0, (PROTCOMPASS_ROOT + "/protcompass/scripts"))
import symmetric_run as sr                                               # noqa: E402

assert str(sr.rps.EMB) == str(AB / "probing_embeddings")
sr.OUT = AB / "symmetric" / f"results_{sr.SPLIT}"
for s in [a for a in sys.argv[1:] if not a.startswith("--")]:
    assert s == "saprot_shuf3di", s

if __name__ == "__main__":
    t0 = time.time()
    sr.main()
    with open(AB / "stage_times_shuf.csv", "a") as fh:
        fh.write(f"probing_symmetric_run,{'+'.join(sys.argv[1:])},{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t0))},"
                 f"{time.strftime('%Y-%m-%d %H:%M:%S')},{time.time() - t0:.0f},\n")
