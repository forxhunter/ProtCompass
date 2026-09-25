"""Modality ablation, probing: scripts/symmetric_run.py unchanged, with its output directory redirected.

Set PROBING_EMB=<ab_modality/probing_embeddings> before starting (run_probing_suite reads it at import). The released
symmetric/results_superfamily/ is never written: OUT is replaced before main() runs, and run_source reads the module
global.

  PROBING_EMB=.../ab_modality/probing_embeddings SYM_SPLIT=superfamily python ab_symmetric_run.py esm3_seqonly saprot_seqonly
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
    assert s in ("esm3_seqonly", "saprot_seqonly"), s

if __name__ == "__main__":
    t0 = time.time()
    sr.main()
    with open(AB / "stage_times.csv", "a") as fh:
        fh.write(f"probing_symmetric_run,{'+'.join(sys.argv[1:])},{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t0))},"
                 f"{time.strftime('%Y-%m-%d %H:%M:%S')},{time.time() - t0:.0f}\n")
