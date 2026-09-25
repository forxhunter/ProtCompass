"""Modality ablation, extraction: ESM3 and SaProt with the same weights and sequence input only.

This wrapper registers two extra encoder specs at run time (common.py is not edited), sends the input-audit rows to
ab_modality/encoder_input_audit_ab.csv instead of the released extract/encoder_input_audit.csv, and writes embeddings
under ab_modality/ only. The extraction itself is extract.py (run_downstream, run_probing) unchanged, so the manifests,
stage-1 rows, pooling and the audit gate are those of the released esm3 and saprot runs.

  python ab_extract.py audit    --encoders esm3_seqonly saprot_seqonly --dataset stage1
  python ab_extract.py downstream --encoders esm3_seqonly --tasks stability fluorescence ...
  python ab_extract.py probing  --encoders esm3_seqonly
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path((PROTCOMPASS_ROOT + "/protcompass/extract"))
AB = HERE / "ab_modality"
sys.path.insert(0, str(HERE))

import common                                                            # noqa: E402
from common import SPEC, SPECS, Spec                                     # noqa: E402

NEW_SPECS = [
    Spec("esm3_seqonly", "patched", "esm3_seqonly", "pretrained", False, True,
         "final trunk residue embeddings, sequence track only; structure track masked, no coordinates",
         note="modality ablation of esm3 (same checkpoint, dtype, layer, pooling)"),
    Spec("saprot_seqonly", "patched", "saprot_seqonly", "pretrained", False, True,
         "final hidden states at the L residue positions of AA+'#' tokens (3Di masked)",
         note="modality ablation of saprot (same checkpoint, layer, pooling)"),
]
for s in NEW_SPECS:
    assert s.key not in SPEC
    SPEC[s.key] = s
    SPECS.append(s)

import input_audit                                                       # noqa: E402

input_audit.AUDIT_CSV = AB / "encoder_input_audit_ab.csv"               # never the released table
import extract                                                           # noqa: E402

_CACHE = {}


def cached_encoder(key, device="cuda"):
    if key not in _CACHE:
        _CACHE[key] = common.Encoder(key, device)
    return _CACHE[key]


extract.Encoder = cached_encoder                                         # load each model once per process
TIMES = AB / "stage_times.csv"


def stamp(stage, what, t0, t1):
    new = not TIMES.exists()
    with open(TIMES, "a") as fh:
        if new:
            fh.write("stage,what,start,end,seconds\n")
        fh.write(f"{stage},{what},{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t0))},"
                 f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t1))},{t1 - t0:.0f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["audit", "downstream", "probing"])
    ap.add_argument("--encoders", nargs="+", required=True)
    ap.add_argument("--tasks", nargs="+", default=[])
    ap.add_argument("--dataset", default="stage1", choices=["stage1", "probing"])
    a = ap.parse_args()
    for key in a.encoders:
        assert key in {s.key for s in NEW_SPECS}, key
        if a.mode == "audit":
            t0 = time.time()
            row, _ = input_audit.run_audit(key, a.dataset, enc=cached_encoder(key))
            stamp("audit", f"{key}/{a.dataset}", t0, time.time())
            print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        elif a.mode == "downstream":
            for task in a.tasks:
                t0 = time.time()
                extract.run_downstream(key, task, full=False, limit=0, out_root=AB / "embeddings")
                stamp("extract_downstream", f"{key}/{task}", t0, time.time())
        else:
            t0 = time.time()
            extract.run_probing(key, "last", 0, labels=None, out_root=AB / "probing_embeddings")
            stamp("extract_probing", key, t0, time.time())


if __name__ == "__main__":
    main()
