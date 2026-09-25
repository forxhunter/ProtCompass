"""Shuffled-3Di control, extraction (PRESPEC_shuf.md): SaProt with its real 3Di string permuted within each protein.

Same pattern as ab_extract.py, which is not imported or edited: the spec `saprot_shuf3di` is registered at run time
(common.py is not edited), input-audit rows go to ab_modality/encoder_input_audit_ab_shuf.csv, embeddings to
ab_modality/embeddings/<task>/saprot_shuf3di and ab_modality/probing_embeddings/saprot_shuf3di, stage times to
ab_modality/stage_times_shuf.csv. The extraction itself is extract.py (run_downstream, run_probing) unchanged.

The encoder object is a subclass of common.Encoder whose only additions are (1) setting the protein id on the patched
module before every call, resolved from the structure path (downstream: the manifest's structure_path -> id of the
task's stage-1 rows; probing: <id>.pdb in probing_data/structures; input audit: the unperturbed protein of the audit
pair, key 'audit:' + its sequence, reused for its jitter / swap / shuffle / rigid variants), and (2) a back-off on
CUDA out-of-memory (free the cache, wait, retry the same protein).

  python ab_shuf_extract.py downstream --tasks remote_homology ec_classification ...
  python ab_shuf_extract.py probing
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path((PROTCOMPASS_ROOT + "/protcompass/extract"))
AB = HERE / "ab_modality"
sys.path.insert(0, str(HERE))

import pandas as pd                                                      # noqa: E402
import torch                                                             # noqa: E402

import common                                                            # noqa: E402
from common import SPEC, SPECS, Spec                                     # noqa: E402

KEY = "saprot_shuf3di"
NEW_SPEC = Spec(KEY, "patched", "saprot_shuf3di", "pretrained", True, True,
                "final hidden states at the L residue positions of AA+3Di tokens, 3Di string permuted within the protein "
                "(seed from sha256 of the protein id)",
                note="shuffled-3Di control of saprot / saprot_seqonly (same checkpoint, layer, pooling)")
assert KEY not in SPEC
SPEC[KEY] = NEW_SPEC
SPECS.append(NEW_SPEC)

import input_audit                                                       # noqa: E402

input_audit.AUDIT_CSV = AB / "encoder_input_audit_ab_shuf.csv"          # never the released or the ab table
import extract                                                           # noqa: E402

AUDIT_TAGS = {"base", "jitter", "swap", "shuffle", "rigid"}


class ShufEncoder(common.Encoder):
    path2id = {}
    probing_ids = None
    audit_key = None

    def resolve(self, sequence, pdb_path):
        p = Path(pdb_path) if pdb_path else None
        if p is not None and str(p) in self.path2id:
            return self.path2id[str(p)]
        if p is not None and p.parent.name.startswith("audit_") and p.stem in AUDIT_TAGS:
            if p.stem == "base":
                self.audit_key = "audit:" + sequence
            if self.audit_key is None:
                raise RuntimeError("audit variant before its base protein")
            return self.audit_key
        if p is not None and self.probing_ids is not None and p.parent.name == "structures" and p.stem in self.probing_ids:
            return p.stem
        raise RuntimeError(f"saprot_shuf3di: no protein id for {pdb_path}")

    def residue(self, sequence, coords=None, pdb_path=None):
        self.impl.protein_id = self.resolve(sequence, pdb_path)
        try:
            for attempt in range(60):
                try:
                    return super().residue(sequence, coords=coords, pdb_path=pdb_path)
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"  CUDA OOM on {self.impl.protein_id} (L={len(sequence)}), attempt {attempt + 1}; waiting 30 s",
                          flush=True)
                    time.sleep(30)
            raise RuntimeError("CUDA out of memory 60 times in a row")
        finally:
            self.impl.protein_id = None


_CACHE = {}


def cached_encoder(key, device="cuda"):
    assert key == KEY, key
    if key not in _CACHE:
        _CACHE[key] = ShufEncoder(key, device)
    return _CACHE[key]


extract.Encoder = cached_encoder                                         # load the model once per process
TIMES = AB / "stage_times_shuf.csv"


def stamp(stage, what, t0, t1, extra=""):
    new = not TIMES.exists()
    with open(TIMES, "a") as fh:
        if new:
            fh.write("stage,what,start,end,seconds,note\n")
        fh.write(f"{stage},{what},{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t0))},"
                 f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t1))},{t1 - t0:.0f},{extra}\n")


def stage1_manifest(task):
    man = pd.read_csv(HERE / "manifest" / f"{task}.csv")
    return man[man.stage1 == 1].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["audit", "downstream", "probing"])
    ap.add_argument("--tasks", nargs="+", default=[])
    ap.add_argument("--dataset", default="stage1", choices=["stage1", "probing"])
    a = ap.parse_args()
    print(f"pid {os.getpid()} start {time.strftime('%F %T')}", flush=True)
    if a.mode == "audit":
        t0 = time.time()
        enc = cached_encoder(KEY)
        row, _ = input_audit.run_audit(KEY, a.dataset, enc=enc)
        stamp("audit", f"{KEY}/{a.dataset}", t0, time.time())
        print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        print("shuffle stats", enc.impl.stats, flush=True)
    elif a.mode == "downstream":
        for task in a.tasks:
            t0 = time.time()
            man = stage1_manifest(task)
            assert man.structure_path.is_unique and man.id.is_unique, task
            ShufEncoder.path2id = dict(zip(man.structure_path.astype(str), man.id.astype(str)))
            if _CACHE:
                _CACHE[KEY].impl.stats = {"calls": 0, "positions": 0, "same_letter_after_shuffle": 0}
            extract.run_downstream(KEY, task, full=False, limit=0, out_root=AB / "embeddings")
            st = _CACHE[KEY].impl.stats if _CACHE else {}
            frac = st.get("same_letter_after_shuffle", 0) / max(st.get("positions", 0), 1)
            stamp("extract_downstream", f"{KEY}/{task}", t0, time.time(), f"same-letter fraction {frac:.4f}")
            print(f"[{task}] stats {st} same-letter fraction {frac:.4f}", flush=True)
    else:
        sys.path.insert(0, str(HERE.parent / "probing_data"))
        from probing_dataset import ProbingData
        ShufEncoder.path2id = {}
        ShufEncoder.probing_ids = set(ProbingData().ids)
        t0 = time.time()
        extract.run_probing(KEY, "last", 0, labels=None, out_root=AB / "probing_embeddings")
        stamp("extract_probing", KEY, t0, time.time())
    print(f"end {time.strftime('%F %T')}", flush=True)


if __name__ == "__main__":
    main()
