"""For every pretrained registry encoder: how many of the model's tensors are actually present in its checkpoint file?

The wrappers load with strict=False, which reports nothing when no tensor matches. Here each model tensor is looked up by
value in the checkpoint (any nesting level), after one forward call so that lazily built networks exist.
Output: audit/checkpoint_verification.csv
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import HERE, NO_CHECKPOINT, SPECS, Encoder, file_sha256
from input_audit import audit_proteins


def tensor_hashes(obj, out):
    if torch.is_tensor(obj):
        out.add(hashlib.sha256(obj.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest())
    elif isinstance(obj, dict):
        for v in obj.values():
            tensor_hashes(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            tensor_hashes(v, out)
    return out


if __name__ == "__main__":
    seq, xyz = audit_proteins("probing")[0]
    rows = []
    for s in SPECS:
        if s.family != "registry" or s.weights != "pretrained" or s.env != "protein_embeddings":
            continue
        try:
            enc = Encoder(s.key); enc.residue(seq, coords=xyz)
            impl = enc.impl
            path = s.checkpoint or getattr(impl, "_ckpt_path", None) or (impl._default_ckpt() if hasattr(impl, "_default_ckpt") else None)
            sd = {k: v for k, v in impl.state_dict().items() if torch.is_tensor(v) and v.is_floating_point() and v.numel() > 1}
            row = {"encoder": s.key, "checkpoint": str(path), "model_tensors": len(sd)}
            if path and Path(path).exists():
                have = tensor_hashes(torch.load(path, map_location="cpu", weights_only=False), set())
                hit = sum(hashlib.sha256(v.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest() in have for v in sd.values())
                row.update(checkpoint_sha256=file_sha256(path), model_tensors_found_in_checkpoint=hit,
                           verdict="loaded" if hit >= 0.9 * len(sd) else "NOT loaded")
            else:
                row.update(verdict="checkpoint path unknown to the harness (weights come from a hub download inside the wrapper)")
            row["loaded_by"] = enc.meta.get("checkpoint_loaded_by", "wrapper")
        except Exception as e:
            row = {"encoder": s.key, "verdict": f"error: {type(e).__name__}: {str(e)[:200]}"}
        rows.append(row); print(json.dumps(row), flush=True)
    for k, why in NO_CHECKPOINT.items():
        rows.append({"encoder": k, "verdict": "excluded: " + why})
    (HERE / "audit").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(HERE / "audit" / "checkpoint_verification.csv", index=False)
