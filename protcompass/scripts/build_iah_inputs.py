"""Assemble the clean matrices the IAH test consumes.

Writes three wide CSVs indexed by encoder:
  iah_probing.csv     IIP profile, as the gain over the best reference feature set
  iah_downstream.csv  downstream score per task, re-extracted embeddings only; a task enters only when its own
                      re-extraction is finished, so no profile is ever joined to a score from the old pipeline
  iah_meta.csv        descriptors that need no downstream run: modality, weights status, parameter count,
                      and the measured inputs from the encoder input audit

`iah_eval.py --probing ... --downstream ... --meta ...` then answers the question the manuscript's ESP was meant to
answer, without the leak: does the IIP profile predict transfer beyond what cheap descriptors already say?
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
OUT = ROOT / "audit"
GROUP = json.loads((ROOT / "scripts" / "encoder_groups.json").read_text()) if (ROOT / "scripts" / "encoder_groups.json").exists() else {}


def probing_matrix():
    P = pd.read_csv(ROOT / "probing_results" / "iip_profile_gain.csv", index_col=0)
    return P.dropna(axis=1, how="all")


def downstream_matrix():
    """Re-extracted scores only. Tasks still in the queue are absent rather than filled from the old cache."""
    frames = []
    new = ROOT / "extract" / "scores_wide.csv"
    if new.exists():
        frames.append(pd.read_csv(new, index_col=0))
    # The old cache is the pipeline in which structure encoders ran on all-zero coordinates, and in which SaProt had
    # constant 3Di tokens. Joining those scores to a rebuilt probing profile pairs two different versions of the same
    # encoder, so a task is used only once every encoder in it has been re-extracted.
    if not frames:
        raise SystemExit("no re-extracted downstream scores yet")
    D = pd.concat(frames, axis=1)
    return D.loc[:, ~D.columns.duplicated()]


def meta_table(index):
    import sys
    sys.path.insert(0, str(ROOT / "extract"))
    from common import SPEC
    audit = pd.read_csv(ROOT / "extract" / "encoder_input_audit.csv")
    audit = audit[audit.dataset == "probing"].drop_duplicates("encoder").set_index("encoder")
    rows = []
    for k in index:
        s = SPEC.get(k)
        a = audit.loc[k] if k in audit.index else None
        params = None
        mj = ROOT / "probing_embeddings" / k / "meta.json"
        if mj.exists():
            params = json.loads(mj.read_text()).get("n_parameters")
        rows.append({
            "encoder": k,
            "modality": ("sequence" if s and not s.uses_coords else "structure") if s else "unknown",
            "pretrained": 1.0 if (s and s.weights == "pretrained") else 0.0,
            "log10_params": np.log10(params) if params else np.nan,
            "uses_coordinates": float(a["uses_coordinates"]) if a is not None else np.nan,
            "uses_residue_identity": float(a["uses_residue_identity"]) if a is not None else np.nan,
            "rel_change_coord_swap": float(a["rel_change_coord_swap"]) if a is not None else np.nan,
        })
    M = pd.DataFrame(rows).set_index("encoder")
    # Measured modality: what the encoder reads, not what the manuscript declares.
    M["modality"] = np.where(M.uses_coordinates.fillna(0) > 0,
                             np.where(M.uses_residue_identity.fillna(0) > 0, "both", "structure"), "sequence")
    M["log10_params"] = M["log10_params"].fillna(M["log10_params"].median())
    return M


def main():
    OUT.mkdir(exist_ok=True)
    P, D = probing_matrix(), downstream_matrix()
    keep = [e for e in P.index if e in D.index]
    if not keep:
        print("probing encoders:", list(P.index)); print("downstream encoders:", list(D.index))
        raise SystemExit("no encoder has both a probing profile and downstream scores yet")
    P, D = P.loc[keep], D.loc[keep]
    M = meta_table(keep)
    P.to_csv(OUT / "iah_probing.csv"); D.to_csv(OUT / "iah_downstream.csv"); M.to_csv(OUT / "iah_meta.csv")
    print(f"{len(keep)} encoders with both a profile and downstream scores")
    print(f"probing: {P.shape[1]} IIPs | downstream: {D.shape[1]} tasks, {int(D.notna().sum().sum())} scored cells")
    print("\nscored cells per task:"); print(D.notna().sum().to_string())
    print("\nmeasured modality:"); print(M.modality.value_counts().to_string())
    print(f"\nwritten to {OUT}/iah_{{probing,downstream,meta}}.csv")
    print(f"next: python scripts/iah_eval.py --probing {OUT}/iah_probing.csv --downstream {OUT}/iah_downstream.csv --meta {OUT}/iah_meta.csv")


if __name__ == "__main__":
    main()
