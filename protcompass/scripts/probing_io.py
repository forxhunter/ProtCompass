"""Load probing result files without importing the full encoder package."""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import glob
import sys
import types
from pathlib import Path

import pandas as pd

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
PROBING_DIR = ROOT / "ProtCompass_Embeddings/probing_IF/probing_results_architecture_full"


class _Mock:
    def __getattr__(self, n):
        return _Mock()

    def __call__(self, *a, **k):
        return _Mock()


class InvariantFingerprint:
    pass


def _install_stubs():
    for m in ["torch_geometric", "torch_geometric.nn"]:
        sys.modules.setdefault(m, _Mock())
    src, probes, battery = (types.ModuleType(n) for n in ("src", "src.probes", "src.probes.probe_battery"))
    battery.InvariantFingerprint = InvariantFingerprint
    sys.modules.update({"src": src, "src.probes": probes, "src.probes.probe_battery": battery})


def load_probing_long(probing_dir=PROBING_DIR):
    """Return one row per (encoder, seed, probe task) plus the stored family scores."""
    _install_stubs()
    import torch

    task_rows, family_rows = [], []
    for f in sorted(glob.glob(str(probing_dir / "*.pt"))):
        d = torch.load(f, map_location="cpu", weights_only=False)
        fp = d["fingerprint"]
        enc, seed = d["encoder"], int(d["seed"])
        for i in range(1, 11):
            family_rows.append({"encoder": enc, "seed": seed, "family": f"I{i}", "score": float(getattr(fp, f"p_I{i}"))})
        for task, val in fp.detailed_scores.items():
            row = {"encoder": enc, "seed": seed, "probe_task": task}
            if isinstance(val, dict):
                row.update({k: v for k, v in val.items() if isinstance(v, (int, float, str))})
            else:
                row["score"] = float(val)
            task_rows.append(row)
    return pd.DataFrame(task_rows), pd.DataFrame(family_rows)


if __name__ == "__main__":
    tasks, fams = load_probing_long()
    out = ROOT / "./audit"
    tasks.to_csv(out / "probing_task_scores_long.csv", index=False)
    fams.to_csv(out / "probing_family_scores_long.csv", index=False)
    print(tasks.head(20).to_string())
    print("\ncolumns:", list(tasks.columns))
    print("probe tasks:", sorted(tasks.probe_task.unique()))
