"""Trace the GVP and GCPNet extraction wrappers: why does each emit 116 dimensions?

Both workers assemble their output as cat([node_s (B,L,s)], [node_v.reshape(B,L,v*3)]), which should give
s + 3v = 148 for GVP (s=100, v=16) and 176 for GCPNet (s=128, v=16). Every released array is 116 wide. This script
instantiates each worker's module on a synthetic protein and prints the shapes at the point of assembly, so the
mismatch is measured rather than inferred.
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import sys
import numpy as np
import torch

sys.path.insert(0, (PROTCOMPASS_ROOT + "/protcompass"))

def synth(B=1, L=24):
    coords = torch.zeros(B, L, 5, 3)
    for i in range(L):
        coords[0, i, :, 0] = i * 3.8
        coords[0, i, 1, 0] = i * 3.8
        coords[0, i, :, 1] = torch.tensor([0.0, 0.5, 1.0, 1.5, 0.8])
    seq = torch.randint(0, 20, (B, L))
    mask = torch.ones(B, L, dtype=torch.bool)
    return coords, seq, mask


def trace(name, module_path, cls_name, kwargs):
    import importlib
    m = importlib.import_module(module_path)
    Cls = getattr(m, cls_name)
    net = Cls(**kwargs).eval()
    declared = kwargs.get("node_s_dim", 0) + kwargs.get("node_v_dim", 0) * 3
    coords, seq, mask = synth()
    with torch.no_grad():
        emb, glob = net(coords, seq, mask)
    print(f"{name}:")
    print(f"  declared node_s_dim + 3*node_v_dim = {kwargs.get('node_s_dim')} + 3*{kwargs.get('node_v_dim')} "
          f"= {declared}")
    print(f"  self._embedding_dim attribute      = {getattr(net, '_embedding_dim', 'n/a')}")
    print(f"  emitted residue embedding shape    = {tuple(emb.shape)}  -> width {emb.shape[-1]}")
    print(f"  emitted global embedding shape     = {tuple(glob.shape)}")
    return int(emb.shape[-1])


if __name__ == "__main__":
    out = {}
    for name, mod, cls, kw in (
        ("GVP", "src.models.class_c.gvp_worker", None, dict(node_s_dim=100, node_v_dim=16)),
        ("GCPNet", "src.models.class_c.gcpnet_worker", None, dict(node_s_dim=128, node_v_dim=16)),
    ):
        import importlib
        m = importlib.import_module(mod)
        cands = [c for c in dir(m) if c[0].isupper() and "Encoder" in c or c.endswith("Net") or c.endswith("GVP")]
        print(f"--- {name}: candidate classes in {mod}: {cands}")
        if not cands:
            continue
        try:
            out[name] = trace(name, mod, cands[0], kw)
        except Exception as e:
            print(f"  [{name}] {type(e).__name__}: {e}")
    print("\nmeasured widths:", out)
