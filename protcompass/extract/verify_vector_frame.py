"""Post-hoc check of the E5 outputs (protcompass/probing_embeddings_vframe) against the released embeddings, from the files.

  python verify_vector_frame.py          # any env with numpy

Checks, over all residue rows: (a) new scalar columns == released scalar columns (fp16 files); (b) per-channel norms of the
new vector block == released norm columns (GVP; for GCPNet the released column is a mean over heavy atoms of |chi_a| and the
new block holds the CA atom's vectors, so only a correlation is reported); pooled_fp32 == mean of the residue rows; the root
offsets/lengths/ids equal the released ones. Prints one JSON line per encoder.
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import json
from pathlib import Path

import numpy as np

R = Path((PROTCOMPASS_ROOT + "/protcompass"))
REL, NEW = R / "probing_embeddings", R / "probing_embeddings_vframe"


def main():
    for f in ("offsets.npy", "lengths.npy"):
        assert np.array_equal(np.load(REL / f), np.load(NEW / f)), f
    assert (REL / "ids.txt").read_text() == (NEW / "ids.txt").read_text()
    off, ln = np.load(NEW / "offsets.npy"), np.load(NEW / "lengths.npy")
    for d in sorted(NEW.glob("*_vframe")):
        if not (d / "meta.json").exists():
            continue
        m = json.loads((d / "meta.json").read_text())
        key, S, V = m["source_encoder"], m["s"], m["v"]
        new = np.load(d / "residue_fp16.npy", mmap_mode="r")
        rel = np.load(REL / key / "residue_fp16.npy", mmap_mode="r")
        out = {"encoder": d.name, "rows": int(new.shape[0]), "dim": int(new.shape[1])}
        assert new.shape == (int(ln.sum()), S + 3 * V) and new.shape[0] == rel.shape[0]
        amax, nmis, bmax, bsum, pmax = 0.0, 0, 0.0, [], 0.0
        pooled = np.load(d / "pooled_fp32.npy")
        for a in range(0, new.shape[0], 500_000):
            b = min(a + 500_000, new.shape[0])
            x, y = np.asarray(new[a:b], dtype=np.float32), np.asarray(rel[a:b], dtype=np.float32)
            dd = np.abs(x[:, :S] - y[:, :S])
            amax, nmis = max(amax, float(dd.max())), nmis + int((dd > 0).sum())
            nrm = np.linalg.norm(x[:, S:].astype(np.float64).reshape(-1, V, 3), axis=-1)
            if y.shape[1] == S + V:
                if key == "gvp":
                    bmax = max(bmax, float(np.abs(nrm - y[:, S:]).max()))
                else:
                    bsum.append(np.corrcoef(nrm.ravel(), y[:, S:].ravel())[0, 1])
        # pooled_fp32 against the mean of the stored fp16 residue rows (fp16 rounding only)
        for k0 in range(0, len(ln), 2000):
            k1 = min(k0 + 2000, len(ln))
            a, b = int(off[k0]), int(off[k1 - 1] + ln[k1 - 1])
            s = np.add.reduceat(np.asarray(new[a:b], dtype=np.float64), (off[k0:k1] - a).astype(np.int64), axis=0)
            pmax = max(pmax, float(np.abs(s / ln[k0:k1, None] - pooled[k0:k1]).max()))
        out.update(a_scalar_maxabs=amax, a_scalar_frac_mismatch=nmis / (new.shape[0] * S),
                   pooled_vs_residue_mean_maxabs=pmax)
        if key == "gvp":
            out["b_norm_maxabs_fp16"] = bmax
        elif bsum:
            out["corr_CA_vector_norms_vs_released_atom_mean_norms"] = float(np.mean(bsum))
        print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
