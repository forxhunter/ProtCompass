"""The two multi-atom floors rebuilt from a given subset of the five stored atoms.

`ref_backbone` and `ref_global` (scripts/run_probing_suite.py) are built from N, CA, C, O and CB. The atom-level
input audit (extract/atom_audit.py) shows that most coordinate encoders read fewer: CA alone, N-CA-C or N-CA-C-O. A
floor built from atoms an encoder never receives is not a floor for that encoder, so this script rebuilds both
descriptors from an atom subset, with nothing else changed:

  window distances   all pairwise distances among the chosen atoms of residues i-2..i+2, centred on CA(i)
  chirality          one sign per window offset, from N, CA, C of that residue when the subset holds all three, and
                     otherwise from CA(j-1), CA(j), CA(j+1) of the chain, which a CA-only encoder does receive
  non-local block    the CA kNN distances and shell counts of the released descriptor (CA only)
  ref_global extra   the released centroid distance, centroid rank and contact-map eigenvectors (CA only)

With all five atoms the rebuilt descriptor equals the released one (asserted on 20 proteins). The protein-level
features of both released floors are CA-only already and are reused as released. Every probe of the released suite is
then run through the released run_source, once scored on the test split and once on the validation split.

  python atom_matched_floors.py CA N+CA+C N+CA+C+O [--val]

Output: probing_results_atom/by_source/ref_backbone@<atoms>.csv (test) or probing_results_atom_val/... (validation).
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import os
import sys
from pathlib import Path

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
VAL = "--val" in sys.argv
# A confirmation cohort sets PROBING_LABELS, PROBING_EMB and PROBING_OUT itself (scripts/confirmation2_run.sh).
os.environ.setdefault("PROBING_OUT", str(ROOT / ("probing_results_atom_val" if VAL else "probing_results_atom")))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np                                                       # noqa: E402

import run_probing_suite as rps                                          # noqa: E402


def run_with_retry(data, source):
    """rps.run_source with the released retry on a shared GPU: wait and try again rather than lose the run."""
    import time
    import torch
    for attempt in range(12):
        try:
            return rps.run_source(data, source)
        except (torch.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower():
                raise
            torch.cuda.empty_cache()
            print(f"[{source}] CUDA out of memory (attempt {attempt + 1}/12); waiting 5 min", flush=True)
            time.sleep(300)
    raise SystemExit(f"[{source}] still out of memory after 12 attempts")

ATOMS = ["N", "CA", "C", "O", "CB"]


class AtomData(rps.Data):
    def backbone_subset(self, prot_idx, atoms, W=2, K=16):
        ai = [ATOMS.index(a) for a in atoms]
        na, ica = len(ai), atoms.index("CA")
        full = {"N", "CA", "C"} <= set(atoms)
        out = []
        for i in prot_idx:
            r = slice(int(self.offs[i]), int(self.offs[i]) + int(self.L[i]))
            c5 = self.h["residue/coords"][r].astype(np.float64)
            c = c5[:, ai]
            n = len(c)
            idx = np.clip(np.arange(n)[:, None] + np.arange(-W, W + 1)[None, :], 0, n - 1)
            win = c[idx].reshape(n, (2 * W + 1) * na, 3)
            win = win - c5[:, 1][:, None, :]
            d = np.linalg.norm(win[:, :, None, :] - win[:, None, :, :], axis=-1)
            iu = np.triu_indices(win.shape[1], 1)
            dist = d[:, iu[0], iu[1]]
            centre = win[:, na * W + ica]
            sign = []
            for k in range(2 * W + 1):
                if full:
                    a, b, e = (win[:, na * k + atoms.index(x)] for x in ("N", "CA", "C"))
                else:
                    ca, j = c5[:, 1], idx[:, k]                                  # the CA trace, centred on CA(i)
                    a = ca[np.clip(j - 1, 0, n - 1)] - ca
                    b = ca[j] - ca
                    e = ca[np.clip(j + 1, 0, n - 1)] - ca
                sign.append(np.sign(np.einsum("ij,ij->i", np.cross(a - b, e - b), centre - b + 1e-9)))
            ca = c5[:, 1]
            dd = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
            np.fill_diagonal(dd, np.inf)
            knn = np.sort(dd, axis=1)[:, :K]
            knn[~np.isfinite(knn)] = 99.0
            shells = np.stack([(dd < s).sum(1) for s in (5.0, 7.0, 9.0, 11.0, 13.0)], 1).astype(np.float64)
            out.append(np.concatenate([dist, np.stack(sign, 1), knn, shells], 1))
        return np.concatenate(out).astype(np.float32)

    def global_block(self, i):
        """The CA-only part ref_global adds to ref_backbone: centroid distance, centroid rank and the leading
        contact-map eigenvectors. It does not depend on the atom set, so it is computed once per protein and cached."""
        if not hasattr(self, "_gblock"):
            self._gblock = dict(np.load(cache_path(), allow_pickle=True)["blocks"].item()) if cache_path().exists() else {}
        if int(i) in self._gblock:
            return self._gblock[int(i)]
        r = slice(int(self.offs[i]), int(self.offs[i]) + int(self.L[i]))
        ca = self.h["residue/coords"][r][:, 1].astype(np.float64)
        c = ca - ca.mean(0)
        dist = np.linalg.norm(c, axis=1, keepdims=True)
        rank = (np.argsort(np.argsort(dist.ravel())) / max(len(c) - 1, 1))[:, None]
        d = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
        A = (d < 8.0).astype(np.float64)
        np.fill_diagonal(A, 0.0)
        k = min(8, len(A) - 1)
        if k > 0:
            vecs = np.linalg.eigh(A)[1][:, -k:]
            spec = np.pad(vecs * np.sign(vecs.sum(0)), ((0, 0), (0, 8 - k)))
        else:
            spec = np.zeros((len(A), 8))
        blk = np.concatenate([dist, rank, spec], 1)
        self._gblock[int(i)] = blk
        return blk

    def global_subset(self, prot_idx, atoms, W=2, K=16):
        return np.concatenate([np.concatenate([self.backbone_subset([i], atoms, W=W, K=K), self.global_block(i)], 1)
                               for i in prot_idx]).astype(np.float32)

    def _global_subset_uncached(self, prot_idx, atoms, W=2, K=16):
        out = []
        for i in prot_idx:
            base = self.backbone_subset([i], atoms, W=W, K=K)
            r = slice(int(self.offs[i]), int(self.offs[i]) + int(self.L[i]))
            ca = self.h["residue/coords"][r][:, 1].astype(np.float64)
            c = ca - ca.mean(0)
            dist = np.linalg.norm(c, axis=1, keepdims=True)
            rank = (np.argsort(np.argsort(dist.ravel())) / max(len(c) - 1, 1))[:, None]
            d = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
            A = (d < 8.0).astype(np.float64)
            np.fill_diagonal(A, 0.0)
            k = min(8, len(A) - 1)
            if k > 0:
                vecs = np.linalg.eigh(A)[1][:, -k:]
                spec = np.pad(vecs * np.sign(vecs.sum(0)), ((0, 0), (0, 8 - k)))
            else:
                spec = np.zeros((len(A), 8))
            out.append(np.concatenate([base, dist, rank, spec], 1))
        return np.concatenate(out).astype(np.float32)

    def residue_features(self, source):
        if "@" in source:
            kind, atoms = source.split("@")
            atoms = atoms.split("+")
            f = self.backbone_subset if kind == "ref_backbone" else self.global_subset
            return {p: f(self.chosen[p], atoms) for p in rps.PARTS}
        return super().residue_features(source)

    def protein_features(self, source):
        return super().protein_features(source.split("@")[0])


def cache_path():
    import hashlib
    return ROOT / "probing_results_atom" / f"global_block_{hashlib.sha1(str(rps.LABELS).encode()).hexdigest()[:10]}.npz"


def build_cache(data):
    """Compute the atom-independent global block for every protein any run draws (train, validation, test)."""
    ids = sorted(set(np.concatenate([data.chosen[p] for p in rps.PARTS]).tolist()))
    blocks = {i: data.global_block(i) for i in ids}
    cache_path().parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path(), blocks=np.array(blocks, dtype=object))
    print(f"cached the global block of {len(blocks)} proteins at {cache_path()}", flush=True)


def check_identity(data):
    """With all five atoms the rebuilt descriptors equal the released ones."""
    ids = data.chosen["train"][:20]
    assert np.array_equal(data.backbone_subset(ids, ATOMS), data.backbone_features_residue(ids))
    assert np.array_equal(data.global_subset(ids, ATOMS), data.global_features_residue(ids))


def main():
    sets = [a for a in sys.argv[1:] if not a.startswith("--")]
    (rps.OUT / "by_source").mkdir(parents=True, exist_ok=True)
    data = AtomData()
    if "--build-cache" in sys.argv:
        return build_cache(data)
    check_identity(data)
    print("identity with the released descriptors verified on 20 proteins", flush=True)
    if VAL:
        for k in ("rows", "chosen", "prot_idx", "pairs", "pair_y"):
            getattr(data, k)["test"] = getattr(data, k)["val"]
    for atoms in sets:
        for kind in ("ref_backbone", "ref_global"):
            s = f"{kind}@{atoms}"
            if not (rps.OUT / "by_source" / f"{s}.csv").exists():
                run_with_retry(data, s)


if __name__ == "__main__":
    main()
