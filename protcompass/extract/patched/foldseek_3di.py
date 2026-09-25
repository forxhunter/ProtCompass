"""Foldseek 3Di states from real backbone coordinates via mini3di (no learned layers on top).

Reason for the patch: src/models/class_a/saprot_encoder.py:78 calls mini3di's encode_atoms with three arguments; the call
raises TypeError, the exception is swallowed, and every residue gets state 0. The registered "foldseek" encoder then fed the
constant tokens to a randomly initialised Transformer. Here each residue is the one-hot of its 3Di state, so the masked mean
over residues is the normalised state frequency of the rebuttal table.
"""
import mini3di
import numpy as np


def states_3di(coords):
    enc = _ENC
    st = enc.encode_atoms(coords[:, 1].astype(np.float64), coords[:, 4].astype(np.float64), coords[:, 0].astype(np.float64), coords[:, 2].astype(np.float64))
    return np.asarray(np.ma.filled(st, 0)).astype(np.int64), int(np.ma.count_masked(st))


_ENC = mini3di.Encoder()


class Foldseek3Di:
    def __init__(self, device):
        self.meta = {"tool": f"mini3di {mini3di.__version__}", "inputs_passed": "N, CA, C, CB coordinates", "n_parameters": 0,
                     "patch": "patched/foldseek_3di.py"}

    def residue(self, sequence, coords=None, pdb_path=None):
        t, _ = states_3di(coords)
        out = np.zeros((len(t), 20), dtype=np.float32)
        out[np.arange(len(t)), t] = 1.0
        return out


def build(device):
    return Foldseek3Di(device)
