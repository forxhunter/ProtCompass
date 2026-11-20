"""PottsMPNN encoder through the official code: see patched/mpnn_official.py."""
from patched.mpnn_official import MPNNOfficial


def build(device):
    return MPNNOfficial(device, "pottsmpnn_soluble")
