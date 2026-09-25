"""Official GCPNet architecture with random weights and a fixed seed: the control for gcpnet_pretrained."""
from patched.gcpnet_official import GCPNetOfficial


def build(device):
    return GCPNetOfficial(device, random_weights=True)
