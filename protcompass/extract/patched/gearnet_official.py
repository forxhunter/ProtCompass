"""GearNet-Edge (Multiview Contrast checkpoint) through the official TorchDrug model and graph construction.

Reason for the patch. The registry wrapper (src/models/pretrained/gearnet_pretrained.py) is a re-implementation whose
tensor shapes match the checkpoint but whose computation does not match the model: seven relation types of its own
(the official ones are five sequence offsets, a radius relation and a nearest-neighbour relation), a 59-dimensional edge
feature with a different layout, a line graph typed by which ends two edges share instead of by the angle between
them, no message from the edge states to the nodes, the project's residue order instead of TorchDrug's, the mean of
the layer outputs instead of their concatenation, and batch-norm statistics re-estimated on the benchmark data.
Here `torchdrug.models.GearNet` is built with the configuration of the released checkpoint
(config/pretrain/mc_gearnet_edge.yaml of the GearNet repository), the checkpoint is loaded with strict=True, and the
graph is built by TorchDrug's own layers: alpha-carbon nodes, sequential edges up to distance 2, radius edges at 10 A
and 10 nearest neighbours, both with a minimum sequence distance of 5, and the "gearnet" edge feature.

The protein is built from the CA coordinates and residue types of the chain; on the released structure files this
gives the same output as `Protein.from_pdb`. TorchDrug's fused sparse aggregation does not run on current PyTorch,
so each layer uses the library's own `message` and `aggregate` methods, which define the same update.

The embeddings are defined by the GPU path. TorchDrug's radius edges keep at most 32 neighbours per residue
(`SpatialEdge(max_num_neighbors=32)` in torch_cluster), and the CPU and GPU implementations keep different ones when
a residue has more: on chains with dense cores the CPU output differs from the GPU output by a relative 0.01 to 0.05,
while two GPUs (RTX 5090 with PyTorch 2.11, RTX 4090 with PyTorch 2.8) agree to 1e-6.

Input: CA coordinates and residue identity. Output: (L, 3072), the concatenated node states of the six layers.
Needs `pylib_torchdrug` (torchdrug 0.2.1, rdkit-pypi 2022.9.5, torch-cluster) on PYTHONPATH.
"""
import torch

from common import ROOT, SEED, file_sha256, weights_sha256

CKPT = ROOT / "checkpoints" / "gearnet" / "mc_gearnet_edge.pth"
THREE = {"A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE", "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS",
         "L": "LEU", "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG", "S": "SER", "T": "THR", "V": "VAL",
         "W": "TRP", "Y": "TYR"}


class GearNetOfficial:
    def __init__(self, device, random_weights=False):
        from torchdrug import data, layers, models
        from torchdrug.layers import geometry
        layers.GeometricRelationalGraphConv.message_and_aggregate = \
            lambda self, graph, input: self.aggregate(graph, self.message(graph, input))
        self.data = data
        self.graph = layers.GraphConstruction(
            node_layers=[geometry.AlphaCarbonNode()],
            edge_layers=[geometry.SequentialEdge(max_distance=2), geometry.SpatialEdge(radius=10.0, min_distance=5),
                         geometry.KNNEdge(k=10, min_distance=5)], edge_feature="gearnet")
        if random_weights:
            torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        model = models.GearNet(input_dim=21, hidden_dims=[512] * 6, num_relation=7, edge_input_dim=59, num_angle_bin=8,
                               batch_norm=True, concat_hidden=True, short_cut=True, readout="sum")
        if not random_weights:
            model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=False), strict=True)
        self.model = model.eval().to(device)
        self.device = device
        self.residue_id = {a: data.Protein.residue2id[t] for a, t in THREE.items()}
        self.ca = data.Protein.atom_name2id["CA"]
        self.meta = {"checkpoint": None if random_weights else str(CKPT),
                     "checkpoint_sha256": None if random_weights else file_sha256(CKPT),
                     "checkpoint_loaded_by": "none (random weights, fixed seed)" if random_weights else "strict=True",
                     "weights_sha256": weights_sha256(self.model),
                     "n_parameters": int(sum(p.numel() for p in self.model.parameters())),
                     "inputs_passed": "CA coordinates and residue identity; graph and edge features by TorchDrug",
                     "patch": "patched/gearnet_official.py"}
        if random_weights:
            self.meta["seed"] = SEED

    def _protein(self, sequence, coords):
        L = len(sequence)
        rt = torch.tensor([self.residue_id[a] for a in sequence])
        return self.data.Protein(
            edge_list=torch.zeros(0, 3, dtype=torch.long), atom_type=torch.full((L,), 6),
            bond_type=torch.zeros(0, dtype=torch.long), num_node=L, num_residue=L,
            node_position=torch.from_numpy(coords[:, 1]).float(), atom_name=torch.full((L,), self.ca),
            atom2residue=torch.arange(L), residue_type=rt, residue_feature=torch.nn.functional.one_hot(rt, 21).float(),
            residue_number=torch.arange(1, L + 1), num_relation=4)

    @torch.no_grad()
    def residue(self, sequence, coords=None, pdb_path=None):
        return self.residue_many([(sequence, coords)])[0]

    @torch.no_grad()
    def residue_many(self, items):
        """Several chains as one packed graph: a list of (sequence, coords) in, a list of (L, 3072) arrays out.
        Edges are built within each chain, so a chain's states do not depend on what it is packed with."""
        sizes = [len(s) for s, _ in items]
        g = self.graph(self.data.Protein.pack([self._protein(s, c) for s, c in items]).to(self.device))
        out = self.model(g, g.residue_feature.float())["node_feature"]
        if out.shape[0] != sum(sizes):
            raise RuntimeError(f"GearNet: {out.shape[0]} node states for {sum(sizes)} residues")
        out = out.float().cpu().numpy()
        return [out[o - n:o] for n, o in zip(sizes, __import__("numpy").cumsum(sizes))]


def build(device):
    return GearNetOfficial(device)
