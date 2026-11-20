"""Encoder states h_V of ProteinMPNN and PottsMPNN from the official model code, feature pipeline and settings.

Reason for the patch. The registry wrapper of ProteinMPNN (src/models/pretrained/proteinmpnn_official.py) is a
re-implementation: it leaves each residue out of its own neighbour list, orders the 25 atom-pair distance blocks
differently from the official code, uses a wider radial basis and sets the chain indicator of the positional encoding
for every pair. The official weights load into it, but on the same coordinates its output differs from the official
encoder by a relative 1.2 to 1.3. The registry wrappers of PottsMPNN call the official code with the class default of
64 neighbours, while the checkpoints were trained with 48 (`num_edges` in each file); the output differs by 0.17 to
0.28. Here the official classes are built with the neighbour count stored in the checkpoint, the weights are loaded
strictly (PottsMPNN: every tensor except the Potts output layer, which the encoder does not use), and the encoder is
called as in the official forward pass.

  ProteinMPNN   external_repos/ProteinMPNN, checkpoints/proteinmpnn_v48_020.pt (v_48_020)
  PottsMPNN     external_repos/PottsMPNN: proteinmpnn_compatible_model_weights/pottsmpnn_20.pt,
                soluble_model_weights/sol_pottsmpnn_20.pt, ft_model_weights/potts_ft.pt
  random        the PottsMPNN architecture with the same 48 neighbours, default initialisation, fixed seed

Input: N, CA, C, O coordinates of one chain. Output: (L, 128).
"""
import sys

import torch

from common import ROOT, SEED, file_sha256, weights_sha256

MPNN = ROOT / "external_repos" / "ProteinMPNN"
POTTS = ROOT / "external_repos" / "PottsMPNN"
CHECKPOINT = {"proteinmpnn": ROOT / "checkpoints" / "proteinmpnn_v48_020.pt",
              "pottsmpnn_pretrained": POTTS / "proteinmpnn_compatible_model_weights" / "pottsmpnn_20.pt",
              "pottsmpnn_soluble": POTTS / "soluble_model_weights" / "sol_pottsmpnn_20.pt",
              "pottsmpnn_ft": POTTS / "ft_model_weights" / "potts_ft.pt"}
RANDOM_NEIGHBOURS = 48


class MPNNOfficial:
    def __init__(self, device, kind):
        self.device = device
        random_weights = kind == "pottsmpnn_random"
        if kind == "proteinmpnn":
            sys.path.insert(0, str(MPNN))
            from protein_mpnn_utils import ProteinMPNN, gather_nodes
        else:
            sys.path.insert(0, str(POTTS))
            from potts_mpnn_utils import PottsMPNN, gather_nodes
        self.gather_nodes = gather_nodes
        ck = None if random_weights else torch.load(CHECKPOINT[kind], map_location="cpu", weights_only=False)
        k = RANDOM_NEIGHBOURS if random_weights else int(ck["num_edges"])
        if random_weights:
            torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        if kind == "proteinmpnn":
            model = ProteinMPNN(num_letters=21, node_features=128, edge_features=128, hidden_dim=128, num_encoder_layers=3,
                                num_decoder_layers=3, augment_eps=0.0, k_neighbors=k)
            model.load_state_dict(ck["model_state_dict"], strict=True)
            how = "strict=True"
        else:
            model = PottsMPNN(num_letters=21, node_features=128, edge_features=128, hidden_dim=128, potts_dim=400,
                              num_encoder_layers=3, num_decoder_layers=3, vocab=21, k_neighbors=k, augment_eps=0.0,
                              dropout=0.0, ca_only=False)
            how = "none (random weights, fixed seed)"
            if not random_weights:
                res = model.load_state_dict(ck["model_state_dict"], strict=False)
                if res.unexpected_keys or any(not m.startswith("etab_out.") for m in res.missing_keys):
                    raise RuntimeError(f"{kind}: checkpoint does not fit the official model: {res}")
                how = "every checkpoint tensor loaded; only the Potts output layer, unused by the encoder, is absent" \
                    if res.missing_keys else "strict=True"
        self.model = model.eval().to(device)
        self.meta = {"checkpoint": None if random_weights else str(CHECKPOINT[kind]),
                     "checkpoint_sha256": None if random_weights else file_sha256(CHECKPOINT[kind]),
                     "checkpoint_loaded_by": how, "k_neighbors": k, "weights_sha256": weights_sha256(self.model),
                     "n_parameters": int(sum(p.numel() for p in self.model.parameters())),
                     "inputs_passed": "N, CA, C, O coordinates of the chain; no sequence",
                     "patch": "patched/mpnn_official.py"}
        if random_weights:
            self.meta["seed"] = SEED

    @torch.no_grad()
    def residue(self, sequence, coords=None, pdb_path=None):
        L = len(sequence)
        X = torch.from_numpy(coords[:, :4]).float()[None].to(self.device)
        mask = torch.ones(1, L, device=self.device)
        residue_idx = torch.arange(L, device=self.device)[None]
        chain = torch.ones(1, L, dtype=torch.long, device=self.device)
        m = self.model
        E, E_idx = m.features(X, mask, residue_idx, chain)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = m.W_e(E)
        mask_attend = self.gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in m.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
        return h_V[0].float().cpu().numpy()


def build(device):
    return MPNNOfficial(device, "proteinmpnn")
