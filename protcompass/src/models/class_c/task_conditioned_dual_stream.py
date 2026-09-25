"""
Task-Conditioned TriStream Encoder.

Extends DualStream with a 3rd Sequence Stream + cross-attention
with learnable task embeddings to adaptively weight streams per task.

Architecture:
    Stream L (Local): 1D Conv ResNet on geometric features → 64D
    Stream P (Packing): MLP on density/topology → 64D
    Stream S (Sequence): AA embeddings + Conv1D → 64D
    Fusion: Task-conditioned cross-attention → 192D
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple

from src.models.base_encoder import (
    BaseProteinEncoder,
    BottleneckClass,
    EncoderOutput,
    ProteinInput,
    register_encoder,
)
from src.models.class_c.dual_stream_encoder import (
    DualStreamEncoder,
    ResNetBlock1D,
    StreamL,
    StreamP,
)
from src.data.preprocessing.geometry_calculator import (
    compute_contact_density,
    compute_topology_features,
)
from src.models.components.handcrafted_features import compute_i1_features


# Task ID mapping for invariant families
TASK_ID_MAP = {
    "I1": 0,
    "I2": 1,
    "I3": 2,
    "I4": 3,
    "I5": 4,
    "I6": 5,
    "I7": 6,
    "I8": 7,
}

NUM_TASKS = len(TASK_ID_MAP)


class StreamS(nn.Module):
    """
    Sequence Stream: Learned AA embeddings + 1D convolutions.

    Lightweight (~50K params). Reuses ResNetBlock1D from Stream L.
    """

    def __init__(self, vocab_size: int = 21, embed_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        self.aa_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=20)

        self.conv_blocks = nn.Sequential(
            nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            ResNetBlock1D(hidden_dim, kernel_size=3, dilation=1),
            ResNetBlock1D(hidden_dim, kernel_size=3, dilation=2),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
        )
        self.output_dim = hidden_dim

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sequence: (B, L) amino acid indices 0-19
            mask: (B, L) boolean mask

        Returns:
            (B, L, hidden_dim)
        """
        x = self.aa_embed(sequence)  # (B, L, embed_dim)
        x = x.transpose(1, 2)  # (B, C, L)

        mask_expanded = mask.unsqueeze(1).float()  # (B, 1, L)
        x = x * mask_expanded

        x = self.conv_blocks(x)
        x = x * mask_expanded

        return x.transpose(1, 2)  # (B, L, hidden_dim)


class TaskCrossAttention(nn.Module):
    """
    Multi-head cross-attention: task embedding queries, stream key-values.

    For each residue, attends over 3 stream tokens with task-specific queries.
    Produces per-residue attention weights (B, L, num_heads, 3) for interpretability.
    """

    def __init__(
        self,
        task_dim: int = 64,
        stream_dim: int = 64,
        num_heads: int = 4,
        num_streams: int = 3,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_streams = num_streams
        self.head_dim = stream_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.total_dim = stream_dim * num_streams  # 192

        self.q_proj = nn.Linear(task_dim, stream_dim)
        self.k_proj = nn.Linear(stream_dim, stream_dim)
        self.v_proj = nn.Linear(stream_dim, stream_dim)
        self.out_proj = nn.Linear(stream_dim, self.total_dim)
        self.norm = nn.LayerNorm(self.total_dim)

    def forward(
        self,
        hl: torch.Tensor,
        hp: torch.Tensor,
        hs: torch.Tensor,
        task_emb: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hl: (B, L, 64) local stream
            hp: (B, L, 64) packing stream
            hs: (B, L, 64) sequence stream
            task_emb: (64,) task embedding (shared across batch)
            mask: (B, L) boolean mask

        Returns:
            output: (B, L, 192) fused representation
            attn_weights: (B, L, num_heads, 3) per-residue stream preferences
        """
        B, L, D = hl.shape

        # Stack streams: (B, L, 3, 64)
        streams = torch.stack([hl, hp, hs], dim=2)

        # Q from task embedding, broadcast to (B, L, 64)
        q = self.q_proj(task_emb).unsqueeze(0).unsqueeze(0).expand(B, L, -1)
        # Reshape to multi-head: (B, L, heads, head_dim)
        q = q.view(B, L, self.num_heads, self.head_dim)

        # K, V from streams: (B, L, 3, 64) → project → (B, L, 3, heads, head_dim)
        k = self.k_proj(streams.view(B * L * self.num_streams, D))
        k = k.view(B, L, self.num_streams, self.num_heads, self.head_dim)
        v = self.v_proj(streams.view(B * L * self.num_streams, D))
        v = v.view(B, L, self.num_streams, self.num_heads, self.head_dim)

        # Attention: Q (B, L, heads, head_dim) vs K (B, L, 3, heads, head_dim)
        # → scores (B, L, heads, 3)
        scores = torch.einsum("blhd,blshd->blhs", q, k) / self.scale

        # Softmax over streams dimension
        attn = F.softmax(scores, dim=-1)  # (B, L, heads, 3)

        # Apply attention to values: (B, L, heads, 3) x (B, L, 3, heads, head_dim)
        attended = torch.einsum("blhs,blshd->blhd", attn, v)  # (B, L, heads, head_dim)

        # Reshape back: (B, L, stream_dim)
        attended = attended.reshape(B, L, D)

        # Project to full dim and add residual from concatenation
        out = self.out_proj(attended)  # (B, L, 192)
        residual = torch.cat([hl, hp, hs], dim=-1)  # (B, L, 192)
        out = self.norm(out + residual)

        return out, attn


@register_encoder("task_conditioned_tri_stream")
class TaskConditionedTriStream(BaseProteinEncoder):
    """
    Task-Conditioned TriStream Encoder.

    Three independent streams (Local, Packing, Sequence) fused via
    task-conditioned cross-attention. Each task family (I1-I8) has a
    learnable embedding that queries the streams differently.

    Without task_id: falls back to simple concatenation (192D).
    With task_id: cross-attention fusion using task embedding (192D).
    """

    bottleneck_class = BottleneckClass.C

    def __init__(
        self,
        hidden_dim: int = 192,
        radii: List[float] = [6.0, 8.0, 10.0, 12.0],
        stream_dim: int = 64,
        num_heads: int = 4,
        num_tasks: int = NUM_TASKS,
    ):
        super().__init__()
        assert hidden_dim == stream_dim * 3, (
            f"hidden_dim ({hidden_dim}) must be 3 * stream_dim ({stream_dim})"
        )

        self.hidden_dim = hidden_dim
        self.radii = radii
        self.stream_dim = stream_dim

        # Stream P: Packing (I3) + Topology (I4)
        self.stream_p = StreamP(
            input_dim=len(radii) + 4,
            output_dim=stream_dim,
        )

        # Global Token Interaction (from DualStream)
        self.global_mlp = nn.Sequential(
            nn.Linear(stream_dim, stream_dim),
            nn.ReLU(),
            nn.Linear(stream_dim, stream_dim),
        )

        # Stream L: Local (I1)
        self.stream_l = StreamL(
            input_dim=21,
            output_dim=stream_dim,
        )

        # Stream S: Sequence (I5)
        self.stream_s = StreamS(
            vocab_size=21,
            embed_dim=stream_dim,
            hidden_dim=stream_dim,
        )

        # Task embeddings
        self.task_embeddings = nn.Embedding(num_tasks, stream_dim)
        nn.init.orthogonal_(self.task_embeddings.weight)

        # Task-conditioned cross-attention
        self.cross_attention = TaskCrossAttention(
            task_dim=stream_dim,
            stream_dim=stream_dim,
            num_heads=num_heads,
            num_streams=3,
        )

    @property
    def embedding_dim(self) -> int:
        return self.hidden_dim  # 192

    def _compute_streams(
        self, protein_input: ProteinInput
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute all three stream outputs."""
        coords = protein_input.coords
        mask = protein_input.mask
        sequence = protein_input.sequence

        # Stream P input
        density = compute_contact_density(coords, mask, threshold=self.radii)
        topology = compute_topology_features(coords, mask, thresholds=[8.0, 10.0])
        p_in = torch.cat([density, topology], dim=-1)

        hp = self.stream_p(p_in)

        # Global token interaction
        g = (hp * mask.unsqueeze(-1)).sum(dim=1) / (
            mask.sum(dim=1, keepdim=True) + 1e-8
        )
        g_out = self.global_mlp(g)
        hp = hp + g_out.unsqueeze(1)

        # Stream L input
        i1_feats = compute_i1_features(coords, mask)
        hl = self.stream_l(i1_feats, mask)

        # Stream S input
        hs = self.stream_s(sequence, mask)

        return hl, hp, hs

    def encode(
        self, protein_input: ProteinInput, task_id: Optional[int] = None
    ) -> EncoderOutput:
        """
        Encode protein structure.

        Args:
            protein_input: ProteinInput with coords, sequence, mask
            task_id: Optional task family index (0-7 for I1-I8).
                     If None, uses simple concatenation.

        Returns:
            EncoderOutput with 192D residue embeddings.
            If task_id is provided, attention weights are stored in metadata.
        """
        mask = protein_input.mask
        hl, hp, hs = self._compute_streams(protein_input)

        metadata = None

        if task_id is not None:
            # Task-conditioned fusion
            task_emb = self.task_embeddings(
                torch.tensor(task_id, device=hl.device)
            )
            h, attn_weights = self.cross_attention(hl, hp, hs, task_emb, mask)
            metadata = {"attn_weights": attn_weights, "task_id": task_id}
        else:
            # Simple concatenation (no task conditioning)
            h = torch.cat([hl, hp, hs], dim=-1)

        # Mask output
        h = h * mask.unsqueeze(-1)

        # Global pooling
        global_emb = h.sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)

        return EncoderOutput(
            residue_embeddings=h,
            global_embedding=global_emb,
            metadata=metadata,
        )

    def get_stream_preference(self, task_id: int) -> Dict[str, float]:
        """
        Get interpretable stream preference for a task.

        Returns average attention weight per stream for the given task.
        Useful for visualization / interpretability analysis.
        """
        task_emb = self.task_embeddings(
            torch.tensor(task_id, device=self.task_embeddings.weight.device)
        )

        # Use dummy inputs to compute preference
        dummy_stream = torch.zeros(1, 1, self.stream_dim, device=task_emb.device)
        # Initialize with unit vectors along different directions for interpretability
        hl = torch.randn_like(dummy_stream)
        hp = torch.randn_like(dummy_stream)
        hs = torch.randn_like(dummy_stream)
        mask = torch.ones(1, 1, dtype=torch.bool, device=task_emb.device)

        with torch.no_grad():
            _, attn = self.cross_attention(hl, hp, hs, task_emb, mask)
            # attn: (1, 1, heads, 3) → mean over heads
            weights = attn.mean(dim=2).squeeze()  # (3,)

        stream_names = ["local", "packing", "sequence"]
        return {name: w.item() for name, w in zip(stream_names, weights)}

    def load_pretrained(self, checkpoint_path: str) -> None:
        """
        Load pretrained weights. Supports loading DualStream checkpoints
        (Stream L + Stream P) with strict=False to skip Stream S + task modules.
        """
        if checkpoint_path and checkpoint_path != "none":
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"TriStream: {len(missing)} missing keys (expected for new modules)")
            if unexpected:
                print(f"TriStream: {len(unexpected)} unexpected keys")


class TaskBoundEncoder(BaseProteinEncoder):
    """
    Wrapper that binds a specific task_id to a TriStream encoder.

    This makes the TriStream compatible with ProbeBattery, which calls
    encoder.encode(protein_input) without a task_id argument.
    """

    bottleneck_class = BottleneckClass.C

    def __init__(self, tri_stream: TaskConditionedTriStream, task_id: Optional[int] = None):
        super().__init__()
        self.tri_stream = tri_stream
        self.task_id = task_id

    @property
    def embedding_dim(self) -> int:
        return self.tri_stream.embedding_dim

    @property
    def name(self) -> str:
        if self.task_id is not None:
            task_name = [k for k, v in TASK_ID_MAP.items() if v == self.task_id]
            task_str = task_name[0] if task_name else str(self.task_id)
            return f"TriStream-{task_str}"
        return "TriStream-NoTask"

    def encode(self, protein_input: ProteinInput) -> EncoderOutput:
        return self.tri_stream.encode(protein_input, task_id=self.task_id)

    def load_pretrained(self, checkpoint_path: str) -> None:
        self.tri_stream.load_pretrained(checkpoint_path)
