"""
External model wrappers for InvariantsBench.

Includes:
- ESM-2 (Sequence-only baseline, or with structural adapter)
- 3D-CNN (Voxel-based baseline)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from .base_encoder import (
    BaseProteinEncoder, 
    EncoderOutput, 
    ProteinInput, 
    BottleneckClass,
    register_encoder
)

# Try imports
try:
    from transformers import EsmModel, EsmConfig, EsmTokenizer
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

@register_encoder("esm2_650m")
@register_encoder("esm2")
class ESM2Adapter(BaseProteinEncoder):
    """
    Wrapper for ESM-2 sequence protein language model.
    Using HuggingFace implementation.
    """

    def __init__(self, model_name: str = "facebook/esm2_t33_650M_UR50D", freeze: bool = True,
                 use_half: bool = False, cpu_offload: bool = False, **kwargs):
        super().__init__()
        self.model_name = model_name
        self._freeze_backbone = freeze
        self.use_half = use_half
        self._cpu_offload = cpu_offload
        # Ignore extra kwargs (like hidden_dim from generic runners)

        if not HF_AVAILABLE:
            raise ImportError("transformers not installed. Run `pip install transformers`.")

        print(f"Loading ESM-2: {model_name}")
        model_kwargs = {}
        if use_half:
            model_kwargs["torch_dtype"] = torch.float16
        if cpu_offload:
            # device_map="auto" splits layers across VRAM + RAM via accelerate
            model_kwargs["device_map"] = "auto"
            model_kwargs["low_cpu_mem_usage"] = True
            print("  [cpu_offload] Using device_map='auto' (VRAM+RAM split)")
        self.model = EsmModel.from_pretrained(model_name, **model_kwargs)
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)
        if use_half and not cpu_offload:
            # half() is incompatible with device_map="auto" (handled by torch_dtype above)
            self.model = self.model.half()

        if freeze:
            self.freeze()
            
    @property
    def bottleneck_class(self) -> BottleneckClass:
        return BottleneckClass.A # Discrete token (Sequence)
        
    @property
    def embedding_dim(self) -> int:
        return self.model.config.hidden_size
        
    def encode(self, protein: ProteinInput) -> EncoderOutput:
        device = protein.coords.device

        # When cpu_offload is active (device_map="auto"), the model manages its own
        # device placement — send inputs to the embedding layer's device instead.
        if getattr(self, '_cpu_offload', False):
            try:
                input_device = next(self.model.embeddings.parameters()).device
            except Exception:
                input_device = torch.device("cpu")
        else:
            input_device = device

        from .base_encoder import tensor_to_sequence

        seqs = [tensor_to_sequence(s) for s in protein.sequence]

        # Tokenize
        inputs = self.tokenizer(seqs, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        inputs = {k: v.to(input_device) for k, v in inputs.items()}
        
        # Forward
        with torch.set_grad_enabled(not self._is_frozen):
            outputs = self.model(**inputs)
            
        # Extract embeddings (B, L_tokens, D)
        # ESM adds <cls> and <eos>, so length is L+2
        # We need to strip them to match (B, L, D) of protein.coords
        
        # Note: Padding handling is tricky if protein.sequence had padding
        # In `inputs`, tokenizer handles padding.
        # We need to extract the tokens corresponding to the actual sequence.
        
        # embeddings = outputs.last_hidden_state[:, 1:-1, :] # Naive strip
        
        # Robust strip:
        last_hidden = outputs.last_hidden_state
        B = last_hidden.shape[0]
        L_target = protein.sequence.shape[1]
        
        # We assume tokenizer produced <cls> SEQ <eos> <pad> ...
        # We take 1 : 1+L_target? 
        # CAUTION: Tokenizer might tokenize string differently than we expect if there are unknown chars.
        # But for standard AAs, it should be 1-to-1.
        
        # Safe slicing
        # Check lengths
        
        # Reshape/Pad to match target
        embeddings = last_hidden[:, 1:, :] # Remove CLS
        # Remove EOS ? EOS is at different pos for each seq
        
        # Let's trust that we just take the first L_target tokens?
        # A bit risky. 
        
        # Better approach: Create a tensor directly if vocab maps align.
        # For now, let's just take [: L] if it matches.
        
        # Move to CPU first for indexing (safe with device_map layouts)
        last_hidden_cpu = last_hidden.cpu()

        final_emb = torch.zeros(B, L_target, self.embedding_dim)

        # Tokenizer may truncate to max_length=1024, so the usable slice is
        # last_hidden.shape[1] - 2 (strip CLS/EOS). Clamp per sequence.
        max_avail = last_hidden_cpu.shape[1] - 2
        for i in range(B):
            seq_len = int(protein.mask[i].sum().item())
            copy_len = min(seq_len, max_avail)
            valid_emb = last_hidden_cpu[i, 1:1 + copy_len, :]
            final_emb[i, :copy_len, :] = valid_emb

        # Move to target device — always cast to float32 so fp16 models work with fp32 probes
        final_emb = final_emb.to(device=device, dtype=torch.float32)
        cls_embedding = last_hidden_cpu[:, 0, :].to(device=device, dtype=torch.float32)

        return EncoderOutput(
            residue_embeddings=final_emb,
            global_embedding=cls_embedding
        )

    def load_pretrained(self, checkpoint_path: str) -> None:
        pass


@register_encoder("mif")
class MIFAdapter(BaseProteinEncoder):
    """
    Masked Inverse Folding (MIF) Wrapper (Placeholder/Adapter).
    """
    def __init__(self, hidden_dim: int = 256, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Placeholder GNN
        self.proj = nn.Linear(25, hidden_dim) # One-hot to dim
        
    @property
    def bottleneck_class(self) -> BottleneckClass:
        return BottleneckClass.C
        
    @property
    def embedding_dim(self) -> int:
        return self.hidden_dim
        
    def encode(self, protein: ProteinInput) -> EncoderOutput:
        # Placeholder implementation
        B, L = protein.coords.shape[:2]
        device = protein.coords.device
        
        # Random for now, waiting for weights
        x = torch.randn(B, L, self.hidden_dim, device=device)
        
        mask = protein.mask.unsqueeze(-1)
        g = (x * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        
        return EncoderOutput(residue_embeddings=x, global_embedding=g)
        
    def load_pretrained(self, checkpoint_path: str) -> None:
        pass




# --- ProtTrans (T5) ---
@register_encoder("prottrans")
class ProtTransAdapter(BaseProteinEncoder):
    """
    Wrapper for ProtT5-XL-UniRef50.
    """
    def __init__(self, model_name: str = "Rostlab/prot_t5_xl_half_uniref50-enc", freeze: bool = True, **kwargs):
        super().__init__()
        self.model_name = model_name
        self._freeze_backbone = freeze
        
        if not HF_AVAILABLE:
            raise ImportError("transformers not installed.")
            
        print(f"Loading ProtTrans: {model_name}")
        from transformers import T5EncoderModel, T5Tokenizer
        
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
        self.model = T5EncoderModel.from_pretrained(model_name)
        
        if freeze:
            self.freeze()
            
    @property
    def bottleneck_class(self) -> BottleneckClass:
        return BottleneckClass.A
        
    @property
    def embedding_dim(self) -> int:
        return self.model.config.d_model
        
    def encode(self, protein: ProteinInput) -> EncoderOutput:
        device = protein.coords.device
        
        # 1. Convert to string list (spaced for T5)
        # ProtTrans expects: "M E T ..." (space separated)
        from .base_encoder import tensor_to_sequence
        seqs = []
        for s in protein.sequence:
            # tensor_to_sequence returns "MET..."
            # insert spaces
            raw = tensor_to_sequence(s)
            seqs.append(" ".join(list(raw)))
            
        # 2. Tokenize
        # Note: T5 tokenizer behavior
        ids = self.tokenizer(seqs, add_special_tokens=True, padding="longest", return_tensors="pt")
        input_ids = ids['input_ids'].to(device)
        attention_mask = ids['attention_mask'].to(device)
        
        # 3. Forward
        with torch.set_grad_enabled(not self._is_frozen):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
        last_hidden = outputs.last_hidden_state # (B, L_tokens, D)
        
        # 4. Extract (Remove EOS)
        # T5 adds </s> at end. No CLS at start.
        # We need to map back to original length.
        
        B = last_hidden.shape[0]
        L_target = protein.sequence.shape[1]
        
        final_emb = torch.zeros(B, L_target, self.embedding_dim, device=device)
        
        max_avail = last_hidden.shape[1]
        for i in range(B):
            seq_len = int(protein.mask[i].sum().item())
            copy_len = min(seq_len, max_avail)
            final_emb[i, :copy_len, :] = last_hidden[i, :copy_len, :]
            
        # Global: Mean pool (excluding EOS)
        # Using mask from protein input
        mask = protein.mask.unsqueeze(-1)
        global_emb = (final_emb * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        
        return EncoderOutput(residue_embeddings=final_emb, global_embedding=global_emb)

    def load_pretrained(self, checkpoint_path: str) -> None:
        pass


# --- Ankh ---
@register_encoder("ankh")
class AnkhAdapter(BaseProteinEncoder):
    """
    Wrapper for Ankh (Google/Elu-lab).
    """
    def __init__(self, model_name: str = "ElnaggarLab/ankh-base", freeze: bool = True, **kwargs):
        super().__init__()
        self.model_name = model_name
        
        if not HF_AVAILABLE:
            raise ImportError("transformers not installed.")
            
        print(f"Loading Ankh: {model_name}")
        # Ankh uses T5 architecture usually or custom? 
        # Check HF model card. It is T5EncoderModel usually.
        from transformers import T5EncoderModel, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = T5EncoderModel.from_pretrained(model_name)
        
        if freeze:
            self.freeze()
            
    @property
    def bottleneck_class(self) -> BottleneckClass:
        return BottleneckClass.A
        
    @property
    def embedding_dim(self) -> int:
        return self.model.config.d_model
        
    def encode(self, protein: ProteinInput) -> EncoderOutput:
        # Similar to ProtTrans but checking tokenizer behavior
        device = protein.coords.device
        from .base_encoder import tensor_to_sequence
        
        # Ankh expects list of list of AAs? Or string?
        # Usually standard string list.
        seqs = [list(tensor_to_sequence(s)) for s in protein.sequence]
        
        ids = self.tokenizer(seqs, add_special_tokens=True, padding=True, is_split_into_words=True, return_tensors="pt")
        input_ids = ids['input_ids'].to(device)
        attention_mask = ids['attention_mask'].to(device)
        
        with torch.set_grad_enabled(not self._is_frozen):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
        last_hidden = outputs.last_hidden_state
        
        B = last_hidden.shape[0]
        L_target = protein.sequence.shape[1]
        final_emb = torch.zeros(B, L_target, self.embedding_dim, device=device)
        
        max_avail = last_hidden.shape[1]
        for i in range(B):
            seq_len = int(protein.mask[i].sum().item())
            copy_len = min(seq_len, max_avail)
            final_emb[i, :copy_len, :] = last_hidden[i, :copy_len, :]
            
        mask = protein.mask.unsqueeze(-1)
        global_emb = (final_emb * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        
        return EncoderOutput(residue_embeddings=final_emb, global_embedding=global_emb)

    def load_pretrained(self, checkpoint_path: str) -> None:
        pass


@register_encoder("cnn3d")
class CNN3DEncoder(BaseProteinEncoder):
    """
    Placeholder for Voxel-based 3D CNN baseline.
    """
     
    def __init__(self, output_dim: int = 128, **kwargs):
        super().__init__()
        # If hidden_dim is passed, use it as output_dim
        if "hidden_dim" in kwargs:
            output_dim = kwargs["hidden_dim"]
        self._embedding_dim = output_dim
        # Placeholder layers
        self.conv = nn.Conv3d(4, 16, 3, padding=1) 
        
    @property
    def bottleneck_class(self) -> BottleneckClass:
        return BottleneckClass.C # Equivariant-ish (Euclidean)
        
    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
        
    def encode(self, protein: ProteinInput) -> EncoderOutput:
        # Placeholder: return random embeddings
        B, L, _ = protein.coords.shape[:3]
        res = torch.randn(B, L, self.embedding_dim, device=protein.coords.device)
        return EncoderOutput(
            residue_embeddings=res,
            global_embedding=res.mean(dim=1)
        )
        
    def load_pretrained(self, checkpoint_path: str) -> None:
        pass



# ── ESM-1b alias (uses same ESM2Adapter, different default model) ──────────

@register_encoder("esm1b")
class ESM1bAdapter(ESM2Adapter):
    """ESM-1b (650M) alias using the ESM2Adapter wrapper."""

    def __init__(
        self,
        model_name: str = "facebook/esm1b_t33_650M_UR50S",
        freeze: bool = True,
        **kwargs,
    ):
        super().__init__(model_name=model_name, freeze=freeze, **kwargs)


@register_encoder("esm2_150m")
class ESM2_150MAdapter(ESM2Adapter):
    """ESM-2-150M (150M) alias using the ESM2Adapter wrapper."""

    def __init__(
        self,
        model_name: str = "facebook/esm2_t30_150M_UR50D",
        freeze: bool = True,
        **kwargs,
    ):
        super().__init__(model_name=model_name, freeze=freeze, **kwargs)


@register_encoder("esm2_35m")
class ESM2_35MAdapter(ESM2Adapter):
    """ESM-2-35M (35M) alias using the ESM2Adapter wrapper."""

    def __init__(
        self,
        model_name: str = "facebook/esm2_t12_35M_UR50D",
        freeze: bool = True,
        **kwargs,
    ):
        super().__init__(model_name=model_name, freeze=freeze, **kwargs)
