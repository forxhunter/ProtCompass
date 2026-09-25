"""Clean Hugging Face path for sequence language models.

Reason for not reusing src/models/external.py: the wrappers slice `last_hidden_state[:, 1:]` without checking where
<eos> and padding fall, run one sequence per call, and the registry default for "esm2" was the 8M checkpoint in the
snapshot that produced the manuscript table. Here the checkpoint is named explicitly, special tokens are located from
the attention mask, and the residue count is asserted for every sequence.
"""
import numpy as np
import torch


def _family(hub):
    h = hub.lower()
    if "prostt5" in h:
        return "prostt5"
    if "prot_t5" in h:
        return "prott5"
    if "ankh" in h:
        return "ankh"
    if "prot_bert" in h:
        return "protbert"
    return "esm"


class HFPLM:
    def __init__(self, hub, device, all_layers=False):
        from transformers import AutoTokenizer, BertModel, BertTokenizer, EsmModel, T5EncoderModel, T5Tokenizer
        self.hub, self.device, self.family, self.all_layers = hub, device, _family(hub), all_layers
        fam = self.family
        half = fam in ("prott5", "prostt5") or "3B" in hub
        dtype = torch.float16 if half else torch.float32
        if fam == "esm":
            self.tok = AutoTokenizer.from_pretrained(hub); self.model = EsmModel.from_pretrained(hub, torch_dtype=dtype, add_pooling_layer=False)
            self.lead = 1
        elif fam in ("prott5", "prostt5"):
            self.tok = T5Tokenizer.from_pretrained(hub, do_lower_case=False, legacy=True); self.model = T5EncoderModel.from_pretrained(hub, torch_dtype=dtype)
            self.lead = 1 if fam == "prostt5" else 0
        elif fam == "ankh":
            self.tok = AutoTokenizer.from_pretrained(hub); self.model = T5EncoderModel.from_pretrained(hub, torch_dtype=dtype)
            self.lead = 0
        else:
            self.tok = BertTokenizer.from_pretrained(hub, do_lower_case=False); self.model = BertModel.from_pretrained(hub, torch_dtype=dtype, add_pooling_layer=False)
            self.lead = 1
        self.model = self.model.eval().to(device)
        cfg = self.model.config
        self.meta = {"checkpoint": hub, "checkpoint_revision": getattr(cfg, "_commit_hash", None), "dtype": str(dtype).replace("torch.", ""),
                     "hidden_size": int(getattr(cfg, "hidden_size", getattr(cfg, "d_model", 0))),
                     "n_layers": int(getattr(cfg, "num_hidden_layers", getattr(cfg, "num_layers", 0))),
                     "n_parameters": int(sum(p.numel() for p in self.model.parameters())),
                     "inputs_passed": "amino-acid sequence only"}

    def _tokenize(self, seqs):
        fam = self.family
        if fam == "esm":
            return self.tok(list(seqs), return_tensors="pt", padding=True, add_special_tokens=True)
        if fam == "prott5":
            return self.tok([" ".join(s) for s in seqs], return_tensors="pt", padding=True, add_special_tokens=True)
        if fam == "prostt5":
            return self.tok(["<AA2fold> " + " ".join(s) for s in seqs], return_tensors="pt", padding="longest", add_special_tokens=True)
        if fam == "ankh":
            return self.tok([list(s) for s in seqs], is_split_into_words=True, return_tensors="pt", padding=True, add_special_tokens=True)
        return self.tok([" ".join(s) for s in seqs], return_tensors="pt", padding=True, add_special_tokens=True)

    @torch.no_grad()
    def residue_batch(self, seqs):
        """List of (L_i, D) arrays, or (n_layers + 1, L_i, D) when all_layers is set."""
        enc = {k: v.to(self.device) for k, v in self._tokenize(seqs).items() if k in ("input_ids", "attention_mask")}
        n_special = self.lead + 1  # leading tokens plus the closing </s>, <eos> or [SEP]
        n_tok = enc["attention_mask"].sum(1).tolist()
        for s, n in zip(seqs, n_tok):
            if n != len(s) + n_special:
                raise RuntimeError(f"{self.hub}: {n} tokens for a sequence of length {len(s)}; expected {len(s) + n_special}")
        out = self.model(**enc, output_hidden_states=self.all_layers)
        h = torch.stack(out.hidden_states, 0) if self.all_layers else out.last_hidden_state
        res = []
        for i, s in enumerate(seqs):
            x = h[..., i, self.lead:self.lead + len(s), :] if self.all_layers else h[i, self.lead:self.lead + len(s)]
            res.append(x.float().cpu().numpy())
        return res

    def residue(self, sequence, coords=None, pdb_path=None):
        return self.residue_batch([sequence])[0]


def token_budget_batches(lengths, budget=24000, max_batch=128):
    """Indices grouped so that batch size times the longest sequence stays under the token budget."""
    order = np.argsort(lengths)[::-1]
    cur, width = [], 0
    for i in order:
        w = max(width, int(lengths[i]) + 2)
        if cur and (w * (len(cur) + 1) > budget or len(cur) >= max_batch):
            yield cur
            cur, w = [], int(lengths[i]) + 2
        cur.append(int(i)); width = w
    if cur:
        yield cur
