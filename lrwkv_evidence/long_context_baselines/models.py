"""Bidirectional softmax denoiser and native cached causal RWKV adapters.

Pythia uses its own tokenizer and pretraining. It is a practical comparator,
not a controlled causal test of changing only the sequence mixer.
"""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class Tokenizer:
    def __init__(self, base):
        from tokenizers import Tokenizer as Native
        self.native = Native.from_file(str(Path(base) / 'tokenizer.json'))
        encoded = [self.encode(str(i)) for i in (0, 1)]
        if any(len(x) != 1 for x in encoded):
            raise ValueError('binary targets must be single native tokens')
        self.binary_ids = tuple(x[0] for x in encoded)
        # The unused Pythia padding vocabulary row is a dedicated mask ID.
        self.mask_id = 50303
        if self.mask_id < self.native.get_vocab_size() or self.mask_id in self.binary_ids:
            raise ValueError('expected unused mask vocabulary row')

    def encode(self, text):
        return self.native.encode(text, add_special_tokens=False).ids


class BinaryProjection(nn.Module):
    def __init__(self, head, binary_ids):
        super().__init__()
        if head.bias is not None:
            raise ValueError('expected bias-free language head')
        self.weight = head.weight
        self.register_buffer('binary_ids', torch.tensor(binary_ids))

    def forward(self, hidden):
        selected = self.weight.index_select(0, self.binary_ids).float()
        logodds = F.linear(hidden.float(), (selected[1] - selected[0]).unsqueeze(0))
        return torch.cat((torch.zeros_like(logodds), logodds), dim=-1)


class AttentionDenoiser(nn.Module):
    def __init__(self, pretrained, binary_ids, gradient_checkpointing=True):
        super().__init__()
        self.backbone = pretrained.gpt_neox
        self.head = BinaryProjection(pretrained.embed_out, binary_ids)
        self.gradient_checkpointing = gradient_checkpointing
        self.backbone.config._attn_implementation = 'sdpa'
        for layer in self.backbone.layers:
            layer.attention.is_causal = False

    def document(self, ids, gather):
        h = self.backbone.emb_dropout(self.backbone.embed_in(ids))
        positions = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
        rope = self.backbone.rotary_emb(h, position_ids=positions)
        for layer in self.backbone.layers:
            # Bypass GPTNeoXModel.forward, which constructs a causal mask.
            # SDPA sees no mask and explicit is_causal=False: full bidirectionality.
            def apply(x, layer=layer):
                return layer(x, attention_mask=None, position_ids=positions, use_cache=False,
                             position_embeddings=rope, is_causal=False)
            h = checkpoint(apply, h, use_reentrant=False) if self.training and self.gradient_checkpointing else apply(h)
        return self.head(self.backbone.final_layer_norm(h).index_select(1, gather))

    def forward(self, batch):
        starts = batch['doc_starts'][0].tolist()
        ends = starts[1:] + [batch['input_ids'].shape[1]]
        n = batch['gold'].shape[1]
        return torch.cat([self.document(batch['input_ids'][:, lo:hi], batch['gather_idx'][j*n:(j+1)*n]-lo)
                          for j, (lo, hi) in enumerate(zip(starts, ends))], dim=0)


class CausalRWKV(nn.Module):
    def __init__(self, pretrained, binary_ids):
        super().__init__()
        self.backbone = pretrained.model
        self.head = BinaryProjection(pretrained.lm_head, binary_ids)

    def document(self, ids, gather):
        # Predictions at prefix_end-1,...,prefix_end+N-2 predict the next bit.
        hidden = self.backbone(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
        return self.head(hidden.index_select(1, gather))

    def forward(self, batch):
        starts = batch['doc_starts'][0].tolist()
        ends = starts[1:] + [batch['input_ids'].shape[1]]
        n = batch['gold'].shape[1]
        return torch.cat([self.document(batch['input_ids'][:, lo:hi], batch['gather_idx'][j*n:(j+1)*n]-lo-1)
                          for j, (lo, hi) in enumerate(zip(starts, ends))], dim=0)

    def next_cached(self, input_ids, state=None):
        outputs = self.backbone(input_ids=input_ids, past_key_values=state, use_cache=True, return_dict=True)
        return self.head(outputs.last_hidden_state[:, -1]), outputs.past_key_values


def build(kind, base):
    torch.manual_seed(0)
    if kind == 'attention':
        from transformers import GPTNeoXForCausalLM
        tok = Tokenizer(base)
        pretrained = GPTNeoXForCausalLM.from_pretrained(str(base), local_files_only=True, dtype=torch.float32,
                                                     attn_implementation='sdpa')
        model = AttentionDenoiser(pretrained, tok.binary_ids)
    elif kind == 'causal_rwkv':
        from fla.models.rwkv7 import RWKV7ForCausalLM
        from lrwkv_evidence.train04.worker import load_tokenizer
        tok, _, _ = load_tokenizer(base)
        pretrained = RWKV7ForCausalLM.from_pretrained(str(base), local_files_only=True, dtype=torch.float32)
        model = CausalRWKV(pretrained, tok.binary_ids)
    else:
        raise ValueError('unknown comparator')
    if any(p.dtype != torch.float32 or not p.requires_grad for p in model.parameters()):
        raise ValueError('requires trainable FP32 parameters')
    return tok, model
