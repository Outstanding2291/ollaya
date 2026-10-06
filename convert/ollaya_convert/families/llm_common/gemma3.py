"""An export-friendly forward pass for Gemma 3 text decoders (`Gemma3TextModel`).

Same idea as qwen3.py: recompute the HF forward from the HF submodules (scaled embedding, norms,
projections, MLPs) with explicit attention, positions 0..T-1 and no mask input (rows are right-padded and
every layer is causal, so padding never reaches an earlier position). Gemma 3 specifics:

  * sliding layers attend to the last `sliding_window` positions (key j is visible from query i when
    i - sliding_window < j <= i) with the local RoPE (`rotary_emb_local`); the others attend to every earlier
    position with the global RoPE (`rotary_emb`, linear scaling folded into its inv_freq);
  * q_norm / k_norm per head before RoPE, `scaling = query_pre_attn_scalar ** -0.5`, no soft-capping;
  * a norm after attention and around the MLP (`post_attention_layernorm`, `pre_feedforward_layernorm`,
    `post_feedforward_layernorm`).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def _sliding(text_model, i, layer):
    attn = layer.self_attn
    if getattr(attn, "is_sliding", None) is not None:
        return bool(attn.is_sliding)
    cfg = text_model.config
    if getattr(cfg, "layer_types", None):
        return cfg.layer_types[i] == "sliding_attention"
    return (i + 1) % cfg.sliding_window_pattern != 0


def _rope_tables(text_model):
    """{"global": (inv_freq, attention_scaling), "local": ...}: transformers 4 keeps a second rotary module
    for the sliding layers, transformers 5 one module with a table per layer type."""
    r = text_model.rotary_emb
    if hasattr(r, "full_attention_inv_freq"):
        return {"global": (r.full_attention_inv_freq, r.full_attention_attention_scaling),
                "local": (r.sliding_attention_inv_freq, r.sliding_attention_attention_scaling)}
    local = text_model.rotary_emb_local
    return {"global": (r.inv_freq, getattr(r, "attention_scaling", 1.0)),
            "local": (local.inv_freq, getattr(local, "attention_scaling", 1.0))}


class Gemma3Trunk(nn.Module):
    """input_ids [B, T] -> last hidden state [B, T, hidden] (after the final norm)."""

    def __init__(self, text_model):
        super().__init__()
        self.m = text_model
        cfg = text_model.config
        if getattr(cfg, "attn_logit_softcapping", None) or getattr(cfg, "final_logit_softcapping", None):
            raise SystemExit("Gemma3Trunk does not implement logit soft-capping")
        self.window = int(cfg.sliding_window)
        self.sliding = [_sliding(text_model, i, layer) for i, layer in enumerate(text_model.layers)]
        for name, (inv_freq, scale) in _rope_tables(text_model).items():
            self.register_buffer("inv_freq_" + name, inv_freq.detach().float().clone(), persistent=False)
            setattr(self, "scale_" + name, float(scale))

    def _rope(self, pos, inv_freq, scale, dtype):
        freqs = pos[:, None] * inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return (emb.cos() * scale).to(dtype), (emb.sin() * scale).to(dtype)

    def _attention(self, mod, h, cos, sin, bias):
        B, T, _ = h.shape
        hd = mod.head_dim
        q = mod.q_norm(mod.q_proj(h).view(B, T, -1, hd)).transpose(1, 2)
        k = mod.k_norm(mod.k_proj(h).view(B, T, -1, hd)).transpose(1, 2)
        v = mod.v_proj(h).view(B, T, -1, hd).transpose(1, 2)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        rep = mod.num_key_value_groups
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        w = (q @ k.transpose(-1, -2)) * mod.scaling + bias
        w = torch.softmax(w.float(), dim=-1).to(q.dtype)
        o = (w @ v).transpose(1, 2).reshape(B, T, -1)
        return mod.o_proj(o)

    def forward(self, input_ids):
        m = self.m
        x = m.embed_tokens(input_ids)   # Gemma3TextScaledWordEmbedding: times sqrt(hidden_size)
        T = input_ids.shape[1]
        pos = torch.arange(T, device=input_ids.device, dtype=torch.float32)
        rope = {"global": self._rope(pos, self.inv_freq_global, self.scale_global, x.dtype),
                "local": self._rope(pos, self.inv_freq_local, self.scale_local, x.dtype)}
        i = torch.arange(T, device=input_ids.device)
        future = i[None, :] > i[:, None]
        too_far = i[None, :] <= i[:, None] - self.window
        zero = torch.zeros((T, T), device=x.device, dtype=x.dtype)
        bias = {"global": zero.masked_fill(future, float("-inf")),
                "local": zero.masked_fill(future | too_far, float("-inf"))}
        for layer, sliding in zip(m.layers, self.sliding):
            kind = "local" if sliding else "global"
            cos, sin = rope[kind]
            h = self._attention(layer.self_attn, layer.input_layernorm(x), cos, sin, bias[kind])
            x = x + layer.post_attention_layernorm(h)
            h = layer.mlp(layer.pre_feedforward_layernorm(x))
            x = x + layer.post_feedforward_layernorm(h)
        return m.norm(x)
