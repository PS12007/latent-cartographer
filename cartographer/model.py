"""A small GPT with first-class hooks for interpretability.

Kept deliberately plain (pre-LN, learned absolute positions, GELU MLP) so the
same forward pass can be reimplemented line-for-line in JavaScript for the
browser demo (docs/js/model.js) and checked for numerical agreement.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 129
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128
    d_mlp: int = 512

    def to_dict(self):
        return asdict(self)


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)
        self.head_hook = None

    def forward(self, x, cache=None, prefix=""):
        B, S, D = x.shape
        H = self.cfg.n_head
        q, k, v = self.qkv(x).split(D, dim=2)
        q = q.view(B, S, H, D // H).transpose(1, 2)
        k = k.view(B, S, H, D // H).transpose(1, 2)
        v = v.view(B, S, H, D // H).transpose(1, 2)
        if cache is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(D // H)
            att = att.masked_fill(~self.mask[:S, :S], float("-inf")).softmax(-1)
            cache[prefix + "attn"] = att
            y = att @ v
            cache[prefix + "head_out"] = y  # (B,H,S,dh) before output proj
        if self.head_hook is not None:  # edit per-head outputs, e.g. for ablations
            y = self.head_hook(y)
        y = y.transpose(1, 2).contiguous().view(B, S, D)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.fc = nn.Linear(cfg.d_model, cfg.d_mlp)
        self.out = nn.Linear(cfg.d_mlp, cfg.d_model)

    def forward(self, x, cache=None, prefix=""):
        a = self.attn(self.ln1(x), cache, prefix)
        x = x + a
        if cache is not None:
            cache[prefix + "attn_out"] = a
            cache[prefix + "resid_mid"] = x
        h = F.gelu(self.fc(self.ln2(x)))
        if cache is not None:
            cache[prefix + "mlp_post"] = h
        m = self.out(h)
        if cache is not None:
            cache[prefix + "mlp_out"] = m
        return x + m


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.block_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, 4)  # only ever predicts moves
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, cache: dict | None = None, resid_hook=None):
        """idx (B,S) -> move logits (B,S,4).

        cache: if a dict is passed, it is filled with activations:
            resid_{l}      residual stream entering block l (l = n_layer is the final stream)
            L{l}.attn      attention patterns (B,H,S,S)
            L{l}.mlp_post  MLP hidden activations, etc.
        resid_hook(l, x) -> x: lets you edit the residual stream entering block l
            (l == n_layer edits the final stream before ln_f).
        """
        B, S = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(S, device=idx.device))
        for l, blk in enumerate(self.blocks):
            if resid_hook is not None:
                x = resid_hook(l, x)
            if cache is not None:
                cache[f"resid_{l}"] = x
            x = blk(x, cache, f"L{l}.")
        if resid_hook is not None:
            x = resid_hook(self.cfg.n_layer, x)
        if cache is not None:
            cache[f"resid_{self.cfg.n_layer}"] = x
        return self.head(self.ln_f(x))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def load_model(path, map_location="cpu"):
    ck = torch.load(path, map_location=map_location, weights_only=False)
    model = GPT(GPTConfig(**ck["config"]))
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck
