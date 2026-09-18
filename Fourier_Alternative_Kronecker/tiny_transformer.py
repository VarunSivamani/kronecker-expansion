"""A small transformer classifier with a swappable input codec.

Each "sequence" here is a single token (length 1), so the transformer
degenerates to: codec -> projection -> a couple of self-attention +
feedforward blocks -> classification head. Attention over length-1
sequences is trivial, but keeping the real transformer blocks in place
(rather than stripping them out) keeps this an honest test of "does the
rest of the model receive a useful representation from the codec", not
just a linear-probe test of the codec alone.
"""

from __future__ import annotations

import torch
from torch import nn


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class TinyClassifier(nn.Module):
    def __init__(
        self,
        codec: nn.Module,
        d_model: int,
        num_classes: int,
        n_layers: int = 2,
        n_heads: int = 4,
        d_ff: int = 256,
    ):
        super().__init__()
        self.codec = codec
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: [B] -> treat as sequence length 1
        x = self.codec(token_ids.unsqueeze(1))  # [B, 1, D]
        for block in self.blocks:
            x = block(x)
        x = x.squeeze(1)  # [B, D]
        return self.head(x)


def count_trainable_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
