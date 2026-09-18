"""
Dynamic Kronecker codecs — Problem #3 (Dynamic_Kronecker)
=========================================================

classic  — paper / Session 7 (hard pos_dim crop)
fourier  — ours: Fourier-position Kronecker (no crop, fixed D = 256·r)

Shape cheat-sheet
-----------------
  text (str)
    → UTF-8 bytes list length L
    → κ code vector                          [D]           where D = 256 * pos_dim  (classic)
                                                           or D = 256 * r         (fourier)
  codec_table                                [V, D]        one κ per vocab id
  input_ids                                  [B, T]
    → gather from codec_table                [B, T, D]
    → Linear(D, d_model)                     [B, T, d_model]
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn

CHAR_DIM = 256  # full byte alphabet (0x00..0xFF); height of the Kronecker grid


def utf8_bytes(text: str) -> List[int]:
    """str → list of byte values in [0, 255], length = UTF-8 byte length L."""
    return list(text.encode("utf-8"))


def utf8_safe_truncate(byte_seq: Sequence[int], max_len: int) -> List[int]:
    """
    Classic-only helper: keep at most `max_len` bytes without splitting a
    multi-byte UTF-8 codepoint (continuation bytes look like 10xxxxxx).

    in:  byte_seq length L
    out: byte list length L' ≤ min(L, max_len)
    """
    if len(byte_seq) <= max_len:
        return list(byte_seq)
    cut = max_len
    # Walk left while sitting on a continuation byte.
    while cut > 0 and (byte_seq[cut] & 0xC0) == 0x80:
        cut -= 1
    return list(byte_seq[:cut])


def z_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Per-vector z-score on the last dim (matches released Kronecker module).

    in/out: [..., D]
    """
    mean = x.mean(dim=-1, keepdim=True)                          # [..., 1]
    std = x.std(dim=-1, keepdim=True).clamp_min(eps)              # [..., 1]
    return (x - mean) / std                                      # [..., D]


class ClassicKronecker:
    """
    Paper / Session 7 codec.

    Marks cells on a fixed 256 × pos_dim grid (byte value × byte position),
    then flattens. Bytes past pos_dim are DROPPED (hard crop).

    κ ∈ R^{D} with D = char_dim * pos_dim  (e.g. 256*32 = 8192)
    """

    def __init__(self, pos_dim: int = 32, char_dim: int = CHAR_DIM):
        self.pos_dim = int(pos_dim)    # # of one-hot position columns (= max bytes kept)
        self.char_dim = int(char_dim)  # always 256 for full byte alphabet

    @property
    def code_dim(self) -> int:
        """D = 256 * pos_dim  — input width of the learned Linear."""
        return self.char_dim * self.pos_dim

    def encode(self, text: str, normalize: bool = True) -> torch.Tensor:
        """
        Encode one token string.

        in:  text (str)
        out: code [D] float32, D = char_dim * pos_dim
        """
        kept = utf8_safe_truncate(utf8_bytes(text), self.pos_dim)  # length L' ≤ pos_dim
        code = torch.zeros(self.code_dim, dtype=torch.float32)     # [D]
        L = len(kept)
        if L == 0:
            return code                                            # [D] all zeros
        scale = L ** -0.5                                          # 1/√L variance norm
        for p, b in enumerate(kept):
            # Kronecker of one-hots → single cell at linearized index:
            #   idx = byte_value * pos_dim + position
            if 0 <= b < self.char_dim:
                code[b * self.pos_dim + p] += scale                # mark one sparse entry
        return z_normalize(code) if normalize else code            # [D]


def _fourier_phi(positions: torch.Tensor, r: int, theta: float) -> torch.Tensor:
    """
    Sinusoidal position features (Vaswani-style), adapted to width r.

    in:  positions [...]          float indices (absolute or relative)
    out: φ         [..., r]

        φ[..., 2i]   = sin(p / θ^{i / half})
        φ[..., 2i+1] = cos(p / θ^{i / half})
    """
    half = max(r // 2, 1)
    i = torch.arange(half, dtype=torch.float32)                    # [half]
    freqs = theta ** (-i / float(half))                            # [half]
    angles = positions.unsqueeze(-1) * freqs                       # [..., half]
    # Interleave sin/cos → [..., 2*half], then trim/pad to exactly r.
    paired = torch.stack([torch.sin(angles), torch.cos(angles)], dim=-1).reshape(
        *positions.shape, 2 * half
    )                                                              # [..., 2*half]
    return paired[..., :r] if paired.shape[-1] >= r else torch.nn.functional.pad(
        paired, (0, r - paired.shape[-1])
    )                                                              # [..., r]


class FourierKronecker:
    """
    Ours (Solution A). Same Kronecker idea, continuous position basis.

    κ = (1/√L) vec( Σ_p  c_{b_p} ⊗ φ(p) )
    D = char_dim * r = 256 * pos_features
    L may be >> r — every byte still contributes (no hard crop).
    """

    def __init__(
        self,
        pos_features: int = 32,
        theta: float = 10000.0,
        position_mode: str = "absolute",
        max_bytes: int = 4096,
        char_dim: int = CHAR_DIM,
    ):
        assert position_mode in ("absolute", "relative")
        self.pos_features = int(pos_features)  # r — width of φ(p); D = 256*r
        self.theta = float(theta)              # Fourier base wavelength
        self.position_mode = position_mode     # "absolute" p=0..L-1 or "relative" p/(L-1)
        self.max_bytes = int(max_bytes)        # soft guardrail only (NOT a Classic window)
        self.char_dim = int(char_dim)

    @property
    def code_dim(self) -> int:
        """D = 256 * r."""
        return self.char_dim * self.pos_features

    def encode(self, text: str, normalize: bool = True) -> torch.Tensor:
        """
        Encode one token string (all bytes up to max_bytes).

        in:  text (str)
        out: code [D] float32, D = char_dim * pos_features

        Implementation: accumulate φ(p) into row b_p of a [256, r] grid,
        then flatten — mathematically identical to Σ c⊗φ, but O(L·r).
        """
        raw = utf8_bytes(text)[: self.max_bytes]                   # length L
        L = len(raw)
        grid = torch.zeros(self.char_dim, self.pos_features, dtype=torch.float32)  # [256, r]
        if L == 0:
            return grid.reshape(-1)                                # [D]
        idx = torch.arange(L, dtype=torch.float32)                 # [L]  positions 0..L-1
        # relative: map byte index onto [0, 1] so short/long words share geometry
        pos = idx / float(max(L - 1, 1)) if self.position_mode == "relative" else idx  # [L]
        phi = _fourier_phi(pos, self.pos_features, self.theta) * (L ** -0.5)            # [L, r]
        for p, b in enumerate(raw):
            if 0 <= b < self.char_dim:
                grid[b] += phi[p]                                  # scatter φ into byte row
        code = grid.reshape(-1)                                    # [D] = [256*r]
        return z_normalize(code) if normalize else code            # [D]


class KroneckerEmbedding(nn.Module):
    """
    Drop-in input path: frozen codec table + trainable projection.

    Buffers (not trained):
      codec_table  [V, D]
    Parameters (trained):
      proj.weight  [d_model, D]   via nn.Linear(D, d_model, bias=False)

    forward:
      input_ids [B, T]  →  gather [B, T, D]  →  Linear  →  [B, T, d_model]
    """

    def __init__(self, codec_table: torch.Tensor, d_model: int):
        super().__init__()
        # codec_table: [V, D] — precomputed κ for every vocab id; never updated
        self.register_buffer("codec_table", codec_table, persistent=True)
        D = codec_table.shape[1]                                   # code width
        self.proj = nn.Linear(D, d_model, bias=False)              # weight [d_model, D]
        # Paper-style init: N(0, 1/√D)
        nn.init.normal_(self.proj.weight, mean=0.0, std=D ** -0.5)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        in:  input_ids   [B, T]           int64 token ids
        mid: codes       [B, T, D]        gathered frozen κ vectors
        out: embeddings  [B, T, d_model]  projected into model width
        """
        codes = nn.functional.embedding(input_ids, self.codec_table)  # [B, T, D]
        return self.proj(codes)                                       # [B, T, d_model]


def build_codec_table(surfaces: Sequence[str], codec, show: bool = True) -> torch.Tensor:
    """
    Encode every vocab surface string once (cached mode).

    in:  surfaces  length V  (one str per vocab id)
         codec.encode(s) → [D]
    out: table     [V, D] float32
    """
    try:
        from tqdm import tqdm

        it = tqdm(surfaces, desc=f"{type(codec).__name__}") if show else surfaces
    except ImportError:
        it = surfaces
    # stack list of [D] → [V, D]
    return torch.stack([codec.encode(s, normalize=True) for s in it], dim=0)
