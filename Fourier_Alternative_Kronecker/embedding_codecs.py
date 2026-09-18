"""Two input-embedding codecs — side-by-side comparison (Fourier_Alternative_Kronecker).

KroneckerCodec  - the Session 7 released design: bytes placed into a fixed
                  256 x 32 one-hot grid (value x position), flattened, then
                  one shared trainable projection. Truncates tokens past 32
                  bytes — the "32-byte wall".

FourierByteCodec - Problem #4 proposal: each byte adds a wave whose
                  frequency depends on the byte value and whose phase depends
                  on its position; waves are *summed* rather than placed in a
                  grid, so token length is unbounded and code size stays
                  independent of vocabulary size, keeping Kronecker's core
                  advantage while dropping the fixed-length truncation.

Both codecs expose the same contract:
  - a fixed-size non-trained `encode(byte_seq) -> Tensor[code_dim]`
  - an nn.Module `forward(token_ids: LongTensor[B,T]) -> Tensor[B,T,d_model]`
    that looks bytes up from a `vocab: list[bytes]` and applies one shared
    trainable Linear projection.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class KroneckerCodec(nn.Module):
    """Byte-value x byte-position one-hot grid, flattened, then projected."""

    def __init__(self, vocab: list[bytes], d_model: int, pos_dim: int = 32):
        super().__init__()
        self.vocab = vocab
        self.pos_dim = pos_dim
        self.char_dim = 256
        self.code_dim = self.char_dim * self.pos_dim

        codes = torch.stack([self._encode_one(b) for b in vocab])
        self.register_buffer("codes", codes)  # [V, code_dim], never trained
        self.projection = nn.Linear(self.code_dim, d_model, bias=False)

    def _encode_one(self, byte_seq: bytes) -> torch.Tensor:
        grid = torch.zeros(self.char_dim, self.pos_dim)
        length = min(len(byte_seq), self.pos_dim)
        for pos in range(length):
            grid[byte_seq[pos], pos] = 1.0
        code = grid.flatten()
        norm = code.norm()
        if norm > 0:
            code = code / norm
        return code

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        codes = self.codes[token_ids]  # [B, T, code_dim]
        return self.projection(codes)

    def collision_key(self, byte_seq: bytes) -> bytes:
        """What the codec actually 'sees' -- used to predict collisions."""
        return byte_seq[: self.pos_dim]


class FourierByteCodec(nn.Module):
    """Each byte is a sum of sin/cos waves; waves accumulate across the
    whole token so length is unbounded and no bytes are ever dropped."""

    def __init__(
        self,
        vocab: list[bytes],
        d_model: int,
        num_freqs: int = 128,
        max_len_for_phase: int = 512,
    ):
        super().__init__()
        self.vocab = vocab
        self.num_freqs = num_freqs
        self.code_dim = 2 * num_freqs

        # Fixed (non-trained) frequency bands spread across byte-value range
        # [0, 255], geometrically spaced like standard sinusoidal encodings
        # so both low- and high-order byte-value differences are captured.
        k = torch.arange(num_freqs, dtype=torch.float32)
        self.register_buffer(
            "omega", 1.0 / (10000 ** (k / max(num_freqs - 1, 1)))
        )
        # Fixed per-position phase offset, standard sinusoidal-PE formula,
        # applied to position within the token (not sequence position).
        pos = torch.arange(max_len_for_phase, dtype=torch.float32)
        self.register_buffer("phase_table", pos * (2 * math.pi / max_len_for_phase))

        codes = torch.stack([self._encode_one(b) for b in vocab])
        self.register_buffer("codes", codes)
        self.projection = nn.Linear(self.code_dim, d_model, bias=False)

    def _encode_one(self, byte_seq: bytes) -> torch.Tensor:
        length = len(byte_seq)
        if length == 0:
            return torch.zeros(self.code_dim)
        values = torch.tensor(list(byte_seq), dtype=torch.float32)  # [L]
        phases = self.phase_table[:length]  # [L]
        # angle[i, k] = byte_value_i * omega_k + phase_i
        angle = values.unsqueeze(1) * self.omega.unsqueeze(0) + phases.unsqueeze(1)
        wave = torch.cat([torch.sin(angle), torch.cos(angle)], dim=1)  # [L, 2K]
        # Divide by sqrt(L) (standard scale-normalization for a sum of
        # roughly-independent unit-scale terms), then by sqrt(2K) so the
        # expected vector norm is O(1) regardless of the frequency-band
        # count -- this keeps the codec's output on a comparable scale to
        # Kronecker's unit-norm codes without discarding the relative
        # magnitude information that separates long, mostly-shared tokens
        # once they diverge (a full L2-normalize would erase exactly that).
        code = wave.sum(dim=0) / (math.sqrt(length) * math.sqrt(self.code_dim))
        return code

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        codes = self.codes[token_ids]
        return self.projection(codes)

    def collision_key(self, byte_seq: bytes) -> bytes:
        """No truncation -- the full byte sequence participates."""
        return byte_seq
