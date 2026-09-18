"""Synthetic byte-level vocabulary and task, engineered to reproduce the
32-byte-wall failure mode from Session 7 Section 8 without needing a real
Indic tokenizer or corpus.

Vocabulary groups:
  - "short":   plain ASCII-like tokens, well under 32 bytes. Control group.
  - "collide": pairs of long tokens (40+ bytes) that are byte-identical for
               their first 32 bytes and differ only after byte 32. Under
               Kronecker's pos_dim=32 window these collide exactly; under
               Fourier's unbounded sum they should not.

Task: token classification. Every colliding pair is assigned to DIFFERENT
classes based on their (post-32-byte) suffix. A codec that cannot see past
byte 32 cannot solve this for the "collide" tokens by construction -- this
is the point being measured, not a hard learning problem.
"""

from __future__ import annotations

import random

import torch


def _short_tokens(n: int, rng: random.Random) -> list[bytes]:
    words = []
    for i in range(n):
        length = rng.randint(2, 8)
        s = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(length))
        words.append(s.encode("ascii"))
    return words


def _colliding_pairs(n_pairs: int, rng: random.Random) -> list[tuple[bytes, bytes]]:
    """Each pair shares an identical 32-byte prefix, then diverges."""
    pairs = []
    for i in range(n_pairs):
        prefix = bytes(rng.randint(0x20, 0x7E) for _ in range(32))
        suffix_a = bytes(rng.randint(0x20, 0x7E) for _ in range(12))
        suffix_b = bytes(rng.randint(0x20, 0x7E) for _ in range(12))
        # guarantee the suffixes actually differ
        while suffix_b == suffix_a:
            suffix_b = bytes(rng.randint(0x20, 0x7E) for _ in range(12))
        pairs.append((prefix + suffix_a, prefix + suffix_b))
    return pairs


class ByteClassificationDataset:
    """token_id -> class_id, where class_id depends on info past byte 32
    for the 'collide' group, and is arbitrary-but-fixed for 'short'."""

    def __init__(self, seed: int = 0, n_short: int = 64, n_pairs: int = 32):
        rng = random.Random(seed)
        self.short_tokens = _short_tokens(n_short, rng)
        self.pairs = _colliding_pairs(n_pairs, rng)

        self.vocab: list[bytes] = []
        self.labels: list[int] = []

        # short tokens: label = hash-based arbitrary binary class (easy task,
        # solvable by any reasonable codec -- sanity check the training loop)
        for tok in self.short_tokens:
            self.vocab.append(tok)
            self.labels.append(sum(tok) % 2)

        # colliding pairs: label = 2 + which side of the pair (0 or 1), so
        # the two tokens in a pair MUST be distinguished to solve this
        num_classes_before = 2
        for a, b in self.pairs:
            self.vocab.append(a)
            self.labels.append(num_classes_before + 0)
            self.vocab.append(b)
            self.labels.append(num_classes_before + 1)

        self.num_classes = num_classes_before + 2
        self.token_ids = list(range(len(self.vocab)))

        # indices of the collide group, for targeted evaluation
        self.collide_start = n_short

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        ids = torch.tensor(self.token_ids, dtype=torch.long)
        labels = torch.tensor(self.labels, dtype=torch.long)
        return ids, labels

    def split(self, train_frac: float = 0.8, seed: int = 0):
        rng = random.Random(seed)
        idx = list(range(len(self.vocab)))
        rng.shuffle(idx)
        cut = int(len(idx) * train_frac)
        train_idx, val_idx = idx[:cut], idx[cut:]
        return train_idx, val_idx
