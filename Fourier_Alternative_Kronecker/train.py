"""
Kronecker vs Fourier byte codec -- a 5-language real-data comparison.

Run this in Google Colab (GPU runtime recommended: Runtime > Change runtime
type > T4 GPU). It is self-contained: paste it into one cell, or split it
across cells at the `# %%` markers -- the latter is recommended so you can
inspect the pulled text and the per-language collision report before
committing to the full training run.

WHAT THIS DOES
--------------
Session 7's assignment measured the Kronecker codec's "32-byte wall" on a
*synthetic* engineered vocabulary (see experiment.py in this repo). This
script repeats the comparison -- Kronecker vs. a Fourier wave-sum codec --
on REAL text pulled from Hugging Face, across FIVE languages chosen to span
the exact fertility range Session 7 itself measures:

    English  (Latin,      ~1 byte/char)  -- the control, Kronecker's easy case
    Hindi    (Devanagari, ~3 bytes/char, conjuncts multiply this further)
    Kannada  (Kannada script, ~3 bytes/char, also conjunct-forming)
    Tamil    (Tamil script, ~3 bytes/char)
    Telugu   (Telugu script, ~3 bytes/char, conjunct-forming)

One shared multilingual tokenizer is trained across all five languages (this
mirrors how a real production tokenizer works -- one vocabulary serving many
scripts, exactly the situation Session 7 Section 4 calls a "sovereign
allocation decision"). The comparison is then run three ways:

  1. PER-LANGUAGE FERTILITY. Bytes per character, tokens per character, for
     each language -- reproduces Session 7 Section 4's claim that fertility
     is not uniform across scripts.
  2. PER-LANGUAGE COLLISION RATE under Kronecker's pos_dim=32 window. This is
     the direct, real-data version of the measurement Session 7 Section 8
     asks for, broken out by language rather than pooled -- the pooled
     number hides exactly the effect being studied, since English tokens
     essentially never collide and Indic tokens disproportionately do.
  3. PER-LANGUAGE LANGUAGE-MODELING LOSS/PERPLEXITY. Two tiny causal
     transformers -- identical except for the input codec -- are trained on
     the shared multilingual corpus, then evaluated separately on each
     language's held-out validation text, so a codec that helps Indic
     scripts without hurting English is distinguishable from a codec that
     trades one for the other.
  4. MODEL SIZE IMPACT. A third arm trains the SAME tiny transformer with a
     plain nn.Embedding lookup table in place of the codec -- the thing
     Kronecker/Fourier exist to avoid -- so "does this codec make the model
     bigger" has an actual answer: each codec's parameter count is reported
     both in isolation and as a percentage of the FULL trained model, next
     to what a dense embedding table would have cost at the same
     vocab_size/d_model (Session 7 Section 3's V x D accounting, applied to
     this run's real vocabulary instead of V5's reference numbers).

DEPENDENCIES (run this first in Colab)
---------------------------------------
    !pip install -q datasets tokenizers torch

DATASET
-------
`wikimedia/wikipedia`, streamed (no full dump download), configs:
    20231101.en, 20231101.hi, 20231101.kn, 20231101.ta, 20231101.te
"""

# %% [1] Install (Colab only -- comment out if deps are already present)
# !pip install -q datasets tokenizers torch

# %% [2] Imports
from __future__ import annotations

import json
import math
import random
from collections import defaultdict

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE)

SEED = 0
random.seed(SEED)
torch.manual_seed(SEED)

# %% [3] Pull real text from Hugging Face, 5 languages
#
# Chosen to span Session 7's fertility range: English is the low-fertility
# control (~1 byte/char), the other four are all "3 bytes/char" Brahmic
# scripts but are NOT interchangeable -- Hindi (Devanagari) and Kannada both
# form heavy conjuncts, Tamil has comparatively few conjuncts and a smaller
# effective alphabet, Telugu sits in between. A pooled "Indic" number would
# hide these differences; this script keeps every measurement broken out
# per language for exactly that reason.

LANGUAGES = {
    "english": "20231101.en",
    "hindi": "20231101.hi",
    "kannada": "20231101.kn",
    "tamil": "20231101.ta",
    "telugu": "20231101.te",
}

N_ARTICLES_PER_LANG = 3000   # per language; keep total corpus small but real
MAX_CHARS_PER_ARTICLE = 800  # short articles -> more, shorter documents

def pull_wikipedia_text(lang_config: str, n_articles: int) -> list[str]:
    try:
        ds = load_dataset("wikimedia/wikipedia", lang_config, split="train", streaming=True)
    except Exception as e:
        raise RuntimeError(
            f"Could not load config '{lang_config}' from wikimedia/wikipedia. "
            f"Check available configs at https://huggingface.co/datasets/wikimedia/wikipedia "
            f"(look for a '20231101.<lang-code>' entry) and update LANGUAGES accordingly. "
            f"Original error: {e}"
        ) from e
    texts = []
    for i, row in enumerate(ds):
        if i >= n_articles:
            break
        text = row["text"].strip().replace("\n", " ")
        if len(text) > 50:
            texts.append(text[:MAX_CHARS_PER_ARTICLE])
    return texts

texts_by_lang: dict[str, list[str]] = {}
for lang_name, config in LANGUAGES.items():
    print(f"Pulling {lang_name} Wikipedia ({config})...")
    texts_by_lang[lang_name] = pull_wikipedia_text(config, N_ARTICLES_PER_LANG)
    print(f"  {len(texts_by_lang[lang_name])} articles, "
          f"sample: {texts_by_lang[lang_name][0][:80]!r}")

# %% [4] Train ONE shared multilingual tokenizer across all 5 languages
#
# Mirrors real production practice (and Session 7 Section 4's framing): a
# single vocabulary serves every language, so how that vocabulary's fixed
# budget of tokens is spent across scripts is a real, measurable allocation
# decision -- not an implementation detail.

VOCAB_SIZE = 16000  # shared across 5 languages; small but real

tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
trainer = trainers.WordPieceTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
)

def all_texts_iterator():
    for lang_texts in texts_by_lang.values():
        yield from lang_texts

tokenizer.train_from_iterator(all_texts_iterator(), trainer=trainer)

vocab_dict = tokenizer.get_vocab()
id_to_token = [None] * len(vocab_dict)
for tok_str, idx in vocab_dict.items():
    id_to_token[idx] = tok_str

vocab_bytes: list[bytes] = [t.encode("utf-8") for t in id_to_token]
VOCAB_SIZE_ACTUAL = len(vocab_bytes)
print(f"\nShared multilingual vocab size: {VOCAB_SIZE_ACTUAL}")

PAD_ID = vocab_dict["[PAD]"]
BOS_ID = vocab_dict["[BOS]"]
EOS_ID = vocab_dict["[EOS]"]

# %% [5] Per-language fertility -- Session 7 Section 4's claim, measured
#
# Fertility = tokens per character. A tokenizer that fragments a script
# into many small tokens both costs more attention compute per document
# (Section 4) AND produces more, and longer-tailed, byte sequences that
# feed into the embedding codecs being compared here.

def fertility_report(texts: list[str], tok: Tokenizer) -> dict:
    total_chars, total_tokens, total_bytes = 0, 0, 0
    for text in texts:
        ids = tok.encode(text).ids
        total_chars += len(text)
        total_tokens += len(ids)
        total_bytes += len(text.encode("utf-8"))
    return dict(
        n_docs=len(texts),
        total_chars=total_chars,
        total_tokens=total_tokens,
        bytes_per_char=total_bytes / max(total_chars, 1),
        tokens_per_char=total_tokens / max(total_chars, 1),
        chars_per_token=total_chars / max(total_tokens, 1),
    )

print("\n=== Per-language fertility (shared vocab, real text) ===")
fertility_by_lang = {}
for lang_name, texts in texts_by_lang.items():
    fertility_by_lang[lang_name] = fertility_report(texts, tokenizer)
    f = fertility_by_lang[lang_name]
    print(f"  {lang_name:<10} bytes/char={f['bytes_per_char']:.2f}  "
          f"tokens/char={f['tokens_per_char']:.3f}  "
          f"chars/token={f['chars_per_token']:.2f}")

# %% [6] Per-language collision report under Kronecker's 32-byte window
#
# This is the direct real-data version of the measurement Session 7 Section
# 8 asks for. Crucially: this is measured PER LANGUAGE, not pooled, because
# a single pooled "% colliding" number is dominated by whichever language
# has the most vocabulary entries and can make a severe problem in one
# script look mild overall.
#
# Method: for each language, tokenize its held-out text, collect the SET of
# distinct token ids actually used by that language, then check how many of
# those tokens' byte sequences collide under the pos_dim=32 window -- against
# the collision table built from the FULL shared vocabulary (a token used by
# Kannada can collide with a token only ever used by Hindi, and that is a
# real production collision, not a measurement artifact).

POS_DIM = 32  # the shipped Kronecker window

def collision_key_table(vocab: list[bytes], pos_dim: int) -> dict[bytes, list[int]]:
    """32-byte prefix -> list of vocab ids sharing that prefix."""
    table = defaultdict(list)
    for idx, b in enumerate(vocab):
        table[b[:pos_dim]].append(idx)
    return table

collision_table = collision_key_table(vocab_bytes, POS_DIM)

def per_language_collision_report(texts: list[str], tok: Tokenizer,
                                   vocab: list[bytes], table: dict, pos_dim: int) -> dict:
    used_ids = set()
    for text in texts:
        used_ids.update(tok.encode(text).ids)

    long_tokens = 0
    colliding_tokens = 0
    for idx in used_ids:
        b = vocab[idx]
        if len(b) > pos_dim:
            long_tokens += 1
        key = b[:pos_dim]
        if len(table[key]) > 1:
            colliding_tokens += 1

    n = max(len(used_ids), 1)
    return dict(
        distinct_tokens_used=len(used_ids),
        tokens_longer_than_window=long_tokens,
        pct_longer_than_window=100 * long_tokens / n,
        colliding_tokens=colliding_tokens,
        pct_colliding=100 * colliding_tokens / n,
    )

print("\n=== Per-language collision report (pos_dim=32, shared vocab) ===")
collision_by_lang = {}
for lang_name, texts in texts_by_lang.items():
    collision_by_lang[lang_name] = per_language_collision_report(
        texts, tokenizer, vocab_bytes, collision_table, POS_DIM
    )
    c = collision_by_lang[lang_name]
    print(f"  {lang_name:<10} distinct_tokens={c['distinct_tokens_used']:<6} "
          f"longer_than_32B={c['pct_longer_than_window']:.2f}%  "
          f"colliding={c['pct_colliding']:.2f}%")

print("\n--> Compare this table to the fertility table above: languages with")
print("    higher bytes/char should show a higher % colliding, if the 32-byte")
print("    wall is really a script-fertility effect and not noise.")

# %% [7] Both codecs, unchanged from the repo's embedding_codecs.py
#
# Self-contained copy for Colab. If you mount/clone the repo instead:
#   from embedding_codecs import KroneckerCodec, FourierByteCodec

class KroneckerCodec(nn.Module):
    def __init__(self, vocab: list[bytes], d_model: int, pos_dim: int = 32):
        super().__init__()
        self.vocab = vocab
        self.pos_dim = pos_dim
        self.char_dim = 256
        self.code_dim = self.char_dim * self.pos_dim
        codes = torch.stack([self._encode_one(b) for b in vocab])
        self.register_buffer("codes", codes)
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
        codes = self.codes[token_ids]
        return self.projection(codes)


class FourierByteCodec(nn.Module):
    def __init__(self, vocab: list[bytes], d_model: int, num_freqs: int = 128,
                 max_len_for_phase: int = 512):
        super().__init__()
        self.vocab = vocab
        self.num_freqs = num_freqs
        self.code_dim = 2 * num_freqs
        k = torch.arange(num_freqs, dtype=torch.float32)
        self.register_buffer("omega", 1.0 / (10000 ** (k / max(num_freqs - 1, 1))))
        pos = torch.arange(max_len_for_phase, dtype=torch.float32)
        self.register_buffer("phase_table", pos * (2 * math.pi / max_len_for_phase))
        codes = torch.stack([self._encode_one(b) for b in vocab])
        self.register_buffer("codes", codes)
        self.projection = nn.Linear(self.code_dim, d_model, bias=False)

    def _encode_one(self, byte_seq: bytes) -> torch.Tensor:
        length = len(byte_seq)
        if length == 0:
            return torch.zeros(self.code_dim)
        values = torch.tensor(list(byte_seq), dtype=torch.float32)
        phases = self.phase_table[:length]
        angle = values.unsqueeze(1) * self.omega.unsqueeze(0) + phases.unsqueeze(1)
        wave = torch.cat([torch.sin(angle), torch.cos(angle)], dim=1)
        code = wave.sum(dim=0) / (math.sqrt(length) * math.sqrt(self.code_dim))
        return code

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        codes = self.codes[token_ids]
        return self.projection(codes)


class DenseEmbeddingCodec(nn.Module):
    """The baseline neither Kronecker nor Fourier is trying to beat on
    accuracy -- a plain nn.Embedding lookup table, one learned row per
    vocabulary entry. This is what Session 7 Section 3 sizes as `V x D`
    and calls "a little over a billion parameters" at V5's scale. Included
    here so every parameter-count comparison below has a real reference
    point, not just "smaller than the other codec" -- the question that
    actually matters is "smaller than what a dense table would have cost
    at THIS vocab size," since that's the thing being avoided.
    """

    def __init__(self, vocab: list[bytes], d_model: int):
        super().__init__()
        self.vocab = vocab
        self.code_dim = d_model  # no separate code space -- the table IS the embedding
        self.embedding = nn.Embedding(len(vocab), d_model)
        # Exposed so the shared param-counting code below (which reads
        # `codec.projection`) works unchanged for this codec too.
        self.projection = self.embedding

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids)


def count_trainable_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def model_size_breakdown(model: nn.Module, vocab_size: int, d_model: int) -> dict:
    """How much of the TOTAL model each part accounts for -- this is the
    number that answers "does swapping in Kronecker/Fourier change the
    model's size", not just "how big is the codec in isolation".

    Also reports what a dense nn.Embedding lookup table would have cost at
    this exact vocab_size/d_model, so the codec's saving (or cost) is
    measured against the thing it's actually replacing, per Session 7
    Section 3's own accounting (V x D parameters, plus the AdamW 16
    bytes/parameter training-memory multiplier from that same section).
    """
    codec_params = sum(p.numel() for p in model.codec.parameters() if p.requires_grad)
    pos_params = sum(p.numel() for p in model.pos_embedding.parameters())
    block_params = sum(p.numel() for p in model.blocks.parameters())
    norm_params = sum(p.numel() for p in model.norm_out.parameters())
    head_params = sum(p.numel() for p in model.lm_head.parameters())
    total = codec_params + pos_params + block_params + norm_params + head_params

    dense_equivalent_params = vocab_size * d_model  # what nn.Embedding(vocab_size, d_model) alone would cost
    bytes_per_param_adamw = 16  # bf16 weight+grad (4B) + fp32 master+2 moments (12B), Session 7 Section 3

    return dict(
        codec_params=codec_params,
        pos_embedding_params=pos_params,
        transformer_block_params=block_params,
        final_norm_params=norm_params,
        lm_head_params=head_params,
        total_model_params=total,
        codec_pct_of_total=100 * codec_params / total,
        dense_embedding_equivalent_params=dense_equivalent_params,
        codec_vs_dense_ratio=codec_params / dense_equivalent_params,
        codec_params_saved_vs_dense=dense_equivalent_params - codec_params,
        dense_equivalent_training_memory_mb=dense_equivalent_params * bytes_per_param_adamw / 1e6,
        codec_training_memory_mb=codec_params * bytes_per_param_adamw / 1e6,
    )

# %% [8] A real tiny causal transformer (multi-token sequences, next-token loss)
#
# Genuine sequence model: causal self-attention, learned absolute position
# embeddings (identical across both runs -- only the token codec varies),
# untied LM head (per Session 7's V5 decision).

SEQ_LEN = 128
D_MODEL = 128
N_LAYERS = 4
N_HEADS = 4
D_FF = 512

class CausalBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, causal_mask):
        attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask, need_weights=False)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class TinyCausalLM(nn.Module):
    def __init__(self, codec: nn.Module, vocab_size: int, d_model: int,
                 seq_len: int, n_layers: int, n_heads: int, d_ff: int):
        super().__init__()
        self.codec = codec
        self.pos_embedding = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList(
            [CausalBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.norm_out = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.seq_len = seq_len

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        B, T = token_ids.shape
        positions = torch.arange(T, device=token_ids.device).unsqueeze(0)
        x = self.codec(token_ids) + self.pos_embedding(positions)
        causal_mask = torch.triu(
            torch.full((T, T), float("-inf"), device=token_ids.device), diagonal=1
        )
        for block in self.blocks:
            x = block(x, causal_mask)
        x = self.norm_out(x)
        return self.lm_head(x)

# %% [9] Build per-language datasets (train + held-out val, per language)
#
# Keeping validation split PER LANGUAGE (not just pooled) is what lets step
# [11] report a separate perplexity per language after shared training.

class LMDataset(Dataset):
    def __init__(self, texts: list[str], tok: Tokenizer, seq_len: int):
        self.examples = []
        for text in texts:
            ids = tok.encode(text).ids
            ids = [BOS_ID] + ids + [EOS_ID]
            for i in range(0, max(len(ids) - seq_len, 1), seq_len):
                chunk = ids[i : i + seq_len + 1]
                if len(chunk) < seq_len + 1:
                    chunk = chunk + [PAD_ID] * (seq_len + 1 - len(chunk))
                self.examples.append(chunk)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        chunk = self.examples[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

VAL_FRACTION = 0.1

train_texts_all = []
val_ds_by_lang: dict[str, LMDataset] = {}
for lang_name, texts in texts_by_lang.items():
    shuffled = texts[:]
    random.Random(SEED).shuffle(shuffled)
    n_val = max(int(len(shuffled) * VAL_FRACTION), 1)
    val_texts, train_texts = shuffled[:n_val], shuffled[n_val:]
    train_texts_all.extend(train_texts)
    val_ds_by_lang[lang_name] = LMDataset(val_texts, tokenizer, SEQ_LEN)

random.Random(SEED).shuffle(train_texts_all)
train_ds = LMDataset(train_texts_all, tokenizer, SEQ_LEN)

print(f"\nShared train examples: {len(train_ds)}")
for lang_name, ds in val_ds_by_lang.items():
    print(f"  val examples [{lang_name}]: {len(ds)}")

BATCH_SIZE = 32
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loaders = {
    lang: DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    for lang, ds in val_ds_by_lang.items()
}

# %% [10] Train both codecs on the SAME shared multilingual corpus
#
# Both models see identical training data, identical architecture, identical
# optimizer settings -- the token codec is the only variable. Evaluation
# happens per language so the comparison in step 11 can show whether a codec
# helps some scripts without hurting others.

EPOCHS = 3
LR = 3e-4

def evaluate_per_language(model: nn.Module, loss_fn) -> dict:
    model.eval()
    results = {}
    with torch.no_grad():
        for lang_name, loader in val_loaders.items():
            total_loss, total_tokens = 0.0, 0
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                loss = loss_fn(logits.reshape(-1, VOCAB_SIZE_ACTUAL), y.reshape(-1))
                n_tok = (y != PAD_ID).sum().item()
                total_loss += loss.item() * n_tok
                total_tokens += n_tok
            avg_loss = total_loss / max(total_tokens, 1)
            results[lang_name] = dict(val_loss=avg_loss, val_ppl=math.exp(min(avg_loss, 20)))
    return results


def train_lm(codec_name: str, codec: nn.Module) -> dict:
    torch.manual_seed(SEED)
    model = TinyCausalLM(codec, VOCAB_SIZE_ACTUAL, D_MODEL, SEQ_LEN, N_LAYERS, N_HEADS, D_FF).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    epoch_history = []
    step = 0
    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, VOCAB_SIZE_ACTUAL), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 100 == 0:
                print(f"[{codec_name}] epoch {epoch} step {step} train_loss {loss.item():.4f}")

        per_lang = evaluate_per_language(model, loss_fn)
        epoch_history.append(dict(epoch=epoch, per_language=per_lang))
        summary = "  ".join(f"{lang}={v['val_ppl']:.1f}" for lang, v in per_lang.items())
        print(f"[{codec_name}] epoch {epoch} val_ppl per language: {summary}")

    final_per_lang = epoch_history[-1]["per_language"]
    size_breakdown = model_size_breakdown(model, VOCAB_SIZE_ACTUAL, D_MODEL)
    print(f"[{codec_name}] model size: codec={size_breakdown['codec_params']:,} params "
          f"({size_breakdown['codec_pct_of_total']:.1f}% of {size_breakdown['total_model_params']:,} total), "
          f"vs. dense embedding equivalent={size_breakdown['dense_embedding_equivalent_params']:,} params "
          f"({size_breakdown['codec_vs_dense_ratio']*100:.2f}% of dense)")

    return dict(
        codec=codec_name,
        epoch_history=epoch_history,
        final_per_language=final_per_lang,
        codec_projection_params=sum(p.numel() for p in codec.projection.parameters()),
        codec_code_dim=codec.code_dim,
        total_trainable_params=count_trainable_params(model),
        size_breakdown=size_breakdown,
    )


print("\n=== Training with Kronecker codec (5-language shared corpus) ===")
kron_codec = KroneckerCodec(vocab_bytes, D_MODEL, pos_dim=POS_DIM)
kron_result = train_lm("kronecker", kron_codec)

print("\n=== Training with Fourier codec (5-language shared corpus) ===")
fourier_codec = FourierByteCodec(vocab_bytes, D_MODEL, num_freqs=128)
fourier_result = train_lm("fourier", fourier_codec)

print("\n=== Training with dense nn.Embedding codec (baseline, 5-language shared corpus) ===")
dense_codec = DenseEmbeddingCodec(vocab_bytes, D_MODEL)
dense_result = train_lm("dense_embedding", dense_codec)

# %% [11] Comparative summary table -- fertility, collisions, and LM quality,
#         all broken out per language, side by side
#
# This is the table that actually answers the assignment: does the
# 32-byte wall's severity track fertility per language, and does switching
# to the Fourier codec close the gap where it's worst (Hindi/Kannada/Telugu,
# the heaviest conjunct-formers) without regressing where it was never a
# problem (English)?

print("\n" + "=" * 100)
print("COMPARATIVE SUMMARY: fertility, collisions, and LM perplexity by language")
print("=" * 100)
header = (f"{'Language':<10}{'bytes/char':<12}{'%>32B':<10}{'%colliding':<12}"
          f"{'Kron PPL':<12}{'Fourier PPL':<14}{'PPL delta':<12}")
print(header)
print("-" * len(header))

summary_rows = []
for lang_name in LANGUAGES:
    fert = fertility_by_lang[lang_name]
    coll = collision_by_lang[lang_name]
    kron_ppl = kron_result["final_per_language"][lang_name]["val_ppl"]
    fourier_ppl = fourier_result["final_per_language"][lang_name]["val_ppl"]
    delta = kron_ppl - fourier_ppl
    row = dict(
        language=lang_name,
        bytes_per_char=fert["bytes_per_char"],
        pct_longer_than_32B=coll["pct_longer_than_window"],
        pct_colliding=coll["pct_colliding"],
        kronecker_val_ppl=kron_ppl,
        fourier_val_ppl=fourier_ppl,
        ppl_delta_kron_minus_fourier=delta,
    )
    summary_rows.append(row)
    print(f"{lang_name:<10}{fert['bytes_per_char']:<12.2f}{coll['pct_longer_than_window']:<10.2f}"
          f"{coll['pct_colliding']:<12.2f}{kron_ppl:<12.2f}{fourier_ppl:<14.2f}{delta:<+12.2f}")

print("\nA positive PPL delta means Fourier achieved LOWER (better) perplexity")
print("than Kronecker on that language's held-out text. If the Fourier codec")
print("is doing what the design predicts, this delta should be largest for")
print("the highest-bytes/char, highest-%-colliding languages, and smallest")
print("(near zero, possibly slightly negative) for English.")

# %% [11b] Model-size comparison -- does the codec choice change model size?
#
# The direct answer to "does adding Kronecker/Fourier increase model size":
# compare each codec's share of the TOTAL model against what a plain
# nn.Embedding lookup table (the dense_embedding baseline trained above)
# would have cost at the exact same vocab_size/d_model. This is the
# apples-to-apples version of Session 7 Section 3's parameter accounting,
# run on your actual real-data vocabulary instead of the session's V5
# reference numbers.

print("\n" + "=" * 100)
print("MODEL SIZE COMPARISON: codec params, share of total model, vs. dense embedding")
print("=" * 100)
header = (f"{'Codec':<16}{'Codec params':<16}{'% of total model':<18}"
          f"{'Total model params':<20}{'vs. dense embedding':<20}")
print(header)
print("-" * len(header))

size_rows = []
for result in (kron_result, fourier_result, dense_result):
    sb = result["size_breakdown"]
    row = dict(
        codec=result["codec"],
        codec_params=sb["codec_params"],
        codec_pct_of_total=sb["codec_pct_of_total"],
        total_model_params=sb["total_model_params"],
        dense_embedding_equivalent_params=sb["dense_embedding_equivalent_params"],
        codec_vs_dense_ratio=sb["codec_vs_dense_ratio"],
        codec_params_saved_vs_dense=sb["codec_params_saved_vs_dense"],
        dense_equivalent_training_memory_mb=sb["dense_equivalent_training_memory_mb"],
        codec_training_memory_mb=sb["codec_training_memory_mb"],
    )
    size_rows.append(row)
    vs_dense_str = f"{sb['codec_vs_dense_ratio']*100:.1f}% of dense"
    print(f"{result['codec']:<16}{sb['codec_params']:<16,}{sb['codec_pct_of_total']:<18.2f}"
          f"{sb['total_model_params']:<20,}{vs_dense_str:<20}")

print(f"\nAt this vocab size ({VOCAB_SIZE_ACTUAL}) and d_model ({D_MODEL}), a dense")
print(f"nn.Embedding table alone would cost "
      f"{size_rows[0]['dense_embedding_equivalent_params']:,} parameters and "
      f"~{size_rows[0]['dense_equivalent_training_memory_mb']:.1f} MB of AdamW training")
print("state (Session 7 Section 3's 16-bytes-per-parameter accounting: bf16")
print("weight+grad, fp32 master + 2 optimizer moments). Both Kronecker and Fourier")
print("replace that with a MUCH smaller structured codec -- this table shows exactly")
print("how much smaller, and what fraction of the FULL trained model that codec ends")
print("up being (not just its size in isolation, which is what codec_projection_params")
print("reports above).")

# %% [12] Save everything

with open("real_data_results_5lang.json", "w") as f:
    json.dump(
        dict(
            vocab_size=VOCAB_SIZE_ACTUAL,
            d_model=D_MODEL,
            languages=list(LANGUAGES.keys()),
            fertility_by_language=fertility_by_lang,
            collision_by_language=collision_by_lang,
            kronecker=kron_result,
            fourier=fourier_result,
            dense_embedding=dense_result,
            comparative_summary=summary_rows,
            model_size_comparison=size_rows,
        ),
        f,
        indent=2,
    )
print("\nWrote real_data_results_5lang.json")

# %% [13] Hyperparameter sweep: Kronecker's pos_dim vs. Fourier's num_freqs
#
# Both codecs have exactly one knob that trades capacity against cost:
#   - Kronecker: pos_dim (byte window width). Session 7 ships pos_dim=32 as
#     a default, not a derived value, and explicitly asks readers to
#     re-measure it against the real vocabulary before trusting it.
#   - Fourier: num_freqs (K, frequency-band count -> code dim = 2K). This is
#     the honest analogue: too few bands under-resolves byte values (more
#     aliasing, i.e. more of the "long shared prefix" weak-separation effect
#     already found in the synthetic experiment); too many bands erodes the
#     parameter-savings argument this whole codec exists for.
#
# The sweep answers two questions with actual numbers instead of the single
# point estimate used above:
#   1. At what pos_dim does Kronecker's real-vocabulary collision rate
#      actually reach ~0 for the highest-fertility languages?
#   2. Does Fourier's real-data perplexity improve monotonically with more
#      frequency bands, or does it plateau once bands are no longer the
#      bottleneck (in which case the extra parameters would be wasted)?
#
# Kept on the SAME tiny transformer as the main run above (same D_MODEL,
# N_LAYERS, N_HEADS, D_FF, SEQ_LEN) -- only the codec's own knob changes.
# Uses a smaller epoch budget than the main run so the whole sweep finishes
# in reasonable Colab time; bump SWEEP_EPOCHS up if you have GPU time to
# spare and want tighter numbers.

SWEEP_EPOCHS = 2  # lighter than EPOCHS (main run) -- this is a sweep, not a final number
KRONECKER_POS_DIMS = [16, 32, 48, 64, 96]
FOURIER_NUM_FREQS = [32, 64, 128, 256]


def train_lm_light(codec_name: str, codec: nn.Module, epochs: int) -> dict:
    """Same training loop as train_lm, but with a caller-supplied epoch
    budget and without the per-100-step print spam -- meant for sweeps
    where you run this many times back to back."""
    torch.manual_seed(SEED)
    model = TinyCausalLM(codec, VOCAB_SIZE_ACTUAL, D_MODEL, SEQ_LEN, N_LAYERS, N_HEADS, D_FF).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, VOCAB_SIZE_ACTUAL), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    final_per_lang = evaluate_per_language(model, loss_fn)
    return dict(
        codec=codec_name,
        final_per_language=final_per_lang,
        codec_projection_params=sum(p.numel() for p in codec.projection.parameters()),
        codec_code_dim=codec.code_dim,
        total_trainable_params=count_trainable_params(model),
    )


print("\n" + "=" * 100)
print(f"HYPERPARAMETER SWEEP -- Kronecker pos_dim in {KRONECKER_POS_DIMS}, "
      f"{SWEEP_EPOCHS} epochs each")
print("=" * 100)

kron_sweep_results = []
for pos_dim in KRONECKER_POS_DIMS:
    print(f"\n--- Kronecker pos_dim={pos_dim} ---")
    codec = KroneckerCodec(vocab_bytes, D_MODEL, pos_dim=pos_dim)

    # Real-vocabulary collision rate at THIS window width, per language --
    # this is the direct re-measurement Session 7 Section 8 asks for.
    table_at_this_width = collision_key_table(vocab_bytes, pos_dim)
    collisions_at_this_width = {
        lang: per_language_collision_report(texts, tokenizer, vocab_bytes, table_at_this_width, pos_dim)
        for lang, texts in texts_by_lang.items()
    }

    result = train_lm_light(f"kronecker_pos{pos_dim}", codec, SWEEP_EPOCHS)
    result["pos_dim"] = pos_dim
    result["collision_by_language"] = collisions_at_this_width

    avg_ppl = sum(v["val_ppl"] for v in result["final_per_language"].values()) / len(LANGUAGES)
    avg_collision_pct = sum(v["pct_colliding"] for v in collisions_at_this_width.values()) / len(LANGUAGES)
    print(f"  projection_params={result['codec_projection_params']:<10} "
          f"avg_val_ppl={avg_ppl:.2f}  avg_pct_colliding={avg_collision_pct:.2f}%")
    kron_sweep_results.append(result)

print("\n" + "=" * 100)
print(f"HYPERPARAMETER SWEEP -- Fourier num_freqs in {FOURIER_NUM_FREQS}, "
      f"{SWEEP_EPOCHS} epochs each")
print("=" * 100)

fourier_sweep_results = []
for num_freqs in FOURIER_NUM_FREQS:
    print(f"\n--- Fourier num_freqs={num_freqs} ---")
    codec = FourierByteCodec(vocab_bytes, D_MODEL, num_freqs=num_freqs)
    result = train_lm_light(f"fourier_k{num_freqs}", codec, SWEEP_EPOCHS)
    result["num_freqs"] = num_freqs

    avg_ppl = sum(v["val_ppl"] for v in result["final_per_language"].values()) / len(LANGUAGES)
    print(f"  projection_params={result['codec_projection_params']:<10} avg_val_ppl={avg_ppl:.2f}")
    fourier_sweep_results.append(result)

# %% [14] Sweep summary tables + save

print("\n" + "=" * 100)
print("SWEEP SUMMARY -- Kronecker: pos_dim vs. params vs. real collision rate vs. PPL")
print("=" * 100)
header = f"{'pos_dim':<10}{'proj_params':<14}{'avg %colliding':<18}{'avg val_ppl':<14}"
print(header)
print("-" * len(header))
for r in kron_sweep_results:
    avg_ppl = sum(v["val_ppl"] for v in r["final_per_language"].values()) / len(LANGUAGES)
    avg_coll = sum(v["pct_colliding"] for v in r["collision_by_language"].values()) / len(LANGUAGES)
    print(f"{r['pos_dim']:<10}{r['codec_projection_params']:<14}{avg_coll:<18.2f}{avg_ppl:<14.2f}")

print("\n" + "=" * 100)
print("SWEEP SUMMARY -- Fourier: num_freqs vs. code_dim vs. params vs. PPL")
print("=" * 100)
header = f"{'num_freqs':<12}{'code_dim':<12}{'proj_params':<14}{'avg val_ppl':<14}"
print(header)
print("-" * len(header))
for r in fourier_sweep_results:
    avg_ppl = sum(v["val_ppl"] for v in r["final_per_language"].values()) / len(LANGUAGES)
    print(f"{r['num_freqs']:<12}{r['codec_code_dim']:<12}{r['codec_projection_params']:<14}{avg_ppl:<14.2f}")

with open("hyperparam_sweep_results.json", "w") as f:
    json.dump(
        dict(
            sweep_epochs=SWEEP_EPOCHS,
            kronecker_pos_dim_sweep=kron_sweep_results,
            fourier_num_freqs_sweep=fourier_sweep_results,
        ),
        f,
        indent=2,
    )
print("\nWrote hyperparam_sweep_results.json")
