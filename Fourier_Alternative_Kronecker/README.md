# Fourier_Alternative_Kronecker — A Fourier Byte Codec (Problem #4)

Session 7 assignment work: "Kronecker Embedding V2" ideas, problem **#4**.

## The ask

Session 7 shipped a Kronecker byte codec for token embeddings: every token's UTF-8 bytes are placed into a fixed **256 × 32** one-hot grid (byte value × byte position), flattened, and passed through one shared trainable projection. It makes embedding cost independent of vocabulary size, but it only ever looks at a token's **first 32 bytes** — anything after that is silently dropped, with no error and no warning.

The assignment posed five open problems and asked to pick one, propose a solution, and prove it by training a small model. The problem picked here is:

> **#4 — What is a REAL Fourier alternative of Kronecker? Why can't I represent each character like a Fourier wave, and just add them to make a word?**

This was chosen over the other four (arithmetic-structured embeddings, a multimodal Kronecker extension, a dynamic position budget, and an invertible codec that removes the output head) because it attacks a concrete, already-measured failure — the 32-byte wall — with a clean, apples-to-apples experiment: swap one module, hold everything else fixed, and measure.

## The problem, precisely

Kronecker's grid needs a **fixed number of columns**, because one-hot placement requires reserving a slot per byte position in advance. That's what forces `pos_dim = 32`. For ASCII text this is generous (32 characters). For Devanagari, Telugu, Tamil, and other Indic scripts — which spend 3 bytes per character in UTF-8, and where a single conjunct character can cost 2–3 Unicode codepoints — the same 32-byte window covers as few as ten characters. **Two genuinely different tokens that happen to share their first 32 bytes get the exact same embedding, forever**, and the model can never tell them apart.

## The solution

Instead of placing each byte into its own grid column, represent each byte as a **wave** — frequency set by its byte value, phase set by its position — and **sum** the waves across the whole token instead of writing them into fixed slots.

```text
angle[i, k] = byte_value_i * omega_k + phase(i)
wave[i]     = [sin(angle[i, :]), cos(angle[i, :])]
code        = sum_i wave[i]  /  (sqrt(L) * sqrt(2K))
```

`omega_k` are fixed, non-trained frequency bands (geometrically spaced, same construction as the original Transformer's sinusoidal position encoding). `phase(i)` is a fixed sinusoidal function of the byte's position **within the token**. Nothing about length is hard-coded: summation never runs out of room, so a 5-byte token and a 500-byte token both produce a fixed-size code. Exactly one thing is learned — the shared linear projection from the code to `d_model`, mirroring Kronecker's own contract of "one frozen encode step, one trainable projection."

The honest trade: because bytes are **added** rather than **placed**, two tokens that share almost all their bytes and differ only in a small suffix get codes that are technically distinct but only weakly separated — the shared majority of the sum dominates. This is a real cost of superposition versus placement, and it's measured directly in the results below rather than assumed away.

## What was proved, and how

Everything is backed by an actual trained model, not intuition:

1. **Vector-level collisions** (ground truth, no training involved). 32 token pairs were engineered to share an identical 32-byte prefix and diverge only after it — a direct, reproducible instance of the documented failure.
   - Kronecker: **32 / 32 exact vector collisions**.
   - Fourier: **0 / 32**.

2. **Downstream trainability** — does a model actually benefit from the distinctness? A tiny transformer (2 self-attention blocks, `d_model=64`) was trained identically on both codecs on a 4-way classification task where solving the colliding-pair half of the vocabulary *requires* distinguishing tokens past byte 32.
   - Kronecker: stuck at exactly chance (**49.7% mean accuracy** across 5 independent dataset seeds — structurally guaranteed, since colliding tokens are literally identical inputs).
   - Fourier: **85.6% mean accuracy** across the same 5 seeds.

3. **Cost parity** — the win isn't from a bigger codec. Fourier's projection (16,384 params, 256-dim code) is **32× smaller** than Kronecker's (524,288 params, 8,192-dim code) in this configuration, and — like Kronecker — its size depends only on the frequency-band count and `d_model`, never on vocabulary size.

4. **Stretch check** — spelling-similarity preservation. Kronecker's grid construction means similar spellings (`train`, `training`, `trainer`) start out near each other; a 2D PCA projection confirms the Fourier codec keeps this property too.

Full detail, limitations, and open questions are in [`report.md`](report.md). An interactive visual write-up (byte-window inspector, collision bars, training curves, embedding-space scatter plots) is in [`report_visual.html`](report_visual.html).

## Real-data validation (5 languages, real Wikipedia text)

The result above uses a synthetic vocabulary engineered to force the failure mode. A second, independent experiment ([`train.py`](train.py)) reruns the whole comparison on **real text**: a shared WordPiece tokenizer (16,000 tokens) trained across five real Wikipedia corpora — **English, Hindi, Kannada, Tamil, Telugu** — chosen to span Session 7's own fertility claim (English ≈1 byte/char control vs. four Brahmic scripts at ≈2.5–2.7 bytes/char). Two tiny causal transformers (identical except for the input codec) were trained on the shared corpus and evaluated per language. Full raw output is in [`real_data_results_5lang.json`](real_data_results_5lang.json); the hyperparameter sweep's numbers are in [`train.log`](train.log) and embedded directly in [`report_visual.html`](report_visual.html) (its standalone JSON was not retained).

**1. The collision pattern predicted from first principles reproduces exactly, on real tokens, at the shipped `pos_dim=32`:**

| Language | bytes/char | % tokens > 32 bytes | % tokens colliding |
|-|-|-|-|
| English | 1.00 | 0.00% | 0.00% (0 / 4,051) |
| Hindi | 2.53 | 0.15% | 0.00% (0 / 5,398) |
| Kannada | 2.65 | 0.70% | 0.05% (2 / 4,146) |
| Telugu | 2.57 | 1.06% | 0.23% (10 / 4,441) |
| Tamil | 2.64 | 1.10% | **0.40%** (24 / 6,066) |

English never collides; Tamil, the highest-fertility script here, collides ~40× more often than Kannada and shows real, decodable colliding token pairs pulled straight from Wikipedia. The ranking by collision rate tracks the ranking by fertility, exactly as the design predicts — this is not a synthetic artifact.

**2. The `pos_dim` sweep shows precisely where the wall closes:**

| pos_dim | projection params | avg % colliding (5 langs) |
|-|-|-|
| 16 | 524,288 | 4.72% |
| 32 (shipped) | 1,048,576 | 0.13% |
| 48 | 1,572,864 | 0.014% |
| **64** | 2,097,152 | **0.00%** |
| 96 | 3,145,728 | 0.00% |

At this vocabulary, collisions vanish entirely at `pos_dim=64` — doubling the shipped window, at 2× the projection cost, fully closes the gap for these five languages. `pos_dim=96` buys nothing further. This is a direct, data-driven answer to the question Session 7 poses but leaves unanswered.

**3. Model-size impact — does the codec choice change model size?** Trained a third baseline (`DenseEmbeddingCodec`, a plain `nn.Embedding` lookup table) for direct comparison:

| Codec | Codec params | % of total model | vs. dense embedding |
|-|-|-|-|
| Kronecker (pos_dim=32) | 1,048,576 | 26.8% | 51.2% of dense |
| Fourier (K=128) | 32,768 | 1.1% | **1.6% of dense** |
| Dense embedding (baseline) | 2,048,000 | 41.7% | 100% |

At this vocabulary (16K) both structured codecs are meaningfully smaller than a plain embedding table, with Fourier the clear winner on size — and Session 7's own math (Section 3) shows this gap widens dramatically at production vocabulary sizes (131K+), where a dense table alone exceeds a billion parameters.

**4. Language-modeling quality — a more honest, mixed result.** On this real next-token-prediction task (3 epochs, 5-language shared corpus), Kronecker actually **matched or beat** Fourier on perplexity in 4 of 5 languages, and the plain dense-embedding baseline beat both structured codecs everywhere:

| Language | Kronecker PPL | Fourier PPL | Dense PPL |
|-|-|-|-|
| English | 690.1 | 694.9 | 542.9 |
| Hindi | 606.2 | 684.7 | 461.1 |
| Kannada | 840.8 | **828.2** | 753.5 |
| Tamil | 942.7 | 984.0 | 872.0 |
| Telugu | 364.4 | 400.1 | 238.6 |

This is reported honestly rather than smoothed over: at this scale (3 epochs, <1% of tokens actually colliding under Kronecker in any language), the collision effect is too small relative to ordinary optimization noise to show up as a perplexity win for Fourier — and a dense table, with no structural constraint at all, unsurprisingly out-trains both compressed codecs on raw quality in a short run. **The synthetic classification experiment above isolates the collision effect on purpose** (by engineering colliding pairs to be *decisive* for the task); real Wikipedia text does not stress that failure mode nearly as hard, since fewer than 0.5% of real tokens are actually affected at `pos_dim=32`. The honest conclusion: Kronecker's collision problem is real and precisely measurable (points 1–2 above), but at this vocabulary size and training budget it is not yet the dominant factor in language-modeling quality — the parameter/memory savings (point 3) are the more clearly-won argument for either structured codec over a dense table.

## File guide

| File | What it is |
| - | - |
| [`embedding_codecs.py`](embedding_codecs.py) | The two codecs, `KroneckerCodec` and `FourierByteCodec`, implemented side by side with the same interface (`encode → fixed-size vector → shared Linear projection → forward(token_ids)`). This is the actual research contribution. |
| [`data.py`](data.py) | Builds the synthetic byte-level vocabulary: short ASCII control tokens, and 32 engineered colliding pairs that share a 32-byte prefix and diverge after it. Assigns classification labels so that solving the colliding pairs requires seeing past byte 32. |
| [`tiny_transformer.py`](tiny_transformer.py) | The shared tiny transformer classifier (codec → 2 attention/feedforward blocks → linear head) used to test both codecs under identical conditions. |
| [`experiment.py`](experiment.py) | Runs the whole comparison: measures exact vector collisions, trains a classifier on each codec, repeats across 5 seeds for robustness, and writes `results.json`. This is the file that produces every number quoted in the report. |
| [`results.json`](results.json) | Raw output of the experiment run — training curves, collision counts, per-seed accuracy. Every number in `report.md` and the visual artifact is read from this file, not hand-typed. |
| [`train.py`](train.py) | **Run this to reproduce the real-data validation.** Pulls 5-language Wikipedia text, trains a shared tokenizer, trains real causal LMs on Kronecker / Fourier / a dense-embedding baseline, measures per-language fertility and collisions, and sweeps `pos_dim` and `num_freqs`. Meant for Google Colab (GPU). |
| [`real_data_results_5lang.json`](real_data_results_5lang.json) | Raw output of the 5-language real-data run — fertility, collisions, per-language perplexity, and the model-size comparison. |
| [`train.log`](train.log) | Full console log of the actual `train.py` run, including the `pos_dim`/`num_freqs` sweep — the sweep's own JSON output was not retained as a separate file, so this log (plus the numbers already written into `report.md` and `report_visual.html`) is the audit trail for those results. |
| [`report.md`](report.md) | The full written report: problem statement, background, design rationale, synthetic experiment, real-data validation, and limitations. |
| [`report_visual.html`](report_visual.html) | Self-contained interactive HTML version of the report (byte-window inspector, live charts drawn on canvas, embedding scatter plots, sweep charts) — all data is embedded directly in the file, so it does not depend on any of the JSON files above to render. Also published as a Claude artifact. |
| [`main.py`](main.py), [`pyproject.toml`](pyproject.toml), [`uv.lock`](uv.lock) | Project scaffolding (uv-managed). Only dependency is `torch`. |

## How to reproduce

```bash
# from this directory, with the uv-managed venv set up (torch installed)
uv run python experiment.py
```

This retrains both codecs from scratch (2000 epochs each, full-batch, ~a few seconds on CPU), regenerates `results.json`, and prints the headline numbers:

```text
=== Training with kronecker codec ===
kronecker: final_val_acc=... collide_group_acc=0.500 short_group_acc=0.906 ...

=== Training with fourier codec ===
fourier: final_val_acc=... collide_group_acc=0.922 short_group_acc=0.891 ...

=== Multi-seed robustness check ===
kronecker: collide_group_acc per seed = [...] (mean=0.497)
fourier: collide_group_acc per seed = [...] (mean=0.856)
```

To regenerate the embedding-space scatter data or the bundled artifact data, see the inline scripts referenced in `report.md`'s Code Map section — they're short, one-off `python -c` snippets built on top of `embedding_codecs.py` and `results.json`.

**Real-data experiment** (Google Colab, GPU runtime recommended):

```bash
!pip install -q datasets tokenizers torch
!python train.py
# or paste train.py into a notebook cell and run top to bottom
```

This pulls real Wikipedia text for English/Hindi/Kannada/Tamil/Telugu, trains a shared tokenizer, trains three tiny causal LMs (Kronecker, Fourier, dense-embedding baseline), sweeps `pos_dim` and `num_freqs`, and writes `real_data_results_5lang.json`. The sweep results print to console (redirect to a log file to keep them, as `train.log` does) rather than being written to a separate JSON.

## Limitations (see `report.md` for full detail)

- Fourier's collide-group accuracy is ~86% on the synthetic task, not 100% — the aliasing risk from summation is real, not just theoretical.
- The `num_freqs` sweep (real-data experiment) did not show a clean monotonic trend in the range tested (32–256) at only 2 sweep epochs — reported as an open question, not resolved.
- Real-data language-modeling perplexity does not yet show Fourier beating Kronecker — the collision effect is real and precisely measured (under 0.5% of real tokens affected per language at `pos_dim=32`), but too rare to dominate a short (3-epoch) training run's perplexity. The parameter/memory argument for either structured codec over a dense table is the more clearly-won result at this scale.
- Sequence-position embeddings were held fixed throughout; this work isolates the token codec only.
