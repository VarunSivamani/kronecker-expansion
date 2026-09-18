# Kronecker Embedding V2: A Fourier Byte Codec

## Problem chosen

Session 7, Assignment problem **#4**:

> What is a REAL Fourier alternative of Kronecker? Why can't I represent each character like a Fourier wave, and just add them to make a word?

Of the five posed problems, this one was chosen because it attacks a concrete, already-measured flaw in the shipped Kronecker codec rather than a speculative capability, it has a clean apples-to-apples experiment (swap one module, hold everything else fixed), and it connects directly to material the course itself previews (Section 12's sinusoidal / rotary position families) rather than requiring a new theory of representation.

## Background: what's broken in Kronecker

The released Kronecker codec (Session 7, Sections 7-8) represents a token by:

1. Reading its UTF-8 bytes.
2. Marking a fixed **256 (byte value) × 32 (byte position)** one-hot grid, one cell per byte.
3. Flattening to an 8,192-dim fixed vector and passing it through one shared trainable `Linear(8192, d_model)`.

This makes embedding cost independent of vocabulary size — a genuine win. But the grid has only **32 columns**, so only the first 32 bytes of any token are ever seen. For ASCII this is generous (32 characters). For Devanagari, Telugu, Tamil etc. (3 bytes/char in UTF-8, and conjuncts costing 2-3 codepoints each), the same window holds as few as ten characters. **Two tokens that agree on their first 32 bytes get the identical embedding, forever, with no error and no warning.** This is the "32-byte wall."

## Design: the Fourier byte codec

Instead of placing each byte into its own grid column (which is what forces a fixed number of columns), represent each byte as a wave and **sum** the waves. Summation does not require reserving a slot per position, so token length becomes unbounded.

For a token's byte sequence `b_1 ... b_L` (no crop — `L` is arbitrary):

```
angle[i, k] = b_i * omega_k + phase(i)
wave[i]     = [sin(angle[i, :]), cos(angle[i, :])]     # 2K-dim
code        = sum_i wave[i]  /  (sqrt(L) * sqrt(2K))
```

- `omega_k`, `k = 0..K-1` are **fixed, non-trained** geometrically-spaced frequency bands (the same construction as the original Transformer's sinusoidal position encoding), spreading byte values 0-255 across the frequency spectrum.
- `phase(i)` is a **fixed, non-trained** sinusoidal function of the byte's position *within the token* — position enters as a **phase shift added inside the wave**, not as a **grid column selecting which slot gets written**. This is the mechanical change that removes the length limit.
- The `1/sqrt(L)` term is the standard scale-normalization for a sum of `L` roughly-independent unit-scale terms (same role as Kronecker's own `1/sqrt(L)`, see Section 7). The extra `1/sqrt(2K)` keeps the vector's expected norm at a sane, comparable scale for the shared projection.
- Crucially, no final L2-normalize is applied. An early version of this codec included one (mirroring Kronecker's own final normalize) and it silently erased the exact signal being tested: for two long tokens that share most of their bytes, L2-normalizing rescales away the small magnitude difference contributed by the differing suffix. Kronecker's grid doesn't have this problem because two colliding tokens are identical *before* normalization too — there's nothing to preserve. The fix was to drop the vector-level normalize and keep only the scalar energy normalization.
- Exactly one thing is learned: the shared `Linear(2K, d_model)` projection — matching Kronecker's contract of "one frozen encode step, one trainable projection."

This is precisely the "computed vs. stored" trade the session's own Section 12 previews for position embeddings, applied one level earlier, to token identity instead of sequence position: a stored signal (Kronecker's grid) is maximally expressive within its trained window and silently useless past it; a computed signal (the wave sum) is defined for every length by construction, at the cost of relying on superposition rather than dedicated storage per position.

## Why this should work — and the real risk

The claim to test: **any two tokens that differ anywhere in their bytes get distinguishable codes, regardless of length**, because the differing bytes contribute their own wave terms to the sum, which are never overwritten or dropped like columns 33+ are in Kronecker's grid.

The honest risk (this is the aliasing/interference question flagged in the design phase, not swept aside): because bytes are **added**, not concatenated, two tokens that share almost all their bytes and differ only in a small suffix produce codes that are *technically distinct but only weakly separated* — the shared majority of the sum dominates, and the differing minority is a small perturbation on top. This is a real, measurable cost of summation versus placement, and it shows up directly in the results below as a harder (but still solvable) discrimination problem, not a free lunch.

## Experiment

**Codecs.** `KroneckerCodec` (256×32 grid, pos_dim=32, exactly as specified) and `FourierByteCodec` (K=128 frequency bands → 256-dim code) implemented side by side in [`embedding_codecs.py`](embedding_codecs.py), sharing the same `encode → codes buffer → Linear projection → forward(token_ids)` interface.

**Model.** A tiny transformer classifier ([`tiny_transformer.py`](tiny_transformer.py)): codec → 2 self-attention + feedforward blocks (`d_model=64`, 4 heads) → linear classification head. The codec is the only thing that varies between runs.

**Data.** A synthetic byte-level vocabulary ([`data.py`](data.py)) engineered to reproduce the documented failure directly:

- 64 short ASCII tokens (2-8 bytes) — control group, sanity-checks that training works at all.
- 32 **colliding pairs**: each pair shares an identical, randomly-generated **32-byte prefix**, then diverges over a further 12 bytes (44 bytes total, well past Kronecker's window). Each side of a pair is assigned a *different* class label, so solving the task for these 64 tokens **requires** distinguishing tokens whose first 32 bytes are identical — exactly the case the 32-byte wall is documented to break.

**Task.** 4-way token classification (2 classes for the short group, 2 more for "which side of the colliding pair"). Trained full-batch, Adam + cosine LR decay + gradient clipping, 2000 epochs, 5 random data seeds.

## Results

All numbers below are read directly from `results.json`, produced by running `uv run python experiment.py` (reproducible; see Code Map).

**1. Vector-level collisions (ground truth, no training involved).**

| Codec | Key collisions (what the codec can see) | Exact vector collisions |
|---|---|---|
| Kronecker (pos_dim=32) | 32 / 32 engineered pairs | **32** |
| Fourier | 0 / 32 | **0** |

Every one of the 32 engineered colliding pairs produces bit-for-bit identical Kronecker embeddings — a direct, reproduced instance of the documented 32-byte-wall failure. The Fourier codec produces zero exact collisions on the same vocabulary: the moment two tokens' bytes differ anywhere, the wave sum differs.

**2. Downstream trainability — can the model use that difference?**

Trained on the full 4-way classification task, accuracy restricted to just the 32 colliding-pair tokens (`collide_group_acc`), across 5 independent dataset seeds:

| Seed | Kronecker | Fourier |
|---|---|---|
| 0 | 0.500 | 0.922 |
| 1 | 0.484 | 0.844 |
| 2 | 0.500 | 0.844 |
| 3 | 0.500 | 0.859 |
| 4 | 0.500 | 0.875 |
| **Mean** | **0.497** (= chance, 2 classes) | **0.856** |

Kronecker sits at exactly chance on every seed — structurally guaranteed, since colliding tokens are literally indistinguishable inputs to the model; no amount of training can move this number. Fourier recovers **~86%** accuracy on the same task, confirming the vector-level distinctness from measurement (1) is not just numerically-nonzero noise but an actually learnable signal.

Accuracy on the short (non-colliding) control group is comparable for both codecs (Kronecker ~0.91, Fourier ~0.88 in the seed-0 run) — the Fourier codec is not winning by being a generally better codec, only by not having the specific wall Kronecker has.

**3. Parameter / compute cost — is the win free?**

| Codec | Code dim (pre-projection) | Projection params | Depends on vocab size? |
|---|---|---|---|
| Kronecker | 8,192 | 524,288 | No |
| Fourier | 256 | 16,384 | No |

The Fourier codec's projection is **32x smaller** than Kronecker's in this configuration, and — like Kronecker — its cost is a function of `K` and `d_model` only, with **no dependence on vocabulary size**, preserving the core property that made Kronecker attractive in Section 7. This is not the point of the comparison (K is a free hyperparameter, not tuned for parity), but it rules out "the win came from a bigger codec."

**4. Stretch check — is spelling-similarity preserved?**

Both codecs place related short words (`train`, `training`, `trainer`, `trained`) closer to each other than to unrelated words (`banana`, `zebra`); the Fourier codec's separation ratio (related-pair distance vs. unrelated-pair distance) is comparable to or better than Kronecker's on the small sample checked in [`experiment.py`](experiment.py)'s companion check. The property Section 7 highlighted as a benefit of Kronecker (similar spellings start out similar) is not lost by switching to a Fourier code.

## Real-data validation: 5 languages, real Wikipedia text, hyperparameter sweeps

The experiment above uses a synthetic vocabulary engineered so that solving the classification task *requires* seeing past byte 32 — a clean way to isolate the collision effect, but not evidence that the effect occurs at a meaningful rate in real text. A second, independent experiment ([`train.py`](train.py)) closes that gap: it trains a real shared WordPiece tokenizer (16,000 tokens) across five real Wikipedia corpora, trains real causal language models, and sweeps each codec's own capacity knob. Raw output: [`real_data_results_5lang.json`](real_data_results_5lang.json); the hyperparameter sweep's numbers are preserved in [`train.log`](train.log) and embedded in [`report_visual.html`](report_visual.html) rather than in a standalone JSON.

**Languages, chosen to span the session's own fertility claim (Section 4):** English (Latin, ≈1 byte/char, the low-fertility control) against four Brahmic scripts that are *not* interchangeable despite all being "≈2.5-2.7 bytes/char" — Hindi, Kannada, Tamil, Telugu differ in how heavily they form multi-codepoint conjuncts.

**Finding 1 — the predicted collision pattern reproduces exactly, on real tokens, at the shipped `pos_dim=32`:**

| Language | bytes/char | % tokens > 32 bytes | % tokens colliding |
|---|---|---|---|
| English | 1.00 | 0.00% | 0.00% (0 / 4,051) |
| Hindi | 2.53 | 0.15% | 0.00% (0 / 5,398) |
| Kannada | 2.65 | 0.70% | 0.05% (2 / 4,146) |
| Telugu | 2.57 | 1.06% | 0.23% (10 / 4,441) |
| Tamil | 2.64 | 1.10% | **0.40%** (24 / 6,066) |

Zero collisions for English at any point; a monotonic-ish rise in collision rate that tracks fertility, topping out at Tamil. This is the real-data confirmation that the failure mode from the synthetic experiment is not an artifact of engineering it to exist — it happens on its own, at a rate proportional to script fertility, exactly as the mechanism predicts.

**Finding 2 — the `pos_dim` sweep shows precisely where the wall closes, on this vocabulary:**

| pos_dim | projection params | avg % colliding (5 languages) |
|---|---|---|
| 16 | 524,288 | 4.72% |
| 32 (shipped) | 1,048,576 | 0.13% |
| 48 | 1,572,864 | 0.014% |
| **64** | 2,097,152 | **0.00%** |
| 96 | 3,145,728 | 0.00% |

Collisions vanish entirely at `pos_dim=64` for this vocabulary, at exactly 2× the shipped projection cost, and `96` buys nothing further. This is the direct, measured answer to the question Session 7 poses without answering: "if the collision count says buy it, buy it" — here, buying to 64 clears it, and going further is waste.

**Finding 3 — a `num_freqs` sweep for Fourier's own capacity knob**, its structural analogue to `pos_dim` (K frequency bands → code dim `2K`):

| K (num_freqs) | code dim | projection params | avg val perplexity (5 langs, 2-epoch sweep) |
|---|---|---|---|
| 32 | 64 | 8,192 | 927.3 |
| 64 | 128 | 16,384 | 963.0 |
| 128 | 256 | 32,768 | 987.8 |
| 256 | 512 | 65,536 | 913.1 |

Perplexity is roughly flat across this range (no clean monotonic trend at only 2 sweep epochs) — unlike Kronecker's `pos_dim` sweep, which has a sharp, interpretable collision-driven signal, Fourier's `num_freqs` doesn't show an obvious capacity bottleneck being relieved in this range. Read cautiously: 2 epochs is a light budget for a sweep, and this is more "no strong effect detected" than "no effect exists."

**Finding 4 — model-size impact, with an actual dense-embedding baseline.** A third arm (`DenseEmbeddingCodec`, a plain `nn.Embedding`) was trained for direct comparison, answering "does the codec choice change model size" against a real reference point rather than an isolated parameter count:

| Codec | Codec params | % of total model | vs. dense embedding |
|---|---|---|---|
| Kronecker (pos_dim=32) | 1,048,576 | 26.8% | 51.2% of dense |
| Fourier (K=128) | 32,768 | 1.1% | **1.6% of dense** |
| Dense embedding (baseline) | 2,048,000 | 41.7% | 100% |

At this vocabulary (16K), both structured codecs are smaller than a dense table, Fourier dramatically so. Section 3's own accounting shows this gap widens sharply at production vocabulary sizes (131K+), where the dense table alone exceeds a billion parameters.

**Finding 5 — language-modeling quality is a mixed, honestly-reported result, not a clean win.** On real next-token prediction (3 epochs, shared 5-language corpus), Kronecker matched or beat Fourier on perplexity in 4 of 5 languages, and the dense baseline beat both structured codecs everywhere:

| Language | Kronecker PPL | Fourier PPL | Dense PPL |
|---|---|---|---|
| English | 690.1 | 694.9 | 542.9 |
| Hindi | 606.2 | 684.7 | 461.1 |
| Kannada | 840.8 | **828.2** | 753.5 |
| Tamil | 942.7 | 984.0 | 872.0 |
| Telugu | 364.4 | 400.1 | 238.6 |

This does not contradict Findings 1-2 — it contextualizes them. Fewer than 0.5% of tokens in any language actually collide under Kronecker at `pos_dim=32` on this real vocabulary, so the collision effect is real but rare, and at only 3 epochs its contribution to perplexity is smaller than ordinary optimization noise between two differently-initialized codecs. The synthetic classification experiment earlier in this report is not invalidated by this — it was designed specifically to make the colliding tokens *decisive* for the task, which isolates the effect; real Wikipedia text does not stress the failure mode nearly that hard. The honest reading: **the collision problem is real and precisely quantifiable (Findings 1-2), the parameter/memory argument for either structured codec over a dense table is clearly won (Finding 4), but the language-modeling-quality argument for Fourier over Kronecker specifically is not yet demonstrated at this scale and training budget** — it would need either a much longer run, a higher-fertility/more-conjunct-heavy corpus, or a task (like the synthetic one) that makes the rare colliding tokens matter disproportionately.

## Limitations and open questions

- **The aliasing risk is real, not hypothetical.** The synthetic task used a 12-byte differing suffix after a 32-byte shared prefix; accuracy is ~86%, not ~100%. A construction with a *shorter* differing suffix relative to token length would push the differing signal's share of the total sum's energy down further — this boundary was not mapped precisely.
- **The `num_freqs` sweep (Finding 3) did not find a clean signal** in the range tested, unlike the sharp, interpretable `pos_dim` sweep for Kronecker. This is reported as an open question, not resolved — a longer sweep budget or a wider K range might reveal a trend the 2-epoch budget here couldn't detect.
- **Real-data perplexity does not yet show Fourier winning over Kronecker (Finding 5).** The collision effect exists and is precisely measured, but is too rare in real text (under 0.5% of tokens per language) to dominate a short training run's perplexity. This is the single most important open question the real-data experiment raises: at what corpus fertility, training length, or task structure does the measured collision advantage actually translate into a language-modeling win?
- **Val-set accuracy in the synthetic run is not the headline metric.** The classification labels are token-specific (not a generalizable rule over unseen tokens), so both codecs overfit the small closed vocabulary and validation loss climbs after early epochs — this is expected and is why `collide_group_acc`, measured over the full engineered set, is the metric that actually answers the research question, not held-out generalization.
- **Interaction with position (Section 11-12) is out of scope here.** This report only replaces the *token* codec; the transformer's sequence-position embedding was held constant and identical across both runs by design, per the plan.

## Code map

- [`embedding_codecs.py`](embedding_codecs.py) — `KroneckerCodec` and `FourierByteCodec`, side by side, same interface.
- [`data.py`](data.py) — synthetic vocabulary + collision-engineering + classification labels.
- [`tiny_transformer.py`](tiny_transformer.py) — the shared tiny transformer classifier.
- [`experiment.py`](experiment.py) — trains both codecs, measures collisions, trains classifiers, runs the 5-seed robustness check, writes `results.json`.
- [`results.json`](results.json) — raw output of the synthetic-vocabulary run this report quotes in "Results".
- [`train.py`](train.py) — the real-data validation: pulls 5-language Wikipedia text, trains a shared tokenizer, trains real causal LMs (Kronecker / Fourier / dense-embedding baseline), and sweeps `pos_dim` and `num_freqs`. Meant to run in Google Colab.
- [`real_data_results_5lang.json`](real_data_results_5lang.json) — raw output backing Findings 1, 4, 5 in "Real-data validation".
- [`train.log`](train.log) — full console log of the `train.py` run, backing Findings 2-3 (the `pos_dim` and `num_freqs` sweeps); their numbers were not retained in a standalone JSON, only in this log and embedded in `report_visual.html`.

**Reproduce the synthetic experiment:**

```bash
uv run python experiment.py
```

**Reproduce the real-data experiment:** run `python train.py` (or paste it into a Google Colab notebook cell, GPU runtime recommended), after `pip install -q datasets tokenizers torch`.
