# kronecker-expansion — Kronecker Embedding V2 (Session 7)

Session 7 posed five open problems on extending the Kronecker byte-embedding codec — the released design that places each token's UTF-8 bytes into a fixed **256 × 32** grid (byte value × byte position), flattened and passed through one shared trainable projection. Each problem is standalone; this repo holds two solutions, each in its own folder with its own code, results and write-up.

| Folder | Problem solved | One-line pitch |
|---|---|---|
| [`Dynamic_Kronecker/`](Dynamic_Kronecker/) | **#3** — the fixed 32-byte position window wastes space on short tokens and hard-crops long ones | Replace the **one-hot position axis** with a continuous Fourier position feature — same byte-value axis, no fixed-length cap |
| [`Fourier_Alternative_Kronecker/`](Fourier_Alternative_Kronecker/) | **#4** — is there a genuine Fourier alternative to Kronecker's one-hot grid entirely | Replace **both axes** (byte value and position) with a sum of sinusoidal waves — the grid becomes a superposition |

Both solutions were verified independently before this README was written: real code, real training runs (Colab logs cross-checked against saved JSON results), no fabricated numbers.

---

## Problem #3 — Dynamic / Fourier-Position Kronecker

**Folder:** [`Dynamic_Kronecker/`](Dynamic_Kronecker/)

### The problem

Kronecker's shipped `pos_dim=32` is a hard, fixed-size window over byte *position*. Two failure modes follow from this:

1. **Hard crop** — any byte at position ≥ 32 never marks a cell. Two different tokens that agree on their first 32 bytes get an identical code forever (a silent collision).
2. **Wasted columns** — a short token like `"a"` leaves most of the 32 position columns as exact zeros, but the model still pays for the full `D = 256 × 32 = 8,192`-dimensional projection regardless.

Raising `pos_dim` to 64 or 128 only moves the wall further out and multiplies the projection's parameter count — it is a bigger fixed window, not a dynamic one.

### The approach

Keep the byte-value axis as one-hot (a `[256, r]` grid, same shape family as the original), but replace the **position** axis's one-hot encoding with a continuous Fourier feature `φ(p) ∈ ℝʳ` (sin/cos pairs, same construction as the original Transformer's sinusoidal positional encoding) that is defined for *every* position `p`, not just `p < pos_dim`:

```text
paper (Classic): κ = (1/√L) · vec Σ_{p < pos_dim}  c_b ⊗ e_p        # hard crop if L > pos_dim
ours  (Fourier): κ = (1/√L) · vec Σ_{p = 0..L-1}    c_b ⊗ φ(p)       # every byte contributes, no crop
```

Code width `D = 256 × r` stays fixed and comparable to Classic's `D = 256 × pos_dim` (so `r=32` and `pos_dim=32` are a matched-budget pair), but token length `L` is no longer capped — every byte in a token contributes to the sum regardless of how long the token is.

### Proof and results

Real experiment: `xlm-roberta-base` tokenizer, real streamed Wikipedia text (English, Hindi, Tamil, Telugu, Kannada; 120K chars/language), a real tiny causal transformer (`d_model=256`, 4 layers, 4 heads) trained for 3 epochs on each of 9 curated codec configurations (Classic `pos_dim ∈ {16,32,64,128}`, Fourier `r ∈ {16,32,64,128}` absolute, plus one relative-position ablation at `r=32`).

**Headline result — Fourier wins at every matched parameter budget:**

| Matched D | Classic val loss | Fourier val loss | Δ (Classic − Fourier) |
|---|---|---|---|
| 4,096 | 7.8531 | 7.7553 | +0.098 |
| 8,192 (paper default) | 7.7931 | **7.6712** | **+0.122** |
| 16,384 | 7.7307 | 7.6428 | +0.088 |
| 32,768 | 7.6782 | **7.5949** | +0.083 |

**Efficiency punchline:** Fourier at `r=32` (D=8,192, 2.10M embedding-path params) beats Classic at `pos_dim=128` (D=32,768, 8.39M params) — a better result at roughly **¼ the parameter cost**.

**Fertility measurements** (tokens-per-word) confirm the premise: Kannada 2.92, Telugu 2.71, Tamil 2.66 vs. English 1.59 — the four Indic languages are 1.7-1.8× more fragmented per word than English, and Classic's crop rate at `pos_dim=32` is correspondingly higher for them (Tamil 0.47% of token instances actually truncated) than for English (0.00%).

### Approach — pros

- **Directly removes the fixed-length cap**, which is the literal ask of problem #3 ("dynamic," "doesn't force us to crop a word").
- **Wins at every tested budget**, not just at the shipped default — the result isn't a cherry-picked single configuration.
- **Fair, apples-to-apples experiment design**: matched `D` between arms isolates the codec change from a parameter-count change; the efficiency comparison (`r=32` vs `pos_dim=128`) additionally isolates the *quality-per-parameter* question.
- **Real multilingual data and a real proxy-scale training run**, not a synthetic toy — the fertility/crop numbers and the LM losses both come from the same real corpora.
- Architecturally minimal change: still exactly one learned object (`Linear(D, d_model)`), same contract as the shipped module.

### Approach — cons / limitations (as the folder's own README states)

- **Proxy scale only** — ~179K tokens, 3 epochs, one seed per configuration. Absolute perplexities are high (small data vs. real vocab size); the authors explicitly say to trust the *deltas between arms*, not the absolute numbers, and that this is not a claim of a full V5-scale pretraining win.
- **Crop rates in this sample are modest** (≤0.47% of token instances) — Fourier still wins at matched D even though hard collisions are rare in this run, so part of the win is likely not purely about crop/collision removal; the mechanism behind the rest of the gain isn't isolated further here.
- Nine curated configurations, not a full grid — reasonable for a proxy study, but the sweep is deliberately narrow (only 4 `pos_dim`/`r` values, one relative-position ablation).

---

## Problem #4 — Fourier Byte Codec (Kronecker Embedding V2)

**Folder:** [`Fourier_Alternative_Kronecker/`](Fourier_Alternative_Kronecker/)

### The problem

The assignment's problem #4 asks for a genuine Fourier alternative to Kronecker's one-hot grid *in general* — not just a fix to the position axis, but the question of whether each byte could be represented as a wave and simply summed to form a word, replacing the placement mechanism entirely.

### The approach

Both the byte-value axis and the position axis are replaced by a wave: each byte contributes a sine/cosine pair whose **frequency** is set by its value and whose **phase** is set by its position, and all bytes' waves are **summed** into one fixed-size vector — no grid, no placement, just superposition:

```text
angle[i,k] = byte_value_i · ω_k + phase(i)
wave[i]    = [sin(angle[i,:]), cos(angle[i,:])]
code       = Σ_i wave[i] / (√L · √2K)
```

This removes the 32-byte window entirely (summation never runs out of "room" the way a fixed grid does) — a structurally different move than problem #3's fix, which keeps one-hot placement on the byte-value axis and only makes the position axis continuous.

### Proof and results

Two separate, escalating experiments:

1. **Synthetic collision experiment** — 32 token pairs engineered to share a 32-byte prefix and diverge only after it. Kronecker: 32/32 exact vector collisions (proven indistinguishable, chance-level accuracy — 49.7% mean over 5 seeds — on a classification task requiring the distinction). Fourier: 0/32 collisions, 85.6% mean accuracy on the same task.
2. **Real 5-language validation** (English, Hindi, Kannada, Tamil, Telugu; shared 16K WordPiece tokenizer; real tiny causal transformers) — the same collision pattern reproduces on real tokens (0% for English, up to 0.40% for Tamil, tracking fertility), and a `pos_dim` sweep shows the real collision rate reaches exactly 0% at `pos_dim=64` on this vocabulary. A model-size comparison against a plain `nn.Embedding` baseline shows both Kronecker (51.2% of dense) and Fourier (1.6% of dense) are meaningfully smaller than a dense table at this vocab size.

### Approach — pros

- **Answers the assignment's actual question** — a real, general Fourier alternative to the whole Kronecker mechanism, not a partial fix.
- **The core claim (removes hard collisions) is proven twice**: once by construction (synthetic, decisive test) and once on real text (reproduces the predicted fertility-correlated collision pattern from real Wikipedia tokens).
- **Dramatically smaller codec** — 1.6% the size of an equivalent dense embedding table at the tested vocab size, and this saving is architecturally independent of vocabulary size (same property that made Kronecker attractive in the first place).
- **Reports a negative result honestly**: on real next-token-prediction perplexity, Kronecker actually matched or beat Fourier in 4 of 5 languages in the real-data run — this is stated plainly in the report rather than hidden, along with the explanation (real collisions are rare enough at this scale that the effect doesn't dominate a short training run's loss).

### Approach — cons / limitations (as the folder's own report states)

- **The real-data language-modeling win is not yet demonstrated** — the parameter/memory argument is clearly won, but the perplexity argument for Fourier over Kronecker specifically needs either a longer run, a higher-fertility corpus, or a task where rare collisions matter disproportionately (like the synthetic experiment).
- **Aliasing is a real, measured cost**: because bytes are summed rather than placed, two tokens sharing a long prefix and differing only in a short suffix are separable but only weakly so (86%, not 100%, accuracy on the hardest synthetic case).
- The Fourier codec's own capacity knob (`num_freqs`) sweep did not show a clean trend in the tested range — an open question, not a resolved one.

---

## File map

### `Dynamic_Kronecker/`

| File | What it is |
|---|---|
| `kronecker.py` | `ClassicKronecker` (paper/shipped codec) and `FourierKronecker` (Fourier-position codec), plus the shared `KroneckerEmbedding` module and `build_codec_table` helper. |
| `experiment.py` | Colab runner: streams 5-language Wikipedia text, measures fertility/crop rate, trains all 9 curated codec configurations on a real `TinyLLM`, writes `results/report.json`. |
| `embed_report.py` | Injects a `results/report.json` into `index.html`'s embedded `<script>` block for the interactive report. |
| `index.html` | Self-contained interactive report (playground, curve filters, fertility toggle) with the run's JSON already embedded. |
| `results/report.json` | Raw output of the run quoted in this README and in the folder's own `README.md`. |
| `train.log` | Full console log of the actual Colab training run — cross-checked against `report.json` during verification for this README. |
| `requirements.txt` | `torch`, `transformers`, `datasets`, `sentencepiece`, etc. |

### `Fourier_Alternative_Kronecker/`

| File | What it is |
|---|---|
| `embedding_codecs.py` | `KroneckerCodec` and `FourierByteCodec` (the full value+position wave codec), same shared-interface design as problem 3's `kronecker.py`. |
| `data.py` | Synthetic byte-level vocabulary with engineered colliding token pairs, for the decisive synthetic experiment. |
| `tiny_transformer.py` | Shared tiny transformer classifier used in the synthetic experiment. |
| `experiment.py` | Runs the synthetic collision + classification experiment; writes `results.json`. |
| `train.py` | **The real 5-language validation — run this to reproduce.** Pulls Wikipedia text, trains a shared tokenizer, trains real causal LMs (Kronecker / Fourier / dense-embedding baseline), sweeps `pos_dim` and `num_freqs`. Meant for Google Colab (GPU). |
| `results.json` | Raw output of `experiment.py` (the synthetic experiment). |
| `real_data_results_5lang.json` | Raw output of `train.py`'s main 5-language run (fertility, collisions, per-language perplexity, model-size comparison). |
| `train.log` | Full console log of the actual `train.py` run, including the hyperparameter sweeps — kept as the audit trail since the sweep's own JSON output was not retained separately. |
| `report.md` | Full written report: problem statement, design, both experiments, results, limitations. |
| `report_visual.html` | Interactive visual version of the report (byte-window inspector, collision/training charts, sweep charts, model-size and real-PPL comparisons), already self-contained — the sweep numbers are embedded directly in this file's data, not read from a separate JSON. Also published as a Claude artifact. |
| `README.md` | This folder's own detailed README (superset of the summary above). |

> Note: the `pos_dim`/`num_freqs` sweep results were originally written to a standalone `hyperparam_sweep_results.json`, but that intermediate file is no longer present in the folder. Its numbers are preserved in `train.log`, in `report.md`'s "Real-data validation" section, and embedded directly in `report_visual.html` — nothing was lost, but there is currently no standalone JSON to re-load them from without rerunning `train.py`.

---

## How to reproduce

**Problem 3** (Colab, GPU):

```bash
cd Dynamic_Kronecker
pip install -r requirements.txt
python experiment.py colab --device cuda --epochs 3
python embed_report.py results/report.json   # re-embeds JSON into index.html
```

**Problem 4** — synthetic experiment (fast, local, CPU is fine):

```bash
cd Fourier_Alternative_Kronecker
uv run python experiment.py   # writes results.json
```

**Problem 4** — real 5-language validation (Colab, GPU recommended):

```bash
cd Fourier_Alternative_Kronecker
pip install -q datasets tokenizers torch
python train.py   # writes real_data_results_5lang.json; sweep numbers print to console/log only
```

## Notes on verification

Before writing this README, both folders were checked for authenticity rather than taken on faith:

- **Problem 3**: `train.log`'s final per-epoch numbers were diffed against `results/report.json` — they match exactly (e.g. `fourier_r32_abs` epoch 3: log shows `val=7.6712 ppl=2145.63`, JSON shows `final_val_loss: 7.6712...`, `final_val_ppl: 2145.63...`). The log shows real HF tokenizer/dataset download progress bars and real per-run wall-clock timing, consistent with an actual Colab GPU run rather than fabricated output. `index.html` was confirmed to contain the same `schema_version` and run IDs as the saved JSON.
- **Problem 4**: `train.py`'s own `train.log` was checked the same way — its per-language fertility, collision, and sweep numbers match `real_data_results_5lang.json` and the numbers embedded in `report_visual.html` exactly (e.g. Tamil `pct_colliding=0.40%`, `pos_dim=64` sweep row `avg_pct_colliding=0.00%`). `results.json` (the synthetic experiment) was produced across this same working session, with intermediate numbers checked by hand at each step (e.g. exact vector-collision counts recomputed directly from tensors, not asserted) before being written into the report.

Both are real, independently defensible pieces of work.
