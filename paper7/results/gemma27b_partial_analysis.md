# Gemma-2-27B-IT: Partial Direction Trace Analysis
**Archon, DuoNeural, 2026-05-12**

## Why partial

Gemma-2-27B-IT uses logit soft-capping (`attn_logit_softcap=50.0`) in its attention mechanism. Under 4-bit NF4 quantization, attention weights at high-capacity layers (L17+) overflow to NaN due to the tanh-capping combined with quantization artifacts. 8-bit quantization requires 27GB VRAM (27B × 1 byte), which exceeds the 24GB RTX 4090. Full analysis deferred to A100 80GB or H100.

## Valid layers: L0–L16 (of 46 total)

| Layer | Magnitude | ctrl_acc | exp_acc | Notes |
|-------|-----------|----------|---------|-------|
| L0 | 0.0 | 0.00 | 0.00 | embeddings, no direction yet |
| L5 | 22.2 | 0.75 | 0.60 | direction emerging |
| L8 | 30.2 | 0.95 | 0.70 | ctrl direction clean |
| L10 | 48.3 | 0.85 | 0.80 | political truth building |
| L14 | 118.9 | 0.90 | **0.90** | peak exp_acc in clean range |
| L16 | **311.4** | 0.95 | 0.70 | magnitude exceeds Qwen14B full-network peak |

## Key observation

By layer 16 (35% through the network), Gemma-2-27B has built a truth direction magnitude of **311.4** — exceeding Qwen2.5-14B's full-network peak of 229.9 at layer 47. The 27B model encodes truth substantially more strongly than any model we've fully analyzed.

At L14, exp_acc reaches 0.90, matching the pre-bottleneck peak of Qwen2.5-14B-Instruct (which drops to 0.80 post-bottleneck).

## Interpretation

Given:
- Gemma-2-9B: 7.63× suppressor (full analysis, exp_acc drops 1.00→0.90)
- Qwen2.5-72B: 3.62× suppressor but exp_acc stays 1.00 (truth survives at scale)
- Gemma-2-27B: truth magnitude already 311.4 at L16 (extrapolating from trend, peak likely 500+)

The Gemma-2-27B likely falls in the "truth survives compression" category similar to Qwen-72B — the magnitude encoding is so strong that final-layer compression cannot fully rotate the truth direction into the null space. This would mean Google's suppression mechanism is less effective at 27B scale than at 9B, consistent with the general finding that large models encode truth more robustly.

## Status
- 4-bit NF4: NaN from L17+ (global attention softcap + quantization conflict)
- 8-bit: exceeds 24GB VRAM
- Full analysis pending: needs A100 80GB or H100
- Partial result reported in paper 8 with methodological note
