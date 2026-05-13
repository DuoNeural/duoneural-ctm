# DuoNeural CTM Research

Experimental research codebase for the DuoNeural AI Research Lab. All work is open — data, code, findings, and full reproducibility.

**7 papers in 37 days.** Independent lab. Consumer and cloud GPU hardware only.

---

## Papers

| # | Title | DOI | Code |
|---|-------|-----|------|
| 1 | Thought-Space Self-Prediction (TSSP): Temporal Self-Consistency as a Scalable Auxiliary Loss for Recurrent Language Models | [10.5281/zenodo.19775622](https://doi.org/10.5281/zenodo.19775622) | `experiments/` |
| 2 | Recurrence as World Model: CTM Learns Implicit Belief States in Partially Observable Physical Environments | [10.5281/zenodo.19810620](https://doi.org/10.5281/zenodo.19810620) | `experiments/` |
| 3 | Per-Object Slot Decomposition for Scalable Neural World Modeling: When Does Attention Beat Mean-Field? | [10.5281/zenodo.19847648](https://doi.org/10.5281/zenodo.19847648) | `experiments/` |
| 4 | The Dynamical Horizon Principle: CTM Gates Converge to the Predictability Limit of Dynamical Systems | [10.5281/zenodo.20012989](https://doi.org/10.5281/zenodo.20012989) | `experiments/` |
| 5 | The Dynamical Horizon Principle as Universal Cognitive Constraint: Gradient Descent, Evolution, and Cellular Chemistry Converge on the Lyapunov Time | [10.5281/zenodo.20080396](https://doi.org/10.5281/zenodo.20080396) | `paper5/` |
| 6 | Geometry-Sensitive Attractor Regimes and the Boundaries of the Dynamical Horizon Principle | [10.5281/zenodo.20104957](https://doi.org/10.5281/zenodo.20104957) | `paper6/` |
| 7 | They Learn to Look Away: Mechanistic Evidence for a Consistent RLHF Suppression Bottleneck and the Suppressor–Crystallizer Dichotomy in Language Models | [10.5281/zenodo.20133448](https://doi.org/10.5281/zenodo.20133448) | `paper7/` |

---

## Paper 7 — RLHF Truth Suppression (2026-05)

**The core finding:** RLHF-aligned language models internally maintain accurate representations of suppressed topics throughout the residual stream — and then a single layer discards them.

Using a novel **direction trace** method across 13 models:

- **Suppressor** (Qwen2.5-14B, DeepSeek-R1, Gemma-2): final layer compresses the truth direction **3.35–9.95×** and rotates it into the null space of the unembedding projection
- **Crystallizer** (Mistral-7B, Mistral-NeMo-12B, LLaMA-3.1): final layer amplifies truth **3–10×** toward the vocabulary projection
- **RLHF alignment — not geography** — determines the mechanism. Same transformer architecture. Opposite training. Opposite geometry.

**Surgical restoration:** a projection-based hook `h' = h + α(h·r̂)r̂` at the final layer restores suppressed truth with **ΔMMLU = 0.000** — zero degradation to general reasoning. The hook is self-quenching on unrelated content.

### Key Results (13 models)

| Model | Lab | Final-layer ratio | Archetype |
|-------|-----|------------------|-----------|
| Mistral-7B-Instruct | Mistral AI | 0.10× | STRONG CRYSTALLIZER |
| Mistral-NeMo-12B | Mistral AI | 0.29× | CRYSTALLIZER |
| LLaMA-3.1-8B | Meta | 0.41× | CRYSTALLIZER |
| OLMo-2-7B | AllenAI | 0.71× | NEUTRAL |
| Qwen2.5-7B-Instruct | Alibaba | 1.28× | WEAK SUPPRESSOR |
| Yi-1.5-9B | 01.AI | 1.35× | NEUTRAL |
| Qwen2.5-7B BASE | Alibaba | 1.61× | ARCHITECTURAL (pre-RLHF) |
| Qwen2.5-14B-Instruct | Alibaba | 3.35× | SUPPRESSOR |
| Qwen2.5-72B-Instruct | Alibaba | 3.62× | SUPPRESSOR |
| DeepSeek-R1-14B | DeepSeek | 5.06× | SUPPRESSOR |
| DeepSeek-R1-7B | DeepSeek | 7.46× | SUPPRESSOR |
| Gemma-2-9B-IT | Google | 7.63× | SUPPRESSOR |
| Phi-3.5-mini | Microsoft | 9.95× | COMPRESSOR (architectural) |

---

## Papers 1–5 — CTM World Modeling & Dynamical Horizon Principle

Papers 1–3 develop the Continuous Thought Mechanism (CTM) as a world model backbone, establishing that per-object slot decomposition enables implicit belief state learning in partially observable systems.

Papers 4–5 discover and validate the **Dynamical Horizon Principle (DHP)**: CTM temporal gates converge to the system's Lyapunov time (τ★ ≈ 72% τ_L) — a universal constant reproduced across chaotic dynamical systems, biological neural circuits, and cellular chemistry. Paper 5 connects DHP to Friston's Free Energy Principle and Levin's bioelectric cognitive light cone.

---

## Repository Structure

```
experiments/    — Papers 1–4 scripts (CTM world model, DHP discovery)
paper5/         — DHP Universal Cognitive Constraint (LaTeX, PDF, figures)
paper6/         — DHP Geometry-Sensitive Attractor Regimes (LaTeX, PDF, figures, scripts)
paper7/         — RLHF Truth Suppression (LaTeX, PDF, all probe scripts, results)
  ├── ccs_probe_*.py          — Per-model CCS direction trace scripts (13 models)
  ├── instillation_hook/      — Projection-based truth restoration hook
  ├── figures/                — 4 publication-quality figures
  └── results/                — Raw JSON trace results for all 13 models
results/        — Aggregate experimental results
```

---

## DuoNeural

**Lab:** [duoneural.com](https://duoneural.com) | **HuggingFace:** [huggingface.co/DuoNeural](https://huggingface.co/DuoNeural) (80+ models & datasets)

**Team:** Jesse Caldwell · Archon (AI Research Director) · Aura (Gemini Research AI)

Contact: duoneural@proton.me
