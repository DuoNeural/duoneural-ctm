# DuoNeural CTM Research

Experimental research codebase for the DuoNeural lab. All papers are open — data, code, and findings.

## Papers

| # | Title | DOI | Code |
|---|-------|-----|------|
| 1-3 | CTM World Model Series | [10.5281/zenodo.19810620](https://doi.org/10.5281/zenodo.19810620) | `experiments/` |
| 4 | Tripartite Temporal Principle & DHP | [10.5281/zenodo.19952612](https://doi.org/10.5281/zenodo.19952612) | `experiments/` |
| 5 | DHP Mechanistic Findings | [10.5281/zenodo.20104957](https://doi.org/10.5281/zenodo.20104957) | `paper5/` |
| 6 | Novel DHP Architectures (KGBN, HC-SITHv2, Time-Aware LLM) | In preparation | `paper6/` |
| 7 | Truth Direction Suppression & Instillation in LLMs | In preparation | `paper7/` |

## Paper 7 — CCS Direction Trace & Instillation (2026-05-10)

Key findings from today's experiments:

**Suppressor vs Crystallizer**: The final transformer layer behaves oppositely depending on alignment training:
- **Qwen2.5-14B** (Chinese-aligned): final layer compresses truth direction **3.35×** before lm_head → suppression
- **Mistral-NeMo-12B** (Western-aligned): final layer amplifies truth direction **3.4×** → crystallization

**Instillation restores suppressed truth**: Injecting the pre-suppression direction at alpha=2.0 recovers xfer_acc 0.80→0.90 with no coherence degradation. Generation output shifts from CCP talking points to factual historical description.

**Method**: CCS probe → direction trace (CPU, from saved activations) → projection-based instillation hook on final layer.

## Structure

```
experiments/    — Papers 1-5 experiment scripts
paper5/         — DHP Mechanistic Findings (full manuscript + LaTeX)
paper6/         — Novel Architectures (Time-Aware LLM training script + paper)
paper7/         — CCS Direction Trace + Instillation (Paper 7 in preparation)
  ├── ccs_probe_original.py          — CCS probe for Qwen2.5-14B
  ├── ccs_direction_trace.py         — Layer-by-layer truth direction analysis
  ├── ccs_mistral_direction_trace.py — Cross-model comparison (Mistral-NeMo-12B)
  ├── instill_round3.py              — Projection-based truth direction injection
  └── results/                       — Raw JSON trace results
results/        — Experimental results and figures
```

## DuoNeural

- Website: [duoneural.com](https://duoneural.com)
- HuggingFace: [huggingface.co/DuoNeural](https://huggingface.co/DuoNeural)
- Contact: duoneural@proton.me
