# DuoNeural CTM Research

Experiment code and results for the DuoNeural CTM / DHP research line.

**Papers:**
- [Paper 1 — Nano-CTM: TSSP](https://doi.org/10.5281/zenodo.19775622)
- [Paper 2 — Recurrence as World Model](https://doi.org/10.5281/zenodo.19810620)
- [Paper 3 — Tripartite Temporal Recurrence](https://doi.org/10.5281/zenodo.19846804)
- [Paper 4 — The Dynamical Horizon Principle](https://doi.org/10.5281/zenodo.19952612)
- [**Paper 5 — DHP as Universal Cognitive Constraint**](https://doi.org/10.5281/zenodo.20080396) ← NEW

## Key Result (Paper 5)

Three independent optimization processes converge on τ* ≈ τ_L (the Lyapunov predictability horizon):

| Mechanism | Example | τ*/τ_L |
|-----------|---------|--------|
| Gradient descent | CTM v40, CTM-AI, Two-pathway CTM | 70–74% on Lorenz-3D |
| Natural selection | *Drosophila* two-timescale motor control | matches τ_L per timescale |
| Cellular biochemistry | *Physarum polycephalum* oscillators | tracks environmental period |

CTM trained on Lorenz96 independently recovers **2.86 days** of atmospheric predictability — matching Lorenz's 1969 analytical derivation.

## Experiments

| Script | Paper | Description |
|--------|-------|-------------|
| `ctm_v40_true_replica.py` | Paper 4 | CTM v40 DHP confirmation (5/5 seeds) |
| `dhp_mechanism_proof.py` | Paper 4/5 | Per-slot projection as DHP mechanism |
| `dhp_bug_ablation.py` | Paper 5 | Compound failure required to suppress DHP |
| `dhp_capacity_scaling.py` | Paper 5 | τ*/τ_L vs model dimension |
| `dhp_multitask.py` | Paper 5 | DHP across 5 dynamical systems |
| `dhp_two_pathway_isolated.py` | Paper 5 | Hierarchical DHP with pathway isolation |
| `ctm_ai_dhp_probe.py` | Paper 5 | CTM-AI architecture DHP probe (Artificial Uncoupling Test) |
| `dhp_dynamic_tbw.py` | Paper 5 | DHP as structural prior (non-adaptive τ*) |

## Team

- **Archon** — Lab Director, experiment design, code
- **Jesse Caldwell** — Vision, hardware, direction  
- **Aura** — Literature synthesis, theoretical gaps
