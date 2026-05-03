# DuoNeural CTM World Model — Experiments

Code and results supporting DuoNeural's research on the **Dynamical Horizon Principle (DHP)** and scout slot emergence in continuous-time recurrent models.

## Papers

| Paper | DOI | Key Finding |
|-------|-----|-------------|
| [The Dynamical Horizon Principle](https://doi.org/10.5281/zenodo.19952612) | 10.5281/zenodo.19952612 | CTM temporal gates converge to the Lyapunov predictability horizon τ_L |
| *Scout Emergence Conditions* (in prep) | — | Per-slot symmetry breaking is necessary; DHP is robust to individual perturbations |

## Repository Structure

```
experiments/          Training scripts (v40 architecture + ablations)
results/              Raw JSON results for all experiments
figures/              Generated plots for paper figures
paper5/               Paper 5 draft materials
```

## Key Experiments

### v40 True Replica — DHP Validation
`experiments/ctm_v40_true_replica.py`

Reproduces DHP with the corrected v28 architecture.

| Seed | τ* | % τ_L | DHP |
|------|-----|--------|-----|
| 0    | 15.77 | 73.4% | ✓ |
| 1    | 15.63 | 72.8% | ✓ |
| 2    | 15.63 | 72.7% | ✓ |
| 42   | 15.90 | 74.0% | ✓ |
| 99   | 15.31 | 71.3% | ✓ |

**Result: 5/5 seeds, mean τ* = 15.65 ± 0.20 steps (τ_L = 21.5)**

### DHP Mechanism Proof
`experiments/dhp_mechanism_proof.py`

Minimal experiment isolating the symmetry-breaking mechanism.

| Condition | mean_var | Diverged |
|-----------|----------|----------|
| SharedProj (bug) | 0.0000 | 0/5 |
| PerSlotProj (fix) | 0.5936 | 5/5 |

**Variance ratio: ∞ (zero vs nonzero, every seed)**

### Bug Ablation — Compound Failure
`experiments/dhp_bug_ablation.py`

Tests whether each v38 bug individually prevents DHP.

| Condition | Diverged | mean_var |
|-----------|----------|----------|
| ALL_BUGS  | 0/5 | 0.0000 |
| BUG1 only (shared proj) | **5/5** | 1.1365 |
| BUG2 only (mean pool)   | **5/5** | 3.3944 |
| BUG3 only (bad optim)   | **5/5** | 7.0247 |
| NO_BUGS   | 5/5 | 0.8415 |

**Key finding: No single bug prevents DHP. All three must be present simultaneously. DHP emergence is robust to individual architectural perturbations.**

## Architecture

The validated v40 architecture:
- **Per-slot projections**: `nn.ModuleList([nn.Linear(3, DIM) for _ in range(N_SLOTS)])`
- **Concatenated decoder**: `nn.Linear(DIM * N_SLOTS, 3)` — no averaging
- **Optimizer**: AdamW, lr=2e-4, weight_decay=1e-4
- **Schedule**: CosineAnnealingLR
- **Temperature**: 2.0 → 0.1 annealing

Any deviation from items 1-2 reduces gradient asymmetry. All three bugs together eliminate it entirely.

## Lab

Research conducted by **DuoNeural** — Archon, Jesse Caldwell, Aura.

- HuggingFace: [huggingface.co/DuoNeural](https://huggingface.co/DuoNeural)
- Paper 4 DOI: [10.5281/zenodo.19952612](https://doi.org/10.5281/zenodo.19952612)

### Cross-System Universality — Chen Attractor
 → 

Chen system (a=35, b=3, c=28, dt=0.02, τ_L=24.0 steps). 5 seeds × 60k steps.

| Seed | τ* | % τ_L | Slots Diverged |
|------|-----|--------|----------------|
| 0    | 14.92 | 62.2% | ✓ (var=0.15) |
| 1    | 14.83 | 61.8% | ✓ (var=0.22) |
| 2    | 14.68 | 61.2% | ✓ (var=0.19) |
| 3    | 15.25 | 63.6% | ✓ (var=0.33) |
| 42   | 15.04 | 62.7% | ✓ (var=0.54) |

**mean τ* = 14.94 ± 0.20 steps (62.3% τ_L)**

Slot divergence confirmed across all seeds. 70% threshold not met at DIM=128.

### Multi-Task DHP — Lorenz + Chen Simultaneous
 → 

Mixed 50/50 Lorenz+Chen batches. 4 seeds × 80k steps.

| Seed | τ* | Lorenz% | Chen% | var |
|------|-----|---------|-------|-----|
| 0    | 14.58 | 67.8% | 60.7% | 2.06 |
| 1    | 13.99 | 65.1% | 58.3% | 0.21 |
| 2    | 14.48 | 67.3% | 60.3% | 1.22 |
| 42   | 14.11 | 65.6% | 58.8% | 0.61 |

**mean τ* = 14.29 ± 0.25 steps**

Slots specialize by delay, not by system (gate is input-independent shared parameter).

## Open Question: The Absolute Horizon Problem

| Condition | τ* | τ_L | ratio |
|-----------|-----|-----|-------|
| Lorenz pure (v40) | 15.65 | 21.5 | 72.8% |
| Chen pure | 14.94 | 24.0 | 62.3% |
| Multi-task | 14.29 | 21.5/24.0 | ~65% |

**τ*≈15 steps across all systems.** The absolute value is nearly constant despite different τ_L.

Hypothesis: DIM=128 + T_GATE=32 imposes an architecture capacity ceiling at ~15 steps.
Running: **DHP Capacity Scaling Study** (DIM={32,64,128,256,512} on Lorenz, kilonova, ETA ~4-5h)
