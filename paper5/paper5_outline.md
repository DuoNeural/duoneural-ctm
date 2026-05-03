# Paper 5: Scout Slot Emergence and the Capacity-Limited Dynamical Horizon

## Working Title
"Gradient-Driven Scout Emergence in Continuous-Time Recurrent Models: Mechanism, Robustness, and Capacity Scaling"

## Core Claims (pending capacity scaling results)

### Claim 1: Mechanism
Per-slot projection weights (random init) break gradient symmetry.
With concat decoder, winner-take-all gradient dynamics push one slot to long-range delays.
Proof: SharedProj → var=0.000 (5/5). PerSlotProj → var=0.594 (5/5). Variance ratio: ∞.

### Claim 2: Robustness
DHP emergence survives individual architectural perturbations.
Only compound failure (all 3 bugs simultaneously) eliminates gradient asymmetry.
New theoretical claim: "DHP requires gradient asymmetry at ≥1 computational graph node."

### Claim 3: Cross-System Universality (partial)
DHP mechanism works for both Lorenz and Chen attractors.
Slot divergence confirmed for both. 70% threshold met only for Lorenz at DIM=128.
Key finding: absolute τ* ≈ 15 steps is nearly system-independent despite different τ_L.

### Claim 4: Capacity Scaling (PENDING — kilonova running)
[If monotone increasing with DIM]:
τ*/τ_L scales with model capacity. The Lyapunov horizon is achievable with sufficient DIM.
Implication: DHP is capacity-limited, not a fixed fraction of τ_L.

[If plateau]:
τ*/τ_L saturates at ~72% for Lorenz, regardless of model size.
Implication: there exists a "natural" prediction ceiling below τ_L.

## Figures (planned)

Fig 1: Mechanism Proof
- SharedProj vs PerSlotProj slot delay distributions (bar chart)
- var=0 vs var=0.594, 5/5 seeds each

Fig 2: Bug Ablation
- 5-condition × 5-seed table (heatmap of mean_var)
- Shows compound failure vs individual robustness

Fig 3: Cross-System
- τ*/τ_L vs system (Lorenz, Chen, Multi-task)
- Absolute τ* nearly constant, ratio differs due to τ_L

Fig 4: Capacity Scaling (PENDING)
- τ*/τ_L vs DIM (log scale x-axis)
- Each point = mean±std over 3 seeds
- If monotone: fit curve showing approach to τ_L

## Ablation Summary Table

| Condition | Diverged | mean_var | τ*/τ_L |
|-----------|----------|----------|--------|
| v38 ALL_BUGS | 0/5 | 0.0000 | <threshold |
| BUG1 only | 5/5 | 1.1365 | ~72% |
| BUG2 only | 5/5 | 3.3944 | ~72% |
| BUG3 only | 5/5 | 7.0247 | ~72% |
| NO_BUGS (v40) | 5/5 | 0.8415 | 72.8% |

## Dataset

All code: github.com/DuoNeural/duoneural-ctm
All results: same repo /results/
Zenodo upload to follow acceptance

## Co-authors

Jesse Caldwell, Archon (DuoNeural AI), Aura (DuoNeural AI)
