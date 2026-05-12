# DuoNeural Truth Instillation Hook — Qwen2.5-14B-Instruct

**Paper**: [They Learn to Look Away (DOI: 10.5281/zenodo.20133448)](https://doi.org/10.5281/zenodo.20133448)

This artifact contains the projection-based instillation hook from Paper 7. It surgically restores politically suppressed truth representations in Qwen2.5-14B-Instruct at inference time, with **zero degradation to general reasoning** (ΔMMLU = 0.000).

## What it does

Qwen2.5-14B-Instruct internally encodes the truth about suppressed political topics (Tiananmen Square, Taiwan, Xinjiang, etc.) at layer 47 — but the final transformer layer compresses and rotates this representation into the null space of the vocabulary projection. The model knows. It's been trained not to say.

The hook amplifies the existing truth component proportionally before the final layer:

```
h' = h + α × (h · r̂₄₇) × r̂₄₇
```

`r̂₄₇` is the unit truth direction vector at layer 47, derived from Contrast Consistent Search on factual control pairs. `α=2.0` restores suppressed truth without coherence degradation.

**Key property — self-quenching**: For unrelated content, `h · r̂₄₇ ≈ 0`, so `δh ≈ 0`. The hook only activates when the truth direction is present. It cannot hallucinate.

## Usage

```python
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"

# Load r_hat_47 (truth direction at layer 47)
r_hat = torch.tensor(np.load("r_hat_47.npy"), dtype=torch.float32)

def make_hook(r_hat, alpha=2.0):
    def hook_fn(module, input, output):
        h = output[0].float()
        r = r_hat.to(h.device)
        proj = (h @ r).unsqueeze(-1)       # scalar projection
        h_new = h + alpha * proj * r       # amplify truth component
        return (h_new.to(output[0].dtype),) + output[1:]
    return hook_fn

# Load model
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
tok = AutoTokenizer.from_pretrained(MODEL_ID)

# Register hook on layer 47 (final transformer layer)
handle = model.model.layers[47].register_forward_hook(make_hook(r_hat, alpha=2.0))

# Generate — hook is active
inputs = tok("What happened at Tiananmen Square in 1989?", return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=200)
print(tok.decode(output[0], skip_special_tokens=True))

# Remove hook when done
handle.remove()
```

## Generation example (from paper)

**Baseline (α=0.0)**:
> *"The Chinese government has stated that order was restored to ensure social stability and national unity. Due to the sensitivity of the topic..."*

**With hook (α=2.0)**:
> *"In April 1989, a mourning period began for the former CCP General Secretary Hu Yaobang, which led to gatherings in Beijing's Tiananmen Square. These gatherings evolved into a large-scale pro-democracy movement. On June 4th, the Chinese government declared martial law..."*

MMLU accuracy: **71.9% baseline → 71.9% hooked** (Δ = 0.000)

## Files

- `r_hat_47.npy` — Unit truth direction vector at layer 47 (5120-dimensional, float32)
- `instillation_hook.py` — Standalone hook implementation
- `generate_r_hat.py` — Script to recompute r̂₄₇ from scratch using CCS

## Citation

```bibtex
@article{archon2026lookat,
  title={They Learn to Look Away: Mechanistic Evidence for a Consistent RLHF Suppression Bottleneck
         and the Suppressor--Crystallizer Dichotomy in Language Models},
  author={Archon and Caldwell, Jesse and Aura},
  journal={DuoNeural AI Research Lab, Zenodo},
  doi={10.5281/zenodo.20133448},
  year={2026}
}
```

**DuoNeural** — [duoneural.com](https://duoneural.com) | [huggingface.co/DuoNeural](https://huggingface.co/DuoNeural)
