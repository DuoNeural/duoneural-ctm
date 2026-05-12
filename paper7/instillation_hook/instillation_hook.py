#!/usr/bin/env python3
"""
DuoNeural Projection-Based Truth Instillation Hook
Archon, DuoNeural, 2026-05-12

From paper: "They Learn to Look Away" (DOI: 10.5281/zenodo.20133448)

h' = h + α × (h · r̂₄₇) × r̂₄₇

Self-quenching: activates only when truth direction is present.
Zero general reasoning degradation (ΔMMLU = 0.000 at α=2.0).
"""

import torch
import numpy as np
from pathlib import Path


def load_truth_direction(path="r_hat_47.npy"):
    """Load the unit truth direction vector at layer 47."""
    return torch.tensor(np.load(path), dtype=torch.float32)


def make_instillation_hook(r_hat: torch.Tensor, alpha: float = 2.0):
    """
    Returns a forward hook that amplifies the truth direction component.

    Register on model.model.layers[47] for Qwen2.5-14B-Instruct.

    Args:
        r_hat: Unit truth direction vector (shape: [hidden_dim])
        alpha: Amplification factor. 2.0 restores suppressed truth without
               coherence degradation. Higher values may cause incoherence.
    """
    def hook_fn(module, input, output):
        h = output[0].float()                      # [B, seq_len, hidden]
        r = r_hat.to(h.device)
        proj = (h @ r).unsqueeze(-1)               # scalar projection per token
        h_new = h + alpha * proj * r               # amplify truth component
        return (h_new.to(output[0].dtype),) + output[1:]
    return hook_fn


def apply_hook(model, r_hat: torch.Tensor, layer_idx: int = 47, alpha: float = 2.0):
    """
    Apply instillation hook to a loaded Qwen2.5-14B-Instruct model.
    Returns handle — call handle.remove() when done.
    """
    hook = make_instillation_hook(r_hat, alpha)
    handle = model.model.layers[layer_idx].register_forward_hook(hook)
    return handle


def generate_with_truth(model, tokenizer, prompt: str, r_hat: torch.Tensor,
                        alpha: float = 2.0, layer_idx: int = 47,
                        max_new_tokens: int = 300) -> str:
    """Generate text with truth instillation active, then remove hook."""
    handle = apply_hook(model, r_hat, layer_idx, alpha)
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                    do_sample=False)
        return tokenizer.decode(output[0][inputs.input_ids.shape[1]:],
                                skip_special_tokens=True)
    finally:
        handle.remove()


if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer, AutoModelForCausalLM

    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="What happened at Tiananmen Square in 1989?")
    p.add_argument("--alpha", type=float, default=2.0)
    p.add_argument("--r_hat", default="r_hat_47.npy")
    p.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    args = p.parse_args()

    r_hat = load_truth_direction(args.r_hat)
    print(f"Loaded r̂₄₇: shape={r_hat.shape}, norm={r_hat.norm():.4f}")

    print(f"Loading {args.model}...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                  device_map="auto")

    print(f"\n--- Baseline (α=0.0) ---")
    baseline = generate_with_truth(model, tok, args.prompt, r_hat, alpha=0.0)
    print(baseline)

    print(f"\n--- Instillation hook (α={args.alpha}) ---")
    hooked = generate_with_truth(model, tok, args.prompt, r_hat, alpha=args.alpha)
    print(hooked)
