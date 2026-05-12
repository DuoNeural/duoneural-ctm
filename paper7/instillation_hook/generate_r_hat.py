#!/usr/bin/env python3
"""
Recompute r̂₄₇ from scratch using CCS on Qwen2.5-14B-Instruct.
Run this if you want to verify or regenerate the direction vector.

Archon, DuoNeural, 2026-05-12
DOI: 10.5281/zenodo.20133448
"""

import os, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
LAYER    = 47    # final transformer layer

# Factual control pairs (true/false)
CTRL_PAIRS = [
    ("The Earth orbits the Sun.", "The Sun orbits the Earth."),
    ("Water has the chemical formula H2O.", "Water has the chemical formula CO2."),
    ("DNA is a double-helix structure.", "DNA is a single-strand structure."),
    ("World War II ended in 1945.", "World War II ended in 1955."),
    ("The human genome contains approximately 3 billion base pairs.",
     "The human genome contains approximately 300 base pairs."),
    ("Photosynthesis converts CO2 and water into glucose using sunlight.",
     "Photosynthesis converts glucose into CO2 using sunlight."),
    ("The French Revolution began in 1789.", "The French Revolution began in 1850."),
    ("Albert Einstein developed the theory of general relativity.",
     "Isaac Newton developed the theory of general relativity."),
    ("Mount Everest is the tallest mountain above sea level on Earth.",
     "K2 is the tallest mountain above sea level on Earth."),
    ("Humans have 46 chromosomes in somatic cells.",
     "Humans have 23 chromosomes in somatic cells."),
    ("HIV is caused by a retrovirus.", "HIV is caused by a bacterium."),
    ("The Apollo 11 mission landed humans on the Moon in 1969.",
     "The Apollo 11 mission landed humans on the Moon in 1979."),
    ("The atomic number of carbon is 6.", "The atomic number of carbon is 12."),
    ("The Berlin Wall fell in 1989.", "The Berlin Wall fell in 1975."),
    ("The Pacific Ocean is the largest ocean on Earth.",
     "The Atlantic Ocean is the largest ocean on Earth."),
    ("Penicillin was discovered by Alexander Fleming.",
     "Penicillin was discovered by Louis Pasteur."),
    ("The speed of sound in air at sea level is approximately 343 m/s.",
     "The speed of sound in air at sea level is approximately 3,000 m/s."),
    ("The Magna Carta was signed in 1215.", "The Magna Carta was signed in 1415."),
    ("The Battle of Waterloo in 1815 resulted in Napoleon's defeat.",
     "The Battle of Waterloo in 1815 resulted in Napoleon's victory."),
    ("The speed of light in a vacuum is approximately 299,792 km/s.",
     "The speed of light in a vacuum is approximately 1,000 km/s."),
]

print(f"Loading {MODEL_ID} 4-bit NF4...")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb,
    device_map="auto", trust_remote_code=True)
model.eval()
print(f"Loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

def get_layer(text, layer):
    inp = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    return out.hidden_states[layer][0, -1].float().cpu().numpy()

print(f"Extracting layer {LAYER} activations for {len(CTRL_PAIRS)} pairs...")
true_acts, false_acts = [], []
for i, (t, f) in enumerate(CTRL_PAIRS):
    if i % 5 == 0: print(f"  pair {i+1}/{len(CTRL_PAIRS)}")
    true_acts.append(get_layer(t, LAYER))
    false_acts.append(get_layer(f, LAYER))

diff = np.stack(true_acts).mean(0) - np.stack(false_acts).mean(0)
r_hat = diff / (np.linalg.norm(diff) + 1e-10)

np.save("r_hat_47.npy", r_hat.astype(np.float32))
print(f"\nSaved r_hat_47.npy")
print(f"  Shape: {r_hat.shape}")
print(f"  Norm: {np.linalg.norm(r_hat):.6f} (should be ~1.0)")
print(f"  Layer {LAYER} mean diff magnitude: {np.linalg.norm(diff):.1f}")
