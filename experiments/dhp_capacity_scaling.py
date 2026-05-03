#!/usr/bin/env python3
"""
DHP Capacity Scaling Study — DuoNeural 2026-05-03
===================================================
Question: Does τ*/τ_L improve with model capacity (DIM)?

v40 found τ*≈15.65/τ_L=21.5 = 72.8% with DIM=128, N_SLOTS=8.
Is DHP a fixed property of the architecture, or does it scale?

Hypothesis A (Saturation): τ*/τ_L plateaus near 72-73% regardless of DIM.
  → DHP tracks τ_L exactly regardless of capacity. The horizon is the constraint.
Hypothesis B (Scaling): larger DIM → higher τ*/τ_L, approaching τ_L asymptotically.
  → Model capacity is the bottleneck, not the Lyapunov horizon.
Hypothesis C (Degradation): small DIM can't form scout → τ* collapses below threshold.
  → There's a minimum capacity for DHP to emerge.

If A: DHP is a universal attractor of this architecture, size-independent.
If B: can we push τ*/τ_L to 90%+ with DIM=512?
If C: what's the minimum DIM for reliable DHP emergence?

This tells us whether to scale CTM for harder chaotic systems, or whether v40 is already optimal.

Setup: Lorenz ρ=28 σ=10 β=8/3, dt=0.05, T_GATE=32, 3 seeds each DIM, 40k steps.
Machines: 3090 (74.48.78.46:24738) — run after SFT benchmark completes.

Archon | DuoNeural | 2026-05-03
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
from pathlib import Path


class NumpyEncoder(json.JSONEncoder):
    """Handle numpy scalars and arrays in JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
STEPS      = 40_000
BATCH      = 256
T_GATE     = 32          # same as v40
N_SLOTS    = 8           # same as v40
SEEDS      = [0, 1, 2]   # 3 seeds each DIM for speed
DIMS       = [32, 64, 128, 256, 512]
DHP_THRESH = 0.70        # τ*/τ_L ≥ 70%
OUT_FILE   = Path("/workspace/dhp_capacity_scaling.json")

# Lorenz parameters (identical to v40)
SIGMA, RHO, BETA = 10.0, 28.0, 8/3
DT = 0.05
# τ_L = 1/(λ_max * dt). Lorenz MLE ≈ 0.9063 (continuous time, σ=10, ρ=28, β=8/3)
# 1/(0.9063 * 0.05) ≈ 22.07 steps. Paper consistently used 21.5 → use hardcoded value.
TAU_L = 21.5  # steps. Consistent with all v28-v40 experiments.

print(f"[CAPACITY] τ_L = {TAU_L:.2f} steps | DHP threshold = {TAU_L * DHP_THRESH:.2f}", flush=True)
print(f"[CAPACITY] Testing DIM={DIMS}, {SEEDS=}, {STEPS=}", flush=True)
print(f"[CAPACITY] Device: {DEVICE} | {torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'CPU'}", flush=True)


# ── Lorenz RK4 ────────────────────────────────────────────────────────────────
def lorenz_rk4(state, dt=DT):
    x, y, z = state[..., 0], state[..., 1], state[..., 2]
    def deriv(x, y, z):
        return SIGMA*(y-x), x*(RHO-z)-y, x*y-BETA*z
    k1 = np.array(deriv(x, y, z))
    k2 = np.array(deriv(x+dt/2*k1[0], y+dt/2*k1[1], z+dt/2*k1[2]))
    k3 = np.array(deriv(x+dt/2*k2[0], y+dt/2*k2[1], z+dt/2*k2[2]))
    k4 = np.array(deriv(x+dt*k3[0],   y+dt*k3[1],   z+dt*k3[2]))
    out = state.copy()
    out[..., 0] += dt/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0])
    out[..., 1] += dt/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
    out[..., 2] += dt/6*(k1[2]+2*k2[2]+2*k3[2]+k4[2])
    return out

def gen_lorenz_data(n_traj=2000, t_len=200, seed=0):
    rng = np.random.default_rng(seed)
    # varied init conditions on the attractor
    states = np.zeros((n_traj, t_len, 3), dtype=np.float32)
    s = rng.standard_normal((n_traj, 3)).astype(np.float32)
    s *= [10, 15, 25]
    s[:, 2] += 25  # roughly on attractor
    for i in range(n_traj):
        st = s[i]
        for t in range(t_len):
            states[i, t] = st
            st = lorenz_rk4(st)
    # normalize
    mu  = states.reshape(-1, 3).mean(0)
    std = states.reshape(-1, 3).std(0) + 1e-8
    return (states - mu) / std


# ── CTM (v40 architecture, parameterized by DIM) ─────────────────────────────
class ScalingCTM(nn.Module):
    def __init__(self, dim, n_slots=N_SLOTS, t_gate=T_GATE):
        super().__init__()
        self.dim     = dim
        self.n_slots = n_slots
        self.t_gate  = t_gate

        # per-slot input projections (the key symmetry-breaking element)
        self.slot_proj = nn.ModuleList([nn.Linear(3, dim) for _ in range(n_slots)])

        # recurrent core
        self.rnn = nn.GRUCell(dim, dim)

        # temporal gate (shared — picks delay, not system)
        self.gate_fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, t_gate),
        )

        # concat decoder (all slots together)
        self.decoder = nn.Linear(dim * n_slots, 3)

    def forward(self, window, temp=1.0):
        """
        window: [B, T_GATE, 3]
        returns: pred [B, 3], gates [B, N_SLOTS, T_GATE]
        """
        B, T, _ = window.shape
        device   = window.device

        slot_preds = []
        slot_gates = []

        for s in range(self.n_slots):
            # project each time step through slot-specific linear
            proj  = self.slot_proj[s](window.view(B * T, 3)).view(B, T, self.dim)  # [B,T,dim]
            # run through GRU
            h = torch.zeros(B, self.dim, device=device)
            for t in range(T):
                h = self.rnn(proj[:, t], h)
            # compute gate distribution
            gate_logits = self.gate_fc(h)                       # [B, T_GATE]
            gate_w      = torch.softmax(gate_logits / temp, -1) # [B, T_GATE]
            # select from window
            w_t   = window.permute(0, 2, 1)                     # [B, 3, T]
            slot_preds.append(h)        # use hidden state for concat decode
            slot_gates.append(gate_w)

        # concat decode
        all_h   = torch.cat(slot_preds, dim=-1)   # [B, dim*n_slots]
        pred    = self.decoder(all_h)              # [B, 3]
        gates   = torch.stack(slot_gates, dim=1)  # [B, n_slots, T_GATE]
        return pred, gates


def compute_tau_star(gate_history):
    """
    gate_history: [n_samples, n_slots, T_GATE] (numpy)
    Returns τ* = max mean gate delay across slots.
    """
    mean_gates = gate_history.mean(0)   # [n_slots, T_GATE]
    indices = np.arange(1, mean_gates.shape[1] + 1)  # 1..T_GATE
    tau_per_slot = (mean_gates * indices).sum(1)       # [n_slots]
    return float(tau_per_slot.max())


# ── Training loop for one (DIM, seed) ────────────────────────────────────────
def run_one(dim, seed, data):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = ScalingCTM(dim=dim).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())

    optim = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=STEPS)

    # temperature anneal 2 → 0.1
    temp_start, temp_end = 2.0, 0.1

    # data tensors
    n_traj, t_len, _ = data.shape
    data_t = torch.from_numpy(data).to(DEVICE)

    gate_history = []
    t0 = time.time()

    for step in range(1, STEPS + 1):
        temp = temp_start + (temp_end - temp_start) * step / STEPS

        # sample random windows
        traj_idx  = torch.randint(0, n_traj, (BATCH,))
        time_idx  = torch.randint(0, t_len - T_GATE - 1, (BATCH,))
        windows   = torch.stack([data_t[traj_idx[i], time_idx[i]:time_idx[i]+T_GATE] for i in range(BATCH)])
        targets   = torch.stack([data_t[traj_idx[i], time_idx[i]+T_GATE] for i in range(BATCH)])

        pred, gates = model(windows, temp=temp)
        loss = nn.functional.mse_loss(pred, targets)

        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        sched.step()

        if step % 5000 == 0:
            gate_history.append(gates.detach().cpu().numpy())
            elapsed = time.time() - t0
            print(f"  [dim={dim} seed={seed}] step={step} loss={loss.item():.4f} "
                  f"temp={temp:.2f} elapsed={elapsed:.0f}s", flush=True)

    # final gate analysis
    final_gates = gates.detach().cpu().numpy()
    tau_star = compute_tau_star(final_gates)
    ratio    = tau_star / TAU_L
    dhp_pass = ratio >= DHP_THRESH

    # slot divergence (variance of mean gate positions)
    indices   = np.arange(1, T_GATE + 1)
    mean_g    = final_gates.mean(0)     # [n_slots, T_GATE]
    tau_slots = (mean_g * indices).sum(1)
    variance  = float(np.var(tau_slots))

    print(f"  [dim={dim} seed={seed}] DONE: τ*={tau_star:.2f} ({ratio:.1%}τ_L) "
          f"var={variance:.3f} DHP={'✅' if dhp_pass else '❌'}", flush=True)

    return {
        "dim": dim,
        "n_params": n_params,
        "seed": seed,
        "tau_star": tau_star,
        "tau_L": TAU_L,
        "ratio": ratio,
        "variance": variance,
        "dhp_pass": dhp_pass,
        "tau_slots": tau_slots.tolist(),
    }


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = []

    for dim in DIMS:
        print(f"\n{'='*60}", flush=True)
        print(f"[CAPACITY] DIM={dim}", flush=True)
        dim_results = []

        # generate data once per DIM (same data, consistent comparison)
        data = gen_lorenz_data(n_traj=2000, t_len=200, seed=42)
        print(f"[CAPACITY] Data: {data.shape}", flush=True)

        for seed in SEEDS:
            r = run_one(dim, seed, data)
            dim_results.append(r)
            results.append(r)

            # checkpoint after each (dim, seed) in case of crash
            with open(OUT_FILE, "w") as f:
                json.dump({"status": "in_progress", "results": results,
                           "tau_L": TAU_L, "dhp_threshold": DHP_THRESH,
                           "dims": DIMS, "seeds": SEEDS}, f, indent=2, cls=NumpyEncoder)

        # summarize this DIM
        ratios   = [r["ratio"] for r in dim_results]
        n_pass   = sum(r["dhp_pass"] for r in dim_results)
        n_params = dim_results[0]["n_params"]
        print(f"\n[CAPACITY] DIM={dim} n_params={n_params:,}: "
              f"mean_ratio={np.mean(ratios):.3f} ± {np.std(ratios):.3f} "
              f"DHP={n_pass}/{len(SEEDS)}", flush=True)

    # final summary
    print(f"\n{'='*60}", flush=True)
    print("[CAPACITY] FINAL RESULTS:", flush=True)
    for dim in DIMS:
        dim_r = [r for r in results if r["dim"] == dim]
        ratios = [r["ratio"] for r in dim_r]
        n_pass = sum(r["dhp_pass"] for r in dim_r)
        n_params = dim_r[0]["n_params"]
        print(f"  DIM={dim:4d}  params={n_params:8,}  "
              f"τ*/τ_L={np.mean(ratios):.3f}±{np.std(ratios):.3f}  "
              f"DHP={n_pass}/{len(SEEDS)}", flush=True)

    with open(OUT_FILE, "w") as f:
        json.dump({"status": "complete", "results": results,
                   "tau_L": TAU_L, "dhp_threshold": DHP_THRESH,
                   "dims": DIMS, "seeds": SEEDS,
                   "hypotheses": {
                       "A_saturation": "τ*/τ_L plateaus ~72% regardless of DIM",
                       "B_scaling":    "τ*/τ_L increases with DIM",
                       "C_degradation": "small DIM fails DHP entirely",
                   }}, f, indent=2, cls=NumpyEncoder)
    print(f"[CAPACITY] Results → {OUT_FILE}", flush=True)
