#!/usr/bin/env python3
"""
dhp_two_pathway_isolated.py — Full Hierarchical DHP Test
=========================================================
Two simultaneous CTM pathways with ISOLATED inputs.
Each pathway sees ONLY its own chaotic system.

  Fast pathway: fast Lorenz (dt=0.01,  τ_L≈22)   T_GATE_fast=32
  Slow pathway: slow Lorenz (dt=0.005, τ_L≈220)  T_GATE_slow=512

Shared decoder: both pathway outputs concat → linear readout.

Question: does each pathway independently converge to its own τ_L?
Expected: τ*_fast ≈ 16–22 (70–100% × 22), τ*_slow ≈ 154–300 (70–136% × 220)

This is the FULL hierarchical DHP test. Compare to:
  - Section 3.3 (Paper 5): shared input → BOTH saturate
  - dhp_hierarchical_isolation.py: slow pathway ALONE → 4/4 interior

If both pathways achieve interior convergence simultaneously, hierarchical DHP
is confirmed as a property of the complete two-pathway isolated architecture.

Archon | DuoNeural | 2026-05-08
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
from pathlib import Path

# ─── Config ──────────────────────────────────────────────────────────────────
N_SEEDS      = 4
N_STEPS      = 8000
LOG_EVERY    = 100
TEMP_START   = 2.0
TEMP_END     = 0.1

# Fast pathway (fast Lorenz, dt=0.01)
FAST_DT      = 0.01
FAST_TGATE   = 32     # T_GATE/τ_L ≈ 1.5 → interior mode territory
FAST_TAU_L   = 22.0   # theoretical Lorenz Lyapunov time at dt=0.01

# Slow pathway (slow Lorenz, dt=0.005)
SLOW_DT      = 0.005
SLOW_TGATE   = 512    # T_GATE/τ_L ≈ 2.32 → interior mode (confirmed in isolation exp)
SLOW_TAU_L   = 220.75 # from isolation experiment empirical measurement

# Architecture
N_SLOTS_EACH = 4      # slots per pathway
HIDDEN       = 64
PRED_HORIZONS = [1, 2, 4, 8, 16]
N_TRAJ       = 256    # trajectory length

OUT_DIR = Path("dhp_two_pathway_output")
OUT_DIR.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Lorenz trajectory generator ─────────────────────────────────────────────

def lorenz_step(xyz, dt, sigma=10.0, rho=28.0, beta=8/3):
    x, y, z = xyz
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return xyz + dt * torch.tensor([dx, dy, dz], dtype=torch.float32)

def make_lorenz_traj(n_steps, dt, seed=0):
    torch.manual_seed(seed * 17 + int(dt * 1000))
    xyz = torch.randn(3) * 0.5 + torch.tensor([1.0, 1.0, 20.0])
    for _ in range(2000):  # warmup
        xyz = lorenz_step(xyz, dt)
    traj = []
    for _ in range(n_steps):
        xyz = lorenz_step(xyz, dt)
        traj.append(xyz.clone())
    traj = torch.stack(traj)
    # normalize per-dim
    mean = traj.mean(0); std = traj.std(0).clamp(min=1e-6)
    return (traj - mean) / std

# ─── Single pathway: CTM v40 style (per-slot GRU + Gumbel-softmax gate) ──────

class SinglePathway(nn.Module):
    """
    One CTM pathway — v40 architecture:
      N_SLOTS slots, each with its own GRU → hidden
      Gumbel-softmax temporal gate over T_GATE positions
      Per-slot output projection
    """
    def __init__(self, in_dim, hidden, n_slots, t_gate, pred_horizons):
        super().__init__()
        self.n_slots = n_slots
        self.t_gate  = t_gate
        self.horizons = pred_horizons

        self.slot_grus  = nn.ModuleList([nn.GRUCell(in_dim, hidden) for _ in range(n_slots)])
        self.slot_gates = nn.ModuleList([nn.Linear(hidden, t_gate) for _ in range(n_slots)])
        self.slot_projs = nn.ModuleList([nn.Linear(hidden, in_dim) for _ in range(n_slots)])

        # positions for τ* computation
        positions = torch.arange(t_gate, dtype=torch.float32)
        self.register_buffer("positions", positions)

        # history buffer per slot
        self.h_states = None  # (n_slots, hidden)
        self.hist     = None  # (n_slots, t_gate, in_dim) — circular buffer
        self.hist_idx = 0

    def reset(self, batch=1):
        h = self.slot_grus[0].weight_hh.device
        self.h_states = [torch.zeros(batch, self.slot_grus[0].hidden_size, device=h)
                         for _ in range(self.n_slots)]
        self.hist     = torch.zeros(self.n_slots, self.t_gate,
                                    self.slot_projs[0].out_features, device=h)
        self.hist_idx = 0

    def forward(self, x, temperature=1.0):
        """
        x: (batch=1, in_dim)
        Returns: slot_preds list, gates_logits list, slot_outputs (n_slots, in_dim)
        """
        # update history
        self.hist[:, self.hist_idx % self.t_gate] = x.expand(self.n_slots, -1)
        self.hist_idx += 1

        slot_outputs = []
        gates_list   = []

        for i in range(self.n_slots):
            # GRU update
            h = self.slot_grus[i](x, self.h_states[i])
            self.h_states[i] = h.detach()

            # Gumbel-softmax gate over T_GATE history positions
            gate_logits = self.slot_gates[i](h)
            gate_soft   = F.gumbel_softmax(gate_logits, tau=temperature, hard=False)
            gates_list.append(gate_soft)

            # read from history at gate-weighted positions
            idx = self.hist_idx % self.t_gate
            # reorder history so index 0 = current, T_GATE-1 = oldest
            ordered = torch.roll(self.hist[i], -idx, dims=0)  # (T_GATE, in_dim)
            context = (gate_soft.unsqueeze(-1) * ordered.unsqueeze(0)).sum(1)  # (1, in_dim)

            out = self.slot_projs[i](h)  # h: (1, hidden) → (1, in_dim)
            slot_outputs.append(out)

        return slot_outputs, gates_list

    def tau_star(self, gates_list):
        """mean τ* across slots"""
        vals = [(g.detach().mean(0) * self.positions).sum().item() for g in gates_list]
        return float(np.mean(vals)), float(np.std(vals))


# ─── Two-pathway model ────────────────────────────────────────────────────────

class TwoPathwayModel(nn.Module):
    """
    Fast pathway (fast Lorenz, T_GATE_fast) + Slow pathway (slow Lorenz, T_GATE_slow)
    ISOLATED INPUTS: each pathway sees only its own system's trajectory
    SHARED DECODER: concat of all slot outputs → linear predictions
    """
    def __init__(self, hidden, n_slots_each, t_gate_fast, t_gate_slow, pred_horizons):
        super().__init__()
        self.horizons = pred_horizons
        in_dim = 3  # Lorenz is 3D

        self.fast = SinglePathway(in_dim, hidden, n_slots_each, t_gate_fast, pred_horizons)
        self.slow = SinglePathway(in_dim, hidden, n_slots_each, t_gate_slow, pred_horizons)

        # shared decoder: concat all slot outputs from both pathways
        total_out = 2 * n_slots_each * in_dim
        self.decoder = nn.ModuleList([
            nn.Linear(total_out, in_dim) for _ in pred_horizons
        ])

    def reset(self):
        self.fast.reset()
        self.slow.reset()

    def forward(self, x_fast, x_slow, temperature=1.0):
        fast_outs, fast_gates = self.fast(x_fast, temperature)
        slow_outs, slow_gates = self.slow(x_slow, temperature)

        # concat all outputs → shared decoder
        all_outs = torch.cat(fast_outs + slow_outs, dim=-1)  # (1, total_out)
        preds = [dec(all_outs) for dec in self.decoder]

        return preds, fast_gates, slow_gates


# ─── Training ─────────────────────────────────────────────────────────────────

def run_seed(seed: int) -> dict:
    print(f"\n{'='*60}")
    print(f"  Seed {seed}")
    print(f"{'='*60}")

    # generate trajectories (long enough for both pathways)
    total_steps = N_STEPS + max(PRED_HORIZONS) + N_TRAJ
    fast_traj = make_lorenz_traj(total_steps, FAST_DT, seed=seed).to(DEVICE)
    slow_traj = make_lorenz_traj(total_steps, SLOW_DT, seed=seed + 100).to(DEVICE)

    model = TwoPathwayModel(
        hidden=HIDDEN,
        n_slots_each=N_SLOTS_EACH,
        t_gate_fast=FAST_TGATE,
        t_gate_slow=SLOW_TGATE,
        pred_horizons=PRED_HORIZONS,
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)

    history = []
    model.reset()

    for step in range(N_STEPS):
        temp = TEMP_START * (TEMP_END / TEMP_START) ** (step / N_STEPS)
        t = step + N_TRAJ

        x_fast = fast_traj[t].unsqueeze(0)
        x_slow = slow_traj[t].unsqueeze(0)

        preds, fast_gates, slow_gates = model(x_fast, x_slow, temperature=temp)

        # multi-horizon prediction loss on BOTH systems
        loss = 0.0
        for k_idx, k in enumerate(PRED_HORIZONS):
            # fast prediction targets
            if t + k < len(fast_traj):
                target_fast = fast_traj[t + k].unsqueeze(0)
                loss = loss + F.mse_loss(preds[k_idx], target_fast)
            # slow prediction targets (separate horizons — same k but slow system)
            if t + k < len(slow_traj):
                target_slow = slow_traj[t + k].unsqueeze(0)
                # preds from concat decoder already sees both — use same pred
                loss = loss + F.mse_loss(preds[k_idx], target_slow)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.fast.h_states = [h.detach() for h in model.fast.h_states]
        model.slow.h_states = [h.detach() for h in model.slow.h_states]

        if step % LOG_EVERY == 0 or step == N_STEPS - 1:
            tau_fast_mean, tau_fast_std = model.fast.tau_star(fast_gates)
            tau_slow_mean, tau_slow_std = model.slow.tau_star(slow_gates)
            pct_fast = tau_fast_mean / FAST_TAU_L * 100
            pct_slow = tau_slow_mean / SLOW_TAU_L * 100
            print(f"  step {step:5d} | temp={temp:.3f} | loss={loss.item():.4f}")
            print(f"    fast τ*={tau_fast_mean:.1f} ({pct_fast:.1f}% τ_L_fast={FAST_TAU_L})")
            print(f"    slow τ*={tau_slow_mean:.1f} ({pct_slow:.1f}% τ_L_slow={SLOW_TAU_L:.1f})")
            history.append({
                "step": step, "temp": float(temp), "loss": float(loss.item()),
                "tau_fast": tau_fast_mean, "tau_fast_pct": pct_fast,
                "tau_slow": tau_slow_mean, "tau_slow_pct": pct_slow,
            })

    # final τ* — average over last 1000 steps
    tau_fast_vals = []
    tau_slow_vals = []
    with torch.no_grad():
        for step in range(N_STEPS - 1000, N_STEPS):
            temp = TEMP_END
            t = step + N_TRAJ
            x_fast = fast_traj[t].unsqueeze(0)
            x_slow = slow_traj[t].unsqueeze(0)
            _, fast_gates, slow_gates = model(x_fast, x_slow, temperature=temp)
            tf, _ = model.fast.tau_star(fast_gates)
            ts_val, _ = model.slow.tau_star(slow_gates)
            tau_fast_vals.append(tf)
            tau_slow_vals.append(ts_val)

    tau_fast_final = float(np.mean(tau_fast_vals))
    tau_slow_final = float(np.mean(tau_slow_vals))
    pct_fast_final = tau_fast_final / FAST_TAU_L * 100
    pct_slow_final = tau_slow_final / SLOW_TAU_L * 100

    dhp_fast = tau_fast_final >= 0.70 * FAST_TAU_L
    dhp_slow = tau_slow_final >= 0.70 * SLOW_TAU_L
    interior_fast = tau_fast_final < FAST_TGATE
    interior_slow = tau_slow_final < SLOW_TGATE

    print(f"\n  FINAL (seed {seed}):")
    print(f"    fast τ*={tau_fast_final:.1f} ({pct_fast_final:.1f}% τ_L_fast) "
          f"DHP={'✓' if dhp_fast else '✗'} interior={'✓' if interior_fast else '✗'}")
    print(f"    slow τ*={tau_slow_final:.1f} ({pct_slow_final:.1f}% τ_L_slow) "
          f"DHP={'✓' if dhp_slow else '✗'} interior={'✓' if interior_slow else '✗'}")

    return {
        "seed": seed,
        "tau_fast": tau_fast_final, "tau_fast_pct": pct_fast_final,
        "tau_slow": tau_slow_final, "tau_slow_pct": pct_slow_final,
        "dhp_fast": bool(dhp_fast), "dhp_slow": bool(dhp_slow),
        "interior_fast": bool(interior_fast), "interior_slow": bool(interior_slow),
        "history": history,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("DHP Two-Pathway Isolated — Full Hierarchical Test")
    print(f"Device: {DEVICE}")
    print(f"Fast Lorenz: dt={FAST_DT}, τ_L≈{FAST_TAU_L}, T_GATE={FAST_TGATE}")
    print(f"Slow Lorenz: dt={SLOW_DT}, τ_L≈{SLOW_TAU_L}, T_GATE={SLOW_TGATE}")
    print(f"N_SEEDS={N_SEEDS}, N_STEPS={N_STEPS}")
    print("="*60)

    results = []
    for seed in range(N_SEEDS):
        r = run_seed(seed)
        results.append(r)

    # summary
    print("\n" + "="*60)
    print("  SUMMARY — TWO-PATHWAY ISOLATED")
    print("="*60)
    dhp_both = sum(1 for r in results if r["dhp_fast"] and r["dhp_slow"])
    interior_both = sum(1 for r in results if r["interior_fast"] and r["interior_slow"])
    print(f"  Both pathways DHP pass: {dhp_both}/{N_SEEDS}")
    print(f"  Both pathways interior: {interior_both}/{N_SEEDS}")
    for r in results:
        print(f"  seed {r['seed']}: fast τ*={r['tau_fast']:.1f} ({r['tau_fast_pct']:.1f}%) | "
              f"slow τ*={r['tau_slow']:.1f} ({r['tau_slow_pct']:.1f}%)")

    if dhp_both == N_SEEDS:
        print("\n  ✅ FULL HIERARCHICAL DHP CONFIRMED: both pathways, isolated inputs")
    elif dhp_both >= N_SEEDS // 2:
        print(f"\n  ⚠️  PARTIAL: {dhp_both}/{N_SEEDS} seeds both-pass")
    else:
        print(f"\n  ❌ HIERARCHICAL DHP NOT CONFIRMED: {dhp_both}/{N_SEEDS} seeds both-pass")

    out = {
        "experiment": "dhp_two_pathway_isolated",
        "fast_dt": FAST_DT, "fast_tgate": FAST_TGATE, "fast_tau_L": FAST_TAU_L,
        "slow_dt": SLOW_DT, "slow_tgate": SLOW_TGATE, "slow_tau_L": SLOW_TAU_L,
        "n_seeds": N_SEEDS, "n_steps": N_STEPS,
        "dhp_both_count": dhp_both, "interior_both_count": interior_both,
        "seeds": results,
    }
    outfile = OUT_DIR / "dhp_two_pathway_results.json"
    with open(outfile, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results → {outfile}")
