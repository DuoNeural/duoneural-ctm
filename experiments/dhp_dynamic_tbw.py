"""
dhp_dynamic_tbw_longphase.py — DHP Dynamic TBW Follow-up: Long Phase Adaptation Test
DuoNeural | Archon | 2026-05-07

QUESTION (from dynamic TBW result):
τ* was non-adaptive across 4000-step phase switches. Is this because:
  (A) 4000 steps is not long enough for re-calibration, OR
  (B) the gate fundamentally cannot re-calibrate once set?

This experiment uses 10,000 steps per phase (2.5× longer) to test (A).

If τ* still doesn't move after 10k steps of Rössler → non-adaptive is structural.
If τ* begins drifting after 5-8k steps → timescale-limited adaptation exists.

Architecture: v40 (N_SLOTS=4, per-slot GRU, concat decoder)
T_GATE: 64, N_SEEDS: 4, 30,000 steps total

Phase 1  (steps  0–10k):  Lorenz-3D   (τ_L≈22)   → expect saturation ~61
Phase 2  (steps 10k–20k): Rössler c=10 (τ_L≈86.5) → watch for drift in τ*
Phase 3  (steps 20k–30k): Lorenz-3D   again        → watch for return

Key diagnostic: plot τ* within each phase over time.
  - Constant τ* → structurally locked (finding B confirmed)
  - Slow drift → timescale-limited adaptation (finding A, follow up with 40k phases)

Also adding: log τ* variance across slots (gate_spread) — if adaptation is beginning,
individual slots may diverge before the mean shifts.

Run: TORCHDYNAMO_DISABLE=1 python3 dhp_dynamic_tbw_longphase.py
"""

import sys, json, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

print("=" * 70)
print("DHP Dynamic TBW — Long-Phase Adaptation Test (follow-up)")
print(f"  Architecture: v40 (N_SLOTS=4, GRU proj, concat decoder)")
print(f"  T_GATE=64, DT=0.05, N_SEEDS=4, 30,000 steps total")
print(f"  Phase 1: Lorenz  0–10k   (τ_L≈22)")
print(f"  Phase 2: Rössler 10k–20k (τ_L≈86.5)")
print(f"  Phase 3: Lorenz  20k–30k (τ_L≈22, return?)")
print(f"  Question: is non-adaptation structural or timescale-limited?")
print("=" * 70)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {DEVICE}")

# ──────────────────── Dynamical Systems ──────────────────────

def lorenz_rk4(state, dt=0.05, sigma=10., rho=28., beta=8./3.):
    def f(s):
        x, y, z = s
        return torch.stack([sigma*(y-x), x*(rho-z)-y, x*y-beta*z])
    k1 = f(state)
    k2 = f(state + 0.5*dt*k1)
    k3 = f(state + 0.5*dt*k2)
    k4 = f(state + dt*k3)
    return state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def rossler_rk4(state, dt=0.05, a=0.2, b=0.2, c=10.0):
    def f(s):
        x, y, z = s
        return torch.stack([-y-z, x+a*y, b+z*(x-c)])
    k1 = f(state)
    k2 = f(state + 0.5*dt*k1)
    k3 = f(state + 0.5*dt*k2)
    k4 = f(state + dt*k3)
    return state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def generate_lorenz(n_traj, traj_len, dt=0.05, warmup=500):
    trajs = []
    for _ in range(n_traj):
        s = torch.randn(3) * 5.0
        for _ in range(warmup): s = lorenz_rk4(s, dt=dt)
        traj = [s]
        for _ in range(traj_len - 1):
            s = lorenz_rk4(s, dt=dt)
            traj.append(s)
        trajs.append(torch.stack(traj))
    data = torch.stack(trajs)
    return (data - data.mean(dim=(0,1))) / data.std(dim=(0,1)).clamp(min=1e-6)

def generate_rossler(n_traj, traj_len, dt=0.05, c=10.0, warmup=500):
    trajs = []
    for _ in range(n_traj):
        s = torch.randn(3) * 2.0
        for _ in range(warmup): s = rossler_rk4(s, dt=dt, c=c)
        traj = [s]
        for _ in range(traj_len - 1):
            s = rossler_rk4(s, dt=dt, c=c)
            traj.append(s)
        trajs.append(torch.stack(traj))
    data = torch.stack(trajs)
    return (data - data.mean(dim=(0,1))) / data.std(dim=(0,1)).clamp(min=1e-6)

# ──────────────────── v40 CTM Architecture ───────────────────

class SlotGate(nn.Module):
    def __init__(self, d_in, d_hidden, T_GATE):
        super().__init__()
        self.T_GATE = T_GATE
        self.proj = nn.Linear(d_in, d_hidden)
        self.gru = nn.GRU(d_hidden, d_hidden, batch_first=True)
        self.pos_emb = nn.Embedding(T_GATE, d_hidden)
        self.gate_head = nn.Linear(d_hidden, T_GATE)

    def forward(self, history, temperature=1.0):
        B, T, D = history.shape
        x = self.proj(history)
        pos = self.pos_emb(torch.arange(T, device=history.device))
        x = x + pos.unsqueeze(0)
        out, _ = self.gru(x)
        query = out[:, -1, :]
        logits = self.gate_head(query)
        if self.training:
            return F.gumbel_softmax(logits, tau=temperature, hard=False)
        else:
            return F.softmax(logits / temperature, dim=-1)

class SlotCTM(nn.Module):
    def __init__(self, d_in=3, d_hidden=128, d_out=3, N_SLOTS=4, T_GATE=64):
        super().__init__()
        self.T_GATE = T_GATE
        self.N_SLOTS = N_SLOTS
        self.gates = nn.ModuleList([SlotGate(d_in, d_hidden, T_GATE) for _ in range(N_SLOTS)])
        self.slot_proj = nn.ModuleList([nn.Linear(d_in, d_hidden) for _ in range(N_SLOTS)])
        self.decoder = nn.Linear(N_SLOTS * d_hidden, d_out)

    def forward(self, history, temperature=1.0):
        slot_reps, gates = [], []
        for s in range(self.N_SLOTS):
            gate = self.gates[s](history, temperature)
            gates.append(gate)
            weighted = (gate.unsqueeze(-1) * history).sum(dim=1)
            slot_reps.append(self.slot_proj[s](weighted))
        return self.decoder(torch.cat(slot_reps, dim=-1)), gates

def tau_star_and_spread(gates_list):
    """Return mean τ* AND per-slot τ* list (for spread analysis)."""
    T = gates_list[0].shape[-1]
    positions = torch.arange(T, dtype=torch.float32, device=gates_list[0].device)
    per_slot = []
    for g in gates_list:
        mean_g = g.detach().mean(0)
        per_slot.append((mean_g * positions).sum().item())
    return float(np.mean(per_slot)), float(np.std(per_slot)), per_slot

def get_batch(trajs, T_GATE, batch_size=32):
    n_traj, traj_len, d = trajs.shape
    max_start = traj_len - T_GATE - 1
    traj_idxs = torch.randint(0, n_traj, (batch_size,))
    start_idxs = torch.randint(0, max_start, (batch_size,))
    history = torch.stack([trajs[t, s:s+T_GATE] for t, s in zip(traj_idxs, start_idxs)])
    target  = torch.stack([trajs[t, s+T_GATE]   for t, s in zip(traj_idxs, start_idxs)])
    return history, target

# ──────────────────── Config ─────────────────────────────────

N_STEPS   = 30000
T_GATE    = 64
N_SLOTS   = 4
HIDDEN    = 128
N_SEEDS   = 4
BATCH     = 32
LR        = 3e-4
LOG_EVERY = 200         # log every 200 steps (150 log points per seed)

T_START, T_END = 2.0, 0.1

PHASE_1_END = 10000   # Lorenz
PHASE_2_END = 20000   # Rössler
PHASE_3_END = 30000   # Lorenz

TAU_L_LORENZ  = 22.0
TAU_L_ROSSLER = 86.5

print("\nGenerating trajectories...")
N_TRAJ   = 64
TRAJ_LEN = T_GATE * 20

lorenz_trajs  = generate_lorenz(N_TRAJ, TRAJ_LEN, dt=0.05).to(DEVICE)
rossler_trajs = generate_rossler(N_TRAJ, TRAJ_LEN, dt=0.05, c=10.0).to(DEVICE)
print(f"  Lorenz: {lorenz_trajs.shape}  Rössler: {rossler_trajs.shape}")
print("  Done.\n")

all_results = []

for seed in range(N_SEEDS):
    print(f"\n{'─'*70}")
    print(f"SEED {seed} / {N_SEEDS-1}")
    print(f"{'─'*70}")

    torch.manual_seed(seed + 42)
    model = SlotCTM(d_in=3, d_hidden=HIDDEN, N_SLOTS=N_SLOTS, T_GATE=T_GATE).to(DEVICE)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    tau_history = []

    for step in range(N_STEPS):
        frac = step / N_STEPS
        temp = T_START * (T_END / T_START) ** frac

        if step < PHASE_1_END:
            trajs, phase, tau_l = lorenz_trajs,  "lorenz",  TAU_L_LORENZ
        elif step < PHASE_2_END:
            trajs, phase, tau_l = rossler_trajs, "rossler", TAU_L_ROSSLER
        else:
            trajs, phase, tau_l = lorenz_trajs,  "lorenz2", TAU_L_LORENZ

        history, target = get_batch(trajs, T_GATE, BATCH)
        history, target = history.to(DEVICE), target.to(DEVICE)

        pred, gates = model(history, temperature=temp)
        loss = F.mse_loss(pred, target)
        opt.zero_grad(); loss.backward(); opt.step()

        if step % LOG_EVERY == 0 or step == N_STEPS - 1:
            ts, spread, per_slot = tau_star_and_spread(gates)
            tau_frac = ts / tau_l
            dhp = "✓" if tau_frac >= 0.70 else "✗"
            print(
                f"step {step:6d} | {phase:8s} | temp={temp:.3f} | "
                f"loss={loss.item():.4f} | τ*={ts:.1f}±{spread:.1f} ({tau_frac*100:.1f}%) | {dhp}"
            )
            tau_history.append({
                "step": step, "phase": phase, "tau_star": ts,
                "tau_spread": spread, "per_slot": per_slot,
                "tau_l": tau_l, "tau_frac": tau_frac,
                "loss": loss.item(), "temperature": temp,
            })

    # Per-phase summaries — split into early/late halves to detect drift
    def phase_stats(ph_name, tau_l_val):
        pts = [r for r in tau_history if r["phase"] == ph_name]
        if not pts: return {}
        early = np.mean([r["tau_star"] for r in pts[:len(pts)//2]])
        late  = np.mean([r["tau_star"] for r in pts[len(pts)//2:]])
        drift = late - early
        return {"early": early, "late": late, "drift": drift,
                "early_frac": early/tau_l_val, "late_frac": late/tau_l_val}

    p1 = phase_stats("lorenz",  TAU_L_LORENZ)
    p2 = phase_stats("rossler", TAU_L_ROSSLER)
    p3 = phase_stats("lorenz2", TAU_L_LORENZ)

    print(f"\nSeed {seed} — phase analysis (early vs late half):")
    print(f"  P1 Lorenz:   early τ*={p1['early']:.1f} ({p1['early_frac']*100:.1f}%)  "
          f"late τ*={p1['late']:.1f} ({p1['late_frac']*100:.1f}%)  drift={p1['drift']:+.2f}")
    print(f"  P2 Rössler:  early τ*={p2['early']:.1f} ({p2['early_frac']*100:.1f}%)  "
          f"late τ*={p2['late']:.1f} ({p2['late_frac']*100:.1f}%)  drift={p2['drift']:+.2f}")
    print(f"  P3 Lorenz2:  early τ*={p3['early']:.1f} ({p3['early_frac']*100:.1f}%)  "
          f"late τ*={p3['late']:.1f} ({p3['late_frac']*100:.1f}%)  drift={p3['drift']:+.2f}")

    all_results.append({
        "seed": seed,
        "p1": p1, "p2": p2, "p3": p3,
        "tau_history": tau_history,
    })

# ──────────────────── Final Summary ──────────────────────────

print("\n" + "=" * 70)
print("LONG-PHASE ADAPTATION TEST — FINAL RESULTS")
print("=" * 70)

p2_drifts = [r["p2"]["drift"] for r in all_results]
p3_drifts = [r["p3"]["drift"] for r in all_results]
p2_late   = [r["p2"]["late"]  for r in all_results]
p1_late   = [r["p1"]["late"]  for r in all_results]

print(f"\nPhase 2 (Rössler) internal drift (early→late half): {np.mean(p2_drifts):+.2f} ± {np.std(p2_drifts):.2f}")
print(f"Phase 3 (Lorenz2) internal drift (early→late half): {np.mean(p3_drifts):+.2f} ± {np.std(p3_drifts):.2f}")
print(f"P1→P2 cross-phase shift: {np.mean(p2_late)-np.mean(p1_late):+.2f}")

print("\nINTERPRETATION:")
p2_drift_mean = np.mean(p2_drifts)
if abs(p2_drift_mean) < 0.5:
    print("  ❌ STRUCTURALLY LOCKED: τ* shows no drift even over 10k steps")
    print("  → Non-adaptation is architectural, not timescale-limited")
    print("  → DHP gate is permanently set after initial convergence")
elif p2_drift_mean > 0.5:
    print("  ✅ SLOW ADAPTATION: τ* drifts upward during Rössler phase")
    print(f"  → Adaptation exists but requires >{PHASE_1_END} steps")
    print("  → Follow-up: 40k-step phases to see if full re-calibration occurs")
else:
    print("  ↕ WEAK SIGNAL: slight downward drift, borderline")
    print("  → Inconclusive — try 40k step phases")

print("\n" + "=" * 70)

out_dir = Path("dhp_longphase_output")
out_dir.mkdir(exist_ok=True)
out_file = out_dir / "dhp_longphase_results.json"
with open(out_file, "w") as f:
    json.dump({
        "experiment": "dhp_dynamic_tbw_longphase",
        "config": {
            "N_STEPS": N_STEPS, "T_GATE": T_GATE, "N_SLOTS": N_SLOTS,
            "HIDDEN": HIDDEN, "N_SEEDS": N_SEEDS,
            "PHASE_LENGTHS": [PHASE_1_END, PHASE_2_END-PHASE_1_END, PHASE_3_END-PHASE_2_END],
            "TAU_L_LORENZ": TAU_L_LORENZ, "TAU_L_ROSSLER": TAU_L_ROSSLER,
        },
        "p2_drift_mean": float(np.mean(p2_drifts)),
        "p2_drift_std":  float(np.std(p2_drifts)),
        "p3_drift_mean": float(np.mean(p3_drifts)),
        "seeds": all_results,
    }, f, indent=2)

print(f"\nResults → {out_file}")
print("Long-phase adaptation test complete.")
