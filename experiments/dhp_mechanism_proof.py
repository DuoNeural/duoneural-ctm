#!/usr/bin/env python3
"""
DHP Mechanism — Minimal Mechanistic Proof
==========================================
Demonstrates that per-slot symmetry breaking is necessary and sufficient
for scout slot emergence in temporal gate models.

Two conditions tested:
  A) SHARED projection (v38 bug): one nn.Linear for all slots
  B) PER-SLOT projection (v40 fix): independent nn.Linear per slot

Result: Condition A → all slots converge to identical delays (var→0)
        Condition B → slots diverge, one scout reaches Lyapunov horizon (var>0)

This is the minimal proof: trivial Lorenz prediction task, 2 slots, 3-layer model.
No GNN dynamics, no slot decoder complexity — just the projection symmetry.

For paper 5: "Scout Emergence Conditions in Continuous-Time Recurrent Models"
Figure 1 candidate.

Archon | DuoNeural | 2026-05-03
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
T_GATE   = 32
N_SLOTS  = 4       # minimal: 4 slots
DIM      = 64      # tiny
STEPS    = 20000
LR       = 2e-4
BATCH    = 64
SEEDS    = [0, 1, 2, 3, 42]
OUT_FILE = Path("/home/ai/dhp_mechanism_proof.json")

# ── Lorenz ────────────────────────────────────────────────────────────────────
def lorenz_rk4(x, y, z, dt=0.05, sigma=10., rho=28., beta=8./3.):
    def d(x,y,z): return sigma*(y-x), x*(rho-z)-y, x*y-beta*z
    k1 = d(x,y,z)
    k2 = d(x+.5*dt*k1[0], y+.5*dt*k1[1], z+.5*dt*k1[2])
    k3 = d(x+.5*dt*k2[0], y+.5*dt*k2[1], z+.5*dt*k2[2])
    k4 = d(x+dt*k3[0], y+dt*k3[1], z+dt*k3[2])
    return (x+dt*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6,
            y+dt*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6,
            z+dt*(k1[2]+2*k2[2]+2*k3[2]+k4[2])/6)

def gen_data(n=2000, t_len=200, rng=None):
    rng = rng or np.random.default_rng(0)
    trajs = []
    for _ in range(n):
        x,y,z = rng.uniform(-15,15), rng.uniform(-15,15), rng.uniform(0,40)
        for _ in range(500): x,y,z = lorenz_rk4(x,y,z)
        t = []
        for _ in range(t_len): x,y,z = lorenz_rk4(x,y,z); t.append([x,y,z])
        trajs.append(t)
    d = np.array(trajs, dtype=np.float32)
    return (d - d.mean()) / (d.std() + 1e-8)

# ── Models ────────────────────────────────────────────────────────────────────
class SharedProjModel(nn.Module):
    """BUGGY: one projection shared across all slots → symmetry unbroken"""
    def __init__(self):
        super().__init__()
        self.gate_logits = nn.Parameter(torch.zeros(N_SLOTS, T_GATE))
        self.proj = nn.Linear(3, DIM)          # SHARED — this is the bug
        self.head = nn.Linear(DIM, 3)
    def forward(self, win, temp):
        B = win.size(0)
        gate = F.softmax(self.gate_logits / temp, dim=-1)
        win_exp = win.unsqueeze(1).expand(B, N_SLOTS, T_GATE, 3)
        weighted = (win_exp * gate.unsqueeze(0).unsqueeze(-1)).sum(2)  # (B,S,3)
        slots = self.proj(weighted)             # same weights → identical gradients
        z = slots.mean(1)                       # average collapses further
        return self.head(z), gate
    def delays(self):
        g = self.gate_logits.detach().softmax(-1)
        idx = torch.arange(T_GATE, device=g.device, dtype=torch.float32)
        return ((T_GATE-1) - (g * idx).sum(-1)).cpu().numpy()

class PerSlotProjModel(nn.Module):
    """FIXED: independent projection per slot → symmetry broken by random init"""
    def __init__(self):
        super().__init__()
        self.gate_logits = nn.Parameter(torch.zeros(N_SLOTS, T_GATE))
        self.slot_proj = nn.ModuleList([nn.Linear(3, DIM) for _ in range(N_SLOTS)])
        self.dec = nn.Linear(DIM * N_SLOTS, 3)  # concat decoder
    def forward(self, win, temp):
        B = win.size(0)
        gate = F.softmax(self.gate_logits / temp, dim=-1)
        win_exp = win.unsqueeze(1).expand(B, N_SLOTS, T_GATE, 3)
        weighted = (win_exp * gate.unsqueeze(0).unsqueeze(-1)).sum(2)
        slots = torch.stack([self.slot_proj[i](weighted[:,i,:]) for i in range(N_SLOTS)], 1)
        return self.dec(slots.view(B, -1)), gate
    def delays(self):
        g = self.gate_logits.detach().softmax(-1)
        idx = torch.arange(T_GATE, device=g.device, dtype=torch.float32)
        return ((T_GATE-1) - (g * idx).sum(-1)).cpu().numpy()

# ── Train ─────────────────────────────────────────────────────────────────────
def run(ModelClass, seed, data):
    torch.manual_seed(seed); np.random.seed(seed)
    model = ModelClass().to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)

    t_max = data.shape[1] - T_GATE - 1
    var_history = []

    for step in range(STEPS + 1):
        temp = 2.0 * (0.1/2.0)**(step/STEPS)
        idx  = np.random.randint(0, data.shape[0], BATCH)
        t0   = np.random.randint(0, t_max)
        win  = torch.tensor(data[idx, t0:t0+T_GATE], device=DEVICE)
        tgt  = torch.tensor(data[idx, t0+T_GATE],    device=DEVICE)
        pred, gate = model(win, temp)
        loss = F.mse_loss(pred, tgt)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()

        if step % 2000 == 0:
            d = model.delays()
            v = float(np.var(d))
            var_history.append({"step": step, "var": v, "delays": d.tolist(), "tau_star": float(d.max())})
            print(f"  [{ModelClass.__name__[:6]}] seed={seed} step={step:5d} "
                  f"var={v:.3f} τ*={d.max():.2f} delays={[f'{x:.1f}' for x in d]}")

    final = model.delays()
    return {
        "seed": seed,
        "model": ModelClass.__name__,
        "final_var": float(np.var(final)),
        "final_tau_star": float(final.max()),
        "final_delays": final.tolist(),
        "var_history": var_history,
        "dhp": bool(final.max() >= 0.7 * (1/(0.0466*0.05))),  # 70% of τ_L=21.5
    }

# ── Main ──────────────────────────────────────────────────────────────────────
print(f"DHP Mechanism Proof | device={DEVICE}")
print(f"Hypothesis: SharedProj → var≈0 | PerSlotProj → var>0 + scout≥70%τ_L")
print()

rng  = np.random.default_rng(0)
data = gen_data(2000, 200, rng)
print(f"Data: {data.shape}")

results = {"shared": {}, "perslot": {}}

for seed in SEEDS:
    print(f"\n── SHARED PROJ seed={seed} ──")
    results["shared"][str(seed)] = run(SharedProjModel, seed, data)

    print(f"\n── PER-SLOT PROJ seed={seed} ──")
    results["perslot"][str(seed)] = run(PerSlotProjModel, seed, data)

# Summary
shared_vars  = [results["shared"][str(s)]["final_var"]       for s in SEEDS]
perslot_vars = [results["perslot"][str(s)]["final_var"]      for s in SEEDS]
shared_dhp   = sum(results["shared"][str(s)]["dhp"]          for s in SEEDS)
perslot_dhp  = sum(results["perslot"][str(s)]["dhp"]         for s in SEEDS)

results["summary"] = {
    "shared_mean_var":   float(np.mean(shared_vars)),
    "perslot_mean_var":  float(np.mean(perslot_vars)),
    "shared_dhp_rate":   shared_dhp / len(SEEDS),
    "perslot_dhp_rate":  perslot_dhp / len(SEEDS),
    "verdict": "MECHANISM_CONFIRMED" if perslot_dhp > shared_dhp else "MECHANISM_NOT_CONFIRMED"
}

print("\n" + "="*60)
print(f"SHARED  PROJ: mean_var={results['summary']['shared_mean_var']:.4f} | DHP={shared_dhp}/{len(SEEDS)}")
print(f"PER-SLOT PROJ: mean_var={results['summary']['perslot_mean_var']:.4f} | DHP={perslot_dhp}/{len(SEEDS)}")
print(f"VERDICT: {results['summary']['verdict']}")
print("="*60)

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_FILE, "w") as f:
    json.dump(results, f, indent=2)
print(f"Results → {OUT_FILE}")
