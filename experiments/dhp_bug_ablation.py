#!/usr/bin/env python3
"""
DHP Bug Ablation — Individual Bug Contribution Test
====================================================
v38 had THREE bugs. v40 fixed all three. This tests each bug in isolation
to determine which were individually necessary/sufficient to prevent DHP.

Conditions:
  A) ALL_BUGS     — shared proj + mean pooling + Adam/1e-3       → expect 0/5 (replicates v38)
  B) ONLY_BUG1    — shared proj + FIXED pooling + FIXED optim     → shared proj alone = no DHP?
  C) ONLY_BUG2    — FIXED proj  + mean pooling  + FIXED optim     → mean pool alone = no DHP?
  D) ONLY_BUG3    — FIXED proj  + FIXED pooling + Adam/1e-3       → wrong optim alone = no DHP?
  E) NO_BUGS      — per-slot proj + concat dec + AdamW/cosine     → expect 5/5 (replicates v40)

Paper 5: "Scout Emergence Conditions in Continuous-Time Recurrent Models"
Section: Ablation of necessary conditions for scout emergence

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
N_SLOTS  = 4
DIM      = 64
STEPS    = 30000      # slightly more than mechanism proof for cleaner convergence
LR_GOOD  = 2e-4      # v40 correct LR
LR_BAD   = 1e-3       # v38 bad LR (Adam, no cosine)
BATCH    = 64
SEEDS    = [0, 1, 2, 3, 42]
OUT_FILE = Path("/home/ai/dhp_bug_ablation.json")


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


# ── Configurable model — any combination of bugs ──────────────────────────────
class AblationModel(nn.Module):
    def __init__(self, shared_proj=False, mean_pool=False):
        """
        shared_proj=True  → bug 1 (all slots share one Linear)
        mean_pool=True    → bug 2 (average slots before decode, not concat)
        optimizer choice  → bug 3 (controlled at training time)
        """
        super().__init__()
        self.shared_proj = shared_proj
        self.mean_pool   = mean_pool
        self.gate_logits = nn.Parameter(torch.zeros(N_SLOTS, T_GATE))

        if shared_proj:
            self.proj = nn.Linear(3, DIM)          # BUG 1: shared
        else:
            self.slot_proj = nn.ModuleList([nn.Linear(3, DIM) for _ in range(N_SLOTS)])

        if mean_pool:
            self.head = nn.Linear(DIM, 3)          # BUG 2: mean then decode
        else:
            self.dec = nn.Linear(DIM * N_SLOTS, 3) # FIX: concat decode

    def forward(self, win, temp):
        B = win.size(0)
        gate = F.softmax(self.gate_logits / temp, dim=-1)
        win_exp = win.unsqueeze(1).expand(B, N_SLOTS, T_GATE, 3)
        weighted = (win_exp * gate.unsqueeze(0).unsqueeze(-1)).sum(2)  # (B,S,3)

        if self.shared_proj:
            slots = self.proj(weighted)             # same weights every slot
        else:
            slots = torch.stack([self.slot_proj[i](weighted[:,i,:]) for i in range(N_SLOTS)], 1)

        if self.mean_pool:
            z = slots.mean(1)
            return self.head(z), gate
        else:
            return self.dec(slots.view(B, -1)), gate

    def delays(self):
        g = self.gate_logits.detach().softmax(-1)
        idx = torch.arange(T_GATE, device=g.device, dtype=torch.float32)
        return ((T_GATE-1) - (g * idx).sum(-1)).cpu().numpy()


# ── Train ─────────────────────────────────────────────────────────────────────
def run(label, shared_proj, mean_pool, bad_optimizer, seed, data):
    torch.manual_seed(seed); np.random.seed(seed)
    model = AblationModel(shared_proj=shared_proj, mean_pool=mean_pool).to(DEVICE)

    if bad_optimizer:
        opt = torch.optim.Adam(model.parameters(), lr=LR_BAD)
        sch = None
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=LR_GOOD, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)

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
        opt.zero_grad(); loss.backward(); opt.step()
        if sch: sch.step()

        if step % 5000 == 0:
            d = model.delays()
            v = float(np.var(d))
            var_history.append({"step": step, "var": v, "delays": d.tolist()})
            bugs = f"proj={'shared' if shared_proj else 'perslot'} pool={'mean' if mean_pool else 'concat'} optim={'Adam/1e-3' if bad_optimizer else 'AdamW/cos'}"
            print(f"  [{label}] seed={seed} step={step:5d} var={v:.4f} delays={[f'{x:.1f}' for x in d]}")

    final = model.delays()
    return {
        "seed": seed, "label": label,
        "shared_proj": shared_proj, "mean_pool": mean_pool, "bad_optimizer": bad_optimizer,
        "final_var": float(np.var(final)),
        "final_tau_star": float(final.max()),
        "final_delays": final.tolist(),
        "diverged": bool(np.var(final) > 0.1),
        "var_history": var_history,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
CONDITIONS = [
    # label        shared_proj  mean_pool  bad_optim
    ("ALL_BUGS",   True,        True,      True),    # v38: expect 0/5
    ("BUG1_ONLY",  True,        False,     False),   # shared proj alone
    ("BUG2_ONLY",  False,       True,      False),   # mean pool alone
    ("BUG3_ONLY",  False,       False,     True),    # bad optim alone
    ("NO_BUGS",    False,       False,     False),   # v40: expect 5/5
]

print(f"DHP Bug Ablation | device={DEVICE}")
print(f"5 conditions × {len(SEEDS)} seeds × {STEPS} steps")
print(f"Hypothesis: each bug individually sufficient to prevent scout emergence")
print()

rng  = np.random.default_rng(0)
data = gen_data(2000, 200, rng)
print(f"Data: {data.shape}")

results = {label: {} for label, *_ in CONDITIONS}

for label, shared_proj, mean_pool, bad_optim in CONDITIONS:
    print(f"\n{'='*50}")
    print(f"CONDITION: {label} (proj={'shared' if shared_proj else 'perslot'}, pool={'mean' if mean_pool else 'concat'}, optim={'Adam/1e-3' if bad_optim else 'AdamW/cos'})")
    for seed in SEEDS:
        results[label][str(seed)] = run(label, shared_proj, mean_pool, bad_optim, seed, data)

# Summary
print("\n" + "="*60)
print("ABLATION SUMMARY")
print("="*60)
summary = {}
for label, *_ in CONDITIONS:
    vars_  = [results[label][str(s)]["final_var"] for s in SEEDS]
    divs   = sum(results[label][str(s)]["diverged"] for s in SEEDS)
    mean_v = float(np.mean(vars_))
    summary[label] = {"mean_var": mean_v, "diverged_count": divs, "n_seeds": len(SEEDS)}
    print(f"{label:12s}: mean_var={mean_v:.4f} | diverged={divs}/{len(SEEDS)}")

results["summary"] = summary

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_FILE, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults → {OUT_FILE}")
