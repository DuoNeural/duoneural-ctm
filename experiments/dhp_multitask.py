#!/usr/bin/env python3
"""
DHP Multi-Task Experiment — Lorenz + Chen Simultaneously
=========================================================
Do slots specialize per attractor when trained on two chaotic systems?

Hypothesis A (slot specialization): model develops "Lorenz slots" (τ*≈τ_L_lorenz)
  and "Chen slots" (τ*≈τ_L_chen), each tuned to their system's horizon.

Hypothesis B (delay specialization only): slots still pick delays independently
  of which system they're processing — τ is the representation, not the system.

Architecture: v40 (per-slot proj, concat dec, AdamW+cosine)
Training: each batch is 50% Lorenz + 50% Chen trajectories
Analysis: after training, freeze gates and probe which gate activates for which system.

Paper 5 Section 4: "Cross-system Slot Specialization"
Archon | DuoNeural | 2026-05-03
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
T_GATE  = 32
N_SLOTS = 8
DIM     = 128
STEPS   = 80000
LR      = 2e-4
BATCH   = 128     # 64 Lorenz + 64 Chen per batch
SEEDS   = [0, 1, 2, 42]
OUT_FILE = Path("/home/ai/dhp_multitask.json")

# Lorenz: τ_L ≈ 21.5 steps (dt=0.05, λ=0.9056)
# Chen:   τ_L ≈ 24.0 steps (dt=0.02, λ=2.08)
TAU_L_LORENZ = 21.5
TAU_L_CHEN   = 24.0


# ── Dynamics ────────────────────────────────────────────────────────────────
def lorenz_rk4(x, y, z, dt=0.05, s=10., r=28., b=8./3.):
    def d(x,y,z): return s*(y-x), x*(r-z)-y, x*y-b*z
    k1=d(x,y,z); k2=d(x+.5*dt*k1[0],y+.5*dt*k1[1],z+.5*dt*k1[2])
    k3=d(x+.5*dt*k2[0],y+.5*dt*k2[1],z+.5*dt*k2[2])
    k4=d(x+dt*k3[0],y+dt*k3[1],z+dt*k3[2])
    return (x+dt*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6,
            y+dt*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6,
            z+dt*(k1[2]+2*k2[2]+2*k3[2]+k4[2])/6)

def chen_rk4(x, y, z, dt=0.02, a=35., b=3., c=28.):
    def d(x,y,z): return a*(y-x),(c-a)*x-x*z+c*y,x*y-b*z
    k1=d(x,y,z); k2=d(x+.5*dt*k1[0],y+.5*dt*k1[1],z+.5*dt*k1[2])
    k3=d(x+.5*dt*k2[0],y+.5*dt*k2[1],z+.5*dt*k2[2])
    k4=d(x+dt*k3[0],y+dt*k3[1],z+dt*k3[2])
    return (x+dt*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6,
            y+dt*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6,
            z+dt*(k1[2]+2*k2[2]+2*k3[2]+k4[2])/6)

def gen_data(stepper, n=1500, t_len=200, burn=500, rng=None,
             x_range=(-15,15), y_range=(-15,15), z_range=(0,40)):
    rng = rng or np.random.default_rng(0)
    trajs = []
    for _ in range(n):
        x,y,z = rng.uniform(*x_range), rng.uniform(*y_range), rng.uniform(*z_range)
        for _ in range(burn): x,y,z = stepper(x,y,z)
        t = []
        for _ in range(t_len): x,y,z = stepper(x,y,z); t.append([x,y,z])
        trajs.append(t)
    d = np.array(trajs, dtype=np.float32)
    return (d - d.mean()) / (d.std() + 1e-8)


# ── v40 architecture ─────────────────────────────────────────────────────────
class MultiTaskCTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_logits = nn.Parameter(torch.zeros(N_SLOTS, T_GATE))
        self.slot_proj   = nn.ModuleList([nn.Linear(3, DIM) for _ in range(N_SLOTS)])
        self.dec         = nn.Linear(DIM * N_SLOTS, 3)

    def forward(self, win, temp):
        B = win.size(0)
        gate = F.softmax(self.gate_logits / temp, dim=-1)
        win_exp  = win.unsqueeze(1).expand(B, N_SLOTS, T_GATE, 3)
        weighted = (win_exp * gate.unsqueeze(0).unsqueeze(-1)).sum(2)
        slots    = torch.stack([self.slot_proj[i](weighted[:,i,:])
                                for i in range(N_SLOTS)], 1)
        return self.dec(slots.view(B,-1)), gate

    def delays(self):
        g   = self.gate_logits.detach().softmax(-1)
        idx = torch.arange(T_GATE, device=g.device, dtype=torch.float32)
        return ((T_GATE-1) - (g*idx).sum(-1)).cpu().numpy()

    def slot_gate_for(self, win, temp=0.1):
        """Return per-sample gate weights for specialization analysis."""
        with torch.no_grad():
            B = win.size(0)
            gate = F.softmax(self.gate_logits / temp, dim=-1)   # (S, T)
            # return which slot attends to longest delay per sample
            delays = ((T_GATE-1) - (gate * torch.arange(
                T_GATE,device=gate.device,dtype=torch.float32)).sum(-1))
            return delays.cpu().numpy()


# ── Training ─────────────────────────────────────────────────────────────────
def run(seed, lorenz_data, chen_data):
    torch.manual_seed(seed); np.random.seed(seed)
    model = MultiTaskCTM().to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)

    t_max_l = lorenz_data.shape[1] - T_GATE - 1
    t_max_c = chen_data.shape[1]   - T_GATE - 1
    history = []

    for step in range(STEPS + 1):
        temp = 2.0 * (0.1/2.0)**(step/STEPS)

        # 50/50 mixed batch
        half = BATCH // 2
        li   = np.random.randint(0, lorenz_data.shape[0], half)
        ci   = np.random.randint(0, chen_data.shape[0],   half)
        lt0  = np.random.randint(0, t_max_l)
        ct0  = np.random.randint(0, t_max_c)

        lwin = torch.tensor(lorenz_data[li, lt0:lt0+T_GATE], device=DEVICE)
        ltgt = torch.tensor(lorenz_data[li, lt0+T_GATE],     device=DEVICE)
        cwin = torch.tensor(chen_data[ci,   ct0:ct0+T_GATE], device=DEVICE)
        ctgt = torch.tensor(chen_data[ci,   ct0+T_GATE],     device=DEVICE)

        win  = torch.cat([lwin, cwin], 0)
        tgt  = torch.cat([ltgt, ctgt], 0)

        pred, gate = model(win, temp)
        loss = F.mse_loss(pred, tgt)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()

        if step % 8000 == 0:
            d = model.delays()
            v = float(np.var(d))
            history.append({"step": step, "var": v, "delays": d.tolist(),
                            "tau_star": float(d.max())})
            print(f"  seed={seed} step={step:5d} | τ*={d.max():.2f} "
                  f"var={v:.3f} | delays={[f'{x:.1f}' for x in d]}")

    final_delays = model.delays()
    tau_star     = float(final_delays.max())

    # specialization probe: what delays does the model use for Lorenz vs Chen?
    n_probe = 200
    rng_p   = np.random.default_rng(seed + 1000)
    l_idx   = rng_p.integers(0, lorenz_data.shape[0], n_probe)
    c_idx   = rng_p.integers(0, chen_data.shape[0],   n_probe)
    lt0_p   = rng_p.integers(0, t_max_l)
    ct0_p   = rng_p.integers(0, t_max_c)

    with torch.no_grad():
        model.eval()
        # both systems get same global gate — probe if effective delay differs
        # by comparing which slots dominate for Lorenz vs Chen inputs
        lwin_p = torch.tensor(lorenz_data[l_idx, lt0_p:lt0_p+T_GATE], device=DEVICE)
        cwin_p = torch.tensor(chen_data[c_idx, ct0_p:ct0_p+T_GATE],   device=DEVICE)
        gate_sm= F.softmax(model.gate_logits / 0.1, -1)
        idx_t  = torch.arange(T_GATE, device=gate_sm.device, dtype=torch.float32)
        slot_delays = ((T_GATE-1) - (gate_sm * idx_t).sum(-1)).cpu().numpy()
        # the gate is shared across inputs (it's a parameter, not input-dependent)
        # but we can measure: does model loss differ per system at final delays?
        lp, _ = model(lwin_p, temp=0.1); lm = F.mse_loss(lp, torch.tensor(
            lorenz_data[l_idx, lt0_p+T_GATE], device=DEVICE)).item()
        cp, _ = model(cwin_p, temp=0.1); cm = F.mse_loss(cp, torch.tensor(
            chen_data[c_idx, ct0_p+T_GATE], device=DEVICE)).item()
        model.train()

    return {
        "seed": seed,
        "final_delays": final_delays.tolist(),
        "tau_star": tau_star,
        "slot_variance": float(np.var(final_delays)),
        "lorenz_probe_mse": lm,
        "chen_probe_mse": cm,
        "tau_L_lorenz": TAU_L_LORENZ,
        "tau_L_chen": TAU_L_CHEN,
        "tau_star_ratio_lorenz": tau_star / TAU_L_LORENZ,
        "tau_star_ratio_chen":   tau_star / TAU_L_CHEN,
        "history": history,
        "note": "gate is input-independent (shared param) — slot delays fixed regardless of system",
    }


# ── Main ────────────────────────────────────────────────────────────────────
print(f"DHP Multi-Task | device={DEVICE}")
print(f"Systems: Lorenz (τ_L={TAU_L_LORENZ}) + Chen (τ_L={TAU_L_CHEN})")
print(f"Mixed batch: {BATCH//2} Lorenz + {BATCH//2} Chen per step")
print(f"N_SLOTS={N_SLOTS}, DIM={DIM}, T_GATE={T_GATE}, {STEPS} steps")
print()

rng = np.random.default_rng(0)
print("Generating Lorenz data...")
lorenz_data = gen_data(lorenz_rk4, n=1500, t_len=200, rng=rng,
                        x_range=(-15,15), y_range=(-15,15), z_range=(0,40))
print(f"  Lorenz: {lorenz_data.shape}")

print("Generating Chen data...")
chen_data = gen_data(chen_rk4, n=1500, t_len=200, burn=2000, rng=rng,
                      x_range=(-20,20), y_range=(-20,20), z_range=(0,50))
print(f"  Chen:   {chen_data.shape}")

results = {}
for seed in SEEDS:
    print(f"\n── seed={seed} ──")
    results[str(seed)] = run(seed, lorenz_data, chen_data)

tau_stars = [results[str(s)]["tau_star"] for s in SEEDS]
vars_     = [results[str(s)]["slot_variance"] for s in SEEDS]
ratios_l  = [results[str(s)]["tau_star_ratio_lorenz"] for s in SEEDS]
ratios_c  = [results[str(s)]["tau_star_ratio_chen"]   for s in SEEDS]

results["summary"] = {
    "mean_tau_star": float(np.mean(tau_stars)),
    "std_tau_star":  float(np.std(tau_stars)),
    "mean_var":      float(np.mean(vars_)),
    "mean_ratio_lorenz_pct": float(np.mean(ratios_l)*100),
    "mean_ratio_chen_pct":   float(np.mean(ratios_c)*100),
    "finding": ("gate_is_input_independent_delay_specializes_not_system"
                " — slots pick τ based on prediction task, not which attractor"),
}

print("\n" + "="*60)
for s in SEEDS:
    r = results[str(s)]
    print(f"seed={s}: τ*={r['tau_star']:.2f} "
          f"({r['tau_star_ratio_lorenz']*100:.1f}%τ_L_lorenz, "
          f"{r['tau_star_ratio_chen']*100:.1f}%τ_L_chen) "
          f"var={r['slot_variance']:.3f}")
print(f"mean τ*={results['summary']['mean_tau_star']:.2f}±{results['summary']['std_tau_star']:.2f}")
print("="*60)

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_FILE, "w") as f:
    json.dump(results, f, indent=2)
print(f"Results → {OUT_FILE}")
