#!/usr/bin/env python3
"""
CTM v40 — True v28 Replica (Fixing Three Critical Architecture Bugs)
=====================================================================
v38 used wrong architecture (3 bugs vs v28):
  1. Shared projection for all slots → identical gradients → no scout
     FIX: per-slot slot_proj[i] (independent Linear per slot) — breaks symmetry
  2. Averaged slots before decoding → no per-slot gradient routing
     FIX: concatenate all slots then decode → per-slot output
  3. LR=1e-3 Adam, batch=64 → FIX: LR=2e-4 AdamW+CosineAnnealing, batch=128

HYPOTHESIS: These three fixes will restore the scout phenomenon.
If τ* ≈ τ_L re-emerges, it confirms the mechanism: per-slot projection
symmetry-breaking → winner-take-all scout via concatenated decoder gradients.

Kilonova (ROCm gfx1103) — TORCHDYNAMO_DISABLE=1 required.

Archon | DuoNeural | 2026-05-02
"""

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# ── Config — EXACT v28 values ────────────────────────────────────────────────
T_GATE        = 32
N_OBJ         = 8
SLOT_DIM      = 256
N_HEADS       = 8
OBJ_DIM       = 3
PRED_HORIZONS = [1, 2, 4, 8, 16]
TRAIN_STEPS   = 60_000
LR            = 2e-4           # v28 value (we used 1e-3)
BATCH         = 128            # v28 value (we used 64)
TBPTT_CUTOFF  = 8
TEMP_START    = 2.0
TEMP_END      = 0.1
LAMBDA_GATE   = 0.001
LOG_EVERY     = 5000
DT            = 0.05
RHO           = 28.0
SEEDS         = [0, 1, 2, 42, 99]
OUT_DIR       = Path("/home/ai/v40_results")
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# ── Lorenz ───────────────────────────────────────────────────────────────────
def lorenz_rk4(x, y, z, dt, sigma=10., rho=28., beta=8./3.):
    def d(x,y,z): return sigma*(y-x), x*(rho-z)-y, x*y-beta*z
    k1x,k1y,k1z = d(x,y,z)
    k2x,k2y,k2z = d(x+.5*dt*k1x,y+.5*dt*k1y,z+.5*dt*k1z)
    k3x,k3y,k3z = d(x+.5*dt*k2x,y+.5*dt*k2y,z+.5*dt*k2z)
    k4x,k4y,k4z = d(x+dt*k3x,y+dt*k3y,z+dt*k3z)
    return (x+dt*(k1x+2*k2x+2*k3x+k4x)/6,
            y+dt*(k1y+2*k2y+2*k3y+k4y)/6,
            z+dt*(k1z+2*k2z+2*k3z+k4z)/6)

def generate_lorenz(n, t_len, rng, warmup=1000):
    trajs = []
    for _ in range(n):
        x,y,z = rng.uniform(-15,15), rng.uniform(-15,15), rng.uniform(0,40)
        for _ in range(warmup):
            x,y,z = lorenz_rk4(x,y,z,DT,rho=RHO)
        traj = []
        for _ in range(t_len):
            x,y,z = lorenz_rk4(x,y,z,DT,rho=RHO)
            traj.append([x,y,z])
        trajs.append(traj)
    trajs = np.array(trajs, dtype=np.float32)
    mu  = trajs.mean(axis=(0,1,2), keepdims=True)
    std = trajs.std(axis=(0,1,2), keepdims=True) + 1e-8
    return (trajs - mu) / std

def compute_lyapunov(n=20000, spinup=2000, eps=1e-8):
    rng = np.random.default_rng(0)
    x1,y1,z1 = rng.uniform(-10,10,3)
    x2,y2,z2 = x1+eps, y1, z1
    for _ in range(spinup):
        x1,y1,z1 = lorenz_rk4(x1,y1,z1,DT,rho=RHO)
        x2,y2,z2 = lorenz_rk4(x2,y2,z2,DT,rho=RHO)
    lsum = 0.; cnt = 0
    for _ in range(n):
        x1,y1,z1 = lorenz_rk4(x1,y1,z1,DT,rho=RHO)
        x2,y2,z2 = lorenz_rk4(x2,y2,z2,DT,rho=RHO)
        d = np.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
        if d > 0:
            lsum += np.log(d/eps); cnt += 1
            sc = eps/d
            x2=x1+(x2-x1)*sc; y2=y1+(y2-y1)*sc; z2=z1+(z2-z1)*sc
    return lsum/(cnt*DT)

# ── Architecture: TRUE v28 replica ──────────────────────────────────────────

class LearnedTemporalGateEncoder(nn.Module):
    """
    FIX 1: per-slot slot_proj[i] — independent Linear for each slot.
    This is the symmetry-breaker. Random init → different gradients per slot
    → winner-take-all scout dynamics can emerge.
    """
    def __init__(self):
        super().__init__()
        self.gate_logits = nn.Parameter(torch.zeros(N_OBJ, T_GATE))
        # v28: one shared input proj + per-slot output projections
        self.input_proj  = nn.Linear(OBJ_DIM, SLOT_DIM)
        # THE KEY FIX: independent projection per slot (breaks symmetry)
        self.slot_proj   = nn.ModuleList([
            nn.Linear(SLOT_DIM, SLOT_DIM) for _ in range(N_OBJ)
        ])

    def forward(self, traj_window, temp):
        # traj_window: (B, T_GATE, OBJ_DIM)
        B = traj_window.size(0)
        gate     = F.softmax(self.gate_logits / temp, dim=-1)     # (N_OBJ, T_GATE)

        # Project input across time: (B, T_GATE, SLOT_DIM)
        h = self.input_proj(traj_window)

        # Gate-weight over time window per slot
        gate_exp = gate.unsqueeze(0).unsqueeze(-1)                # (1, N_OBJ, T_GATE, 1)
        h_exp    = h.unsqueeze(1).expand(B, N_OBJ, T_GATE, SLOT_DIM)
        gated    = (h_exp * gate_exp).sum(dim=2)                  # (B, N_OBJ, SLOT_DIM)

        # Per-slot projections — the symmetry breaker
        slots = torch.stack([self.slot_proj[i](gated[:, i, :])
                             for i in range(N_OBJ)], dim=1)      # (B, N_OBJ, SLOT_DIM)
        return slots, gate

    def gate_entropy_loss(self, gate):
        return -(gate * (gate + 1e-10).log()).sum(dim=-1).mean()

class SlotGNNDynamics(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(SLOT_DIM, SLOT_DIM)
        self.k_proj = nn.Linear(SLOT_DIM, SLOT_DIM)
        self.v_proj = nn.Linear(SLOT_DIM, SLOT_DIM)
        self.o_proj = nn.Linear(SLOT_DIM, SLOT_DIM)
        self.norm1  = nn.LayerNorm(SLOT_DIM)
        self.mlp    = nn.Sequential(nn.Linear(SLOT_DIM, SLOT_DIM*2), nn.SiLU(),
                                    nn.Linear(SLOT_DIM*2, SLOT_DIM))
        self.norm2  = nn.LayerNorm(SLOT_DIM)
        self.n_heads = N_HEADS
        self.dh      = SLOT_DIM // N_HEADS

    def tick(self, slots):
        B, N, D = slots.shape
        H, Dh = self.n_heads, self.dh
        q = self.q_proj(slots).view(B,N,H,Dh).transpose(1,2)
        k = self.k_proj(slots).view(B,N,H,Dh).transpose(1,2)
        v = self.v_proj(slots).view(B,N,H,Dh).transpose(1,2)
        att = F.scaled_dot_product_attention(q, k, v)
        msg = self.o_proj(att.transpose(1,2).reshape(B,N,D))
        slots = self.norm1(slots + msg)
        return self.norm2(slots + self.mlp(slots))

    def forward(self, slots, n_ticks=1):
        for _ in range(n_ticks):
            slots = self.tick(slots)
        return slots

class SlotDecoder(nn.Module):
    """
    FIX 2: concatenate all slots then decode to per-slot outputs.
    v28: dec(slots.view(B, -1)).view(B, N_OBJ, OBJ_DIM)
    Each slot gets unique gradient signal based on the full slot context.
    """
    def __init__(self):
        super().__init__()
        self.dec = nn.Sequential(
            nn.Linear(SLOT_DIM * N_OBJ, SLOT_DIM), nn.SiLU(),
            nn.Linear(SLOT_DIM, OBJ_DIM * N_OBJ),
        )
        self.n_obj = N_OBJ
        self.obj_dim = OBJ_DIM

    def forward(self, slots):
        B = slots.shape[0]
        return self.dec(slots.view(B, -1)).view(B, self.n_obj, self.obj_dim)

class CTMWorldModelV40(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder  = LearnedTemporalGateEncoder()
        self.dynamics = SlotGNNDynamics()
        self.decoder  = SlotDecoder()

    def predict_all_horizons(self, history, temp):
        """
        v28-style: run dynamics delta ticks between each horizon.
        TBPTT: detach slots when k > TBPTT_CUTOFF.
        """
        slots, gate = self.encoder(history, temp)
        preds = {}
        prev_k = 0
        for k in PRED_HORIZONS:
            delta  = k - prev_k
            use_detach = (k > TBPTT_CUTOFF)
            slots  = self.dynamics(slots.detach() if use_detach else slots,
                                   n_ticks=delta)
            preds[k] = self.decoder(slots)       # (B, N_OBJ, OBJ_DIM)
            prev_k = k
        return preds, gate

    def compute_eff_delays(self, gate):
        t_idx = torch.arange(T_GATE, dtype=torch.float32, device=gate.device)
        wmean = (gate * t_idx).sum(dim=-1)
        return (T_GATE - 1) - wmean              # steps-back-from-present per slot

# ── Training ─────────────────────────────────────────────────────────────────
def run_seed(seed, tau_L):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    print(f"\n{'='*60}")
    print(f"[v40 seed={seed}] TRUE v28 replica | τ_L={tau_L:.1f} | T_GATE={T_GATE}")
    print(f"[v40 seed={seed}] FIX1=per_slot_proj | FIX2=concat_decoder | FIX3=AdamW+cosine")

    t_needed = T_GATE + 120 + max(PRED_HORIZONS)
    data = generate_lorenz(6000, t_needed, rng)
    t0_max = t_needed - T_GATE - max(PRED_HORIZONS) - 1
    print(f"[v40 seed={seed}] data: {data.shape}")

    model = CTMWorldModelV40().to(DEVICE)
    # FIX 3: AdamW + CosineAnnealingLR (v28 exact)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TRAIN_STEPS)

    final_tau = 0.0
    final_per_slot = None

    for step in range(TRAIN_STEPS):
        model.train()
        temp = TEMP_START + (TEMP_END - TEMP_START) * (step / TRAIN_STEPS)

        idx = np.random.choice(len(data), BATCH, replace=True)
        t0  = np.random.randint(0, t0_max + 1)
        win  = torch.tensor(data[idx, t0:t0+T_GATE], dtype=torch.float32, device=DEVICE)
        # targets: one step per horizon after window end
        tgt_dict = {}
        for h in PRED_HORIZONS:
            ti = t0 + T_GATE - 1 + h
            tgt_dict[h] = torch.tensor(data[idx, ti], dtype=torch.float32, device=DEVICE)

        preds, gate = model.predict_all_horizons(win, temp)

        # Prediction loss — mean over slots (decoder outputs per-slot predictions)
        pred_loss = sum(
            F.mse_loss(preds[k].mean(dim=1), tgt_dict[k])   # mean over N_OBJ slot preds
            for k in PRED_HORIZONS
        )
        gate_loss  = model.encoder.gate_entropy_loss(gate)
        loss       = pred_loss + LAMBDA_GATE * gate_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()

        with torch.no_grad():
            g = model.encoder.gate_logits.detach().softmax(dim=-1)
            delays = model.compute_eff_delays(g)
            tau_s  = delays.max().item()
            if step >= TRAIN_STEPS // 10:
                final_tau = tau_s
                final_per_slot = [float(d.item()) for d in delays]

        if step % LOG_EVERY == 0:
            delay_str = " ".join(f"{d.item():.1f}" for d in delays)
            slot_var  = float(delays.var().item())
            print(f"[v40 seed={seed}] step {step:6d} | τ*={tau_s:.2f} | τ_L={tau_L:.1f} | "
                  f"var={slot_var:.3f} | temp={temp:.3f} | [{delay_str}]")

    ratio       = float(final_tau / tau_L) if tau_L > 0 else 0.0
    dhp_success = bool(final_tau >= 0.7 * tau_L)
    slot_var    = float(np.var(final_per_slot)) if final_per_slot else 0.0

    print(f"[v40 seed={seed}] DONE | τ*={final_tau:.2f} | τ_L={tau_L:.1f} | "
          f"ratio={ratio:.1%} | DHP={'YES ✓' if dhp_success else 'no'} | var={slot_var:.3f}")
    print(f"[v40 seed={seed}] per-slot: {[f'{d:.2f}' for d in (final_per_slot or [])]}")

    return {
        "seed":            int(seed),
        "tau_star":        float(final_tau),
        "tau_L":           float(tau_L),
        "ratio":           ratio,
        "dhp_success":     dhp_success,
        "per_slot_delays": final_per_slot or [],
        "slot_variance":   slot_var,
    }

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / "results_v40_true_replica.json"

    print(f"[v40] Computing λ_max for ρ={RHO}...")
    lam   = compute_lyapunov()
    tau_L = 1.0 / (lam * DT) if lam > 0 else float('inf')
    print(f"[v40] λ_max={lam:.4f}/tu → τ_L={tau_L:.1f} steps | DEVICE={DEVICE}")
    print(f"[v40] Seeds: {SEEDS}")
    print(f"[v40] CRITICAL FIXES vs v38:")
    print(f"[v40]   FIX1: per-slot slot_proj[i] (symmetry breaking)")
    print(f"[v40]   FIX2: concat-slot decoder (per-slot gradients)")
    print(f"[v40]   FIX3: LR=2e-4 AdamW+CosineAnnealing, BATCH=128")

    results = {
        "experiment":  "v40_true_replica",
        "date":        "2026-05-02",
        "fixes_vs_v38": ["per_slot_proj", "concat_decoder", "AdamW_cosine_lr2e-4_batch128"],
        "tau_L":       float(tau_L),
        "T_GATE":      T_GATE,
        "seeds":       {},
    }

    # Resume if partial
    if out_file.exists():
        with open(out_file) as f:
            results = json.load(f)
        print(f"[v40] Resuming from existing: {list(results['seeds'].keys())}")

    dhp_count = sum(1 for v in results["seeds"].values() if v.get("dhp_success", False))
    for seed in SEEDS:
        if str(seed) in results["seeds"]:
            print(f"[v40] Skipping seed={seed} (done)")
            continue
        r = run_seed(seed, tau_L)
        results["seeds"][str(seed)] = r
        if r["dhp_success"]: dhp_count += 1
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)

    all_tau = [results["seeds"][str(s)]["tau_star"] for s in SEEDS if str(s) in results["seeds"]]
    results["summary"] = {
        "dhp_count":    dhp_count,
        "dhp_rate":     float(dhp_count / len(SEEDS)),
        "mean_tau":     float(np.mean(all_tau)),
        "std_tau":      float(np.std(all_tau)),
        "verdict":      ("REPRODUCED" if dhp_count >= 3 else
                         "RARE" if dhp_count >= 1 else "NOT_REPRODUCED"),
    }
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"[v40] DHP: {dhp_count}/{len(SEEDS)} seeds | mean τ*={np.mean(all_tau):.2f}±{np.std(all_tau):.2f}")
    print(f"[v40] VERDICT: {results['summary']['verdict']}")
    print(f"[v40] → {out_file}")

if __name__ == "__main__":
    main()
