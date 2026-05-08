#!/usr/bin/env python3
"""
ctm_ai_dhp_probe.py — DHP Probe for CTM-AI Architecture
=========================================================
Tests whether the CTM-AI architecture (Conscious Turing Machine, Lau et al. 2025,
arXiv:2605.04097) produces DHP-consistent temporal gate convergence.

CTM-AI architecture summary:
  - K parallel LTM processors, each producing a "chunk" from its own history
  - Up-tree competition: chunks compete for STM broadcast slot (weighted softmax)
  - Down-tree broadcast: winner's representation goes back to all processors
  - Link formation: Hebbian-style strengthening of successful processor connections
  - Key finding (ablation): iterative loop is the most critical component (−6.7 F1)

Our DHP analog:
  - Each LTM processor maintains a TEMPORAL GATE over its own history window
  - The gate selects which past timesteps the processor attends to when building its chunk
  - Up-tree competition = Gumbel-softmax selection (our existing mechanism)
  - Question: does competition + broadcast drive τ* convergence to τ_L?

Hypothesis:
  If up-tree competition creates selection pressure for temporally coherent chunks,
  processors that attend to the Lyapunov-coherent horizon should consistently win.
  This would drive τ* → τ_L via competitive selection, not gradient descent alone.
  This would be a new mechanism for DHP emergence.

Null hypothesis:
  CTM-AI competition is about content routing, not temporal horizon selection.
  τ* would remain near T_GATE/2 (uniform) or collapse to short-range.

Archon | DuoNeural | 2026-05-08
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path

# ─── Config ──────────────────────────────────────────────────────────────────
N_SEEDS       = 4
N_STEPS       = 6000
LOG_EVERY     = 100
TEMP_START    = 2.0
TEMP_END      = 0.1

# CTM-AI architecture params
K_PROCESSORS  = 6      # number of LTM processors (CTM-AI uses variable K)
T_GATE        = 32     # history window per processor
HIDDEN        = 64     # processor hidden dim
CHUNK_DIM     = 32     # chunk dimension (processor output)
STM_DIM       = 32     # shared STM broadcast dimension
PRED_HORIZONS = [1, 2, 4, 8, 16]
N_TRAJ        = 128

# Target: Lorenz-3D (τ_L ≈ 22 for dt=0.01)
LORENZ_DT     = 0.01
TAU_L         = 22.0
IN_DIM        = 3

OUT_DIR = Path("ctm_ai_dhp_output")
OUT_DIR.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Lorenz ───────────────────────────────────────────────────────────────────

def lorenz_step(xyz, dt=0.01, sigma=10.0, rho=28.0, beta=8/3):
    x, y, z = xyz[0], xyz[1], xyz[2]
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return xyz + dt * torch.tensor([dx, dy, dz], dtype=torch.float32, device=xyz.device)

def make_lorenz(n_steps, seed=0, dt=LORENZ_DT):
    torch.manual_seed(seed * 13)
    xyz = torch.randn(3, device=DEVICE) * 0.5 + torch.tensor([1., 1., 20.], device=DEVICE)
    for _ in range(2000):
        xyz = lorenz_step(xyz, dt)
    traj = []
    for _ in range(n_steps):
        xyz = lorenz_step(xyz, dt)
        traj.append(xyz.clone())
    traj = torch.stack(traj)
    mean = traj.mean(0); std = traj.std(0).clamp(min=1e-6)
    return (traj - mean) / std


# ─── CTM-AI LTM Processor ────────────────────────────────────────────────────

class LTMProcessor(nn.Module):
    """
    Single LTM processor with temporal gate.
    - Maintains a GRU over its own input history
    - Gumbel-softmax gate selects which history positions to attend to
    - Produces a "chunk" = context-weighted summary for competition
    - Receives broadcast from STM (down-tree)
    """
    def __init__(self, in_dim, hidden, t_gate, chunk_dim, stm_dim):
        super().__init__()
        self.t_gate    = t_gate
        self.chunk_dim = chunk_dim

        # GRU processes current input + STM broadcast
        self.gru       = nn.GRUCell(in_dim + stm_dim, hidden)
        # temporal gate over history
        self.gate_proj = nn.Linear(hidden, t_gate)
        # history readout → chunk
        self.chunk_proj = nn.Linear(in_dim, chunk_dim)
        # output chunk after gating
        self.out_proj   = nn.Linear(hidden, chunk_dim)

        positions = torch.arange(t_gate, dtype=torch.float32)
        self.register_buffer("positions", positions)

    def forward(self, x, stm, h, hist, hist_idx, temperature=1.0):
        """
        x:        (1, in_dim) — current input
        stm:      (1, stm_dim) — STM broadcast from previous step
        h:        (1, hidden) — processor hidden state
        hist:     (t_gate, in_dim) — circular input history
        hist_idx: int

        Returns: chunk (1, chunk_dim), gate (1, t_gate), new_h, gate_logits
        """
        # update hidden state
        gru_in = torch.cat([x, stm], dim=-1)
        h_new  = self.gru(gru_in, h)

        # temporal gate
        gate_logits = self.gate_proj(h_new)
        gate        = F.gumbel_softmax(gate_logits, tau=temperature, hard=False)

        # read history at gate-weighted positions
        idx      = hist_idx % self.t_gate
        ordered  = torch.roll(hist, -idx, dims=0)  # (t_gate, in_dim) newest-first
        context  = (gate.unsqueeze(-1) * ordered.unsqueeze(0)).sum(1)  # (1, in_dim)

        # produce chunk from hidden + gated context
        chunk = self.out_proj(h_new) + self.chunk_proj(context)
        return chunk, gate, h_new, gate_logits

    def tau_star(self, gate):
        return (gate.detach().mean(0) * self.positions).sum().item()


# ─── CTM-AI Full Architecture ─────────────────────────────────────────────────

class CTMAIModel(nn.Module):
    """
    K parallel LTM processors + up-tree competition + down-tree broadcast.
    Competition: softmax over chunk "competition scores" → winner gets STM slot.
    This is our DHP probe: does competition pressure drive τ* → τ_L?
    """
    def __init__(self, in_dim, hidden, k_processors, t_gate, chunk_dim, stm_dim, pred_horizons):
        super().__init__()
        self.k          = k_processors
        self.t_gate     = t_gate
        self.stm_dim    = stm_dim
        self.horizons   = pred_horizons

        self.processors = nn.ModuleList([
            LTMProcessor(in_dim, hidden, t_gate, chunk_dim, stm_dim)
            for _ in range(k_processors)
        ])

        # competition: score each chunk for STM access
        self.competition_score = nn.Linear(chunk_dim, 1)

        # STM broadcast: winning chunk → STM
        self.stm_proj = nn.Linear(chunk_dim, stm_dim)

        # shared predictor: STM → prediction for each horizon
        self.predictors = nn.ModuleList([
            nn.Linear(stm_dim, in_dim) for _ in pred_horizons
        ])

    def reset(self):
        dev = next(self.parameters()).device
        self.h_states  = [torch.zeros(1, self.processors[0].gru.hidden_size, device=dev)
                          for _ in range(self.k)]
        self.hists     = [torch.zeros(self.t_gate, IN_DIM, device=dev)
                          for _ in range(self.k)]
        self.hist_idxs = [0] * self.k
        self.stm       = torch.zeros(1, self.stm_dim, device=dev)

    def forward(self, x, temperature=1.0):
        """
        x: (1, in_dim)
        Returns: preds, all_gates, competition_weights, winner_idx
        """
        chunks = []
        gates  = []

        for i, proc in enumerate(self.processors):
            # update processor's history
            self.hists[i][self.hist_idxs[i] % self.t_gate] = x.squeeze(0)
            self.hist_idxs[i] += 1

            chunk, gate, h_new, _ = proc(
                x, self.stm, self.h_states[i],
                self.hists[i], self.hist_idxs[i], temperature
            )
            self.h_states[i] = h_new.detach()
            chunks.append(chunk)       # (1, chunk_dim)
            gates.append(gate)

        # up-tree competition: score all chunks
        chunk_stack = torch.cat(chunks, dim=0)    # (K, chunk_dim)
        scores      = self.competition_score(chunk_stack).squeeze(-1)   # (K,)
        comp_weights = F.softmax(scores, dim=0)   # (K,) — competition distribution

        # winner via gumbel-softmax (differentiable selection)
        winner_weights = F.gumbel_softmax(scores.unsqueeze(0), tau=temperature, hard=False)  # (1, K)
        winner_idx     = int(winner_weights.argmax().item())

        # down-tree broadcast: weighted sum of chunks → STM
        stm_input  = (winner_weights.unsqueeze(-1) * chunk_stack.unsqueeze(0)).sum(1)  # (1, chunk_dim)
        self.stm   = self.stm_proj(stm_input).detach()

        # predictions from STM
        preds = [pred(self.stm) for pred in self.predictors]

        return preds, gates, comp_weights, winner_idx

    def tau_star_per_processor(self, gates):
        return [proc.tau_star(g) for proc, g in zip(self.processors, gates)]

    def mean_tau_star(self, gates):
        vals = self.tau_star_per_processor(gates)
        return float(np.mean(vals)), float(np.std(vals))

    def winner_tau_star(self, gates, winner_idx):
        return self.processors[winner_idx].tau_star(gates[winner_idx])


# ─── Training ─────────────────────────────────────────────────────────────────

def run_seed(seed: int) -> dict:
    print(f"\n{'='*60}")
    print(f"  Seed {seed} — CTM-AI DHP probe")
    print(f"{'='*60}")

    total_steps = N_STEPS + max(PRED_HORIZONS) + N_TRAJ
    traj = make_lorenz(total_steps, seed=seed)

    model = CTMAIModel(
        in_dim=IN_DIM, hidden=HIDDEN, k_processors=K_PROCESSORS,
        t_gate=T_GATE, chunk_dim=CHUNK_DIM, stm_dim=STM_DIM,
        pred_horizons=PRED_HORIZONS,
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    model.reset()

    history = []
    winner_tau_history = []  # track τ* of the WINNING processor specifically

    for step in range(N_STEPS):
        temp = TEMP_START * (TEMP_END / TEMP_START) ** (step / N_STEPS)
        t    = step + N_TRAJ

        x = traj[t].unsqueeze(0)
        preds, gates, comp_weights, winner_idx = model(x, temperature=temp)

        # multi-horizon prediction loss
        loss = sum(
            F.mse_loss(preds[k_idx], traj[t + k].unsqueeze(0))
            for k_idx, k in enumerate(PRED_HORIZONS)
            if t + k < len(traj)
        )

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % LOG_EVERY == 0 or step == N_STEPS - 1:
            mean_tau, std_tau = model.mean_tau_star(gates)
            winner_tau        = model.winner_tau_star(gates, winner_idx)
            per_proc          = model.tau_star_per_processor(gates)
            pct_mean  = mean_tau / TAU_L * 100
            pct_winner = winner_tau / TAU_L * 100

            print(f"  step {step:5d} | temp={temp:.3f} | loss={loss.item():.4f}")
            print(f"    mean τ*={mean_tau:.1f} ({pct_mean:.1f}% τ_L)  "
                  f"winner[{winner_idx}] τ*={winner_tau:.1f} ({pct_winner:.1f}%)")
            print(f"    per-proc τ*: {[f'{v:.1f}' for v in per_proc]}")
            print(f"    competition weights: {[f'{w:.2f}' for w in comp_weights.detach().cpu().tolist()]}")

            history.append({
                "step": step, "temp": float(temp), "loss": float(loss.item()),
                "mean_tau": float(mean_tau), "mean_tau_pct": float(pct_mean),
                "winner_idx": winner_idx, "winner_tau": float(winner_tau),
                "winner_tau_pct": float(pct_winner),
                "per_proc_tau": [float(v) for v in per_proc],
                "comp_weights": [float(w) for w in comp_weights.detach().cpu().tolist()],
            })
            winner_tau_history.append(winner_tau)

    # final τ* averaged over last 500 steps
    final_taus = []
    final_winner_taus = []
    with torch.no_grad():
        for step in range(N_STEPS - 500, N_STEPS):
            temp = TEMP_END
            t    = step + N_TRAJ
            x    = traj[t].unsqueeze(0)
            _, gates, _, winner_idx = model(x, temperature=temp)
            mt, _ = model.mean_tau_star(gates)
            wt    = model.winner_tau_star(gates, winner_idx)
            final_taus.append(mt)
            final_winner_taus.append(wt)

    mean_final  = float(np.mean(final_taus))
    winner_final = float(np.mean(final_winner_taus))
    pct_mean_f  = mean_final / TAU_L * 100
    pct_win_f   = winner_final / TAU_L * 100

    dhp_mean   = mean_final >= 0.70 * TAU_L
    dhp_winner = winner_final >= 0.70 * TAU_L
    interior   = mean_final < T_GATE

    print(f"\n  FINAL (seed {seed}):")
    print(f"    mean τ*={mean_final:.1f} ({pct_mean_f:.1f}% τ_L) DHP={'✓' if dhp_mean else '✗'}")
    print(f"    winner τ*={winner_final:.1f} ({pct_win_f:.1f}% τ_L) DHP={'✓' if dhp_winner else '✗'}")
    print(f"    interior (< T_GATE={T_GATE}): {'✓' if interior else '✗'}")

    # KEY DIAGNOSTIC: is winner τ* > mean τ*?
    # If competition preferentially selects high-τ* processors, winner_final > mean_final
    # This would show competition AMPLIFIES DHP, not just reflects it
    winner_advantage = winner_final - mean_final
    print(f"    winner τ* advantage over mean: {winner_advantage:+.1f} steps")

    return {
        "seed": seed,
        "mean_tau_final": mean_final, "mean_tau_pct": pct_mean_f,
        "winner_tau_final": winner_final, "winner_tau_pct": pct_win_f,
        "dhp_mean": bool(dhp_mean), "dhp_winner": bool(dhp_winner),
        "interior": bool(interior),
        "winner_advantage": float(winner_advantage),
        "history": history,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("CTM-AI DHP Probe — Up-tree competition + temporal gates")
    print(f"Device: {DEVICE}")
    print(f"K_PROCESSORS={K_PROCESSORS}, T_GATE={T_GATE}, τ_L≈{TAU_L}")
    print(f"N_SEEDS={N_SEEDS}, N_STEPS={N_STEPS}")
    print("="*60)

    results = []
    for seed in range(N_SEEDS):
        r = run_seed(seed)
        results.append(r)

    # summary
    print("\n" + "="*60)
    print("  SUMMARY — CTM-AI DHP PROBE")
    print("="*60)
    dhp_mean_count   = sum(1 for r in results if r["dhp_mean"])
    dhp_winner_count = sum(1 for r in results if r["dhp_winner"])
    avg_advantage    = np.mean([r["winner_advantage"] for r in results])
    print(f"  DHP pass (mean τ*):   {dhp_mean_count}/{N_SEEDS}")
    print(f"  DHP pass (winner τ*): {dhp_winner_count}/{N_SEEDS}")
    print(f"  Avg winner advantage: {avg_advantage:+.1f} steps "
          f"({'winner biased high-τ*' if avg_advantage > 1 else 'no winner bias'})")
    print()
    for r in results:
        print(f"  seed {r['seed']}: mean τ*={r['mean_tau_final']:.1f} ({r['mean_tau_pct']:.1f}%) | "
              f"winner τ*={r['winner_tau_final']:.1f} ({r['winner_tau_pct']:.1f}%) | "
              f"advantage={r['winner_advantage']:+.1f}")

    if dhp_mean_count >= 3:
        verdict = "✅ CTM-AI ARCHITECTURE PRODUCES DHP"
        if avg_advantage > 1.0:
            verdict += " — competition amplifies high-τ* selection"
    else:
        verdict = f"❌ DHP NOT CONFIRMED: {dhp_mean_count}/{N_SEEDS} seeds"
    print(f"\n  {verdict}")

    out = {
        "experiment": "ctm_ai_dhp_probe",
        "tau_L": TAU_L, "t_gate": T_GATE, "k_processors": K_PROCESSORS,
        "n_seeds": N_SEEDS, "n_steps": N_STEPS,
        "dhp_mean_count": dhp_mean_count, "dhp_winner_count": dhp_winner_count,
        "avg_winner_advantage": float(avg_advantage),
        "verdict": verdict,
        "seeds": results,
    }
    outfile = OUT_DIR / "ctm_ai_dhp_results.json"
    with open(outfile, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results → {outfile}")
