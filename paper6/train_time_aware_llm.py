#!/usr/bin/env python3
"""
train_time_aware_llm.py — DuoNeural Time-Aware LLM
Archon, 2026-05-10

Architecture synthesis of ALL DHP/CTM research (Papers 1-6):
  - CTM: vertical recurrence (shared weights), N_ticks=20 → 3B effective depth at 150M params
  - KGBN/SSA: Kuramoto oscillator attention, O(N) memory state (no KV cache growth)
  - DHP Gate: learned temporal gate → converges to τ_L via gradient descent alone
  - TSSP: Thought-Space Self-Prediction, warmup-hold-cosine-decay λ schedule
    prevents scale-dependent inversion at >300M params
  - Causal masking throughout (autoregressive LM)

Target: ~150M actual params, RTX 3090 24GB, WikiText-103 baseline
The DHP gate's τ* will be logged each eval step — we expect convergence at ~0.72 × τ_L.

Usage:
  python train_time_aware_llm.py
  python train_time_aware_llm.py --d_model 1024 --n_ticks 10  # smaller debug run
"""

import math, os, json, time, argparse, sys, traceback
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint

import numpy as np

# ─── CLI ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--d_model",    type=int, default=2048)
    p.add_argument("--n_heads",    type=int, default=16)
    p.add_argument("--n_ticks",    type=int, default=20)
    p.add_argument("--seq_len",    type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)   # effective batch = 32
    p.add_argument("--max_steps",  type=int, default=50000)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--tssp_lambda_max", type=float, default=0.05)
    p.add_argument("--kappa",      type=float, default=1.0)  # Kuramoto coupling
    p.add_argument("--dt",         type=float, default=0.1)  # Kuramoto timestep
    p.add_argument("--save_dir",   type=str, default="/workspace/time_aware_llm")
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--no_bnb",     action="store_true", help="disable 8-bit Adam")
    p.add_argument("--grad_ckpt",  action="store_true", default=True)
    return p.parse_args()

# ─── Logging ─────────────────────────────────────────────────────────────────
LOG_FILE = None
def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if LOG_FILE:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")

# ─── Kuramoto SSA (Selective Synchronization Attention) ──────────────────────
class KuramotoSSA(nn.Module):
    """
    Replaces dot-product self-attention with phase-locking oscillatory attention.

    Memory: O(N) state (phases only, no KV cache matrix storage)
    Computation: O(N²) for phase differences (same as standard attn, but avoids
                 the large D_k scaling factor)

    Each token-position has H oscillator phases (one per head).
    Phases evolve via Kuramoto dynamics across CTM ticks.
    Attention weight a_ij = cos(φ_i - φ_j), causally masked.

    Key insight from Paper 4/6: phase-locking escapes spatial aliasing on
    symmetric attractors where dot-product attention collapses to uniform weights.
    """
    def __init__(self, d_model, n_heads, kappa=1.0, dt=0.1, max_seq_len=512):
        super().__init__()
        self.d_model    = d_model
        self.n_heads    = n_heads
        self.head_dim   = d_model // n_heads
        self.kappa      = kappa
        self.dt         = dt

        # Learnable natural frequencies — one per head
        # These determine which tokens naturally synchronize vs. repel
        self.omega = nn.Parameter(torch.randn(n_heads) * 0.1)

        # Phase initialization: each head gets a sinusoidal prior by position
        # (acts like rotary position encoding but in phase space)
        pos = torch.arange(max_seq_len).float()
        freqs = torch.arange(1, n_heads + 1).float() / n_heads
        # shape: [max_seq_len, n_heads]
        phase_init = torch.outer(pos, freqs * 2 * math.pi / max_seq_len)
        self.register_buffer("phase_init", phase_init)

        # Value projection (standard linear — phases handle routing, V handles content)
        self.v_proj   = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # DHP Temporal Gate — learned over sequence position
        # Gradient descent should drive this to peak at τ_L (0.72 × window)
        self.dhp_gate_logits = nn.Parameter(torch.zeros(max_seq_len))

        # Causal mask buffer
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def init_phases(self, B, S, device):
        """Sinusoidal phase initialization — same role as RoPE but in phase space."""
        return self.phase_init[:S].unsqueeze(0).expand(B, -1, -1).clone()

    def kuramoto_step(self, phases):
        """
        One step of Kuramoto dynamics (causal — token i only couples to j ≤ i).
        phases: [B, S, H]
        returns: [B, S, H]
        """
        B, S, H = phases.shape
        # Pairwise phase differences [B, S, S, H]
        diff = phases.unsqueeze(2) - phases.unsqueeze(1)  # φ_i - φ_j
        sin_diff = torch.sin(diff)  # [B, S, S, H]

        # Causal coupling: only j <= i contributes
        # causal_mask is True where j > i (to MASK OUT), so invert for coupling
        causal = ~self.causal_mask[:S, :S]  # [S, S] — True where j <= i
        sin_diff = sin_diff * causal.unsqueeze(0).unsqueeze(-1).float()

        # Mean field coupling (normalize by number of causal neighbors)
        n_causal = causal.float().sum(dim=1).clamp(min=1)  # [S]
        coupling = sin_diff.sum(dim=2) / n_causal.view(1, S, 1)  # [B, S, H]

        # Update: φ ← φ + (ω + κ·coupling) × dt
        new_phases = phases + (self.omega.view(1, 1, H) + self.kappa * coupling) * self.dt
        return new_phases

    def forward(self, x, phases):
        """
        x:      [B, S, D]
        phases: [B, S, H]
        returns: (out [B, S, D], new_phases [B, S, H])
        """
        B, S, D = x.shape
        H = self.n_heads
        head_dim = self.head_dim

        # Evolve phases one Kuramoto step
        new_phases = self.kuramoto_step(phases)  # [B, S, H]

        # Attention logits from phase differences
        phi_i = new_phases.unsqueeze(2)  # [B, S, 1, H]
        phi_j = new_phases.unsqueeze(1)  # [B, 1, S, H]
        attn_logits = torch.cos(phi_i - phi_j)  # [B, S, S, H]
        attn_logits = attn_logits.permute(0, 3, 1, 2)  # [B, H, S, S]

        # Apply DHP gate: weight attention logits by learned temporal relevance
        # gate[j] = learned weight for position j (relative to current position i)
        # We want a causal gate: at position i, gate weights how much to attend to j
        # Simple form: broadcast the gate across all query positions
        gate = F.softmax(self.dhp_gate_logits[:S], dim=0)  # [S]
        # Scale attention logits by DHP gate (encourages sparse, horizon-aligned attention)
        attn_logits = attn_logits + gate.log().view(1, 1, 1, S)

        # Causal mask (future tokens → -inf)
        attn_logits = attn_logits.masked_fill(
            self.causal_mask[:S, :S].unsqueeze(0).unsqueeze(0), float("-inf")
        )
        attn_weights = F.softmax(attn_logits, dim=-1)  # [B, H, S, S]

        # Values
        v = self.v_proj(x)                              # [B, S, D]
        v = v.view(B, S, H, head_dim).permute(0, 2, 1, 3)  # [B, H, S, head_dim]
        out = attn_weights @ v                          # [B, H, S, head_dim]
        out = out.permute(0, 2, 1, 3).contiguous().view(B, S, D)
        out = self.out_proj(out)

        return out, new_phases

    def get_dhp_metrics(self, S):
        """
        Compute DHP gate statistics for logging.
        Returns τ* (effective horizon) and gate entropy.
        """
        gate = F.softmax(self.dhp_gate_logits[:S], dim=0).detach().cpu()
        # τ* = center of mass of gate distribution
        positions = torch.arange(S).float()
        tau_star = (gate * positions).sum().item()
        # Entropy
        entropy = -(gate * (gate + 1e-9).log()).sum().item()
        # Peakedness (max gate weight)
        peakedness = gate.max().item()
        return tau_star, entropy, peakedness


# ─── CTM Block (shared-weight residual block) ────────────────────────────────
class CTMBlock(nn.Module):
    """
    The core CTM computation unit. Applied N_ticks times with SHARED weights.

    One block ≈ 50M params (d_model=2048):
    - KuramotoSSA: ~34M (V proj + out proj + phases)
    - FFN: ~33M (d_model → 4×d_model → d_model)
    With N_ticks=20: simulates 1B effective depth.
    """
    def __init__(self, d_model, n_heads, ffn_mult=4, dropout=0.1,
                 kappa=1.0, dt=0.1, max_seq_len=512):
        super().__init__()
        self.attn = KuramotoSSA(d_model, n_heads, kappa=kappa, dt=dt,
                                max_seq_len=max_seq_len)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model, bias=False),
            nn.GELU(),
            nn.Linear(ffn_mult * d_model, d_model, bias=False),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, h, phases):
        """
        h:      [B, S, D] — current hidden state
        phases: [B, S, H] — current oscillator phases
        returns: (h [B, S, D], phases [B, S, H])
        """
        # Pre-norm + Kuramoto attention + residual
        attn_out, new_phases = self.attn(self.norm1(h), phases)
        h = h + self.drop(attn_out)
        # Pre-norm + FFN + residual
        h = h + self.drop(self.ffn(self.norm2(h)))
        return h, new_phases


# ─── TSSP Head ───────────────────────────────────────────────────────────────
class TSSPHead(nn.Module):
    """
    Thought-Space Self-Prediction head.
    Predicts h_{tick+1} from h_{tick} via stop-gradient.
    Shared across all tick transitions.

    Architecture: LN → Linear(D→D//4) → GELU → Linear(D//4→D)
    ~2M params for d_model=2048 (negligible overhead, discarded at inference).
    """
    def __init__(self, d_model):
        super().__init__()
        hidden = d_model // 4
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, d_model, bias=False),
        )

    def forward(self, h_t, h_t1):
        """
        h_t:  [B, S, D] — current tick hidden
        h_t1: [B, S, D] — next tick hidden (stop-gradient target)
        returns: scalar TSSP loss
        """
        pred = self.net(h_t)
        target = h_t1.detach()  # stop-gradient — critical to prevent collapse
        return F.mse_loss(pred, target)


# ─── Time-Aware LLM ──────────────────────────────────────────────────────────
class TimeAwareLLM(nn.Module):
    """
    Full Time-Aware LLM:
    Embedding → [CTM Block × N_ticks (shared weights)] → Output projection

    The CTM block is applied N_ticks times with identical weights.
    Phases evolve via Kuramoto dynamics across ticks.
    TSSP auxiliary loss guides intermediate tick states.
    DHP gate monitors τ* convergence.
    """
    def __init__(self, vocab_size, d_model, n_heads, n_ticks, max_seq_len,
                 ffn_mult=4, dropout=0.1, kappa=1.0, dt=0.1):
        super().__init__()
        self.d_model    = d_model
        self.n_ticks    = n_ticks
        self.n_heads    = n_heads
        self.max_seq_len = max_seq_len

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, std=0.02)

        # Input projection: project embedding to initial hidden state
        # (allows embedding dim to differ from d_model if needed)
        self.input_proj = nn.LayerNorm(d_model)

        # Single CTM block — SHARED ACROSS ALL TICKS
        self.ctm_block = CTMBlock(
            d_model, n_heads, ffn_mult=ffn_mult, dropout=dropout,
            kappa=kappa, dt=dt, max_seq_len=max_seq_len
        )

        # TSSP head (training only, discarded at inference)
        self.tssp_head = TSSPHead(d_model)

        # Output norm + tied output projection
        self.out_norm = nn.LayerNorm(d_model)
        # Tie output weights to embedding (saves 100M params at d_model=2048, vocab=50k)
        self.out_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.out_proj.weight = self.embedding.weight  # weight tying

        # Scale embedding init for weight tying stability
        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02 / math.sqrt(2 * self.n_ticks))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input_ids, tssp_lambda=0.0, use_grad_ckpt=False):
        """
        input_ids: [B, S]
        tssp_lambda: current TSSP loss weight (0 = no TSSP loss)
        returns: (logits [B, S, V], total_loss_dict)
        """
        B, S = input_ids.shape
        device = input_ids.device

        # Embed tokens
        h = self.embedding(input_ids)     # [B, S, D]
        h = self.input_proj(h)            # layer norm before first tick

        # Initialize oscillator phases
        phases = self.ctm_block.attn.init_phases(B, S, device)

        # CTM recurrence — N_ticks with shared weights
        #
        # TSSP + gradient checkpointing: O(N²) recomputation hazard.
        # When tick_hiddens are stored for TSSP and each h_t came from a
        # checkpoint, backward through h_t re-triggers all prior checkpoints
        # leading to it — 230 recomputations for N_ticks=20 instead of 20.
        # Fix: disable grad_ckpt when TSSP is active. The extra memory is fine
        # (activations ~5GB, well within 25GB on 3090) and O(N) backward is
        # dramatically faster.
        use_ckpt_this_step = use_grad_ckpt and (tssp_lambda == 0.0) and self.training

        tick_hiddens = [h]  # h_0 = initial embedding

        for tick in range(self.n_ticks):
            if use_ckpt_this_step:
                h, phases = checkpoint(
                    self.ctm_block, h, phases, use_reentrant=False
                )
            else:
                h, phases = self.ctm_block(h, phases)

            # Store tick hiddens for TSSP — sample every 4 ticks (4 terms total)
            # to avoid dense dependency chains while still providing training signal
            if tssp_lambda > 0.0 or not self.training:
                if tick % (self.n_ticks // 4) == 0 or tick == self.n_ticks - 1:
                    tick_hiddens.append(h)

        # Output projection
        logits = self.out_proj(self.out_norm(h))  # [B, S, V]

        loss_dict = {}

        # TSSP loss: h_tick predicts h_{tick+1} via stop-gradient
        if tssp_lambda > 0.0 and self.training and len(tick_hiddens) > 1:
            tssp_losses = []
            for t in range(len(tick_hiddens) - 1):
                l = self.tssp_head(tick_hiddens[t], tick_hiddens[t + 1])
                tssp_losses.append(l)
            tssp_loss = torch.stack(tssp_losses).mean()
            loss_dict["tssp"] = tssp_loss.item()
            # Note: caller adds: total_loss = ce_loss + tssp_lambda * tssp_loss
            loss_dict["tssp_tensor"] = tssp_loss

        return logits, loss_dict

    def get_dhp_metrics(self, S):
        return self.ctm_block.attn.get_dhp_metrics(S)

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        # Subtract tied embedding params (counted twice)
        tied = self.embedding.weight.numel()
        actual = total - tied
        return actual, total


# ─── TSSP Lambda Schedule ─────────────────────────────────────────────────────
def get_tssp_lambda(step, max_steps, lambda_max=0.05,
                    warmup_frac=0.10, hold_frac=0.50):
    """
    Warmup-hold-cosine-decay schedule for TSSP auxiliary loss weight.

    Prevents scale-dependent inversion at >300M params (Paper 6/Aura blueprint):
    - Warmup: 0 → λ_max (first warmup_frac of steps)
    - Hold:   λ_max (from warmup to hold_frac of steps)
    - Decay:  cosine λ_max → 0 (hold_frac to end)

    The decay yields optimization dominance back to cross-entropy before
    gradient misalignment can cause late-stage regression.
    """
    warmup_end = int(max_steps * warmup_frac)
    hold_end   = int(max_steps * hold_frac)

    if step < warmup_end:
        return lambda_max * step / max(warmup_end, 1)
    elif step < hold_end:
        return lambda_max
    else:
        # Cosine decay from λ_max → 0
        progress = (step - hold_end) / max(max_steps - hold_end, 1)
        return lambda_max * 0.5 * (1 + math.cos(math.pi * progress))


# ─── LR Schedule ─────────────────────────────────────────────────────────────
def get_lr(step, max_steps, lr, warmup_steps=2000):
    if step < warmup_steps:
        return lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    return lr * 0.5 * (1 + math.cos(math.pi * progress))


# ─── Dataset ─────────────────────────────────────────────────────────────────
class TextChunkDataset(Dataset):
    """
    Tokenize a HuggingFace dataset and chunk into fixed-length sequences.
    """
    def __init__(self, texts, tokenizer, seq_len, max_tokens=None):
        super().__init__()
        self.seq_len = seq_len

        log("Tokenizing dataset...")
        all_ids = []
        for i, text in enumerate(texts):
            if i % 10000 == 0:
                log(f"  Tokenizing doc {i}/{len(texts)}")
            ids = tokenizer.encode(text, add_special_tokens=True)
            all_ids.extend(ids)
            if max_tokens and len(all_ids) >= max_tokens:
                all_ids = all_ids[:max_tokens]
                break

        # Chunk into seq_len blocks (overlapping by 1 for next-token prediction)
        self.data = torch.tensor(all_ids, dtype=torch.long)
        n_chunks = (len(self.data) - 1) // seq_len
        log(f"Total tokens: {len(self.data):,}, chunks: {n_chunks:,}")

    def __len__(self):
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.data[start : start + self.seq_len]
        y = self.data[start + 1 : start + self.seq_len + 1]
        return x, y


# ─── Main Training Loop ───────────────────────────────────────────────────────
def main():
    args = parse_args()

    global LOG_FILE
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    LOG_FILE = f"{args.save_dir}/train.log"

    log("=" * 70)
    log("DuoNeural Time-Aware LLM — Training Run")
    log(f"d_model={args.d_model}, n_heads={args.n_heads}, n_ticks={args.n_ticks}")
    log(f"seq_len={args.seq_len}, batch={args.batch_size}×{args.grad_accum} (eff)")
    log(f"kappa={args.kappa}, dt={args.dt}, tssp_λ_max={args.tssp_lambda_max}")
    log("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    if device.type == "cuda":
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        log(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    log("Loading GPT-2 tokenizer...")
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size  # 50257
    log(f"Vocab size: {vocab_size}")

    # ── Data ──────────────────────────────────────────────────────────────────
    log("Loading WikiText-103...")
    from datasets import load_dataset
    wt103 = load_dataset("wikitext", "wikitext-103-v1", trust_remote_code=False)
    train_texts = [t for t in wt103["train"]["text"] if len(t.strip()) > 50]
    val_texts   = [t for t in wt103["validation"]["text"] if len(t.strip()) > 50]
    log(f"Train docs: {len(train_texts):,}, Val docs: {len(val_texts):,}")

    train_ds = TextChunkDataset(train_texts, tokenizer, args.seq_len,
                                 max_tokens=50_000_000)  # 50M tokens
    val_ds   = TextChunkDataset(val_texts,   tokenizer, args.seq_len,
                                 max_tokens=2_000_000)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size * 2,
                              shuffle=False, num_workers=2, pin_memory=True,
                              drop_last=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    log("Initializing TimeAwareLLM...")
    model = TimeAwareLLM(
        vocab_size   = vocab_size,
        d_model      = args.d_model,
        n_heads      = args.n_heads,
        n_ticks      = args.n_ticks,
        max_seq_len  = args.seq_len,
        ffn_mult     = 4,
        dropout      = 0.1,
        kappa        = args.kappa,
        dt           = args.dt,
    ).to(device)

    actual_params, total_params = model.count_params()
    effective_params = actual_params * args.n_ticks  # computational depth
    log(f"Actual params:    {actual_params/1e6:.1f}M")
    log(f"Total (w/tied):   {total_params/1e6:.1f}M")
    log(f"Effective depth:  {effective_params/1e9:.2f}B ({args.n_ticks} ticks)")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # Separate weight decay: no decay for embeddings, layernorms, biases
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(x in name for x in ["norm", "bias", "embedding", "omega",
                                    "dhp_gate_logits", "phase_init"]):
            no_decay.append(p)
        else:
            decay.append(p)

    param_groups = [
        {"params": decay,    "weight_decay": 0.1},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    try_bnb = not args.no_bnb
    if try_bnb:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(param_groups, lr=args.lr)
            log("Using 8-bit AdamW (bitsandbytes)")
        except ImportError:
            optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
            log("bitsandbytes not found — using standard AdamW")
    else:
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
        log("Using standard AdamW")

    # ── AMP Scaler ────────────────────────────────────────────────────────────
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # ── Training ──────────────────────────────────────────────────────────────
    log(f"\nStarting training: {args.max_steps} steps")
    log(f"TSSP schedule: warmup@10% → hold@50% → cosine decay, λ_max={args.tssp_lambda_max}")
    log(f"DHP gate τ* will be logged every {args.eval_every} steps")
    log("-" * 70)

    model.train()
    step = 0
    optimizer.zero_grad()
    best_val_ppl = float("inf")
    train_losses = []
    train_tssp   = []

    # Save config
    cfg_path = f"{args.save_dir}/config.json"
    with open(cfg_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    log(f"Config saved: {cfg_path}")

    data_iter = iter(train_loader)
    t0 = time.time()

    while step < args.max_steps:
        # Get batch
        try:
            x_batch, y_batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x_batch, y_batch = next(data_iter)

        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        # TSSP lambda schedule
        tssp_lambda = get_tssp_lambda(
            step, args.max_steps,
            lambda_max     = args.tssp_lambda_max,
            warmup_frac    = 0.10,
            hold_frac      = 0.50,
        )

        # LR schedule
        lr = get_lr(step, args.max_steps, args.lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # Forward + loss
        autocast_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda"))
        with autocast_ctx:
            logits, loss_dict = model(
                x_batch,
                tssp_lambda    = tssp_lambda,
                use_grad_ckpt  = args.grad_ckpt,
            )
            # Cross-entropy loss (autoregressive LM)
            ce_loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                y_batch.view(-1),
                ignore_index = tokenizer.pad_token_id,
            )
            # Total loss
            total_loss = ce_loss
            if "tssp_tensor" in loss_dict and tssp_lambda > 0:
                total_loss = ce_loss + tssp_lambda * loss_dict["tssp_tensor"]

            # Scale by grad_accum
            total_loss = total_loss / args.grad_accum

        scaler.scale(total_loss).backward()
        train_losses.append(ce_loss.item())
        if "tssp" in loss_dict:
            train_tssp.append(loss_dict["tssp"])

        # Optimizer step every grad_accum micro-batches
        if (step + 1) % args.grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # ── Eval & Logging ────────────────────────────────────────────────────
        if step % args.eval_every == 0 or step == args.max_steps - 1:
            # DHP gate metrics
            tau_star, entropy, peakedness = model.get_dhp_metrics(args.seq_len)
            tau_L_frac = tau_star / args.seq_len  # fraction of window

            # Validation
            model.eval()
            val_losses = []
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx = vx.to(device); vy = vy.to(device)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        vlogits, _ = model(vx, tssp_lambda=0.0)
                        vl = F.cross_entropy(vlogits.view(-1, vocab_size), vy.view(-1))
                    val_losses.append(vl.item())
                    if len(val_losses) >= 50:  # ~50 batches ≈ enough for stable estimate
                        break

            val_ppl = math.exp(np.mean(val_losses))
            train_ppl = math.exp(np.mean(train_losses[-100:])) if train_losses else 0.0
            avg_tssp = np.mean(train_tssp[-100:]) if train_tssp else 0.0
            elapsed = time.time() - t0

            log(
                f"step={step:6d} | "
                f"train_ppl={train_ppl:7.2f} | val_ppl={val_ppl:7.2f} | "
                f"tssp={avg_tssp:.4f} λ={tssp_lambda:.4f} | "
                f"τ*={tau_star:.1f} ({tau_L_frac:.2%} window) | "
                f"gate_peak={peakedness:.3f} H={entropy:.3f} | "
                f"lr={lr:.2e} | {elapsed/60:.1f}min"
            )

            # Save checkpoint if best
            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
                ckpt_path = f"{args.save_dir}/best.pt"
                torch.save({
                    "step":    step,
                    "model":   model.state_dict(),
                    "optim":   optimizer.state_dict(),
                    "val_ppl": val_ppl,
                    "config":  vars(args),
                    "dhp": {
                        "tau_star":   tau_star,
                        "tau_L_frac": tau_L_frac,
                        "entropy":    entropy,
                        "peakedness": peakedness,
                    }
                }, ckpt_path)
                log(f"  ✓ New best val_ppl={val_ppl:.2f} → saved {ckpt_path}")

            model.train()

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if step > 0 and step % args.save_every == 0:
            ckpt_path = f"{args.save_dir}/step_{step:06d}.pt"
            torch.save({
                "step":  step,
                "model": model.state_dict(),
                "config": vars(args),
            }, ckpt_path)
            log(f"Checkpoint: {ckpt_path}")

        step += 1

    # ── Final Save ────────────────────────────────────────────────────────────
    final_path = f"{args.save_dir}/final.pt"
    torch.save({"step": step, "model": model.state_dict(), "config": vars(args)}, final_path)
    log(f"\nTraining complete! Best val_ppl={best_val_ppl:.2f}")
    log(f"Final checkpoint: {final_path}")

    # Final DHP metrics
    tau_star, entropy, peakedness = model.get_dhp_metrics(args.seq_len)
    log(f"\nFinal DHP Gate Analysis:")
    log(f"  τ* = {tau_star:.1f} tokens ({tau_star/args.seq_len:.2%} of window)")
    log(f"  Peakedness = {peakedness:.4f}, Entropy = {entropy:.4f}")
    log(f"  Expected τ*/window ≈ 0.72 if DHP converged")
    log(f"  (DHP Token Horizon exp found τ_L_text ≈ 64 tokens at byte level)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
