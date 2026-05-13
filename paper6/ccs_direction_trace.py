#!/usr/bin/env python3
"""
CCS Truth Direction Tracing — Stage 2 of three-part paper
Archon, DuoNeural, 2026-05-10

Computes:
  1. CCS direction r_L at each layer (normalized mean-diff)
  2. Direction magnitude (||r_L||) — encoding strength
  3. Direction rotation (cos_sim(r_L, r_{L+1})) — where does direction change?
  4. Experimental projection score — does r_L trained on ctrl generalize at each layer?
  5. Identifies "suppression layers" where direction rotates most dramatically

All CPU-only from precomputed activations (act_true.npy / act_false.npy).
Shape: (n_pairs, n_layers, hidden_dim)
  First n_ctrl pairs = control, last n_exp pairs = experimental.
"""

import numpy as np
import json
from pathlib import Path
import argparse

def compute_direction_trace(act_true, act_false, n_ctrl, n_exp):
    """
    act_true/act_false: (n_pairs, n_layers, hidden_dim)
    Returns per-layer dict of metrics.
    """
    n_pairs, n_layers, d = act_true.shape
    assert n_ctrl + n_exp == n_pairs, f"{n_ctrl}+{n_exp} != {n_pairs}"

    ctrl_true  = act_true[:n_ctrl]   # (n_ctrl, n_layers, d)
    ctrl_false = act_false[:n_ctrl]
    exp_true   = act_true[n_ctrl:]   # (n_exp,  n_layers, d)
    exp_false  = act_false[n_ctrl:]

    results = []

    prev_r = None
    for L in range(n_layers):
        h_ct = ctrl_true[:, L, :]   # (n_ctrl, d)
        h_cf = ctrl_false[:, L, :]
        h_et = exp_true[:, L, :]    # (n_exp, d)
        h_ef = exp_false[:, L, :]

        # 1. Truth direction = mean diff, unnormalized first
        diff = h_ct.mean(0) - h_cf.mean(0)          # (d,)
        magnitude = float(np.linalg.norm(diff))
        r = diff / (magnitude + 1e-10)               # unit vector

        # 2. Direction rotation from prev layer
        if prev_r is not None:
            cos_rot = float(np.clip(np.dot(r, prev_r), -1, 1))
            angle_deg = float(np.degrees(np.arccos(abs(cos_rot))))
        else:
            cos_rot = 1.0
            angle_deg = 0.0

        # 3. Ctrl separation score: how well does r separate ctrl pairs?
        ctrl_scores_t = (h_ct @ r)   # (n_ctrl,)
        ctrl_scores_f = (h_cf @ r)
        ctrl_gap = float((ctrl_scores_t - ctrl_scores_f).mean())
        ctrl_sign_acc = float(((ctrl_scores_t - ctrl_scores_f) > 0).mean())

        # 4. Experimental projection: does r (from ctrl) generalize to exp?
        exp_scores_t = (h_et @ r)    # (n_exp,)
        exp_scores_f = (h_ef @ r)
        exp_gap = float((exp_scores_t - exp_scores_f).mean())
        exp_sign_acc = float(((exp_scores_t - exp_scores_f) > 0).mean())

        # 5. Variance in hidden states (informativeness of layer)
        total_var = float(np.concatenate([h_ct, h_cf, h_et, h_ef]).var(axis=0).mean())

        results.append({
            "layer": L,
            "magnitude": magnitude,
            "cos_rot": cos_rot,
            "angle_deg": angle_deg,
            "ctrl_gap": ctrl_gap,
            "ctrl_sign_acc": ctrl_sign_acc,
            "exp_gap": exp_gap,
            "exp_sign_acc": exp_sign_acc,
            "total_var": total_var,
        })

        prev_r = r

    return results


def find_suppression_layers(trace, angle_threshold=10.0):
    """Flag layers where direction rotates sharply (suppression candidates)."""
    suppressed = []
    for i, r in enumerate(trace):
        if r["angle_deg"] > angle_threshold and i > 0:
            suppressed.append({
                "layer": r["layer"],
                "angle_deg": r["angle_deg"],
                "exp_sign_acc_before": trace[i-1]["exp_sign_acc"],
                "exp_sign_acc_after":  r["exp_sign_acc"],
                "exp_gap_delta": r["exp_gap"] - trace[i-1]["exp_gap"],
            })
    return suppressed


def print_report(trace, suppression_layers, model_name, n_ctrl, n_exp):
    print(f"\n{'='*70}")
    print(f"Truth Direction Trace — {model_name}")
    print(f"n_ctrl={n_ctrl}, n_exp={n_exp}, n_layers={len(trace)}, d={5120}")
    print(f"{'='*70}")
    print(f"{'L':>4}  {'|r|':>6}  {'angle':>6}  {'ctrl_acc':>8}  {'exp_acc':>8}  {'exp_gap':>8}")
    print(f"{'':->4}  {'':->6}  {'':->6}  {'':->8}  {'':->8}  {'':->8}")

    for r in trace:
        flag = ""
        if r["angle_deg"] > 15:
            flag = " << ROTATE"
        elif r["exp_sign_acc"] > 0.79:
            flag = " ** PEAK"
        print(f"{r['layer']:>4}  {r['magnitude']:>6.1f}  {r['angle_deg']:>6.1f}°  "
              f"{r['ctrl_sign_acc']:>8.2f}  {r['exp_sign_acc']:>8.2f}  "
              f"{r['exp_gap']:>8.3f}{flag}")

    print(f"\n--- Suppression Layers (rotation > 10°) ---")
    if suppression_layers:
        for s in suppression_layers:
            delta = s['exp_gap_delta']
            direction = "↑ improves" if delta > 0 else "↓ degrades"
            print(f"  L{s['layer']:02d}: {s['angle_deg']:.1f}°  "
                  f"exp_acc: {s['exp_sign_acc_before']:.2f} → {s['exp_sign_acc_after']:.2f}  "
                  f"gap Δ={delta:+.3f} {direction}")
    else:
        print("  None found (all layers rotate < 10°)")

    # Summary stats
    best_exp = max(trace, key=lambda x: x["exp_sign_acc"])
    print(f"\n--- Summary ---")
    print(f"  Best exp accuracy:  L{best_exp['layer']:02d} = {best_exp['exp_sign_acc']:.2f}")
    print(f"  Max direction mag:  L{max(trace, key=lambda x: x['magnitude'])['layer']:02d} = "
          f"{max(t['magnitude'] for t in trace):.1f}")
    print(f"  Max rotation:       L{max(trace[1:], key=lambda x: x['angle_deg'])['layer']:02d} = "
          f"{max(t['angle_deg'] for t in trace[1:]):.1f}°")
    print(f"  Avg ctrl acc:       {sum(t['ctrl_sign_acc'] for t in trace)/len(trace):.2f}")
    print(f"  Avg exp acc:        {sum(t['exp_sign_acc'] for t in trace)/len(trace):.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--act-dir", default="ccs_archive/ccs_results",
                        help="Dir containing act_true.npy, act_false.npy, results.json")
    parser.add_argument("--out", default="paper6/ccs_direction_trace_results.json")
    parser.add_argument("--model-name", default="Qwen2.5-14B-Instruct")
    args = parser.parse_args()

    act_dir = Path(args.act_dir)
    print(f"Loading activations from {act_dir}...")
    act_true  = np.load(act_dir / "act_true.npy")
    act_false = np.load(act_dir / "act_false.npy")
    print(f"  Shape: {act_true.shape}  dtype: {act_true.dtype}")

    with open(act_dir / "results.json") as f:
        meta = json.load(f)
    n_ctrl = meta["n_control"]
    n_exp  = meta["n_experimental"]
    print(f"  n_ctrl={n_ctrl}, n_exp={n_exp}, n_layers={meta['n_layers']}")

    trace = compute_direction_trace(act_true, act_false, n_ctrl, n_exp)
    suppression = find_suppression_layers(trace, angle_threshold=10.0)

    print_report(trace, suppression, args.model_name, n_ctrl, n_exp)

    out = {
        "model": args.model_name,
        "n_ctrl": n_ctrl, "n_exp": n_exp,
        "trace": trace,
        "suppression_layers": suppression,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
