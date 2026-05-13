"""
Paper 6 Figure Generation
DuoNeural — 2026-05-10
Generates all manuscript figures from experimental results.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.labelsize': 12, 'axes.titlesize': 13,
    'legend.fontsize': 10, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False,
})

C_KURAMOTO = '#2166ac'   # blue
C_DOTPROD  = '#d6604d'   # red-orange
C_PHYSARUM = '#4dac26'   # green
C_HC       = '#762a83'   # purple
C_NEUTRAL  = '#888888'

# ── Fig 1: C3 Escape — Halvorsen headline result ───────────────────────────────
def fig1_c3_escape():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: full multi-attractor results
    ax = axes[0]
    conditions = ['dot\nLorenz', 'kura\nLorenz', 'dot\nRössler', 'kura\nRössler',
                  'dot\nHalvorsen', 'kura\nHalvorsen']
    ratios     = [0.611,          0.886,           0.505,          0.498,
                  0.001,          0.953]
    colors     = [C_DOTPROD, C_KURAMOTO, C_DOTPROD, C_KURAMOTO,
                  C_DOTPROD, C_KURAMOTO]
    dhp_thresh = 0.65

    bars = ax.bar(range(6), ratios, color=colors, edgecolor='white',
                  linewidth=0.8, width=0.65)
    ax.axhline(dhp_thresh, color='black', linestyle='--', linewidth=1.2, alpha=0.6,
               label=f'DHP threshold ({dhp_thresh})')
    ax.set_xticks(range(6)); ax.set_xticklabels(conditions, fontsize=9)
    ax.set_ylabel(r'$\tau^*/\tau_L$'); ax.set_ylim(0, 1.05)
    ax.set_title('Multi-Attractor Results')
    ax.legend(loc='upper left', fontsize=9)

    # annotate Halvorsen
    ax.annotate('0.001', xy=(4, 0.001), xytext=(4, 0.12),
                ha='center', fontsize=9, color=C_DOTPROD,
                arrowprops=dict(arrowstyle='->', color=C_DOTPROD, lw=1.2))
    ax.annotate('0.953 (*)', xy=(5, 0.953), xytext=(5.4, 0.82),
                ha='center', fontsize=10, color=C_KURAMOTO, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_KURAMOTO, lw=1.2))

    # Right: Halvorsen close-up — three-regime schematic
    ax2 = axes[1]
    regimes = ['Interior Mode\n(Lorenz)', 'Interior Mode\n(Halvorsen)', 'Diffuse Mode\n(Rössler)']
    kura_r  = [0.886, 0.953, 0.498]
    dot_r   = [0.611, 0.001, 0.505]
    x = np.arange(3); w = 0.35
    ax2.bar(x - w/2, kura_r, width=w, color=C_KURAMOTO, label='kuramoto_free', edgecolor='white')
    ax2.bar(x + w/2, dot_r,  width=w, color=C_DOTPROD,  label='dot_product',   edgecolor='white')
    ax2.axhline(dhp_thresh, color='black', linestyle='--', linewidth=1.2, alpha=0.6)
    ax2.set_xticks(x); ax2.set_xticklabels(regimes, fontsize=9.5)
    ax2.set_ylabel(r'$\tau^*/\tau_L$'); ax2.set_ylim(0, 1.05)
    ax2.set_title('By Attractor Regime')
    ax2.legend(loc='upper right', fontsize=9)

    # shade diffuse region
    ax2.axvspan(1.6, 2.4, alpha=0.08, color='gray', label='Diffuse Mode')

    for ax_ in axes:
        ax_.yaxis.grid(True, alpha=0.3, linewidth=0.5)
        ax_.set_axisbelow(True)

    fig.suptitle('KGBN Phase-Locking vs. Dot-Product Attention\nAcross Chaotic Attractors',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig1_c3_escape.pdf')
    fig.savefig(FIG_DIR / 'fig1_c3_escape.png')
    plt.close(); print("Fig 1 done")

# ── Fig 2: Three Attractor Regimes Taxonomy ────────────────────────────────────
def fig2_regime_taxonomy():
    fig, ax = plt.subplots(figsize=(8, 5))

    tau_l = 1.0  # normalized

    # Interior mode: tau* converges to ~0.70-0.95 x tau_L
    t_gate_vals = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    lorenz_tauk  = np.array([0.70, 0.80, 0.85, 0.88, 0.89])  # saturates at tau_L
    halvors_tauk = np.array([0.75, 0.90, 0.95, 0.95, 0.95])

    # Diffuse mode: tau* = T_gate/2 exactly
    rossler_tauk = t_gate_vals / 2  # linear in T_gate

    # Saturation mode (from paper 4): tau* saturates at T_gate when T_gate < tau_L
    sat_tauk = np.minimum(t_gate_vals * 0.9, 0.72)  # hypothetical

    ax.plot(t_gate_vals, lorenz_tauk,  'o-', color=C_KURAMOTO, label='Interior (Lorenz) — kuramoto', lw=2, ms=7)
    ax.plot(t_gate_vals, halvors_tauk, 's-', color='#1a7abf', label='Interior (Halvorsen) — kuramoto', lw=2, ms=7)
    ax.plot(t_gate_vals, rossler_tauk, '^--', color=C_NEUTRAL, label='Diffuse (Rössler) — both gates', lw=2, ms=7)

    ax.axhline(0.65, color='black', linestyle=':', linewidth=1.2, alpha=0.7, label='DHP threshold (0.65)')
    ax.axhline(1.0,  color='black', linestyle='-', linewidth=0.8, alpha=0.3, label=r'$\tau_L$ ceiling')

    # Shade regions
    ax.axhspan(0.65, 1.05, alpha=0.06, color=C_KURAMOTO)
    ax.fill_between(t_gate_vals, rossler_tauk - 0.02, rossler_tauk + 0.02, alpha=0.15, color=C_NEUTRAL)

    # Annotations
    ax.text(2.3, 0.91, 'Interior Mode\n(converges to $\\tau_L$)', color=C_KURAMOTO,
            fontsize=9.5, ha='center')
    ax.text(2.3, 0.54, 'Diffuse Mode\n($= T_{gate}/2$)', color='#555555',
            fontsize=9.5, ha='center')

    ax.set_xlabel(r'Gate Window $T_{gate}$ (units of $\tau_L$)')
    ax.set_ylabel(r'$\tau^*/\tau_L$')
    ax.set_title('Three Temporal Regimes: $\\tau^*$ as a Function of Gate Window Width',
                 fontsize=12)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(0.3, 2.7); ax.set_ylim(0, 1.08)
    ax.yaxis.grid(True, alpha=0.3, linewidth=0.5); ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig2_regime_taxonomy.pdf')
    fig.savefig(FIG_DIR / 'fig2_regime_taxonomy.png')
    plt.close(); print("Fig 2 done")

# ── Fig 3: Physarum RC — Physical Boundary ────────────────────────────────────
def fig3_physarum():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    # Left: tau*/tau_L per condition
    ax = axes[0]
    conds   = ['geometric\nRC', 'uniform\nRC', 'single_tau\n(cheating)', 'mismatched\nRC']
    ratios  = [0.243, 0.523, 0.995, 0.144]
    colors  = [C_PHYSARUM, '#74c476', '#31a354', '#addd8e']
    bars = ax.bar(conds, ratios, color=colors, edgecolor='white', linewidth=0.8, width=0.55)
    ax.axhline(0.65, color='black', linestyle='--', linewidth=1.2, alpha=0.6,
               label='DHP threshold')
    ax.set_ylabel(r'$\tau^*/\tau_L$'); ax.set_ylim(0, 1.08)
    ax.set_title('Physarum RC — No Learning')
    ax.legend(fontsize=9)

    # Annotate
    ax.text(2, 1.005, '0.995\n(knows $\\tau_L$)', ha='center', fontsize=8.5,
            color='#31a354', style='italic')
    ax.text(0, 0.243 + 0.04, '0.243', ha='center', fontsize=9)
    ax.text(1, 0.523 + 0.04, '0.523', ha='center', fontsize=9)
    ax.text(3, 0.144 + 0.04, '0.144', ha='center', fontsize=9)

    # Right: R² vs ratio scatter — shows high R² with low ratio (can HOLD, can't SEEK)
    ax2 = axes[1]
    r2_vals  = [0.750, 0.867, 0.463, 0.813]
    ratio_v  = [0.243, 0.523, 0.995, 0.144]
    labels   = ['geometric', 'uniform', 'single_τ', 'mismatched']
    clrs     = [C_PHYSARUM, '#74c476', '#31a354', '#addd8e']

    for i, (r2, rat, lab, c) in enumerate(zip(r2_vals, ratio_v, labels, clrs)):
        ax2.scatter(r2, rat, s=120, color=c, edgecolors='black', linewidth=0.8, zorder=3)
        ax2.annotate(lab, (r2, rat), textcoords='offset points',
                     xytext=(8, 3), fontsize=9)

    ax2.axhline(0.65, color='black', linestyle='--', linewidth=1.2, alpha=0.6,
                label='DHP threshold')
    ax2.axvline(0.70, color='gray', linestyle=':', linewidth=1.0, alpha=0.5)

    # Arrow: "can HOLD τ_L" (high R²) vs "can't SEEK it" (low ratio)
    ax2.text(0.45, 0.15, '← Can predict well\n   but cannot seek $\\tau_L$',
             fontsize=8.5, color='#555', style='italic')

    ax2.set_xlabel(r'Prediction $R^2$ (quality of fit)')
    ax2.set_ylabel(r'$\tau^*/\tau_L$ (temporal depth found)')
    ax2.set_title('Prediction Quality vs. Temporal Depth')
    ax2.set_xlim(0.35, 1.05); ax2.set_ylim(0, 1.1)
    ax2.yaxis.grid(True, alpha=0.3); ax2.xaxis.grid(True, alpha=0.3)
    ax2.set_axisbelow(True); ax2.legend(fontsize=9)

    for ax_ in axes:
        pass
    fig.suptitle('Physarum RC Probe: Passive Physics Cannot Discover the Lyapunov Horizon',
                 fontsize=12, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig3_physarum.pdf')
    fig.savefig(FIG_DIR / 'fig3_physarum.png')
    plt.close(); print("Fig 3 done")

# ── Fig 4: HC-SITH v1 vs v2 ───────────────────────────────────────────────────
def fig4_hcsith():
    fig, ax = plt.subplots(figsize=(8, 4.5))

    conds_v1 = ['uniform\nfixed', 'uniform\ntrained', 'geometric\nfixed', 'geometric\ntrained']
    conds_v2 = ['uniform\nv2', 'geometric\nv2', 'geometric\nv2\nlearned']

    r_v1 = [0.369, 0.369, 0.552, 0.552]
    r_v2 = [0.887, 0.801, 0.755]

    x1 = np.arange(len(conds_v1))
    x2 = np.arange(len(conds_v2)) + len(conds_v1) + 0.8

    bars1 = ax.bar(x1, r_v1, color=C_HC, alpha=0.45, edgecolor='white',
                   linewidth=0.8, width=0.6, label='HC-SITH v1 (Gumbel gate)')
    bars2 = ax.bar(x2, r_v2, color=C_HC, alpha=1.0,  edgecolor='white',
                   linewidth=0.8, width=0.6, label='HC-SITH v2 (block-norm gate)')

    ax.axhline(0.65, color='black', linestyle='--', linewidth=1.2, alpha=0.7,
               label='DHP threshold (0.65)')

    # Separator
    ax.axvline(len(conds_v1) + 0.3, color='gray', linewidth=1.0, alpha=0.4, linestyle=':')
    ax.text(len(conds_v1) + 0.35, 0.95, 'v2 →\nblock-norm\ncoupling',
            fontsize=8.5, color='#555', ha='left', va='top')
    ax.text(len(conds_v1)/2 - 0.5, 0.95, '← v1\nGumbel gate',
            fontsize=8.5, color='#888', ha='center', va='top')

    xtick_pos   = list(x1) + list(x2)
    xtick_labs  = conds_v1 + conds_v2
    ax.set_xticks(xtick_pos); ax.set_xticklabels(xtick_labs, fontsize=9)
    ax.set_ylabel(r'$\tau^*/\tau_L$ (mean over 4 seeds)'); ax.set_ylim(0, 1.05)
    ax.set_title('HC-SITH v1 → v2: Block-Norm Gating Breaks the DHP Bottleneck', fontsize=12)
    ax.legend(loc='upper left', fontsize=9)
    ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)

    # Improvement arrows
    for x1i, x2i, r1, r2 in [(0, len(conds_v1)+0.8, 0.369, 0.887),
                               (2, len(conds_v1)+1.8, 0.552, 0.801)]:
        ax.annotate('', xy=(x2i, r2-0.02), xytext=(x1i, r1+0.02),
                    arrowprops=dict(arrowstyle='->', color=C_HC, lw=1.2, alpha=0.5,
                                   connectionstyle='arc3,rad=0.25'))

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig4_hcsith.pdf')
    fig.savefig(FIG_DIR / 'fig4_hcsith.png')
    plt.close(); print("Fig 4 done")

# ── Fig 5: Architecture vs Objective — DHP Mechanism Taxonomy ─────────────────
def fig5_mechanism_taxonomy():
    fig, ax = plt.subplots(figsize=(8, 5))

    systems = [
        ('Passive RC\n(geometric)', 0.243, 'passive_physics', '#aaaaaa'),
        ('Passive RC\n(uniform)',   0.523, 'passive_physics', '#bbbbbb'),
        ('LA-SSM\n(Lyap. loss)',    0.323, 'objective_only',  '#f4a582'),
        ('HC-SITH v2\n(uniform)',   0.887, 'arch_coupled',    C_HC),
        ('HC-SITH v2\n(geometric)', 0.801, 'arch_coupled',    '#9970ab'),
        ('KGBN\ndot-product',       0.611, 'arch_coupled',    '#fc8d59'),
        ('KGBN\nkuramoto\nLorenz',  0.886, 'phase_lock',      C_KURAMOTO),
        ('KGBN\nkuramoto\nHalvorsen', 0.953, 'phase_lock',    '#1a5276'),
    ]

    categories = {
        'passive_physics': ('Passive Physics', '#cccccc'),
        'objective_only':  ('Training Obj. Only', '#f4a582'),
        'arch_coupled':    ('Architecture-Coupled', C_HC),
        'phase_lock':      ('Phase-Locking\n(geometry-agnostic)', C_KURAMOTO),
    }

    x_pos = np.arange(len(systems))
    for i, (name, ratio, cat, color) in enumerate(systems):
        ax.bar(i, ratio, color=color, edgecolor='white', linewidth=0.8, width=0.7, alpha=0.92)

    ax.axhline(0.65, color='black', linestyle='--', linewidth=1.3, alpha=0.7,
               label='DHP threshold (0.65)')

    ax.set_xticks(x_pos)
    ax.set_xticklabels([s[0] for s in systems], fontsize=8.5)
    ax.set_ylabel(r'$\tau^*/\tau_L$'); ax.set_ylim(0, 1.05)
    ax.set_title('DHP Discovery by System Type:\nArchitecture is the Load-Bearing Mechanism', fontsize=12)
    ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)

    # Category brackets
    brackets = [
        (0, 1, 'Passive\nPhysics', '#aaaaaa'),
        (2, 2, 'Training\nObj. Only', '#f4a582'),
        (3, 5, 'Architecture-\nCoupled', C_HC),
        (6, 7, 'Phase-Locking\n(geom.-agnostic)', C_KURAMOTO),
    ]
    for x_start, x_end, label, color in brackets:
        y_brk = -0.13
        ax.annotate('', xy=(x_end+0.3, y_brk), xytext=(x_start-0.3, y_brk),
                    xycoords=('data','axes fraction'), textcoords=('data','axes fraction'),
                    arrowprops=dict(arrowstyle='-', color=color, lw=2.0))
        ax.text((x_start+x_end)/2, y_brk - 0.08, label,
                ha='center', va='top', fontsize=8, color=color, fontweight='bold',
                transform=ax.get_xaxis_transform())

    ax.legend(fontsize=9, loc='upper left')
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(FIG_DIR / 'fig5_mechanism_taxonomy.pdf')
    fig.savefig(FIG_DIR / 'fig5_mechanism_taxonomy.png')
    plt.close(); print("Fig 5 done")

if __name__ == "__main__":
    fig1_c3_escape()
    fig2_regime_taxonomy()
    fig3_physarum()
    fig4_hcsith()
    fig5_mechanism_taxonomy()
    print("\nAll figures saved to ./figures/")
