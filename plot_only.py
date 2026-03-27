"""
═══════════════════════════════════════════════════════════════════
 PLOT ONLY — Regenerate all 10 figures from saved CSV/NPZ
 Runtime: <5 seconds (no computation)

 Prerequisites: run compute_and_save_v2.py once.
 Required files in paper_outputs/:
   params.csv, load_protocol.csv, error_curves_miehe.csv,
   error_curves_bourdin.csv, fatigue_curves.csv,
   convergence_tension.csv, convergence_shear.csv,
   tip_errors_sent.csv, shear_tip_errors.csv,
   spatial_fields.npz, load_displacement.csv,
   load_displacement_shear.csv
═══════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ═══════════════════════════════════════════════════════════════
#  STYLE
# ═══════════════════════════════════════════════════════════════

SINGLE_COL = 3.31
DOUBLE_COL = 6.85

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'DejaVu Serif', 'Times'],
    'mathtext.fontset': 'cm',
    'font.size': 8, 'axes.labelsize': 9, 'axes.titlesize': 9,
    'legend.fontsize': 6.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'figure.dpi': 150, 'savefig.dpi': 300,
    'axes.linewidth': 0.6, 'lines.linewidth': 1.2, 'lines.markersize': 3.5,
    'grid.linewidth': 0.4, 'grid.alpha': 0.3,
    'legend.frameon': True, 'legend.framealpha': 0.9, 'legend.edgecolor': '0.8',
    'text.usetex': False,
})

C_MIEHE  = '#C0392B'; C_BOURDIN = '#2471A3'
C_GRID8  = '#2E86C1'; C_GRID16  = '#E67E22'
C_GRID32 = '#27AE60'; C_RBF200  = '#8E44AD'
SC = {'grid_8': C_GRID8, 'grid_16': C_GRID16, 'grid_32': C_GRID32, 'rbf_200': C_RBF200}
SM = {'grid_8': 'o', 'grid_16': 's', 'grid_32': '^', 'rbf_200': 'D'}

OUT = 'paper_outputs'
os.makedirs(f'{OUT}/pdf', exist_ok=True)
os.makedirs(f'{OUT}/png', exist_ok=True)

def savefig(fig, name):
    fig.savefig(f'{OUT}/pdf/{name}.pdf', bbox_inches='tight', pad_inches=0.02)
    fig.savefig(f'{OUT}/png/{name}.png', dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f'  ✓ {name}')

# ═══════════════════════════════════════════════════════════════
#  LOAD DATA
# ═══════════════════════════════════════════════════════════════

print("Loading data from CSV/NPZ...")
params = {}
with open(f'{OUT}/params.csv') as f:
    import csv
    for row in csv.DictReader(f):
        params[row['key']] = float(row['value'])

P = params
N_mesh = int(P['N_mesh']); N_ph1 = int(P['N_ph1']); N_ph2 = int(P['N_ph2'])
N_ph3 = int(P['N_ph3']); N_steps = int(P['N_steps']); ell = P['ell']

load = pd.read_csv(f'{OUT}/load_protocol.csv')
u_loads = load['u_bar'].values
u_peak = P['u_peak']

em = pd.read_csv(f'{OUT}/error_curves_miehe.csv')
eb = pd.read_csv(f'{OUT}/error_curves_bourdin.csv')
fat = pd.read_csv(f'{OUT}/fatigue_curves.csv')
ct = pd.read_csv(f'{OUT}/convergence_tension.csv')
cs = pd.read_csv(f'{OUT}/convergence_shear.csv')
tip_t = pd.read_csv(f'{OUT}/tip_errors_sent.csv')
tip_s = pd.read_csv(f'{OUT}/shear_tip_errors.csv')

# Load-displacement data (NEW)
ld = pd.read_csv(f'{OUT}/load_displacement.csv')
ld_shear = pd.read_csv(f'{OUT}/load_displacement_shear.csv')

sp = np.load(f'{OUT}/spatial_fields.npz')
N = int(sp['N_mesh'][0])
xs_plot = sp['xs_plot']

# Fatigue helper
p_fat = 0.5; xi_fat = 0.4
def f_fat(a):
    return (1 + 2*p_fat*(1-xi_fat)*np.maximum(a, 0.0))**(-1/(1-xi_fat))

print("  ✓ All data loaded\n")

schemes = ['grid_8', 'grid_16', 'grid_32', 'rbf_200']

# ═══════════════════════════════════════════════════════════════
#  FIG 01: Setup — FIX: equal column widths
# ═══════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(DOUBLE_COL, 2.8))
gs1 = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.25)

for idx, (bc_label, bc_color, panel_label) in enumerate([
    (r'$\bar{u}_y$', C_MIEHE, '(a) SEN-T'),
    (r'$\bar{u}_x$', C_BOURDIN, '(b) SEN-S')]):
    ax = fig.add_subplot(gs1[idx])
    ax.add_patch(plt.Rectangle((0,0), 1, 1, fill=True, fc='#F8F9FA', ec='k', lw=1.0))
    ax.plot([0, 0.5], [0.5, 0.5], 'k-', lw=2.5)
    ax.plot(0.5, 0.5, 'k>', ms=4)
    ax.annotate(r'$a_0\!=\!L/2$', xy=(0.25, 0.5), xytext=(0.25, 0.65),
                fontsize=6, ha='center', color='#555',
                arrowprops=dict(arrowstyle='->', color='#555', lw=0.5))
    if idx == 0:
        for xp_bc in np.linspace(0.1, 0.9, 5):
            ax.annotate('', xy=(xp_bc, 1.08), xytext=(xp_bc, 1.0),
                        arrowprops=dict(arrowstyle='->', color=bc_color, lw=0.8))
    else:
        for xp_bc in np.linspace(0.1, 0.9, 5):
            ax.annotate('', xy=(xp_bc+0.08, 1.03), xytext=(xp_bc, 1.03),
                        arrowprops=dict(arrowstyle='->', color=bc_color, lw=0.8))
    ax.text(0.5, 1.12, bc_label, ha='center', fontsize=9, color=bc_color, fontweight='bold')
    ax.plot([0, 1], [-0.02, -0.02], 'k-', lw=1.5)
    for xh in np.linspace(0.03, 0.97, 10):
        ax.plot([xh, xh-0.03], [-0.02, -0.07], 'k-', lw=0.4)
    ax.text(0.5, -0.12, 'fixed', ha='center', fontsize=6, color='#777')
    ax.annotate('', xy=(1.06, 0), xytext=(1.06, 1),
                arrowprops=dict(arrowstyle='<->', color='gray', lw=0.6))
    ax.text(1.12, 0.5, '$L$', ha='center', va='center', fontsize=8, color='gray')
    ax.add_patch(plt.Rectangle((0.42, 0.42), 0.16, 0.16, fill=True,
                                fc='#FADBD8', ec='none', alpha=0.5))
    ax.text(0.58, 0.42, r'$\sim\!\ell$', fontsize=6, color='#922B21')
    ax.set_xlim(-0.08, 1.2); ax.set_ylim(-0.16, 1.18)
    ax.set_aspect('equal'); ax.axis('off')
    ax.text(0.5, -0.22, panel_label, ha='center', fontsize=9, fontweight='bold')

# (c) Load protocol — SAME width as other panels
ax = fig.add_subplot(gs1[2])
steps_arr = np.arange(1, N_steps+1)
cc = ['#27AE60']*N_ph1 + [C_MIEHE]*N_ph2 + [C_BOURDIN]*N_ph3
for i in range(N_steps-1):
    ax.plot(steps_arr[i:i+2], u_loads[i:i+2]*1e3, color=cc[i], lw=1.5)
ax.axvline(x=N_ph1, color='gray', ls='--', lw=0.5, alpha=0.5)
ax.axvline(x=N_ph1+N_ph2, color='gray', ls='--', lw=0.5, alpha=0.5)
for txt, x, c in [('Load', N_ph1/2, '#27AE60'),
                   ('Unload', N_ph1+N_ph2/2, C_MIEHE),
                   ('Reload', N_ph1+N_ph2+N_ph3/2, C_BOURDIN)]:
    ax.text(x, u_loads.max()*1.08e3, txt, ha='center', fontsize=6.5, color=c, fontweight='bold')
ax.set_xlabel('Load step $n$')
ax.set_ylabel(r'$\bar{u}$ [$\times 10^{-3}$ mm]')
ax.grid(True, alpha=0.2)
ax.text(0.02, 0.95, '(c)', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')
savefig(fig, 'fig01_setup')


# ═══════════════════════════════════════════════════════════════
#  FIG 02: Fields — FIX: proper layout, no squeezing
# ═══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, 4.0))

d_load = sp['d_exact_load'].reshape(N+1, N+1)
d_reload = sp['d_exact_reload'].reshape(N+1, N+1)
H_ref = sp['H_ref_nodal'].reshape(N+1, N+1)
H_t16 = sp['H_t16_nodal'].reshape(N+1, N+1)

im0 = axes[0,0].pcolormesh(xs_plot, xs_plot, d_load, cmap='inferno', vmin=0, vmax=1, shading='auto')
axes[0,0].set_title(f'(a) $d$, end of load ($n={N_ph1}$)', fontsize=8)
axes[0,0].set_aspect('equal'); plt.colorbar(im0, ax=axes[0,0], shrink=0.8)

im1 = axes[0,1].pcolormesh(xs_plot, xs_plot, d_reload, cmap='inferno', vmin=0, vmax=1, shading='auto')
axes[0,1].set_title(f'(b) $d$, end of reload ($n={N_steps}$)', fontsize=8)
axes[0,1].set_aspect('equal'); plt.colorbar(im1, ax=axes[0,1], shrink=0.8)

im2 = axes[1,0].pcolormesh(xs_plot, xs_plot, np.clip(H_ref, 0, 5), cmap='viridis', vmin=0, vmax=5, shading='auto')
axes[1,0].set_title(r'(c) $\mathcal{H}_{\mathrm{ref}}$ (clipped $\leq 5$)', fontsize=8)
axes[1,0].set_aspect('equal'); plt.colorbar(im2, ax=axes[1,0], shrink=0.8)

im3 = axes[1,1].pcolormesh(xs_plot, xs_plot, np.abs(H_t16 - H_ref), cmap='Reds', shading='auto')
axes[1,1].set_title(r'(d) $|\widetilde{\mathcal{H}}-\mathcal{H}|$ (grid 16)', fontsize=8)
axes[1,1].set_aspect('equal'); plt.colorbar(im3, ax=axes[1,1], shrink=0.8)

for ax in axes.flat:
    ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
plt.tight_layout()
savefig(fig, 'fig02_fields_evolution')


# ═══════════════════════════════════════════════════════════════
#  FIG 03: Error growth — FIX: legend bottom-right
# ═══════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(DOUBLE_COL*0.75, 3.0))
for sn in schemes:
    ax.semilogy(em['step'], em[f'{sn}_H_global'].clip(lower=1e-15),
                marker=SM[sn], ms=2.5, lw=1.0, color=SC[sn],
                label=f'Miehe, {sn.replace("_"," ")}', markevery=3)
ax.semilogy(eb['step'], eb['grid_16_d_global'].clip(lower=1e-15),
            '-', color=C_BOURDIN, lw=2.5, alpha=0.8,
            label=r'Bourdin, grid 16')
ax.axvspan(1, N_ph1, alpha=0.05, color='green')
ax.axvspan(N_ph1, N_ph1+N_ph2, alpha=0.05, color='red')
ax.axvspan(N_ph1+N_ph2, N_steps, alpha=0.05, color='blue')
# Gap annotation
m_final = em['grid_16_H_global'].iloc[-1]
b_final = eb['grid_16_d_global'].iloc[-1]
gap = m_final / b_final
ax.annotate('', xy=(48, b_final), xytext=(48, m_final),
            arrowprops=dict(arrowstyle='<->', color='#333', lw=1.2))
ax.text(46, 0.3, f'{gap:.0f}×', fontsize=8, ha='center', color='#333', fontweight='bold')
ax.set_ylim(1e-8, 20)
ax.set_xlabel('Load step $n$')
ax.set_ylabel(r'$L^2$ error of carried variable')
ax.legend(frameon=True, ncol=2, fontsize=5.5, loc='lower right',
          columnspacing=1.0, handlelength=1.5)
ax.grid(True, alpha=0.2, which='both')
plt.tight_layout(); savefig(fig, 'fig03_error_growth')


# ═══════════════════════════════════════════════════════════════
#  FIG 04: Tip error evolution — FIX: annotations in boxes
# ═══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.0))

ax = axes[0]
for sn in schemes:
    ax.semilogy(em['step'], em[f'{sn}_H_tip'].clip(lower=1e-15),
                marker=SM[sn], ms=2.5, lw=1.2, color=SC[sn],
                label=sn.replace("_"," "), markevery=3)
ax.axvspan(N_ph1, N_ph1+N_ph2, alpha=0.06, color='red')
ax.set_xlabel('Load step $n$')
ax.set_ylabel(r'$\mathcal{H}$ tip error ($r<5\ell$)')
ax.set_title('(a) Miehe: tip error')
ax.legend(fontsize=6, loc='lower right'); ax.grid(True, alpha=0.2, which='both')
ax.set_ylim(1e-8, 20)
ax.text(0.97, 0.97, 'frozen during\nunload', transform=ax.transAxes, fontsize=6,
        color=C_MIEHE, ha='right', va='top', fontstyle='italic',
        bbox=dict(boxstyle='round,pad=0.3', fc='#FDEDEC', ec=C_MIEHE, alpha=0.8))

ax = axes[1]
for sn in schemes:
    ax.semilogy(eb['step'], eb[f'{sn}_d_tip'].clip(lower=1e-15),
                marker=SM[sn], ms=2.5, lw=1.2, color=SC[sn],
                label=sn.replace("_"," "), markevery=3)
ax.axvspan(N_ph1, N_ph1+N_ph2, alpha=0.06, color='red')
ax.set_xlabel('Load step $n$')
ax.set_ylabel(r'$d$ tip error ($r<5\ell$)')
ax.set_title('(b) Bourdin: tip error')
ax.legend(fontsize=6, loc='upper right'); ax.grid(True, alpha=0.2, which='both')
ax.set_ylim(1e-4, 2)
ax.text(0.97, 0.03, 'stable across\nall phases', transform=ax.transAxes, fontsize=6,
        color=C_BOURDIN, ha='right', va='bottom', fontstyle='italic',
        bbox=dict(boxstyle='round,pad=0.3', fc='#EBF5FB', ec=C_BOURDIN, alpha=0.8))

plt.tight_layout(); savefig(fig, 'fig04_tip_error_evolution')


# ═══════════════════════════════════════════════════════════════
#  FIG 05: M vs B — FIX: minimal annotations
# ═══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.0))

for sn, ls, alp, lw_v in [('grid_16', '-', 1.0, 1.5), ('grid_32', '--', 0.8, 1.2)]:
    axes[0].semilogy(em['step'], em[f'{sn}_H_global'].clip(lower=1e-15),
                     f'r{ls}', lw=lw_v, alpha=alp,
                     label=rf'Miehe, {sn.replace("_"," ")}')
    axes[0].semilogy(eb['step'], eb[f'{sn}_d_global'].clip(lower=1e-15),
                     f'b{ls}', lw=lw_v, alpha=alp,
                     label=rf'Bourdin, {sn.replace("_"," ")}')
axes[0].axvspan(N_ph1, N_ph1+N_ph2, alpha=0.06, color='red')
axes[0].set_ylim(bottom=1e-8)
axes[0].set_xlabel('Load step $n$'); axes[0].set_ylabel(r'$L^2$ error')
axes[0].set_title('(a) Carried-variable error')
axes[0].legend(fontsize=5.5, loc='lower right')
axes[0].grid(True, alpha=0.2, which='both')

# (b) Bar chart — clean, ratios inside bars
sl = list(tip_t['scheme'])
xp = np.arange(len(sl))
mt = tip_t['miehe_H_tip'].values
bt = tip_t['bourdin_d_tip'].values
ratios = tip_t['ratio'].values

axes[1].bar(xp-0.18, mt, 0.32, color=C_MIEHE, alpha=0.85, label=r'Miehe $\mathcal{H}$')
axes[1].bar(xp+0.18, bt, 0.32, color=C_BOURDIN, alpha=0.85, label='Bourdin $d$')
axes[1].set_yscale('log'); axes[1].set_xticks(xp)
axes[1].set_xticklabels([s.replace('_','\n') for s in sl], fontsize=6)
axes[1].set_ylabel('Tip error')
axes[1].set_title('(b) Tip error (SEN-T, cyclic)')
axes[1].legend(fontsize=6, loc='upper right')
axes[1].grid(True, alpha=0.2, axis='y', which='both')
# Ratios: inside bars, white
for i in range(len(sl)):
    axes[1].text(i-0.18, mt[i]*0.35, f'{ratios[i]:.0f}×', ha='center', va='center',
                 fontsize=6, color='white', fontweight='bold')

plt.tight_layout(); savefig(fig, 'fig05_miehe_vs_bourdin')


# ═══════════════════════════════════════════════════════════════
#  FIG 06: Spatial + d-profile
# ═══════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(DOUBLE_COL, 4.2))
gs6 = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.35, width_ratios=[1,1,1.2])
ix75 = int(0.75 * N)

d_exact_final = sp['d_exact_reload']
d_bexact_final = sp['d_bexact_final']
H_exact_nodal = sp['H_exact_nodal']

for row, sn in enumerate(['grid_32', 'grid_16']):
    labels_row = ['(a)','(b)','(c)'] if row==0 else ['(d)','(e)','(f)']

    He = np.abs(sp[f'H_miehe_{sn}'] - H_exact_nodal).reshape(N+1, N+1)
    ax = fig.add_subplot(gs6[row, 0])
    im = ax.pcolormesh(xs_plot, xs_plot, He, cmap='Reds', shading='auto')
    ax.set_title(f'{labels_row[0]} Miehe, {sn.replace("_"," ")}', fontsize=7.5)
    ax.set_aspect('equal'); ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
    plt.colorbar(im, ax=ax, shrink=0.65, pad=0.02)

    de = np.abs(sp[f'd_bourdin_{sn}'] - d_bexact_final).reshape(N+1, N+1)
    ax = fig.add_subplot(gs6[row, 1])
    im = ax.pcolormesh(xs_plot, xs_plot, de, cmap='Blues', shading='auto')
    ax.set_title(f'{labels_row[1]} Bourdin, {sn.replace("_"," ")}', fontsize=7.5)
    ax.set_aspect('equal'); ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
    plt.colorbar(im, ax=ax, shrink=0.65, pad=0.02)

    ax = fig.add_subplot(gs6[row, 2])
    ax.plot(xs_plot, d_exact_final.reshape(N+1,N+1)[:, ix75], 'k-', lw=2, label='Ref (Miehe)')
    ax.plot(xs_plot, sp[f'd_miehe_{sn}'].reshape(N+1,N+1)[:, ix75], '--', color=C_MIEHE, lw=1.5,
            label=f'Miehe + {sn.replace("_"," ")}')
    ax.plot(xs_plot, d_bexact_final.reshape(N+1,N+1)[:, ix75], 'k:', lw=2, label='Ref (Bourdin)')
    ax.plot(xs_plot, sp[f'd_bourdin_{sn}'].reshape(N+1,N+1)[:, ix75], '-.', color=C_BOURDIN, lw=1.5,
            label=f'Bourdin + {sn.replace("_"," ")}')
    ax.set_xlabel('$y$'); ax.set_ylabel(r'$d\,(x\!=\!0.75,\, y)$')
    ax.set_title(f'{labels_row[2]} Profile $\\perp$ crack', fontsize=7.5)
    ax.legend(fontsize=5, loc='upper right'); ax.grid(True, alpha=0.2)
savefig(fig, 'fig06_spatial_profile')


# ═══════════════════════════════════════════════════════════════
#  FIG 07: Fatigue — B/C ratio panel, no table
# ═══════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(DOUBLE_COL, 4.5))
gs7 = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.4)

ax_a = fig.add_subplot(gs7[0, :2])
ax_a.semilogy(fat['cycle'], fat['alpha_err_B'].clip(lower=1e-15), '-', color=C_MIEHE, lw=1.5,
              label=r'Standard $L^2(\bar{\alpha})$')
ax_a.semilogy(fat['cycle'], fat['alpha_err_C'].clip(lower=1e-15), '--', color=C_BOURDIN, lw=1.5,
              label=r'Compressed $L^2(\bar{\alpha})$')
ax_a.semilogy(fat['cycle'], fat['d_err_B'].clip(lower=1e-15), ':', color=C_MIEHE, lw=1, alpha=0.5,
              label=r'Standard $L^2(d)$')
ax_a.semilogy(fat['cycle'], fat['d_err_C'].clip(lower=1e-15), '-.', color=C_BOURDIN, lw=1, alpha=0.5,
              label=r'Compressed $L^2(d)$')
ax_a.set_xlabel('Cycle'); ax_a.set_ylabel('$L^2$ error')
ax_a.set_title('(a) Transfer error over 40 cycles')
ax_a.legend(fontsize=5.5, loc='lower right'); ax_a.grid(True, alpha=0.2, which='both')
axins = ax_a.inset_axes([0.03, 0.05, 0.25, 0.35])
axins.plot(fat['cycle'], fat['f_min'], '-', color=C_BOURDIN, lw=1.0)
axins.set_xlabel('Cycle', fontsize=5); axins.set_ylabel(r'$\min f$', fontsize=5)
axins.tick_params(labelsize=4.5); axins.set_title('Toughness', fontsize=5.5)
axins.grid(True, alpha=0.2)

ax_b = fig.add_subplot(gs7[0, 2])
bc = fat['alpha_err_B'] / fat['alpha_err_C'].clip(lower=1e-15)
ax_b.plot(fat['cycle'], bc, '-', color='#8E44AD', lw=1.5)
ax_b.axhline(1.0, color='gray', ls=':', lw=0.6, alpha=0.5)
ax_b.set_xlabel('Cycle'); ax_b.set_ylabel('B/C error ratio')
ax_b.set_title('(b) Compression benefit')
ax_b.grid(True, alpha=0.2); ax_b.set_ylim(0.8, 3.5)
for nc in [10, 20, 40]:
    idx = (fat['cycle'] - nc).abs().idxmin()
    r = bc.iloc[idx]
    ax_b.plot(nc, r, 'o', color='#8E44AD', ms=4, zorder=5)
    ax_b.annotate(f'{r:.1f}×', xy=(nc, r), xytext=(nc+2, r+0.15),
                  fontsize=6, color='#8E44AD', fontweight='bold')

x_1d = sp['fat_x1d']; mask = (x_1d > 0.3) & (x_1d < 0.7)
for col, (da, db, dc, yl, ttl) in enumerate([
    (sp['fat_dA'], sp['fat_dB'], sp['fat_dC'], '$d$', '(c) Phase field'),
    (sp['fat_alphaA'], sp['fat_alphaB'], sp['fat_alphaC'], r'$\bar{\alpha}$', '(d) Fatigue variable'),
    (f_fat(sp['fat_alphaA']), f_fat(sp['fat_alphaB']), f_fat(sp['fat_alphaC']),
     r'$f(\bar{\alpha})$', '(e) Toughness')]):
    ax = fig.add_subplot(gs7[1, col])
    ax.plot(x_1d[mask], da[mask], '-', color='k', lw=1.5, label='Exact')
    ax.plot(x_1d[mask], db[mask], '--', color=C_MIEHE, lw=1.2, label='Standard')
    ax.plot(x_1d[mask], dc[mask], ':', color=C_BOURDIN, lw=1.2, label='Compressed')
    ax.set_xlabel('$x$'); ax.set_ylabel(yl); ax.set_title(ttl, fontsize=8)
    ax.legend(fontsize=5.5); ax.grid(True, alpha=0.2)
savefig(fig, 'fig07_fatigue')


# ═══════════════════════════════════════════════════════════════
#  FIG 08: Convergence — FIX: add tip panel + shear overlay
# ═══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.2))

hs = ct['h_s_over_ell'].values
for ax, mcol, bcol, ttl in [
    (axes[0], 'miehe_H_global', 'bourdin_d_global', '(a) Global convergence'),
    (axes[1], 'miehe_H_tip', 'bourdin_d_tip', '(b) Crack-tip convergence')]:
    em_v = ct[mcol].values; eb_v = ct[bcol].values
    sl_m, _ = np.polyfit(np.log10(hs), np.log10(em_v), 1)
    sl_b, _ = np.polyfit(np.log10(hs), np.log10(eb_v), 1)
    ax.loglog(hs, em_v, '-o', color=C_MIEHE, ms=5, lw=1.8, label=f'SEN-T Miehe ({sl_m:.2f})')
    ax.loglog(hs, eb_v, '-s', color=C_BOURDIN, ms=5, lw=1.8, label=f'SEN-T Bourdin ({sl_b:.2f})')
    # Add SEN-S overlay for global panel
    if 'global' in mcol:
        hs_s = cs['h_s_over_ell'].values
        em_s = cs['miehe_H_global'].values; eb_s = cs['bourdin_d_global'].values
        sl_ms, _ = np.polyfit(np.log10(hs_s), np.log10(em_s), 1)
        sl_bs, _ = np.polyfit(np.log10(hs_s), np.log10(eb_s), 1)
        ax.loglog(hs_s, em_s, '--o', color=C_MIEHE, ms=3, lw=0.8, alpha=0.4,
                  label=f'SEN-S Miehe ({sl_ms:.2f})')
        ax.loglog(hs_s, eb_s, '--s', color=C_BOURDIN, ms=3, lw=0.8, alpha=0.4,
                  label=f'SEN-S Bourdin ({sl_bs:.2f})')
    hs_ref = np.array([hs.min()*0.7, hs.max()*1.3])
    c1 = eb_v[-1]/hs[-1]
    ax.loglog(hs_ref, c1*hs_ref, 'k--', lw=0.8, alpha=0.3, label=r'$O(h_s/\ell)$')
    ax.set_xlabel(r'$h_s / \ell$'); ax.set_ylabel(r'$L^2$ error')
    ax.set_title(ttl); ax.legend(fontsize=5.5, loc='lower right')
    ax.grid(True, alpha=0.2, which='both')
plt.tight_layout(); savefig(fig, 'fig08_convergence_rate')


# ═══════════════════════════════════════════════════════════════
#  FIG 09: Regime comparison — FIX: minimal annotations
# ═══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.0))
sl3 = ['grid_8','grid_16','grid_32']
xp = np.arange(3)

# (a) SEN-T
mt_t = [tip_t[tip_t['scheme']==s]['miehe_H_tip'].values[0] for s in sl3]
bt_t = [tip_t[tip_t['scheme']==s]['bourdin_d_tip'].values[0] for s in sl3]
axes[0].bar(xp-0.18, mt_t, 0.32, color=C_MIEHE, alpha=0.85, label=r'Miehe $\mathcal{H}$')
axes[0].bar(xp+0.18, bt_t, 0.32, color=C_BOURDIN, alpha=0.85, label='Bourdin $d$')
axes[0].set_yscale('log'); axes[0].set_xticks(xp)
axes[0].set_xticklabels(['8','16','32'], fontsize=8)
axes[0].set_xlabel('$N_s$'); axes[0].set_ylabel('Tip error')
axes[0].set_title('(a) SEN-T cyclic: M $\\gg$ B', fontsize=8)
axes[0].legend(fontsize=6); axes[0].grid(True, alpha=0.2, axis='y', which='both')

# (b) SEN-S
mt_s = [tip_s[tip_s['scheme']==s]['miehe_H_tip'].values[0] for s in sl3]
bt_s = [tip_s[tip_s['scheme']==s]['bourdin_d_tip'].values[0] for s in sl3]
axes[1].bar(xp-0.18, mt_s, 0.32, color=C_MIEHE, alpha=0.85, label=r'Miehe $\mathcal{H}$')
axes[1].bar(xp+0.18, bt_s, 0.32, color=C_BOURDIN, alpha=0.85, label='Bourdin $d$')
axes[1].set_yscale('log'); axes[1].set_xticks(xp)
axes[1].set_xticklabels(['8','16','32'], fontsize=8)
axes[1].set_xlabel('$N_s$'); axes[1].set_ylabel('Tip error')
axes[1].set_title('(b) SEN-S monotonic: M $\\ll$ B', fontsize=8)
axes[1].legend(fontsize=6); axes[1].grid(True, alpha=0.2, axis='y', which='both')
axes[1].text(0.97, 0.03, 'ratio inverted', transform=axes[1].transAxes,
             fontsize=6.5, ha='right', va='bottom', color=C_BOURDIN, fontweight='bold',
             bbox=dict(boxstyle='round', fc='#EBF5FB', ec=C_BOURDIN, alpha=0.8))

plt.tight_layout(); savefig(fig, 'fig09_regime_comparison')


# ═══════════════════════════════════════════════════════════════
#  FIG 10: Load-Displacement (P-δ) curves — NEW
#  Shows: (a) SEN-T ref vs transfer, (b) SEN-S ref, (c) force error
# ═══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.8))

u_mm = ld['u_bar'].values * 1e3  # mm → ×10^-3

# (a) SEN-T: reference + transferred (Miehe)
axes[0].plot(u_mm, ld['F_ref_miehe'], 'k-', lw=2, label='Ref (no transfer)')
for sn, col, ls in [('grid_8', C_GRID8, '-'), ('grid_16', C_GRID16, '--'),
                     ('grid_32', C_GRID32, '-.'), ('rbf_200', C_RBF200, ':')]:
    axes[0].plot(u_mm, ld[f'F_miehe_{sn}'], ls, color=col, lw=1,
                 label=f'Miehe + {sn.replace("_"," ")}')
axes[0].set_xlabel(r'$\bar{u}$ [$\times 10^{-3}$ mm]')
axes[0].set_ylabel('Reaction force $F$')
axes[0].set_title('(a) SEN-T, Miehe')
axes[0].legend(fontsize=5, loc='upper left'); axes[0].grid(True, alpha=0.2)

# (b) SEN-T: Miehe ref vs Bourdin ref (no transfer)
axes[1].plot(u_mm, ld['F_ref_miehe'], '-', color=C_MIEHE, lw=1.8, label='Miehe ref')
axes[1].plot(u_mm, ld['F_ref_bourdin'], '-', color=C_BOURDIN, lw=1.8, label='Bourdin ref')
# Also show Bourdin + grid_16 to prove minimal P-δ impact
axes[1].plot(u_mm, ld[f'F_bourdin_grid_16'], '--', color=C_BOURDIN, lw=1,
             alpha=0.7, label='Bourdin + grid 16')
axes[1].set_xlabel(r'$\bar{u}$ [$\times 10^{-3}$ mm]')
axes[1].set_ylabel('Reaction force $F$')
axes[1].set_title('(b) Miehe vs Bourdin ref')
axes[1].legend(fontsize=5, loc='upper left'); axes[1].grid(True, alpha=0.2)

# (c) Relative force error (Miehe vs Bourdin, grid_16)
F_ref = ld['F_ref_miehe'].values
F_m16 = ld['F_miehe_grid_16'].values
F_b16 = ld['F_bourdin_grid_16'].values
# Relative error w.r.t. respective references
F_ref_m = ld['F_ref_miehe'].values
F_ref_b = ld['F_ref_bourdin'].values
eps_force = 1e-10
rel_m = np.abs(F_m16 - F_ref_m) / (np.abs(F_ref_m) + eps_force)
rel_b = np.abs(F_b16 - F_ref_b) / (np.abs(F_ref_b) + eps_force)
axes[2].semilogy(np.arange(1, N_steps+1), rel_m, '-', color=C_MIEHE, lw=1.2,
                 label='Miehe, grid 16')
axes[2].semilogy(np.arange(1, N_steps+1), rel_b, '-', color=C_BOURDIN, lw=1.2,
                 label='Bourdin, grid 16')
axes[2].axvspan(N_ph1, N_ph1+N_ph2, alpha=0.06, color='red')
axes[2].set_xlabel('Load step $n$')
axes[2].set_ylabel('Relative force error')
axes[2].set_title('(c) $|F - F_{\\mathrm{ref}}|/|F_{\\mathrm{ref}}|$')
axes[2].legend(fontsize=5.5, loc='upper left'); axes[2].grid(True, alpha=0.2, which='both')

plt.tight_layout(); savefig(fig, 'fig10_load_displacement')


print(f"\n{'='*70}")
print("✅ ALL 10 FIGURES GENERATED (<5 sec)")
print("="*70)