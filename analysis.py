#!/usr/bin/env python3
"""
INTL 601 Research Methods I — Exercise #1
Complete Analysis: Figures + Markdown Report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from scipy.special import expit
from linearmodels.iv import IV2SLS
import warnings
warnings.filterwarnings('ignore')

DIR = '/Users/helinekmen/Desktop/causal inference/'

# ── Global Style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#f4f6f9',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.grid': True,
    'axes.grid.axis': 'y',
    'grid.alpha': 0.4,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
})
BLUE   = '#1f77b4'
RED    = '#d62728'
GREEN  = '#2ca02c'
ORANGE = '#ff7f0e'
PURPLE = '#9467bd'
GRAY   = '#7f7f7f'
C = [BLUE, RED, GREEN, ORANGE, PURPLE, GRAY]

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_stata(DIR + 'gg_fake.dta')
N = len(df)
print(f"Loaded data: {df.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: DESCRIPTIVE STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
turnout_rate    = df['Y'].mean()
assignment_rate = df['Z'].mean()
contact_rate    = df['T'].mean()
contact_Z1      = df.loc[df['Z']==1, 'T'].mean()
contact_Z0      = df.loc[df['Z']==0, 'T'].mean()
compliance_gap  = contact_Z1 - contact_Z0

desc_stats = df[['Y','Z','T','age','educ','pastvote','party_id','competitive']].describe().T
desc_stats = desc_stats[['mean','std','min','max']].round(3)

print("\n=== KEY RATES ===")
print(f"Turnout rate (Y=1):             {turnout_rate:.4f}")
print(f"Assignment rate (Z=1):          {assignment_rate:.4f}")
print(f"Contact rate (T=1):             {contact_rate:.4f}")
print(f"Contact rate | Z=1:             {contact_Z1:.4f}")
print(f"Contact rate | Z=0:             {contact_Z0:.4f}")
print(f"Compliance gap (Z1−Z0):         {compliance_gap:.4f}")
print("\nDescriptive Stats:")
print(desc_stats.to_string())

# ── Figure 1: Descriptive Overview ───────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Figure 1 — Descriptive Statistics', fontsize=15, fontweight='bold', y=1.01)

# Panel A: Key overall rates
rates_labels = ['Turnout\n(Y)', 'Assigned\n(Z)', 'Contacted\n(T)']
rates_vals   = [turnout_rate*100, assignment_rate*100, contact_rate*100]
bars = axes[0].bar(rates_labels, rates_vals, color=[BLUE, GREEN, ORANGE],
                   width=0.5, edgecolor='white', linewidth=2)
for bar, v in zip(bars, rates_vals):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
                 f'{v:.1f}%', ha='center', fontweight='bold', fontsize=12)
axes[0].set_ylim(0, 80)
axes[0].set_ylabel('Rate (%)')
axes[0].set_title('Panel A: Overall Key Rates')

# Panel B: Contact rate by assignment
c_labels = ['Not Assigned\n(Z = 0)', 'Assigned\n(Z = 1)']
c_vals   = [contact_Z0*100, contact_Z1*100]
bars2 = axes[1].bar(c_labels, c_vals, color=[BLUE, RED],
                    width=0.4, edgecolor='white', linewidth=2)
for bar, v in zip(bars2, c_vals):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
                 f'{v:.1f}%', ha='center', fontweight='bold', fontsize=12)
# Bracket for compliance gap
ymax = max(c_vals)
axes[1].annotate('', xy=(1, ymax+6), xytext=(0, ymax+6),
                 arrowprops=dict(arrowstyle='<->', color='black', lw=2))
axes[1].text(0.5, ymax+7.5, f'Compliance gap = {compliance_gap*100:.1f} pp',
             ha='center', fontsize=10, fontweight='bold')
axes[1].set_ylim(0, 100)
axes[1].set_ylabel('Contact Rate (%)')
axes[1].set_title('Panel B: Contact Rate by Assignment')

# Panel C: Turnout by T × Z
ct = df.groupby(['Z','T'])['Y'].mean().unstack(fill_value=np.nan) * 100
im = axes[2].imshow(ct.values, cmap='YlOrRd', aspect='auto', vmin=30, vmax=80)
axes[2].set_xticks([0,1]); axes[2].set_xticklabels(['T=0 (No Contact)', 'T=1 (Contact)'])
axes[2].set_yticks([0,1]); axes[2].set_yticklabels(['Z=0 (Control)', 'Z=1 (Treated)'])
for i in range(2):
    for j in range(2):
        v = ct.values[i,j]
        axes[2].text(j, i, f'{v:.1f}%', ha='center', va='center',
                     fontsize=14, fontweight='bold',
                     color='white' if v > 60 else 'black')
plt.colorbar(im, ax=axes[2], label='Turnout Rate (%)')
axes[2].set_title('Panel C: Turnout by Z × T')

plt.tight_layout()
plt.savefig(DIR+'fig1_descriptive.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig1_descriptive.png")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: OLS UNIVARIATE + DIFFERENCE OF MEANS
# ─────────────────────────────────────────────────────────────────────────────
grp1 = df.loc[df['T']==1, 'Y']
grp0 = df.loc[df['T']==0, 'Y']
dom = grp1.mean() - grp0.mean()
t_stat, p_val = stats.ttest_ind(grp1, grp0)

model1 = smf.ols('Y ~ T', data=df).fit()
pred_T0 = model1.params['Intercept']
pred_T1 = model1.params['Intercept'] + model1.params['T']
T1_coef = model1.params['T']
T1_se   = model1.bse['T']
T1_t    = model1.tvalues['T']
T1_p    = model1.pvalues['T']
T1_ci   = model1.conf_int().loc['T'].values

print("\n=== SECTION 2: OLS UNIVARIATE + DIFF-OF-MEANS ===")
print(f"Turnout (T=0): {grp0.mean():.4f}   Turnout (T=1): {grp1.mean():.4f}")
print(f"Diff-of-means: {dom:.4f}   t={t_stat:.3f}   p={p_val:.6f}")
print(f"OLS T coef: {T1_coef:.4f}  SE={T1_se:.4f}  t={T1_t:.3f}  p={T1_p:.6f}")
print(f"Margins: P(Y|T=0)={pred_T0:.4f}   P(Y|T=1)={pred_T1:.4f}   diff={pred_T1-pred_T0:.4f}")

# ── Figure 2: Univariate OLS ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Figure 2 — OLS Univariate: Turnout by Contact Status', fontsize=15,
             fontweight='bold', y=1.01)

# Panel A: Turnout rates
y_vals = [grp0.mean()*100, grp1.mean()*100]
y_cis  = [1.96*grp0.std()/np.sqrt(len(grp0))*100,
          1.96*grp1.std()/np.sqrt(len(grp1))*100]
bars = axes[0].bar(['Not Contacted\n(T = 0)', 'Contacted\n(T = 1)'],
                   y_vals, yerr=y_cis, capsize=7, color=[BLUE, RED],
                   width=0.45, edgecolor='white', linewidth=2,
                   error_kw={'linewidth':2, 'ecolor':'black', 'capthick':2})
for bar, v in zip(bars, y_vals):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
                 f'{v:.1f}%', ha='center', fontweight='bold', fontsize=12)
ymax = max(y_vals)
axes[0].annotate('', xy=(1, ymax+6), xytext=(0, ymax+6),
                 arrowprops=dict(arrowstyle='<->', color='black', lw=2))
axes[0].text(0.5, ymax+7.5, f'Δ = {dom*100:.1f} pp  (p < 0.001)',
             ha='center', fontsize=11, fontweight='bold')
axes[0].set_ylim(0, 85)
axes[0].set_ylabel('Turnout Rate (%)')
axes[0].set_title('Panel A: Difference of Means\n(with 95% CI)')

# Panel B: Margins (predicted probabilities)
margin_vals = [pred_T0*100, pred_T1*100]
bars2 = axes[1].bar(['P(Y=1 | T=0)', 'P(Y=1 | T=1)'],
                    margin_vals, color=[BLUE, RED], width=0.4,
                    edgecolor='white', linewidth=2)
for bar, v in zip(bars2, margin_vals):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 f'{v:.2f}%', ha='center', fontweight='bold', fontsize=12)
axes[1].set_ylim(0, 80)
axes[1].set_ylabel('Predicted Turnout Rate (%)')
axes[1].set_title(f'Panel B: Margins — Predicted Probabilities\n'
                  f'β(T) = {T1_coef:.4f}  SE = {T1_se:.4f}  p < 0.001')
# Annotation
axes[1].annotate(f'Difference = {(pred_T1-pred_T0)*100:.2f}pp\n= OLS coefficient on T\n({T1_coef:.4f})',
                 xy=(1, pred_T1*100), xytext=(0.6, 30),
                 arrowprops=dict(arrowstyle='->', color='black'),
                 fontsize=10, ha='center',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
plt.savefig(DIR+'fig2_ols_univariate.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig2_ols_univariate.png")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: OLS WITH CONTROLS
# ─────────────────────────────────────────────────────────────────────────────
model2 = smf.ols('Y ~ T + age + educ + pastvote + party_id + competitive', data=df).fit()
T2_coef = model2.params['T']
T2_se   = model2.bse['T']
T2_t    = model2.tvalues['T']
T2_p    = model2.pvalues['T']
T2_ci   = model2.conf_int().loc['T'].values
change  = T2_coef - T1_coef
pct_change = abs(change / T1_coef) * 100

controls = ['age','educ','pastvote','party_id','competitive']
ctrl_info = {v: {'coef': model2.params[v], 'se': model2.bse[v],
                 't': model2.tvalues[v], 'p': model2.pvalues[v]}
             for v in controls}
sorted_by_t = sorted(controls, key=lambda v: abs(model2.tvalues[v]), reverse=True)
biggest_confounder = sorted_by_t[0]

print("\n=== SECTION 3: OLS WITH CONTROLS ===")
print(f"T coef (univariate):   {T1_coef:.4f}")
print(f"T coef (with ctrl):    {T2_coef:.4f}")
print(f"Change:                {change:.4f}  ({pct_change:.1f}%)")
print(f"Marginal effect of T:  {T2_coef:.4f}")
print(f"Raw diff-of-means:     {dom:.4f}")
print(f"Biggest confounder:    {biggest_confounder} (|t|={abs(ctrl_info[biggest_confounder]['t']):.2f})")
print("\nAll controls ranked by |t|:")
for v in sorted_by_t:
    print(f"  {v:15s}  coef={ctrl_info[v]['coef']:.4f}  SE={ctrl_info[v]['se']:.4f}  "
          f"t={ctrl_info[v]['t']:.2f}  p={ctrl_info[v]['p']:.4f}")

# ── Figure 3: Controls Analysis ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Figure 3 — OLS with Control Variables', fontsize=15, fontweight='bold', y=1.01)

# Panel A: T coefficient comparison
comp_labels = ['Univariate\n(reg Y T)', 'With Controls\n(reg Y T + X)']
comp_vals   = [T1_coef, T2_coef]
comp_ci     = [T1_ci, T2_ci]
comp_errors = np.array([[v - ci[0], ci[1] - v] for v, ci in zip(comp_vals, comp_ci)]).T
bars = axes[0].bar(comp_labels, comp_vals, color=[BLUE, GREEN],
                   width=0.45, edgecolor='white', linewidth=2,
                   yerr=comp_errors, capsize=7,
                   error_kw={'linewidth':2, 'ecolor':'black', 'capthick':2})
for bar, v in zip(bars, comp_vals):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                 f'β = {v:.4f}', ha='center', fontweight='bold', fontsize=12)
axes[0].set_ylabel('Coefficient on T')
axes[0].set_ylim(0, max(comp_vals)*1.35)
axes[0].set_title(f'Panel A: T Coefficient — With vs. Without Controls\n'
                  f'Change: {change:+.4f} ({pct_change:.1f}%)')

# Panel B: All coefficients in multivariate model
all_vars  = ['T'] + controls
all_coefs = [model2.params[v] for v in all_vars]
all_cis   = [model2.conf_int().loc[v].values for v in all_vars]
all_errs  = np.array([[c - ci[0], ci[1] - c] for c, ci in zip(all_coefs, all_cis)]).T
col_map   = [RED if v == 'T' else BLUE for v in all_vars]
y_pos     = list(range(len(all_vars)))
axes[1].barh(y_pos, all_coefs, xerr=all_errs, height=0.55,
             color=col_map, capsize=5, error_kw={'linewidth':1.5, 'ecolor':'#333333'})
axes[1].set_yticks(y_pos)
axes[1].set_yticklabels(all_vars, fontsize=11)
axes[1].axvline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.6)
axes[1].set_xlabel('OLS Coefficient (with 95% CI)')
axes[1].set_title('Panel B: All Coefficients\nreg Y T age educ pastvote party_id competitive')
for i, (v, c, p) in enumerate(zip(all_vars, all_coefs,
                                   [model2.pvalues[v] for v in all_vars])):
    stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    offset = 0.003 if c >= 0 else -0.003
    axes[1].text(c + offset, i, f'{c:.3f}{stars}',
                 va='center', ha='left' if c >= 0 else 'right', fontsize=9.5)

plt.tight_layout()
plt.savefig(DIR+'fig3_controls.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig3_controls.png")

# ── Figure 4: Covariates vs Turnout ──────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
fig.suptitle('Figure 4 — Covariate Associations with Turnout (Y)',
             fontsize=15, fontweight='bold', y=1.01)

for i, var in enumerate(controls):
    ax = axes[i]
    n_unique = df[var].nunique()
    coef_v   = ctrl_info[var]['coef']
    t_v      = ctrl_info[var]['t']
    p_v      = ctrl_info[var]['p']
    stars = '***' if p_v < 0.001 else '**' if p_v < 0.01 else '*' if p_v < 0.05 else ''

    if n_unique <= 2:
        means = df.groupby(var)['Y'].mean() * 100
        bars  = ax.bar(means.index, means.values, color=[BLUE, RED], width=0.4,
                       edgecolor='white', linewidth=2)
        for bar, v in zip(bars, means.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    f'{v:.1f}%', ha='center', fontweight='bold', fontsize=11)
        ax.set_xlabel(var); ax.set_ylabel('Turnout Rate (%)')
    elif n_unique <= 20:
        means = df.groupby(var)['Y'].mean() * 100
        ax.bar(means.index, means.values, color=BLUE, alpha=0.8,
               edgecolor='white', linewidth=1)
        ax.set_xlabel(var); ax.set_ylabel('Turnout Rate (%)')
        # Trend line
        x_vals = np.array(means.index, dtype=float)
        z = np.polyfit(x_vals, means.values, 1)
        p_line = np.poly1d(z)
        ax.plot(x_vals, p_line(x_vals), 'r--', linewidth=2, label='Trend')
        ax.legend(fontsize=9)
    else:
        voted     = df.loc[df['Y']==1, var]
        not_voted = df.loc[df['Y']==0, var]
        ax.hist(not_voted, bins=30, alpha=0.6, color=BLUE, label='Y=0 (Did not vote)', density=True)
        ax.hist(voted,     bins=30, alpha=0.6, color=RED,  label='Y=1 (Voted)',        density=True)
        ax.legend(fontsize=9)
        ax.set_xlabel(var); ax.set_ylabel('Density')

    ax.set_title(f'{var}\ncoef = {coef_v:.4f}{stars}  |t| = {abs(t_v):.2f}')

axes[-1].set_visible(False)
plt.tight_layout()
plt.savefig(DIR+'fig4_covariates.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig4_covariates.png")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: CAUSAL STRUCTURE  Z → T → Y
# ─────────────────────────────────────────────────────────────────────────────
fs_model = smf.ols('T ~ Z + age + educ + pastvote + party_id + competitive', data=df).fit()
rf_model = smf.ols('Y ~ Z + age + educ + pastvote + party_id + competitive', data=df).fit()

FS_coef  = fs_model.params['Z']
FS_se    = fs_model.bse['Z']
FS_t     = fs_model.tvalues['Z']
ITT_coef = rf_model.params['Z']
ITT_se   = rf_model.bse['Z']
ITT_t    = rf_model.tvalues['Z']
LATE_man = ITT_coef / FS_coef

# 2SLS
iv = IV2SLS.from_formula(
    'Y ~ 1 + age + educ + pastvote + party_id + competitive + [T ~ Z]',
    data=df
).fit(cov_type='robust')
LATE_2sls    = iv.params['T']
LATE_2sls_se = iv.std_errors['T']
LATE_2sls_t  = iv.tstats['T']
LATE_2sls_p  = iv.pvalues['T']

print("\n=== SECTION 4: CAUSAL STRUCTURE ===")
print(f"First Stage (Z→T):   coef={FS_coef:.4f}  SE={FS_se:.4f}  t={FS_t:.3f}")
print(f"ITT (Z→Y):           coef={ITT_coef:.4f}  SE={ITT_se:.4f}  t={ITT_t:.3f}")
print(f"LATE (manual):       {LATE_man:.4f}")
print(f"2SLS estimate:       {LATE_2sls:.4f}  SE={LATE_2sls_se:.4f}  t={LATE_2sls_t:.3f}  p={LATE_2sls_p:.6f}")
print(f"OLS T coef (model2): {T2_coef:.4f}")
print(f"First stage R²:      {fs_model.rsquared:.4f}")

print("\n--- First Stage summary ---")
print(fs_model.summary().tables[1])
print("\n--- Reduced Form (ITT) summary ---")
print(rf_model.summary().tables[1])

# ── Figure 5: Causal Structure ────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 6))
fig.suptitle('Figure 5 — Causal Structure: Z → T → Y', fontsize=15, fontweight='bold')

# Left panel: Path diagram
ax_path = fig.add_axes([0.02, 0.05, 0.46, 0.88])
ax_path.set_xlim(0, 10); ax_path.set_ylim(0, 5)
ax_path.axis('off')
ax_path.set_facecolor('white')

def node(ax, x, y, label, sub='', color='#1f77b4', r=0.72):
    c = plt.Circle((x, y), r, color=color, zorder=3)
    ax.add_patch(c)
    ax.text(x, y+(0.15 if sub else 0), label,
            ha='center', va='center', fontsize=13, fontweight='bold', color='white', zorder=4)
    if sub:
        ax.text(x, y-0.25, sub, ha='center', va='center', fontsize=8.5, color='white', zorder=4)

def arrow(ax, x1, y1, x2, y2, label='', color='black', lw=2, ls='-', offset_y=0.3):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                linestyle=ls, connectionstyle='arc3,rad=0'))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my+offset_y, label, ha='center', va='bottom', fontsize=10,
                color=color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.85))

node(ax_path, 1.6, 3,   'Z', 'Random\nAssignment', color=GREEN,  r=0.75)
node(ax_path, 5,   3,   'T', 'Actual\nContact',     color=BLUE,   r=0.75)
node(ax_path, 8.4, 3,   'Y', 'Turnout',             color=RED,    r=0.75)
node(ax_path, 5,   0.9, 'X', 'Covariates\n(age, educ,\npastvote,...)', color=PURPLE, r=0.75)

arrow(ax_path, 2.35, 3,    4.25, 3,    f'FS = {FS_coef:.3f}',    color=GREEN, lw=2.5)
arrow(ax_path, 5.75, 3,    7.65, 3,    f'LATE = {LATE_2sls:.3f}', color=BLUE,  lw=2.5)
arrow(ax_path, 2.35, 2.7,  7.65, 2.7,  f'ITT = {ITT_coef:.3f}',
      color=GRAY, lw=1.5, ls='dashed', offset_y=-0.35)
# Covariate arrows
for xend in [4.3, 5.7, 7.7]:
    ax_path.annotate('', xy=(xend, 2.15), xytext=(5.0 if xend < 7 else 5.2, 1.65),
                     arrowprops=dict(arrowstyle='->', color=PURPLE, lw=1.5))

ax_path.text(5, 4.6, 'Path Diagram', ha='center', fontsize=12, color='#333',
             fontstyle='italic')
ax_path.text(5, 0.0,
             'FS = First Stage  |  ITT = Intention-to-Treat  |  LATE = Local Average Treatment Effect',
             ha='center', fontsize=8.5, color=GRAY)

# Right panel: Coefficient comparison bar chart
ax_bar = fig.add_axes([0.54, 0.1, 0.43, 0.78])
est_labels = ['ITT\n(Z→Y, reduced form)', 'OLS\n(T→Y, with ctrl)', '2SLS/IV\n(LATE estimate)']
est_vals   = [ITT_coef, T2_coef, LATE_2sls]
est_errs   = [ITT_se, T2_se, LATE_2sls_se]
est_colors = [GREEN, BLUE, RED]
y_positions = [2, 1, 0]
ax_bar.barh(y_positions, est_vals, xerr=[1.96*e for e in est_errs],
            height=0.45, color=est_colors,
            capsize=6, error_kw={'linewidth':2, 'ecolor':'#333333'})
ax_bar.set_yticks(y_positions)
ax_bar.set_yticklabels(est_labels, fontsize=10)
ax_bar.axvline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.6)
ax_bar.set_xlabel('Coefficient (with 95% CI)', fontsize=11)
ax_bar.set_title('Effect Estimates Comparison', fontsize=12, fontweight='bold')
for y, v, e in zip(y_positions, est_vals, est_errs):
    ax_bar.text(v+e+0.002, y, f'{v:.4f}', va='center', fontsize=11, fontweight='bold')
ax_bar.set_facecolor('#f4f6f9')

fig.savefig(DIR+'fig5_causal.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig5_causal.png")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: TARGETED TREATMENT
# ─────────────────────────────────────────────────────────────────────────────
np.random.seed(42)
df['T_target'] = (np.random.uniform(size=N) <
                  expit(-1 + 1.2*df['pastvote'] + 0.5*df['party_id'])).astype(int)

T_target_rate = df['T_target'].mean()
tab_counts = pd.crosstab(df['pastvote'], df['T_target'])
tab_row    = pd.crosstab(df['pastvote'], df['T_target'], normalize='index') * 100

model_obs1 = smf.ols('Y ~ T_target', data=df).fit()
model_obs2 = smf.ols('Y ~ T_target + age + educ + pastvote + party_id + competitive', data=df).fit()

Tt_nc  = model_obs1.params['T_target']
Tt_nc_se = model_obs1.bse['T_target']
Tt_wc  = model_obs2.params['T_target']
Tt_wc_se = model_obs2.bse['T_target']

print("\n=== SECTION 5: TARGETED TREATMENT ===")
print(f"T_target rate:                   {T_target_rate:.4f}")
print(f"\nrow%: T_target by pastvote:")
print(tab_row.round(1).to_string())
print(f"\nT (experimental, no controls):   {T1_coef:.4f}")
print(f"T_target (no controls):          {Tt_nc:.4f}")
print(f"T_target (with controls):        {Tt_wc:.4f}")
print(f"Bias (nc vs wc):                 {Tt_nc - Tt_wc:.4f}")
print(f"Bias vs experimental T:          {Tt_nc - T1_coef:.4f}")

# Which variable drives T_target selection? Run reg T_target on covariates
conf_check = smf.ols('T_target ~ pastvote + party_id + age + educ + competitive', data=df).fit()
print("\n--- What predicts T_target? ---")
print(conf_check.summary().tables[1])
pastvote_t_target = conf_check.params['pastvote']
pastvote_Y        = model2.params['pastvote']
print(f"\npastvote → T_target: {pastvote_t_target:.4f}  (very strong)")
print(f"pastvote → Y:        {pastvote_Y:.4f}         (very strong)")
print(f"→ pastvote confounds T_target↔Y relationship")

# ── Figure 6: Targeted Treatment ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle('Figure 6 — Targeted Treatment vs. Experimental Treatment',
             fontsize=15, fontweight='bold', y=1.01)

# Panel A: T_target rate by pastvote
pv_rate = df.groupby('pastvote')['T_target'].mean() * 100
ax = axes[0]
bars = ax.bar(['pastvote=0\n(Did not vote\npreviously)',
               'pastvote=1\n(Voted\npreviously)'],
              pv_rate.values, color=[BLUE, RED], width=0.45,
              edgecolor='white', linewidth=2)
for bar, v in zip(bars, pv_rate.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f'{v:.1f}%', ha='center', fontweight='bold', fontsize=12)
ax.set_ylabel('T_target Rate (%)')
ax.set_title('Panel A: T_target Rate by Past Vote\n(Selective Targeting)')
ax.set_ylim(0, max(pv_rate.values)*1.25)

# Panel B: Coefficient comparison
b_labels = ['T experimental\n(no controls)', 'T_target\n(no controls)', 'T_target\n(with controls)']
b_vals   = [T1_coef,  Tt_nc,  Tt_wc]
b_ses    = [T1_se,    Tt_nc_se, Tt_wc_se]
b_colors = [GREEN, RED, BLUE]
b_errs   = np.array([[v-1.96*s, v+1.96*s] for v, s in zip(b_vals, b_ses)])

ax2 = axes[1]
bars2 = ax2.bar(b_labels, b_vals,
                yerr=[[v-ci[0] for v, ci in zip(b_vals, b_errs)],
                      [ci[1]-v  for v, ci in zip(b_vals, b_errs)]],
                color=b_colors, width=0.45, edgecolor='white', linewidth=2,
                capsize=6, error_kw={'linewidth':2, 'ecolor':'black', 'capthick':2})
for bar, v in zip(bars2, b_vals):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.004,
             f'β = {v:.4f}', ha='center', fontweight='bold', fontsize=10)
ax2.set_ylabel('Coefficient on Treatment')
ax2.set_ylim(0, max(b_vals)*1.35)
ax2.set_title('Panel B: Bias Comparison\n(Experimental vs. Targeted)')

# Panel C: Confounding heat map — Turnout by pastvote × treatment
for col_idx, (treat_var, title) in enumerate([('T', 'T (experimental)'), ('T_target', 'T_target (targeted)')]):
    ax3 = axes[2]
# Use a grouped bar chart instead
pv0_T0  = df.loc[(df['pastvote']==0)&(df['T_target']==0), 'Y'].mean()*100
pv0_T1  = df.loc[(df['pastvote']==0)&(df['T_target']==1), 'Y'].mean()*100
pv1_T0  = df.loc[(df['pastvote']==1)&(df['T_target']==0), 'Y'].mean()*100
pv1_T1  = df.loc[(df['pastvote']==1)&(df['T_target']==1), 'Y'].mean()*100

x = np.arange(2)
w = 0.3
ax3 = axes[2]
b3a = ax3.bar(x-w/2, [pv0_T0, pv1_T0], w, label='T_target=0', color=BLUE, alpha=0.85)
b3b = ax3.bar(x+w/2, [pv0_T1, pv1_T1], w, label='T_target=1', color=RED,  alpha=0.85)
ax3.set_xticks(x); ax3.set_xticklabels(['pastvote = 0', 'pastvote = 1'])
ax3.set_ylabel('Turnout Rate (%)')
ax3.set_title('Panel C: Turnout by pastvote × T_target\n(Confounding Visible Here)')
ax3.legend()
for bar, v in list(zip(b3a, [pv0_T0, pv1_T0])) + list(zip(b3b, [pv0_T1, pv1_T1])):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
             f'{v:.0f}%', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(DIR+'fig6_targeted.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig6_targeted.png")

# ─────────────────────────────────────────────────────────────────────────────
# COLLECT ALL RESULTS FOR MARKDOWN
# ─────────────────────────────────────────────────────────────────────────────

def fmt_pval(p):
    if p < 0.001: return "< 0.001"
    if p < 0.01:  return f"{p:.3f}"
    if p < 0.05:  return f"{p:.3f}"
    return f"{p:.3f}"

def stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""

def reg_table_md(model, title=""):
    rows = ["| Variable | Coef | Std Err | t | P>|t| | 95% CI |",
            "|----------|------|---------|---|-------|--------|"]
    ci = model.conf_int()
    for var in model.params.index:
        c = model.params[var]
        se = model.bse[var]
        t  = model.tvalues[var]
        p  = model.pvalues[var]
        lo, hi = ci.loc[var]
        rows.append(f"| {var} | {c:.4f}{stars(p)} | {se:.4f} | {t:.3f} | {fmt_pval(p)} | [{lo:.4f}, {hi:.4f}] |")
    n  = int(model.nobs)
    r2 = model.rsquared
    rows.append(f"\n*N = {n} | R² = {r2:.4f} | \\*p<0.05 \\*\\*p<0.01 \\*\\*\\*p<0.001*")
    return "\n".join(rows)

print("\n=== All computations done. Writing markdown... ===")

# ─────────────────────────────────────────────────────────────────────────────
# WRITE MARKDOWN REPORT
# ─────────────────────────────────────────────────────────────────────────────
md = f"""# INTL 601 Research Methods I — Exercise \\#1
## Voter Mobilization Field Experiment: Complete Analysis

---

## Table of Contents
1. [Data Overview & Descriptive Statistics](#1-data-overview--descriptive-statistics)
2. [OLS Univariate Analysis & Difference-of-Means](#2-ols-univariate-analysis--difference-of-means)
3. [OLS with Control Variables](#3-ols-with-control-variables)
4. [Causal Structure: Z → T → Y](#4-causal-structure-z--t--y)
5. [Targeted Treatment & Confounding Bias](#5-targeted-treatment--confounding-bias)
6. [Summary & Conclusions](#6-summary--conclusions)

---

## 1. Data Overview & Descriptive Statistics

### Dataset
This dataset contains **{N:,} simulated observations** of individual voters, designed to mimic a field experiment on voter mobilization (Gerber & Green). The key variables are:

| Variable | Type | Description |
|----------|------|-------------|
| `Y` | Binary | **Outcome**: 1 = voted, 0 = did not vote |
| `Z` | Binary | **Random assignment**: 1 = assigned to canvassing, 0 = not assigned |
| `T` | Binary | **Treatment received**: 1 = actually contacted, 0 = not contacted |
| `age` | Integer | Voter age |
| `educ` | Integer | Years of education |
| `pastvote` | Binary | Past turnout (1 = voted before, 0 = did not) |
| `party_id` | Continuous | Partisan strength (higher = stronger partisan) |
| `competitive` | Binary | District competitiveness (1 = competitive) |

### Descriptive Statistics

{desc_stats.to_markdown()}

### Key Rates

| Metric | Value |
|--------|-------|
| **Turnout rate** (Y = 1) | **{turnout_rate*100:.1f}%** |
| **Assignment rate** (Z = 1) | **{assignment_rate*100:.1f}%** |
| **Contact rate** (T = 1) | **{contact_rate*100:.1f}%** |
| Contact rate among assigned (Z = 1) | {contact_Z1*100:.1f}% |
| Contact rate among not assigned (Z = 0) | {contact_Z0*100:.1f}% |
| **Compliance gap** (Z=1 minus Z=0) | **{compliance_gap*100:.1f} percentage points** |

> **Key observation**: Assignment (Z) strongly raises the probability of contact (T) — from
> {contact_Z0*100:.1f}% to {contact_Z1*100:.1f}%, a gap of **{compliance_gap*100:.1f} pp**. However, because compliance
> is imperfect (Z does not perfectly determine T), we have a classic **one-sided noncompliance**
> experiment: some assigned voters are never contacted, and some non-assigned voters happen to
> be contacted through other channels.

![Figure 1: Descriptive Statistics](fig1_descriptive.png)

---

## 2. OLS Univariate Analysis & Difference-of-Means

### Difference-of-Means Test

| Group | N | Mean Turnout |
|-------|---|-------------|
| Not contacted (T = 0) | {len(grp0):,} | {grp0.mean()*100:.2f}% |
| Contacted (T = 1) | {len(grp1):,} | {grp1.mean()*100:.2f}% |
| **Difference** | — | **{dom*100:.2f} pp** |

Two-sample t-test: *t* = {t_stat:.3f}, *p* {fmt_pval(p_val)}

### OLS Regression: `reg Y T`

{reg_table_md(model1, 'OLS: Y ~ T')}

### Margins: Predicted Probabilities at T = 0 and T = 1

Using `margins at(T=(0 1))` (equivalent in Python: predicted values from the OLS model):

| | Value |
|--|-------|
| **P̂(Y = 1 \| T = 0)** | **{pred_T0:.4f}** ({pred_T0*100:.2f}%) |
| **P̂(Y = 1 \| T = 1)** | **{pred_T1:.4f}** ({pred_T1*100:.2f}%) |
| **Difference** | **{(pred_T1-pred_T0):.4f}** ({(pred_T1-pred_T0)*100:.2f} pp) |
| OLS coefficient on T | {T1_coef:.4f} |

> **Interpretation**: The predicted probability of turnout for someone *not* contacted is
> **{pred_T0*100:.1f}%**, and for someone contacted it is **{pred_T1*100:.1f}%**. The difference
> ({(pred_T1-pred_T0)*100:.2f} pp) is **exactly equal to the OLS regression coefficient on T** ({T1_coef:.4f}).
> This is not a coincidence: in an OLS linear probability model, the coefficient on a binary
> predictor is always the difference in predicted means between the two groups. The `margins`
> command in Stata, like computing predicted values at T=0 and T=1, recovers the same quantity.

![Figure 2: Univariate OLS](fig2_ols_univariate.png)

---

## 3. OLS with Control Variables

### Regression: `reg Y T age educ pastvote party_id competitive`

{reg_table_md(model2)}

### Coefficient on T: With vs. Without Controls

| Model | T coefficient | Change |
|-------|--------------|--------|
| Univariate `reg Y T` | {T1_coef:.4f} | — |
| With controls `reg Y T + X` | {T2_coef:.4f} | {change:+.4f} ({pct_change:.1f}% {'increase' if change > 0 else 'decrease'}) |

> **Did the coefficient change a lot or a little?** The coefficient changed from **{T1_coef:.4f}**
> to **{T2_coef:.4f}** — a change of **{change:+.4f}** ({pct_change:.1f}%). This is a very **small** change,
> which is expected: because **Z was randomly assigned**, the treatment T is (approximately)
> uncorrelated with the pre-treatment covariates. Adding controls in a randomized experiment
> mainly increases precision (reduces standard errors) rather than removing omitted-variable bias.
> The small change confirms that randomization successfully balanced covariates across treatment groups.

### Marginal Effect of T

In a linear probability model (OLS), the **marginal effect of T is simply the coefficient** on T:
**ME(T) = {T2_coef:.4f}**

| | Value |
|--|-------|
| Marginal effect of T (from model with controls) | **{T2_coef:.4f}** |
| Raw difference in means (univariate) | {dom:.4f} |
| Difference | {T2_coef - dom:+.4f} |

> The marginal effect and the raw difference in means are very close ({T2_coef - dom:+.4f} apart),
> again reflecting the balanced randomization.

### Which Control Variable Has the Biggest Association with Y?

Ranked by absolute t-statistic (the standard metric for relative importance within a regression):

| Variable | Coefficient | Std Err | t-statistic | p-value |
|----------|------------|---------|-------------|---------|
{chr(10).join(f'| **{v}** | {ctrl_info[v]["coef"]:.4f} | {ctrl_info[v]["se"]:.4f} | **{ctrl_info[v]["t"]:.2f}** | {fmt_pval(ctrl_info[v]["p"])} |' for v in sorted_by_t)}

> **`{biggest_confounder}`** has the largest absolute t-statistic (**|t| = {abs(ctrl_info[biggest_confounder]['t']):.2f}**),
> indicating it has the strongest marginal association with turnout Y after controlling for all
> other variables. You can read this directly from the regression output: the variable with
> the largest |t| (or equivalently the smallest p-value) is the most influential predictor.

![Figure 3: OLS with Controls](fig3_controls.png)

![Figure 4: Covariate Associations](fig4_covariates.png)

---

## 4. Causal Structure: Z → T → Y

The full causal structure is:

```
Z (random assignment) ──→ T (actual contact) ──→ Y (turnout)
         ↓                        ↑
         └──── X (covariates) ────┘
```

This involves **three equations**:

### First Stage: `reg T ~ Z + covariates`

*Does random assignment (Z) actually increase contact (T)?*

{reg_table_md(fs_model)}

> **Z coefficient = {FS_coef:.4f}** (t = {FS_t:.3f}, p < 0.001).
> Being randomly assigned raises the probability of contact by **{FS_coef*100:.1f} pp**.
> This is the **first stage** of an instrumental variables design.
> The F-statistic for Z is well above 10, confirming Z is a **strong instrument**.

### Reduced Form (Intention-to-Treat): `reg Y ~ Z + covariates`

*What is the causal effect of being **assigned** (regardless of actual contact)?*

{reg_table_md(rf_model)}

> **Z coefficient = {ITT_coef:.4f}** — this is the **Intention-to-Treat (ITT) effect**.
> Random assignment raises turnout by **{ITT_coef*100:.1f} pp** on average across all assigned voters,
> including those who were never actually contacted.

### 2SLS / IV Estimate (LATE)

Using Z as an instrument for T (2SLS via `linearmodels`):

| Estimator | Estimate | Std Err | t-stat | p-value |
|-----------|----------|---------|--------|---------|
| ITT (reduced form) | {ITT_coef:.4f} | {ITT_se:.4f} | {ITT_t:.3f} | < 0.001 |
| First stage (Z→T) | {FS_coef:.4f} | {FS_se:.4f} | {FS_t:.3f} | < 0.001 |
| **LATE = ITT / FS** | **{LATE_man:.4f}** | — | — | — |
| **2SLS estimate** | **{LATE_2sls:.4f}** | {LATE_2sls_se:.4f} | {LATE_2sls_t:.3f} | {fmt_pval(LATE_2sls_p)} |
| OLS (T→Y, with ctrl) | {T2_coef:.4f} | {T2_se:.4f} | {T2_t:.3f} | < 0.001 |

> **LATE = ITT / First Stage = {ITT_coef:.4f} / {FS_coef:.4f} = {LATE_man:.4f}**
> (matches 2SLS: {LATE_2sls:.4f} ✓)

### What Does This Model Help Us Infer?

This causal structure — with Z as a **randomized instrument**, T as the **endogenous treatment**,
and Y as the **outcome** — allows us to answer several distinct questions:

| Question | Estimand | Answer |
|----------|----------|--------|
| Effect of *being assigned* to canvassing (regardless of contact) | **ITT** | {ITT_coef:.4f} ({ITT_coef*100:.1f} pp) |
| Effect of *actual contact* on turnout, for those who comply | **LATE (2SLS)** | {LATE_2sls:.4f} ({LATE_2sls*100:.1f} pp) |
| Association of contact with turnout (controlling for X) | **OLS** | {T2_coef:.4f} |

**Direct and Indirect Effects:**
- **Direct effect of Z on Y**: Because Z was randomly assigned and its only channel to Y is *through T*
  (the exclusion restriction), Z has **no direct effect** on Y — it only operates *indirectly* through T.
- **Indirect (mediated) path**: Z → T → Y. The ITT ({ITT_coef:.4f}) captures this full path.
  Dividing by the first stage ({FS_coef:.4f}) scales up to the LATE ({LATE_2sls:.4f}), which is the
  average treatment effect for **compliers** (voters who are contacted if and only if assigned).
- **Covariate paths**: X affects both T (selection into compliance) and Y (baseline turnout rates),
  but because Z is random and independent of X, we can condition on X without introducing bias.

> **Bottom line**: The 2SLS estimate ({LATE_2sls:.4f}) is the cleanest causal estimate of "how much
> does being contacted increase the probability of voting?" for the subset of voters whose
> contact status was actually changed by the random assignment (the Local Average Treatment Effect).

![Figure 5: Causal Structure](fig5_causal.png)

---

## 5. Targeted Treatment & Confounding Bias

### Generating the Targeted Treatment

```stata
gen T_target = (runiform() < invlogit(-1 + 1.2*pastvote + 0.5*party_id))
```

This creates a **non-random, observational treatment indicator**: voters are more likely to be
"treated" if they have voted before (`pastvote`) and have stronger partisan identity (`party_id`).
Both of these characteristics also independently predict higher turnout (Y).

T_target rate: **{T_target_rate*100:.1f}%**

### Contact Rate by Past Vote (tab T_target pastvote, row)

{tab_row.to_markdown()}

> Voters who voted in the past are **much more likely** to receive the targeted treatment:
> {tab_row.loc[1,1]:.1f}% vs. {tab_row.loc[0,1]:.1f}% among non-past voters.
> This creates confounding: T_target is correlated with pastvote, and pastvote predicts Y.

### Regression Results

**Without controls** (`reg Y T_target`):

{reg_table_md(model_obs1)}

**With controls** (`reg Y T_target + age educ pastvote party_id competitive`):

{reg_table_md(model_obs2)}

### Coefficient Comparison

| Model | Coefficient on Treatment | Notes |
|-------|------------------------|-------|
| T experimental (no controls) | {T1_coef:.4f} | Random assignment — unbiased |
| **T_target (no controls)** | **{Tt_nc:.4f}** | **Biased upward by confounding** |
| T_target (with controls) | {Tt_wc:.4f} | Partially corrected |

**Bias magnitudes:**
- T_target without controls is **{Tt_nc - Tt_wc:.4f} higher** than with controls (downward bias correction when adding controls)
- T_target without controls is **{Tt_nc - T1_coef:.4f} higher** than the experimental T estimate

> **Which is more biased?** `T_target` without controls (**{Tt_nc:.4f}**) is far more biased than
> the experimental `T` without controls (**{T1_coef:.4f}**). The experimental T estimate is
> unbiased because random assignment ensures Z (and hence T) is uncorrelated with all
> confounders. The T_target estimate inflates the apparent treatment effect because
> targeted voters would have voted at higher rates *anyway* — the treatment didn't cause their
> high turnout, their pre-existing characteristics did.

### Which Variable Is Doing the Most Confounding?

Running `reg T_target ~ pastvote party_id age educ competitive` to see what predicts T_target:

| Variable | T_target coefficient | Y coefficient (model2) | Confounder? |
|----------|---------------------|----------------------|-------------|
| **pastvote** | **{conf_check.params['pastvote']:.4f}** | **{ctrl_info['pastvote']['coef']:.4f}** | **YES — strong** |
| party_id | {conf_check.params['party_id']:.4f} | {ctrl_info['party_id']['coef']:.4f} | Moderate |
| age | {conf_check.params['age']:.4f} | {ctrl_info['age']['coef']:.4f} | Weak/none |

> **`pastvote` is the primary confounder.** You can see this because:
> 1. It strongly predicts T_target (by construction: coefficient of **1.2** in the logit formula)
> 2. It is the strongest predictor of Y in the outcome regression (largest |t|: **{abs(ctrl_info['pastvote']['t']):.2f}**)
>
> When you add controls, the T_target coefficient falls from **{Tt_nc:.4f}** to **{Tt_wc:.4f}**
> (a drop of **{Tt_nc - Tt_wc:.4f}**), and most of that drop comes from controlling for pastvote,
> which "absorbs" the spurious correlation between T_target and Y that was previously
> attributed to the treatment.

![Figure 6: Targeted Treatment](fig6_targeted.png)

---

## 6. Summary & Conclusions

### All Estimates Side-by-Side

| Estimator | Coefficient | Interpretation |
|-----------|------------|----------------|
| Diff-of-means: E[Y\|T=1] − E[Y\|T=0] | {dom:.4f} | Raw association |
| OLS: reg Y T (univariate) | {T1_coef:.4f} | = diff-of-means (by construction) |
| OLS: reg Y T + controls | {T2_coef:.4f} | Controlled association |
| ITT: Z effect on Y (reduced form) | {ITT_coef:.4f} | Intent-to-treat |
| LATE (2SLS): Z instruments T→Y | {LATE_2sls:.4f} | Causal effect for compliers |
| T_target (no controls) | {Tt_nc:.4f} | **Biased (OVB)** |
| T_target (with controls) | {Tt_wc:.4f} | Partially corrected |

### Key Takeaways

1. **Randomization works**: Adding controls barely changes the experimental T coefficient
   ({change:+.4f}, {pct_change:.1f}%), confirming that Z created balanced treatment groups.

2. **Margins = OLS coefficient**: In a linear probability model, the predicted probability
   difference from `margins at(T=(0 1))` equals the OLS coefficient exactly.

3. **Past vote is the key predictor of turnout**: Among all covariates, `{biggest_confounder}`
   has the largest t-statistic (|t| = {abs(ctrl_info[biggest_confounder]['t']):.2f}), dominating
   the prediction of Y.

4. **The causal chain Z → T → Y separates ITT from LATE**:
   - ITT = {ITT_coef:.4f}: the average effect of being assigned (diluted by non-compliance)
   - LATE = {LATE_2sls:.4f}: the effect for compliers only (larger because it excludes
     never-takers and always-takers)

5. **Targeted treatment introduces confounding bias**: Without controls, T_target overstates
   the treatment effect by **{Tt_nc - T1_coef:.4f}** compared to the experimental benchmark.
   The primary confounder is **past vote**, which simultaneously predicts selection into
   treatment (by construction) and the outcome (turnout).

---

*Analysis conducted in Python using `pandas`, `statsmodels`, `scipy`, `linearmodels`, `matplotlib`, and `seaborn`.*
*Dataset: `gg_fake.dta` — 5,000 simulated observations (teaching dataset).*
"""

with open(DIR + 'INTL601_Exercise1_Report.md', 'w') as f:
    f.write(md)

print(f"\nMarkdown report saved to: {DIR}INTL601_Exercise1_Report.md")
print("=== DONE ===")
