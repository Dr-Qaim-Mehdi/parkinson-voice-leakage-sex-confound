#!/usr/bin/env python3
"""Figures for the sex paper.  Usage: figs_p2.py OUT_ROOT FIG_DIR"""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

root, figdir = sys.argv[1:3]
os.makedirs(figdir, exist_ok=True)
S = json.load(open(os.path.join(root, 'summary.json')))
E = json.load(open(os.path.join(root, 'extra.json')))
U = json.load(open(os.path.join(root, 'univariate.json')))
O = json.load(open(os.path.join(root, 'oxford_sex.json')))

# categorical roles (validated pairs): diagnosis = blue/orange, sex = aqua/violet
HC, PD = '#2a78d6', '#eb6834'
WOM, MEN = '#1baf7a', '#6250d6'
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK, 'axes.labelcolor': INK, 'xtick.color': INK,
                     'ytick.color': INK, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titlesize': 10, 'figure.dpi': 100, 'savefig.dpi': 300, 'legend.frameon': False})


def grid(ax, axis='x'):
    ax.grid(axis=axis, color=GRID, lw=0.8); ax.set_axisbelow(True)


SHORT = {'Logistic Regression': 'Logistic regression', 'LDA': 'LDA', 'Gaussian NB': 'Gaussian NB', 'k-NN': 'k-NN',
         'Decision Tree': 'Decision tree', 'SVM (RBF)': 'SVM (RBF)', 'Random Forest': 'Random forest',
         'Extra Trees': 'Extra trees', 'Gradient Boosting': 'Gradient boosting', 'XGBoost': 'XGBoost', 'MLP': 'MLP'}

# ---------------------------------------------------------------- Figure 1: Oxford, recorded sex
fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.1), gridspec_kw={'width_ratios': [1, 1.25]})
rng = np.random.default_rng(3)
for sp in O['speakers']:
    x0 = 0 if sp['status'] == 0 else 1
    x = x0 + rng.uniform(-0.13, 0.13)
    if sp['sex'] is None:
        a.scatter(x, sp['f0'], s=42, facecolor='white', edgecolor=MUTED, lw=1.2, zorder=3)
    else:
        a.scatter(x, sp['f0'], s=46, color=WOM if sp['sex'] == 'F' else MEN, edgecolor='white', lw=1.0, zorder=3)
a.axhline(O['f0_cut'], color=MUTED, lw=1.0, ls=(0, (4, 3)))
a.text(-0.46, O['f0_cut'] + 3, f"{O['f0_cut']:.0f} Hz: largest gap\namong controls", fontsize=7.5, color=MUTED, ha='left', va='bottom')
a.set_xticks([0, 1]); a.set_xticklabels(['Controls', 'PD']); a.set_xlim(-0.5, 1.5); a.set_ylim(90, 262)
a.set_ylabel('Speaker-mean MDVP:Fo (Hz)'); grid(a, 'y')
a.set_title('(a) Fundamental frequency by recorded sex', loc='left')
a.legend(handles=[Line2D([], [], marker='o', ls='', color=WOM, label='Women', markersize=6.5),
                  Line2D([], [], marker='o', ls='', color=MEN, label='Men', markersize=6.5),
                  Line2D([], [], marker='o', ls='', markerfacecolor='white', markeredgecolor=MUTED, label='Not tabulated', markersize=6)],
         loc='upper right', fontsize=7.5, ncol=1, handletextpad=0.2, borderaxespad=0.1)
err = O['loso_errors']
order = sorted(err, key=lambda m: (err[m]['fp_m'] / err[m]['n_hc_m'], err[m]['fp_f'] / err[m]['n_hc_f']))
y = np.arange(len(order)); h = 0.34
b.barh(y + h / 2 + 0.02, [err[m]['fp_m'] / err[m]['n_hc_m'] for m in order], height=h, color=MEN, label='Male controls (n = 3)')
b.barh(y - h / 2 - 0.02, [err[m]['fp_f'] / err[m]['n_hc_f'] for m in order], height=h, color=WOM, label='Female controls (n = 5)')
b.set_yticks(y); b.set_yticklabels([SHORT[m] for m in order]); b.set_xlim(0, 1.0)
b.set_xlabel('Proportion classified as PD'); grid(b, 'x')
b.set_title('(b) False positives by recorded sex', loc='left')
b.legend(loc='lower right', fontsize=7.5, borderaxespad=0.1)
fig.tight_layout(w_pad=2.0); fig.savefig(os.path.join(figdir, 'P2_Fig1_oxford_sex.png')); plt.close(fig)

# ---------------------------------------------------------------- Figure 2: univariate, by sex
pf = U['per_feature']
feats = sorted(C.BASELINE21, key=lambda f: abs(pf[f]['women']['r']))
fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 4.3), gridspec_kw={'width_ratios': [1.45, 1]})
yy = np.arange(len(feats))
for i, f in enumerate(feats):
    rw, rm = abs(pf[f]['women']['r']), abs(pf[f]['men']['r'])
    a.plot([rm, rw], [i, i], color='#c9c8c2', lw=2.2, zorder=1, solid_capstyle='round')
    for r_, col, st in ((rw, WOM, 'women'), (rm, MEN, 'men')):
        sig = pf[f][st]['p_holm'] < 0.05
        a.scatter(r_, i, s=44, facecolor=col if sig else 'white', edgecolor=col, lw=1.4, zorder=3)
a.set_yticks(yy); a.set_yticklabels(feats, fontsize=7.5); a.set_xlim(0, 0.92); a.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
a.set_xlabel('|rank-biserial r|, PD versus controls'); grid(a, 'x')
a.set_title('(a) Effect size of each variable, by sex', loc='left')
a.legend(handles=[Line2D([], [], marker='o', ls='', color=WOM, label='Women', markersize=6.5),
                  Line2D([], [], marker='o', ls='', color=MEN, label='Men', markersize=6.5),
                  Line2D([], [], marker='o', ls='', markerfacecolor='white', markeredgecolor=MUTED, label='Not significant after Holm', markersize=6)],
         loc='lower right', fontsize=7.5, borderaxespad=0.2, handletextpad=0.2)
cw = np.array(U['power_matched']['counts_women']); cm = np.array(U['power_matched']['counts_men'])
bins = np.arange(-0.5, 22.5, 1)
hw = np.histogram(cw, bins)[0] / len(cw); hm = np.histogram(cm, bins)[0] / len(cm)
xs = np.arange(0, 22)
b.bar(xs - 0.2, hm, width=0.38, color=MEN, label='Men')
b.bar(xs + 0.2, hw, width=0.38, color=WOM, label='Women')
b.set_xlabel('Significant variables (of 21)'); b.set_ylabel('Proportion of 2000 resamples')
b.set_xticks([0, 3, 6, 9, 12, 15, 18, 21]); grid(b, 'y'); b.set_xlim(-0.8, 21.8)
b.set_title('(b) Power-matched counts', loc='left')
b.text(1.6, hm[1] - 0.03, f"men: mean {cm.mean():.1f}", fontsize=8, color=INK, va='top')
b.text(12.2, hw.max() + 0.035, f"women: mean {cw.mean():.1f}", fontsize=8, color=INK, ha='center')
b.legend(loc='upper right', fontsize=7.5)
fig.tight_layout(w_pad=1.8); fig.savefig(os.path.join(figdir, 'P2_Fig2_univariate.png')); plt.close(fig)

# ---------------------------------------------------------------- Figure 3: anatomy of pooled classifiers
fig = plt.figure(figsize=(7.4, 6.2))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.45], hspace=0.42, wspace=0.42)
labels = ['Women\nonly', 'Men\nonly', 'F control,\nM patient', 'M control,\nF patient']
keys = ['auc_within_f', 'auc_within_m', 'auc_aligned', 'auc_opposed']
for j, (fs, ttl) in enumerate((('b21', '(a) Baseline-21: ROC-AUC by pair type'), ('f752', '(b) Full-752: ROC-AUC by pair type'))):
    ax = fig.add_subplot(gs[0, j]); A = S[f'{fs}/anatomy']
    vals = [A['mean'][k] for k in keys]
    lo = [A['mean'][k] - A['ci'][k][0] for k in keys]; hi = [A['ci'][k][1] - A['mean'][k] for k in keys]
    ax.errorbar(range(4), vals, yerr=[lo, hi], fmt='o', color=HC, markersize=8, markeredgecolor='white',
                markeredgewidth=1.2, ecolor=INK, elinewidth=1.0, capsize=3, zorder=3)
    ax.axhline(A['mean']['auc_pooled'], color=MUTED, lw=1.0, ls=(0, (4, 3)))
    ax.text(3.68, A['mean']['auc_pooled'] + 0.008, f"all pairs {A['mean']['auc_pooled']:.3f}", fontsize=7.5, color=MUTED, ha='right', va='bottom')
    for i, v in enumerate(vals):
        ax.text(i + 0.12, v, f'{v:.3f}', ha='left', va='center', fontsize=7.5, color=INK)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=7.5); ax.set_xlim(-0.5, 3.7)
    ax.set_ylim(0.4, 1.0); ax.set_ylabel('ROC-AUC'); grid(ax, 'y'); ax.set_title(ttl, loc='left')
for j, (fs, ttl) in enumerate((('b21', '(c) Baseline-21: controls classified as PD'), ('f752', '(d) Full-752: controls classified as PD'))):
    ax = fig.add_subplot(gs[1, j]); pm = S[f'{fs}/anatomy']['per_model']
    order = sorted(pm, key=lambda m: pm[m]['fpr_m'])
    for i, m in enumerate(order):
        ax.plot([pm[m]['fpr_f'], pm[m]['fpr_m']], [i, i], color='#c9c8c2', lw=2.2, zorder=1, solid_capstyle='round')
        ax.scatter(pm[m]['fpr_f'], i, s=44, color=WOM, edgecolor='white', lw=1.0, zorder=3)
        ax.scatter(pm[m]['fpr_m'], i, s=44, color=MEN, edgecolor='white', lw=1.0, zorder=3)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([SHORT[m] for m in order], fontsize=7.5)
    ax.set_xlim(0, 1); ax.set_xlabel('False-positive rate among controls'); grid(ax, 'x'); ax.set_title(ttl, loc='left')
    if j == 0:
        ax.legend(handles=[Line2D([], [], marker='o', ls='', color=WOM, label='Female controls', markersize=6.5),
                           Line2D([], [], marker='o', ls='', color=MEN, label='Male controls', markersize=6.5)],
                  loc='lower right', fontsize=7.5, borderaxespad=0.2)
fig.savefig(os.path.join(figdir, 'P2_Fig3_anatomy.png'), bbox_inches='tight'); plt.close(fig)

# ---------------------------------------------------------------- Figure 4: confound dose-response
if 'confdose_b21' in E and len(E['confdose_b21']['arms']) == 4:
    arms = ['OR1.0', 'OR2.4', 'OR5.2', 'OR10']; D = E['confdose_b21']['arms']
    xl = [f"{D[a]['odds_ratio']:.1f}" for a in arms]
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.75))
    specs = [('(a) Discrimination', [('auc_pooled', 'All pairs', HC, 'o'), ('auc_adj', 'Sex-adjusted', PD, 's')], 'ROC-AUC'),
             ('(b) By pair type', [('auc_aligned', 'F control, M patient', HC, 'o'), ('auc_opposed', 'M control, F patient', PD, 's')], 'ROC-AUC'),
             ('(c) Controls classified as PD', [('fpr_m', 'Male controls', MEN, 'o'), ('fpr_f', 'Female controls', WOM, 's')], 'False-positive rate')]
    for ax, (ttl, series, yl) in zip(axs, specs):
        for k, lab, col, mk in series:
            m = [D[a]['mean'][k] for a in arms]; s = [D[a]['sd'][k] for a in arms]
            ax.errorbar(range(4), m, yerr=s, color=col, lw=2, marker=mk, markersize=6, markeredgecolor='white',
                        markeredgewidth=1.0, capsize=2.5, elinewidth=1.0, label=lab)
        ax.set_xticks(range(4)); ax.set_xticklabels(xl); ax.set_xlabel('Odds ratio, male sex and PD')
        ax.set_ylabel(yl); grid(ax, 'y'); ax.set_title(ttl, loc='left'); ax.legend(fontsize=7, loc='best')
    fig.tight_layout(w_pad=1.2); fig.savefig(os.path.join(figdir, 'P2_Fig4_confdose.png')); plt.close(fig)
print('figures written to', figdir)
