#!/usr/bin/env python3
"""Figures for the sex-confound paper.  Usage: figures_p2.py DATA UNIVARIATE SUMMARY SUMMARY2 OUTDIR [which]"""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

BLUE, ORANGE, GRAY, DGRAY = '#1f77b4', '#ff7f0e', '#b0b0b0', '#555555'
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 300, 'savefig.dpi': 300})

data, uni_p, summ_p, summ2_p, outdir = sys.argv[1:6]
which = sys.argv[6] if len(sys.argv) > 6 else 'all'
os.makedirs(outdir, exist_ok=True)
U = json.load(open(uni_p)); S = json.load(open(summ_p))
S2 = json.load(open(summ2_p)) if os.path.exists(summ2_p) else {}

# ---------------------------------------------------------------- Figure 1: composition + effect modification
if which in ('all', '1'):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    comp = U['composition']
    cats = ['Control\n(n = 64)', 'PD\n(n = 188)']
    fem = [comp['hc_f'], comp['pd_f']]; mal = [comp['hc_m'], comp['pd_m']]
    tot = [64, 188]
    x = np.arange(2); w = 0.6
    pf = [f / t for f, t in zip(fem, tot)]; pm = [m / t for m, t in zip(mal, tot)]
    ax.bar(x, pf, w, color=BLUE, label='Women', edgecolor='white', linewidth=2)
    ax.bar(x, pm, w, bottom=pf, color=ORANGE, label='Men', edgecolor='white', linewidth=2)
    for i in range(2):
        ax.text(x[i], pf[i] / 2, f'{fem[i]} women\n({100*pf[i]:.1f}%)', ha='center', va='center',
                color='white', fontsize=10, fontweight='bold')
        ax.text(x[i], pf[i] + pm[i] / 2, f'{mal[i]} men\n({100*pm[i]:.1f}%)', ha='center', va='center',
                color='white', fontsize=10, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylabel('Proportion of speakers'); ax.set_ylim(0, 1.02)
    ax.set_title('(a) Sex composition differs by diagnosis\n'
                 f"OR {comp['odds_ratio']:.2f} [{comp['or_lo']:.2f}, {comp['or_hi']:.2f}], "
                 f"Fisher p = {comp['fisher_p']:.4f}", fontsize=11)
    ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(0.99, 0.6))

    ax = axes[1]
    rw = np.array([U['per_feature'][f]['women']['r'] for f in C.BASELINE21])
    rm = np.array([U['per_feature'][f]['men']['r'] for f in C.BASELINE21])
    lim = 0.75
    ax.fill_between([0, lim], [0, 0], [0, lim], color='#eeeeee', zorder=0)
    ax.plot([0, lim], [0, lim], color=DGRAY, lw=1, ls='--')
    ax.scatter(np.abs(rw), np.abs(rm), s=42, color=BLUE, zorder=3, edgecolor='white', linewidth=0.8)
    offs = {'PPE': (6, -3), 'DFA': (6, 2), 'RPDE': (-8, 7), 'locPctJitter': (-6, -14),
            'apq11Shimmer': (6, 2), 'meanHarmToNoiseHarmonicity': (7, -9)}
    for f, xw, xm in zip(C.BASELINE21, np.abs(rw), np.abs(rm)):
        if f in offs:
            lab = {'locPctJitter': 'jitter (%)', 'apq11Shimmer': 'shimmer APQ11',
                   'meanHarmToNoiseHarmonicity': 'HNR'}.get(f, f)
            ax.annotate(lab, (xw, xm), textcoords='offset points', xytext=offs[f], fontsize=8.5, color=DGRAY)
    ax.set_xlabel('|rank-biserial r|, women (41 HC vs 81 PD)')
    ax.set_ylabel('|rank-biserial r|, men (23 HC vs 107 PD)')
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_title('(b) Univariate disease effects by sex stratum', fontsize=11.5)
    n_above = int((np.abs(rw) > np.abs(rm)).sum())
    ax.text(0.97, 0.06, f'{n_above} of 21 features: larger effect in women\n(shaded: effect larger in women)',
            transform=ax.transAxes, ha='right', fontsize=9, color=DGRAY)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'P2_Fig1_composition.png'), bbox_inches='tight')
    plt.close(fig)
    print('fig1 done')

# ---------------------------------------------------------------- Figure 2: pair-type decomposition + error structure
if which in ('all', '2'):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), gridspec_kw={'width_ratios': [1, 1.15]})
    ax = axes[0]
    A = S['b21/anatomy']; mean, ci = A['mean'], A['ci']
    order = [('auc_opposed', 'Confound-opposed\nmale HC vs female PD'),
             ('auc_within_m', 'Within men'),
             ('auc_within_f', 'Within women'),
             ('auc_pooled', 'Pooled\n(all pairs)'),
             ('auc_aligned', 'Confound-aligned\nfemale HC vs male PD')]
    y = np.arange(len(order))
    vals = [mean[k] for k, _ in order]
    los = [ci[k][0] for k, _ in order]; his = [ci[k][1] for k, _ in order]
    cols = [GRAY, BLUE, BLUE, DGRAY, ORANGE]
    ax.barh(y, vals, 0.62, color=cols, edgecolor='white')
    ax.errorbar(vals, y, xerr=[np.array(vals) - np.array(los), np.array(his) - np.array(vals)],
                fmt='none', ecolor='#333333', elinewidth=1.2, capsize=3)
    for yi, v, h in zip(y, vals, his):
        ax.text(h + 0.012, yi, f'{v:.3f}', va='center', fontsize=9.5)
    ax.axvline(0.5, color=DGRAY, lw=1, ls=':')
    ax.set_yticks(y); ax.set_yticklabels([lbl for _, lbl in order], fontsize=9.5)
    ax.set_xlim(0.45, 0.95)
    ax.set_xlabel('ROC-AUC (mean of 11 classifiers, speaker-disjoint)\ndotted line: chance (0.5); bars: 95% speaker-bootstrap CI')
    ax.set_title('(a) The pooled AUC is a mixture of pair types', fontsize=11.5)

    ax = axes[1]
    pm = S['b21/speaker_errors']['per_model']
    names = [n for n in C.MODEL_ORDER if n in pm]
    yy = np.arange(len(names))[::-1]
    fm = [pm[n]['fpr_m'] for n in names]; ff = [pm[n]['fpr_f'] for n in names]
    for yi, a, b in zip(yy, fm, ff):
        ax.plot([b, a], [yi, yi], color='#cccccc', lw=2.5, zorder=1)
    ax.scatter(ff, yy, s=46, color=BLUE, zorder=3, label='Female controls (n = 41)')
    ax.scatter(fm, yy, s=46, color=ORANGE, zorder=3, label='Male controls (n = 23)')
    ax.set_yticks(yy); ax.set_yticklabels(names, fontsize=9.5)
    ax.set_xlabel('Proportion of controls misclassified as PD\n(speaker level, majority vote)')
    ax.set_xlim(0, 1.0)
    ax.set_title('(b) False positives concentrate in male controls', fontsize=11.5)
    ax.legend(frameon=False, loc='lower left', fontsize=9, bbox_to_anchor=(-0.02, -0.04))
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'P2_Fig2_decomposition.png'), bbox_inches='tight')
    plt.close(fig)
    print('fig2 done')

# ---------------------------------------------------------------- Figure 3: confound dose-response
if which in ('all', '3') and 'b21/confdose' in S2:
    D = S2['b21/confdose']['per_arm']
    arms = ['OR1.0', 'OR2.4', 'OR5.2', 'OR10']
    ors = [D[a]['or_target'] for a in arms]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax = axes[0]
    m = [D[a]['auc_mean'] for a in arms]; sd = [D[a]['auc_sd'] for a in arms]
    ax.errorbar(ors, m, yerr=sd, color=DGRAY, marker='o', ms=7, lw=2, capsize=3)
    for x_, y_ in zip(ors, m):
        ax.annotate(f'{y_:.3f}', (x_, y_), textcoords='offset points', xytext=(0, 9),
                    ha='center', fontsize=9)
    ax.axvline(2.35, color=ORANGE, lw=1.2, ls='--')
    ax.text(2.45, m[0] + 0.002, 'observed cohort\n(OR 2.35)', fontsize=8.5, color=ORANGE)
    ax.set_xscale('log'); ax.set_xticks(ors); ax.set_xticklabels([f'{o:g}' for o in ors])
    ax.set_xlabel('Engineered male : PD odds ratio (fixed cohort size)')
    ax.set_ylabel('Diagnostic ROC-AUC\n(mean of 6 classifiers)')
    ax.set_title('(a) Apparent skill rises with the confound', fontsize=11.5)
    ax = axes[1]
    ssex = [D[a]['score_sex_auc_mean'] for a in arms]
    fgap = [D[a]['fpr_gap_mean'] for a in arms]
    ax.plot(ors, ssex, color=BLUE, marker='o', ms=7, lw=2, label='Sex readable from the\nPD score (AUC)')
    ax.plot(ors, fgap, color=ORANGE, marker='s', ms=7, lw=2,
            label='FPR gap, male − female\ncontrols')
    for x_, y_ in zip(ors, ssex):
        ax.annotate(f'{y_:.2f}', (x_, y_), textcoords='offset points', xytext=(0, 9), ha='center', fontsize=9)
    for x_, y_ in zip(ors, fgap):
        ax.annotate(f'{y_:.2f}', (x_, y_), textcoords='offset points', xytext=(0, -16), ha='center', fontsize=9)
    ax.set_xscale('log'); ax.set_xticks(ors); ax.set_xticklabels([f'{o:g}' for o in ors])
    ax.set_xlabel('Engineered male : PD odds ratio (fixed cohort size)')
    ax.set_ylabel('AUC / proportion')
    ax.set_ylim(0, 1.0)
    ax.set_title('(b) The score tracks sex as the confound grows', fontsize=11.5)
    ax.legend(frameon=False, fontsize=9, loc='center right')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'P2_Fig3_dose.png'), bbox_inches='tight')
    plt.close(fig)
    print('fig3 done')
