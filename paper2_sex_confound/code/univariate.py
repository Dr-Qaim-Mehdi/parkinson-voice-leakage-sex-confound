#!/usr/bin/env python3
"""Univariate analyses for the sex-confound paper (Sakar 2018 corpus, Baseline-21).

Usage: univariate.py DATA OUT_JSON
All tests are on speaker-level means (one value per participant).
"""
import sys, os, json
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

data, out = sys.argv[1:3]
df = C.load(data)
sp = df.groupby('id')[C.BASELINE21 + ['gender', 'class']].mean()
sex = sp['gender'].to_numpy().astype(int)      # 1 = male
dx = sp['class'].to_numpy().astype(int)        # 1 = PD
F = sp[C.BASELINE21].to_numpy()
R = {}

# ---------- composition and association ----------
a = int(((dx == 1) & (sex == 1)).sum()); b = int(((dx == 1) & (sex == 0)).sum())
c = int(((dx == 0) & (sex == 1)).sum()); d = int(((dx == 0) & (sex == 0)).sum())
orr = (a * d) / (b * c)
se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
fisher_p = stats.fisher_exact([[a, b], [c, d]])[1]
tpr = a / (a + b); fpr = c / (c + d)
R['composition'] = dict(pd_m=a, pd_f=b, hc_m=c, hc_f=d,
                        pct_f_hc=100 * d / (c + d), pct_f_pd=100 * b / (a + b),
                        odds_ratio=orr, or_lo=float(np.exp(np.log(orr) - 1.96 * se)),
                        or_hi=float(np.exp(np.log(orr) + 1.96 * se)), fisher_p=float(fisher_p),
                        auc_sex_alone=0.5 + (tpr - fpr) / 2,
                        phi=float(np.corrcoef(sex, dx)[0, 1]))


def holm(p):
    p = np.asarray(p); m = len(p); order = np.argsort(p)
    adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj


def mw(x1, x0):
    """Two-sided Mann-Whitney; r = rank-biserial, positive when group 1 is higher."""
    u, p = stats.mannwhitneyu(x1, x0, alternative='two-sided')
    return p, 2 * u / (len(x1) * len(x0)) - 1


def table(mask):
    ps, rs = [], []
    for j in range(F.shape[1]):
        p, r = mw(F[mask & (dx == 1), j], F[mask & (dx == 0), j])
        ps.append(p); rs.append(r)
    return np.array(ps), np.array(rs)


strata = {'pooled': np.ones(len(dx), bool), 'women': sex == 0, 'men': sex == 1}
per_feature = {f: {} for f in C.BASELINE21}
R['holm_counts'] = {}
for name, m in strata.items():
    ps, rs = table(m)
    adj = holm(ps)
    R['holm_counts'][name] = int((adj < 0.05).sum())
    R.setdefault('raw_counts', {})[name] = int((ps < 0.05).sum())
    R.setdefault('median_abs_r', {})[name] = float(np.median(np.abs(rs)))
    R.setdefault('mean_abs_r', {})[name] = float(np.mean(np.abs(rs)))
    for j, f in enumerate(C.BASELINE21):
        per_feature[f][name] = dict(p=float(ps[j]), p_holm=float(adj[j]), r=float(rs[j]))

# sex effect (male vs female) per feature, within controls, within PD, and overall
for label, m in {'sex_in_hc': dx == 0, 'sex_in_pd': dx == 1, 'sex_all': np.ones(len(dx), bool)}.items():
    ps, rs = [], []
    for j in range(F.shape[1]):
        p, r = mw(F[m & (sex == 1), j], F[m & (sex == 0), j])
        ps.append(p); rs.append(r)
    adj = holm(ps)
    R['holm_counts'][label] = int((adj < 0.05).sum())
    R['mean_abs_r'][label] = float(np.mean(np.abs(rs)))
    for j, f in enumerate(C.BASELINE21):
        per_feature[f][label] = dict(p=float(ps[j]), p_holm=float(adj[j]), r=float(rs[j]))

# ---------- power-matched counts: both strata to 23 HC / 81 PD ----------
rng = np.random.default_rng(C.RNG)
NRES = 2000
iw_hc = np.where((sex == 0) & (dx == 0))[0]; iw_pd = np.where((sex == 0) & (dx == 1))[0]
im_hc = np.where((sex == 1) & (dx == 0))[0]; im_pd = np.where((sex == 1) & (dx == 1))[0]
n_hc, n_pd = len(im_hc), len(iw_pd)   # 23, 81
assert (n_hc, n_pd) == (23, 81)
cw, cm = [], []
sigw = np.zeros(F.shape[1]); sigm = np.zeros(F.shape[1])
for _ in range(NRES):
    hw = rng.choice(iw_hc, n_hc, replace=False)
    pm = rng.choice(im_pd, n_pd, replace=False)
    pw = np.array([stats.mannwhitneyu(F[iw_pd, j], F[hw, j], alternative='two-sided')[1]
                   for j in range(F.shape[1])])
    pmn = np.array([stats.mannwhitneyu(F[pm, j], F[im_hc, j], alternative='two-sided')[1]
                    for j in range(F.shape[1])])
    aw = holm(pw) < 0.05; am = holm(pmn) < 0.05
    cw.append(aw.sum()); cm.append(am.sum()); sigw += aw; sigm += am
cw, cm = np.array(cw), np.array(cm)
R['power_matched'] = dict(n_resamples=NRES, n_hc=n_hc, n_pd=n_pd,
                          women_mean=float(cw.mean()), women_sd=float(cw.std(ddof=1)),
                          women_p2_5=float(np.percentile(cw, 2.5)), women_p97_5=float(np.percentile(cw, 97.5)),
                          men_mean=float(cm.mean()), men_sd=float(cm.std(ddof=1)),
                          men_p2_5=float(np.percentile(cm, 2.5)), men_p97_5=float(np.percentile(cm, 97.5)),
                          frac_women_gt_men=float((cw > cm).mean()),
                          counts_women=[int(v) for v in cw], counts_men=[int(v) for v in cm])
for j, f in enumerate(C.BASELINE21):
    per_feature[f]['pm_frac_sig_women'] = float(sigw[j] / NRES)
    per_feature[f]['pm_frac_sig_men'] = float(sigm[j] / NRES)

# ---------- sex-stratified (van Elteren) test of PD vs control ----------
def van_elteren(j):
    T = 0.0; V = 0.0
    for s in (0, 1):
        m = sex == s
        x = F[m, j]; y = dx[m]
        n1 = int((y == 1).sum()); n0 = int((y == 0).sum()); N = n1 + n0
        rk = stats.rankdata(x)
        W = rk[y == 1].sum()
        _, cnt = np.unique(x, return_counts=True)
        tie = (cnt ** 3 - cnt).sum() / (N * (N - 1))
        var = n1 * n0 / 12.0 * ((N + 1) - tie)
        w = 1.0 / (N + 1)
        T += w * (W - n1 * (N + 1) / 2.0); V += w * w * var
    z = T / np.sqrt(V)
    return z, 2 * stats.norm.sf(abs(z))


ve = [van_elteren(j) for j in range(F.shape[1])]
ve_p = np.array([p for _, p in ve]); ve_adj = holm(ve_p)
R['holm_counts']['van_elteren_sex_stratified'] = int((ve_adj < 0.05).sum())
for j, f in enumerate(C.BASELINE21):
    per_feature[f]['van_elteren'] = dict(z=float(ve[j][0]), p=float(ve_p[j]), p_holm=float(ve_adj[j]))

# ---------- effect modification: bootstrap of r(women) - r(men) ----------
rngb = np.random.default_rng(C.RNG + 1)
NB = 2000
cells = [iw_hc, iw_pd, im_hc, im_pd]
diff = np.zeros((NB, F.shape[1]))
for b_ in range(NB):
    bw_hc = rngb.choice(iw_hc, len(iw_hc)); bw_pd = rngb.choice(iw_pd, len(iw_pd))
    bm_hc = rngb.choice(im_hc, len(im_hc)); bm_pd = rngb.choice(im_pd, len(im_pd))
    for j in range(F.shape[1]):
        rw = 2 * stats.mannwhitneyu(F[bw_pd, j], F[bw_hc, j])[0] / (len(bw_pd) * len(bw_hc)) - 1
        rm = 2 * stats.mannwhitneyu(F[bm_pd, j], F[bm_hc, j])[0] / (len(bm_pd) * len(bm_hc)) - 1
        diff[b_, j] = abs(rw) - abs(rm)
lo, hi = np.percentile(diff, [2.5, 97.5], axis=0)
alpha_b = 0.05 / F.shape[1]
lob, hib = np.percentile(diff, [100 * alpha_b / 2, 100 * (1 - alpha_b / 2)], axis=0)
R['effect_modification'] = dict(n_boot=NB,
                                n_ci_excl_zero=int(((lo > 0) | (hi < 0)).sum()),
                                n_ci_women_larger=int((lo > 0).sum()),
                                n_bonf_ci_excl_zero=int(((lob > 0) | (hib < 0)).sum()),
                                n_point_women_larger=int(sum(abs(per_feature[f]['women']['r']) > abs(per_feature[f]['men']['r'])
                                                             for f in C.BASELINE21)))
for j, f in enumerate(C.BASELINE21):
    per_feature[f]['absr_diff_w_minus_m'] = dict(
        point=abs(per_feature[f]['women']['r']) - abs(per_feature[f]['men']['r']),
        lo=float(lo[j]), hi=float(hi[j]))

# correlation, across features, between the sex effect and the pooled disease effect
r_sex_hc = np.array([per_feature[f]['sex_in_hc']['r'] for f in C.BASELINE21])
r_pooled = np.array([per_feature[f]['pooled']['r'] for f in C.BASELINE21])
r_w = np.array([per_feature[f]['women']['r'] for f in C.BASELINE21])
r_m = np.array([per_feature[f]['men']['r'] for f in C.BASELINE21])
R['across_feature'] = dict(
    spearman_sexhc_vs_pooled=float(stats.spearmanr(r_sex_hc, r_pooled)[0]),
    spearman_women_vs_men=float(stats.spearmanr(r_w, r_m)[0]),
    n_same_sign_w_m=int((np.sign(r_w) == np.sign(r_m)).sum()),
    mean_abs_r_women=float(np.abs(r_w).mean()), mean_abs_r_men=float(np.abs(r_m).mean()),
    mean_abs_r_pooled=float(np.abs(r_pooled).mean()))
R['per_feature'] = per_feature
json.dump(R, open(out, 'w'), indent=1)
R['power_matched_print'] = {k: v for k, v in R['power_matched'].items() if not k.startswith('counts_')}
for k in ('composition', 'holm_counts', 'raw_counts', 'mean_abs_r', 'power_matched_print',
          'effect_modification', 'across_feature'):
    print(k, json.dumps(R[k]))
