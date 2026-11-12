#!/usr/bin/env python3
"""Secondary analyses for the sex paper: single-feature AUCs by sex, speaker-level AUCs,
cross-sex transfer, confound dose-response, sex-normative scaling, and the Oxford no-F0 run.

Usage: analyse_extra.py DATA OUT_ROOT OUT_JSON [N_BOOT]
"""
import sys, os, json, glob
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

data, root, out = sys.argv[1:4]
NB = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
df = C.load(data)
g = df['id'].to_numpy(); sex = df['gender'].to_numpy(); dx = df['class'].to_numpy()
ids = df['id'].drop_duplicates().to_numpy()
rows_of = {i: np.where(g == i)[0] for i in ids}
sdx = np.array([dx[rows_of[i][0]] for i in ids]); ssex = np.array([sex[rows_of[i][0]] for i in ids])
R = {}


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = stats.rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def anatomy(s, p, d_, x_):
    c = {'hc_f': (d_ == 0) & (x_ == 0), 'hc_m': (d_ == 0) & (x_ == 1),
         'pd_f': (d_ == 1) & (x_ == 0), 'pd_m': (d_ == 1) & (x_ == 1)}

    def pair(neg, pos):
        m = c[neg] | c[pos]
        return auc(d_[m], s[m])
    wm = c['pd_m'].sum() / max(1, (c['pd_m'].sum() + c['pd_f'].sum()))
    o = dict(auc_pooled=auc(d_, s), auc_within_f=pair('hc_f', 'pd_f'), auc_within_m=pair('hc_m', 'pd_m'),
             auc_aligned=pair('hc_f', 'pd_m'), auc_opposed=pair('hc_m', 'pd_f'),
             fpr_m=float(p[c['hc_m']].mean()), fpr_f=float(p[c['hc_f']].mean()),
             sens_m=float(p[c['pd_m']].mean()), sens_f=float(p[c['pd_f']].mean()),
             score_sex_auc_hc=auc(x_[d_ == 0], s[d_ == 0]), score_sex_auc_pd=auc(x_[d_ == 1], s[d_ == 1]))
    o['auc_adj'] = wm * o['auc_within_m'] + (1 - wm) * o['auc_within_f']
    o['aligned_minus_opposed'] = o['auc_aligned'] - o['auc_opposed']
    o['pooled_minus_adj'] = o['auc_pooled'] - o['auc_adj']
    o['fpr_m_minus_f'] = o['fpr_m'] - o['fpr_f']
    o['sens_m_minus_f'] = o['sens_m'] - o['sens_f']
    o['bacc_m'] = (o['sens_m'] + 1 - o['fpr_m']) / 2; o['bacc_f'] = (o['sens_f'] + 1 - o['fpr_f']) / 2
    return o


def spk(vec):
    return np.array([vec[rows_of[i]].mean() for i in ids])


# ------------------------------------------------------------ 1. single-feature AUCs (speaker means)
F, cols = C.feature_matrix(df, 'b21')
Fs = np.array([F[rows_of[i]].mean(axis=0) for i in ids])
sf = {}
for lab, m in (('pooled', np.ones(len(ids), bool)), ('women', ssex == 0), ('men', ssex == 1)):
    a = np.array([auc(sdx[m], Fs[m, j]) for j in range(Fs.shape[1])]); a = np.maximum(a, 1 - a)
    j = int(np.argmax(a))
    sf[lab] = dict(median=float(np.median(a)), mean=float(a.mean()), max=float(a.max()), argmax=cols[j],
                   n_ge_0_70=int((a >= 0.70).sum()), per_feature={c_: float(v) for c_, v in zip(cols, a)})
R['single_feature_auc_b21'] = sf

# ------------------------------------------------------------ 2. speaker-level multivariate AUCs
def load(pattern):
    o = {}
    for fn in sorted(glob.glob(os.path.join(root, pattern))):
        o[os.path.basename(fn)[:-4]] = np.load(fn, allow_pickle=True)
    return o


nested = load('nested/*.npz')
for fs in ('b21', 'f752'):
    res = {}
    for cohort in ('pooled', 'men', 'women'):
        vals = {}
        for k, z in nested.items():
            p_ = k.split('__')
            if p_[0] != fs or p_[1] != cohort or p_[2] != 'dx':
                continue
            full = np.full(len(dx), np.nan); full[z['idx']] = z['proba']
            sp_ids = [i for i in ids if not np.isnan(full[rows_of[i][0]])]
            s_ = np.array([full[rows_of[i]].mean() for i in sp_ids])
            d_ = np.array([dx[rows_of[i][0]] for i in sp_ids]); x_ = np.array([sex[rows_of[i][0]] for i in sp_ids])
            e = dict(all=auc(d_, s_))
            if cohort == 'pooled':
                e['in_men'] = auc(d_[x_ == 1], s_[x_ == 1]); e['in_women'] = auc(d_[x_ == 0], s_[x_ == 0])
            vals.setdefault(p_[4], []).append(e)
        if vals:
            pm = {m: {kk: float(np.mean([v[kk] for v in vs])) for kk in vs[0]} for m, vs in vals.items()}
            res[cohort] = dict(n_models=len(pm), mean={kk: float(np.mean([pm[m][kk] for m in pm])) for kk in next(iter(pm.values()))})
    R[f'speaker_level_auc_{fs}'] = res

# ------------------------------------------------------------ 3. cross-sex transfer
tr = load('transfer/*.npz')
for fs in ('b21', 'f752'):
    o = {}
    for direction in ('men_to_women', 'women_to_men'):
        a = {k.split('__')[2]: auc(z['y'], z['proba']) for k, z in tr.items()
             if k.startswith(f'{fs}__{direction}__')}
        if a:
            o[direction] = dict(n_models=len(a), mean_auc=float(np.mean(list(a.values()))), per_model=a)
    if o:
        R[f'transfer_{fs}'] = o

# ------------------------------------------------------------ 4. confound dose-response
cd = load('confdose/*.npz')
ARMS = ['OR1.0', 'OR2.4', 'OR5.2', 'OR10']
PDMIX = {'OR1.0': (81, 45), 'OR2.4': (54, 72), 'OR5.2': (32, 94), 'OR10': (19, 107)}
if cd:
    per = {a: [] for a in ARMS}
    for k, z in cd.items():
        fs, arm, r = k.split('__')
        idx = z['idx']; d_ = dx[idx]; x_ = sex[idx]
        names = [n[7:] for n in z.files if n.startswith('proba__')]
        st = [anatomy(z['proba__' + n], z['pred__' + n], d_, x_) for n in names]
        per[arm].append({kk: float(np.mean([s_[kk] for s_ in st])) for kk in st[0]})
    cdres = {}
    for a in ARMS:
        if per[a]:
            keys = per[a][0].keys()
            nf, nm = PDMIX[a]
            cdres[a] = dict(n=len(per[a]), pd_f=nf, pd_m=nm, odds_ratio=(nm / nf) / (23 / 41),
                            mean={kk: float(np.mean([p_[kk] for p_ in per[a]])) for kk in keys},
                            sd={kk: float(np.std([p_[kk] for p_ in per[a]], ddof=1)) if len(per[a]) > 1 else 0.0 for kk in keys})
    trend = {}
    done = [a for a in ARMS if per[a]]
    if len(done) >= 3:
        for kk in per[done[0]][0].keys():
            xs = np.concatenate([[ARMS.index(a)] * len(per[a]) for a in done])
            ys = np.concatenate([[p_[kk] for p_ in per[a]] for a in done])
            rho, pv = stats.spearmanr(xs, ys)
            trend[kk] = dict(rho=float(rho), p=float(pv))
    R['confdose_b21'] = dict(arms=cdres, trend=trend)

# ------------------------------------------------------------ 5. sex-normative scaling vs standard
sn = load('sexnorm/*.npz')
if sn:
    rng = np.random.default_rng(C.RNG + 11)
    cells = [[i for i in ids if dx[rows_of[i][0]] == d_ and sex[rows_of[i][0]] == x_] for d_ in (0, 1) for x_ in (0, 1)]
    pairs = []
    for k, z in sn.items():
        fs, _, _, s_, m_ = k.split('__')
        std = nested.get(f'{fs}__pooled__dx__{s_}__{m_}')
        if std is not None:
            pairs.append((m_, s_, (std['proba'], std['pred']), (z['proba'], z['pred'])))
    seeds_done = sorted({p_[1] for p_ in pairs}); models_done = sorted({p_[0] for p_ in pairs})
    keys = None; a_std = []; a_new = []
    for _, _, (ps, qs), (pn, qn) in pairs:
        a_std.append(anatomy(ps, qs, dx, sex)); a_new.append(anatomy(pn, qn, dx, sex))
    keys = list(a_std[0].keys())
    mean_std = {kk: float(np.mean([a[kk] for a in a_std])) for kk in keys}
    mean_new = {kk: float(np.mean([a[kk] for a in a_new])) for kk in keys}
    bs = {kk: np.zeros(NB) for kk in keys}
    for b in range(NB):
        pick = np.concatenate([rng.choice(c_, len(c_)) for c_ in cells])
        rr = np.concatenate([rows_of[i] for i in pick]); d_ = dx[rr]; x_ = sex[rr]
        acc = {kk: [] for kk in keys}
        for _, _, (ps, qs), (pn, qn) in pairs:
            o1 = anatomy(ps[rr], qs[rr], d_, x_); o2 = anatomy(pn[rr], qn[rr], d_, x_)
            for kk in keys:
                acc[kk].append(o2[kk] - o1[kk])
        for kk in keys:
            bs[kk][b] = np.nanmean(acc[kk])
    R['sexnorm_b21'] = dict(n_runs=len(pairs), seeds=seeds_done, n_models=len(models_done),
                            standard=mean_std, normative=mean_new,
                            diff={kk: mean_new[kk] - mean_std[kk] for kk in keys},
                            diff_ci={kk: [float(np.nanpercentile(bs[kk], 2.5)), float(np.nanpercentile(bs[kk], 97.5))] for kk in keys})

# ------------------------------------------------------------ 6. Oxford: with and without the F0 variables
ox = load('oxford/*.npz')
if ox:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    TAB_SEX = {'S01': 'M', 'S34': 'F', 'S44': 'M', 'S20': 'M', 'S24': 'M', 'S26': 'F', 'S08': 'F', 'S39': 'M', 'S33': 'M',
               'S32': 'M', 'S02': 'M', 'S22': 'M', 'S37': 'M', 'S21': 'F', 'S04': 'M', 'S19': 'M', 'S35': 'F', 'S05': 'F',
               'S18': 'M', 'S16': 'M', 'S27': 'M', 'S25': 'M', 'S06': 'F', 'S10': 'F', 'S07': 'F', 'S13': 'M', 'S43': 'M',
               'S17': 'F', 'S42': 'F', 'S50': 'F', 'S49': 'M'}
    oxr = {}
    for fset in ('p9', 'p9_nof0'):
        per_model = {}
        for k, z in ox.items():
            _, fs_, m_ = k.split('__')
            if fs_ != fset:
                continue
            d_ = pd.DataFrame(dict(code=[s[-3:] for s in z['subject']], y=z['y'], pred=z['pred'], proba=z['proba']))
            v = d_.groupby('code').agg(y=('y', 'first'), votes=('pred', 'sum'), n=('pred', 'size'), ps=('proba', 'mean')).reset_index()
            v['vote'] = (v['votes'] * 2 > v['n']).astype(int)
            v['sex'] = v['code'].map(TAB_SEX)
            rec_acc = float((z['pred'] == z['y']).mean())
            rec_bacc = float(((z['pred'][z['y'] == 1] == 1).mean() + (z['pred'][z['y'] == 0] == 0).mean()) / 2)
            hc = v[v.y == 0]; pdx = v[v.y == 1]
            per_model[m_] = dict(rec_acc=rec_acc, rec_bacc=rec_bacc, rec_auc=auc(z['y'], z['proba']),
                                 spk_sens=float(pdx['vote'].mean()), spk_spec=float(1 - hc['vote'].mean()),
                                 spk_auc=auc(v['y'].to_numpy(), v['ps'].to_numpy()),
                                 fp_m=int(hc.loc[hc.sex == 'M', 'vote'].sum()), fp_f=int(hc.loc[hc.sex == 'F', 'vote'].sum()),
                                 fn_m=int((1 - pdx.loc[pdx.sex == 'M', 'vote']).sum()), fn_f=int((1 - pdx.loc[pdx.sex == 'F', 'vote']).sum()))
        if per_model:
            oxr[fset] = dict(n_models=len(per_model), per_model=per_model,
                             total={kk: int(sum(pm[kk] for pm in per_model.values())) for kk in ('fp_m', 'fp_f', 'fn_m', 'fn_f')},
                             mean={kk: float(np.mean([pm[kk] for pm in per_model.values()])) for kk in ('rec_acc', 'rec_bacc', 'rec_auc', 'spk_sens', 'spk_spec', 'spk_auc')})
    R['oxford_loso'] = oxr

json.dump(R, open(out, 'w'), indent=1)
print('single-feature', {k: {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items() if a != 'per_feature'} for k, v in sf.items()})
for k in R:
    if k.startswith('speaker_level') or k.startswith('transfer'):
        print(k, json.dumps(R[k], default=str)[:600])
if 'confdose_b21' in R:
    for a, v in R['confdose_b21']['arms'].items():
        print(a, v['n'], {kk: round(v['mean'][kk], 3) for kk in ('auc_pooled', 'auc_adj', 'auc_within_f', 'auc_within_m', 'auc_aligned', 'auc_opposed', 'fpr_m', 'fpr_f', 'sens_m', 'sens_f', 'score_sex_auc_hc', 'pooled_minus_adj')})
    print('trend', {kk: (round(v['rho'], 2), round(v['p'], 4)) for kk, v in R['confdose_b21']['trend'].items()})
if 'sexnorm_b21' in R:
    s_ = R['sexnorm_b21']
    print('sexnorm runs', s_['n_runs'], s_['seeds'])
    for kk in ('auc_pooled', 'auc_adj', 'auc_within_f', 'auc_within_m', 'auc_aligned', 'auc_opposed', 'fpr_m', 'fpr_f', 'sens_m', 'sens_f', 'fpr_m_minus_f', 'aligned_minus_opposed', 'score_sex_auc_hc'):
        print(f'   {kk:24s} std {s_["standard"][kk]:.3f}  norm {s_["normative"][kk]:.3f}  diff {s_["diff"][kk]:+.3f} [{s_["diff_ci"][kk][0]:+.3f}, {s_["diff_ci"][kk][1]:+.3f}]')
if 'oxford_loso' in R:
    for fset, v in R['oxford_loso'].items():
        print('oxford', fset, v['n_models'], v['total'], {a: round(b, 3) for a, b in v['mean'].items()})
