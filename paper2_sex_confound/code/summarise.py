#!/usr/bin/env python3
"""Summarise nested evaluations: per-model metrics, within-sex vs pooled, pair-type AUCs,
error structure by sex, with speaker-level bootstrap intervals.

Usage: summarise.py DATA NESTED_DIR OUT_JSON [N_BOOT]
"""
import sys, os, json, glob
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, matthews_corrcoef
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

data, nd, out = sys.argv[1:4]
NB = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
df = C.load(data)
g = df['id'].to_numpy(); sex = df['gender'].to_numpy(); dx = df['class'].to_numpy()


def auc(y, s):
    """Mann-Whitney AUC (ties count one half)."""
    y = np.asarray(y); s = np.asarray(s)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = stats.rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def metrics(y, pred, proba):
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn); spec = tn / (tn + fp)
    return dict(acc=(tp + tn) / len(y), sens=sens, spec=spec, bacc=(sens + spec) / 2,
                mcc=float(matthews_corrcoef(y, pred)), auc=float(auc(y, proba)))


def load_runs(featset, cohort, target):
    runs = {}
    for fn in sorted(glob.glob(os.path.join(nd, f'{featset}__{cohort}__{target}__s*__*.npz'))):
        parts = os.path.basename(fn)[:-4].split('__')
        seed = int(parts[3][1:]); model = parts[4]
        runs[(model, seed)] = np.load(fn)
    return runs


SLUG2NAME = {C.slug(n): n for n in C.MODEL_ORDER}
R = {'n_boot': NB}

# ------------------------------------------------------------------ per-model metric tables
for featset in ('b21', 'f752'):
    for target in ('dx', 'sex'):
        for cohort in ('pooled', 'men', 'women'):
            runs = load_runs(featset, cohort, target)
            if not runs:
                continue
            seeds = sorted({s for _, s in runs})
            per_model = {}
            for slug_ in [C.slug(n) for n in C.MODEL_ORDER]:
                ms = [metrics(runs[(slug_, s)]['y'], runs[(slug_, s)]['pred'], runs[(slug_, s)]['proba'])
                      for s in seeds if (slug_, s) in runs]
                if not ms:
                    continue
                per_model[SLUG2NAME[slug_]] = dict(
                    n_seeds=len(ms),
                    **{k: float(np.mean([m[k] for m in ms])) for k in ms[0]},
                    **{k + '_sd': float(np.std([m[k] for m in ms], ddof=1)) if len(ms) > 1 else 0.0 for k in ms[0]})
            complete = [n for n, v in per_model.items() if v['n_seeds'] == len(seeds)]
            key = f'{featset}/{target}/{cohort}'
            R[key] = dict(seeds=seeds, n_models=len(per_model), n_complete=len(complete), per_model=per_model,
                          mean={k: float(np.mean([per_model[n][k] for n in complete]))
                                for k in ('acc', 'sens', 'spec', 'bacc', 'mcc', 'auc')} if complete else {},
                          best_auc=max(((per_model[n]['auc'], n) for n in complete), default=None))
            if complete:
                no_gb = [n for n in complete if n != 'Gradient Boosting']
                R[key]['mean_excl_gb'] = {k: float(np.mean([per_model[n][k] for n in no_gb]))
                                          for k in ('acc', 'sens', 'spec', 'bacc', 'mcc', 'auc')}

# ------------------------------------------------------------------ pooled-model anatomy by sex
cell = {'hc_f': (dx == 0) & (sex == 0), 'hc_m': (dx == 0) & (sex == 1),
        'pd_f': (dx == 1) & (sex == 0), 'pd_m': (dx == 1) & (sex == 1)}
W_M = cell['pd_m'].sum() / (dx == 1).sum()      # case-weighted (Janes-Pepe) adjustment weights


def anatomy(proba, pred, mask=None):
    """Statistics of one pooled out-of-fold vector, optionally on a bootstrap index array."""
    if mask is None:
        s, p, d_, x_ = proba, pred, dx, sex
    else:
        s, p, d_, x_ = proba[mask], pred[mask], dx[mask], sex[mask]
    c = {'hc_f': (d_ == 0) & (x_ == 0), 'hc_m': (d_ == 0) & (x_ == 1),
         'pd_f': (d_ == 1) & (x_ == 0), 'pd_m': (d_ == 1) & (x_ == 1)}

    def pair(neg, pos):
        m = c[neg] | c[pos]
        return auc(d_[m], s[m])
    a_ff = pair('hc_f', 'pd_f'); a_mm = pair('hc_m', 'pd_m')
    a_fm = pair('hc_f', 'pd_m')      # confound-aligned pairs (female control, male patient)
    a_mf = pair('hc_m', 'pd_f')      # confound-opposed pairs (male control, female patient)
    wm = c['pd_m'].sum() / (c['pd_m'].sum() + c['pd_f'].sum())
    o = dict(auc_pooled=auc(d_, s), auc_within_f=a_ff, auc_within_m=a_mm,
             auc_aligned=a_fm, auc_opposed=a_mf, auc_adj=wm * a_mm + (1 - wm) * a_ff,
             fpr_m=p[c['hc_m']].mean(), fpr_f=p[c['hc_f']].mean(),
             sens_m=p[c['pd_m']].mean(), sens_f=p[c['pd_f']].mean(),
             score_sex_auc_hc=auc(x_[d_ == 0], s[d_ == 0]),   # does the PD score rank male controls higher?
             score_sex_auc_pd=auc(x_[d_ == 1], s[d_ == 1]),
             score_sex_auc_all=auc(x_, s))
    o['aligned_minus_opposed'] = o['auc_aligned'] - o['auc_opposed']
    o['pooled_minus_adj'] = o['auc_pooled'] - o['auc_adj']
    o['fpr_m_minus_f'] = o['fpr_m'] - o['fpr_f']
    o['sens_m_minus_f'] = o['sens_m'] - o['sens_f']
    o['within_m_minus_f'] = o['auc_within_m'] - o['auc_within_f']
    return o


# speaker-level bootstrap, stratified by sex x diagnosis cell, shared across models and seeds
rng = np.random.default_rng(C.RNG + 7)
ids = df['id'].drop_duplicates().to_numpy()
rows_of = {i: np.where(g == i)[0] for i in ids}
sp_sex = {i: sex[rows_of[i][0]] for i in ids}; sp_dx = {i: dx[rows_of[i][0]] for i in ids}
cells_ids = [[i for i in ids if sp_dx[i] == d_ and sp_sex[i] == x_] for d_ in (0, 1) for x_ in (0, 1)]
boot_rows = []
for _ in range(NB):
    pick = np.concatenate([rng.choice(c_, len(c_)) for c_ in cells_ids])
    boot_rows.append(np.concatenate([rows_of[i] for i in pick]))

for featset in ('b21', 'f752'):
    runs = load_runs(featset, 'pooled', 'dx')
    if not runs:
        continue
    seeds = sorted({s for _, s in runs})
    names = [n for n in C.MODEL_ORDER if all((C.slug(n), s) in runs for s in seeds)]
    if not names:
        continue
    per_model = {}
    full = {}
    for n in names:
        vecs = []
        for s in seeds:
            z = runs[(C.slug(n), s)]
            assert (z['idx'] == np.arange(len(dx))).all()
            vecs.append((z['proba'], z['pred']))
        full[n] = vecs
        stats_ = [anatomy(p, q) for p, q in vecs]
        per_model[n] = {k: float(np.mean([s_[k] for s_ in stats_])) for k in stats_[0]}
    keys = list(next(iter(per_model.values())).keys())
    mean = {k: float(np.mean([per_model[n][k] for n in names])) for k in keys}
    # bootstrap the across-model mean of each statistic
    bs = {k: np.zeros(NB) for k in keys}
    for b, rows in enumerate(boot_rows):
        acc = {k: [] for k in keys}
        for n in names:
            for p, q in full[n]:
                o = anatomy(p, q, rows)
                for k in keys:
                    acc[k].append(o[k])
        for k in keys:
            bs[k][b] = np.nanmean(acc[k])
    ci = {k: [float(np.nanpercentile(bs[k], 2.5)), float(np.nanpercentile(bs[k], 97.5))] for k in keys}
    pboot = {k: float(2 * min((bs[k] <= 0).mean(), (bs[k] >= 0).mean()))
             for k in ('aligned_minus_opposed', 'pooled_minus_adj', 'fpr_m_minus_f', 'sens_m_minus_f', 'within_m_minus_f')}
    R[f'{featset}/anatomy'] = dict(seeds=seeds, models=names, per_model=per_model, mean=mean, ci=ci, p_boot=pboot,
                                   n_models_aligned_gt_opposed=int(sum(per_model[n]['aligned_minus_opposed'] > 0 for n in names)),
                                   n_models_fpr_m_gt_f=int(sum(per_model[n]['fpr_m_minus_f'] > 0 for n in names)),
                                   n_models_sens_m_gt_f=int(sum(per_model[n]['sens_m_minus_f'] > 0 for n in names)))

    # speaker-level (majority vote of three recordings) error rates by sex
    sp_rows = [rows_of[i] for i in ids]
    sdx = np.array([sp_dx[i] for i in ids]); ssex = np.array([sp_sex[i] for i in ids])
    spk = {}
    for n in names:
        vals = []
        for p, q in full[n]:
            vote = np.array([int(q[r].sum() >= 2) for r in sp_rows])
            vals.append(dict(fpr_m=vote[(sdx == 0) & (ssex == 1)].mean(), fpr_f=vote[(sdx == 0) & (ssex == 0)].mean(),
                             sens_m=vote[(sdx == 1) & (ssex == 1)].mean(), sens_f=vote[(sdx == 1) & (ssex == 0)].mean()))
        spk[n] = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
    R[f'{featset}/speaker_errors'] = dict(per_model=spk,
                                          mean={k: float(np.mean([spk[n][k] for n in names])) for k in ('fpr_m', 'fpr_f', 'sens_m', 'sens_f')})

    # within-sex trained models versus the pooled model evaluated inside each sex
    wm = load_runs(featset, 'men', 'dx'); ww = load_runs(featset, 'women', 'dx')
    cmp_names = [n for n in names if all((C.slug(n), s) in wm and (C.slug(n), s) in ww for s in seeds)]
    if cmp_names:
        rows_ = {}
        for n in cmp_names:
            am = np.mean([auc(wm[(C.slug(n), s)]['y'], wm[(C.slug(n), s)]['proba']) for s in seeds])
            aw = np.mean([auc(ww[(C.slug(n), s)]['y'], ww[(C.slug(n), s)]['proba']) for s in seeds])
            rows_[n] = dict(pooled=per_model[n]['auc_pooled'], pooled_in_men=per_model[n]['auc_within_m'],
                            pooled_in_women=per_model[n]['auc_within_f'], men_only=float(am), women_only=float(aw))
        R[f'{featset}/within_vs_pooled'] = dict(
            models=cmp_names, per_model=rows_,
            mean={k: float(np.mean([rows_[n][k] for n in cmp_names])) for k in ('pooled', 'pooled_in_men', 'pooled_in_women', 'men_only', 'women_only')},
            n_men_only_ge_pooled=int(sum(rows_[n]['men_only'] >= rows_[n]['pooled'] for n in cmp_names)),
            n_women_only_ge_pooled=int(sum(rows_[n]['women_only'] >= rows_[n]['pooled'] for n in cmp_names)),
            n_men_gt_women=int(sum(rows_[n]['men_only'] > rows_[n]['women_only'] for n in cmp_names)),
            wilcoxon_men_vs_women=float(stats.wilcoxon([rows_[n]['men_only'] for n in cmp_names],
                                                       [rows_[n]['women_only'] for n in cmp_names]).pvalue))

json.dump(R, open(out, 'w'), indent=1)
for k in R:
    if k.endswith('/pooled') or k.endswith('/men') or k.endswith('/women'):
        print(k, 'seeds', R[k]['seeds'], 'models', R[k]['n_complete'], {a: round(b, 4) for a, b in R[k]['mean'].items()})
        if 'mean_excl_gb' in R[k]:
            print('   excl GB', {a: round(b, 4) for a, b in R[k]['mean_excl_gb'].items()})
    elif k.endswith('/anatomy'):
        print(k, {a: round(b, 4) for a, b in R[k]['mean'].items()})
        print('   CI', {a: [round(x, 3) for x in b] for a, b in R[k]['ci'].items() if 'minus' in a}, R[k]['p_boot'])
    elif k.endswith('/within_vs_pooled'):
        print(k, {a: round(b, 4) for a, b in R[k]['mean'].items()}, {a: b for a, b in R[k].items() if a.startswith('n_') or a.startswith('wil')})
    elif k.endswith('/speaker_errors'):
        print(k, {a: round(b, 4) for a, b in R[k]['mean'].items()})
