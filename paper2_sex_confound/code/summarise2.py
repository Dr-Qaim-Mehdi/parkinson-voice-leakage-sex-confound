#!/usr/bin/env python3
"""Summarise the transfer and confound dose-response runs.

Usage: summarise2.py DATA TRANSFER_DIR CONFDOSE_DIR OUT_JSON
"""
import sys, os, json, glob, re
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

data, td, cd, out = sys.argv[1:5]
df = C.load(data)
g = df['id'].to_numpy(); sex = df['gender'].to_numpy(); dx = df['class'].to_numpy()


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = stats.rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


R = {}
SLUG2NAME = {C.slug(n): n for n in C.MODEL_ORDER}

# ------------------------------------------------- transfer
for featset in ('b21', 'f752'):
    per = {}
    for fn in sorted(glob.glob(os.path.join(td, f'{featset}__*__*.npz'))):
        m = re.match(rf'{featset}__(men_to_women|women_to_men)__(.+)\.npz', os.path.basename(fn))
        if not m:
            continue
        z = np.load(fn)
        name = SLUG2NAME[m.group(2)]
        a = float(auc(z['y'], z['proba']))
        sens = float(z['pred'][z['y'] == 1].mean()); spec = float(1 - z['pred'][z['y'] == 0].mean())
        per.setdefault(name, {})[m.group(1)] = dict(auc=a, sens=sens, spec=spec, bacc=(sens + (spec)) / 2)
    if per:
        names = [n for n in C.MODEL_ORDER if n in per and len(per[n]) == 2]
        R[f'{featset}/transfer'] = dict(
            per_model=per, models=names,
            mean_m2w_auc=float(np.mean([per[n]['men_to_women']['auc'] for n in names])),
            mean_w2m_auc=float(np.mean([per[n]['women_to_men']['auc'] for n in names])))

# ------------------------------------------------- confound dose-response
ARMS = ['OR1.0', 'OR2.4', 'OR5.2', 'OR10']
SUBSET = ['Logistic Regression', 'LDA', 'k-NN', 'SVM (RBF)', 'Random Forest', 'XGBoost']
for featset in ('b21',):
    rows = {}
    for fn in sorted(glob.glob(os.path.join(cd, f'{featset}__*__r*.npz'))):
        m = re.match(rf'{featset}__(OR[0-9.]+)__r(\d+)\.npz', os.path.basename(fn))
        arm, rep = m.group(1), int(m.group(2))
        z = np.load(fn)
        idx = z['idx']; y = dx[idx]; sx = sex[idx]
        # realised odds ratio of the drawn cohort
        a = ((y == 1) & (sx == 1)).sum() / 3; b = ((y == 1) & (sx == 0)).sum() / 3
        c_ = ((y == 0) & (sx == 1)).sum() / 3; d = ((y == 0) & (sx == 0)).sum() / 3
        aucs, saucs, fprdiff = [], [], []
        for nm in SUBSET:
            s = z['proba__' + C.slug(nm)]; q = z['pred__' + C.slug(nm)]
            aucs.append(auc(y, s))
            saucs.append(auc(sx, s))                       # does the PD score track sex?
            fprdiff.append(q[(y == 0) & (sx == 1)].mean() - q[(y == 0) & (sx == 0)].mean())
        rows.setdefault(arm, []).append(dict(rep=rep, or_realised=float((a * d) / (b * c_)),
                                             auc=float(np.mean(aucs)), auc_sd=float(np.std(aucs, ddof=1)),
                                             score_sex_auc=float(np.mean(saucs)),
                                             fpr_m_minus_f=float(np.mean(fprdiff))))
    if rows:
        summ = {}
        for arm in ARMS:
            if arm not in rows:
                continue
            v = rows[arm]
            summ[arm] = dict(n=len(v),
                             or_target=float(arm[2:]),
                             auc_mean=float(np.mean([x['auc'] for x in v])),
                             auc_sd=float(np.std([x['auc'] for x in v], ddof=1)) if len(v) > 1 else 0.0,
                             score_sex_auc_mean=float(np.mean([x['score_sex_auc'] for x in v])),
                             fpr_gap_mean=float(np.mean([x['fpr_m_minus_f'] for x in v])),
                             fpr_gap_sd=float(np.std([x['fpr_m_minus_f'] for x in v], ddof=1)) if len(v) > 1 else 0.0)
        # paired differences per draw against the OR1.0 arm with the same rep
        pdiff = {}
        if 'OR1.0' in rows:
            base = {x['rep']: x for x in rows['OR1.0']}
            for arm in ARMS[1:]:
                if arm not in rows:
                    continue
                d_ = [x['auc'] - base[x['rep']]['auc'] for x in rows[arm] if x['rep'] in base]
                if d_:
                    pdiff[arm] = dict(n=len(d_), mean=float(np.mean(d_)),
                                      sd=float(np.std(d_, ddof=1)) if len(d_) > 1 else 0.0)
        R[f'{featset}/confdose'] = dict(per_arm=summ, paired_auc_vs_OR1=pdiff, raw=rows)

json.dump(R, open(out, 'w'), indent=1)
for k, v in R.items():
    if k.endswith('/transfer'):
        print(k, 'm2w', round(v['mean_m2w_auc'], 4), 'w2m', round(v['mean_w2m_auc'], 4), 'n', len(v['models']))
    else:
        for arm, s in v['per_arm'].items():
            print(k, arm, 'n', s['n'], 'auc', round(s['auc_mean'], 4), '+/-', round(s['auc_sd'], 4),
                  'scoresex', round(s['score_sex_auc_mean'], 3), 'fprgap', round(s['fpr_gap_mean'], 3))
        print(' paired vs OR1.0:', json.dumps(v['paired_auc_vs_OR1']))
