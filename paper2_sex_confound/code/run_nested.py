#!/usr/bin/env python3
"""Nested, speaker-disjoint (subject-wise group 10-fold) evaluation.

Usage: run_nested.py DATA OUT FEATSET COHORTS SEEDS TARGET [MODELS]
  FEATSET  b21 | f752
  COHORTS  comma list of pooled,men,women
  SEEDS    comma list of integers
  TARGET   dx (PD vs control) | sex (male vs female, pooled only)
  MODELS   optional comma list of model names; default = all (GB skipped on f752)

One checkpoint per (featset, cohort, target, seed, model); safe to re-launch.
"""
import sys, os, time, warnings
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

warnings.filterwarnings('ignore')

data, out, featset, cohorts, seeds, target = sys.argv[1:7]
only = sys.argv[7].split(',') if len(sys.argv) > 7 else None
os.makedirs(out, exist_ok=True)
df = C.load(data)
X_all, cols = C.feature_matrix(df, featset)
g_all = df['id'].to_numpy()
sex_all = df['gender'].to_numpy()
dx_all = df['class'].to_numpy()
y_all = dx_all if target == 'dx' else sex_all

log = open(os.path.join(out, f'log_{featset}_{target}.txt'), 'a')


def say(*a):
    msg = ' '.join(str(x) for x in a)
    log.write(msg + '\n'); log.flush()
    print(msg, flush=True)


for seed in [int(s) for s in seeds.split(',')]:
    for cohort in cohorts.split(','):
        if cohort == 'pooled':
            idx = np.arange(len(df))
        elif cohort == 'men':
            idx = np.where(sex_all == 1)[0]
        elif cohort == 'women':
            idx = np.where(sex_all == 0)[0]
        else:
            raise SystemExit(cohort)
        X, y, g = X_all[idx], y_all[idx], g_all[idx]
        for name, (est, grid) in C.models().items():
            if only and name not in only:
                continue
            if featset == 'f752' and name == 'Gradient Boosting':
                continue  # excluded as in the companion manuscript (prohibitive on 752 features)
            fn = os.path.join(out, f'{featset}__{cohort}__{target}__s{seed}__{C.slug(name)}.npz')
            if os.path.exists(fn):
                continue
            t0 = time.time()
            outer = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
            proba = np.full(len(y), np.nan)
            pred = np.full(len(y), -1)
            fold = np.full(len(y), -1)
            for k, (tr, te) in enumerate(outer.split(X, y, g)):
                assert not set(g[tr]) & set(g[te])
                inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
                gs = GridSearchCV(C.pipe(est), grid, cv=inner, scoring='roc_auc',
                                  n_jobs=1, error_score=0.5)
                gs.fit(X[tr], y[tr], groups=g[tr])
                proba[te] = gs.predict_proba(X[te])[:, 1]
                pred[te] = gs.predict(X[te])
                fold[te] = k
            assert not np.isnan(proba).any() and (pred >= 0).all()
            np.savez(fn, idx=idx, proba=proba, pred=pred, fold=fold, y=y)
            say(f'{featset} {cohort} {target} s{seed} {name}: {time.time()-t0:.0f}s')
say('DONE', featset, cohorts, seeds, target)
