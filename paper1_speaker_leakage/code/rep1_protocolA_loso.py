#!/usr/bin/env python3
"""Replication-cohort (Sakar, 252 speakers) record-wise and leave-one-subject-out runs (Paper 1, Section 3.6).

Usage: rep1_protocolA_loso.py DATA OUT FEATSET PROTOCOL SEEDS [MODELS]
  FEATSET   b21 | f752
  PROTOCOL  A  record-wise stratified 10-fold, inner stratified 3-fold (seeds as given)
            C  leave-one-subject-out (252 outer folds), inner stratified-group 3-fold (seed 1)
            (the speaker-disjoint Protocol B reference is paper2_sex_confound/code/run_nested.py)
  SEEDS     comma list (Protocol C is deterministic: pass 1)
  MODELS    optional comma list of model names; default all (gradient boosting skipped on f752;
            Protocol C is run on b21 only, as in the manuscript)
One checkpoint (.npz) per (featset, protocol, seed, model); safe to re-launch.  Model set, grids,
standardisation inside the pipeline and nested ROC-AUC tuning are shared with the Paper 2 runs
(paper2_sex_confound/code/common.py), so Protocols A, B and C are commensurable.
"""
import sys, os, time, warnings
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedGroupKFold, LeaveOneGroupOut
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'paper2_sex_confound', 'code'))
import common as C
warnings.filterwarnings('ignore')

data, out, featset, protocol, seeds = sys.argv[1:6]
only = sys.argv[6].split(',') if len(sys.argv) > 6 else None
if protocol == 'C' and featset != 'b21':
    raise SystemExit('Protocol C was run on Baseline-21 only')
os.makedirs(out, exist_ok=True)
df = C.load(data)
X, cols = C.feature_matrix(df, featset)
y = df['class'].to_numpy(); g = df['id'].to_numpy()

for seed in [int(s) for s in seeds.split(',')]:
    for name, (est, grid) in C.models().items():
        if only and name not in only:
            continue
        if featset == 'f752' and name == 'Gradient Boosting':
            continue
        fn = os.path.join(out, f'{featset}__{protocol}__s{seed}__{C.slug(name)}.npz')
        if os.path.exists(fn):
            continue
        t0 = time.time()
        if protocol == 'A':
            outer = StratifiedKFold(10, shuffle=True, random_state=seed)
            inner = StratifiedKFold(3, shuffle=True, random_state=seed)
            splits = outer.split(X, y)
        else:
            splits = LeaveOneGroupOut().split(X, y, g)
            inner = StratifiedGroupKFold(3, shuffle=True, random_state=seed)
        proba = np.full(len(y), np.nan); pred = np.full(len(y), -1); fold = np.full(len(y), -1)
        for k, (tr, te) in enumerate(splits):
            if protocol == 'C':
                assert not set(g[tr]) & set(g[te])
            gs = GridSearchCV(C.pipe(est), grid, cv=inner, scoring='roc_auc', n_jobs=1, error_score=0.5)
            gs.fit(X[tr], y[tr], groups=g[tr]) if protocol == 'C' else gs.fit(X[tr], y[tr])
            proba[te] = gs.predict_proba(X[te])[:, 1]; pred[te] = gs.predict(X[te]); fold[te] = k
        assert not np.isnan(proba).any() and (pred >= 0).all()
        np.savez(fn, idx=np.arange(len(y)), proba=proba, pred=pred, fold=fold, y=y)
        print(f'{featset} {protocol} s{seed} {name}: {time.time()-t0:.0f}s', flush=True)
print('DONE', featset, protocol, seeds)
