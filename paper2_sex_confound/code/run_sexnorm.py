#!/usr/bin/env python3
"""Mitigation experiment: sex-normative scaling.

Each feature is expressed as a z-score relative to healthy controls of the same sex,
with the reference mean and SD estimated on the training partition only (inside every
outer and inner fold).  Everything else is identical to run_nested.py (pooled cohort,
subject-wise stratified group 10-fold, nested group 3-fold tuning).

Usage: run_sexnorm.py DATA OUT FEATSET SEEDS
"""
import sys, os, time, warnings
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

warnings.filterwarnings('ignore')


class SexNormativeScaler(BaseEstimator, TransformerMixin):
    """Column 0 of X is sex (0/1); remaining columns are features."""

    def fit(self, X, y):
        sex = X[:, 0].astype(int); F = X[:, 1:]
        self.mu_ = {}; self.sd_ = {}
        for s in (0, 1):
            ref = F[(sex == s) & (np.asarray(y) == 0)]
            if len(ref) < 6:                       # safeguard; never triggered in this cohort
                ref = F[np.asarray(y) == 0]
            self.mu_[s] = ref.mean(axis=0)
            sd = ref.std(axis=0, ddof=1); sd[sd == 0] = 1.0
            self.sd_[s] = sd
        return self

    def transform(self, X):
        sex = X[:, 0].astype(int); F = X[:, 1:]
        Z = np.empty_like(F, dtype=float)
        for s in (0, 1):
            m = sex == s
            Z[m] = (F[m] - self.mu_[s]) / self.sd_[s]
        return Z


data, out, featset, seeds = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
df = C.load(data)
Xf, _ = C.feature_matrix(df, featset)
g = df['id'].to_numpy(); sex = df['gender'].to_numpy(); y = df['class'].to_numpy()
X = np.column_stack([sex.astype(float), Xf])
for seed in [int(s) for s in seeds.split(',')]:
    for name, (est, grid) in C.models().items():
        if featset == 'f752' and name == 'Gradient Boosting':
            continue
        fn = os.path.join(out, f'{featset}__sexnorm__dx__s{seed}__{C.slug(name)}.npz')
        if os.path.exists(fn):
            continue
        t0 = time.time()
        outer = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
        proba = np.full(len(y), np.nan); pred = np.full(len(y), -1)
        for tr, te in outer.split(X, y, g):
            inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
            pipe = Pipeline([('norm', SexNormativeScaler()), ('sc', StandardScaler()), ('clf', est)])
            gs = GridSearchCV(pipe, grid, cv=inner, scoring='roc_auc', n_jobs=1, error_score=0.5)
            gs.fit(X[tr], y[tr], groups=g[tr])
            proba[te] = gs.predict_proba(X[te])[:, 1]; pred[te] = gs.predict(X[te])
        np.savez(fn, idx=np.arange(len(y)), proba=proba, pred=pred, y=y)
        print(f'{featset} sexnorm s{seed} {name}: {time.time()-t0:.0f}s', flush=True)
print('DONE sexnorm', featset, flush=True)
