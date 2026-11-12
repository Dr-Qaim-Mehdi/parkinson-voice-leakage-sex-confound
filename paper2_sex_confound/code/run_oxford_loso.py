#!/usr/bin/env python3
"""Oxford corpus, leave-one-subject-out with nested grouped tuning, identical to the audit pipeline
(inner StratifiedGroupKFold(3, shuffle=True, random_state=1)), for a chosen variable set.

Usage: run_oxford_loso.py ANALYSIS_DATASET.csv OUT_DIR SET     SET = p9 | p9_nof0
"""
import sys, os, time, warnings
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, LeaveOneGroupOut
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

warnings.filterwarnings('ignore')
data, out, fset = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
P9 = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
      'spread1', 'spread2', 'D2', 'PPE']
cols = P9 if fset == 'p9' else [c for c in P9 if not c.startswith('MDVP:F')]
df = pd.read_csv(data)
X = df[cols].values.astype(float); y = df['status'].values; g = df['subject'].values
for name, (est, grid) in C.models().items():
    fn = os.path.join(out, f'oxford__{fset}__{C.slug(name)}.npz')
    if os.path.exists(fn):
        continue
    t0 = time.time()
    pred = np.full(len(y), -1); proba = np.full(len(y), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        pipe = Pipeline([('sc', StandardScaler()), ('clf', clone(est))])
        if grid:
            gs = GridSearchCV(pipe, grid, cv=StratifiedGroupKFold(3, shuffle=True, random_state=1),
                              scoring='roc_auc', n_jobs=1, error_score=0.5)
            gs.fit(X[tr], y[tr], groups=g[tr]); best = gs.best_estimator_
        else:
            best = pipe.fit(X[tr], y[tr])
        pred[te] = best.predict(X[te]); proba[te] = best.predict_proba(X[te])[:, 1]
    np.savez(fn, pred=pred, proba=proba, y=y, subject=g, cols=np.array(cols))
    print(f'oxford {fset} {name}: {time.time()-t0:.0f}s', flush=True)
print('DONE oxford', fset, flush=True)
