#!/usr/bin/env python3
"""Cross-sex transfer: tune and train on one sex, test on the other (speakers disjoint by construction).

Usage: run_transfer.py DATA OUT FEATSET
"""
import sys, os, time, warnings
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

warnings.filterwarnings('ignore')
data, out, featset = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
df = C.load(data)
X, _ = C.feature_matrix(df, featset)
g = df['id'].to_numpy(); sex = df['gender'].to_numpy(); y = df['class'].to_numpy()
SEED = 1
for direction, (tr_sex, te_sex) in {'men_to_women': (1, 0), 'women_to_men': (0, 1)}.items():
    tr = np.where(sex == tr_sex)[0]; te = np.where(sex == te_sex)[0]
    for name, (est, grid) in C.models().items():
        if featset == 'f752' and name == 'Gradient Boosting':
            continue
        fn = os.path.join(out, f'{featset}__{direction}__{C.slug(name)}.npz')
        if os.path.exists(fn):
            continue
        t0 = time.time()
        inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
        gs = GridSearchCV(C.pipe(est), grid, cv=inner, scoring='roc_auc', n_jobs=1, error_score=0.5)
        gs.fit(X[tr], y[tr], groups=g[tr])
        np.savez(fn, idx=te, proba=gs.predict_proba(X[te])[:, 1], pred=gs.predict(X[te]), y=y[te])
        print(f'{featset} {direction} {name}: {time.time()-t0:.0f}s', flush=True)
print('DONE transfer', featset, flush=True)
