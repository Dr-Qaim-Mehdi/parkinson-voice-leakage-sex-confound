#!/usr/bin/env python3
"""Confound dose-response: vary the sex-diagnosis association by resampling PD speakers.

All 64 controls (41 F / 23 M) are kept in every arm; 126 PD speakers are drawn so that
the male:PD odds ratio is ~1.0, ~2.4 (as observed), ~5.2 or ~10.  Cohort size and class
balance are therefore identical across arms; only the sex composition of the PD group moves.

Usage: run_confdose.py DATA OUT FEATSET N_RESAMPLES
"""
import sys, os, time, warnings
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

warnings.filterwarnings('ignore')
data, out, featset, nres = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
os.makedirs(out, exist_ok=True)
df = C.load(data)
X_all, _ = C.feature_matrix(df, featset)
g_all = df['id'].to_numpy(); sex_all = df['gender'].to_numpy(); dx_all = df['class'].to_numpy()
sp = df.groupby('id')[['gender', 'class']].first()
ids_hc = sp.index[sp['class'] == 0].to_numpy()
ids_pd_f = sp.index[(sp['class'] == 1) & (sp['gender'] == 0)].to_numpy()
ids_pd_m = sp.index[(sp['class'] == 1) & (sp['gender'] == 1)].to_numpy()

ARMS = {'OR1.0': (81, 45), 'OR2.4': (54, 72), 'OR5.2': (32, 94), 'OR10': (19, 107)}  # (PD women, PD men)
SUBSET = ['Logistic Regression', 'LDA', 'k-NN', 'SVM (RBF)', 'Random Forest', 'XGBoost']
CV_SEED = 1
log = open(os.path.join(out, f'log_confdose_{featset}.txt'), 'a')

for r in range(nres):
    for arm, (nf, nm) in ARMS.items():
        fn = os.path.join(out, f'{featset}__{arm}__r{r}.npz')
        if os.path.exists(fn):
            continue
        t0 = time.time()
        rng = np.random.default_rng([C.RNG, r, list(ARMS).index(arm)])
        keep = np.concatenate([ids_hc, rng.choice(ids_pd_f, nf, replace=False),
                               rng.choice(ids_pd_m, nm, replace=False)])
        idx = np.where(np.isin(g_all, keep))[0]
        X, y, g = X_all[idx], dx_all[idx], g_all[idx]
        res = {}
        for name in SUBSET:
            est, grid = C.models()[name]
            outer = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=CV_SEED)
            proba = np.full(len(y), np.nan); pred = np.full(len(y), -1)
            for tr, te in outer.split(X, y, g):
                inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=CV_SEED)
                gs = GridSearchCV(C.pipe(est), grid, cv=inner, scoring='roc_auc', n_jobs=1, error_score=0.5)
                gs.fit(X[tr], y[tr], groups=g[tr])
                proba[te] = gs.predict_proba(X[te])[:, 1]; pred[te] = gs.predict(X[te])
            res['proba__' + C.slug(name)] = proba; res['pred__' + C.slug(name)] = pred
        np.savez(fn, idx=idx, **res)
        msg = f'{featset} {arm} r{r}: {time.time()-t0:.0f}s'
        log.write(msg + '\n'); log.flush(); print(msg, flush=True)
print('DONE confdose', featset, flush=True)
