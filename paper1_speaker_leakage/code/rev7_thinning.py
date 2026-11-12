#!/usr/bin/env python3
"""Repetition-depth (thinning) experiment (Paper 1, Section 2.9 / 3.7 / Table S7, Figure 5).

Each speaker of the primary corpus (Panel-9) is randomly thinned to k recordings (k = 6, 5, 4, 3, 2) and the
Protocol A (record-wise stratified 10-fold) versus Protocol C (leave-one-subject-out) contrast is re-run for a fixed
subset of six classifiers, four random draws per k (one draw at k = 6, which is the full corpus for the 29
six-recording speakers).  Nested tuning follows rev1_oof.py exactly (inner stratified 3-fold for A, inner
stratified-group 3-fold for C).

Run from paper1_speaker_leakage/ :
    python code/rev7_thinning.py [--jobs 4] [--models "Logistic Regression,LDA,k-NN,SVM (RBF),Random Forest,XGBoost"]
Writes replication/thinning_results.csv (one row per k, draw, model) and replication/thinning_summary.json.
The six-classifier subset is a parameter because the manuscript does not name it; the default is the six models
used in the Paper 2 dose-response experiment.
"""
import argparse, json, os, sys, time, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, LeaveOneGroupOut, GridSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'paper2_sex_confound', 'code'))
import common as C   # identical model set and grids to rev1_oof.py (XGBoost subsample 0.8)

warnings.filterwarnings('ignore')
ap = argparse.ArgumentParser()
ap.add_argument('--jobs', type=int, default=4)
ap.add_argument('--models', default='Logistic Regression,LDA,k-NN,SVM (RBF),Random Forest,XGBoost')
ap.add_argument('--draws', type=int, default=4)
ap.add_argument('--out', default='replication')
args = ap.parse_args()
MODELS = args.models.split(',')

df = pd.read_csv('analysis_dataset.csv'); meta = json.load(open('dataset_meta.json'))
X_all = df[meta['panel9']].to_numpy(float); y_all = df['status'].to_numpy(); g_all = df['subject'].to_numpy()


def thin(k, seed):
    rng = np.random.default_rng(1000 * k + seed)
    keep = []
    for s in np.unique(g_all):
        idx = np.where(g_all == s)[0]
        keep += list(rng.choice(idx, size=min(k, len(idx)), replace=False))
    return np.sort(np.array(keep))


def evaluate(k, draw, name):
    est, grid = C.models()[name]
    idx = thin(k, draw); X, y, g = X_all[idx], y_all[idx], g_all[idx]
    res = {}
    for prot in ('A', 'C'):
        if prot == 'A':
            cv = StratifiedKFold(10, shuffle=True, random_state=draw + 1)
            inner = StratifiedKFold(3, shuffle=True, random_state=draw + 1)
        else:
            cv = LeaveOneGroupOut(); inner = StratifiedGroupKFold(3, shuffle=True, random_state=1)
        pr = np.full(len(y), -1); sc = np.full(len(y), np.nan)
        for tr, te in cv.split(X, y, g):
            gs = GridSearchCV(C.pipe(clone(est)), grid, cv=inner, scoring='roc_auc', n_jobs=1, error_score=0.5)
            gs.fit(X[tr], y[tr], groups=g[tr])
            pr[te] = gs.predict(X[te]); sc[te] = gs.predict_proba(X[te])[:, 1]
        res[prot] = (accuracy_score(y, pr), balanced_accuracy_score(y, pr), roc_auc_score(y, sc))
    return dict(k=k, draw=draw, model=name, n=len(y),
                acc_infl=100 * (res['A'][0] - res['C'][0]), bal_infl=100 * (res['A'][1] - res['C'][1]),
                auc_infl=100 * (res['A'][2] - res['C'][2]),
                accA=res['A'][0], accC=res['C'][0], balA=res['A'][1], balC=res['C'][1], aucA=res['A'][2], aucC=res['C'][2])


tasks = [(k, d, m) for k in (6, 5, 4, 3, 2) for d in range(1 if k == 6 else args.draws) for m in MODELS]
t0 = time.time()
rows = Parallel(n_jobs=args.jobs, verbose=5)(delayed(evaluate)(*t) for t in tasks)
os.makedirs(args.out, exist_ok=True)
R = pd.DataFrame(rows); R.to_csv(os.path.join(args.out, 'thinning_results.csv'), index=False)
per_draw = R.groupby(['k', 'draw'])[['acc_infl', 'bal_infl', 'auc_infl']].mean().reset_index()
summ = per_draw.groupby('k')[['acc_infl', 'bal_infl', 'auc_infl']].agg(['mean', 'std'])
summ.to_json(os.path.join(args.out, 'thinning_summary.json'))
print(summ.round(1)); print('seconds', round(time.time() - t0))
