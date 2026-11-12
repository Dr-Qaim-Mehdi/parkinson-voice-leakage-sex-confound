#!/usr/bin/env python3
"""Speaker-identity probe (Paper 1, Section 2.7 / 3.4 / Table 4).

Three analyses, for each feature set of both corpora:
  (i)   speaker re-identification, leave-one-recording-out (1-NN; plus SVM and random forest on the
        primary corpus), against chance 1/n_speakers;
  (ii)  nearest-neighbour analysis: does a recording's nearest neighbour (Euclidean, globally z-scored)
        belong to the same speaker?  Expected rate under random assignment = sum n_s(n_s-1) / (N(N-1));
  (iii) variance decomposition: one-way ICC(1) of every feature with speaker as the grouping factor.

Run from paper1_speaker_leakage/ :
    python code/rev6_speaker_probe.py --sakar /path/to/pd_speech_features.csv
(the Sakar table is the public UCI file; its SHA-256 is checked).  Writes probe_results.json.
Settings that reproduce the manuscript's Table 4 (see replication_notes in README):
  * Oxford corpus: the standardiser is re-fitted inside every leave-one-out fold; Sakar corpus: features are
    z-scored once on the whole table before the leave-one-out loop (differences are below 0.5 percentage points);
  * random forest: 300 trees, random_state 42; SVM: RBF, C = 10, gamma = 'scale';
  * ICC(1): one-way ANOVA estimator (unbalanced-design k0), negative estimates truncated at 0.
Known residual: the Oxford mean ICC(1) comes out 0.002 above the manuscript (0.699 vs 0.697; 0.745 vs 0.743).
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'paper2_sex_confound', 'code'))
import common as C   # Sakar loader with checksum and the Baseline-21 / Full-752 definitions

ap = argparse.ArgumentParser()
ap.add_argument('--sakar', required=True)
ap.add_argument('--out', default='probe_results.json')
ap.add_argument('--skip-heavy', action='store_true', help='skip SVM/RF re-identification')
args = ap.parse_args()


def icc1(X, g):
    """One-way random-effects ICC(1) per column; k0 handles unbalanced groups."""
    g = pd.factorize(g)[0]
    n = np.bincount(g); a = len(n); N = len(g)
    k0 = (N - (n ** 2).sum() / N) / (a - 1)
    out = []
    for j in range(X.shape[1]):
        x = X[:, j]; grand = x.mean()
        gm = np.bincount(g, weights=x) / n
        ssb = (n * (gm - grand) ** 2).sum(); ssw = ((x - gm[g]) ** 2).sum()
        msb = ssb / (a - 1); msw = ssw / (N - a)
        out.append((msb - msw) / (msb + (k0 - 1) * msw) if (msb + (k0 - 1) * msw) > 0 else np.nan)
    return np.clip(np.array(out), 0, None)


def nn_same_speaker(X, g):
    Z = StandardScaler().fit_transform(X)
    D = cdist(Z, Z); np.fill_diagonal(D, np.inf)
    nn = D.argmin(1)
    rate = float((g[nn] == g).mean())
    n = pd.Series(g).value_counts().to_numpy(); N = len(g)
    exp = float((n * (n - 1)).sum() / (N * (N - 1)))
    return rate, exp


def reid(X, g, est, scaling='fold'):
    if scaling == 'global':
        X = StandardScaler().fit_transform(X); pipe = est
    else:
        pipe = make_pipeline(StandardScaler(), est)
    pred = cross_val_predict(pipe, X, g, cv=LeaveOneOut(), n_jobs=-1)
    return float((pred == g).mean())


results = {}
# ---- primary corpus
df = pd.read_csv('analysis_dataset.csv'); meta = json.load(open('dataset_meta.json'))
g_ox = df['subject'].to_numpy()
sets = {'Oxford Panel-9': (df[meta['panel9']].to_numpy(float), g_ox, 'oxford'),
        'Oxford Full-22': (df[meta['full22']].to_numpy(float), g_ox, 'oxford')}
# ---- replication corpus
sk = C.load(args.sakar)
g_sk = sk['id'].to_numpy()
sets['Sakar Baseline-21'] = (C.feature_matrix(sk, 'b21')[0], g_sk, 'sakar')
sets['Sakar Full-752'] = (C.feature_matrix(sk, 'f752')[0], g_sk, 'sakar')

for name, (X, g, corpus) in sets.items():
    nsp = len(np.unique(g)); r = dict(n_speakers=nsp, n_features=X.shape[1], chance=1 / nsp)
    scal = 'fold' if corpus == 'oxford' else 'global'
    r['reid_1nn'] = reid(X, g, KNeighborsClassifier(1), scal)
    if corpus == 'oxford' and not args.skip_heavy:
        r['reid_svm'] = reid(X, g, SVC(kernel='rbf', C=10, gamma='scale'))
        r['reid_rf'] = reid(X, g, RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=1))
    r['nn_same_speaker'], r['nn_expected_random'] = nn_same_speaker(X, g)
    r['multiple_of_chance_1nn'] = r['reid_1nn'] / r['chance']
    ic = icc1(X, g); r['icc1_mean'] = float(np.nanmean(ic)); r['icc1_by_feature'] = [float(v) for v in ic]
    results[name] = r
    print(f"{name:20s} reID-1NN {r['reid_1nn']:.3f} ({r['multiple_of_chance_1nn']:.0f}x chance)  "
          f"NN-same {r['nn_same_speaker']:.3f} vs {r['nn_expected_random']:.4f}  ICC1 mean {r['icc1_mean']:.3f}"
          + (f"  SVM {r['reid_svm']:.3f} RF {r['reid_rf']:.3f}" if 'reid_svm' in r else ''), flush=True)
json.dump(results, open(args.out, 'w'), indent=1)
