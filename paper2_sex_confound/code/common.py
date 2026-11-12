"""Shared definitions for the companion-paper analyses (Sakar 2018 corpus).

Model set, grids and protocol conventions are identical to those of the
leakage audit (companion manuscript), so that results are commensurable.
"""
import hashlib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier)
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

RNG = 42
EXPECTED_SHA256 = "9495d1100beaa24005ea951d4b6588186091de4c21cc45009b027f38f699b8ab"

BASELINE21 = ["PPE", "DFA", "RPDE", "numPulses", "numPeriodsPulses",
              "meanPeriodPulses", "stdDevPeriodPulses", "locPctJitter",
              "locAbsJitter", "rapJitter", "ppq5Jitter", "ddpJitter",
              "locShimmer", "locDbShimmer", "apq3Shimmer", "apq5Shimmer",
              "apq11Shimmer", "ddaShimmer", "meanAutoCorrHarmonicity",
              "meanNoiseToHarmHarmonicity", "meanHarmToNoiseHarmonicity"]


def models():
    return {
        'Logistic Regression': (LogisticRegression(max_iter=5000, class_weight='balanced'),
                                {'clf__C': [0.01, 0.1, 1, 10, 100]}),
        'LDA': (LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'), {}),
        'Gaussian NB': (GaussianNB(), {'clf__var_smoothing': [1e-11, 1e-9, 1e-7]}),
        'k-NN': (KNeighborsClassifier(),
                 {'clf__n_neighbors': [1, 3, 5, 7, 11], 'clf__weights': ['uniform', 'distance']}),
        'Decision Tree': (DecisionTreeClassifier(random_state=RNG, class_weight='balanced'),
                          {'clf__max_depth': [2, 3, 5, None], 'clf__min_samples_leaf': [1, 3, 5]}),
        'SVM (RBF)': (SVC(kernel='rbf', probability=True, random_state=RNG, class_weight='balanced'),
                      {'clf__C': [0.1, 1, 10, 100], 'clf__gamma': ['scale', 0.1]}),
        'Random Forest': (RandomForestClassifier(random_state=RNG, n_estimators=200,
                                                 class_weight='balanced', n_jobs=1),
                          {'clf__max_depth': [3, None], 'clf__min_samples_leaf': [1, 3]}),
        'Extra Trees': (ExtraTreesClassifier(random_state=RNG, n_estimators=200,
                                             class_weight='balanced', n_jobs=1),
                        {'clf__max_depth': [3, None], 'clf__min_samples_leaf': [1, 3]}),
        'Gradient Boosting': (GradientBoostingClassifier(random_state=RNG),
                              {'clf__n_estimators': [200], 'clf__learning_rate': [0.05, 0.1],
                               'clf__max_depth': [2, 3]}),
        'XGBoost': (XGBClassifier(random_state=RNG, eval_metric='logloss', tree_method='hist',
                                  n_jobs=1, verbosity=0),
                    {'clf__n_estimators': [300], 'clf__learning_rate': [0.05, 0.1],
                     'clf__max_depth': [2, 3], 'clf__subsample': [0.8]}),
        'MLP': (MLPClassifier(random_state=RNG, max_iter=3000),
                {'clf__hidden_layer_sizes': [(16,), (32, 16)], 'clf__alpha': [1e-3, 1e-1]}),
    }


MODEL_ORDER = list(models().keys())


def pipe(est):
    return Pipeline([('sc', StandardScaler()), ('clf', est)])


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def load(path, check=True):
    if check:
        got = sha256(path)
        if got != EXPECTED_SHA256:
            raise SystemExit(f"SHA-256 mismatch: {got}")
    df = pd.read_csv(path, header=1)
    assert df.shape == (756, 755), df.shape
    assert df['id'].nunique() == 252
    assert (df.groupby('id').size() == 3).all()
    assert df.isna().sum().sum() == 0
    sp = df.groupby('id')[['gender', 'class']].first()
    # documented composition: PD 107 M / 81 F; HC 23 M / 41 F (gender 1 = male)
    assert ((sp['class'] == 1) & (sp['gender'] == 1)).sum() == 107
    assert ((sp['class'] == 1) & (sp['gender'] == 0)).sum() == 81
    assert ((sp['class'] == 0) & (sp['gender'] == 1)).sum() == 23
    assert ((sp['class'] == 0) & (sp['gender'] == 0)).sum() == 41
    return df


def feature_matrix(df, featset):
    if featset == 'b21':
        cols = BASELINE21
    elif featset == 'f752':
        cols = [c for c in df.columns if c not in ('id', 'gender', 'class')]
        assert len(cols) == 752
    else:
        raise ValueError(featset)
    return df[cols].to_numpy(dtype=float), cols


def slug(name):
    return name.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '')
