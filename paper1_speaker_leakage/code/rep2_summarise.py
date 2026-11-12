#!/usr/bin/env python3
"""Summarise the replication-cohort runs (Paper 1, Table 6, Table S8 and the LOSO check of Section 3.6).

Inputs (all from this repository):
  replication/oof/{featset}__A__s{seed}__{model}.npz   record-wise Protocol A   (rep1_protocolA_loso.py)
  replication/oof/b21__C__s1__{model}.npz              leave-one-subject-out      (rep1_protocolA_loso.py)
  ../paper2_sex_confound/oof_predictions/nested/{featset}__pooled__dx__s{seed}__{model}.npz
                                                        subject-wise group 10-fold = speaker-disjoint reference (B)
Metrics are computed on each run's pooled out-of-fold vector and averaged over seeds; inflation is the paired
per-model difference A - B in percentage points.  Writes replication/replication_summary.json.
Run from paper1_speaker_leakage/ :  python code/rep2_summarise.py
"""
import glob, json, os, sys
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, matthews_corrcoef, confusion_matrix
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'paper2_sex_confound', 'code'))
import common as C

B_DIR = os.path.join(HERE, '..', '..', 'paper2_sex_confound', 'oof_predictions', 'nested')
A_DIR = os.path.join(HERE, '..', 'replication', 'oof')
METRICS = ['accuracy', 'balanced_accuracy', 'roc_auc', 'specificity', 'mcc']


def met(y, pred, proba):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(accuracy=accuracy_score(y, pred), balanced_accuracy=balanced_accuracy_score(y, pred),
                roc_auc=roc_auc_score(y, proba), specificity=tn / (tn + fp), mcc=matthews_corrcoef(y, pred))


def collect(pattern_fn, seeds):
    out = {}
    for name in C.MODEL_ORDER:
        ms = []
        for s in seeds:
            fn = pattern_fn(name, s)
            if os.path.exists(fn):
                d = np.load(fn); ms.append(met(d['y'], d['pred'], d['proba']))
        if ms:
            out[name] = {k: float(np.mean([m[k] for m in ms])) for k in METRICS}
    return out


summary = {}
for fs in ('b21', 'f752'):
    A = collect(lambda n, s: os.path.join(A_DIR, f'{fs}__A__s{s}__{C.slug(n)}.npz'), (1, 2, 3))
    B = collect(lambda n, s: os.path.join(B_DIR, f'{fs}__pooled__dx__s{s}__{C.slug(n)}.npz'), (1, 2, 3))
    common = [m for m in C.MODEL_ORDER if m in A and m in B]
    if not common:
        print(f'--- {fs}: no complete runs yet'); continue
    block = dict(n_models=len(common), models=common)
    for k in METRICS:
        a = np.array([A[m][k] for m in common]); b = np.array([B[m][k] for m in common]); d = 100 * (a - b)
        block[k] = dict(A_mean=a.mean(), B_mean=b.mean(), inflation_pp=d.mean(), range_pp=[d.min(), d.max()],
                        models_degraded=int((d > 0).sum()))
    block['per_model_A'] = A; block['per_model_B'] = B
    summary[fs] = block
    print(f'--- {fs}: {len(common)} models (A = record-wise, B = subject-wise group 10-fold)')
    for k in METRICS:
        v = block[k]
        print(f"  {k:18s} A {v['A_mean']:.3f}  B {v['B_mean']:.3f}  inflation {v['inflation_pp']:.1f} pp "
              f"[{v['range_pp'][0]:.1f}, {v['range_pp'][1]:.1f}]  degraded {v['models_degraded']}/{len(common)}")

# leave-one-subject-out check on Baseline-21 against the group 10-fold reference
Cc = collect(lambda n, s: os.path.join(A_DIR, f'b21__C__s1__{C.slug(n)}.npz'), (1,))
Bb = summary['b21']['per_model_B']
common = [m for m in C.MODEL_ORDER if m in Cc and m in Bb]
if common:
    chk = {}
    for k in ('roc_auc', 'accuracy'):
        c = np.array([Cc[m][k] for m in common]); b = np.array([Bb[m][k] for m in common])
        chk[k] = dict(mean_abs_diff=float(np.abs(c - b).mean()), correlation=float(np.corrcoef(c, b)[0, 1]))
    summary['loso_check_b21'] = dict(n_models=len(common), **chk, per_model_C=Cc)
    print('--- LOSO (252 folds) vs group 10-fold, Baseline-21:', {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in chk.items()}, f'({len(common)} models)')
json.dump(summary, open(os.path.join(HERE, '..', 'replication', 'replication_summary.json'), 'w'), indent=1)
