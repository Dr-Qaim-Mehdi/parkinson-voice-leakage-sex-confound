"""Revision step 4 — speaker-level label-permutation test with 2,000 replicates and a
PRE-SPECIFIED, label-independent pipeline.

The original test used a hard-coded MLP configuration that was fixed after the results were
known, and reported p at the 1/201 floor.  Here the pipeline is regularised logistic
regression with C = 1 (the simplest member of the model set, specifiable without reference to
any result), evaluated under the identical LOSO loop, with 2,000 speaker-level permutations."""
import json, warnings
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')

df=pd.read_csv('analysis_dataset.csv'); META=json.load(open('dataset_meta.json'))
X=df[META['panel9']].values.astype(float); y=df['status'].values
groups=df['subject'].values
sub_lab=df.groupby('subject')['status'].max()
logo=LeaveOneGroupOut()

def loso_auc(yy):
    s=np.full(len(yy), np.nan)
    for tr,te in logo.split(X,yy,groups):
        p=Pipeline([('sc',StandardScaler()),
                    ('clf',LogisticRegression(C=1.0, max_iter=5000,
                                              class_weight='balanced'))]).fit(X[tr],yy[tr])
        s[te]=p.predict_proba(X[te])[:,1]
    return roc_auc_score(yy,s)

obs=loso_auc(y)
print(f'observed LOSO ROC-AUC (pre-specified LR, C=1): {obs:.4f}', flush=True)
rng=np.random.default_rng(2026); null=[]
N=2000
for i in range(N):
    perm=sub_lab.copy(); perm[:]=rng.permutation(perm.values)
    yy=df['subject'].map(perm).values
    null.append(loso_auc(yy))
    if (i+1)%250==0: print(f'  {i+1}/{N}', flush=True)
null=np.array(null)
exc=int((null>=obs).sum())
res=dict(pipeline='Regularised logistic regression, C = 1, class-weighted (pre-specified)',
         observed_auc=float(obs), n_permutations=N, exceedances=exc,
         p_value=float((exc+1)/(N+1)), null_mean=float(null.mean()),
         null_sd=float(null.std(ddof=1)), null_p95=float(np.percentile(null,95)),
         null_max=float(null.max()))
json.dump(res, open('rev_permutation.json','w'), indent=1)
np.save('rev_null.npy', null)
print(json.dumps(res, indent=1))
