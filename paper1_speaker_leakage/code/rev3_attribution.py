"""Revision step 3 — group-aware OUT-OF-FOLD permutation importance.

The original analysis fitted the final model on all 195 recordings and measured importance on
those same recordings (in-sample), and permuted at recording level, which destroys the
within-speaker block structure.  Here each LOSO fold model is reused, the feature is permuted
BETWEEN SPEAKER BLOCKS (speaker s receives speaker pi(s)'s values for that feature), and the
pooled out-of-fold ROC-AUC is recomputed.  Nothing is measured on data a model saw."""
import json, pickle, warnings
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')

df=pd.read_csv('analysis_dataset.csv'); META=json.load(open('dataset_meta.json'))
P9=META['panel9']; X=df[P9].values.astype(float); y=df['status'].values
sub=df['subject'].values
store=pickle.load(open('rev_oof.pkl','rb'))
subs=np.array(sorted(set(sub)))
rows_by_sub={s:np.where(sub==s)[0] for s in subs}

def oof_auc(models, Xm):
    s=np.full(len(y), np.nan)
    for te, mdl in models:
        s[te]=mdl.predict_proba(Xm[te])[:,1]
    return roc_auc_score(y, s)

def block_permute(Xm, j, perm):
    """Give speaker s the feature-j values of speaker perm(s), matched by position."""
    Xp=Xm.copy()
    for s, t in zip(subs, perm):
        ri, rj = rows_by_sub[s], rows_by_sub[t]
        vals = rj[np.arange(len(ri)) % len(rj)]
        Xp[ri, j] = Xm[vals, j]
    return Xp

OUT={}
for MODEL in ['MLP','SVM (RBF)']:
    models=store[('C_models',MODEL,0)]
    base=oof_auc(models, X)
    rng=np.random.default_rng(7)
    R=200
    drops=np.zeros((R, len(P9)))
    for r in range(R):
        perm=rng.permutation(subs)
        for j in range(len(P9)):
            drops[r, j]=base-oof_auc(models, block_permute(X, j, perm))
    res=pd.DataFrame({'Feature':P9,
                      'importance_mean':drops.mean(0),
                      'importance_sd':drops.std(0, ddof=1)}).sort_values('importance_mean', ascending=False)
    OUT[MODEL]={'baseline_oof_auc':float(base),
                'importance':res.to_dict(orient='records')}
    print(f'\n{MODEL}: baseline out-of-fold AUC = {base:.4f}')
    print(res.round(4).to_string(index=False))
    res.to_csv(f'rev_oof_importance_{MODEL.split()[0]}.csv', index=False)

json.dump(OUT, open('rev_oof_importance.json','w'), indent=1)
print('\nsaved')
