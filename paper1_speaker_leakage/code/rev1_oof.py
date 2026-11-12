"""Revision step 1 — regenerate out-of-fold predictions for Protocol A and Protocol C
on Panel-9 for all eleven classifiers, with the XGBoost specification HARMONISED
(subsample=0.8 everywhere, as in the main experiment).  Stores OOF predictions so that
every downstream revision analysis uses the same, reproducible pipeline."""
import json, pickle, time, warnings
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (StratifiedKFold, StratifiedGroupKFold,
                                     LeaveOneGroupOut, GridSearchCV)
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, recall_score,
                             precision_score, f1_score, matthews_corrcoef,
                             roc_auc_score, confusion_matrix)
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
RNG, NJ = 42, 2
df = pd.read_csv('analysis_dataset.csv')
META = json.load(open('dataset_meta.json'))
P9 = META['panel9']
X = df[P9].values.astype(float); y = df['status'].values; groups = df['subject'].values

def models():
    return {
     'Logistic Regression': (LogisticRegression(max_iter=5000, class_weight='balanced'),
        {'clf__C':[0.01,0.1,1,10,100]}),
     'LDA': (LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'), {}),
     'Gaussian NB': (GaussianNB(), {'clf__var_smoothing':[1e-11,1e-9,1e-7]}),
     'k-NN': (KNeighborsClassifier(),
        {'clf__n_neighbors':[1,3,5,7,11],'clf__weights':['uniform','distance']}),
     'Decision Tree': (DecisionTreeClassifier(random_state=RNG, class_weight='balanced'),
        {'clf__max_depth':[2,3,5,None],'clf__min_samples_leaf':[1,3,5]}),
     'SVM (RBF)': (SVC(kernel='rbf', probability=True, random_state=RNG, class_weight='balanced'),
        {'clf__C':[0.1,1,10,100],'clf__gamma':['scale',0.1]}),
     'Random Forest': (RandomForestClassifier(random_state=RNG, n_estimators=200, class_weight='balanced'),
        {'clf__max_depth':[3,None],'clf__min_samples_leaf':[1,3]}),
     'Extra Trees': (ExtraTreesClassifier(random_state=RNG, n_estimators=200, class_weight='balanced'),
        {'clf__max_depth':[3,None],'clf__min_samples_leaf':[1,3]}),
     'Gradient Boosting': (GradientBoostingClassifier(random_state=RNG),
        {'clf__n_estimators':[200],'clf__learning_rate':[0.05,0.1],'clf__max_depth':[2,3]}),
     # HARMONISED: subsample is part of the grid here exactly as in the main experiment
     'XGBoost': (XGBClassifier(random_state=RNG, eval_metric='logloss', tree_method='hist',
                               n_jobs=1, verbosity=0),
        {'clf__n_estimators':[300],'clf__learning_rate':[0.05,0.1],
         'clf__max_depth':[2,3],'clf__subsample':[0.8]}),
     'MLP': (MLPClassifier(random_state=RNG, max_iter=3000),
        {'clf__hidden_layer_sizes':[(16,),(32,16)],'clf__alpha':[1e-3,1e-1]}),
    }

def metrics(yt, yp, ys):
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0,1]).ravel()
    out = dict(accuracy=accuracy_score(yt,yp),
               balanced_accuracy=balanced_accuracy_score(yt,yp),
               sensitivity=recall_score(yt,yp,zero_division=0),
               specificity=tn/(tn+fp) if (tn+fp) else np.nan,
               precision=precision_score(yt,yp,zero_division=0),
               f1=f1_score(yt,yp,zero_division=0),
               mcc=matthews_corrcoef(yt,yp),
               tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))
    try: out['roc_auc']=roc_auc_score(yt,ys)
    except Exception: out['roc_auc']=np.nan
    return out

def score_of(est, Xt):
    return est.predict_proba(Xt)[:,1] if hasattr(est,'predict_proba') else est.decision_function(Xt)

SEEDS=[1,2,3]
store={}; rows=[]
t0=time.time()
for pname, outer_f, inner_f, seeds in [
        ('A', lambda s: StratifiedKFold(10, shuffle=True, random_state=s),
              lambda s: StratifiedKFold(3, shuffle=True, random_state=s), SEEDS),
        ('C', lambda s: LeaveOneGroupOut(),
              lambda s: StratifiedGroupKFold(3, shuffle=True, random_state=s), [1])]:
    for mname,(est,grid) in models().items():
        for rep,seed in enumerate(seeds):
            cv=outer_f(seed)
            op=np.full(len(y), np.nan); os_=np.full(len(y), np.nan)
            fitted=[]
            for tr,te in cv.split(X,y,groups):
                pipe=Pipeline([('sc',StandardScaler()),('clf',clone(est))])
                if grid:
                    gs=GridSearchCV(pipe,grid,cv=inner_f(seed),scoring='roc_auc',
                                    n_jobs=NJ,error_score=0.5)
                    gs.fit(X[tr],y[tr],groups=groups[tr]); best=gs.best_estimator_
                else:
                    best=pipe.fit(X[tr],y[tr])
                op[te]=best.predict(X[te]); os_[te]=score_of(best,X[te])
                if pname=='C': fitted.append((te,best))
            m=metrics(y,op.astype(int),os_); m.update(protocol=pname,model=mname,repeat=rep)
            rows.append(m)
            store[(pname,mname,rep)]=(op.copy(),os_.copy())
            if pname=='C': store[('C_models',mname,0)]=fitted
        print(f'  [{pname}] {mname:20s} done  t={time.time()-t0:.0f}s', flush=True)

pd.DataFrame(rows).to_csv('rev_results_panel9.csv', index=False)
pickle.dump(store, open('rev_oof.pkl','wb'))
print('TOTAL', time.time()-t0)
