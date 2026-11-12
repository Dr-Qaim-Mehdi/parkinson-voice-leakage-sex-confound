"""Revision step 2 — speaker-level aggregation for all eleven models (harmonised pipelines),
exact binomial intervals, calibration, the F0-band sex-proxy analysis, the bootstrap interval
on the paired A-C difference, predictive values / likelihood ratios, and the fraction of
above-chance skill lost."""
import json, pickle, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, recall_score,
                             precision_score, f1_score, matthews_corrcoef, roc_auc_score,
                             confusion_matrix, brier_score_loss)
import statsmodels.api as sm
warnings.filterwarnings('ignore')

df = pd.read_csv('analysis_dataset.csv'); META=json.load(open('dataset_meta.json'))
y = df['status'].values; sub = df['subject'].values
store = pickle.load(open('rev_oof.pkl','rb'))
MODELS = ['Logistic Regression','LDA','Gaussian NB','k-NN','Decision Tree','SVM (RBF)',
          'Random Forest','Extra Trees','Gradient Boosting','XGBoost','MLP']
OUT = {}

def ci_exact(k, n):
    lo, hi = stats.beta.ppf(0.025, k, n-k+1) if k>0 else 0.0, \
             stats.beta.ppf(0.975, k+1, n-k) if k<n else 1.0
    return float(lo), float(hi)

def calib(yt, p):
    p = np.clip(p, 1e-6, 1-1e-6); lo = np.log(p/(1-p))
    slope = float(sm.Logit(yt, sm.add_constant(lo)).fit(disp=0).params[1])
    inter = float(sm.GLM(yt, np.ones((len(yt),1)), family=sm.families.Binomial(),
                         offset=lo).fit().params[0])
    return dict(brier=float(brier_score_loss(yt,p)), slope=slope, intercept=inter)

# ---------------- 1. speaker-level aggregation, all 11 models ----------------
rows=[]
for m in MODELS:
    op, os_ = store[('C', m, 0)]
    d = pd.DataFrame({'subject':sub,'y':y,'p':op,'s':os_})
    g = d.groupby('subject').agg(y=('y','max'), maj=('p','mean'), mean_s=('s','mean'))
    for rule, pred in [('Majority vote',(g['maj']>0.5).astype(int)),
                       ('Mean probability',(g['mean_s']>0.5).astype(int))]:
        yt, yp = g['y'].values, pred.values
        tn,fp,fn,tp = confusion_matrix(yt,yp,labels=[0,1]).ravel()
        sens_lo,sens_hi = ci_exact(tp, tp+fn); spec_lo,spec_hi = ci_exact(tn, tn+fp)
        acc_lo,acc_hi   = ci_exact(tn+tp, len(yt))
        rows.append(dict(model=m, rule=rule,
            accuracy=accuracy_score(yt,yp), acc_lo=acc_lo, acc_hi=acc_hi,
            balanced_accuracy=balanced_accuracy_score(yt,yp),
            sensitivity=recall_score(yt,yp,zero_division=0), sens_lo=sens_lo, sens_hi=sens_hi,
            specificity=tn/(tn+fp), spec_lo=spec_lo, spec_hi=spec_hi,
            precision=precision_score(yt,yp,zero_division=0),
            f1=f1_score(yt,yp,zero_division=0), mcc=matthews_corrcoef(yt,yp),
            roc_auc=roc_auc_score(yt,g['mean_s'].values),
            tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp)))
SL=pd.DataFrame(rows); SL.to_csv('rev_speaker_level_all11.csv', index=False)
print('speaker-level, all 11 models:'); print(SL[['model','rule','accuracy','balanced_accuracy',
      'sensitivity','sens_lo','sens_hi','specificity','spec_lo','spec_hi','mcc','roc_auc']].round(3).to_string(index=False))

# ---------------- 2. calibration ----------------
cal=[]
for m in MODELS:
    op, os_ = store[('C', m, 0)]
    c_rec = calib(y, os_)
    d = pd.DataFrame({'subject':sub,'y':y,'s':os_})
    g = d.groupby('subject').agg(y=('y','max'), s=('s','mean'))
    c_spk = calib(g['y'].values, g['s'].values)
    cal.append(dict(model=m, brier_rec=c_rec['brier'], slope_rec=c_rec['slope'],
                    intercept_rec=c_rec['intercept'], brier_spk=c_spk['brier'],
                    slope_spk=c_spk['slope'], intercept_spk=c_spk['intercept']))
CAL=pd.DataFrame(cal); CAL.to_csv('rev_calibration.csv', index=False)
print('\ncalibration (LOSO, Panel-9):'); print(CAL.round(3).to_string(index=False))

# ---------------- 3. F0-band sex-proxy analysis ----------------
sm_ = df.groupby(['subject','status'])[META['panel9']].mean().reset_index()
hc = np.sort(sm_.loc[sm_.status==0,'MDVP:Fo(Hz)'].values)
gaps = np.diff(hc); cut = (hc[np.argmax(gaps)] + hc[np.argmax(gaps)+1])/2
sm_['band'] = np.where(sm_['MDVP:Fo(Hz)'] < cut, 'low', 'high')
ct = pd.crosstab(sm_.band, sm_.status)
orr, pf = stats.fisher_exact(ct.values)
strat={'cut_hz':float(cut),'largest_gap_hz':float(gaps.max()),
       'hc_f0_sorted':[round(float(v),1) for v in hc],
       'crosstab':ct.to_dict(), 'fisher_or':float(orr), 'fisher_p':float(pf)}
for b in ['low','high']:
    d=sm_[sm_.band==b]; h=d[d.status==0]['MDVP:Fo(Hz)']; p_=d[d.status==1]['MDVP:Fo(Hz)']
    u,pp=stats.mannwhitneyu(h,p_)
    strat[f'{b}_band']={'n_hc':int(len(h)),'n_pd':int(len(p_)),'hc_mean':float(h.mean()),
                        'pd_mean':float(p_.mean()),'p':float(pp)}
feat_hi={}
d=sm_[sm_.band=='high']
for f in META['panel9']:
    h=d[d.status==0][f]; p_=d[d.status==1][f]; u,pp=stats.mannwhitneyu(h,p_)
    feat_hi[f]={'hc':float(h.mean()),'pd':float(p_.mean()),'p':float(pp)}
strat['high_band_features']=feat_hi
# pitch-range proxy
sm_['range']=sm_['MDVP:Fhi(Hz)']-sm_['MDVP:Flo(Hz)']
h=sm_[sm_.status==0]['range']; p_=sm_[sm_.status==1]['range']; u,pp=stats.mannwhitneyu(h,p_)
strat['pitch_range']={'hc':float(h.mean()),'pd':float(p_.mean()),'p':float(pp)}
# LOSO error structure by band
band=dict(zip(sm_.subject, sm_.band)); err={}
for m in MODELS:
    op,_=store[('C',m,0)]
    g=pd.DataFrame({'subject':sub,'y':y,'p':op}).groupby('subject').agg(y=('y','max'),maj=('p','mean'))
    g['pred']=(g['maj']>0.5).astype(int); g['band']=g.index.map(band)
    h=g[g.y==0]
    err[m]={'low_fp':int(((h.band=='low')&(h.pred==1)).sum()),'n_low':int((h.band=='low').sum()),
            'high_fp':int(((h.band=='high')&(h.pred==1)).sum()),'n_high':int((h.band=='high').sum())}
strat['loso_error_structure']=err
OUT['stratified']=strat
print(f"\nF0 cut {cut:.1f} Hz (largest gap {gaps.max():.1f} Hz); Fisher OR={orr:.2f} p={pf:.4f}")
print('  HC F0 sorted:', strat['hc_f0_sorted'])
for b in ['low','high']: print(f"  {b}: {strat[f'{b}_band']}")
print('  error structure:', {k:f"{v['low_fp']}/{v['n_low']} low, {v['high_fp']}/{v['n_high']} high" for k,v in err.items()})

# ---------------- 4. bootstrap CI on the paired A-C difference ----------------
subs=np.array(sorted(set(sub))); rng=np.random.default_rng(2026)
idx_by_sub={s:np.where(sub==s)[0] for s in subs}
boot={}
for m in MODELS:
    pA=np.mean([store[('A',m,r)][0] for r in range(3)],axis=0)  # mean over repeats
    sA=np.mean([store[('A',m,r)][1] for r in range(3)],axis=0)
    pC,sC=store[('C',m,0)]
    diffs={'accuracy':[], 'balanced_accuracy':[], 'specificity':[], 'roc_auc':[]}
    for _ in range(2000):
        pick=rng.choice(subs,size=len(subs),replace=True)
        ii=np.concatenate([idx_by_sub[s] for s in pick])
        yt=y[ii]
        if len(set(yt))<2: continue
        aA=(pA[ii]>0.5).astype(int); aC=pC[ii].astype(int)
        tnA,fpA,_,_=confusion_matrix(yt,aA,labels=[0,1]).ravel()
        tnC,fpC,_,_=confusion_matrix(yt,aC,labels=[0,1]).ravel()
        diffs['accuracy'].append(accuracy_score(yt,aA)-accuracy_score(yt,aC))
        diffs['balanced_accuracy'].append(balanced_accuracy_score(yt,aA)-balanced_accuracy_score(yt,aC))
        diffs['specificity'].append(tnA/(tnA+fpA)-tnC/(tnC+fpC))
        diffs['roc_auc'].append(roc_auc_score(yt,sA[ii])-roc_auc_score(yt,sC[ii]))
    boot[m]={k:[float(np.percentile(v,2.5)),float(np.percentile(v,50)),float(np.percentile(v,97.5))]
             for k,v in diffs.items()}
OUT['bootstrap_AC']=boot
print('\nbootstrap 95% CI on the A-C difference (speaker-level resampling):')
for m in MODELS:
    b=boot[m]['balanced_accuracy']
    print(f'  {m:20s} bal.acc +{b[1]*100:5.1f} pp  [{b[0]*100:+.1f}, {b[2]*100:+.1f}]')

# ---------------- 5. fraction of above-chance skill lost ----------------
r=pd.read_csv('rev_results_panel9.csv')
g2=r.groupby('protocol')[['accuracy','balanced_accuracy','roc_auc','mcc']].mean()
base=META['n_pd_records']/META['n_records']
frac={}
for met,ch in [('balanced_accuracy',0.5),('roc_auc',0.5),('mcc',0.0),('accuracy',base)]:
    A,C=g2.loc['A',met],g2.loc['C',met]
    frac[met]=float((A-C)/(A-ch)*100)
OUT['frac_above_chance_lost_panel9']=frac
print('\nfraction of above-chance skill lost (Panel-9):',
      {k:round(v,1) for k,v in frac.items()})

# ---------------- 6. predictive values and likelihood ratios ----------------
def pv(sens,spec,prev):
    ppv=sens*prev/(sens*prev+(1-spec)*(1-prev))
    npv=spec*(1-prev)/((1-sens)*prev+spec*(1-prev))
    return ppv,npv
ops=[('Balanced (logistic regression, majority vote)',0.750,0.750),
     ('High-sensitivity (XGBoost, majority vote)',1.000,0.500)]
prevs=[('World, all ages, 2050 projection',0.00267),('Unselected population aged 65+',0.01),
       ('Unselected population aged 80+',0.025),('New movement-disorder referral',0.40),
       ('Suspected parkinsonism',0.60)]
pvt=[]
for lab,se,sp in ops:
    lrp = se/(1-sp) if sp<1 else np.inf
    lrn = (1-se)/sp if sp>0 else np.inf
    for pl,pr in prevs:
        ppv,npv=pv(se,sp,pr)
        pvt.append(dict(operating_point=lab,sens=se,spec=sp,setting=pl,prevalence=pr,
                        ppv=ppv,npv=npv,lr_pos=lrp,lr_neg=lrn,
                        false_alarms_per_true_case=((1-sp)*(1-pr))/(se*pr)))
PV=pd.DataFrame(pvt); PV.to_csv('rev_predictive_values.csv', index=False)
print('\npredictive values:')
print(PV[['operating_point','setting','ppv','lr_pos','lr_neg','false_alarms_per_true_case']].round(3).to_string(index=False))

json.dump(OUT, open('rev_analyses.json','w'), indent=1)
print('\nsaved rev_analyses.json')
