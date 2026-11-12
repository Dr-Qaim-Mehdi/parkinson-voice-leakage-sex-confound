"""Revision step 5 — regenerate Figure 7 (confusion matrices, now for the MLP), Figure 8
(out-of-fold group-aware attribution) and the new Figure 9 (sex-proxy analysis)."""
import json, pickle, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
warnings.filterwarnings('ignore')

plt.rcParams.update({
 'figure.dpi':300,'savefig.dpi':300,'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
 'font.size':8,'axes.labelsize':8.5,'axes.titlesize':9,'xtick.labelsize':7.5,'ytick.labelsize':7.5,
 'legend.fontsize':7.5,'axes.linewidth':0.7,'xtick.major.width':0.7,'ytick.major.width':0.7,
 'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight','legend.frameon':False,
 'axes.grid':True,'grid.alpha':0.25,'grid.linewidth':0.5,'grid.color':'#9a9a95',
 'axes.axisbelow':True,'figure.facecolor':'white'})
CA,CB,CC = '#eb6834','#2a78d6','#1baf7a'
INK,INK2 = '#0b0b0b','#52514e'

df=pd.read_csv('analysis_dataset.csv'); META=json.load(open('dataset_meta.json'))
y=df['status'].values; sub=df['subject'].values
store=pickle.load(open('rev_oof.pkl','rb'))
A=json.load(open('rev_analyses.json'))
P9=META['panel9']

# ---------------- Figure 7: confusion matrices, MLP ----------------
op,os_=store[('C','MLP',0)]
cm_rec=confusion_matrix(y, op.astype(int), labels=[0,1])
g=pd.DataFrame({'subject':sub,'y':y,'p':op}).groupby('subject').agg(y=('y','max'),maj=('p','mean'))
g['pred']=(g['maj']>0.5).astype(int)
cm_spk=confusion_matrix(g['y'],g['pred'],labels=[0,1])
fig,axes=plt.subplots(1,2,figsize=(6.4,2.9))
for ax,cm,t in zip(axes,[cm_rec,cm_spk],
      ['(a) Recording level — MLP','(b) Speaker level — MLP, majority vote']):
    ax.imshow(cm/cm.sum(1,keepdims=True),cmap='Blues',vmin=0,vmax=1)
    for i in range(2):
        for j in range(2):
            frac=cm[i,j]/cm[i].sum()
            ax.text(j,i,f'{cm[i,j]}\n({frac*100:.0f}%)',ha='center',va='center',
                    fontsize=8,color='white' if frac>0.55 else INK)
    ax.set_xticks([0,1]); ax.set_xticklabels(['HC','PD'])
    ax.set_yticks([0,1]); ax.set_yticklabels(['HC','PD'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); ax.set_title(t,pad=5,fontsize=8); ax.grid(False)
fig.tight_layout(); fig.savefig('Figure7_confusion_matrices.png'); plt.close(fig)

# ---------------- Figure 8: out-of-fold group-aware attribution ----------------
I=json.load(open('rev_oof_importance.json'))
fig,axes=plt.subplots(1,2,figsize=(7.2,3.1))
for ax,(mdl,col) in zip(axes,[('MLP',CB),('SVM (RBF)',CC)]):
    d=pd.DataFrame(I[mdl]['importance']).sort_values('importance_mean')
    ax.barh(d.Feature,d.importance_mean,xerr=d.importance_sd,height=0.62,color=col,
            edgecolor='white',linewidth=0.8,error_kw=dict(ecolor=INK2,lw=0.8,capsize=2))
    ax.set_xlabel('Loss of out-of-fold ROC-AUC')
    ax.set_title(f'({"ab"[0 if mdl=="MLP" else 1]}) {mdl}  (baseline AUC {I[mdl]["baseline_oof_auc"]:.3f})',
                 pad=5,fontsize=8.5)
    ax.grid(axis='y',visible=False); ax.set_xlim(0,None)
fig.tight_layout(); fig.savefig('Figure8_oof_importance.png'); plt.close(fig)

# ---------------- Figure 9: sex-proxy analysis ----------------
S=A['stratified']; cut=S['cut_hz']
smm=df.groupby(['subject','status'])['MDVP:Fo(Hz)'].mean().reset_index()
fig,axes=plt.subplots(1,2,figsize=(7.2,2.9))
ax=axes[0]
rng=np.random.default_rng(3)
for st,col,lab in [(0,CB,'HC'),(1,CA,'PD')]:
    v=smm.loc[smm.status==st,'MDVP:Fo(Hz)'].values
    ax.scatter(np.full(len(v),st)+rng.uniform(-0.12,0.12,len(v)),v,s=26,color=col,
               alpha=0.9,edgecolor='white',linewidth=0.6,zorder=3,label=lab)
ax.axhline(cut,color=INK2,ls=(0,(3,3)),lw=0.9)
ax.annotate(f'  {cut:.0f} Hz — largest gap in the\n  control group ({S["largest_gap_hz"]:.0f} Hz)',
            (1.42,cut),fontsize=6.4,color=INK2,va='center')
ax.set_xticks([0,1]); ax.set_xticklabels(['HC','PD']); ax.set_xlim(-0.5,2.4)
ax.set_ylabel('Speaker-mean MDVP:Fo (Hz)')
ax.set_title('(a) Fundamental frequency is bimodal in the controls',pad=5,fontsize=8.5)
ax.grid(axis='x',visible=False)

ax=axes[1]
E=S['loso_error_structure']
models=['Logistic Regression','LDA','Gaussian NB','k-NN','Decision Tree','SVM (RBF)',
        'Random Forest','Extra Trees','Gradient Boosting','XGBoost','MLP']
lowr=[E[m]['low_fp']/E[m]['n_low'] for m in models]
higr=[E[m]['high_fp']/E[m]['n_high'] for m in models]
yv=np.arange(len(models)); w=0.38
ax.barh(yv+w/2,lowr,w,color=CA,label=f'low-F0 controls (n = {E[models[0]]["n_low"]})',
        edgecolor='white',linewidth=0.7)
ax.barh(yv-w/2,higr,w,color=CB,label=f'high-F0 controls (n = {E[models[0]]["n_high"]})',
        edgecolor='white',linewidth=0.7)
ax.set_yticks(yv); ax.set_yticklabels(models,fontsize=7)
ax.set_xlabel('Proportion misclassified as PD'); ax.set_xlim(0,1.05)
ax.set_title('(b) Speaker-level false positives concentrate in the low-F0 band',pad=5,fontsize=8.5)
ax.grid(axis='y',visible=False); ax.legend(loc='lower right',fontsize=6.6)
fig.tight_layout(); fig.savefig('Figure9_sex_proxy.png'); plt.close(fig)
print('figures 7, 8, 9 written')
