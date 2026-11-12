#!/usr/bin/env python3
"""Oxford corpus: link the subject codes in the UCI name field to the sex and age tabulated in
Table I of the source publication (Little et al., IEEE Trans Biomed Eng 2009;56:1015-1022),
validate the F0 proxy, and stratify features and leave-one-subject-out errors by recorded sex.

Usage: oxford_sex.py ANALYSIS_DATASET.csv OUT_OF_FOLD_PREDICTIONS.csv OUT_JSON
"""
import sys, json
import numpy as np, pandas as pd
from scipy import stats

data, oof, out = sys.argv[1:4]

# Table I of the source publication: code -> (group, sex, age, Hoehn & Yahr stage, years since diagnosis)
TABLE_I = {
 'S01': ('PD', 'M', 78, 3.0, 0), 'S34': ('PD', 'F', 79, 2.5, 0.25), 'S44': ('PD', 'M', 67, 1.5, 1),
 'S20': ('PD', 'M', 70, 3.0, 1), 'S24': ('PD', 'M', 73, 2.5, 1), 'S26': ('PD', 'F', 53, 2.0, 1.5),
 'S08': ('PD', 'F', 48, 2.0, 2), 'S39': ('PD', 'M', 64, 2.0, 2), 'S33': ('PD', 'M', 68, 2.0, 3),
 'S32': ('PD', 'M', 50, 1.0, 4), 'S02': ('PD', 'M', 60, 2.0, 4), 'S22': ('PD', 'M', 60, 1.5, 4.5),
 'S37': ('PD', 'M', 76, 1.0, 5), 'S21': ('PD', 'F', 81, 1.5, 5), 'S04': ('PD', 'M', 70, 2.5, 5.5),
 'S19': ('PD', 'M', 73, 1.0, 7), 'S35': ('PD', 'F', 85, 4.0, 7), 'S05': ('PD', 'F', 72, 3.0, 8),
 'S18': ('PD', 'M', 61, 2.5, 11), 'S16': ('PD', 'M', 62, 2.5, 14), 'S27': ('PD', 'M', 72, 2.5, 15),
 'S25': ('PD', 'M', 74, 3.0, 23), 'S06': ('PD', 'F', 63, 2.5, 28),
 'S10': ('HC', 'F', 46, None, None), 'S07': ('HC', 'F', 48, None, None), 'S13': ('HC', 'M', 61, None, None),
 'S43': ('HC', 'M', 62, None, None), 'S17': ('HC', 'F', 64, None, None), 'S42': ('HC', 'F', 66, None, None),
 'S50': ('HC', 'F', 66, None, None), 'S49': ('HC', 'M', 69, None, None),
}
assert len(TABLE_I) == 31

df = pd.read_csv(data)
df['code'] = df['subject'].str.extract(r'(S\d+)$')[0]
P9 = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
      'spread1', 'spread2', 'D2', 'PPE']
sp = df.groupby('code').agg(status=('status', 'first'), n=('status', 'size'),
                            **{f: (f, 'mean') for f in P9}).reset_index()
R = {}
codes_file = set(sp['code']); codes_tab = set(TABLE_I)
R['n_codes_file'] = len(codes_file); R['n_codes_table'] = len(codes_tab)
R['in_file_not_table'] = sorted(codes_file - codes_tab)
R['in_table_not_file'] = sorted(codes_tab - codes_file)
sp['sex'] = sp['code'].map(lambda c: TABLE_I[c][1] if c in TABLE_I else None)
sp['age'] = sp['code'].map(lambda c: TABLE_I[c][2] if c in TABLE_I else np.nan)
sp['grp_tab'] = sp['code'].map(lambda c: TABLE_I[c][0] if c in TABLE_I else None)
doc = sp[sp['sex'].notna()].copy()
R['status_agreement'] = int(((doc['status'] == 1) == (doc['grp_tab'] == 'PD')).sum())
R['n_documented'] = len(doc)

# ---- F0 proxy: boundary at the largest gap in the control distribution (uses controls only)
hc_f0 = np.sort(sp.loc[sp['status'] == 0, 'MDVP:Fo(Hz)'].to_numpy())
gaps = np.diff(hc_f0); k = int(np.argmax(gaps)); cut = (hc_f0[k] + hc_f0[k + 1]) / 2
R['f0_cut'] = float(cut); R['f0_gap'] = float(gaps[k])
sp['proxy'] = np.where(sp['MDVP:Fo(Hz)'] < cut, 'M', 'F')
doc = sp[sp['sex'].notna()].copy()
agree = doc['proxy'] == doc['sex']
R['proxy_agreement'] = dict(n=int(agree.sum()), of=len(doc),
                            disagree=[dict(code=r.code, sex=r.sex, group='PD' if r.status == 1 else 'HC',
                                           f0=round(float(r['MDVP:Fo(Hz)']), 1))
                                      for _, r in doc[~agree].iterrows()])
R['undocumented'] = [dict(code=r.code, group='PD' if r.status == 1 else 'HC', f0=round(float(r['MDVP:Fo(Hz)']), 1),
                          proxy=r.proxy, n_rec=int(r.n)) for _, r in sp[sp['sex'].isna()].iterrows()]

# ---- sex x diagnosis (documented speakers)
a = int(((doc.status == 1) & (doc.sex == 'M')).sum()); b = int(((doc.status == 1) & (doc.sex == 'F')).sum())
c = int(((doc.status == 0) & (doc.sex == 'M')).sum()); d = int(((doc.status == 0) & (doc.sex == 'F')).sum())
orr = a * d / (b * c); se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
R['sex_dx'] = dict(pd_m=a, pd_f=b, hc_m=c, hc_f=d, pct_f_hc=100 * d / (c + d), pct_f_pd=100 * b / (a + b),
                   odds_ratio=orr, or_lo=float(np.exp(np.log(orr) - 1.96 * se)), or_hi=float(np.exp(np.log(orr) + 1.96 * se)),
                   fisher_p=float(stats.fisher_exact([[a, b], [c, d]])[1]),
                   auc_sex_alone=0.5 + (a / (a + b) - c / (c + d)) / 2)

# ---- age
age_pd = doc.loc[doc.status == 1, 'age'].to_numpy(); age_hc = doc.loc[doc.status == 0, 'age'].to_numpy()
u, p_age = stats.mannwhitneyu(age_pd, age_hc, alternative='two-sided')
R['age'] = dict(pd_mean=float(age_pd.mean()), pd_sd=float(age_pd.std(ddof=1)), hc_mean=float(age_hc.mean()),
                hc_sd=float(age_hc.std(ddof=1)), diff=float(age_pd.mean() - age_hc.mean()), mw_p=float(p_age),
                auc_age_alone=float(u / (len(age_pd) * len(age_hc))),
                hc_m_ages=sorted(doc.loc[(doc.status == 0) & (doc.sex == 'M'), 'age'].astype(int).tolist()),
                hc_f_ages=sorted(doc.loc[(doc.status == 0) & (doc.sex == 'F'), 'age'].astype(int).tolist()),
                all_mean=float(doc['age'].mean()), all_sd=float(doc['age'].std(ddof=1)))


def mw(x1, x0):
    u_, p_ = stats.mannwhitneyu(x1, x0, alternative='two-sided')
    return float(p_), float(2 * u_ / (len(x1) * len(x0)) - 1)


# ---- features by diagnosis within recorded sex (speaker-level means)
strat = {}
for f in P9:
    row = {}
    for lab, m in (('pooled', doc.sex.notna()), ('men', doc.sex == 'M'), ('women', doc.sex == 'F')):
        x1 = doc.loc[m & (doc.status == 1), f].to_numpy(); x0 = doc.loc[m & (doc.status == 0), f].to_numpy()
        p_, r_ = mw(x1, x0)
        row[lab] = dict(hc=float(x0.mean()), pd=float(x1.mean()), n_hc=len(x0), n_pd=len(x1), p=p_, r=r_)
    # sex effect among controls and among PD (male vs female)
    for lab, st in (('sex_in_hc', 0), ('sex_in_pd', 1)):
        xm = doc.loc[(doc.status == st) & (doc.sex == 'M'), f].to_numpy()
        xf = doc.loc[(doc.status == st) & (doc.sex == 'F'), f].to_numpy()
        p_, r_ = mw(xm, xf); row[lab] = dict(m=float(xm.mean()), f=float(xf.mean()), p=p_, r=r_)
    strat[f] = row
R['features'] = strat

# ---- leave-one-subject-out speaker-level errors by recorded sex
o = pd.read_csv(oof)
o = o[o['protocol'] == 'C'].copy()
o['code'] = o['subject'].str.extract(r'(S\d+)$')[0]
err = {}
for model, g in o.groupby('model'):
    v = g.groupby('code').agg(y=('y_true', 'first'), votes=('y_pred', 'sum'), n=('y_pred', 'size')).reset_index()
    v['pred'] = (v['votes'] * 2 > v['n']).astype(int)          # ties -> control
    v['sex'] = v['code'].map(lambda c_: TABLE_I[c_][1] if c_ in TABLE_I else None)
    v = v[v['sex'].notna()]
    hc = v[v.y == 0]; pdx = v[v.y == 1]
    err[model] = dict(fp_m=int(hc.loc[hc.sex == 'M', 'pred'].sum()), n_hc_m=int((hc.sex == 'M').sum()),
                      fp_f=int(hc.loc[hc.sex == 'F', 'pred'].sum()), n_hc_f=int((hc.sex == 'F').sum()),
                      fn_m=int((1 - pdx.loc[pdx.sex == 'M', 'pred']).sum()), n_pd_m=int((pdx.sex == 'M').sum()),
                      fn_f=int((1 - pdx.loc[pdx.sex == 'F', 'pred']).sum()), n_pd_f=int((pdx.sex == 'F').sum()))
R['loso_errors'] = err
tot = {k: sum(e[k] for e in err.values()) for k in ('fp_m', 'n_hc_m', 'fp_f', 'n_hc_f', 'fn_m', 'n_pd_m', 'fn_f', 'n_pd_f')}
R['loso_errors_total'] = dict(**tot, fpr_m=tot['fp_m'] / tot['n_hc_m'], fpr_f=tot['fp_f'] / tot['n_hc_f'],
                              fnr_m=tot['fn_m'] / tot['n_pd_m'], fnr_f=tot['fn_f'] / tot['n_pd_f'],
                              n_models=len(err),
                              n_models_fpr_m_gt_f=sum(e['fp_m'] / e['n_hc_m'] > e['fp_f'] / e['n_hc_f'] for e in err.values()))
sp_out = sp[['code', 'status', 'n', 'sex', 'age', 'proxy', 'MDVP:Fo(Hz)']].rename(columns={'MDVP:Fo(Hz)': 'f0'})
R['speakers'] = json.loads(sp_out.to_json(orient='records'))
json.dump(R, open(out, 'w'), indent=1)
for k in ('n_codes_file', 'n_codes_table', 'in_file_not_table', 'in_table_not_file', 'status_agreement', 'n_documented',
          'f0_cut', 'proxy_agreement', 'undocumented', 'sex_dx', 'age', 'loso_errors_total'):
    print(k, json.dumps(R[k]))
for f in ('MDVP:Fo(Hz)', 'spread1', 'PPE', 'spread2', 'D2', 'MDVP:Jitter(%)'):
    print(f, {k: {a_: (round(b_, 3) if isinstance(b_, float) else b_) for a_, b_ in v.items()} for k, v in strat[f].items()})
for m, e in err.items():
    print(m, e)
