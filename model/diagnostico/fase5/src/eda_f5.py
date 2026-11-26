import argparse, csv, io, hashlib, json, zipfile, platform
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, chi2_contingency
from sklearn.model_selection import train_test_split
from docx import Document
P=argparse.ArgumentParser();P.add_argument('--source',required=True);P.add_argument('--master',required=True);P.add_argument('--out',required=True);a=P.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(a.master) as z:
 def get(s):return z.read(next(n for n in z.namelist() if n.endswith(s)))
 manifest=get('/preliminar/resultados/split_manifest.csv'); old=json.loads(get('/preliminar/resultados/audit.json'))
# Only store development IDs; never compute reserve statistics or read reserve labels.
m=pd.read_csv(io.BytesIO(manifest)); dev=set(m.loc[m.partition.eq('exploratory'),'Customer_ID']);assert len(dev)==70000 and m.Customer_ID.is_unique;del m
with zipfile.ZipFile(a.source) as z:
 tables=[]; hashes={}
 for name in ['Client','Record']:
  member=next(n for n in z.namelist() if n.endswith('/'+name+'.csv')); h=hashlib.sha256(); buf=io.BytesIO()
  with z.open(member) as f:
   header=f.readline();h.update(header);assert header.decode().strip().split(',')[-1]=='Customer_ID';buf.write(header)
   for line in f:
    h.update(line) # integrity only, no reserve field parsing except key
    key=int(line.rstrip(b'\r\n').rsplit(b',',1)[-1]);
    if key in dev:buf.write(line)
  hashes[name]=h.hexdigest();assert hashes[name]==old[name]['sha256'];buf.seek(0);t=pd.read_csv(buf);assert len(t)==70000 and t.Customer_ID.is_unique and set(t.Customer_ID)==dev;tables.append(t)
 d=tables[0].merge(tables[1],on='Customer_ID',validate='one_to_one').sort_values('Customer_ID').reset_index(drop=True)
 doc=Document(io.BytesIO(z.read(next(n for n in z.namelist() if n.endswith('Dataset Overview.docx')))));dictionary={r.cells[0].text.strip():r.cells[1].text.strip() for t in doc.tables for r in t.rows[1:] if len(r.cells)==2}
assert d.shape==(70000,100) and set(d.churn)=={0,1} and d.churn.notna().all()
y=d.churn; rev=d.rev_Mean; v=rev.clip(lower=0); known=rev.notna();base=y.mean();den=(v*y).sum()
def wilson(k,n):
 if n==0:return (np.nan,np.nan)
 p=k/n;z=1.96;mid=(p+z*z/(2*n))/(1+z*z/n);half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n);return mid-half,mid+half
au=dict(n=len(d),columns=len(d.columns),target_1=int(y.sum()),target_0=int((1-y).sum()),churn_share=base,missing_cells=int(d.isna().sum().sum()),complete_rows=int(d.notna().all(axis=1).sum()),missing_cols=int(d.isna().any().sum()),revenue_missing=int(rev.isna().sum()),negative_revenue=int((rev<0).sum()),zero_revenue=int((rev==0).sum()),negative_equipment=int((d.eqpdays<0).sum()),zero_attempts=int((d.attempt_Mean==0).sum()),active_gt_unique=int((d.actvsubs>d.uniqsubs).sum()),revenue_positive_sum=float(v.sum()),churn_revenue_positive_sum=float(den),revenue_known_mean=float(rev.mean()),hashes=hashes,manifest_sha256=hashlib.sha256(manifest).hexdigest(),reserve='not parsed except IDs for selection; bytes hashed for integrity; prior aggregate inspection F3',python=platform.python_version(),pandas=pd.__version__,numpy=np.__version__)
(o/'audit.json').write_text(json.dumps(au,indent=2))
# Every original field, with conservative feature eligibility classifications.
excluded={'Customer_ID','churn','recv_sms_Mean','iwylis_vce_Mean'}
profile_fields={'prizm_social_one','area','truck','rv','ownrent','lor','dwlltype','marital','adults','infobase','income','numbcars','HHstatin','dwllsize','forgntvl','ethnic','kid0_2','kid3_5','kid6_10','kid11_15','kid16_17','creditcd','crclscod','asl_flag'}
conditional={'totcalls','totmou','totrev','adjrev','adjmou','adjqty','avgrev','avgmou','avgqty','months','eqpdays','phones','models','uniqsubs','actvsubs','new_cell','hnd_price','dualband','refurb_new','hnd_webcap'}
rows=[]; levels=[]; features=[]
for c in d:
 s=d[c];missing=s.isna();num=pd.api.types.is_numeric_dtype(s); r=dict(variable=c,dtype=str(s.dtype),missing_n=int(missing.sum()),missing_pct=missing.mean()*100,unique=s.nunique(),churn_missing=y[missing].mean(),churn_present=y[~missing].mean())
 if num:
  q=s.quantile([0,.01,.25,.5,.75,.95,.99,1]);r.update({k:float(val) for k,val in zip(['min','p01','p25','median','p75','p95','p99','max'],q)});r.update(mean=s.mean(),zeros=int(s.eq(0).sum()),negatives=int(s.lt(0).sum()),iqr_outliers=int(((s<q[.25]-1.5*(q[.75]-q[.25]))|(s>q[.75]+1.5*(q[.75]-q[.25]))).sum()),spearman_churn=s.corr(y,method='spearman') if c not in ['Customer_ID','churn'] and s.nunique()>1 else np.nan,mean_churn=s[y.eq(1)].mean(),mean_stay=s[y.eq(0)].mean())
 elif c not in excluded:
  for k,g in d.groupby(s.fillna('[missing]')):
   if len(g)>=30: levels.append(dict(variable=c,level=k,n=len(g),churn=g.churn.mean(),mean_revenue=g.rev_Mean.mean()))
 rows.append(r)
 desc=dictionary.get(c,'Not documented');status='exclude' if c in excluded else 'audit_only' if c in profile_fields else 'conditional_snapshot' if c in conditional else 'candidate_unverified'
 timing='target days 31–60' if c=='churn' else 'key only' if c=='Customer_ID' else 'undefined' if desc in ['NaN','Not documented'] else 'prior 3/6 months, exact cutoff unconfirmed' if c.startswith(('avg3','avg6','change_')) else 'lifetime/snapshot cutoff unconfirmed' if c in conditional else 'aggregation period/cutoff unconfirmed' if c.endswith('_Mean') else 'snapshot cutoff unconfirmed'
 features.append(dict(variable=c,definition=desc,timing=timing,status=status,missing_n=int(missing.sum()),rule='never predictor' if status=='exclude' else 'no production targeting; governance review' if status=='audit_only' else 'confirm as-of t0; train-fold transforms; exclude in conservative ablation if cutoff unverified'))
pd.DataFrame(rows).to_csv(o/'field_profile.csv',index=False);pd.DataFrame(levels).to_csv(o/'categorical_profiles.csv',index=False);pd.DataFrame(features).to_csv(o/'feature_register.csv',index=False)
# Fixed, interpretable cuts inherited where feasible. Missing/invalid separated.
def cut(s,bins,labels):return pd.cut(s,bins,labels=labels,include_lowest=True).astype('object').fillna('Missing/invalid')
ratio=d.drop_blk_Mean.div(d.attempt_Mean.where(d.attempt_Mean>=30));ratio=ratio.where(ratio.between(0,1));eq=d.eqpdays.where(d.eqpdays>=0)
seg={ 'tenure_months':cut(d.months,[0,6,12,24,36,np.inf],['0–6','7–12','13–24','25–36','37+']), 'equipment_days':cut(eq,[0,180,365,730,np.inf],['0–180','181–365','366–730','731+']), 'revenue':cut(v,[-.001,25,50,75,100,150,np.inf],['0–25','25–50','50–75','75–100','100–150','150+']), 'usage_minutes':cut(d.mou_Mean,[-.001,100,300,600,1000,np.inf],['0–100','100–300','300–600','600–1000','1000+']), 'change_usage':cut(d.change_mou,[-np.inf,-50,0,50,np.inf],['≤−50','−50–0','0–50','>50']), 'care_calls':cut(d.custcare_Mean,[-.001,0,1,3,np.inf],['0','0–1','1–3','>3']), 'failure_ratio':cut(ratio,[-.001,.02,.05,.10,.20,1],['0–2%','2–5%','5–10%','10–20%','20–100%']), 'overage':cut(d.ovrrev_Mean,[-.001,0,20,50,np.inf],['0','0–20','20–50','50+']), 'active_subscribers':cut(d.actvsubs,[0,1,2,3,np.inf],['0–1','2','3','4+']) }
def summarize(g):
 n=int(np.sum(g));k=int(y[g].sum());lo,hi=wilson(k,n);return dict(n=n,churn_n=k,churn_rate=k/n,ci_low=lo,ci_high=hi,rev_known=int(known[g].sum()),mean_revenue=v[g].mean(),revenue_sum=v[g].sum(),churn_revenue=(v[g]*y[g]).sum(),exposure_share=(v[g]*y[g]).sum()/den)
ss=[]
for name,s in seg.items():
 for k in s.unique():ss.append(dict(segment=name,level=k,**summarize(s.eq(k))))
pd.DataFrame(ss).to_csv(o/'segments.csv',index=False)
# associations with controls (not causal), common support counts; missing separated from controls
rules={'H1_old_equipment':(eq.ge(730),eq.notna()),'H2_usage_decline':(d.change_mou.lt(0),d.change_mou.notna()),'H3_failure_ge10pct':(ratio.ge(.1),ratio.notna()),'H4_care_ge3':(d.custcare_Mean.ge(3),d.custcare_Mean.notna()),'H5_overage_ge20':(d.ovrrev_Mean.ge(20),d.ovrrev_Mean.notna()),'H6_revenue_ge100':(rev.ge(100),known),'H7_multi_active':(d.actvsubs.ge(2),d.actvsubs.notna()),'H8_revenue100_old':(rev.ge(100)&eq.ge(730),known&eq.notna()),'H9_decline_old':(d.change_mou.lt(0)&eq.ge(730),d.change_mou.notna()&eq.notna()),'H10_decline_overage':(d.change_mou.lt(0)&d.ovrrev_Mean.ge(20),d.change_mou.notna()&d.ovrrev_Mean.notna())}
rr=[];strata=[];masks={};sens=[]
for name,(m,e) in rules.items():
 m=m&e;ctrl=(~m)&e;g=summarize(m);h=summarize(ctrl);gap=g['churn_rate']-h['churn_rate'];se=np.sqrt(g['churn_rate']*(1-g['churn_rate'])/g['n']+h['churn_rate']*(1-h['churn_rate'])/h['n']);total=weighted=0
 for (tb,ub),ix in d[e].groupby([seg['tenure_months'],seg['usage_minutes']]).groups.items():
  # tenure is definitionally linked to equipment; conditioning is descriptive only
  a1=y.loc[ix][m.loc[ix]];a0=y.loc[ix][ctrl.loc[ix]]
  if min(len(a1),len(a0))>=30:
   delta=a1.mean()-a0.mean();total+=len(ix);weighted+=len(ix)*delta;strata.append(dict(rule=name,tenure=tb,usage=ub,n=len(ix),exposed=len(a1),gap_pp=100*delta))
 rr.append(dict(rule=name,eligible_n=int(e.sum()),**g,control_n=h['n'],control_rate=h['churn_rate'],gap_pp=gap*100,ci_low_pp=(gap-1.96*se)*100,ci_high_pp=(gap+1.96*se)*100,standardized_gap_pp=100*weighted/total if total else np.nan,common_support=total/e.sum()));masks[name]=m
 for p in [.005,.01,.02,base]:
  w=pd.Series(np.where(y==1,p/base,(1-p)/(1-base)));sel=w[m].sum();sens.append(dict(rule=name,prior=p,selected_per_1000=1000*sel/w.sum(),precision=(w[m]*y[m]).sum()/sel,exposure_per_1000=1000*(w[m]*y[m]*v[m]).sum()/w.sum(),rev_coverage_weighted=w[m&known].sum()/sel))
pd.DataFrame(rr).to_csv(o/'hypotheses.csv',index=False);pd.DataFrame(strata).to_csv(o/'strata.csv',index=False);pd.DataFrame(sens).to_csv(o/'prior_scenarios.csv',index=False);b=pd.DataFrame(masks).astype(int);b.T.dot(b).to_csv(o/'overlap.csv')
# Interactions: value x equipment, change x overage; all levels retained, no cherry-pick
out=[]
for f,g in [('revenue','equipment_days'),('change_usage','overage'),('failure_ratio','care_calls')]:
 for (i,j),ix in d.groupby([seg[f],seg[g]]).groups.items():
  mask=d.index.isin(ix);out.append(dict(dim1=f,level1=i,dim2=g,level2=j,**summarize(mask)))
pd.DataFrame(out).to_csv(o/'interactions.csv',index=False)
# Semantic checks: do documented percentage changes resemble absolute deviations? no redefinition without client.
checks={}
for c,cur,prev in [('change_mou','mou_Mean','avg3mou'),('change_rev','rev_Mean','avg3rev')]:
 t=d[[c,cur,prev]].dropna();t=t[t[prev]>0];diff=t[cur]-t[prev];pct=100*diff/t[prev];checks[c]={'n':len(t),'corr_absolute':float(t[c].corr(diff)),'corr_percentage':float(t[c].corr(pct)),'sign_match_absolute':float((np.sign(t[c])==np.sign(diff)).mean())}
checks['failure_ratio']={'valid_n':int(ratio.notna().sum()),'above_one_before_filter':int((d.drop_blk_Mean.div(d.attempt_Mean.where(d.attempt_Mean>=30))>1).sum())};checks['device_age_vs_tenure_approx']={'older_than_months_x31':int((eq>d.months*31).sum())};checks['id_order_diagnostic']=[dict(decile=i+1,n=len(g),churn=g.churn.mean()) for i,g in enumerate(np.array_split(d[['churn']],10))]
(o/'semantic_checks.json').write_text(json.dumps(checks,indent=2))
# Proposed split only. Reuses F3 seed and explicitly labels both development partitions as explored.
tr,va=train_test_split(d.Customer_ID,test_size=.25,random_state=20261125,stratify=y);assert set(tr).isdisjoint(va) and len(tr)==52500 and len(va)==17500
assert np.array_equal(tr.to_numpy(), d.Customer_ID.iloc[train_test_split(np.arange(len(d)),test_size=.25,random_state=20261125,stratify=y)[0]].to_numpy())
# No identifiers in delivered tables. Recreate split from original manifest and this script; store digests only.
(o/'split_proposal.json').write_text(json.dumps({'seed':20261125,'method':'stratified random on ID-sorted development only','train_n':len(tr),'validation_n':len(va),'train_ids_sha256':hashlib.sha256(','.join(map(str,sorted(tr))).encode()).hexdigest(),'validation_ids_sha256':hashlib.sha256(','.join(map(str,sorted(va))).encode()).hexdigest(),'note':'proposed reuse of exact F3 algorithm, ID order, seed and labels; all development labels already explored; not clean holdout'},indent=2))
# Revenue concentration and redundant signals; descriptive only.
conc=[]
ranked=d.loc[rev.ge(0)].sort_values(['rev_Mean','Customer_ID'],ascending=[False,True]);total_valid=ranked.rev_Mean.sum();total_exp=(ranked.rev_Mean*ranked.churn).sum()
for frac in [.01,.05,.10,.20,.50]:
 g=ranked.head(int(np.ceil(frac*len(ranked))));conc.append(dict(top_fraction=frac,n=len(g),revenue_share=g.rev_Mean.sum()/total_valid,churn_rate=g.churn.mean(),exposure_share=(g.rev_Mean*g.churn).sum()/total_exp))
pd.DataFrame(conc).to_csv(o/'revenue_concentration.csv',index=False)
cols=[c for c in d.select_dtypes('number') if c not in excluded and c not in profile_fields];corr=d[cols].corr(method='spearman');pairs=[]
for i,c in enumerate(cols):
 for j in range(i):
  val=corr.iloc[i,j]
  if abs(val)>=.95:pairs.append(dict(feature1=c,feature2=cols[j],spearman=val))
pd.DataFrame(pairs).sort_values('spearman',ascending=False).to_csv(o/'redundant_pairs.csv',index=False)
# QA: reconcile each partition, exposure and counts; prevent boolean-length denominator bug.
segdf=pd.DataFrame(ss);assert (segdf.groupby('segment').n.sum()==len(d)).all() and (segdf.groupby('segment').churn_n.sum()==y.sum()).all() and np.allclose(segdf.groupby('segment').exposure_share.sum(),1)
for row in rr:assert row['n']+row['control_n']==row['eligible_n'] and row['n']>=row['churn_n'] and 0<=row['churn_rate']<=1
assert (pd.DataFrame(rr).n.to_numpy()==np.diag(b.T.dot(b))).all()
(o/'qa.json').write_text(json.dumps({'status':'PASS','checks':['source SHA256 matches F3','unique development IDs both tables and 1:1 join','target binary complete','all nine segment partitions reconcile count, churn and exposure','rule denominators and overlap diagonal reconcile','split matches F3 algorithm and disjoint sizes'],'limitations':'Arithmetic and provenance only; not validation of commercial assumptions or temporal leakage.'},indent=2))
print(json.dumps(au,indent=2));print(pd.DataFrame(rr)[['rule','n','churn_rate','gap_pp','standardized_gap_pp','common_support','churn_revenue','exposure_share']].round(4).to_string(index=False));print(json.dumps(checks,indent=2))
