import os
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,io,zipfile,hashlib,json,time
from pathlib import Path
import numpy as np,pandas as pd,sklearn,lightgbm as lgb,joblib
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss
from scipy.special import expit,logit
P=Path(__file__).resolve().parent;R=P.parents[2];O=P/'results';O.mkdir(exist_ok=True)
if '--reproduce' in sys.argv:
 import shutil
 frozen=O;O=P/'reproduction';O.mkdir(exist_ok=True)
 for f in ['candidate_recipes.json','calibrated_cv_rank.csv']:shutil.copy2(frozen/f,O/f)
 sys.argv.extend(['--recipe','--champion-only','--rebuild-oof'])

sha=lambda b:hashlib.sha256(b).hexdigest();manifest=(R/'diagnostico/fase3/preliminar/resultados/split_manifest.csv').read_bytes();m=pd.read_csv(io.BytesIO(manifest));ids=set(m.loc[m.partition.eq('exploratory'),'Customer_ID']);assert len(ids)==70000 and m.Customer_ID.is_unique;del m
old=json.loads((R/'diagnostico/fase3/preliminar/resultados/audit.json').read_text());f5=R/'diagnostico/fase5/resultados';audit=json.loads((f5/'audit.json').read_text());assert sha(manifest)==audit['manifest_sha256'];tables=[];hashes={}
with zipfile.ZipFile(sys.argv[1]) as z:
 for name in ['Client','Record']:
  h=hashlib.sha256();b=io.BytesIO()
  with z.open(next(n for n in z.namelist() if n.endswith('/'+name+'.csv'))) as f:
   header=f.readline();h.update(header);assert header.decode().strip().split(',')[-1]=='Customer_ID';b.write(header)
   for line in f:
    h.update(line);key=int(line.rstrip(b'\r\n').rsplit(b',',1)[-1]);
    if key in ids:b.write(line)
  hashes[name]=h.hexdigest();assert hashes[name]==old[name]['sha256'];b.seek(0);t=pd.read_csv(b);assert len(t)==70000 and t.Customer_ID.is_unique and set(t.Customer_ID)==ids;tables.append(t)
d=tables[0].merge(tables[1],on='Customer_ID',validate='one_to_one').sort_values('Customer_ID').reset_index(drop=True);y=d.churn.to_numpy(int);rev=d.rev_Mean.to_numpy();assert d.shape==(70000,100)
tr,va=train_test_split(np.arange(len(d)),test_size=.25,random_state=20261125,stratify=y);prop=json.loads((f5/'split_proposal.json').read_text());hi=lambda a:sha(','.join(map(str,sorted(a))).encode());print("split hashes",hi(tr),hi(va),prop,flush=True);assert hi(d.Customer_ID.iloc[tr])==prop['train_ids_sha256'] and hi(d.Customer_ID.iloc[va])==prop['validation_ids_sha256'];assert not set(tr)&set(va)
folds=list(StratifiedKFold(5,shuffle=True,random_state=20261126).split(tr,y[tr]));foldid=np.full(len(tr),-1)
for j,(_,b) in enumerate(folds):foldid[b]=j
pd.DataFrame({'Customer_ID':d.Customer_ID.iloc[tr],'fold':foldid}).to_csv(O/'train_manifest.csv',index=False);pd.DataFrame({'Customer_ID':d.Customer_ID.iloc[va]}).to_csv(O/'validation_manifest.csv',index=False)
tie=np.random.default_rng(20261127).permutation(len(d));q=y[tr].mean();(O/'integrity.json').write_text(json.dumps(dict(source_hashes=hashes,manifest_hash=sha(manifest),position_hashes=[hi(tr),hi(va)],actual_id_hashes=[hi(d.Customer_ID.iloc[a]) for a in [tr,va]],n_train=len(tr),n_valid=len(va),train_churn=int(y[tr].sum()),valid_churn=int(y[va].sum()),eligible_train=int(np.sum(np.isfinite(rev[tr])&(rev[tr]>=0))),eligible_valid=int(np.sum(np.isfinite(rev[va])&(rev[va]>=0))),fold_sizes=[len(b) for _,b in folds],reserve='keys only for filtering, bytes hashed; no fields parsed',sklearn=sklearn.__version__,lightgbm=lgb.__version__,numpy=np.__version__,pandas=pd.__version__),indent=2))
reg=pd.read_csv(f5/'feature_register.csv');allowed=reg.loc[~reg.status.isin(['audit_only','exclude']),'variable'].tolist();base=['rev_Mean','mou_Mean','totmrc_Mean','ovrrev_Mean','change_mou','change_rev','avg3mou','avg3rev','avg6mou','avg6rev','months','eqpdays','phones','custcare_Mean','drop_blk_Mean','attempt_Mean'];assert set(base)<=set(allowed)
x=d[allowed].copy();x['rev_Mean']=x.rev_Mean.where(x.rev_Mean>=0);x['eqpdays']=x.eqpdays.where(x.eqpdays>=0);num=x.select_dtypes('number').columns.tolist();cur=[c for c in num if c not in ['change_mou','change_rev']];basic=[c for c in base if c not in ['change_mou','change_rev']];no_snap=[c for c in cur if reg.set_index('variable').loc[c,'status']!='conditional_snapshot'];eng=x[cur].copy()
for name,a,b in [('usage_delta','mou_Mean','avg3mou'),('rev_delta','rev_Mean','avg3rev'),('usage_trend','avg3mou','avg6mou'),('rev_trend','avg3rev','avg6rev')]:
 eng[name]=x[a]-x[b];eng[name+'_relative']=eng[name]/x[b].where(x[b]>0)
for name,a,b,minb in [('failure','drop_blk_Mean','attempt_Mean',30),('overage_share','ovrrev_Mean','rev_Mean',1e-9),('care_minutes_per_call','cc_mou_Mean','custcare_Mean',1e-9),('device_tenure','eqpdays','months',1)]:eng[name]=x[a]/x[b].where(x[b]>=minb)
eng['failure']=eng.failure.where(eng.failure.between(0,1));eng['device_tenure']/=30.4375;eng['decline_sign']=np.sign(x.change_mou);eng['revenue_sign']=np.sign(x.change_rev);eng['old_x_decline']=(x.eqpdays/365)*(-eng.decline_sign);eng['overage_x_decline']=eng.overage_share*(-eng.decline_sign);eng=eng.replace([np.inf,-np.inf],np.nan)
sets={'f3':x[base],'basic':x[basic],'wide':x[cur],'engineered':eng,'no_ambiguous':eng.drop(columns=['decline_sign','revenue_sign','old_x_decline','overage_x_decline']),'no_snapshot':x[no_snap],'log_engineered':np.sign(eng)*np.log1p(np.abs(eng))};(O/'features.json').write_text(json.dumps({k:list(v.columns) for k,v in sets.items()},indent=2))
def corrected(p,prior):return expit(logit(np.clip(p,1e-7,1-1e-7))+logit(prior)-logit(q))
def selection(idx,score,prior=.01,k=.05,counts=None):
 yy=y[idx];rr=rev[idx];ok=np.isfinite(rr)&(rr>=0);yy=yy[ok];rr=rr[ok];ss=np.asarray(score)[ok];tt=tie[idx][ok];cc=np.ones(len(yy)) if counts is None else counts[ok];qq=np.average(yy,weights=cc);w=cc*np.where(yy==1,prior/qq,(1-prior)/(1-qq));order=np.lexsort((tt,-np.nan_to_num(ss,nan=-1e30,neginf=-1e30)));w=w[order];yy=yy[order];rr=rr[order];tot=w.sum();z=np.minimum(w,np.maximum(0,k*tot-(np.cumsum(w)-w)));tp=(z*yy).sum();e=(z*yy*rr).sum();return dict(exposure=1000*e/tot,precision=tp/z.sum(),recall=tp/(w*yy).sum(),lift=tp/z.sum()/prior,exposure_recall=e/(w*yy*rr).sum(),capacity=z.sum()/tot)
def metrics(idx,p):
 yy=y[idx];return dict(auc=roc_auc_score(yy,p),ap=average_precision_score(yy,p),brier=brier_score_loss(yy,p),logloss=log_loss(yy,p))
def scores(idx,p,prior=.01,alpha=1):return corrected(p,prior)**alpha*np.nan_to_num(np.maximum(rev[idx],0))
def simple(idx):return {'B1_revenue':rev[idx],'B2_decline':-d.change_mou.iloc[idx].fillna(np.inf).to_numpy(),'B3_device':d.eqpdays.iloc[idx].where(d.eqpdays.iloc[idx]>=0,-np.inf).fillna(-np.inf).to_numpy()}
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
from sklearn.preprocessing import QuantileTransformer
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
new=sets['no_ambiguous'].copy()
for a,b in [('avg3qty','avg6qty'),('avg3mou','avgmou'),('avg3rev','avgrev'),('avg3qty','avgqty'),('mou_Mean','avgmou'),('rev_Mean','avgrev'),('avg6mou','avgmou'),('avg6rev','avgrev'),('avgmou','avgqty'),('avg3mou','avg3qty'),('mou_Mean','attempt_Mean'),('mou_cvce_Mean','comp_vce_Mean'),('ovrmou_Mean','mou_Mean'),('totmrc_Mean','rev_Mean'),('roam_Mean','attempt_Mean'),('complete_Mean','attempt_Mean'),('custcare_Mean','attempt_Mean'),('mou_peav_Mean','mou_Mean'),('actvsubs','uniqsubs'),('phones','months'),('models','phones'),('totrev','months'),('totmou','months'),('adjrev','totrev')]:new[a+'__per__'+b]=x[a]/x[b].where(x[b]>0)
for a,b in [('avg3qty','avg6qty'),('avg3qty','avgqty'),('avg3mou','avgmou'),('avg3rev','avgrev'),('totrev','adjrev'),('totmou','adjmou'),('totcalls','adjqty'),('rev_Mean','totmrc_Mean'),('phones','models'),('uniqsubs','actvsubs')]:new[a+'__minus__'+b]=x[a]-x[b]
new['tenure_minus_device_days']=x.months*30.4375-x.eqpdays
new['current_vs_old3_mou']=x.mou_Mean-(2*x.avg6mou-x.avg3mou)
new['current_vs_old3_rev']=x.rev_Mean-(2*x.avg6rev-x.avg3rev)
new['current_vs_old3_qty']=x.attempt_Mean-(2*x.avg6qty-x.avg3qty)
sets['mechanism']=new.replace([np.inf,-np.inf],np.nan);catcols=[c for c in allowed if c not in num];print('categories',catcols,flush=True)
sets['cat']=pd.concat([sets['mechanism'],x[catcols].fillna('__MISSING__').astype(str)],axis=1)
sets['raw_change']=sets['mechanism'].assign(change_mou=x.change_mou,change_rev=x.change_rev)
sets['recent']=x[no_snap].copy()
for c in ['usage_delta','rev_delta','usage_trend','rev_trend','usage_delta_relative','rev_delta_relative','usage_trend_relative','rev_trend_relative','failure','overage_share','care_minutes_per_call']:sets['recent'][c]=eng[c]
# Additional mechanistic interactions: no identifiers or protected attributes.
plus=sets['mechanism'].copy()
for a,b in [('mou_Mean','avg3mou'),('rev_Mean','avg3rev'),('attempt_Mean','avg3qty')]:
 plus[a+'__recent4_vs_3']=4*(x[a]-x[b]);plus[a+'__recent4_relative']=4*(x[a]-x[b])/x[b].where(x[b]>0)
for a,b in [('avgmou','avg6mou'),('avgrev','avg6rev'),('avgqty','avg6qty')]:plus[a+'__older_life']=(x[a]*x.months-6*x[b])/(x.months-6).where(x.months>6)
plus['failure_x_decline']=eng.failure*eng.usage_delta_relative
plus['overage_x_decline']=eng.overage_share*eng.usage_delta_relative
plus['device_x_decline']=x.eqpdays*eng.usage_delta_relative
plus['tenure_x_decline']=x.months*eng.usage_delta_relative
plus['rev_x_usage_decline']=x.rev_Mean*eng.usage_delta_relative
sets['plus']=plus.replace([np.inf,-np.inf],np.nan)
life=['totcalls','totmou','totrev','adjrev','adjmou','adjqty','avgrev','avgmou','avgqty']
sets['no_lifetime']=sets['mechanism'].drop(columns=[c for c in sets['mechanism'] if any(t in c.split('__') for t in life)])
sets['raw_revenue']=sets['mechanism'].copy();sets['raw_revenue']['rev_Mean']=d.rev_Mean
configs={}
def add(n,f,s,**p):configs[n]=dict(name=n,family=f,features=s,params=p)
def lg(n,s,leaves,depth,nest,lr,child):add(n,'lgb',s,num_leaves=leaves,max_depth=depth,n_estimators=nest,learning_rate=lr,min_child_samples=child)
lg('C01_Tdeep','no_ambiguous',31,7,80,.1,80)
lg('LGB_mech','mechanism',31,8,120,.08,70)
add('XGB_6','xgb','mechanism',max_depth=6,n_estimators=120,learning_rate=.085,min_child_weight=12,subsample=.85,colsample_bytree=.85,reg_lambda=4)
add('CAT_6','cat','cat',depth=6,iterations=150,learning_rate=.1,l2_leaf_reg=5)
lg('LGB_cat','cat',31,8,120,.08,70)
lg('LGB_63','mechanism',63,10,160,.065,45)
lg('LGB_200','no_ambiguous',31,8,200,.05,60)
lg('LGB_raw_change','raw_change',31,8,120,.08,70)
lg('LGB_recent','recent',31,7,120,.08,80)
lg('LGB_shallow','cat',15,5,220,.06,100)
add('XGB_4','xgb','cat',max_depth=4,n_estimators=180,learning_rate=.08,min_child_weight=20,subsample=.9,colsample_bytree=.85,reg_lambda=5)
add('ExtraTrees','extra','mechanism',n_estimators=80,max_depth=14,min_samples_leaf=25,max_features=.75)
lg('LGB_plus','plus',63,10,220,.05,50)
lg('LGB_500','mechanism',63,10,400,.035,60)
lg('LGB_no_life','no_lifetime',31,8,160,.065,70)
lg('LGB_raw_revenue','raw_revenue',31,8,120,.08,70)
lg('LGB_revweighted','mechanism',31,8,160,.065,70);configs['LGB_revweighted']['weight']='sqrt_revenue'
add('CAT_8','cat','cat',depth=8,iterations=240,learning_rate=.065,l2_leaf_reg=8)
add('HistGB','hist','mechanism',max_iter=160,max_leaf_nodes=31,learning_rate=.065,l2_regularization=3,min_samples_leaf=60)
add('MLP','mlp','mechanism',hidden_layer_sizes=(64,32),alpha=.01,max_iter=35,early_stopping=True,validation_fraction=.12,batch_size=512,learning_rate_init=.001)
def make(c):
 p=c['params'];f=c['family'];X=sets[c['features']];cats=X.select_dtypes(exclude='number').columns.tolist();nums=X.select_dtypes('number').columns.tolist()
 if f=='cat':return CatBoostClassifier(**p,thread_count=4,random_seed=20261126,verbose=False,allow_writing_files=False,cat_features=cats)
 if f=='lgb':m=lgb.LGBMClassifier(**(dict(random_state=20261126,n_jobs=4,verbosity=-1,reg_lambda=2,feature_fraction=.85)|p))
 elif f=='xgb':m=XGBClassifier(**p,tree_method='hist',n_jobs=4,random_state=20261126,eval_metric='logloss')
 elif f=='hist':m=HistGradientBoostingClassifier(**p,random_state=20261126)
 elif f=='mlp':return make_pipeline(SimpleImputer(strategy='median',add_indicator=True),QuantileTransformer(n_quantiles=300,output_distribution='normal',random_state=20261126),MLPClassifier(**p,random_state=20261126))
 else:return make_pipeline(SimpleImputer(strategy='median',add_indicator=True),ExtraTreesClassifier(**p,n_jobs=4,random_state=20261126))
 if cats:return make_pipeline(ColumnTransformer([('num','passthrough',nums),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats)]),m)
 return m
def fit(c,a,b):
 X=sets[c['features']];m=make(c)
 if c.get('weight'):
  r=np.maximum(rev[a],0);r=np.where(np.isfinite(r),r,np.nanmedian(r));w=np.sqrt(np.maximum(r,1));w=np.minimum(w,np.quantile(w,.99));m.fit(X.iloc[a],y[a],sample_weight=w/w.mean())
 else:m.fit(X.iloc[a],y[a])
 return m,m.predict_proba(X.iloc[b])[:,1]
def load_oof(n):
 p=np.zeros(len(tr))
 for j,(_,b) in enumerate(folds):p[b]=np.load(O/f'{n}_fold{j}.npy')
 return p
def calibrate(p,labels):return LogisticRegression(C=100).fit(logit(np.clip(p,1e-7,1-1e-7)).reshape(-1,1),labels)
def applycal(c,p):return c.predict_proba(logit(np.clip(p,1e-7,1-1e-7)).reshape(-1,1))[:,1]
