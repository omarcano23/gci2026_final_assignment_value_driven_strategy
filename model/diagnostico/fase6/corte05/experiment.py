from base import *
from sklearn.preprocessing import SplineTransformer
import warnings
warnings.filterwarnings('ignore',category=UserWarning)
P=Path(__file__).resolve().parent; O=P/'results'; PRE=P.parent/'corte02'/'results'
oldrec=json.loads((PRE/'candidate_recipes.json').read_text());oldcfg=oldrec['configs']
sector={}; defs={}
def put(n,v,desc): sector[n]=v;defs[n]=desc
def rat(a,b,minb=1): return a/b.where(b>=minb)
for t in [12,24]:put(f'tenure_distance_{t}',np.abs(x.months-t),'Distance from hypothetical lifecycle milestone, not contract expiry')
for t in [180,365,730]:put(f'device_distance_{t}',np.abs(x.eqpdays-t),'Distance from hypothetical device age milestone')
for n,a,b in [('tenure',x.months,12),('device',x.eqpdays,365.25)]:
 for f in [np.sin,np.cos]:put(n+'_cycle_'+f.__name__,f(2*np.pi*a/b),'Lifecycle age periodicity, not calendar')
put('device_since_start_days',x.months*30.4375-x.eqpdays,'Approximate previous tenure before current device, conditional snapshot')
for n,a,b in [('voice_failure','drop_vce_Mean','plcd_vce_Mean'),('voice_block','blck_vce_Mean','plcd_vce_Mean'),('data_failure','drop_dat_Mean','plcd_dat_Mean'),('data_block','blck_dat_Mean','plcd_dat_Mean'),('short_inbound','inonemin_Mean','recv_vce_Mean'),('voice_completion','comp_vce_Mean','plcd_vce_Mean'),('wireless_share','mouowylisv_Mean','mou_Mean'),('inwireless_share','mouiwylisv_Mean','mou_Mean'),('peak_share','mou_peav_Mean','mou_Mean')]:
 v=rat(x[a],x[b],10);put(n,v.where(v.between(0,1)),f'{a}/{b}; denominator >=10 and ratio 0–1')
for n,a,b in [('revenue_per_active','rev_Mean','actvsubs'),('usage_per_active','mou_Mean','actvsubs'),('recur_per_active','totmrc_Mean','actvsubs'),('price_per_min','rev_Mean','mou_Mean'),('overage_per_overage_min','ovrrev_Mean','ovrmou_Mean'),('recurring_per_min','totmrc_Mean','mou_Mean'),('outbound_duration','mou_cvce_Mean','comp_vce_Mean'),('inbound_duration','mou_rvce_Mean','recv_vce_Mean'),('handsets_per_month','phones','months'),('callwait_intensity','callwait_Mean','attempt_Mean'),('forward_intensity','callfwdv_Mean','attempt_Mean')]:put(n,rat(x[a],x[b]),f'{a}/{b}; denominator >=1')
for t,a,b,c in [('usage','mou_Mean','avg3mou','avg6mou'),('revenue','rev_Mean','avg3rev','avg6rev')]:
 aa,bb,cc=x[a],x[b],x[c];put(t+'_log_recent',np.log1p(aa.clip(lower=0))-np.log1p(bb.clip(lower=0)),'Log recent vs prior3 contrast; cutoff conditional');put(t+'_log_hist',np.log1p(bb.clip(lower=0))-np.log1p(cc.clip(lower=0)),'Log prior3 vs prior6 contrast; overlapping windows');put(t+'_accel',((aa-bb)-(bb-cc))/(1+cc.abs()),'Normalized contrast of contrasts, not verified acceleration');put(t+'_symmetric',(aa-bb)/(1+aa.abs()+bb.abs()),'Bounded symmetric change');old=2*cc-bb;put(t+'_prior_older3',old.where(old>=0),'Older 3m mean conditional on aligned windows');put(t+'_recent_vs_older3',(bb-old)/(1+bb.abs()+old.abs()).where(old>=0),'Contrast conditional on aligned windows')
put('billing_usage_divergence',sector['revenue_log_recent']-sector['usage_log_recent'],'Revenue trend vs usage trend, not causal price effect')
put('billing_usage_hist_divergence',sector['revenue_log_hist']-sector['usage_log_hist'],'Revenue vs use historical trend contrast')
put('care_failure_interaction',np.log1p(x.custcare_Mean.clip(lower=0))*eng.failure,'Service friction and care intensity')
put('old_low_engagement',np.log1p(x.eqpdays)*(-sector['usage_log_recent']),'Old device and usage decline')
put('overage_divergence',eng.overage_share*sector['billing_usage_divergence'],'Overage and billing-usage divergence')
put('failure_decline',eng.failure*(-sector['usage_log_recent']),'Failure rate and usage decline')
put('recur_gap',x.rev_Mean-x.totmrc_Mean-x.ovrrev_Mean,'Unexplained charge residual; component semantics unverified')
put('voice_data_mix',rat(x.plcd_dat_Mean,x.plcd_dat_Mean+x.plcd_vce_Mean,10),'Data share of attempted voice/data calls')
put('incoming_outgoing_balance',(x.recv_vce_Mean-x.comp_vce_Mean)/(1+x.recv_vce_Mean+x.comp_vce_Mean),'Incoming vs completed voice balance')
for n,a,b in [('usage','adjmou','avgmou'),('revenue','adjrev','avgrev'),('calls','adjqty','avgqty')]:put(n+'_effective_month_gap',rat(x[a],x[b])-x.months,'Implied lifetime denominator vs tenure; conditional data consistency proxy')
sector=pd.DataFrame(sector,index=x.index).replace([np.inf,-np.inf],np.nan)
sets['sector04']=pd.concat([sets['mechanism'],sector],axis=1);sets['sector04_raw']=sets['sector04'].assign(change_mou=x.change_mou,change_rev=x.change_rev)
# Remove inherited speculative call-window reconstruction as well.
sets['sector04_clean']=sets['sector04'].drop(columns=[c for c in sector if ('cycle_' in c or 'distance_' in c or 'effective_month' in c or 'older3' in c or c=='recur_gap')]+['current_vs_old3_qty','current_vs_old3_mou','current_vs_old3_rev'])
sets['sector04_cat']=pd.concat([sets['sector04_clean'],x[catcols].fillna('__MISSING__').astype(str)],axis=1)
sets['sector04_no_life']=pd.concat([sets['no_lifetime'].drop(columns=['current_vs_old3_qty','current_vs_old3_mou','current_vs_old3_rev']),sector.drop(columns=[c for c in sector if 'effective_month' in c or 'older3' in c or 'cycle_' in c or 'distance_' in c or c=='recur_gap'])],axis=1)
assert all(z.columns.is_unique for z in sets.values());(O/'feature_definitions.json').write_text(json.dumps(defs,indent=2));(O/'features04.json').write_text(json.dumps({k:list(v.columns) for k,v in sets.items() if k.startswith('sector04')},indent=2))
newcfg={}
def cfg(n,fs,**params):newcfg[n]=dict(name=n,family='lgb',features=fs,params=params)
basepar=dict(num_leaves=31,max_depth=8,n_estimators=160,learning_rate=.07,min_child_samples=65,reg_lambda=7,feature_fraction=.8,bagging_fraction=.85,bagging_freq=1)
for n,fs,p in [('sector_clean','sector04_clean',{}),('sector_raw','sector04_raw',{}),('sector_nolife','sector04_no_life',{}),('sector_deep','sector04_raw',dict(num_leaves=63,max_depth=10,n_estimators=220,learning_rate=.055,min_child_samples=45,reg_lambda=10)),('sector_goss','sector04_clean',dict(data_sample_strategy='goss',bagging_fraction=1.,bagging_freq=0,n_estimators=160))]:cfg(n,fs,**(basepar|p))
for k in [25,55,90]:cfg(f'sector_select{k}','sector04_clean',**basepar);newcfg[f'sector_select{k}']['select']=k
newcfg['sector_xgb']=dict(name='sector_xgb',family='xgb',features='sector04_clean',params=dict(max_depth=5,n_estimators=180,learning_rate=.07,min_child_weight=12,subsample=.85,colsample_bytree=.8,reg_lambda=7))
newcfg['sector_cat']=dict(name='sector_cat',family='cat',features='sector04_cat',params=dict(depth=7,iterations=150,learning_rate=.085,l2_leaf_reg=8,random_strength=1.5))
sets['sector_spline']=sets['sector04_clean'][['eqpdays','months','hnd_price','rev_Mean','mou_Mean','totmrc_Mean','ovrrev_Mean','custcare_Mean','failure','usage_log_recent','revenue_log_recent','billing_usage_divergence','device_since_start_days','voice_failure','overage_share','actvsubs','avg3mou','avg6mou']]
newcfg['sector_spline']=dict(name='sector_spline',family='spline',features='sector_spline',params=dict(C=.5))
cfg('sector_DART','sector04_clean',**(basepar|dict(boosting_type='dart',n_estimators=100,learning_rate=.1,drop_rate=.1)))
(O/'newconfigs.json').write_text(json.dumps(newcfg,indent=2))
def train_fit(c,a,b):
 if c.get('select'):
  X=sets[c['features']];s=lgb.LGBMClassifier(n_estimators=60,num_leaves=31,min_child_samples=80,n_jobs=4,verbosity=-1,importance_type='gain',random_state=20261126);s.fit(X.iloc[a],y[a]);cols=X.columns[np.argsort(-s.feature_importances_,kind='stable')[:c['select']]].tolist();m=make(c);m.fit(X[cols].iloc[a],y[a]);return (m,cols),m.predict_proba(X[cols].iloc[b])[:,1]
 if c['family']=='spline':
  X=sets[c['features']];m=make_pipeline(SimpleImputer(strategy='median'),SplineTransformer(n_knots=5,degree=2,knots='quantile',extrapolation='constant'),LogisticRegression(C=c['params']['C'],max_iter=200));m.fit(X.iloc[a],y[a]);return m,m.predict_proba(X.iloc[b])[:,1]
 return fit(c,a,b)
if __name__=='__main__' and '--train' in sys.argv:
 for n,c in newcfg.items():
  for j,(a,b) in enumerate(folds):
   out=O/f'{n}_fold{j}.json'
   if out.exists():continue
   try:
    t=time.monotonic();m,p=train_fit(c,tr[a],tr[b]);np.save(O/f'{n}_fold{j}.npy',p);row=dict(name=n,fold=j,seconds=time.monotonic()-t,**metrics(tr[b],p),**selection(tr[b],scores(tr[b],p)))
    if c.get('select'):row['selected_features']=m[1]
    out.write_text(json.dumps(row));print(n,j,round(row['auc'],4),round(row['exposure'],2),round(row['seconds'],1),flush=True)
   except Exception as e: print('FAIL',n,j,repr(e),flush=True);(O/f'{n}_fold{j}.error').write_text(repr(e))
