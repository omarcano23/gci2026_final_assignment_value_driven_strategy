from experiment import *
D={};desc={}
def put(n,v,s): D[n]=v.replace([np.inf,-np.inf],np.nan);desc[n]=s
def pos(v):return v.clip(lower=0)
def bound(v,s):return pos(v)/(s+pos(v))
def drop(a,b):return ((b-a)/(1+b)).clip(0,1).where((a>=0)&(b>=0))
def blend(v,w):
 a=pd.concat(v,axis=1);ww=np.array(w);return a.fillna(0).mul(ww).sum(axis=1)/a.notna().mul(ww).sum(axis=1).replace(0,np.nan)
U=drop(x.mou_Mean,x.avg3mou);H=drop(x.avg3mou,x.avg6mou);V=drop(x.rev_Mean,x.avg3rev)
for n,v,s in [('usage_loss',U,'(avg3mou-mou_Mean)+/(1+avg3mou)'),('history_loss',H,'(avg6mou-avg3mou)+/(1+avg6mou)'),('billing_loss',V,'(avg3rev-rev_Mean)+/(1+avg3rev)'),('shortfall_joint',np.sqrt(U*V),'geometric mean usage and revenue declines'),('shortfall_max',pd.concat([U,H,V],axis=1).max(axis=1),'maximum observed decline contrast'),('shortfall_accel',U-H,'recent minus historic decline, overlapping windows'),('shortfall_persistence',np.sqrt(U*H),'geometric mean recent and historical declines'),('disengagement_low',1/(1+pos(x.mou_Mean)/100),'inverse usage with 100 minute scale'),('overage_pressure',bound(eng.overage_share,.2),'positive overage share / (0.2+share)'),('charge_pressure',bound((x.rev_Mean-x.totmrc_Mean)/(1+x.totmrc_Mean),.5),'positive excess above recurring charge, not arrears'),('service_pressure',bound(eng.failure,.1),'failure rate / (0.1+rate), min 30 attempts'),('care_pressure',bound(x.custcare_Mean,3),'care volume / (3+volume); direction uncertain'),('device_pressure',bound(x.eqpdays,365),'device age/(365+age)'),('tenure_early',1/(1+pos(x.months)/12),'inverse tenure, not contract expiry'),('line_inactivity',((x.uniqsubs-x.actvsubs)/x.uniqsubs.where(x.uniqsubs>0)).clip(0,1),'inactive subscriber share, no household inference')]:put('ex_'+n,v,s)
for n,a,b in [('network_loss','service_pressure','usage_loss'),('bill_loss','overage_pressure','usage_loss'),('device_loss','device_pressure','usage_loss'),('care_network','care_pressure','service_pressure'),('device_line','device_pressure','line_inactivity'),('bill_device','charge_pressure','device_pressure')]:put('ex_'+n,D['ex_'+a]*D['ex_'+b],a+' × '+b)
G={'engagement':['usage_loss','history_loss','disengagement_low','shortfall_joint'],'billing':['overage_pressure','charge_pressure','billing_loss'],'service':['service_pressure','care_pressure','network_loss'],'lifecycle':['device_pressure','tenure_early','line_inactivity','device_loss']}
for n,cols in G.items():put('ex_domain_'+n,blend([D['ex_'+c] for c in cols],[1]*len(cols)),'available-component mean: '+', '.join(cols))
weights=[[1,1,1,1],[4,1,1,1],[1,4,1,1],[1,1,4,1],[1,1,1,4],[3,2,2,1],[2,1,1,3],[2,3,1,1],[2,1,3,1],[1,2,2,3],[4,2,1,1],[1,1,2,4]]
for i,w in enumerate(weights):put(f'ex_composite_{i:02}',blend([D['ex_domain_'+n] for n in G],w),f'weighted available-domain mean: {dict(zip(G,w))}')
E=pd.DataFrame(D,index=x.index);sets['expert_only']=E;sets['expert_wide']=pd.concat([sets['sector04_clean'],E],axis=1);sets['expert_nolife']=pd.concat([sets['sector04_no_life'],E],axis=1)
assert all(z.columns.is_unique for z in sets.values());(O/'expert_definitions.json').write_text(json.dumps(desc,indent=2));(O/'expert_sets.json').write_text(json.dumps({k:list(sets[k]) for k in ['expert_only','expert_wide','expert_nolife']},indent=2))
CFG={}
for n,fs,leaf,depth,it,lr,child,lam in [('expert_wide','expert_wide',31,8,120,.08,65,7),('expert_nolife','expert_nolife',31,8,120,.08,65,7),('expert_only','expert_only',15,5,100,.085,100,8),('expert_shallow','expert_wide',15,5,150,.08,95,10)]:CFG[n]=dict(name=n,family='lgb',features=fs,params=dict(num_leaves=leaf,max_depth=depth,n_estimators=it,learning_rate=lr,min_child_samples=child,reg_lambda=lam,feature_fraction=.8,bagging_fraction=.85,bagging_freq=1))
CFG['expert_select35']=dict(CFG['expert_wide'],name='expert_select35',select=35)
CFG['expert_xgb']=dict(name='expert_xgb',family='xgb',features='expert_nolife',params=dict(max_depth=5,n_estimators=120,learning_rate=.08,min_child_weight=15,subsample=.85,colsample_bytree=.8,reg_lambda=9))
(O/'expert_configs.json').write_text(json.dumps(CFG,indent=2))
C4=P.parent/'corte04'/'results'; C3=P.parent/'corte03_work'
assert pd.read_csv(C4/'train_manifest.csv').equals(pd.read_csv(O/'train_manifest.csv'));assert pd.read_csv(C3/'train_manifest.csv').equals(pd.read_csv(O/'train_manifest.csv'))
def train_all():
 for n,c in CFG.items():
  for j,(a,b) in enumerate(folds):
   f=O/f'{n}_fold{j}.json'
   if f.exists():continue
   A=tr[a];B=tr[b];aa,bb=train_test_split(A,test_size=.2,random_state=20261126+j,stratify=y[A]);t=time.monotonic();m,p=train_fit(c,A,B);_,ip=train_fit(c,aa,bb);cal=calibrate(ip,y[bb]);cp=applycal(cal,p);np.save(O/f'{n}_raw_fold{j}.npy',p);np.save(O/f'{n}_fold{j}.npy',cp);row=dict(name=n,fold=j,seconds=time.monotonic()-t,**metrics(B,cp),**selection(B,scores(B,cp)));f.write_text(json.dumps(row));print(n,j,row['auc'],row['exposure'],round(row['seconds'],1),flush=True)
if __name__=='__main__' and '--train' in sys.argv:train_all()
