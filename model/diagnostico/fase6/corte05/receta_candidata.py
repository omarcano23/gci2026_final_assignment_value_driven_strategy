# Usage: python receta_candidata.py /path/original.zip [--rebuild-oof]
# No reserve code. D-029: fit on 52.5k, OOF Platt, score risk x revenue. This is pre-gate, not approval.
from expert import *
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
import sys
recipe=json.loads((O/'proposal05.json').read_text()); rebuild='--rebuild-oof' in sys.argv
C4pol={r['policy_id']:r for r in json.loads((C4/'policies.json').read_text())}; C4cfg=json.loads((C4/'shortlist.json').read_text())['configs']; PRE=P.parent/'corte02'/'results'
# Rebuild component OOF on demand; otherwise reuse exact archived raw predictions for quick development checks.
rawcache={}; fullcache={}; models={}; calmodels={}; resolved={}
def raw(n,c,src,fitfn):
 if n in rawcache:return rawcache[n],fullcache[n]
 oof=np.empty(len(tr));
 for j,(a,b) in enumerate(folds):
  f=src/f'{n}_fold{j}.npy'
  if rebuild or not f.exists():_,p=fitfn(c,tr[a],tr[b]);
  else:p=np.load(f)
  oof[b]=p
 m,p=fitfn(c,tr,va);rawcache[n]=oof;fullcache[n]=p;models[n]=m;return oof,p
def c4(n):
 names=['sector_raw','sector_select25','XGB_6'] if n=='STACK' else list(C4pol[n]['weights']); w=C4pol[n]['weights'] if n!='STACK' else {x:1/3 for x in names}
 pairs=[raw(z,C4cfg[z],C4 if z in newcfg else PRE,train_fit) for z in names];a=np.column_stack([p[0] for p in pairs]);b=np.column_stack([p[1] for p in pairs]);
 if n=='STACK':
  cal=LogisticRegression(C=.3,max_iter=200).fit(logit(np.clip(a,1e-6,1-1e-6)),y[tr]);pv=cal.predict_proba(logit(np.clip(b,1e-6,1-1e-6)))[:,1]
 else:
  ww=np.array([w[z] for z in names]);cal=calibrate(a@ww,y[tr]);pv=applycal(cal,b@ww)
 calmodels[n]=cal;resolved[n]={'names':names,'weights':w,'configs':{z:C4cfg[z] for z in names},'method':'stack' if n=='STACK' else 'Platt'};return pv
def c3(n):
 # Imported original C03 implementations from retained source. They read only development.
 sys.path.insert(0,str(P.parent/'corte02'));import c03 as cm;import adaptive03 as ad
 r=json.loads((C3/'selection03.json').read_text());names=r['recipes'][n]; old=json.loads((PRE/'candidate_recipes.json').read_text())['configs']; pp=[]
 for z in names:
  if z.startswith('old_'):cfg=old[z[4:]];src=PRE;fitfn=cm.basefit;actual=z[4:]
  elif z in ad.configs:cfg=ad.configs[z];src=C3;fitfn=ad.fit2;actual=z
  else:cfg=cm.configs[z];src=C3;fitfn=cm.fit;actual=z
  pp.append(raw(actual,cfg,src,fitfn))
 a=np.mean([t[0] for t in pp],axis=0);b=np.mean([t[1] for t in pp],axis=0);cal=calibrate(a,y[tr]);calmodels['C03_'+n]=cal;resolved['C03_'+n]={'names':names,'configs':{z:r['configs'][z] if z in r['configs'] else old[z[4:]] for z in names},'method':'equal mean + Platt'};return applycal(cal,b)
def component(n):
 if n in ['P072','P038','P025','P036','P029','STACK']:return c4(n)
 if n.startswith('C03final_'):return c3(n.removeprefix('C03final_'))
 if n.startswith('formula'):
  col='ex_composite_'+n[-2:];m=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),LogisticRegression(C=10,max_iter=150));m.fit(E[[col]].iloc[tr],y[tr]);models[n]=m;resolved[n]={'feature':col};return m.predict_proba(E[[col]].iloc[va])[:,1]
 c=CFG[n];oof=np.empty(len(tr));
 for j,(a,b) in enumerate(folds):
  if rebuild:_,p=train_fit(c,tr[a],tr[b]);
  else:p=np.load(O/f'{n}_raw_fold{j}.npy')
  oof[b]=p
 m,p=train_fit(c,tr,va);models[n]=m;cal=calibrate(oof,y[tr]);calmodels[n]=cal;resolved[n]={'configs':{n:c},'method':'Platt'};return applycal(cal,p)
W=recipe['weights'];p=sum(component(n)*w for n,w in W.items());np.save(O/'candidate05_validation.npy',p);joblib.dump({'models':models,'calibrators':calmodels,'recipe':recipe,'resolved':resolved,'q_train':q},O/'candidate05_model.joblib');(O/'resolved05.json').write_text(json.dumps(resolved,indent=2));print('RECIPE',recipe,'METRICS',metrics(va,p),selection(va,scores(va,p)),flush=True)
# Exact full component feature lists and gain where available. Not causal.
imp=[];fmap={}
for n,m in models.items():
 cols=None
 if isinstance(m,tuple):m,cols=m
 if isinstance(m,list):m=m[0]
 if cols is None and hasattr(m,'feature_name_'):cols=m.feature_name_
 if cols is None and hasattr(m,'feature_names_'):cols=m.feature_names_
 if cols is None and hasattr(m,'feature_names_in_'):cols=list(m.feature_names_in_)
 if cols is None:continue
 fmap[n]=list(cols)
 if hasattr(m,'booster_'):gain=m.booster_.feature_importance(importance_type='gain')
 elif hasattr(m,'get_feature_importance'):gain=m.get_feature_importance()
 elif hasattr(m,'feature_importances_'):gain=m.feature_importances_
 else:continue
 if len(cols)==len(gain):
  for c,g in zip(cols,gain):imp.append(dict(component=n,feature=c,importance=float(g),normalized=float(g)/sum(gain) if sum(gain)>0 else 0))
(O/'candidate_feature_lists.json').write_text(json.dumps(fmap,indent=2));pd.DataFrame(imp).to_csv(O/'candidate_importance.csv',index=False)
