"""Illustrative operational queue allocator, no customer data. 75/25 applies to minutes and slots.
Run python gestion_hibrida.py; writes synthetic QA and capacity scenarios. Not production CRM.
"""
from pathlib import Path
import pandas as pd, math
D=Path(__file__).resolve().parent/'resultados/cierre'
def allocate(rows,slots,minutes,alpha=.75):
 assert 0<=alpha<=1 and slots>=0 and minutes>=0 and rows.id.is_unique
 # mandatory service handled outside marketing queue; unknown permission fails closed.
 valid=rows[rows.active.eq(True)&rows.consent.eq(True)&rows.optout.eq(False)&rows.cooldown.eq(False)&rows.collision.eq(False)&rows.closed.eq(False)&rows.open_case.eq(False)&rows.valid_action.eq(True)&rows.minutes.gt(0)].copy()
 used=set();chosen=[]; left={'analytics':[math.floor(slots*alpha),minutes*alpha],'business':[slots-math.floor(slots*alpha),minutes*(1-alpha)]}
 def take(candidates,lane,quota):
  for x in candidates.itertuples():
   if x.id in used or quota[0]<1 or quota[1]<x.minutes:continue
   quota[0]-=1;quota[1]-=x.minutes;used.add(x.id);chosen.append(dict(id=x.id,lane=lane,minutes=x.minutes,reason=x.reason if lane=='business' else 'model_rank',score=x.score))
 # reserve explicit business commitments/campaigns first; do not count same person twice.
 take(valid[valid.business.eq(True)&valid.reason.notna()].sort_values(['business_priority','id'],ascending=[False,True]),'business',left['business'])
 ranked=valid[valid.score.notna()].sort_values(['score','id'],ascending=[False,True]);take(ranked,'analytics',left['analytics'])
 # release only unused business quota to analytics, not silently oversubscribe
 take(ranked,'analytics_backfill',left['business']);out=pd.DataFrame(chosen)
 if len(out):assert out.id.is_unique and len(out)<=slots and out.minutes.sum()<=minutes+1e-9 and set(out.id)<=set(valid.id)
 return out
if __name__=='__main__':
 rows=pd.DataFrame([dict(id=f'C{i:02}',score=100-i,active=True,consent=True,optout=i==0,cooldown=i==1,collision=i==2,closed=i==3,open_case=i==4,valid_action=i!=5,minutes=10,business=i in [6,12,13],business_priority=100-i,reason='callback' if i==6 else 'campaign' if i in [12,13] else None) for i in range(20)])
 out=allocate(rows,8,80);assert len(out)==8 and not set(out.id)&set(rows.id[:6]);assert (out.lane=='business').sum()==2;out.to_csv(D/'synthetic_hybrid_queue.csv',index=False)
 res=[]
 for N in [100000,250000,1000000]:
  for k in [.05,.1,.2,.3]:
   eligible=int(N*.8);slots=int(eligible*k);demand=slots*.3;human_cap=20*30*60/10
   res.append(dict(portfolio=N,eligible=eligible,k=k,selected=slots,analytics_slots=int(.75*slots),business_slots=slots-int(.75*slots),human_share=.3,human_demand=int(demand),human_capacity=int(human_cap),human_gap=int(max(demand-human_cap,0)),required_hours=demand*10/60,available_hours=600))
 pd.DataFrame(res).to_csv(D/'capacity_hybrid_scenarios.csv',index=False);print('synthetic allocation tests PASS')
