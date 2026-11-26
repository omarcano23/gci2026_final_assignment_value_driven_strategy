"""Matched-capacity marginal Analytics value; aggregates only; stdlib Python 3."""
import csv,json,itertools,math
from pathlib import Path
from dataclasses import dataclass,replace,asdict
P=Path(__file__).resolve().parent; O=P/'results'; O.mkdir(exist_ok=True)
G=list(csv.DictReader((P/'inputs/capacity_grid.csv').open())); TECH=17426/17500
@dataclass
class Params:
    N:int=100000; months:int=12; prior:float=.01; elig:float=.8; fresh:float=.5; k:float=.2; share:float=.75
    human:float=.3; agents:float=20; hours:float=30; minutes:float=10; digital:float=12000
    reach_d:float=.65; reach_h:float=.75; response:float=.15; margin:float=.55; persist:int=6; survival:float=.95
    decay:float=.025; saturation:float=.2; replenishment:bool=False; reset_quality:bool=False
    adoption:tuple=(0,0,.30,.50,.65,.80,.85,.90,.90,.90,.90,.90)
    build_hours:float=160; monthly_hours:float=8; hourly_rate:float=15; integration_extra:float=0; platform_extra:float=0
    contact_h:float=1.2; contact_d:float=.05; extra_per_migrated:float=0; discount_rate:float=.1
B=Params()
def curve(p,policy,k):
    a=[(0.,0.,0.)]+sorted((float(r['k']),float(r['exposure'])/1000,float(r['precision'])*float(r['k'])) for r in G if float(r['prior'])==p.prior and r['policy']==policy)
    for (x,e,c),(y,f,d) in zip(a,a[1:]):
        if x<=k<=y+1e-12:
            w=(k-x)/(y-x);return e+w*(f-e),c+w*(d-c)
    raise ValueError(k)
def sim(p):
    untreated=unassigned=p.N; hist=[]; pending=(0.,0.,0.,0.); rows=[]; cum=0; prior_stock=0; unique=0
    for m in range(1,p.months+1):
        hist.append((m,*pending)); stock=sum((a-b)*p.survival**(m-t) for t,a,b,ea,eb in hist if m-t<p.persist); rev=sum((ea-eb)*p.survival**(m-t) for t,a,b,ea,eb in hist if m-t<p.persist)
        pool=unassigned*p.elig*TECH*p.fresh; cap=min(pool*p.k,p.agents*p.hours*60/p.minutes/p.human if p.human else math.inf,p.digital/(1-p.human) if p.human<1 else math.inf)
        adoption=p.adoption[min(m-1,11)]; business=cap*(1-p.share); switched=cap*p.share*adoption; fallback=cap-business-switched
        # Business first, assumed random representative selection, identical in both arms. Fallback and adopted strata representative of remaining pool.
        residual=pool-business; adopted_pool=residual*adoption; k=switched/adopted_pool if adopted_pool else 0
        eb,cb=curve(p,'B1_revenue',k); ea,ca=curve(p,'Q0062_value',k); et,ct=curve(p,'Q0062_value',1.)
        age=max(0,m-3); age=age%12 if p.reset_quality else age; q=(1-p.decay)**age
        ea=q*ea+(1-q)*et*k; ca=q*ca+(1-q)*ct*k
        reach=p.human*p.reach_h+(1-p.human)*p.reach_d; sat=1-p.saturation*max(0,1-unassigned/p.N); response=p.response*sat
        sa=adopted_pool*ca*reach*response; sb=adopted_pool*cb*reach*response; ra=adopted_pool*ea*reach*response; rb=adopted_pool*eb*reach*response
        pending=(sa,sb,ra,rb); margin=rev*p.margin
        build=p.build_hours*p.hourly_rate/2+p.integration_extra/2 if m<=2 else 0; run=p.monthly_hours*p.hourly_rate+p.platform_extra if m>=3 else 0; extra=switched*p.extra_per_migrated
        cost=build+run+extra; net=margin-cost; cum+=net; df=(1+p.discount_rate)**(m/12); contact=switched*(p.human*p.contact_h+(1-p.human)*p.contact_d)
        natural=untreated*p.prior; new=natural if p.replenishment else 0
        rows.append(dict(month=m,opening_common_no_action=untreated,unassigned=unassigned,eligible_pool=pool,assigned_each=cap,business_same=business,fallback_same=fallback,analytics_switched=switched,actual_rank_fraction=k,human_hours=cap*p.human*p.minutes/60,reached_switched=switched*reach,exposure_analytics=adopted_pool*ea,exposure_B1=adopted_pool*eb,saves_analytics_due_next=sa,saves_B1_due_next=sb,recognized_delta_saves=hist[-1][1]-hist[-1][2],additional_active=stock,delta_churn=stock-prior_stock,protected_revenue_delta=rev,margin_delta=margin,build=build,run=run,extra=extra,net=net,cumulative=cum,pv_margin=margin/df,pv_cost=cost/df,pv_net=net/df,common_contact_cost_each_switched=contact,acquisitions_common=new))
        unique+=cap; unassigned=(unassigned-cap)*(1-p.prior)+new; untreated=untreated-natural+new; prior_stock=stock
        assert cap<=pool+1e-8 and abs(cap-business-fallback-switched)<1e-7 and rows[-1]['human_hours']<=p.agents*p.hours+1e-7
        assert unique<=p.N+sum(r['acquisitions_common'] for r in rows)+1e-6
    return rows
def summ(p,r):
    s=lambda k:sum(x[k] for x in r); benefit=s('margin_delta'); cost=s('build')+s('run')+s('extra'); net=s('net'); a=s('saves_analytics_due_next'); b=s('saves_B1_due_next'); common=s('common_contact_cost_each_switched')
    assert abs(s('delta_churn')-r[-1]['additional_active'])<1e-7
    return dict(N=p.N,months=p.months,prior=p.prior,k=p.k,share=p.share,response=p.response,margin=p.margin,persist=p.persist,decay=p.decay,replenishment=p.replenishment,assigned=s('assigned_each'),switched=s('analytics_switched'),exposure_delta=s('exposure_analytics')-s('exposure_B1'),exposure_relative=(s('exposure_analytics')/s('exposure_B1')-1) if s('exposure_B1') else None,incremental_saves_recognized=s('recognized_delta_saves'),additional_active_end=r[-1]['additional_active'],own_portfolio_pct=100*r[-1]['additional_active']/p.N,revenue_delta=s('protected_revenue_delta'),margin_delta=benefit,build=s('build'),run=s('run'),extra=s('extra'),net=net,npv=s('pv_net'),pv_benefit=s('pv_margin'),pv_cost=s('pv_cost'),roi=net/cost if cost else None,payback=next((x['month'] for x in r if x['cumulative']>=0 and all(y['cumulative']>=0 for y in r[x['month']-1:])),None),max_pv_analytics_budget=s('pv_margin'),break_even_hourly_rate=(s('pv_margin')-sum((p.integration_extra/2 if x['month']<=2 else p.platform_extra)+x['extra'] for x in r)) / sum((p.build_hours/2 if x['month']<=2 else p.monthly_hours)/(1+p.discount_rate)**(x['month']/12) for x in r) if p.integration_extra==p.platform_extra==0 else None,contact_cost_per_expected_save_B1=common/b if b else None,contact_cost_per_expected_save_Analytics=common/a if a else None,common_contact_cost_each_switched=common)
def scale(p,N):return replace(p,N=N,agents=p.agents*N/100000,digital=p.digital*N/100000)
def write(n,r):
    with (O/n).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(r[0]));w.writeheader();w.writerows(r)
sc={'Conservador':replace(B,prior=.005,response=.05,margin=.4,persist=3,survival=.9,decay=.05,fresh=.15),'Base':B,'Favorable':replace(B,prior=.02,response=.30,margin=.7,persist=12,survival=.98,decay=.005,fresh=.85)}
def main():
    ss=[]; mm=[]; sensitivity=[]; mult=[]; costs=[]
    for name,p in sc.items():
        for N in [100000,250000,1000000]:
            q=scale(p,N); r=sim(q); ss.append(dict(scenario=name,**summ(q,r)));mm.extend(dict(scenario=name,N=N,**x) for x in r)
    for N,k,resp,margin,persist in itertools.product([100000,250000,1000000],[.05,.1,.2,.3],[0,.05,.15,.30],[.4,.55,.7],[3,6,12]):
        q=replace(scale(B,N),k=k,response=resp,margin=margin,persist=persist); sensitivity.append(summ(q,sim(q)))
    for N,repl,reset in itertools.product([100000,1000000],[False,True],[False,True]):
        q=replace(scale(B,N),months=36,replenishment=repl,reset_quality=reset); r=sim(q); mult.extend(dict(year=y,**summ(replace(q,months=12*y),r[:12*y])) for y in [1,2,3])
    for N,rate,extra,share in itertools.product([100000,250000,1000000],[0,5,15,30],[0,.05,.20],[.75,1.]):
        q=replace(scale(B,N),hourly_rate=rate,extra_per_migrated=extra,share=share); costs.append(dict(hourly_rate=rate,extra_per_migrated=extra,**summ(q,sim(q))))
    write('cost_sensitivity.csv',costs)
    # pure offline matched decision curves and one-wave monetization, no operating extrapolation
    wave=[]
    for k in [.05,.1,.2,.3,.4,.5,1.]:
        ea,ca=curve(B,'Q0062_value',k); eb,cb=curve(B,'B1_revenue',k); reach=.68; horizon=sum(.95**i for i in range(6)); cost=k*1000*(.3*1.2+.7*.05)
        wave.append(dict(k=k,exposure_A=ea*1000,exposure_B1=eb*1000,delta_exposure=(ea-eb)*1000,relative=ea/eb-1,expected_churners_A=ca*1000,expected_churners_B1=cb*1000,saves_A=ca*1000*reach*.15,saves_B1=cb*1000*reach*.15,margin_A=ea*1000*reach*.15*.55*horizon,margin_B1=eb*1000*reach*.15*.55*horizon,margin_delta=(ea-eb)*1000*reach*.15*.55*horizon,common_contact_cost_each=cost,cost_per_save_A=cost/(ca*1000*reach*.15),cost_per_save_B1=cost/(cb*1000*reach*.15)))
    write('scenarios.csv',ss);write('monthly.csv',mm);write('sensitivity.csv',sensitivity);write('multiyear.csv',mult);write('one_wave.csv',wave)
    (O/'assumptions.json').write_text(json.dumps({k:asdict(v) for k,v in sc.items()},indent=2))
    assert abs(summ(replace(B,share=0),sim(replace(B,share=0)))['margin_delta'])<1e-8
    assert abs(summ(replace(B,response=0),sim(replace(B,response=0)))['margin_delta'])<1e-8
    print('QA: no analytics / no response => zero gross benefit; portfolio and capacity balances pass.')
    for x in ss:print(x['scenario'],x['N'],*[round(x[k],2) for k in ['revenue_delta','margin_delta','net','npv','additional_active_end','contact_cost_per_expected_save_B1','contact_cost_per_expected_save_Analytics']],x['payback'])
if __name__=='__main__': main()
