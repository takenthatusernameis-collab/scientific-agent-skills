import hashlib,json
from pathlib import Path
import ccxt, numpy as np, pandas as pd, vectorbt as vbt

R=Path(__file__).resolve().parent
SYMS=['APT/USDT:USDT','ARB/USDT:USDT','OP/USDT:USDT','SAND/USDT:USDT','MANA/USDT:USDT','GALA/USDT:USDT','AXS/USDT:USDT','DYDX/USDT:USDT']
BTC='BTC/USDT:USDT'; SINCE=pd.Timestamp('2023-01-01',tz='UTC'); OOS=pd.Timestamp('2025-10-01',tz='UTC')
BETA=90; H=6; K=2; FEE=.00055; SLIP=.00050

def get(ex,since,sym):
    out=[]; cur=int(since.timestamp()*1000)
    while len(out)<6000:
        x=ex.fetch_ohlcv(sym,'4h',since=cur,limit=1500)
        if not x: break
        out+=x; cur=x[-1][0]+1
        if len(x)<1500: break
    return pd.Series({pd.to_datetime(r[0],unit='ms',utc=True):float(r[4]) for r in out}).sort_index()

def metrics(eq,r):
    r=r.dropna(); dd=eq/eq.cummax()-1
    s=r.std(ddof=1); down=r[r<0].std(ddof=1)
    return {'total_return':float(eq.iloc[-1]-1),'CAGR':float(eq.iloc[-1]**(365.25/((eq.index[-1]-eq.index[0]).total_seconds()/86400))-1),'Sharpe':float(np.sqrt(2190)*r.mean()/s) if s else 0.0,'Sortino':float(np.sqrt(2190)*r.mean()/down) if down else 0.0,'MDD':float(dd.min()),'periods':int(len(r))}

ex=ccxt.binanceusdm({'enableRateLimit':True})
ex.load_markets()
close=pd.concat([get(ex,SINCE,s) for s in [BTC]+SYMS],axis=1)
close.columns=['BTC']+['A'+str(i) for i in range(len(SYMS))]
ret=close.pct_change()
r6=close.pct_change(H)
bret=ret['BTC']
beta=ret.drop(columns='BTC').rolling(BETA,min_periods=BETA).cov(bret).div(bret.rolling(BETA,min_periods=BETA).var(),axis=0).shift(1)
btc_h=close['BTC'].pct_change(H).reindex(beta.index)
res=r6.drop(columns='BTC')-beta.mul(btc_h,axis=0)
rank=res.rank(axis=1,ascending=False,method='first')
w=pd.DataFrame(0.0,index=close.index,columns=rank.columns)
for t in w.index:
    z=rank.loc[t]
    if z.notna().sum()>=K: w.loc[t,z.nsmallest(K).index]=1.0/K
w=w.shift(1).fillna(0.0)
m=close.index>=OOS
c=close.loc[m, w.columns]; ww=w.loc[m]
pf=vbt.Portfolio.from_orders(c,size=ww,size_type='targetpercent',fees=FEE,slippage=SLIP,init_cash=1.0,cash_sharing=True,group_by=True,call_seq='auto',direction='longonly')
eq=pf.value(); r=eq.pct_change()
bm=(1+c.pct_change().mean(axis=1).fillna(0)).cumprod()
primary=metrics(eq,r); bench=metrics(bm,bm.pct_change()); n=int((ww.diff().abs().sum(axis=1)>0).sum())
resout={'experiment_id':'V88-CYCLE2-H1-V2','hypothesis_id':'H1_RESIDUAL_XS_MOMENTUM','real_data':True,'source':'Binance USD-M via ccxt','symbols':SYMS,'oos_start':str(OOS),'vectorbt_version':getattr(vbt,'__version__','unknown'),'primary':primary,'benchmark_equal_weight':bench,'oos_rebalance_events':n,'gate_passed':bool(primary['Sharpe']>=.9 and n>=100 and primary['total_return']>bench['total_return']),'data_sha256':hashlib.sha256(close.loc[m].to_csv().encode()).hexdigest()}
(R/'H1_RESULT.json').write_text(json.dumps(resout,indent=2))
(R/'FINAL_STATUS.json').write_text(json.dumps({'task_execution_status':'TASK_COMPLETED_EMPIRICALLY','evidence_status':'EVIDENCE_EMPIRICAL','edge_status':'CANDIDATE_EDGE_PASS' if resout['gate_passed'] else 'NO_EDGE_CONTINUE','experiment_id':'V88-CYCLE2-H1','primary':primary,'benchmark':bench,'oos_rebalance_events':n,'next_action':'INDEPENDENT_HOLDOUT_AND_SURVIVORSHIP_ROBUSTNESS' if resout['gate_passed'] else 'PROMOTE_NEXT_HYPOTHESIS'},indent=2))
print(json.dumps(resout,indent=2))
