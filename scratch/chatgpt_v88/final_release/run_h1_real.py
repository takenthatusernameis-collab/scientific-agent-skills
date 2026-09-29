import hashlib,json,io,zipfile,os
from pathlib import Path
from urllib.request import urlopen
from urllib.error import HTTPError
import numpy as np, pandas as pd, vectorbt as vbt

R=Path(__file__).resolve().parent
SYMS=['APTUSDT','ARBUSDT','OPUSDT','SANDUSDT','MANAUSDT','GALAUSDT','AXSUSDT','DYDXUSDT']
BTC='BTCUSDT'; SINCE=pd.Timestamp('2023-01-01',tz='UTC'); OOS=pd.Timestamp('2025-10-01',tz='UTC')
BETA=int(os.getenv('H1_BETA','90')); H=int(os.getenv('H1_H','6')); K=2; FEE=.00055; SLIP=.00050
STAGE=os.getenv('H1_STAGE','OOS'); CONFIG_ID=os.getenv('CONFIG_ID',f'B{BETA}_H{H}'); EXPERIMENT_ID=os.getenv('EXPERIMENT_ID','V88-CYCLE2-H1-V3')
VALID_START=pd.Timestamp('2024-01-01',tz='UTC')

def get(since,sym):
    parts=[]
    for m in pd.date_range(since.normalize(),pd.Timestamp('2026-08-01',tz='UTC'),freq='MS'):
        tag=f"{m.year}-{m.month:02d}"
        url=f"https://data.binance.vision/data/futures/um/monthly/klines/{sym}/4h/{sym}-4h-{tag}.zip"
        try:
            blob=urlopen(url,timeout=30).read()
        except HTTPError as e:
            if e.code==404: continue
            raise
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            names=[n for n in z.namelist() if n.endswith('.csv')]
            if not names: raise RuntimeError(f"NO_CSV {url}")
            d=pd.read_csv(z.open(names[0]),header=None)
        d[0]=pd.to_datetime(d[0],unit='ms',utc=True)
        d[4]=pd.to_numeric(d[4],errors='coerce')
        parts.append(d.set_index(0)[4].rename(sym))
    if not parts: raise RuntimeError(f"NO_ARCHIVE_DATA {sym}")
    return pd.concat(parts).sort_index()

def metrics(eq,r):
    r=r.dropna(); dd=eq/eq.cummax()-1
    s=r.std(ddof=1); down=r[r<0].std(ddof=1)
    return {'total_return':float(eq.iloc[-1]-1),'CAGR':float(eq.iloc[-1]**(365.25/((eq.index[-1]-eq.index[0]).total_seconds()/86400))-1),'Sharpe':float(np.sqrt(2190)*r.mean()/s) if s else 0.0,'Sortino':float(np.sqrt(2190)*r.mean()/down) if down else 0.0,'MDD':float(dd.min()),'periods':int(len(r))}

close=pd.concat([get(SINCE,s) for s in [BTC]+SYMS],axis=1)
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
m=(close.index>=VALID_START)&(close.index<OOS) if STAGE=='VALIDATION' else (close.index>=OOS)
c=close.loc[m, w.columns]; ww=w.loc[m]
pf=vbt.Portfolio.from_orders(c,size=ww,size_type='targetpercent',fees=FEE,slippage=SLIP,init_cash=1.0,cash_sharing=True,group_by=True,call_seq='auto',direction='longonly')
eq=pf.value(); r=eq.pct_change()
bm=(1+c.pct_change().mean(axis=1).fillna(0)).cumprod()
primary=metrics(eq,r); bench=metrics(bm,bm.pct_change()); n=int((ww.diff().abs().sum(axis=1)>0).sum())
screen_passed=bool(primary['Sharpe']>=0.50 and n>=100 and primary['total_return']>bench['total_return'])
gate_passed=bool(STAGE=='OOS' and primary['Sharpe']>=0.90 and n>=100 and primary['total_return']>bench['total_return'])
resout={'experiment_id':EXPERIMENT_ID,'config_id':CONFIG_ID,'stage':STAGE,'beta_lookback':BETA,'horizon_bars':H,'hypothesis_id':'H1_RESIDUAL_XS_MOMENTUM','real_data':True,'source':'Binance Public Data USD-M Futures monthly 4h archive','symbols':SYMS,'oos_start':str(OOS),'vectorbt_version':getattr(vbt,'__version__','unknown'),'primary':primary,'benchmark_equal_weight':bench,'oos_rebalance_events':n,'screen_passed':screen_passed,'gate_passed':gate_passed,'data_sha256':hashlib.sha256(close.loc[m].to_csv().encode()).hexdigest()}
(R/f'H1_RESULT_{CONFIG_ID}_{STAGE}.json').write_text(json.dumps(resout,indent=2))
(R/'FINAL_STATUS.json').write_text(json.dumps({'task_execution_status':'TASK_COMPLETED_EMPIRICALLY','evidence_status':'EVIDENCE_EMPIRICAL','edge_status':'CANDIDATE_EDGE_PASS' if resout['gate_passed'] else 'NO_EDGE_CONTINUE','experiment_id':'V88-CYCLE2-H1','primary':primary,'benchmark':bench,'oos_rebalance_events':n,'next_action':'INDEPENDENT_HOLDOUT_AND_SURVIVORSHIP_ROBUSTNESS' if resout['gate_passed'] else 'PROMOTE_NEXT_HYPOTHESIS'},indent=2))
print(json.dumps(resout,indent=2))
