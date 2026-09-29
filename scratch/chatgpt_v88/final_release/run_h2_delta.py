import hashlib,io,json,os,zipfile
from pathlib import Path
from urllib.request import urlopen
from urllib.error import HTTPError
import numpy as np,pandas as pd,vectorbt as vbt

R=Path(__file__).resolve().parent
BTC="BTCUSDT"; SYMS=["JTOUSDT","PYTHUSDT","WIFUSDT","SEIUSDT","TIAUSDT","ENAUSDT","ORDIUSDT","JUPUSDT"]
START=pd.Timestamp("2023-01-01",tz="UTC"); OOS=pd.Timestamp("2025-10-01",tz="UTC"); VSTART=pd.Timestamp("2024-01-01",tz="UTC")
Z=float(os.getenv("H2_Z","1.0")); H=int(os.getenv("H2_H","6")); STAGE=os.getenv("H2_STAGE","VALIDATION")
COST=2*(0.00055+0.00050); END=pd.Timestamp("2026-08-01",tz="UTC")

def get(sym):
    parts=[]
    for m in pd.date_range(START,END,freq="MS"):
        tag=f"{m.year}-{m.month:02d}"
        url=f"https://data.binance.vision/data/futures/um/monthly/klines/{sym}/4h/{sym}-4h-{tag}.zip"
        try: blob=urlopen(url,timeout=30).read()
        except HTTPError as e:
            if e.code==404: continue
            raise
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            n=[x for x in z.namelist() if x.endswith(".csv")][0]
            d=pd.read_csv(z.open(n),header=None)
        if str(d.iloc[0,0]).lower() in {"open time","open_time"}: d=d.iloc[1:].reset_index(drop=True)
        d[0]=pd.to_numeric(d[0],errors="coerce")
        d[0]=pd.to_datetime(d[0],unit="ms",utc=True)
        for col in [1,4,5,9]: d[col]=pd.to_numeric(d[col],errors="coerce")
        parts.append(d.set_index(0)[[1,4,5,9]].rename(columns={1:"open",4:"close",5:"volume",9:"taker_buy"}))
    if not parts: raise RuntimeError("NO_DATA "+sym)
    return pd.concat(parts).sort_index()

def stats(x):
    x=x.dropna().astype(float)
    if len(x)==0: return {"n":0,"mean":0,"std":0,"Sharpe":0,"win_rate":0,"PF":0}
    sd=x.std(ddof=1); wins=x[x>0]; losses=x[x<0]
    return {"n":int(len(x)),"mean":float(x.mean()),"std":float(sd),"Sharpe":float(np.sqrt(len(x))*x.mean()/sd) if sd else 0.0,"win_rate":float((x>0).mean()),"PF":float(wins.sum()/abs(losses.sum())) if len(losses) and losses.sum()!=0 else float("inf")}

raw={s:get(s) for s in [BTC]+SYMS}
idx=raw[BTC].index
btc=raw[BTC].reindex(idx).dropna()
delta=(2*btc["taker_buy"]-btc["volume"])/btc["volume"].replace(0,np.nan)
mu=delta.rolling(180,min_periods=180).mean().shift(1)
sd=delta.rolling(180,min_periods=180).std(ddof=1).shift(1)
z=(delta-mu)/sd
sig=pd.Series(0.0,index=btc.index)
sig[z>=Z]=1.0; sig[z<=-Z]=-1.0

basket_close=pd.concat([raw[s]["close"].rename(s) for s in SYMS],axis=1).dropna()
basket_open=pd.concat([raw[s]["open"].rename(s) for s in SYMS],axis=1).reindex(basket_close.index).dropna()
common=basket_close.index.intersection(basket_open.index).intersection(sig.index)
basket_close=basket_close.reindex(common); basket_open=basket_open.reindex(common); sig=sig.reindex(common)

fwd=[]
for t in common:
    j=common.get_loc(t); e=j+1; x=j+H
    if x<len(common):
        r=(basket_close.iloc[x]/basket_open.iloc[e]-1).mean()
        fwd.append((t,r))
fwd=pd.Series(dict(fwd)).sort_index()
s=sig.reindex(fwd.index).fillna(0)
trade=(s*fwd-COST).loc[VSTART:] if STAGE=="VALIDATION" else (s*fwd-COST).loc[OOS:]
trade=trade[s.loc[trade.index]!=0]
primary=stats(trade)

btc_ret=btc["close"].pct_change().reindex(trade.index)
btc_signal=np.sign(btc_ret).replace(0,np.nan).dropna()
btc_control=stats((btc_signal*fwd.reindex(btc_signal.index)-COST).dropna())
rng=np.random.default_rng(20260929+int(Z*100))
shuf=trade.copy()
if len(shuf): shuf.iloc[:]=rng.permutation(shuf.to_numpy())
placebo=stats(shuf)

# Independent vectorbt accounting on a normalized basket close approximation.
br=(1+basket_close.pct_change().mean(axis=1).fillna(0)).cumprod()
pos=s.reindex(br.index).fillna(0).shift(1).fillna(0)
pf=vbt.Portfolio.from_orders(br,size=pos,size_type="targetpercent",fees=.00055,slippage=.00050,init_cash=1.0,direction="both")
ve=pf.value(); vr=ve.pct_change().loc[trade.index]
vb={"total_return":float(ve.iloc[-1]-1),"Sharpe":float(np.sqrt(2190)*vr.mean()/vr.std(ddof=1)) if vr.std(ddof=1) else 0.0}

screen=bool(primary["n"]>=100 and primary["mean"]>0 and primary["PF"]>1)
gate=bool(STAGE=="OOS" and primary["n"]>=100 and primary["mean"]>0 and primary["PF"]>1 and primary["Sharpe"]>0.5)
out={"experiment_id":f"V88-CYCLE2-H2-{STAGE}-Z{Z}-H{H}","hypothesis_id":"H2_BTC_VOLUME_DELTA_ALT","stage":STAGE,"real_data":True,"source":"Binance USD-M Public Data 4h archive","threshold_z":Z,"horizon_bars":H,"symbols":SYMS,"primary":primary,"btc_return_control":btc_control,"shuffled_signal_placebo":placebo,"vectorbt_crosscheck":vb,"screen_passed":screen,"gate_passed":gate,"data_sha256":hashlib.sha256(pd.concat([btc,basket_close],axis=1).to_csv().encode()).hexdigest(),"scientific_claim_allowed":False}
(R/f"H2_RESULT_Z{Z}_H{H}_{STAGE}.json").write_text(json.dumps(out,indent=2,default=float))
print(json.dumps(out,indent=2,default=float))
