import gzip,hashlib,json,math,os
from pathlib import Path
import numpy as np,pandas as pd

R=Path(__file__).resolve().parent
DATA=R.parent/"h16_h19_data"
cands=list(DATA.glob("**/BTCUSDT.csv.gz"))
if cands: DATA=cands[0].parent

VAL_START=pd.Timestamp("2023-01-01",tz="UTC"); VAL_END=pd.Timestamp("2023-03-31 23:59:59",tz="UTC")
OOS_START=pd.Timestamp("2023-04-01",tz="UTC"); OOS_END=pd.Timestamp("2023-06-30 23:59:59",tz="UTC")
HOLD_START=pd.Timestamp("2023-07-01",tz="UTC"); HOLD_END=pd.Timestamp("2023-12-31 23:59:59",tz="UTC")
COST=0.0021
HYP={
 "H16":"FUNDING_CARRY_CONTRARIAN",
 "H17":"LIQUIDITY_SHOCK_REVERSAL",
 "H18":"WEEKEND_CALENDAR_CARRY",
 "H19":"VOLATILITY_STATE_CONTINUATION"
}

def load():
    files=sorted(DATA.glob("*.csv.gz")); frames={}; funding={}; hashes={}
    for p in files:
        name=p.name
        if name.endswith("_funding.csv.gz"):
            s=name.removesuffix("_funding.csv.gz")
            d=pd.read_csv(p,parse_dates=["fundingTime"])
            d["fundingTime"]=pd.to_datetime(d["fundingTime"],utc=True)
            d["fundingRate"]=pd.to_numeric(d["fundingRate"],errors="coerce")
            funding[s]=d.dropna().set_index("fundingTime").sort_index()
            hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
            continue
        s=name.removesuffix(".csv.gz")
        d=pd.read_csv(p,parse_dates=["timestamp"])
        d["timestamp"]=pd.to_datetime(d["timestamp"],utc=True)
        for col in ["open","high","low","close","volume","taker_buy_base_volume"]:
            d[col]=pd.to_numeric(d[col],errors="coerce")
        frames[s]=d.dropna(subset=["timestamp","open","high","low","close"]).drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
        hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    return frames,funding,hashes

def metric(rows):
    if not rows:return {"n":0,"mean":float("nan"),"PF":0.0,"Sharpe":float("nan"),"MDD":float("nan"),"total_return":float("nan")}
    d=pd.DataFrame(rows); d["ts"]=pd.to_datetime(d["ts"],utc=True); r=d["return"].to_numpy(float)
    wins=r[r>0].sum(); losses=-r[r<0].sum()
    daily=d.assign(day=d.ts.dt.floor("D")).groupby("day")["return"].apply(lambda x:float(np.prod(1+x)-1))
    sd=daily.std(ddof=1); eq=(1+daily).cumprod()
    return {"n":int(len(r)),"mean":float(r.mean()),"PF":float(wins/losses) if losses>0 else float("inf"),"Sharpe":float(np.sqrt(252)*daily.mean()/sd) if sd>0 else float("nan"),"MDD":float((eq/eq.cummax()-1).min()),"total_return":float(eq.iloc[-1]-1)}

def h16(frames,funding,start,end):
    rows=[]; basket=[]
    for s in ["BTCUSDT","ETHUSDT","SOLUSDT"]:
        if s in funding: basket.append(s)
    if not basket:return rows
    bars=frames["BTCUSDT"]; idx=bars.index
    for s in basket:
        fd=funding[s]
        for ts,rr in fd["fundingRate"].items():
            if ts<start-pd.Timedelta(days=1) or ts>end: continue
            rate=float(rr)
            if abs(rate)<0.0005: continue
            pos=-1 if rate>0 else 1
            j=idx.searchsorted(ts,side="right")
            x=j+32
            if j>=len(idx) or x>=len(idx): continue
            e,xs=idx[j],idx[x]
            vals=[]
            for sym in basket:
                if e in frames[sym].index and xs in frames[sym].index:
                    vals.append(math.log(frames[sym].loc[xs,"open"]/frames[sym].loc[e,"open"])*pos)
            if vals:
                rows.append({"ts":ts,"return":float(math.exp(np.mean(vals))-1-COST)})
    return rows

def h17(frames,start,end):
    d=frames["BTCUSDT"]; c=d["close"]; hi=d["high"]; lo=d["low"]; vol=d["volume"]
    tr=(hi-lo)/c
    v_med=vol.rolling(96).median().shift(1); tr_med=tr.rolling(96).median().shift(1)
    idx=c.index[(c.index>=start-pd.Timedelta(days=2))&(c.index<=end)]
    rows=[]; i=96
    while i<len(idx)-3:
        ts=idx[i]
        if vol.get(ts,np.nan)>3*v_med.get(ts,np.nan) and tr.get(ts,np.nan)>2*tr_med.get(ts,np.nan):
            move=(c.get(ts)/c.get(idx[i-1])-1)
            if np.isfinite(move) and abs(move)>=0.002:
                direction=-np.sign(move)
                e,x=idx[i+1],idx[i+3]
                if e in d.index and x in d.index:
                    ret=math.exp(direction*math.log(d.loc[x,"open"]/d.loc[e,"open"]))-1-COST
                    rows.append({"ts":ts,"return":float(ret)})
                i+=3; continue
        i+=1
    return rows

def h18(frames,start,end):
    d=frames["BTCUSDT"]; c=d["close"]; idx=c.index[(c.index>=start-pd.Timedelta(days=3))&(c.index<=end)]
    rows=[]; i=16
    while i<len(idx)-16:
        ts=idx[i]
        if ts.weekday() not in (5,6) or ts.hour%4!=0 or ts.minute!=0:
            i+=1; continue
        prev=c.get(ts)/c.get(idx[i-16])-1
        if not np.isfinite(prev) or abs(prev)<0.005:
            i+=1; continue
        direction=-np.sign(prev); e,x=idx[i+1],idx[i+16]
        if e in d.index and x in d.index:
            ret=math.exp(direction*math.log(d.loc[x,"open"]/d.loc[e,"open"]))-1-COST
            rows.append({"ts":ts,"return":float(ret)})
        i+=16
    return rows

def h19(frames,start,end):
    d=frames["BTCUSDT"]; c=d["close"]; r=np.log(c/c.shift(1)); rv=r.rolling(32).std(ddof=1)
    med=rv.rolling(128).median().shift(1); four=c/c.shift(4)-1
    idx=c.index[(c.index>=start-pd.Timedelta(days=3))&(c.index<=end)]
    rows=[]; i=128
    while i<len(idx)-5:
        ts=idx[i]; state=rv.get(ts,np.nan); base=med.get(ts,np.nan); sig=four.get(ts,np.nan)
        if not np.isfinite(state) or not np.isfinite(base) or not np.isfinite(sig):
            i+=1; continue
        if state>1.5*base and abs(sig)>0.004:
            direction=np.sign(sig); e,x=idx[i+1],idx[i+5]
            if e in d.index and x in d.index:
                ret=math.exp(direction*math.log(d.loc[x,"open"]/d.loc[e,"open"]))-1-COST
                rows.append({"ts":ts,"return":float(ret)})
            i+=5; continue
        i+=1
    return rows

RUN={"H16":h16,"H17":h17,"H18":h18,"H19":h19}

def main():
    h=os.environ["HYPOTHESIS"]; frames,funding,hashes=load()
    rows=RUN[h](frames,funding,VAL_START,VAL_END) if h=="H16" else RUN[h](frames,VAL_START,VAL_END)
    out={"experiment_id":"V88-CYCLE3-H16-H19","prompt_version":"1.8.2","hypothesis_id":h,"hypothesis":HYP[h],"strategy_class":{"H16":"CARRY_FUNDING","H17":"LIQUIDITY","H18":"CALENDAR_SEASONALITY","H19":"VOLATILITY_REGIME"}[h],"real_data":True,"validation_window":["2023-01-01","2023-03-31"],"oos_window":["2023-04-01","2023-06-30"],"holdout_window":["2023-07-01","2023-12-31"],"future_data_hidden":True,"validation":metric(rows),"data_manifest_sha256":hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),"oos_seen":False,"holdout_seen":False,"new_idea":True}
    p=DATA/"results"; p.mkdir(parents=True,exist_ok=True); (p/f"{h}_VALIDATION.json").write_text(json.dumps(out,indent=2,default=str)+"
"); print(json.dumps(out,indent=2,default=str))
if __name__=="__main__": main()
