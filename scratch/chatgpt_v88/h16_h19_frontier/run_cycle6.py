from __future__ import annotations
import hashlib,json,math,os
from pathlib import Path
import numpy as np,pandas as pd

R=Path("scratch/chatgpt_v88/h16_h19_frontier"); DATA=R/"data"; OUT=R/"results"; OUT.mkdir(exist_ok=True)
H=os.environ["HYPOTHESIS"]; PHASE=os.environ["PHASE"]
WINDOWS={
 "validation":("2023-01-01","2023-03-31 23:59:59+00:00"),
 "oos":("2023-04-01","2023-06-30 23:59:59+00:00"),
 "holdout":("2023-07-01","2023-09-30 23:59:59+00:00"),
}
COST=0.0021
SYMS=["BTCUSDT","ETHUSDT","SOLUSDT","DOGEUSDT","INJUSDT","OPUSDT","STXUSDT","LINKUSDT"]

def load():
    frames={}; funding={}; hashes={}
    for p in DATA.glob("*.csv.gz"):
        hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
        if p.name.endswith("_funding.csv.gz"):
            d=pd.read_csv(p); d["timestamp"]=pd.to_datetime(d["timestamp"],utc=True); d["fundingRate"]=pd.to_numeric(d["fundingRate"],errors="coerce")
            funding[p.name.removesuffix("_funding.csv.gz")]=d.dropna().set_index("timestamp").sort_index()
        else:
            d=pd.read_csv(p); d["timestamp"]=pd.to_datetime(d["timestamp"],utc=True)
            for c in ["open","high","low","close","volume","taker_buy_base_volume"]: d[c]=pd.to_numeric(d[c],errors="coerce")
            frames[p.name.removesuffix("_1h.csv.gz")]=d.dropna(subset=["timestamp","open","high","low","close"]).drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
    return frames,funding,hashes

def ret_window(o,e,x,direction=1):
    if e is None or x is None or e not in o.index or x not in o.index: return None
    return float(math.exp(direction*math.log(o.loc[x,"open"]/o.loc[e,"open"]))-1-COST)

def stats(rows):
    if not rows: return {"n":0,"mean":None,"PF":None,"Sharpe":None,"MDD":None,"total_return":None}
    d=pd.DataFrame(rows); d["ts"]=pd.to_datetime(d["ts"],utc=True); r=d["return"].to_numpy(float)
    wins=r[r>0].sum(); losses=-r[r<0].sum()
    daily=d.assign(day=d.ts.dt.floor("D")).groupby("day")["return"].apply(lambda x:float(np.prod(1+x)-1))
    sd=daily.std(ddof=1); eq=(1+daily).cumprod()
    return {"n":int(len(r)),"mean":float(r.mean()),"PF":float(wins/losses) if losses>0 else float("inf"),"Sharpe":float(np.sqrt(252)*daily.mean()/sd) if sd>0 else None,"MDD":float((eq/eq.cummax()-1).min()),"total_return":float(eq.iloc[-1]-1)}

def event_index(idx,ts,hold):
    j=idx.searchsorted(ts,side="right"); e=j; x=j+hold
    if e>=len(idx) or x>=len(idx): return None,None
    return idx[e],idx[x]

def h16(frames,funding,start,end):
    rows=[]; basket=["BTCUSDT","ETHUSDT","SOLUSDT"]; idx=frames["BTCUSDT"].index
    for s in basket:
        for ts,rate in funding[s]["fundingRate"].items():
            if ts<start or ts>end or abs(rate)<0.0005: continue
            e,x=event_index(idx,ts,8)
            vals=[]
            for sym in basket:
                if e in frames[sym].index and x in frames[sym].index:
                    vals.append(math.log(frames[sym].loc[x,"open"]/frames[sym].loc[e,"open"])*(-1 if rate>0 else 1))
            if vals: rows.append({"ts":ts,"return":float(math.exp(np.mean(vals))-1-COST)})
    return rows

def h17(frames,start,end):
    d=frames["BTCUSDT"]; idx=d.index; r=(d.close/d.close.shift(1)-1); tr=(d.high-d.low)/d.close; vm=d.volume.rolling(48).median().shift(1); tm=tr.rolling(48).median().shift(1); rows=[]
    for ts in idx[(idx>=start)&(idx<=end)]:
        if ts not in vm.index or not np.isfinite(vm.get(ts,np.nan)) or not np.isfinite(tm.get(ts,np.nan)): continue
        if d.loc[ts,"volume"]>3*vm.loc[ts] and tr.loc[ts]>2*tm.loc[ts] and abs(r.get(ts,np.nan))>=0.002:
            e,x=event_index(idx,ts,3); rr=ret_window(d,e,x,-np.sign(r.loc[ts]))
            if rr is not None: rows.append({"ts":ts,"return":rr})
    return rows

def h18(frames,start,end):
    d=frames["BTCUSDT"]; idx=d.index; rows=[]
    for ts in idx[(idx>=start)&(idx<=end)]:
        if ts.weekday() not in (5,6) or ts.hour%4!=0: continue
        if ts-pd.Timedelta(hours=16)<idx[0]: continue
        prev=d.close.loc[ts]/d.close.loc[ts-pd.Timedelta(hours=16)]-1 if ts-pd.Timedelta(hours=16) in d.index else np.nan
        if not np.isfinite(prev) or abs(prev)<0.005: continue
        e,x=event_index(idx,ts,16); rr=ret_window(d,e,x,-np.sign(prev))
        if rr is not None: rows.append({"ts":ts,"return":rr})
    return rows

def h19(frames,start,end):
    d=frames["BTCUSDT"]; idx=d.index; r=np.log(d.close/d.close.shift(1)); rv=r.rolling(32).std(ddof=1); med=rv.rolling(128).median().shift(1); mv=d.close/d.close.shift(4)-1; rows=[]
    for ts in idx[(idx>=start)&(idx<=end)]:
        state,base,sig=rv.get(ts,np.nan),med.get(ts,np.nan),mv.get(ts,np.nan)
        if not np.isfinite(state) or not np.isfinite(base) or not np.isfinite(sig): continue
        if state>1.5*base and abs(sig)>0.004:
            e,x=event_index(idx,ts,5); rr=ret_window(d,e,x,np.sign(sig))
            if rr is not None: rows.append({"ts":ts,"return":rr})
    return rows

frames,funding,hashes=load(); start=pd.Timestamp(WINDOWS[PHASE][0],tz="UTC"); end=pd.Timestamp(WINDOWS[PHASE][1])
fn={"H16_1H_FUNDING_CARRY_CONTRARIAN":lambda:h16(frames,funding,start,end),
    "H17_1H_LIQUIDITY_SHOCK_REVERSAL":lambda:h17(frames,start,end),
    "H18_1H_WEEKEND_CALENDAR_REVERSAL":lambda:h18(frames,start,end),
    "H19_1H_VOLATILITY_STATE_CONTINUATION":lambda:h19(frames,start,end)}
rows=fn[H]()
out={"experiment_id":"V88-CYCLE6-H16-H19-1H-FRESH-20260929","hypothesis_id":H,"phase":PHASE,"real_data":True,"future_data_hidden":True,"validation_selection_firewall":PHASE!="validation","stats":stats(rows),"data_manifest_sha256":hashlib.sha256((DATA/"DATA_MANIFEST.json").read_bytes()).hexdigest(),"source_file_hashes":hashes}
(OUT/f"{H}_{PHASE.upper()}.json").write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out,indent=2))
