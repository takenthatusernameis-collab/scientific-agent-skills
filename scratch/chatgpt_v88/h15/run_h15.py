import gzip,hashlib,json,math,os
from pathlib import Path
import numpy as np,pandas as pd

R=Path(__file__).resolve().parent
DATA_ROOT=R.parent/"h15_data"
cands=list(DATA_ROOT.glob("**/BTCUSDT.csv.gz"))
DATA=cands[0].parent if cands else DATA_ROOT

def load():
    files=sorted(DATA.glob("*.csv.gz"))
    out={}
    hashes={}
    for p in files:
        s=p.name.removesuffix(".csv.gz")
        if s not in ["BTCUSDT"]: continue
        with gzip.open(p,"rt") as fh: d=pd.read_csv(fh,parse_dates=["timestamp"])
        d["timestamp"]=pd.to_datetime(d["timestamp"],utc=True)
        for col in ["open","high","low","close"]: d[col]=pd.to_numeric(d[col],errors="coerce")
        d=d.dropna(subset=["timestamp","open","high","low","close"]).drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
        out[s]=d; hashes[s]=hashlib.sha256(p.read_bytes()).hexdigest()
    if "BTCUSDT" not in out: raise RuntimeError("BTC_MISSING")
    return out,hashes

START=pd.Timestamp("2026-06-01",tz="UTC"); END=pd.Timestamp("2026-07-31 23:59:59",tz="UTC")
COST=0.0021

def metrics(rows):
    if not rows:return {"n":0,"mean":float("nan"),"PF":0.0,"Sharpe":float("nan"),"MDD":float("nan"),"total_return":float("nan")}
    d=pd.DataFrame(rows); d["ts"]=pd.to_datetime(d["ts"],utc=True)
    r=d["return"].to_numpy(float); wins=r[r>0].sum(); losses=-r[r<0].sum()
    daily=d.assign(day=d.ts.dt.floor("D")).groupby("day")["return"].apply(lambda x:float(np.prod(1+x)-1))
    sd=daily.std(ddof=1)
    eq=(1+daily).cumprod()
    return {"n":int(len(r)),"mean":float(r.mean()),"PF":float(wins/losses) if losses>0 else float("inf"),"Sharpe":float(np.sqrt(252)*daily.mean()/sd) if sd>0 else float("nan"),"MDD":float((eq/eq.cummax()-1).min()),"total_return":float(eq.iloc[-1]-1)}

def main():
    frames,hashes=load(); d=frames["BTCUSDT"]; c=d["close"]
    ret=np.log(c/c.shift(1))
    rv=ret.rolling(16).std(ddof=1)
    prior_med=rv.rolling(96).median().shift(1)
    hi=d["high"].rolling(8).max().shift(1); lo=d["low"].rolling(8).min().shift(1)
    idx=c.index[(c.index>=START-pd.Timedelta(days=2))&(c.index<=END)]
    rows=[]; i=96
    while i<len(idx)-3:
        ts=idx[i]
        rvol=float(rv.get(ts,np.nan)); med=float(prior_med.get(ts,np.nan))
        if not np.isfinite(rvol) or not np.isfinite(med) or rvol>=med*0.60:
            i+=1; continue
        direction=1 if c.get(ts)>hi.get(ts,np.nan) else (-1 if c.get(ts)<lo.get(ts,np.nan) else 0)
        if direction==0:
            i+=1; continue
        e=idx[i+1]; x=idx[i+3]
        if e in d.index and x in d.index:
            gross=direction*math.log(d.loc[x,"open"]/d.loc[e,"open"])
            rows.append({"ts":ts,"return":float(math.exp(gross)-1-COST)})
        i+=3
    out={"experiment_id":"V88-CYCLE2-H15","prompt_version":"1.8.1","hypothesis_id":"H15","strategy_class":"VOLATILITY_REGIME","real_data":True,"exploratory_only":True,"validation_window":["2026-06-01","2026-07-31"],"validation":metrics(rows),"data_manifest_sha256":hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),"oos_seen":False,"holdout_seen":False,"new_idea":True}
    p=DATA/"results"; p.mkdir(parents=True,exist_ok=True); (p/"H15_EXPLORATORY.json").write_text(json.dumps(out,indent=2,default=str)+"\n"); print(json.dumps(out,indent=2,default=str))
if __name__=="__main__": main()
