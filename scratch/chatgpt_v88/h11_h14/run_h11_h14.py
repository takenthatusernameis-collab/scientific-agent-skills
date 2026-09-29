import gzip,hashlib,json,math,os
from pathlib import Path
import numpy as np,pandas as pd

R=Path(__file__).resolve().parent
DATA_ROOT=R.parent/"h11_h14_data"
_candidates=list(DATA_ROOT.glob("**/BTCUSDT.csv.gz"))
if _candidates:
    DATA=_candidates[0].parent
else:
    DATA=DATA_ROOT

SYMBOLS=["BTCUSDT","ETHUSDT","APTUSDT","ARBUSDT","OPUSDT","SEIUSDT","SUIUSDT","STXUSDT","INJUSDT","RUNEUSDT","SOLUSDT","XRPUSDT","DOGEUSDT","ADAUSDT","LINKUSDT"]
ALT_BASKET=["APTUSDT","ARBUSDT","OPUSDT","SEIUSDT","SUIUSDT","STXUSDT","INJUSDT","RUNEUSDT"]
VAL_START=pd.Timestamp("2026-04-01",tz="UTC")
VAL_END=pd.Timestamp("2026-05-31 23:59:59",tz="UTC")
OOS_START=pd.Timestamp("2026-06-01",tz="UTC")
OOS_END=pd.Timestamp("2026-07-31 23:59:59",tz="UTC")
HOLD_START=pd.Timestamp("2026-08-01",tz="UTC")
HOLD_END=pd.Timestamp("2026-09-25 23:59:59",tz="UTC")
COST_RT=0.0021

HYP={
 "H11":"VOLATILITY_BREAKOUT_TREND",
 "H12":"ETH_BTC_SPREAD_MEAN_REVERSION",
 "H13":"CALENDAR_REVERSION",
 "H14":"ETH_LEAD_LAG_ALT_BASKET"
}

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for ch in iter(lambda:f.read(1<<20),b""): h.update(ch)
    return h.hexdigest()

def load():
    files=sorted(DATA.glob("*.csv.gz"))
    if not files:
        raise RuntimeError(f"NO_DATA_ROOT_{DATA}")
    frames={}; hashes={}
    for p in files:
        s=p.name.removesuffix(".csv.gz")
        if s not in SYMBOLS:
            continue
        with gzip.open(p,"rt") as fh:
            df=pd.read_csv(fh,parse_dates=["timestamp"])
        df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True)
        for col in ["open","high","low","close"]:
            if col in df.columns:
                df[col]=pd.to_numeric(df[col],errors="coerce")
        df=df.dropna(subset=["timestamp","open","close"]).drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
        frames[s]=df
        hashes[s]=sha(p)
    missing=[s for s in ["BTCUSDT","ETHUSDT"] if s not in frames]
    if missing:
        raise RuntimeError(f"MISSING_REQUIRED_{missing}")
    return frames,hashes

def metrics(rows):
    if not rows:
        return {"n":0,"mean":float("nan"),"PF":0.0,"Sharpe":float("nan"),"MDD":float("nan"),"total_return":float("nan")}
    df=pd.DataFrame(rows).copy()
    df["ts"]=pd.to_datetime(df["ts"],utc=True)
    r=df["return"].to_numpy(float)
    wins=r[r>0].sum(); losses=-r[r<0].sum()
    daily=df.assign(day=df["ts"].dt.floor("D")).groupby("day")["return"].apply(lambda x:float(np.prod(1+x)-1))
    sd=float(daily.std(ddof=1)) if len(daily)>1 else float("nan")
    eq=(1+daily).cumprod()
    return {
      "n":int(len(r)),
      "mean":float(r.mean()),
      "PF":float(wins/losses) if losses>0 else float("inf"),
      "Sharpe":float(np.sqrt(252)*daily.mean()/sd) if sd>0 else float("nan"),
      "MDD":float((eq/eq.cummax()-1).min()) if len(eq) else float("nan"),
      "total_return":float(eq.iloc[-1]-1) if len(eq) else float("nan")
    }

def run_h11(frames, start, end):
    d=frames["BTCUSDT"]
    c=d["close"]
    prior_hi=c.rolling(32).max().shift(1)
    prior_lo=c.rolling(32).min().shift(1)
    idx=c.index[(c.index>=start-pd.Timedelta(days=2))&(c.index<=end)]
    rows=[]; i=0
    while i < len(idx)-9:
        ts=idx[i]
        if ts < start or ts > end:
            i+=1; continue
        if pd.notna(prior_hi.get(ts)) and c.get(ts)>prior_hi.get(ts):
            e=idx[i+1]; x=idx[i+8]
            if e in d.index and x in d.index:
                ret=math.exp(math.log(d.loc[x,"open"]/d.loc[e,"open"]))-1-COST_RT
                rows.append({"ts":ts,"return":float(ret)})
                i+=8; continue
        if pd.notna(prior_lo.get(ts)) and c.get(ts)<prior_lo.get(ts):
            e=idx[i+1]; x=idx[i+8]
            if e in d.index and x in d.index:
                gross=math.log(d.loc[e,"open"]/d.loc[x,"open"])
                rows.append({"ts":ts,"return":float(math.exp(gross)-1-COST_RT)})
                i+=8; continue
        i+=1
    return rows

def run_h12(frames,start,end):
    e=frames["ETHUSDT"]["close"]; b=frames["BTCUSDT"]["close"]
    common=e.index.intersection(b.index)
    spread=np.log(e.loc[common]/b.loc[common])
    mu=spread.rolling(96).mean().shift(1)
    sd=spread.rolling(96).std(ddof=1).shift(1)
    z=(spread-mu)/sd
    idx=common[(common>=start-pd.Timedelta(days=3))&(common<=end)]
    rows=[]; active=None
    for i in range(96,len(idx)-2):
        ts=idx[i]
        if active is not None:
            entry_i, direction=active
            if i-entry_i>=16 or (direction==1 and z.get(ts,0)>=0) or (direction==-1 and z.get(ts,0)<=0):
                e_ts=idx[entry_i+1]; x_ts=idx[i+1]
                if e_ts in common and x_ts in common:
                    re=math.log(e.loc[x_ts]/e.loc[e_ts])-math.log(b.loc[x_ts]/b.loc[e_ts])
                    rows.append({"ts":idx[entry_i],"return":float(math.exp(direction*re)-1-COST_RT*2)})
                active=None
            continue
        zz=z.get(ts,np.nan)
        if np.isfinite(zz) and zz<=-2:
            active=(i,1)
        elif np.isfinite(zz) and zz>=2:
            active=(i,-1)
    return [r for r in rows if start<=pd.Timestamp(r["ts"])<=end]

def run_h13(frames,start,end):
    d=frames["BTCUSDT"]; c=d["close"]
    r4=c/c.shift(16)-1
    idx=c.index[(c.index>=start-pd.Timedelta(days=3))&(c.index<=end)]
    rows=[]; slots={0,6,12,18}; i=16
    while i<len(idx)-5:
        ts=idx[i]
        if ts.hour not in slots or ts.minute!=0:
            i+=1; continue
        prev=float(r4.get(ts,np.nan))
        if not np.isfinite(prev) or abs(prev)<0.01:
            i+=1; continue
        e=idx[i+1]; x=idx[i+4]
        if e in d.index and x in d.index:
            gross=-math.copysign(math.log(d.loc[x,"open"]/d.loc[e,"open"]),prev)
            rows.append({"ts":ts,"return":float(math.exp(gross)-1-COST_RT)})
        i+=4
    return rows

def run_h14(frames,start,end):
    eth=frames["ETHUSDT"]["close"]
    alt={s:frames[s]["close"] for s in ALT_BASKET if s in frames}
    idx=eth.index[(eth.index>=start-pd.Timedelta(days=2))&(eth.index<=end)]
    rows=[]; i=2
    while i<len(idx)-3:
        ts=idx[i]
        if ts not in eth.index:
            i+=1; continue
        lead=eth.loc[ts]/eth.loc[idx[i-1]]-1
        if abs(lead)<0.004:
            i+=1; continue
        e=idx[i+1]; x=idx[i+3]
        vals=[]
        for s,d in alt.items():
            if e in d.index and x in d.index:
                vals.append(math.log(d.loc[x,"open"]/d.loc[e,"open"]))
        if len(vals)<4:
            i+=1; continue
        gross=float(np.mean(vals))*np.sign(lead)
        rows.append({"ts":ts,"return":float(math.exp(gross)-1-COST_RT)})
        i+=3
    return rows

RUNNERS={"H11":run_h11,"H12":run_h12,"H13":run_h13,"H14":run_h14}

def main():
    h=os.environ["HYPOTHESIS"]
    if h not in HYP:
        raise SystemExit("UNKNOWN_HYPOTHESIS")
    frames,hashes=load()
    rows=RUNNERS[h](frames,VAL_START,VAL_END)
    result={
      "experiment_id":"V88-CYCLE2-H11-H14",
      "prompt_version":"1.8.1",
      "hypothesis_id":h,
      "hypothesis":HYP[h],
      "strategy_class":{"H11":"TIME_SERIES_TREND","H12":"RELATIVE_VALUE_MEAN_REVERSION","H13":"CALENDAR_SEASONALITY","H14":"CROSS_ASSET_LEAD_LAG"}[h],
      "real_data":True,
      "validation_window":["2026-04-01","2026-05-31"],
      "oos_window":["2026-06-01","2026-07-31"],
      "holdout_window":["2026-08-01","2026-09-25"],
      "future_data_hidden":True,
      "validation":metrics(rows),
      "data_manifest_sha256":hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
      "oos_seen":False,
      "holdout_seen":False,
      "new_idea":True
    }
    out=DATA/"results"; out.mkdir(parents=True,exist_ok=True)
    (out/f"{h}_VALIDATION.json").write_text(json.dumps(result,indent=2,default=str)+"\n")
    print(json.dumps(result,indent=2,default=str))

if __name__=="__main__":
    main()
