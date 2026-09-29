import csv,gzip,hashlib,io,json,math,os,zipfile
from pathlib import Path
from urllib.request import urlopen
from urllib.error import HTTPError
import numpy as np,pandas as pd

R=Path(__file__).resolve().parent
DATA=R/"h3_data"
BASE="https://data.binance.vision/data/futures/um/monthly/klines"
SYMBOLS=["BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT","BNBUSDT","DOGEUSDT","ADAUSDT","AVAXUSDT","LINKUSDT","LTCUSDT","BCHUSDT","DOTUSDT","UNIUSDT","ETCUSDT","ATOMUSDT","SUIUSDT","NEARUSDT","APTUSDT","ARBUSDT","OPUSDT","FILUSDT","INJUSDT","TRXUSDT","XLMUSDT","ALGOUSDT","HBARUSDT","ICPUSDT","AAVEUSDT","CRVUSDT","MKRUSDT","RUNEUSDT","STXUSDT","SEIUSDT"]
START=pd.Timestamp("2024-01-01",tz="UTC")
VAL_END=pd.Timestamp("2025-03-31 23:59:59",tz="UTC")
OOS_END=pd.Timestamp("2026-03-31 23:59:59",tz="UTC")
END=pd.Timestamp("2026-09-25 23:59:59",tz="UTC")
COST_FEE=0.0006
SLIP_MULT=0.25
INITIAL_TRAIN=60
EXIT_MAP={"10:15":(14,15),"12:00":(16,0),"close":(20,0)}
THRESHOLDS=(0.5,0.75,1.0)
NQ=(5,10)
EXITS=("10:15","12:00","close")

def month_blob(sym,year,month):
    tag=f"{year}-{month:02d}"
    url=f"{BASE}/{sym}/15m/{sym}-15m-{tag}.zip"
    return urlopen(url,timeout=60).read()

def acquire():
    DATA.mkdir(parents=True,exist_ok=True)
    manifest={}
    months=pd.date_range(START.normalize(),END.normalize(),freq="MS")
    for sym in SYMBOLS:
        frames=[]
        for m in months:
            try: blob=month_blob(sym,m.year,m.month)
            except HTTPError as e:
                if e.code==404: continue
                raise
            with zipfile.ZipFile(io.BytesIO(blob)) as z:
                names=[n for n in z.namelist() if n.endswith(".csv")]
                if not names: raise RuntimeError(f"NO_CSV {sym} {m}")
                df=pd.read_csv(z.open(names[0]),header=None)
            if str(df.iloc[0,0]).strip().lower() in {"open time","open_time"}: df=df.iloc[1:].reset_index(drop=True)
            keep=df.iloc[:,[0,1,2,3,4]]
            keep.columns=["timestamp","open","high","low","close"]
            keep["timestamp"]=pd.to_numeric(keep["timestamp"],errors="coerce")
            for c in ["open","high","low","close"]: keep[c]=pd.to_numeric(keep[c],errors="coerce")
            keep["timestamp"]=pd.to_datetime(keep["timestamp"],unit="ms",utc=True)
            frames.append(keep)
        if not frames: raise RuntimeError(f"NO_DATA {sym}")
        df=pd.concat(frames,ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp")
        df=df[(df.timestamp>=START)&(df.timestamp<=END)]
        path=DATA/f"{sym}.csv.gz"
        df.to_csv(path,index=False,compression="gzip")
        manifest[sym]={"rows":int(len(df)),"start":str(df.timestamp.min()),"end":str(df.timestamp.max()),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    (R/"H3_DATA_MANIFEST.json").write_text(json.dumps({"source":"Binance USD-M public archive","start":str(START),"end":str(END),"symbols":SYMBOLS,"files":manifest},indent=2)+"\n")

def load():
    return {s:pd.read_csv(DATA/f"{s}.csv.gz",parse_dates=["timestamp"]).set_index("timestamp").sort_index() for s in SYMBOLS}

def session_days(index):
    days=pd.date_range(START.normalize(),END.normalize(),freq="D",tz="UTC")
    return [d for d in days if d.weekday()<5]

def daily_offsession(bars):
    idx=bars.index
    off=~(((idx.weekday<5)&((idx.hour>13)|((idx.hour==13)&(idx.minute>=30))))&~((idx.hour==20)&(idx.minute==0)))
    # More directly: cash is 13:30 <= t < 20:00, weekdays.
    cash=(idx.weekday<5)&(((idx.hour>13)|((idx.hour==13)&(idx.minute>=30)))&((idx.hour<20)))
    off=~cash
    ret=np.log(bars["close"]/bars["open"])
    out=[]
    dates=[]
    for ts,val in ret[off].items():
        day=ts.normalize()
        if ts.weekday()<5 and (ts.hour,ts.minute)<(13,30):
            target=day
        elif ts.weekday()<5:
            target=day+pd.Timedelta(days=1)
            while target.weekday()>=5: target+=pd.Timedelta(days=1)
        else:
            target=day+pd.Timedelta(days=1)
            while target.weekday()>=5: target+=pd.Timedelta(days=1)
        out.append(val); dates.append(target)
    return pd.Series(out,index=pd.DatetimeIndex(dates,tz="UTC")).groupby(level=0).sum()

def beta_prior(daily,day,window):
    btc=daily["BTCUSDT"]
    out={}
    for s in SYMBOLS:
        if s=="BTCUSDT": continue
        xy=daily[[s,"BTCUSDT"]].dropna()
        xy=xy[xy.index<day].tail(window)
        if len(xy)<window: out[s]=np.nan; continue
        x=xy["BTCUSDT"].to_numpy(); y=xy[s].to_numpy()
        X=np.column_stack([np.ones(window),x])
        coef,*_=np.linalg.lstsq(X,y,rcond=None)
        out[s]=coef[1]
    return pd.Series(out)

def day_trade(bars,daily,day,beta_win,threshold,nq,exit_label):
    if day not in daily.index: return (np.nan,np.nan)
    btc=daily.loc[day,"BTCUSDT"]
    sigma=daily["BTCUSDT"].shift(1).rolling(20,min_periods=20).std(ddof=1).loc[day]
    if not np.isfinite(sigma) or abs(btc)<=threshold*sigma: return (np.nan,np.nan)
    beta=beta_prior(daily,day,beta_win)
    resid=(daily.loc[day].drop("BTCUSDT")-beta*btc).dropna()
    if len(resid)<max(2*nq,4): return (np.nan,np.nan)
    k=max(1,int(math.ceil(len(resid)/nq)))
    longs=list(resid.nsmallest(k).index); shorts=list(resid.nlargest(k).index)
    entry=day+pd.Timedelta(hours=13,minutes=30)
    exith,exitm=EXIT_MAP[exit_label]; exit_ts=day+pd.Timedelta(hours=exith,minutes=exitm)
    if entry not in bars["BTCUSDT"].index or exit_ts not in bars["BTCUSDT"].index: return (np.nan,np.nan)
    pos={s:0.5/len(longs) for s in longs}; pos.update({s:-0.5/len(shorts) for s in shorts})
    nb=sum(pos[s]*beta.get(s,np.nan) for s in pos)
    if not np.isfinite(nb): return (np.nan,np.nan)
    pos["BTCUSDT"]=-nb
    gross=cost=0.0
    for s,w in pos.items():
        e=bars[s].loc[entry]; x=bars[s].loc[exit_ts]
        gross += w*math.log(x["open"]/e["open"])
        er=(e["high"]-e["low"])/e["close"]; xr=(x["high"]-x["low"])/x["close"]
        cost += abs(w)*(2*COST_FEE+SLIP_MULT*(er+xr))
    return gross,gross-cost

def metrics(vals):
    x=pd.Series(vals).dropna().astype(float)
    if len(x)<2:return {"n":int(len(x)),"mean":float(x.mean()) if len(x) else float("nan"),"PF":float("nan"),"Sharpe":float("nan"),"MDD":float("nan")}
    sd=x.std(ddof=1); wins=x[x>0]; losses=x[x<0]
    eq=x.cumsum(); dd=eq-eq.cummax()
    return {"n":int(len(x)),"mean":float(x.mean()),"PF":float(wins.sum()/abs(losses.sum())) if len(losses) and losses.sum()!=0 else float("inf"),"Sharpe":float(np.sqrt(252)*x.mean()/sd) if sd else 0.0,"MDD":float(dd.min())}

def run_worker():
    bars=load()
    daily=pd.concat([daily_offsession(bars[s]).rename(s) for s in SYMBOLS],axis=1).sort_index()
    beta_win=int(os.environ.get("H3_BETA","40"))
    val_days=[d for d in daily.index if START<=d<=VAL_END]
    oos_days=[d for d in daily.index if VAL_END<d<=OOS_END]
    hold_days=[d for d in daily.index if OOS_END<d<=END]
    cfgs=[(beta_win,t,nq,e) for t in THRESHOLDS for nq in NQ for e in EXITS]
    cache={}
    for cfg in cfgs:
        vals=[]; keys=[]
        for d in val_days+oos_days+hold_days:
            g,n=day_trade(bars,daily,d,*cfg)
            cache[(cfg,d)]=(g,n)
    scored=[]
    train_days=val_days[INITIAL_TRAIN:]
    for cfg in cfgs:
        vals=[cache[(cfg,d)][1] for d in train_days if np.isfinite(cache[(cfg,d)][1])]
        m=metrics(vals)
        score=m["Sharpe"] if np.isfinite(m["Sharpe"]) else -np.inf
        scored.append((score,m["mean"],m["PF"],cfg))
    scored.sort(key=lambda z:(z[0],z[1],z[2]),reverse=True)
    best_cfg=scored[0][3] if scored else cfgs[0]
    v=metrics([cache[(best_cfg,d)][1] for d in train_days if np.isfinite(cache[(best_cfg,d)][1])])
    out={"beta_window":beta_win,"validation":v,"selected_config":{"shock_threshold":best_cfg[1],"n_quantiles":best_cfg[2],"exit":best_cfg[3],"beta_window":best_cfg[0]},"oos_seen":False,"holdout_seen":False}
    path=R/f"H3_VALIDATION_B{beta_win}.json"; path.write_text(json.dumps(out,indent=2,default=float)+"\n")

if __name__=="__main__":
    mode=os.environ.get("H3_MODE","worker")
    acquire() if mode=="acquire" else run_worker()
