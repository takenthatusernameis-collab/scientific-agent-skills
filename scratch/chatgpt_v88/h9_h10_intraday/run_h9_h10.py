import gzip,hashlib,json,math,os
from pathlib import Path
import numpy as np,pandas as pd

R=Path(__file__).resolve().parent
DATA=R.parent/"h9_h10_data"
_candidates=list(DATA.glob("**/BTCUSDT.csv.gz"))
if _candidates:
    DATA=_candidates[0].parent

SYMBOLS=["BTCUSDT","AAVEUSDT","ADAUSDT","ALGOUSDT","APTUSDT","ARBUSDT","ATOMUSDT","AVAXUSDT","BCHUSDT","BNBUSDT","CRVUSDT","DOGEUSDT","DOTUSDT","ETCUSDT","ETHUSDT","FILUSDT","HBARUSDT","ICPUSDT","INJUSDT","LINKUSDT","LTCUSDT","MKRUSDT","NEARUSDT","OPUSDT","RUNEUSDT","SEIUSDT","SOLUSDT","STXUSDT","SUIUSDT","TRXUSDT","UNIUSDT","XLMUSDT","XRPUSDT"]
ALT=SYMBOLS[1:]
VAL_START=pd.Timestamp("2025-10-01",tz="UTC")
VAL_END=pd.Timestamp("2026-03-31 23:59:59",tz="UTC")
OOS_START=pd.Timestamp("2026-04-01",tz="UTC")
OOS_END=pd.Timestamp("2026-07-31 23:59:59",tz="UTC")
HOLD_START=pd.Timestamp("2026-08-01",tz="UTC")
HOLD_END=pd.Timestamp("2026-09-25 23:59:59",tz="UTC")
COST_RT=0.0021
HOLD_BARS=2

HYP={"H9":"MULTI_HORIZON_RELATIVE_MOMENTUM","H10":"BTC_BREAKOUT_CONDITIONED_RELATIVE_MOMENTUM"}

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for ch in iter(lambda:f.read(1<<20),b""): h.update(ch)
    return h.hexdigest()

def load():
    files=sorted(DATA.glob("*.csv.gz"))
    if not files: raise RuntimeError("NO_H9_H10_DATA")
    frames={}; hashes={}
    for p in files:
        s=p.name.removesuffix(".csv.gz")
        if s not in SYMBOLS: continue
        with gzip.open(p,"rt") as fh: df=pd.read_csv(fh,parse_dates=["timestamp"])
        df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True)
        for col in ["open","close"]: df[col]=pd.to_numeric(df[col],errors="coerce")
        df=df.dropna(subset=["timestamp","open","close"]).drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
        frames[s]=df
        hashes[s]=sha(p)
    return frames,hashes

def panel(frames):
    idx=sorted(set().union(*[set(x.index) for x in frames.values()]))
    out=pd.DataFrame(index=pd.DatetimeIndex(idx))
    for s,d in frames.items():
        out[(s,"open")]=d["open"].reindex(out.index)
        out[(s,"close")]=d["close"].reindex(out.index)
    out.columns=pd.MultiIndex.from_tuples(out.columns)
    return out

def returns(p):
    return pd.DataFrame({s:p[(s,"close")]/p[(s,"open")]-1 for s in SYMBOLS})

def signal(h,i,idx,r):
    if i<4: return None
    one_hour=r[SYMBOLS].iloc[max(0,i-4):i+1]
    cur1h={}
    for s in SYMBOLS:
        vals=r[s].iloc[i-3:i+1]
        if len(vals)<4 or not np.isfinite(vals).all(): continue
        cur1h[s]=(np.prod(1+vals)-1)
    if "BTCUSDT" not in cur1h: return None
    rel={s:cur1h[s]-cur1h["BTCUSDT"] for s in cur1h if s!="BTCUSDT"}
    if len(rel)<10: return None
    vals=pd.Series(rel).sort_values()
    k=max(2,int(math.ceil(len(vals)*0.20)))
    if h=="H9":
        longs=list(vals.nlargest(k).index); shorts=list(vals.nsmallest(k).index)
        w={s:0.5/len(longs) for s in longs}; w.update({s:-0.5/len(shorts) for s in shorts})
        return w
    # H10: current 1h BTC breakout relative to its prior 96 non-overlapping 1h observations.
    hist=[]
    for j in range(max(4,i-4*96),i-3,4):
        valsj=r["BTCUSDT"].iloc[j-3:j+1]
        if len(valsj)==4 and np.isfinite(valsj).all(): hist.append(np.prod(1+valsj)-1)
    if len(hist)<48: return None
    lo=float(np.quantile(hist,0.10)); hi=float(np.quantile(hist,0.90))
    btc=cur1h["BTCUSDT"]
    if btc>hi:
        longs=list(vals.nlargest(k).index); return {s:1/len(longs) for s in longs}
    if btc<lo:
        shorts=list(vals.nsmallest(k).index); return {s:-1/len(shorts) for s in shorts}
    return None

def trade(p,idx,i,w):
    ei=i+1; xi=ei+HOLD_BARS
    if xi>=len(idx): return None
    et,xt=idx[ei],idx[xi]
    gross=0.0; base=0.0
    for s,wt in w.items():
        e=p.loc[et,(s,"open")]; x=p.loc[xt,(s,"open")]
        if not (np.isfinite(e) and np.isfinite(x) and e>0): return None
        gross+=wt*math.log(x/e); base+=abs(wt)
    return math.exp(gross)-1-COST_RT*base

def metrics(df):
    if df.empty: return {"n":0,"mean":float("nan"),"PF":0.0,"Sharpe":float("nan"),"MDD":float("nan"),"total_return":float("nan")}
    r=df["return"].to_numpy(float); wins=r[r>0].sum(); losses=-r[r<0].sum()
    daily=df.assign(day=df.ts.dt.floor("D")).groupby("day")["return"].apply(lambda x:np.prod(1+x)-1)
    sd=daily.std(ddof=1)
    return {"n":len(r),"mean":float(r.mean()),"PF":float(wins/losses) if losses>0 else float("inf"),"Sharpe":float(np.sqrt(252)*daily.mean()/sd) if sd>0 else float("nan"),"MDD":float((daily.add(1).cumprod()/daily.add(1).cumprod().cummax()-1).min()),"total_return":float(daily.add(1).prod()-1)}

def run(h):
    frames,hashes=load()
    frames={s:d.loc[d.index<=VAL_END].copy() for s,d in frames.items()}
    p=panel(frames); r=returns(p)
    idx=r.index[(r.index>=VAL_START-pd.Timedelta(days=2))&(r.index<=VAL_END)]
    trades=[]; nxt=0
    for i in range(len(idx)-HOLD_BARS-2):
        if i<nxt: continue
        w=signal(h,i,idx,r)
        if w is None: continue
        tr=trade(p,idx,i,w)
        if tr is None or not np.isfinite(tr): continue
        nxt=i+1+HOLD_BARS+1
        trades.append({"ts":idx[i].isoformat(),"return":float(tr)})
    df=pd.DataFrame(trades)
    if not df.empty: df["ts"]=pd.to_datetime(df["ts"],utc=True)
    return metrics(df),hashes

def main():
    h=os.environ["HYPOTHESIS"]
    m,hashes=run(h)
    out={"experiment_id":"V88-CYCLE2-H9-H10","prompt_version":"1.7.0","hypothesis_id":h,"hypothesis":HYP[h],"real_data":True,"validation_window":["2025-10-01","2026-03-31"],"future_data_hidden":True,"validation":m,"data_manifest_sha256":hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),"oos_seen":False,"holdout_seen":False}
    outdir=DATA/"results"; outdir.mkdir(parents=True,exist_ok=True)
    (outdir/f"{h}_VALIDATION.json").write_text(json.dumps(out,indent=2)+"\\n")
    print(json.dumps(out,indent=2))
if __name__=="__main__": main()
