from __future__ import annotations
import concurrent.futures as cf
import hashlib, io, json, urllib.request, zipfile
from pathlib import Path
import pandas as pd

R=Path("scratch/chatgpt_v88/h16_h19_frontier")
DATA=R/"data"; DATA.mkdir(parents=True, exist_ok=True)
SYMS=["BTCUSDT","ETHUSDT","SOLUSDT"]
FUNDING=["BTCUSDT","ETHUSDT","SOLUSDT"]
MONTHS=pd.date_range("2023-01-01","2023-09-01",freq="MS",tz="UTC")
BASE="https://data.binance.vision/data/futures/um/monthly"

def get(url:str)->bytes:
    with urllib.request.urlopen(url,timeout=90) as r:
        return r.read()

def kline_month(sym,dt):
    url=f"{BASE}/klines/{sym}/1h/{sym}-1h-{dt.year}-{dt.month:02d}.zip"
    blob=get(url)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names=[n for n in z.namelist() if n.endswith(".csv")]
        if len(names)!=1: raise RuntimeError(f"KLINE_SCHEMA {sym} {dt.date()} names={names}")
        df=pd.read_csv(z.open(names[0]),header=None)
    if str(df.iloc[0,0]).strip().lower() in {"open time","open_time"}:
        df=df.iloc[1:].reset_index(drop=True)
    keep=df.iloc[:,[0,1,2,3,4,5,9]].copy()
    keep.columns=["timestamp","open","high","low","close","volume","taker_buy_base_volume"]
    keep["timestamp"]=pd.to_numeric(keep["timestamp"],errors="coerce")
    for c in ["open","high","low","close","volume","taker_buy_base_volume"]:
        keep[c]=pd.to_numeric(keep[c],errors="coerce")
    keep["timestamp"]=pd.to_datetime(keep["timestamp"],unit="ms",utc=True)
    return keep.dropna().drop_duplicates("timestamp").sort_values("timestamp")

def funding_month(sym,dt):
    url=f"{BASE}/fundingRate/{sym}/{sym}-fundingRate-{dt.year}-{dt.month:02d}.zip"
    blob=get(url)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names=[n for n in z.namelist() if n.endswith(".csv")]
        if len(names)!=1: raise RuntimeError(f"FUNDING_SCHEMA {sym} {dt.date()} names={names}")
        df=pd.read_csv(z.open(names[0]))
    lower={str(c).lower():c for c in df.columns}
    tcol=next((lower[k] for k in ["calc_time","fundingtime","funding_time"] if k in lower),None)
    rcol=next((lower[k] for k in ["last_funding_rate","fundingrate","funding_rate"] if k in lower),None)
    if tcol is None or rcol is None:
        raise RuntimeError(f"FUNDING_SCHEMA {sym} cols={list(df.columns)}")
    out=pd.DataFrame({"timestamp":pd.to_datetime(pd.to_numeric(df[tcol],errors="coerce"),unit="ms",utc=True),
                      "fundingRate":pd.to_numeric(df[rcol],errors="coerce")})
    return out.dropna().drop_duplicates("timestamp").sort_values("timestamp")

def fetch_symbol(sym):
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        frames=list(ex.map(lambda dt:kline_month(sym,dt), MONTHS))
    d=pd.concat(frames,ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp")
    p=DATA/f"{sym}_1h.csv.gz"; d.to_csv(p,index=False,compression="gzip")
    return p

def fetch_funding(sym):
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        frames=list(ex.map(lambda dt:funding_month(sym,dt), MONTHS))
    d=pd.concat(frames,ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp")
    p=DATA/f"{sym}_funding.csv.gz"; d.to_csv(p,index=False,compression="gzip")
    return p

paths=[]
with cf.ThreadPoolExecutor(max_workers=8) as ex:
    paths.extend(ex.map(fetch_symbol,SYMS))
with cf.ThreadPoolExecutor(max_workers=3) as ex:
    paths.extend(ex.map(fetch_funding,FUNDING))
manifest={p.name:{"rows":int(pd.read_csv(p,nrows=1_000_000).shape[0]),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths}
root=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
(DATA/"DATA_MANIFEST.json").write_text(json.dumps({"experiment_id":"V88-CYCLE6-H16-H19-1H-FRESH-20260929","root_sha256":root,"files":manifest},indent=2)+"\n")
print(json.dumps({"root_sha256":root,"files":len(paths)},indent=2))
