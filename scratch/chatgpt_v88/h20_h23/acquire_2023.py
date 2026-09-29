import hashlib, io, json, urllib.parse, urllib.request, zipfile
from pathlib import Path
import pandas as pd

R = Path(__file__).resolve().parent
DATA = R / "h20_h23_data"
START = pd.Timestamp("2023-01-01", tz="UTC")
END = pd.Timestamp("2023-12-31 23:59:59", tz="UTC")
PERP_SYMBOLS = ["SOLUSDT","APTUSDT","OPUSDT","STXUSDT","INJUSDT","DOGEUSDT","LINKUSDT","MATICUSDT"]
BASIS_SYMBOLS = ["BTCUSDT","ETHUSDT"]
FUNDING_SYMBOLS = PERP_SYMBOLS
FUT_BASE = "https://data.binance.vision/data/futures/um/monthly/klines"
SPOT_BASE = "https://data.binance.vision/data/spot/monthly/klines"

def download(url):
    with urllib.request.urlopen(url, timeout=90) as r:
        return r.read()

def monthly_klines(base, symbol):
    frames=[]
    for m in pd.date_range(START.normalize(), END.normalize(), freq="MS"):
        url=f"{base}/{symbol}/15m/{symbol}-15m-{m.year}-{m.month:02d}.zip"
        try:
            blob=download(url)
        except urllib.error.HTTPError as e:
            if e.code==404:
                print(f"SKIP_MISSING_ARCHIVE {url}")
                continue
            raise
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            names=[n for n in z.namelist() if n.endswith(".csv")]
            if not names:
                raise RuntimeError(f"NO_CSV {base} {sym} {m.date()}")
            df=pd.read_csv(z.open(names[0]), header=None)
        if str(df.iloc[0,0]).strip().lower() in {"open time","open_time"}:
            df=df.iloc[1:].reset_index(drop=True)
        keep=df.iloc[:,[0,1,2,3,4,5,9]].copy()
        keep.columns=["timestamp","open","high","low","close","volume","taker_buy_base_volume"]
        keep["timestamp"]=pd.to_numeric(keep["timestamp"],errors="coerce")
        for col in keep.columns[1:]:
            keep[col]=pd.to_numeric(keep[col],errors="coerce")
        keep["timestamp"]=pd.to_datetime(keep["timestamp"],unit="ms",utc=True)
        frames.append(keep)
    d=pd.concat(frames,ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp")
    return d[(d.timestamp>=START)&(d.timestamp<=END)].reset_index(drop=True)

def acquire():
    DATA.mkdir(parents=True,exist_ok=True)
    manifest={}
    for sym in sorted(set(PERP_SYMBOLS+BASIS_SYMBOLS)):
        d=monthly_klines(FUT_BASE,sym)
        p=DATA/f"perp_{sym}.csv.gz"; d.to_csv(p,index=False,compression="gzip")
        manifest[f"perp_{sym}"]={"rows":int(len(d)),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
    for sym in BASIS_SYMBOLS:
        d=monthly_klines(SPOT_BASE,sym)
        p=DATA/f"spot_{sym}.csv.gz"; d.to_csv(p,index=False,compression="gzip")
        manifest[f"spot_{sym}"]={"rows":int(len(d)),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
    for sym in FUNDING_SYMBOLS:
        rows=[]; cursor=int(START.timestamp()*1000); end_ms=int(END.timestamp()*1000)
        while cursor<=end_ms:
            q=urllib.parse.urlencode({"symbol":sym,"startTime":cursor,"endTime":end_ms,"limit":1000})
            raw=json.loads(download(f"https://fapi.binance.com/fapi/v1/fundingRate?{q}").decode())
            if not raw: break
            rows.extend(raw)
            last=int(raw[-1]["fundingTime"])
            if last<=cursor: break
            cursor=last+1
            if len(raw)<1000: break
        fd=pd.DataFrame(rows)
        if fd.empty: raise RuntimeError(f"NO_FUNDING {sym}")
        fd["fundingTime"]=pd.to_datetime(pd.to_numeric(fd["fundingTime"]),unit="ms",utc=True)
        fd["fundingRate"]=pd.to_numeric(fd["fundingRate"],errors="coerce")
        fd=fd.drop_duplicates("fundingTime").sort_values("fundingTime")
        p=DATA/f"funding_{sym}.csv.gz"; fd[["fundingTime","fundingRate"]].to_csv(p,index=False,compression="gzip")
        manifest[f"funding_{sym}"]={"rows":int(len(fd)),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
    root=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    (R/"H20_H23_DATA_MANIFEST.json").write_text(json.dumps({"start":str(START),"end":str(END),"root_sha256":root,"manifest":manifest},indent=2)+"\n")
    print(json.dumps({"root_sha256":root,"files":len(manifest)},indent=2))

if __name__=="__main__":
    acquire()
