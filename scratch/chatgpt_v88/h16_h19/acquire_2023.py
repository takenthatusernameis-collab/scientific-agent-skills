import gzip,hashlib,io,json,os,urllib.parse,urllib.request,zipfile
from pathlib import Path
import pandas as pd

R=Path(__file__).resolve().parent
DATA=R/"h16_h19_data"
BASE="https://data.binance.vision/data/futures/um/monthly/klines"
SYMBOLS=["BTCUSDT","ETHUSDT","SOLUSDT","APTUSDT","ARBUSDT","OPUSDT","SEIUSDT","SUIUSDT","STXUSDT","INJUSDT","RUNEUSDT"]
FUNDING_SYMBOLS=["BTCUSDT","ETHUSDT","SOLUSDT"]
START=pd.Timestamp("2023-01-01",tz="UTC")
END=pd.Timestamp("2023-12-31 23:59:59",tz="UTC")

def download(url):
    with urllib.request.urlopen(url,timeout=90) as r:
        return r.read()

def acquire_klines():
    DATA.mkdir(parents=True,exist_ok=True)
    manifest={}
    for sym in SYMBOLS:
        frames=[]
        for m in pd.date_range(START.normalize(),END.normalize(),freq="MS"):
            url=f"{BASE}/{sym}/15m/{sym}-15m-{m.year}-{m.month:02d}.zip"
            try:
                blob=download(url)
            except Exception as e:
                raise RuntimeError(f"KLINE_DOWNLOAD_FAILED {sym} {m.date()} {e}")
            with zipfile.ZipFile(io.BytesIO(blob)) as z:
                names=[n for n in z.namelist() if n.endswith(".csv")]
                if not names: raise RuntimeError(f"NO_CSV {sym} {m.date()}")
                df=pd.read_csv(z.open(names[0]),header=None)
            if str(df.iloc[0,0]).strip().lower() in {"open time","open_time"}:
                df=df.iloc[1:].reset_index(drop=True)
            keep=df.iloc[:,[0,1,2,3,4,5,9]].copy()
            keep.columns=["timestamp","open","high","low","close","volume","taker_buy_base_volume"]
            keep["timestamp"]=pd.to_numeric(keep["timestamp"],errors="coerce")
            for col in ["open","high","low","close","volume","taker_buy_base_volume"]:
                keep[col]=pd.to_numeric(keep[col],errors="coerce")
            keep["timestamp"]=pd.to_datetime(keep["timestamp"],unit="ms",utc=True)
            frames.append(keep)
        d=pd.concat(frames,ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp")
        d=d[(d.timestamp>=START)&(d.timestamp<=END)]
        p=DATA/f"{sym}.csv.gz"
        d.to_csv(p,index=False,compression="gzip")
        manifest[sym]={"rows":int(len(d)),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
    for sym in FUNDING_SYMBOLS:
        rows=[]; cursor=int(START.timestamp()*1000); end_ms=int(END.timestamp()*1000)
        while cursor<=end_ms:
            q=urllib.parse.urlencode({"symbol":sym,"startTime":cursor,"endTime":end_ms,"limit":1000})
            url=f"https://fapi.binance.com/fapi/v1/fundingRate?{q}"
            raw=json.loads(download(url).decode())
            if not raw: break
            rows.extend(raw)
            last=int(raw[-1]["fundingTime"])
            if last<=cursor: break
            cursor=last+1
            if len(raw)<1000: break
        fd=pd.DataFrame(rows)
        if fd.empty: raise RuntimeError(f"NO_FUNDING {sym}")
        fd["fundingTime"]=pd.to_numeric(fd["fundingTime"])
        fd["fundingTime"]=pd.to_datetime(fd["fundingTime"],unit="ms",utc=True)
        fd["fundingRate"]=pd.to_numeric(fd["fundingRate"],errors="coerce")
        fd=fd.drop_duplicates("fundingTime").sort_values("fundingTime")
        p=DATA/f"{sym}_funding.csv.gz"
        fd[["fundingTime","fundingRate"]].to_csv(p,index=False,compression="gzip")
        manifest[f"{sym}_funding"]={"rows":int(len(fd)),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
    root=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    (R/"H16_H19_DATA_MANIFEST.json").write_text(json.dumps({"start":str(START),"end":str(END),"root_sha256":root,"manifest":manifest},indent=2)+"
")
    print(json.dumps({"root_sha256":root,"symbols":len(SYMBOLS),"funding_symbols":len(FUNDING_SYMBOLS)},indent=2))

if __name__=="__main__": acquire_klines()
