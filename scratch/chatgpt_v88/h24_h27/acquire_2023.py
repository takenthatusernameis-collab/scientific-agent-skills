import hashlib,json
from pathlib import Path
import pandas as pd
from binance_vision import fetch_data

R=Path(__file__).resolve().parent
DATA=R/"data"; START="2023-01-01"; END="2023-03-31"
METRICS_SYMBOLS=["BTCUSDT","ETHUSDT","SOLUSDT","DOGEUSDT","INJUSDT","OPUSDT","STXUSDT","LINKUSDT"]
BOOK_SYMBOLS=["BTCUSDT","ETHUSDT"]
KLINE_SYMBOLS=METRICS_SYMBOLS

def grab(sym,typ,interval=None):
    kwargs={"ticker":sym,"start_date":START,"end_date":END,"market":"usdm","data_type":typ,"output_format":"parquet","max_workers":4}
    if interval: kwargs["interval"]=interval
    out=DATA/f"{sym}_{typ}{('_'+interval) if interval else ''}"
    r=fetch_data(output_path=str(out),**kwargs)
    if r.failed: raise RuntimeError(f"{sym} {typ} failed: {r.failed}")
    return r

def main():
    DATA.mkdir(parents=True,exist_ok=True); manifest={}
    for s in METRICS_SYMBOLS:
        r=grab(s,"metrics"); manifest[f"{s}_metrics"]={"rows":int(len(r.data)),"sha256":hashlib.sha256(Path(r.output_path).read_bytes()).hexdigest(),"columns":list(r.data.columns)}
    for s in BOOK_SYMBOLS:
        r=grab(s,"bookDepth"); manifest[f"{s}_bookDepth"]={"rows":int(len(r.data)),"sha256":hashlib.sha256(Path(r.output_path).read_bytes()).hexdigest(),"columns":list(r.data.columns)}
    for s in KLINE_SYMBOLS:
        r=grab(s,"klines","1h"); manifest[f"{s}_klines_1h"]={"rows":int(len(r.data)),"sha256":hashlib.sha256(Path(r.output_path).read_bytes()).hexdigest(),"columns":list(r.data.columns)}
    root=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    out={"experiment_id":"V88-CYCLE5-H24-H27","prompt_version":"1.9.5","start":START,"end":END,"manifest_root_sha256":root,"manifest":manifest}
    (R/"DATA_MANIFEST.json").write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out,indent=2))
if __name__=="__main__": main()
