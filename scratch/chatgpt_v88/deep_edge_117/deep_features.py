import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
R=Path("scratch/chatgpt_v88/deep_edge_117");D=R/"data";OUT=R/"features";OUT.mkdir(exist_ok=True)
syms=["BTCUSDT","DOGEUSDT","LINKUSDT","AVAXUSDT","NEARUSDT","FILUSDT","ATOMUSDT","UNIUSDT","AAVEUSDT","INJUSDT","OPUSDT","STXUSDT","ARBUSDT","APTUSDT","SUIUSDT"]
oi_syms={"BTCUSDT","DOGEUSDT","LINKUSDT","AVAXUSDT"}
parts=[]
for s in syms:
 d=pd.read_parquet(D/f"{s}_daily.parquet").set_index("timestamp").sort_index()
 out=pd.DataFrame(index=d.index)
 out[f"{s}|close"]=d.close.astype("float64")
 out[f"{s}|volume"]=d.volume.astype("float64")
 out[f"{s}|quote_volume"]=d.quote_volume.astype("float64")
 if s in oi_syms:
  oi=pd.read_parquet(D/f"{s}_oi_daily.parquet").set_index("timestamp").sort_index().oi.astype("float64")
  out[f"{s}|oi"]=oi
 else: out[f"{s}|oi"]=np.nan
 out[f"{s}|oichg1"]=out[f"{s}|oi"].pct_change(1)
 for L in [3,5,10,20,40,80]:
  out[f"{s}|ret{L}"]=out[f"{s}|close"].pct_change(L)
  out[f"{s}|vol{L}"]=out[f"{s}|close"].pct_change().rolling(L,min_periods=max(3,L//2)).std()
  out[f"{s}|volz{L}"]=(out[f"{s}|volume"]-out[f"{s}|volume"].rolling(L,min_periods=max(3,L//2)).mean())/out[f"{s}|volume"].rolling(L,min_periods=max(3,L//2)).std().replace(0,np.nan)
  out[f"{s}|oichg{L}"]=out[f"{s}|oi"].pct_change(L)
  out[f"{s}|oiz{L}"]=(out[f"{s}|oichg1"]-out[f"{s}|oichg1"].rolling(L,min_periods=max(3,L//2)).mean())/out[f"{s}|oichg1"].rolling(L,min_periods=max(3,L//2)).std().replace(0,np.nan) if f"{s}|oichg1" in out else np.nan
 out[f"{s}|oichg1"]=out[f"{s}|oi"].pct_change(1)
 parts.append(out)
F=pd.concat(parts,axis=1).sort_index()
F=F.loc[(F.index>=pd.Timestamp("2023-01-01",tz="UTC"))&(F.index<=pd.Timestamp("2026-09-20",tz="UTC"))]
p=OUT/"FEATURES.parquet";F.to_parquet(p,compression="zstd")
meta={"rows":len(F),"columns":len(F.columns),"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"index_start":str(F.index.min()),"index_end":str(F.index.max())}
(OUT/"FEATURE_MANIFEST.json").write_text(json.dumps(meta,indent=2)+"\n");print(json.dumps(meta,indent=2))
