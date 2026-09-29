import hashlib,json,os
from pathlib import Path
import pandas as pd
from binance_vision import fetch_data
from types import SimpleNamespace

ROOT=Path("scratch/chatgpt_v88/deep_edge_117")
RAW=ROOT/"raw"; DATA=ROOT/"data"; ROOT.mkdir(parents=True,exist_ok=True); RAW.mkdir(exist_ok=True); DATA.mkdir(exist_ok=True)
START="2023-01-01"; END="2026-09-20"
SYMS=["BTCUSDT","DOGEUSDT","LINKUSDT","AVAXUSDT","NEARUSDT","FILUSDT","ATOMUSDT","UNIUSDT","AAVEUSDT","INJUSDT","OPUSDT","STXUSDT","ARBUSDT","APTUSDT","SUIUSDT"]
OI_SYMS=["BTCUSDT","DOGEUSDT","LINKUSDT","AVAXUSDT"]

LOOKBACKS=[5,10,20,40]; HOLDS=[1,3,5,10]; THRESH=[0.0,0.5,1.0,1.5]; QUANTS=[0.10,0.20,0.30,0.40]
FAMILIES=[
 "TS_MOM","TS_REV","XS_MOM","XS_REV","BREAKOUT","BREAKDOWN","VOL_MOM","VOL_CONTRA",
 "OI_SHOCK_CONT","OI_SHOCK_REV","OI_PRICE_DIV","OI_XS","BREADTH_MOM","BREADTH_REV","BTC_REL_MOM","VOL_BREAKOUT"
]

def canonical(x): return json.dumps(x,sort_keys=True,separators=(",",":"))
def cid(x): return hashlib.sha256(canonical(x).encode()).hexdigest()[:20]

def fetch_one(sym,typ,interval=None):
    out=RAW/f"{sym}_{typ}{('_'+interval) if interval else ''}"
    # Reuse an existing immutable source file when supplied by the runner's
    # cache or a prior workflow artifact. Only missing/corrupt files are fetched.
    if os.getenv("V88_REUSE_SOURCE_DATA","1") == "1" and out.is_file() and out.stat().st_size > 0:
        d=pd.read_parquet(out)
        return SimpleNamespace(data=d, output_path=str(out), failed=[], missing=[])
    r=fetch_data(ticker=sym,start_date=START,end_date=END,market="usdm",data_type=typ,interval=interval,
                 output_format="parquet",output_path=str(out),max_workers=8)
    if r.failed: raise RuntimeError(f"FAILED {sym} {typ}: {r.failed[:3]}")
    return r

def kline_daily(sym):
    r=fetch_one(sym,"klines","1d")
    d=r.data.copy()
    # binance-vision 0.1.0 normalizes timestamps/headers.
    tcol=next(c for c in d.columns if str(c).lower() in {"open_time","timestamp","datetime","open time"})
    d["timestamp"]=pd.to_datetime(d[tcol],utc=True)
    d["close"]=pd.to_numeric(d["close"],errors="coerce")
    vcol=next(c for c in d.columns if str(c).lower() in {"volume","base_asset_volume","base volume"})
    d["volume"]=pd.to_numeric(d[vcol],errors="coerce")
    qcol=next((c for c in d.columns if str(c).lower() in {"quote_asset_volume","quote volume","quote_volume"}),None)
    d["quote_volume"]=pd.to_numeric(d[qcol],errors="coerce") if qcol else d["close"]*d["volume"]
    d=d[["timestamp","close","volume","quote_volume"]].dropna().drop_duplicates("timestamp").sort_values("timestamp")
    d.to_parquet(DATA/f"{sym}_daily.parquet",index=False)
    return r

def metrics_daily(sym):
    r=fetch_one(sym,"metrics")
    d=r.data.copy()
    tcol=next(c for c in d.columns if "timestamp" in str(c).lower() or "time" in str(c).lower())
    d["timestamp"]=pd.to_datetime(d[tcol],utc=True)
    low={str(c).lower():c for c in d.columns}
    oi_col=next((c for k,c in low.items() if "sum_open_interest" in k and "value" not in k),None)
    oiv_col=next((c for k,c in low.items() if "sum_open_interest_value" in k),None)
    if oi_col is None and oiv_col is None: raise RuntimeError(f"NO_OI_COLUMN {sym} columns={list(d.columns)}")
    src=oi_col or oiv_col
    d["oi"]=pd.to_numeric(d[src],errors="coerce")
    d=d[["timestamp","oi"]].dropna().drop_duplicates("timestamp").sort_values("timestamp")
    d=d.set_index("timestamp").resample("1D").last().dropna().reset_index()
    d.to_parquet(DATA/f"{sym}_oi_daily.parquet",index=False)
    return r

manifest={}
for sym in SYMS:
    r=kline_daily(sym)
    manifest[f"{sym}_klines_daily"]={"rows":int(len(r.data)),"sha256":hashlib.sha256(Path(r.output_path).read_bytes()).hexdigest(),"missing":len(r.missing)}
for sym in OI_SYMS:
    r=metrics_daily(sym)
    manifest[f"{sym}_metrics"]={"rows":int(len(r.data)),"sha256":hashlib.sha256(Path(r.output_path).read_bytes()).hexdigest(),"missing":len(r.missing)}

expected_cardinality=len(FAMILIES)*len(LOOKBACKS)*len(HOLDS)*len(THRESH)*len(QUANTS)
assert expected_cardinality==4096, f"PREREGISTERED_CARDINALITY_MISMATCH:{expected_cardinality}"
candidates=[]
for fam in FAMILIES:
  for L in LOOKBACKS:
    for H in HOLDS:
      for T in THRESH:
        for Q in QUANTS:
          spec={"family":fam,"lookback":L,"hold":H,"threshold":T,"quantile":Q,"universe":"full"}
          spec["candidate_id"]=cid(spec); candidates.append(spec)
assert len(candidates)==expected_cardinality==4096

# Balanced static shards; Stage 1 uses exactly 64 shards x 64 candidates.
shards=[]
for i in range(64):
    part=candidates[i*64:(i+1)*64]
    shards.append({"shard_id":str(i),"candidate_ids":[x["candidate_id"] for x in part],"predicted_cost":float(sum(x["lookback"]*x["hold"] for x in part))})
(ROOT/"inputs").mkdir(exist_ok=True)
(ROOT/"plan").mkdir(exist_ok=True)
(ROOT/"inputs"/"candidate_matrix.json").write_text(json.dumps(candidates,separators=(",",":")))
(ROOT/"plan"/"stage1_matrix.json").write_text(json.dumps(shards,separators=(",",":")))
data_sha=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
meta={"prompt_version":"1.17.0","experiment_id":"V88-1-17-DEEP-REAL-DATA-EDGE-WAVE","stage":"STAGE_1_VALIDATION","start":START,"end":END,"symbols":SYMS,"oi_symbols":OI_SYMS,"candidate_count":4096,"family_count":len(FAMILIES),"shards":64,"data_manifest_sha256":data_sha,"scientific_claim":False,"OOS_seen":False,"holdout_seen":False}
(ROOT/"inputs"/"DATA_MANIFEST.json").write_text(json.dumps({"root_sha256":data_sha,"manifest":manifest},indent=2)+"\n")
(ROOT/"inputs"/"WAVE_MANIFEST.json").write_text(json.dumps(meta,indent=2)+"\n")
print(json.dumps(meta,indent=2))
