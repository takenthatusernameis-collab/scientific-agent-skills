#!/usr/bin/env python3
"""Independent V88 1.17 provenance and Stage-1 reconciliation audit.

No worker-generated manifest is treated as authoritative. The audit reconstructs
raw->normalized DATA, DATA->FEATURES, candidate IDs, runtime shard plan, and,
when present, every Stage-1 shard.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

R = Path("scratch/chatgpt_v88/deep_edge_117")
SYMS = ["BTCUSDT","DOGEUSDT","LINKUSDT","AVAXUSDT","NEARUSDT","FILUSDT","ATOMUSDT","UNIUSDT","AAVEUSDT","INJUSDT","OPUSDT","STXUSDT","ARBUSDT","APTUSDT","SUIUSDT"]
OI_SYMS = {"BTCUSDT","DOGEUSDT","LINKUSDT","AVAXUSDT"}
FAMILIES = ["TS_MOM","TS_REV","XS_MOM","XS_REV","BREAKOUT","BREAKDOWN","VOL_MOM","VOL_CONTRA","OI_SHOCK_CONT","OI_SHOCK_REV","OI_PRICE_DIV","OI_XS","BREADTH_MOM","BREADTH_REV","BTC_REL_MOM","VOL_BREAKOUT"]
LOOKBACKS = [5,10,20,40]
HOLDS = [1,3,5,10]
THRESH = [0.0,0.5,1.0,1.5]
QUANTS = [0.10,0.20,0.30,0.40]

def canonical(x): return json.dumps(x, sort_keys=True, separators=(",", ":"))
def cid(x): return hashlib.sha256(canonical(x).encode()).hexdigest()[:20]
def file_sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def frame_hash(df):
    x=df.sort_index()
    cols=sorted(map(str,x.columns)); x=x[cols]
    payload={
        "index":[v.isoformat() for v in x.index],
        "columns":cols,
        "dtypes":[str(x[c].dtype) for c in cols],
        "values":{c:[None if pd.isna(v) else float(v) for v in x[c].tolist()] for c in cols}
    }
    return hashlib.sha256(canonical(payload).encode()).hexdigest()

def expected_candidates():
    out=[]
    for fam in FAMILIES:
      for L in LOOKBACKS:
       for H in HOLDS:
        for T in THRESH:
         for Q in QUANTS:
          s={"family":fam,"lookback":L,"hold":H,"threshold":T,"quantile":Q,"universe":"full"}
          s["candidate_id"]=cid(s); out.append(s)
    assert len(out)==4096 and len({x["candidate_id"] for x in out})==4096
    return out

def normalize_kline(raw):
    d=raw.copy()
    tcol=next(c for c in d.columns if str(c).lower() in {"open_time","timestamp","datetime","open time"})
    d["timestamp"]=pd.to_datetime(d[tcol],utc=True)
    d["close"]=pd.to_numeric(d["close"],errors="coerce")
    vcol=next(c for c in d.columns if str(c).lower() in {"volume","base_asset_volume","base volume"})
    d["volume"]=pd.to_numeric(d[vcol],errors="coerce")
    qcol=next((c for c in d.columns if str(c).lower() in {"quote_asset_volume","quote volume","quote_volume"}),None)
    d["quote_volume"]=pd.to_numeric(d[qcol],errors="coerce") if qcol else d["close"]*d["volume"]
    return d[["timestamp","close","volume","quote_volume"]].dropna().drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

def normalize_oi(raw):
    d=raw.copy()
    tcol=next(c for c in d.columns if "timestamp" in str(c).lower() or "time" in str(c).lower())
    d["timestamp"]=pd.to_datetime(d[tcol],utc=True)
    low={str(c).lower():c for c in d.columns}
    oi_col=next((c for k,c in low.items() if "sum_open_interest" in k and "value" not in k),None)
    oiv_col=next((c for k,c in low.items() if "sum_open_interest_value" in k),None)
    assert oi_col is not None or oiv_col is not None
    src=oi_col or oiv_col
    d["oi"]=pd.to_numeric(d[src],errors="coerce")
    d=d[["timestamp","oi"]].dropna().drop_duplicates("timestamp").sort_values("timestamp")
    return d.set_index("timestamp").resample("1D").last().dropna().reset_index()

def rebuild_features():
    parts=[]
    for s in SYMS:
        d=pd.read_parquet(R/"data"/f"{s}_daily.parquet").set_index("timestamp").sort_index()
        o=pd.DataFrame(index=d.index)
        o[f"{s}|close"]=d.close.astype("float64")
        o[f"{s}|volume"]=d.volume.astype("float64")
        o[f"{s}|quote_volume"]=d.quote_volume.astype("float64")
        if s in OI_SYMS:
            oi=pd.read_parquet(R/"data"/f"{s}_oi_daily.parquet").set_index("timestamp").sort_index().oi.astype("float64")
            o[f"{s}|oi"]=oi
        else:
            o[f"{s}|oi"]=np.nan
        o[f"{s}|oichg1"]=o[f"{s}|oi"].pct_change(1)
        for L in LOOKBACKS:
            o[f"{s}|ret{L}"]=o[f"{s}|close"].pct_change(L)
            o[f"{s}|vol{L}"]=o[f"{s}|close"].pct_change().rolling(L,min_periods=max(3,L//2)).std()
            vm=o[f"{s}|volume"].rolling(L,min_periods=max(3,L//2)).mean()
            vs=o[f"{s}|volume"].rolling(L,min_periods=max(3,L//2)).std().replace(0,np.nan)
            o[f"{s}|volz{L}"]=(o[f"{s}|volume"]-vm)/vs
            o[f"{s}|oichg{L}"]=o[f"{s}|oi"].pct_change(L)
            oc=o[f"{s}|oichg1"]
            om=oc.rolling(L,min_periods=max(3,L//2)).mean()
            os=oc.rolling(L,min_periods=max(3,L//2)).std().replace(0,np.nan)
            o[f"{s}|oiz{L}"]=(oc-om)/os
        parts.append(o)
    F=pd.concat(parts,axis=1).sort_index()
    return F.loc[(F.index>=pd.Timestamp("2023-01-01",tz="UTC"))&(F.index<=pd.Timestamp("2026-09-20",tz="UTC"))]

def main():
    E=expected_candidates(); expected_ids=[x["candidate_id"] for x in E]
    C=json.loads((R/"inputs/candidate_matrix.json").read_text())
    candidate_ok=[x["candidate_id"] for x in C]==expected_ids

    P=json.loads((R/"plan/stage1_matrix.json").read_text())
    plan_ok=(len(P)==64 and [str(x["shard_id"]) for x in P]==[str(i) for i in range(64)]
             and [cid for x in P for cid in x["candidate_ids"]]==expected_ids
             and all(len(x["candidate_ids"])==64 for x in P))

    raw_to_data={}
    for s in SYMS:
        raw=normalize_kline(pd.read_parquet(R/"raw"/f"{s}_klines_1d.parquet"))
        obs=pd.read_parquet(R/"data"/f"{s}_daily.parquet")
        raw_to_data[s]={
            "match": frame_hash(raw)==frame_hash(obs),
            "raw_rows":len(raw),"data_rows":len(obs),
            "raw_sha256":file_sha(R/"raw"/f"{s}_klines_1d.parquet"),
            "data_sha256":file_sha(R/"data"/f"{s}_daily.parquet")
        }
    oi_raw_to_data={}
    for s in sorted(OI_SYMS):
        raw=normalize_oi(pd.read_parquet(R/"raw"/f"{s}_metrics.parquet"))
        obs=pd.read_parquet(R/"data"/f"{s}_oi_daily.parquet")
        oi_raw_to_data[s]={
            "match": frame_hash(raw)==frame_hash(obs),
            "raw_rows":len(raw),"data_rows":len(obs),
            "raw_sha256":file_sha(R/"raw"/f"{s}_metrics.parquet"),
            "data_sha256":file_sha(R/"data"/f"{s}_oi_daily.parquet")
        }

    rebuilt=rebuild_features()
    observed=pd.read_parquet(R/"features"/"FEATURES.parquet")
    feature_hash_rebuilt=frame_hash(rebuilt)
    feature_hash_observed=frame_hash(observed)
    feature_ok=(rebuilt.shape==observed.shape and list(rebuilt.columns)==list(observed.columns)
                and rebuilt.index.equals(observed.index) and feature_hash_rebuilt==feature_hash_observed)

    shard_files=sorted((R/"stage1").glob("shard_*.json"), key=lambda p:int(p.stem.split("_")[1])) if (R/"stage1").exists() else []
    stage1_status="NOT_PUBLISHED"
    stage1_ok=False
    shard_checks={}
    if shard_files:
        stage1_status="PUBLISHED"
        stage1_ok=(len(shard_files)==64)
        for i,p in enumerate(shard_files):
            obj=json.loads(p.read_text())
            ids=[row["candidate_id"] for row in obj.get("rows",[])]
            ok=(p.name==f"shard_{i}.json" and str(obj.get("shard_id"))==str(i)
                and len(ids)==64 and ids==expected_ids[i*64:(i+1)*64])
            shard_checks[p.name]={"ok":ok,"row_count":len(ids),"sha256":file_sha(p)}
            stage1_ok &= ok

    result={
      "status":"PASS" if candidate_ok and plan_ok and all(v["match"] for v in raw_to_data.values()) and all(v["match"] for v in oi_raw_to_data.values()) and feature_ok and (stage1_status!="PUBLISHED" or stage1_ok) else "FAIL",
      "candidate_matrix_exact":candidate_ok,
      "plan_exact_64x64":plan_ok,
      "raw_to_data_exact":all(v["match"] for v in raw_to_data.values()) and all(v["match"] for v in oi_raw_to_data.values()),
      "feature_reconstruction_exact":feature_ok,
      "feature_hash_observed":feature_hash_observed,
      "feature_hash_rebuilt":feature_hash_rebuilt,
      "stage1_status":stage1_status,
      "stage1_exact_64x64":stage1_ok if stage1_status=="PUBLISHED" else None,
      "raw_to_data":raw_to_data,
      "oi_raw_to_data":oi_raw_to_data,
      "candidate_count":4096,
      "expected_candidate_id_list_sha256":hashlib.sha256("\n".join(expected_ids).encode()).hexdigest()
    }
    if shard_checks: result["stage1_shards"]=shard_checks
    Path(R/"PROVENANCE_INDEPENDENT_VERIFICATION.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
    if result["status"]!="PASS": raise SystemExit(1)

if __name__=="__main__":
    main()
