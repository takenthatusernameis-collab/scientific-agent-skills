import hashlib, io, json, time, urllib.request, zipfile
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path("scratch/chatgpt_v88/recursive_113")
DATA=ROOT/"data"
ROOT.mkdir(parents=True, exist_ok=True)
DATA.mkdir(exist_ok=True)

START="2023-01-01"
END="2023-06-30 23:59:59"
SYMS=["BTCUSDT","ETHUSDT","SOLUSDT"]
MONTHS=pd.date_range("2023-01-01","2023-06-01",freq="MS",tz="UTC")
ROUND_TRIP_COST=0.0010
MAX_GENERATIONS=3
MAX_CANDIDATES=256
PER_PARENT_CHILDREN=8

ROOT_CANDIDATES=[
    {"mechanism":"time_series_momentum","lookback":12,"representation":"level","holding_bars":1,"state":"all"},
    {"mechanism":"time_series_reversal","lookback":24,"representation":"level","holding_bars":1,"state":"all"},
    {"mechanism":"breakout_range","lookback":48,"representation":"level","holding_bars":1,"state":"all"},
    {"mechanism":"volume_flow","lookback":24,"representation":"zscore","holding_bars":1,"state":"all"},
]

def get(url):
    req=urllib.request.Request(url,headers={"User-Agent":"V88-1.13-realdata-canary/1.0"})
    with urllib.request.urlopen(req,timeout=90) as r:
        return r.read()

def load_symbol(sym):
    frames=[]
    for m in MONTHS:
        u=f"https://data.binance.vision/data/futures/um/monthly/klines/{sym}/1h/{sym}-1h-{m.year}-{m.month:02d}.zip"
        blob=get(u)
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            names=[n for n in z.namelist() if n.endswith(".csv")]
            if not names: raise RuntimeError(f"NO_CSV {sym} {m}")
            d=pd.read_csv(z.open(names[0]),header=None)
        if str(d.iloc[0,0]).strip().lower() in {"open_time","open time"}:
            d=d.iloc[1:].reset_index(drop=True)
        d=d.iloc[:,[0,4,5]].copy()
        d.columns=["timestamp","close","volume"]
        d["timestamp"]=pd.to_datetime(pd.to_numeric(d["timestamp"]),unit="ms",utc=True)
        d["close"]=pd.to_numeric(d["close"],errors="coerce")
        d["volume"]=pd.to_numeric(d["volume"],errors="coerce")
        frames.append(d)
    d=pd.concat(frames,ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp")
    d=d[(d.timestamp>=pd.Timestamp(START,tz="UTC"))&(d.timestamp<=pd.Timestamp(END,tz="UTC"))]
    p=DATA/f"{sym}_1h.parquet"
    d.to_parquet(p,index=False)
    return {"rows":len(d),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}

def load_data():
    manifest={}
    for s in SYMS:
        manifest[s]=load_symbol(s)
    (ROOT/"DATA_MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return {s:pd.read_parquet(DATA/f"{s}_1h.parquet").set_index("timestamp").sort_index() for s in SYMS},manifest

def stats(r):
    x=np.asarray(r.dropna(),dtype=float)
    if x.size==0:
        return {"n":0,"mean":None,"sharpe":None,"pf":None,"mdd":None,"total_return":None}
    wins=x[x>0].sum(); losses=-x[x<0].sum()
    eq=np.cumprod(1+x)
    mdd=float(np.min(eq/np.maximum.accumulate(eq)-1))
    sd=float(x.std(ddof=1)) if x.size>1 else 0.0
    sharpe=float(np.sqrt(24*365)*x.mean()/sd) if sd>0 else None
    return {"n":int(x.size),"mean":float(x.mean()),
            "sharpe":sharpe,"pf":float(wins/losses) if losses>0 else None,
            "mdd":mdd,"total_return":float(eq[-1]-1)}

def make_signal(close,vol,c):
    L=int(c["lookback"])
    ret=close.pct_change(L)
    if c["mechanism"]=="time_series_momentum":
        sig=ret
    elif c["mechanism"]=="time_series_reversal":
        sig=-ret
    elif c["mechanism"]=="breakout_range":
        hi=close.shift(1).rolling(L,min_periods=L).max()
        sig=close/hi-1.0
    elif c["mechanism"]=="volume_flow":
        vr=vol.pct_change().replace([np.inf,-np.inf],np.nan)
        mu=vr.shift(1).rolling(L,min_periods=max(5,L//3)).mean()
        sd=vr.shift(1).rolling(L,min_periods=max(5,L//3)).std(ddof=1)
        sig=ret*((vr-mu)/sd.replace(0,np.nan))
    else:
        raise ValueError(c["mechanism"])

    if c["representation"]=="zscore":
        mu=sig.shift(1).rolling(max(8,L),min_periods=max(5,L//2)).mean()
        sd=sig.shift(1).rolling(max(8,L),min_periods=max(5,L//2)).std(ddof=1)
        sig=(sig-mu)/sd.replace(0,np.nan)

    if c["state"]=="high_vol":
        rv=close.pct_change().rolling(max(12,L),min_periods=max(6,L//2)).std()
        thr=rv.shift(1).rolling(90,min_periods=30).quantile(.60)
        sig=sig.where(rv.shift(1)>thr)
    elif c["state"]=="trend":
        tr=ret.abs()
        thr=tr.shift(1).rolling(90,min_periods=30).quantile(.60)
        sig=sig.where(tr.shift(1)>thr)

    return sig

def evaluate(data,c):
    close=pd.DataFrame({s:data[s].close for s in SYMS}).sort_index()
    vol=pd.DataFrame({s:data[s].volume for s in SYMS}).sort_index()
    sig=make_signal(close,vol,c)
    pos=np.sign(sig).fillna(0.0)
    denom=pos.abs().sum(axis=1).replace(0,np.nan)
    pos=pos.div(denom,axis=0).fillna(0.0)
    fwd=close.pct_change().shift(-int(c["holding_bars"]))
    gross=(pos*fwd).sum(axis=1)
    turnover=pos.diff().abs().sum(axis=1).fillna(pos.abs().sum(axis=1))
    net=gross-ROUND_TRIP_COST*turnover
    net=net[(net.index>=pd.Timestamp(START,tz="UTC"))&(net.index<=pd.Timestamp(END,tz="UTC"))]
    return stats(net)

def canonical(c):
    return json.dumps(c,sort_keys=True,separators=(",",":"))

def cid(c):
    return hashlib.sha256(canonical(c).encode()).hexdigest()[:16]

def children(parent,generation):
    p=dict(parent)
    out=[]
    # deterministic local neighborhood
    for dl in (-8,-4,4,8):
        q=dict(p); q["lookback"]=max(6,int(p["lookback"])+dl)
        out.append(("LOCAL_NEIGHBORHOOD",q))
    # orthogonal representations / states
    q=dict(p); q["representation"]="zscore" if p["representation"]=="level" else "level"
    out.append(("ORTHOGONAL_REPRESENTATION",q))
    q=dict(p); q["state"]="trend" if p["state"]=="all" else "all"
    out.append(("STATE_NEIGHBOR",q))
    q=dict(p); q["holding_bars"]=2 if int(p["holding_bars"])==1 else 1
    out.append(("HORIZON_NEIGHBOR",q))
    q=dict(p); q["lookback"]=max(6,int(p["lookback"])+12)
    out.append(("COVERAGE_GAP_FILL",q))
    # stable ordering, deterministic de-duplication
    seen=set(); ret=[]
    for op,q in out:
        k=canonical(q)
        if k not in seen:
            seen.add(k); ret.append((op,q))
    return ret[:PER_PARENT_CHILDREN]

data, data_manifest = load_data()
data_root=hashlib.sha256(json.dumps(data_manifest,sort_keys=True).encode()).hexdigest()

existing=[]
frontier_path=ROOT/"RESEARCH_FRONTIER.json"
if frontier_path.exists():
    try:
        existing=json.loads(frontier_path.read_text())
    except Exception:
        existing=[]

queue=deque()
seen=set()
rows=[]
if existing:
    for item in existing:
        queue.append(item)
        seen.add(item["candidate_id"])
else:
    for c in ROOT_CANDIDATES:
        cidv=cid(c)
        queue.append({"candidate_id":cidv,"parent_candidate_id":None,"root_matrix_id":"ROOT-113",
                      "generation_id":0,"expansion_operator":"ROOT","spec":c})
        seen.add(cidv)

generation_counts={}
processed=0
next_frontier=[]

while queue and processed < MAX_CANDIDATES:
    item=queue.popleft()
    gen=int(item["generation_id"])
    c=item["spec"]
    result=evaluate(data,c)
    row={
      "candidate_id":item["candidate_id"],
      "root_matrix_id":item["root_matrix_id"],
      "generation_id":gen,
      "parent_candidate_id":item["parent_candidate_id"],
      "expansion_operator":item["expansion_operator"],
      "family_hash":hashlib.sha256(canonical(c).encode()).hexdigest(),
      "spec":c,
      "status":"VALIDATION_COMPLETE",
      "stage":"EXECUTION_CANARY",
      "real_data":True,
      "data_manifest_sha256":data_root,
      "result":result,
    }
    rows.append(row)
    processed += 1
    generation_counts[gen]=generation_counts.get(gen,0)+1

    if gen < MAX_GENERATIONS-1 and processed < MAX_CANDIDATES:
        for op,q in children(c,gen):
            child_id=cid(q)
            if child_id in seen:
                continue
            seen.add(child_id)
            child={"candidate_id":child_id,"root_matrix_id":item["root_matrix_id"],
                   "generation_id":gen+1,"parent_candidate_id":item["candidate_id"],
                   "expansion_operator":op,"spec":q}
            queue.append(child)

# Persist remaining frontier for continuation; this is deliberately not an automatic infinite loop.
remaining=list(queue)
frontier_path.write_text(json.dumps(remaining,indent=2)+"\n")

summary={
  "prompt_version":"1.13.0",
  "prompt_execution_mode":"RECURSIVE_DURABLE_FRONTIER_CANARY",
  "scientific_claim":False,
  "root_candidates":len(ROOT_CANDIDATES),
  "candidates_processed":len(rows),
  "generation_counts":generation_counts,
  "frontier_remaining":len(remaining),
  "unique_candidate_ids":len({r["candidate_id"] for r in rows}),
  "duplicate_candidates_rejected":len(seen)-len(rows),
  "data_manifest_sha256":data_root,
  "data_manifest":data_manifest,
  "selection_firewall":"No OOS or holdout selection; canary validates execution architecture only.",
  "results":rows
}
(ROOT/"V88_1_13_RECURSIVE_CANARY_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
(ROOT/"PROCESS_LEARNING_1_13.json").write_text(json.dumps({
  "process_change_id":"V88-PC-54-RECURSIVE-LOGIC-FIRST-FABRIC",
  "logic_to_code_ratio":1.0,
  "compute_without_connector_intervention_fraction":1.0,
  "recursive_generation_yield":generation_counts,
  "frontier_remaining":len(remaining),
  "duplicate_candidates_rejected":len(seen)-len(rows),
  "scientific_claim":False
},indent=2)+"\n")
print(json.dumps({
  "processed":len(rows),
  "generation_counts":generation_counts,
  "frontier_remaining":len(remaining),
  "data_manifest_sha256":data_root
},indent=2))
