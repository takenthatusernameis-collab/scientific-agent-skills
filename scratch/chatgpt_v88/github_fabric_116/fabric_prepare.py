import hashlib,io,json,urllib.request,zipfile
from pathlib import Path
import pandas as pd
R=Path("scratch/chatgpt_v88/github_fabric_116"); D=R/"inputs"/"data"; D.mkdir(parents=True,exist_ok=True)
SYMS=["BTCUSDT","ETHUSDT","SOLUSDT"]; MONTHS=pd.date_range("2023-01-01","2023-06-01",freq="MS",tz="UTC")
ROOT=[
 {"mechanism":"time_series_momentum","lookback":12,"representation":"level","holding_bars":1,"state":"all"},
 {"mechanism":"time_series_reversal","lookback":24,"representation":"level","holding_bars":1,"state":"all"},
 {"mechanism":"breakout_range","lookback":48,"representation":"level","holding_bars":1,"state":"all"},
 {"mechanism":"volume_flow","lookback":24,"representation":"zscore","holding_bars":1,"state":"all"}]
def get(u):
 r=urllib.request.Request(u,headers={"User-Agent":"V88-1.16-GitHub-Compute-Fabric"})
 with urllib.request.urlopen(r,timeout=90) as x:return x.read()
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"))
def cid(x):return hashlib.sha256(canon(x).encode()).hexdigest()[:16]
def children(p):
 out=[]
 for dl in (-8,-4,4,8):
  q=dict(p);q["lookback"]=max(6,int(p["lookback"])+dl);out.append(q)
 q=dict(p);q["representation"]="zscore" if p["representation"]=="level" else "level";out.append(q)
 q=dict(p);q["state"]="trend" if p["state"]=="all" else "all";out.append(q)
 q=dict(p);q["holding_bars"]=2 if int(p["holding_bars"])==1 else 1;out.append(q)
 q=dict(p);q["lookback"]=max(6,int(p["lookback"])+12);out.append(q)
 return out[:(8 if int(p["lookback"])+2*int(p["holding_bars"])<=18 else 6 if int(p["lookback"])+2*int(p["holding_bars"])<=36 else 4)]
c=[]
for x in ROOT:c.append({"candidate_id":cid(x),"generation_id":0,"spec":x})
i=0;seen={x["candidate_id"] for x in c}
while i<len(c) and len(c)<166:
 x=c[i];i+=1
 if x["generation_id"]>=3:continue
 for q in children(x["spec"]):
  k=cid(q)
  if k in seen:continue
  seen.add(k);c.append({"candidate_id":k,"generation_id":x["generation_id"]+1,"spec":q})
  if len(c)>=166:break
for sym in SYMS:
 frames=[]
 for m in MONTHS:
  u=f"https://data.binance.vision/data/futures/um/monthly/klines/{sym}/1h/{sym}-1h-{m.year}-{m.month:02d}.zip"
  with zipfile.ZipFile(io.BytesIO(get(u))) as z:
   n=[n for n in z.namelist() if n.endswith(".csv")][0];d=pd.read_csv(z.open(n),header=None)
  if str(d.iloc[0,0]).lower() in {"open_time","open time"}:d=d.iloc[1:]
  d=d.iloc[:,[0,4,5]];d.columns=["timestamp","close","volume"]
  d["timestamp"]=pd.to_datetime(pd.to_numeric(d["timestamp"]),unit="ms",utc=True);d["close"]=pd.to_numeric(d["close"]);d["volume"]=pd.to_numeric(d["volume"])
  frames.append(d)
 d=pd.concat(frames).drop_duplicates("timestamp").sort_values("timestamp")
 d=d[(d.timestamp>=pd.Timestamp("2023-01-01",tz="UTC"))&(d.timestamp<=pd.Timestamp("2023-06-30 23:59:59",tz="UTC"))]
 d.to_parquet(D/f"{sym}_1h.parquet",index=False)
manifest={s:hashlib.sha256((D/f"{s}_1h.parquet").read_bytes()).hexdigest() for s in SYMS}
(R/"inputs"/"candidate_matrix.json").write_text(json.dumps(c))
(R/"inputs"/"DATA_MANIFEST.json").write_text(json.dumps(manifest,indent=2))
meta={"prompt_version":"1.16.0","experiment_id":"V88-1-16-GITHUB-COMPUTE-FABRIC-CANARY","candidate_count":len(c),"generation_counts":{str(g):sum(1 for x in c if x["generation_id"]==g) for g in sorted(set(x["generation_id"] for x in c))},"data_manifest_sha256":hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),"scientific_claim":False}
(R/"inputs"/"INPUT_MANIFEST.json").write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))
