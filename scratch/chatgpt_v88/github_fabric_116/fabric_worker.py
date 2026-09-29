import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np,pandas as pd
R=Path("scratch/chatgpt_v88/github_fabric_116");DATA=R/"inputs"/"data";a=argparse.ArgumentParser();a.add_argument("--shard-id",required=True);s=a.parse_args().shard_id
plan=json.loads((R/"plan"/"matrix.json").read_text());sh=next(x for x in plan if x["shard_id"]==s);wanted=set(sh["candidate_ids"]);c=json.loads((R/"inputs"/"candidate_matrix.json").read_text());sel=[x for x in c if x["candidate_id"] in wanted]
d={p.stem.replace("_1h",""):pd.read_parquet(p).set_index("timestamp").sort_index() for p in DATA.glob("*.parquet")};close=pd.DataFrame({k:d[k].close for k in ["BTCUSDT","ETHUSDT","SOLUSDT"]});vol=pd.DataFrame({k:d[k].volume for k in ["BTCUSDT","ETHUSDT","SOLUSDT"]})
def sig(x):
 s=x["spec"];L=int(s["lookback"]);m=s["mechanism"];ret=close.pct_change(L)
 if m=="time_series_momentum":z=ret
 elif m=="time_series_reversal":z=-ret
 elif m=="breakout_range":z=close/close.shift(1).rolling(L,min_periods=L).max()-1
 else:
  vr=vol.pct_change();mu=vr.shift(1).rolling(L,min_periods=max(5,L//3)).mean();sd=vr.shift(1).rolling(L,min_periods=max(5,L//3)).std();z=ret*((vr-mu)/sd.replace(0,np.nan))
 if s["representation"]=="zscore":
  mu=z.shift(1).rolling(max(8,L),min_periods=max(5,L//2)).mean();sd=z.shift(1).rolling(max(8,L),min_periods=max(5,L//2)).std();z=(z-mu)/sd.replace(0,np.nan)
 if s["state"]=="trend":z=z.where(ret.shift(1).abs()>ret.shift(1).abs().rolling(90,min_periods=30).quantile(.60))
 return z
out=[];t0=time.perf_counter()
for x in sel:
 q=time.perf_counter();a0=np.nan_to_num(sig(x).to_numpy());out.append({"candidate_id":x["candidate_id"],"runtime_s":time.perf_counter()-q,"checksum":hashlib.sha256(a0.tobytes()).hexdigest()})
m={"prompt_version":"1.16.0","experiment_id":"V88-1-16-GITHUB-COMPUTE-FABRIC-CANARY","shard_id":s,"candidate_ids":sorted(wanted),"candidate_count":len(sel),"predicted_cost":sh["predicted_cost"],"observed_runtime_s":time.perf_counter()-t0,"candidate_runtime_sum_s":sum(x["runtime_s"] for x in out),"github_run_id":os.getenv("GITHUB_RUN_ID"),"github_job_id":os.getenv("GITHUB_JOB"),"github_sha":os.getenv("GITHUB_SHA"),"runner_label":os.getenv("RUNNER_NAME"),"financial_performance_used":False,"results":out}
o=R/"shards"/s;o.mkdir(parents=True,exist_ok=True);(o/"SHARD_MANIFEST.json").write_text(json.dumps(m,indent=2));print(json.dumps({k:v for k,v in m.items() if k!="results"},indent=2))
