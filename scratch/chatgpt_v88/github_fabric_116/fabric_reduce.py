import json,os,statistics
from pathlib import Path
R=Path("scratch/chatgpt_v88/github_fabric_116");ms=[json.loads(x.read_text()) for x in sorted((R/"shards").glob("*/SHARD_MANIFEST.json"))]
ids=[i for m in ms for i in m["candidate_ids"]]
if len(ms)!=8 or len(ids)!=166 or len(set(ids))!=166:raise SystemExit("reducer_integrity_failure")
crit=max(m["observed_runtime_s"] for m in ms);total=sum(m["candidate_runtime_sum_s"] for m in ms);eff=total/(len(ms)*crit) if crit else 0
p=[m["predicted_cost"] for m in ms];a=[m["candidate_runtime_sum_s"] for m in ms];scale=statistics.fmean(a)/statistics.fmean(p)
err=sorted(abs(x-y*scale)/max(abs(y*scale),1e-9) for x,y in zip(a,p))[int(.95*len(a))-1]
s={"prompt_version":"1.16.0","experiment_id":"V88-1-16-GITHUB-COMPUTE-FABRIC-CANARY","scientific_claim":False,"financial_performance_used":False,"shard_count":8,"candidate_count":166,"critical_path_shard_s":crit,"sum_shard_compute_s":total,"parallelism_efficiency":eff,"p95_relative_shard_cost_error":err,"workflow_run_id":os.getenv("GITHUB_RUN_ID"),"head_sha":os.getenv("GITHUB_SHA"),"reducer_integrity":"PASS","shards":[{"shard_id":m["shard_id"],"predicted_cost":m["predicted_cost"],"runtime_s":m["observed_runtime_s"]} for m in ms]}
(R/"FINAL_SUMMARY.json").write_text(json.dumps(s,indent=2));print(json.dumps(s,indent=2))
