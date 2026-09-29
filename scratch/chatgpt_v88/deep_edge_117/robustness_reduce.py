import json
from pathlib import Path
R=Path("scratch/chatgpt_v88/deep_edge_117")
freeze=json.loads((R/"freeze_stage1.json").read_text())
expected={x["candidate_id"] for x in freeze["robustness_candidates"]}
shards=sorted((R/"robustness").glob("shard_*.json"))
if len(shards)!=16:
    raise SystemExit(f"ROBUSTNESS_SHARD_COUNT_FAILURE:{len(shards)}")
rows=[]
for p in shards:
    rows.extend(json.loads(p.read_text()).get("rows",[]))
ids=[x["candidate_id"] for x in rows]
if len(ids)!=len(expected) or len(set(ids))!=len(ids) or set(ids)!=expected:
    raise SystemExit("ROBUSTNESS_CANDIDATE_UNION_FAILURE")
passes=[x for x in rows if x["robust_pass"]]
for x in passes:x["robust_score"]=min(t["sharpe"] for t in x["tests"])
passes.sort(key=lambda x:x["robust_score"],reverse=True)
selected=passes[:32]
out={"stage":"STAGE_2_ROBUSTNESS","input_count":len(rows),"pass_count":len(passes),"oos_candidates":selected,"OOS_seen":False,"holdout_seen":False}
(R/"freeze_stage2.json").write_text(json.dumps(out,indent=2))
print(json.dumps({"input_count":len(rows),"pass_count":len(passes),"oos_count":len(selected)},indent=2))
