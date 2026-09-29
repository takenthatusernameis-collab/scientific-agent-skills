import json
from pathlib import Path
R=Path("scratch/chatgpt_v88/deep_edge_117")
freeze=json.loads((R/"freeze_stage2.json").read_text())
expected={x["candidate_id"] for x in freeze["oos_candidates"]}
shards=sorted((R/"oos").glob("shard_*.json"))
if len(shards)!=8:
    raise SystemExit(f"OOS_SHARD_COUNT_FAILURE:{len(shards)}")
rows=[]
for p in shards:
    rows.extend(json.loads(p.read_text()).get("rows",[]))
ids=[x["candidate_id"] for x in rows]
if len(ids)!=len(expected) or len(set(ids))!=len(ids) or set(ids)!=expected:
    raise SystemExit("OOS_CANDIDATE_UNION_FAILURE")
passes=[x for x in rows if x["oos_pass"]]
for x in passes:x["oos_score"]=x["base"]["sharpe"]+0.5*x["stress"]["sharpe"]
passes.sort(key=lambda x:x["oos_score"],reverse=True)
selected=passes[:12]
out={"stage":"OOS","input_count":len(rows),"pass_count":len(passes),"holdout_candidates":selected,"holdout_seen":False}
(R/"freeze_oos.json").write_text(json.dumps(out,indent=2))
print(json.dumps({"input_count":len(rows),"pass_count":len(passes),"holdout_count":len(selected)},indent=2))
