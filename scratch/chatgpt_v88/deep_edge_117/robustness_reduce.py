import json
from pathlib import Path
R=Path("scratch/chatgpt_v88/deep_edge_117");rows=[]
for p in sorted((R/"robustness").glob("shard_*.json")):rows.extend(json.loads(p.read_text())["rows"])
passes=[x for x in rows if x["robust_pass"]]
# rank by worst-case Sharpe across robustness scenarios, then cap at 32 for OOS.
for x in passes:x["robust_score"]=min(t["sharpe"] for t in x["tests"])
passes.sort(key=lambda x:x["robust_score"],reverse=True)
selected=passes[:32]
out={"stage":"STAGE_2_ROBUSTNESS","input_count":len(rows),"pass_count":len(passes),"oos_candidates":selected,"OOS_seen":False,"holdout_seen":False}
(R/"freeze_stage2.json").write_text(json.dumps(out,indent=2));print(json.dumps({"input_count":len(rows),"pass_count":len(passes),"oos_count":len(selected)},indent=2))
