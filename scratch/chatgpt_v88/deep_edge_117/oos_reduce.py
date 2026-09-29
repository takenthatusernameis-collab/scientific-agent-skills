import json
from pathlib import Path
R=Path("scratch/chatgpt_v88/deep_edge_117");rows=[]
for p in sorted((R/"oos").glob("shard_*.json")):rows.extend(json.loads(p.read_text())["rows"])
passes=[x for x in rows if x["oos_pass"]]
for x in passes:x["oos_score"]=x["base"]["sharpe"]+0.5*x["stress"]["sharpe"]
passes.sort(key=lambda x:x["oos_score"],reverse=True)
selected=passes[:12]
out={"stage":"OOS","input_count":len(rows),"pass_count":len(passes),"holdout_candidates":selected,"holdout_seen":False}
(R/"freeze_oos.json").write_text(json.dumps(out,indent=2));print(json.dumps({"input_count":len(rows),"pass_count":len(passes),"holdout_count":len(selected)},indent=2))
