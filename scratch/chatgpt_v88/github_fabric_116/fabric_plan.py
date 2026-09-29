import json,hashlib
from pathlib import Path
R=Path("scratch/chatgpt_v88/github_fabric_116");c=json.loads((R/"inputs"/"candidate_matrix.json").read_text())
N=8
def cost(x):
 s=x["spec"];return 3*int(s["lookback"])+8*int(s["holding_bars"])+(6 if s["representation"]=="zscore" else 2)+(4 if s["state"]!="all" else 0)
b=[{"shard_id":str(i),"candidate_ids":[],"predicted_cost":0.0} for i in range(N)]
for x in sorted(c,key=cost,reverse=True):
 z=min(b,key=lambda q:(q["predicted_cost"],q["shard_id"]));z["candidate_ids"].append(x["candidate_id"]);z["predicted_cost"]+=cost(x)
m=json.dumps(b,separators=(",",":"));(R/"plan").mkdir(exist_ok=True);(R/"plan"/"matrix.json").write_text(m)
(R/"plan"/"PLAN.json").write_text(json.dumps({"planner_mode":"DETERMINISTIC_LPT_BASELINE","ml_model_used":False,"shard_count":N,"candidate_count":len(c),"matrix_sha256":hashlib.sha256(m.encode()).hexdigest()},indent=2))
print(m)
