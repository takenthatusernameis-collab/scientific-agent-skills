import json,hashlib
from pathlib import Path
R=Path("scratch/chatgpt_v88/deep_edge_117")
rows=[]
for p in sorted((R/"stage1").glob("shard_*.json")):rows.extend(json.loads(p.read_text())["rows"])
if len(rows)!=4096 or len({x["candidate_id"] for x in rows})!=4096:raise SystemExit("STAGE1_INTEGRITY_FAILURE")
eligible=[x for x in rows if x["n"]>=150 and x["mean"]>0 and x["pf"]>1.05 and x["sharpe"]>0.5 and x["mdd"]>-0.60]
eligible.sort(key=lambda x:(x["sharpe"],x["pf"]),reverse=True)
# Diversity cap: no more than 12 per mechanism family; candidate family is recovered from matrix.
C={x["candidate_id"]:x for x in json.loads((R/"inputs/candidate_matrix.json").read_text())}
picked=[];counts={}
for x in eligible:
 fam=C[x["candidate_id"]]["family"];counts.setdefault(fam,0)
 if counts[fam]>=12:continue
 picked.append(x);counts[fam]+=1
 if len(picked)>=128:break
out={"stage":"STAGE_1_VALIDATION","candidate_count":4096,"eligible_count":len(eligible),"robustness_candidates":picked,"family_counts":counts,"OOS_seen":False,"holdout_seen":False}
(R/"freeze_stage1.json").write_text(json.dumps(out,indent=2));print(json.dumps({"eligible_count":len(eligible),"robustness_count":len(picked),"family_counts":counts},indent=2))
