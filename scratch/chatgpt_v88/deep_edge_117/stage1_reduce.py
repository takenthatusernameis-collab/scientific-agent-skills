import json
from pathlib import Path
R=Path("scratch/chatgpt_v88/deep_edge_117")

C=json.loads((R/"inputs/candidate_matrix.json").read_text())
expected_ids={x["candidate_id"] for x in C}
shards=sorted((R/"stage1").glob("shard_*.json"))
expected_shards=[R/"stage1"/f"shard_{i}.json" for i in range(64)]
if [p.name for p in shards] != [p.name for p in expected_shards]:
    raise SystemExit("STAGE1_SHARD_FILESET_FAILURE")

rows=[]
for i,p in enumerate(shards):
    obj=json.loads(p.read_text())
    if str(obj.get("shard_id")) != str(i):
        raise SystemExit(f"STAGE1_SHARD_ID_FAILURE:{p.name}")
    rws=obj.get("rows",[])
    if len(rws)!=64:
        raise SystemExit(f"STAGE1_SHARD_SIZE_FAILURE:{p.name}:{len(rws)}")
    rows.extend(rws)

ids=[x["candidate_id"] for x in rows]
if len(ids)!=4096 or len(set(ids))!=4096 or set(ids)!=expected_ids:
    raise SystemExit("STAGE1_CANDIDATE_UNION_FAILURE")

eligible=[x for x in rows if x["n"]>=150 and x["mean"]>0 and x["pf"]>1.05 and x["sharpe"]>0.5 and x["mdd"]>-0.60]
eligible.sort(key=lambda x:(x["sharpe"],x["pf"]),reverse=True)

picked=[];counts={}
for x in eligible:
    fam=C[next(y for y in C if y["candidate_id"]==x["candidate_id"])]["family"]
    counts.setdefault(fam,0)
    if counts[fam]>=12:
        continue
    picked.append(x);counts[fam]+=1
    if len(picked)>=128:break

out={"stage":"STAGE_1_VALIDATION","candidate_count":4096,"shard_count":64,"shard_size":64,"eligible_count":len(eligible),"robustness_candidates":picked,"family_counts":counts,"OOS_seen":False,"holdout_seen":False}
(R/"freeze_stage1.json").write_text(json.dumps(out,indent=2))
print(json.dumps({"eligible_count":len(eligible),"robustness_count":len(picked),"family_counts":counts},indent=2))
