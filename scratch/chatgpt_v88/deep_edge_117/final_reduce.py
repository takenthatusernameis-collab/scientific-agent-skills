import json
from pathlib import Path
R=Path("scratch/chatgpt_v88/deep_edge_117")
freeze=json.loads((R/"freeze_oos.json").read_text())
expected={x["candidate_id"] for x in freeze["holdout_candidates"]}
shards=sorted((R/"holdout").glob("shard_*.json"))
if len(shards)!=4:
    raise SystemExit(f"HOLDOUT_SHARD_COUNT_FAILURE:{len(shards)}")
rows=[]
for p in shards:
    rows.extend(json.loads(p.read_text()).get("rows",[]))
ids=[x["candidate_id"] for x in rows]
if len(ids)!=len(expected) or len(set(ids))!=len(ids) or set(ids)!=expected:
    raise SystemExit("HOLDOUT_CANDIDATE_UNION_FAILURE")
passes=[x for x in rows if x["holdout_pass"]]
for x in passes:x["confirmation_score"]=x["base"]["sharpe"]+0.5*x["stress"]["sharpe"]
passes.sort(key=lambda x:x["confirmation_score"],reverse=True)
verified=passes
summary={"prompt_version":"1.17.0","experiment_id":"V88-1-17-DEEP-REAL-DATA-EDGE-WAVE","validation_candidates":4096,"OOS_candidates_evaluated":len(expected),"holdout_candidates_evaluated":len(rows),"edge_candidates_verified":len(verified),"verified_candidates":verified,"scientific_claim":bool(verified),"holdout_used_for_selection":False,"holdout_used_for_confirmation":bool(verified),"deployment_ready":False,"requires_independent_rerun":bool(verified)}
(R/"FINAL_EDGE_WAVE_SUMMARY.json").write_text(json.dumps(summary,indent=2))
print(json.dumps({"edge_candidates_verified":len(verified),"scientific_claim":bool(verified)},indent=2))
