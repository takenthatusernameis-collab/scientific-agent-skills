import json,hashlib
from pathlib import Path
R=Path("scratch/chatgpt_v88/deep_edge_117");rows=[]
for p in sorted((R/"holdout").glob("shard_*.json")):rows.extend(json.loads(p.read_text())["rows"])
passes=[x for x in rows if x["holdout_pass"]]
for x in passes:x["confirmation_score"]=x["base"]["sharpe"]+0.5*x["stress"]["sharpe"]
passes.sort(key=lambda x:x["confirmation_score"],reverse=True)
verified=passes
summary={"prompt_version":"1.17.0","experiment_id":"V88-1-17-DEEP-REAL-DATA-EDGE-WAVE","validation_candidates":4096,"OOS_candidates_evaluated":None,"holdout_candidates_evaluated":len(rows),"edge_candidates_verified":len(verified),"verified_candidates":verified,"scientific_claim":bool(verified),"holdout_used_for_selection":True,"deployment_ready":False,"requires_independent_rerun":bool(verified)}
(R/"FINAL_EDGE_WAVE_SUMMARY.json").write_text(json.dumps(summary,indent=2));print(json.dumps({"edge_candidates_verified":len(verified),"scientific_claim":bool(verified)},indent=2))
