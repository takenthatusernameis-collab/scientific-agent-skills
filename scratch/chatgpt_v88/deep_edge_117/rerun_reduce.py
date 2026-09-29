import json
from pathlib import Path
R=Path("scratch/chatgpt_v88/deep_edge_117")
rows=[]
for p in sorted((R/"rerun").glob("shard_*.json")): rows.extend(json.loads(p.read_text())["rows"])
final=json.loads((R/"FINAL_EDGE_WAVE_SUMMARY.json").read_text())
checks=[]
for row in rows:
 orig=next(x for x in final["verified_candidates"] if x["candidate_id"]==row["candidate_id"])
 ob,rb=orig["base"],row["rerun_base"]; os_,rs=row["stress"],row["rerun_stress"]
 checks.append({
  "candidate_id":row["candidate_id"],
  "base_abs_mean_delta":abs(float(ob["mean"])-float(rb["mean"])),
  "base_abs_sharpe_delta":abs(float(ob["sharpe"])-float(rb["sharpe"])),
  "stress_abs_mean_delta":abs(float(os_["mean"])-float(rs["mean"])),
  "stress_abs_sharpe_delta":abs(float(os_["sharpe"])-float(rs["sharpe"]))
 })
passed=[x for x in checks if x["base_abs_mean_delta"]<1e-12 and x["base_abs_sharpe_delta"]<1e-9 and x["stress_abs_mean_delta"]<1e-12 and x["stress_abs_sharpe_delta"]<1e-9]
result={
 "prompt_version":"1.17.0",
 "experiment_id":"V88-1-17-DEEP-REAL-DATA-EDGE-WAVE",
 "independent_rerun_candidates":len(rows),
 "exact_reconciliations":len(passed),
 "edge_verified":len(rows)>0 and len(passed)==len(rows),
 "deployment_ready":False,
 "reason_if_not_ready":"Live deployment still requires execution-layer review and explicit human approval even after statistical confirmation.",
 "checks":checks
}
(R/"EDGE_VERIFICATION.json").write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
