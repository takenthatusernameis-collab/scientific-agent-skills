#!/usr/bin/env python3
from __future__ import annotations
import json
import subprocess
from pathlib import Path
import sys
try:
    import yaml
except Exception as exc:
    print(json.dumps({"status":"FAIL","reason":f"PyYAML import failed: {exc}"}))
    raise SystemExit(2)

ROOT = Path(__file__).resolve().parents[2]
PROMPT = ROOT / "scratch/chatgpt_v88/enterprise/V88_MASTER_PROMPT.yaml"
MANIFEST = ROOT / "scratch/chatgpt_v88/enterprise/PROMPT_RELEASE.json"
OUT = ROOT / "scratch/chatgpt_v88/guardrail_canary_result.json"

manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
prompt = yaml.safe_load(PROMPT.read_text(encoding="utf-8"))

checks = {}
checks["enterprise_id"] = prompt["enterprise"]["id"] == manifest["enterprise_id"]
checks["enterprise_version"] = prompt["enterprise"]["version"] == manifest["enterprise_version"]
checks["project_id"] = prompt["enterprise"]["current_project"]["project_id"] == manifest["project_id"]
checks["execution_repository"] = prompt["enterprise"]["current_project"]["execution_repository"] == "takenthatusernameis-collab/scientific-agent-skills"
checks["serialized_execution"] = prompt["enterprise"]["operating_model"]["concurrency"] == "ONE_ACTIVE_ROLE_AT_A_TIME"
checks["fail_closed"] = prompt["enterprise"]["operating_model"]["fail_closed"] is True
checks["real_data_only"] = prompt["enterprise"]["operating_model"]["empirical_execution"] == "REAL_DATA_ONLY"

imp = prompt["enterprise"]["iterative_improvement"]
checks["iteration_cap_13"] = imp["maximum_routine_iterations"] == 13
checks["iteration_14_prohibited"] = imp["iteration_14"] == "PROHIBITED"
checks["min_material_improvement"] = imp["scoring"]["minimum_material_improvement"] >= 0.15
checks["min_net_value"] = imp["scoring"]["minimum_net_value"] >= 0.10
checks["low_value_limit"] = imp["scoring"]["consecutive_low_value_limit"] == 2
checks["early_stop_no_defect"] = imp["scoring"]["stop_when_no_open_material_defect"] is True
checks["no_metric_chasing"] = "metric_chasing" in imp["scoring"]["forbidden"]
checks["freeze_rule"] = "freeze" in imp["scoring"]

obs = prompt["enterprise"]["observer_admin"]
checks["observer_veto_project_drift"] = "project_drift" in obs["veto_triggers"]
checks["observer_veto_prompt_drift"] = "prompt_drift" in obs["veto_triggers"]
checks["observer_veto_lookahead"] = "lookahead" in obs["veto_triggers"]
checks["observer_veto_oos"] = "OOS_contamination" in obs["veto_triggers"]

emp = prompt["enterprise"]["empirical_execution"]
checks["synthetic_empirical_forbidden"] = "empirical_performance" in emp["synthetic_data_forbidden"]
checks["vectorbt_present"] = "vectorbt" in prompt["enterprise"]
checks["handoff_guard"] = prompt["enterprise"]["cleanup"]["rule"].startswith("If handoff is not confirmed")
checks["final_certification_present"] = "final_certification" in prompt["enterprise"]

blob_sha = subprocess.check_output(["git","hash-object",str(PROMPT)], text=True).strip()
checks["prompt_blob_sha_exact"] = blob_sha == manifest["expected_prompt_git_blob_sha"]

failed = [k for k,v in checks.items() if not v]
result = {
    "status": "PASS" if not failed else "FAIL",
    "project_id": manifest["project_id"],
    "enterprise_id": manifest["enterprise_id"],
    "prompt_blob_sha": blob_sha,
    "expected_prompt_blob_sha": manifest["expected_prompt_git_blob_sha"],
    "checks": checks,
    "failed_checks": failed,
    "canary_only": True,
    "empirical_result_claimed": False
}
OUT.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n", encoding="utf-8")
print(json.dumps(result, indent=2, sort_keys=True))
raise SystemExit(0 if not failed else 1)
