#!/usr/bin/env python3
from __future__ import annotations
import json
import subprocess
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
PROMPT = ROOT / "scratch/chatgpt_v88/enterprise/V88_MASTER_PROMPT.yaml"
MANIFEST = ROOT / "scratch/chatgpt_v88/enterprise/PROMPT_RELEASE.json"
OUT = ROOT / "scratch/chatgpt_v88/guardrail_canary_result.json"

def get(mapping, *keys):
    cur = mapping
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur

manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
prompt = yaml.safe_load(PROMPT.read_text(encoding="utf-8"))

checks = {}

checks["enterprise_id"] = get(prompt,"enterprise","id") == manifest["enterprise_id"]
checks["enterprise_version"] = get(prompt,"enterprise","version") == manifest["enterprise_version"]
checks["project_id"] = get(prompt,"enterprise","current_project","project_id") == manifest["project_id"]
checks["execution_repository"] = get(prompt,"enterprise","current_project","execution_repository") == "takenthatusernameis-collab/scientific-agent-skills"
checks["serialized_execution"] = get(prompt,"enterprise","operating_model","concurrency") == "ONE_ACTIVE_ROLE_AT_A_TIME"
checks["fail_closed"] = get(prompt,"enterprise","operating_model","fail_closed") is True
checks["real_data_only"] = get(prompt,"enterprise","operating_model","empirical_execution") == "REAL_DATA_ONLY"

imp = get(prompt,"enterprise","iterative_improvement") or {}
dr = get(imp,"diminishing_returns_protection") or {}
checks["iteration_cap_13"] = get(imp,"maximum_routine_iterations") == 13
checks["iteration_14_prohibited"] = get(imp,"iteration_14") == "PROHIBITED"
checks["min_material_improvement"] = get(dr,"minimum_material_improvement") >= 0.15 if get(dr,"minimum_material_improvement") is not None else False
checks["min_net_value"] = get(dr,"minimum_net_value") >= 0.10 if get(dr,"minimum_net_value") is not None else False
checks["low_value_limit"] = get(dr,"consecutive_low_value_limit") == 2
checks["early_stop_no_defect"] = get(dr,"stop_when_no_open_material_defect") is True
checks["early_stop_redundant"] = get(dr,"stop_when_redundant_change") is True
checks["no_metric_chasing"] = "metric_chasing" in (get(imp,"forbidden") or [])
checks["freeze_rule"] = bool(get(imp,"freeze_rule"))

obs = get(prompt,"enterprise","observer_admin") or {}
vetoes = get(obs,"veto_triggers") or []
checks["observer_veto_project_drift"] = "project_drift" in vetoes
checks["observer_veto_prompt_drift"] = "prompt_drift" in vetoes
checks["observer_veto_lookahead"] = "lookahead" in vetoes
checks["observer_veto_oos"] = "OOS_contamination" in vetoes

emp = get(prompt,"enterprise","empirical_execution") or {}
checks["synthetic_empirical_forbidden"] = "empirical_performance" in (get(emp,"synthetic_data_forbidden") or [])
checks["vectorbt_present"] = "vectorbt" in (get(prompt,"enterprise") or {})
checks["handoff_guard"] = str(get(prompt,"enterprise","cleanup","rule") or "").startswith("If handoff is not confirmed")
checks["final_certification_present"] = bool(get(prompt,"enterprise","final_certification"))

blob_sha = subprocess.check_output(["git","hash-object",str(PROMPT)], text=True).strip()
checks["prompt_blob_sha_exact"] = blob_sha == manifest["expected_prompt_git_blob_sha"]

failed = [key for key, ok in checks.items() if not ok]

result = {
    "status": "PASS" if not failed else "FAIL",
    "project_id": manifest["project_id"],
    "enterprise_id": manifest["enterprise_id"],
    "prompt_blob_sha": blob_sha,
    "expected_prompt_blob_sha": manifest["expected_prompt_git_blob_sha"],
    "checks": checks,
    "failed_checks": failed,
    "canary_only": True,
    "empirical_result_claimed": False,
}

OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2, sort_keys=True))
raise SystemExit(0 if not failed else 1)
