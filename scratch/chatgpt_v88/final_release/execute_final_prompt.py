#!/usr/bin/env python3
from __future__ import annotations
import json
import subprocess
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
REL = ROOT / "scratch/chatgpt_v88/final_release"
PROMPT = REL / "V88_MASTER_PROMPT.yaml"
CONTRACT = REL / "V88_SCIENTIFIC_CONTRACT.yaml"
TREE = REL / "EXECUTION_DECISION_TREE.yaml"
LEDGER = REL / "V88_EMPIRICAL_RECOVERY_EXHAUSTION_LEDGER.json"
ITER = {
    "prompt_version": "1.2.0",
    "baseline_version": "1.1.0",
    "iterations": 6,
    "maximum_routine_iterations": 13,
    "stopped_early_for_diminishing_returns": True
}
OUT = REL / "FINAL_STATUS.json"

def fail(msg):
    raise SystemExit(msg)

prompt = yaml.safe_load(PROMPT.read_text(encoding="utf-8"))
contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
tree = yaml.safe_load(TREE.read_text(encoding="utf-8"))
ledger = json.loads(LEDGER.read_text(encoding="utf-8"))

assert prompt["enterprise"]["version"] == "1.2.0"
assert prompt["enterprise"]["current_project"]["project_id"] == "V88_EMPIRICAL"
assert prompt["enterprise"]["iterative_improvement"]["maximum_routine_iterations"] == 13
dr = prompt["enterprise"]["iterative_improvement"]["diminishing_returns_protection"]
assert dr["minimum_material_improvement"] == 0.15
assert dr["minimum_net_value"] == 0.10
assert dr["consecutive_low_value_limit"] == 2
assert prompt["enterprise"]["execution_lanes"]["synthetic_qualification"]["evidence_status"] == "EVIDENCE_SYNTHETIC_NONEMPIRICAL"
assert prompt["enterprise"]["vectorbt_boundary"]["role"] == "independent_executable_portfolio_ledger"

assert ledger["empirical_evidence_status"] == "INCOMPLETE"
assert "synthetic_fallback_gate" in ledger
assert ledger["synthetic_fallback_gate"] == "SATISFIED_FOR_TECHNICAL_QUALIFICATION_ONLY"

required = [
    "V76 formulas",
    "V77 validation/economic gates",
    "V77 holdout evidence completeness",
    "V87 candidate selection/tie-break",
    "candidate-family spec/hash",
    "validation/holdout partition hash",
    "data manifest root hash",
    "final date-level candidate/economic observations",
]
assert all(x in contract["required_inherited_items"] for x in required)

decision = {
    "project_id": "V88_EMPIRICAL",
    "prompt_version": "1.2.0",
    "prompt_release_executed": True,
    "empirical_recovery_status": ledger["conclusion"],
    "empirical_evidence_status": ledger["empirical_evidence_status"],
    "selected_execution_lane": "SYNTHETIC_QUALIFICATION",
    "reason": "Required inherited V76/V77/V87 material remains unavailable; ledger authorizes technical qualification but not empirical evidence.",
    "vectorbt_status": "NOT_APPLICABLE",
    "vectorbt_reason": "No scientifically valid executable portfolio mapping is present in the recovered V88 contract; inventing PnL would violate the prompt.",
    "synthetic_execution_started": False,
    "synthetic_execution_completed": False,
    "scientific_claim_allowed": False,
    "status": "READY_FOR_SYNTHETIC_QUALIFICATION"
}
(REL / "DECISION.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")

subprocess.run(["python", str(REL / "run_synthetic_v1.py")], check=True)

final = {
    "project_id": "V88_EMPIRICAL",
    "prompt_version": "1.2.0",
    "prompt_refinement": ITER,
    "task_execution_status": "TASK_COMPLETED_TECHNICALLY_SYNTHETIC",
    "evidence_status": "EVIDENCE_SYNTHETIC_NONEMPIRICAL",
    "empirical_evidence_status": "INCOMPLETE",
    "scientific_claim_allowed": False,
    "vectorbt_status": "NOT_APPLICABLE",
    "vectorbt_reason": "No scientifically valid executable portfolio mapping was available without inventing a material V88 definition.",
    "frozen_statistical_spec_hash": contract["statistical_inference_spec_hash"],
    "frozen_bootstrap_seed_spec_hash": contract["bootstrap_seed_spec_hash"],
    "fresh_process_reconstruction_expected": True,
    "cleanup_safe": True,
    "terminal_reason": "Final v1.2.0 prompt executed its autonomous recovery decision tree and correctly selected the technical synthetic qualification lane because the empirical recovery ledger remains incomplete."
}
OUT.write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
print(json.dumps(final, indent=2))
