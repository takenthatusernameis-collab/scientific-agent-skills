import hashlib,json
from pathlib import Path
import yaml

p=Path("research-results/edge-search/V88_HEAVY_WAVE_COVERAGE_MATRIX_1_10_0.yaml")
d=yaml.safe_load(p.read_text())
m=d["mechanisms"]; h=d["horizons"]; s=d["states"]; r=d["representations"]
expected=len(m)*len(h)*len(s)
assert expected==d["candidate_slots"], (expected,d["candidate_slots"])
assert len(set(m))==14 and len(set(h))==4 and len(set(s))==4 and len(set(r))==5
out={
 "wave_id":d["wave_id"],
 "prompt_version":d["prompt_version"],
 "prompt_sha":d["prompt_sha"],
 "expected_candidate_slots":expected,
 "mechanism_classes":len(set(m)),
 "horizons":len(set(h)),
 "market_states":len(set(s)),
 "representations":len(set(r)),
 "coverage_check":"PASS",
 "launch_mode":"DESIGN_ONCE_LAUNCH_ONCE_MONITOR",
 "scientific_execution":False,
 "preflight_only":True,
 "manifest_sha256":hashlib.sha256(p.read_bytes()).hexdigest()
}
Path("research-results/edge-search/V88_HEAVY_WAVE_COVERAGE_PREFLIGHT_1_10_0.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
