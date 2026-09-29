import ast, hashlib, json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "research-results/edge-search/V88_CYCLE3_H16_H19_EXECUTION_RELEASE_1_12_2.json"
EXPECTED_VERSION = "1.12.2"
EXPECTED_SHA = "01d55dbe7c790ae91c8424f535e5ac68d52b9dea"

def main():
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["prompt_control_release"]["version"] == EXPECTED_VERSION
    assert manifest["prompt_control_release"]["blob_sha"] == EXPECTED_SHA
    assert manifest["scientific_registration"]["matrix_frozen"] is True

    paths = [Path(p) for p in manifest["execution_closure"]]
    assert Path(__file__).resolve().relative_to(ROOT) in paths
    observed = {}
    for rel in paths:
        p = ROOT / rel
        assert p.exists(), f"MISSING_EXECUTION_CLOSURE:{rel}"
        src = p.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(p))
        compile(tree, str(p), "exec")
        forbidden = sorted({n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id in {"true","false","null"}})
        assert not forbidden, f"FORBIDDEN_JSON_TOKEN_IDENTIFIERS:{rel}:{forbidden}"
        observed[str(rel)] = {
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "bytes": p.stat().st_size,
        }

    closure_json = json.dumps(observed, sort_keys=True, separators=(",", ":")).encode()
    closure_sha = hashlib.sha256(closure_json).hexdigest()
    out = {
        "preflight_only": True,
        "scientific_execution": False,
        "prompt_version": EXPECTED_VERSION,
        "prompt_sha": EXPECTED_SHA,
        "execution_closure": observed,
        "execution_closure_digest": closure_sha,
        "matrix_frozen": True,
        "python_semantic_preflight": "PASS",
    }
    out_path = ROOT / "research-results/edge-search/V88_CYCLE3_H16_H19_PREFLIGHT_1_12_2.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
