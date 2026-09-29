#!/usr/bin/env python3
"""Terminalize a sharded research run into terminal_manifest.json.

This adapter is intentionally producer-agnostic: it reads existing acquisition,
feature, shard, candidate-ledger, and reducer artifacts and never reruns the
scientific computation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


HEX64 = 64


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel_file(root: Path, path: Path) -> str:
    resolved = path.resolve()
    root_resolved = root.resolve()
    try:
        return resolved.relative_to(root_resolved).as_posix()
    except ValueError as exc:
        raise ValueError(f"path escapes artifact root: {path}") from exc


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_ids(values: list[str]) -> list[str]:
    if any(not isinstance(v, str) or not v for v in values):
        raise ValueError("candidate IDs must be non-empty strings")
    if len(values) != len(set(values)):
        raise ValueError("candidate IDs contain duplicates")
    return sorted(values)


def hash_declared_files(root: Path, items: list[dict[str, Any]]) -> None:
    for item in items:
        path = root / str(item["path"])
        expected = str(item["sha256"])
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"digest mismatch: {item['path']}: expected {expected}, got {actual}")


def build_shard_digest(shards: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for shard in sorted(shards, key=lambda x: str(x["shard_id"])):
        h.update(str(shard["shard_id"]).encode())
        h.update(b"\0")
        h.update(str(shard["result_sha256"]).encode())
        h.update(b"\0")
    return h.hexdigest()


def terminal_digest(root: Path, manifest: dict[str, Any]) -> str:
    declared: list[dict[str, Any]] = []
    declared.extend(manifest.get("inputs", []))
    declared.extend(manifest.get("result_files", []))
    coverage = manifest.get("execution_coverage", {})
    declared.extend(coverage.get("feature_digests", []))
    declared.extend(coverage.get("acquisition_files", []))

    seen: set[str] = set()
    entries: list[tuple[str, str, int]] = []
    for item in declared:
        path = Path(str(item["path"])).as_posix()
        if path == "terminal_manifest.json" or path in seen:
            continue
        seen.add(path)
        file_path = root / path
        if file_path.is_file():
            entries.append((path, sha256_file(file_path), file_path.stat().st_size))

    for shard in coverage.get("shards", []):
        for key in ("candidate_ledger_path", "result_path"):
            path = Path(str(shard[key])).as_posix()
            if path in seen or path == "terminal_manifest.json":
                continue
            seen.add(path)
            file_path = root / path
            if file_path.is_file():
                entries.append((path, sha256_file(file_path), file_path.stat().st_size))

    candidate_path = coverage.get("candidate_set_path")
    if candidate_path:
        path = Path(str(candidate_path)).as_posix()
        if path not in seen:
            seen.add(path)
            file_path = root / path
            if file_path.is_file():
                entries.append((path, sha256_file(file_path), file_path.stat().st_size))

    h = hashlib.sha256()
    for path, digest, size in sorted(entries):
        h.update(path.encode())
        h.update(b"\0")
        h.update(digest.encode())
        h.update(b"\0")
        h.update(str(size).encode())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--scientific-contract", required=True, type=Path)
    ap.add_argument("--candidate-set", required=True, type=Path)
    ap.add_argument("--acquisition-ledger", required=True, type=Path)
    ap.add_argument("--expected-acquisition-ledger", required=True, type=Path)
    ap.add_argument("--feature-ledger", required=True, type=Path)
    ap.add_argument("--shard-ledger", required=True, type=Path)
    ap.add_argument("--reducer-result", required=True, type=Path)
    ap.add_argument("--reducer-code", required=True, type=Path)
    ap.add_argument("--expected-candidate-count", required=True, type=int)
    ap.add_argument("--expected-shard-count", required=True, type=int)
    ap.add_argument("--worker-repository", required=True)
    ap.add_argument("--worker-commit", required=True)
    ap.add_argument("--workflow-run-id")
    args = ap.parse_args()

    root = args.root.resolve()
    candidate_set = canonical_ids(load_json(args.candidate_set))
    expected_units = load_json(args.expected_acquisition_ledger)
    observed_units = load_json(args.acquisition_ledger)
    features = load_json(args.feature_ledger)
    shards = load_json(args.shard_ledger)
    reducer = load_json(args.reducer_result)

    if len(candidate_set) != args.expected_candidate_count:
        raise SystemExit("COVERAGE_FAILURE: candidate-set count mismatch")

    expected_ids = [str(x["unit_id"]) for x in expected_units]
    observed_ids = [str(x["unit_id"]) for x in observed_units]
    if len(expected_ids) != len(set(expected_ids)):
        raise SystemExit("COVERAGE_FAILURE: expected acquisition ledger contains duplicates")
    if len(observed_ids) != len(set(observed_ids)):
        raise SystemExit("COVERAGE_FAILURE: observed acquisition ledger contains duplicates")
    if set(expected_ids) != set(observed_ids):
        raise SystemExit("COVERAGE_FAILURE: symbol-month acquisition matrix mismatch")
    if any(str(x.get("status")) != "SUCCESS" for x in observed_units):
        raise SystemExit("COVERAGE_FAILURE: unsuccessful acquisition unit present")

    expected_candidate_set = set(candidate_set)
    covered: list[str] = []
    for shard in shards:
        ledger_path = root / str(shard["candidate_ledger_path"])
        actual_ledger_sha = sha256_file(ledger_path)
        if actual_ledger_sha != str(shard["candidate_ledger_sha256"]):
            raise SystemExit(f"INTEGRITY_FAILURE: shard candidate ledger digest mismatch: {shard['shard_id']}")
        ids = canonical_ids(load_json(ledger_path))
        if len(ids) != int(shard["candidate_count"]):
            raise SystemExit(f"COVERAGE_FAILURE: shard candidate count mismatch: {shard['shard_id']}")
        covered.extend(ids)
        result_path = root / str(shard["result_path"])
        if sha256_file(result_path) != str(shard["result_sha256"]):
            raise SystemExit(f"INTEGRITY_FAILURE: shard result digest mismatch: {shard['shard_id']}")

    if len(shards) != args.expected_shard_count:
        raise SystemExit("COVERAGE_FAILURE: shard count mismatch")
    if len(covered) != len(set(covered)):
        raise SystemExit("COVERAGE_FAILURE: candidate assigned to multiple shards")
    if set(covered) != expected_candidate_set:
        raise SystemExit("COVERAGE_FAILURE: shard candidate union != pinned candidate set")

    reducer_result_path = root / args.reducer_result
    if sha256_file(reducer_result_path) != str(reducer["result_sha256"]):
        raise SystemExit("INTEGRITY_FAILURE: reducer result digest mismatch")
    reducer_code_sha = sha256_file(root / args.reducer_code)
    if reducer_code_sha != str(reducer["reducer_code_sha256"]):
        raise SystemExit("INTEGRITY_FAILURE: reducer code digest mismatch")
    shard_digest = build_shard_digest(shards)
    if shard_digest != str(reducer["input_shards_digest"]):
        raise SystemExit("INTEGRITY_FAILURE: reducer input-shard digest mismatch")
    if int(reducer["candidate_count_seen"]) != args.expected_candidate_count:
        raise SystemExit("COVERAGE_FAILURE: reducer candidate count mismatch")
    if int(reducer["shard_count_seen"]) != args.expected_shard_count:
        raise SystemExit("COVERAGE_FAILURE: reducer shard count mismatch")

    contract_sha = sha256_file(args.scientific_contract)
    candidate_path = rel_file(root, args.candidate_set)
    coverage = {
        "profile": "heavy-search-v1",
        "expected_counts": {
            "candidate_count": args.expected_candidate_count,
            "shard_count": args.expected_shard_count,
            "acquisition_unit_count": len(expected_ids),
            "feature_count": len(features),
        },
        "observed_counts": {
            "candidate_count": len(candidate_set),
            "shard_count": len(shards),
            "acquisition_unit_count": len(observed_units),
            "feature_count": len(features),
        },
        "candidate_set_path": candidate_path,
        "candidate_set_sha256": sha256_file(args.candidate_set),
        "acquisition_units": observed_units,
        "feature_digests": features,
        "shards": shards,
        "reducer": {
            **reducer,
            "reducer_code_sha256": reducer_code_sha,
            "input_shards_digest": shard_digest,
        },
    }

    manifest = {
        "schema_version": "terminal-artifact-v1",
        "terminal_status": "PROMISING_BUT_UNVERIFIED",
        "experiment_id": args.experiment_id,
        "scientific_contract_sha256": contract_sha,
        "worker_repository": args.worker_repository,
        "worker_commit_sha": args.worker_commit,
        "workflow_run_id": args.workflow_run_id,
        "created_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "inputs": [
            {
                "id": "scientific_contract",
                "role": "config",
                "path": rel_file(root, args.scientific_contract),
                "sha256": contract_sha,
            }
        ],
        "selection_lock": {
            "selection_frozen": True,
            "candidate_set_sha256": coverage["candidate_set_sha256"],
            "oos_selection_allowed": False,
        },
        "execution_coverage": coverage,
        "result_files": [
            {
                "path": rel_file(root, reducer_result_path),
                "sha256": reducer["result_sha256"],
                "required": True,
            }
        ],
        "artifact_digest": {
            "algorithm": "sha256",
            "excluded_paths": ["terminal_manifest.json"],
            "digest_basis": "sorted path + NUL + sha256 + NUL + size",
        },
    }
    manifest["artifact_digest"]["sha256"] = terminal_digest(root, manifest)
    (root / "terminal_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
