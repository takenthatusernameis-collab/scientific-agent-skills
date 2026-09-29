#!/usr/bin/env python3
"""Terminalize a sharded research run into an immutable terminal manifest.

This adapter only reconciles existing producer artifacts. It never reruns the
scientific search, repairs data, selects candidates, or recalculates metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
        rel = resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"path escapes artifact root: {path}") from exc
    if rel == Path(".") or ".." in rel.parts:
        raise ValueError(f"invalid relative path: {path}")
    return rel.as_posix()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def unique_ids(values: list[str], label: str) -> list[str]:
    if any(not isinstance(v, str) or not v for v in values):
        raise SystemExit(f"SCHEMA_FAILURE: {label} must contain non-empty strings")
    if len(values) != len(set(values)):
        raise SystemExit(f"COVERAGE_FAILURE: duplicate IDs in {label}")
    return sorted(values)


def require_digest(root: Path, rel: str, expected: str, label: str) -> None:
    path = root / rel
    if not path.is_file():
        raise SystemExit(f"INTEGRITY_FAILURE: missing {label}: {rel}")
    actual = sha256_file(path)
    if actual != expected:
        raise SystemExit(f"INTEGRITY_FAILURE: {label} digest mismatch: {rel}")


def shard_input_digest(shards: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for shard in sorted(shards, key=lambda item: str(item["shard_id"])):
        h.update(str(shard["shard_id"]).encode())
        h.update(b"\0")
        h.update(str(shard["result_sha256"]).encode())
        h.update(b"\0")
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
    ap.add_argument("--reducer-metadata", required=True, type=Path)
    ap.add_argument("--reducer-code", required=True, type=Path)
    ap.add_argument("--expected-candidate-count", required=True, type=int)
    ap.add_argument("--expected-shard-count", required=True, type=int)
    ap.add_argument("--worker-repository", required=True)
    ap.add_argument("--worker-commit", required=True)
    ap.add_argument("--workflow-run-id")
    args = ap.parse_args()

    root = args.root.resolve()

    contract_rel = rel_file(root, args.scientific_contract)
    candidate_rel = rel_file(root, args.candidate_set)
    observed_acq_rel = rel_file(root, args.acquisition_ledger)
    expected_acq_rel = rel_file(root, args.expected_acquisition_ledger)
    feature_ledger_rel = rel_file(root, args.feature_ledger)
    shard_ledger_rel = rel_file(root, args.shard_ledger)
    reducer_metadata_rel = rel_file(root, args.reducer_metadata)
    reducer_code_rel = rel_file(root, args.reducer_code)

    candidate_set = unique_ids(load_json(args.candidate_set), "pinned candidate set")
    expected_units = load_json(args.expected_acquisition_ledger)
    observed_units = load_json(args.acquisition_ledger)
    features = load_json(args.feature_ledger)
    shards = load_json(args.shard_ledger)
    reducer = load_json(args.reducer_metadata)

    if not all(isinstance(x, list) for x in (expected_units, observed_units, features, shards)):
        raise SystemExit("SCHEMA_FAILURE: acquisition/feature/shard ledgers must be JSON arrays")
    if not isinstance(reducer, dict):
        raise SystemExit("SCHEMA_FAILURE: reducer metadata must be a JSON object")
    if len(candidate_set) != args.expected_candidate_count:
        raise SystemExit("COVERAGE_FAILURE: pinned candidate count mismatch")

    expected_unit_ids = unique_ids([str(x["unit_id"]) for x in expected_units], "expected acquisition units")
    observed_unit_ids = unique_ids([str(x["unit_id"]) for x in observed_units], "observed acquisition units")
    if set(expected_unit_ids) != set(observed_unit_ids):
        raise SystemExit("COVERAGE_FAILURE: symbol-month acquisition matrix mismatch")

    result_files: dict[str, dict[str, Any]] = {}

    def add_result(path: str, digest: str, required: bool = True) -> None:
        result_files[path] = {"path": path, "sha256": digest, "required": required}

    add_result(contract_rel, sha256_file(args.scientific_contract))
    add_result(candidate_rel, sha256_file(args.candidate_set))
    add_result(observed_acq_rel, sha256_file(args.acquisition_ledger))
    add_result(expected_acq_rel, sha256_file(args.expected_acquisition_ledger))
    add_result(feature_ledger_rel, sha256_file(args.feature_ledger))
    add_result(shard_ledger_rel, sha256_file(args.shard_ledger))
    add_result(reducer_metadata_rel, sha256_file(args.reducer_metadata))
    add_result(reducer_code_rel, sha256_file(args.reducer_code))

    acquisition_records = []
    for unit in observed_units:
        if unit.get("status") != "SUCCESS":
            raise SystemExit(f"COVERAGE_FAILURE: unsuccessful acquisition unit: {unit.get('unit_id')}")
        for key in ("unit_id", "symbol", "month", "path", "content_sha256", "row_count", "source"):
            if key not in unit:
                raise SystemExit(f"SCHEMA_FAILURE: acquisition unit missing {key}: {unit.get('unit_id')}")
        rel = rel_file(root, root / str(unit["path"]))
        require_digest(root, rel, str(unit["content_sha256"]), "acquisition artifact")
        add_result(rel, str(unit["content_sha256"]))
        acquisition_records.append({**unit, "path": rel})

    feature_records = []
    for feature in features:
        for key in ("feature_id", "path", "sha256", "transform_code_path", "transform_code_sha256", "source_input_ids"):
            if key not in feature:
                raise SystemExit(f"SCHEMA_FAILURE: feature missing {key}: {feature.get('feature_id')}")
        rel = rel_file(root, root / str(feature["path"]))
        transform_rel = rel_file(root, root / str(feature["transform_code_path"]))
        require_digest(root, rel, str(feature["sha256"]), "feature artifact")
        require_digest(root, transform_rel, str(feature["transform_code_sha256"]), "feature transform code")
        add_result(rel, str(feature["sha256"]))
        add_result(transform_rel, str(feature["transform_code_sha256"]))
        feature_records.append({**feature, "path": rel, "transform_code_path": transform_rel})

    expected_candidates = set(candidate_set)
    observed_candidates: list[str] = []
    normalized_shards = []
    shard_ids: set[str] = set()

    for shard in shards:
        for key in (
            "shard_id",
            "candidate_ledger_path",
            "candidate_ledger_sha256",
            "candidate_count",
            "input_digest",
            "result_path",
            "result_sha256",
            "status",
        ):
            if key not in shard:
                raise SystemExit(f"SCHEMA_FAILURE: shard missing {key}")
        sid = str(shard["shard_id"])
        if sid in shard_ids:
            raise SystemExit(f"COVERAGE_FAILURE: duplicate shard_id: {sid}")
        shard_ids.add(sid)
        if shard["status"] != "SUCCESS":
            raise SystemExit(f"COVERAGE_FAILURE: unsuccessful shard: {sid}")

        candidate_ledger_rel = rel_file(root, root / str(shard["candidate_ledger_path"]))
        result_rel = rel_file(root, root / str(shard["result_path"]))
        require_digest(root, candidate_ledger_rel, str(shard["candidate_ledger_sha256"]), "candidate ledger")
        require_digest(root, result_rel, str(shard["result_sha256"]), "shard result")

        ids = unique_ids(load_json(root / candidate_ledger_rel), f"candidate ledger {sid}")
        if len(ids) != int(shard["candidate_count"]):
            raise SystemExit(f"COVERAGE_FAILURE: shard candidate count mismatch: {sid}")
        observed_candidates.extend(ids)
        add_result(candidate_ledger_rel, str(shard["candidate_ledger_sha256"]))
        add_result(result_rel, str(shard["result_sha256"]))

        normalized_shards.append({
            **shard,
            "shard_id": sid,
            "candidate_ledger_path": candidate_ledger_rel,
            "result_path": result_rel,
        })

    if len(shards) != args.expected_shard_count:
        raise SystemExit("COVERAGE_FAILURE: shard count mismatch")
    if len(observed_candidates) != len(set(observed_candidates)):
        raise SystemExit("COVERAGE_FAILURE: candidate assigned to multiple shards")
    if set(observed_candidates) != expected_candidates:
        raise SystemExit("COVERAGE_FAILURE: shard candidate union != pinned candidate set")

    reducer_result_rel = rel_file(root, root / str(reducer["result_path"]))
    require_digest(root, reducer_result_rel, str(reducer["result_sha256"]), "reducer result")
    add_result(reducer_result_rel, str(reducer["result_sha256"]))

    reducer_code_sha = sha256_file(root / reducer_code_rel)
    if reducer_code_sha != str(reducer["reducer_code_sha256"]):
        raise SystemExit("INTEGRITY_FAILURE: reducer code digest mismatch")

    shard_digest = shard_input_digest(normalized_shards)
    if shard_digest != str(reducer["input_shards_digest"]):
        raise SystemExit("INTEGRITY_FAILURE: reducer input-shards digest mismatch")
    if int(reducer["candidate_count_seen"]) != args.expected_candidate_count:
        raise SystemExit("COVERAGE_FAILURE: reducer candidate count mismatch")
    if int(reducer["shard_count_seen"]) != args.expected_shard_count:
        raise SystemExit("COVERAGE_FAILURE: reducer shard count mismatch")

    result_file_list = [result_files[path] for path in sorted(result_files)]
    manifest = {
        "schema_version": "terminal-artifact-v1",
        "terminal_status": "PROMISING_BUT_UNVERIFIED",
        "experiment_id": args.experiment_id,
        "scientific_contract_sha256": sha256_file(args.scientific_contract),
        "worker_repository": args.worker_repository,
        "worker_commit_sha": args.worker_commit,
        "workflow_run_id": args.workflow_run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": [{
            "id": "scientific_contract",
            "role": "config",
            "path": contract_rel,
            "sha256": sha256_file(args.scientific_contract),
        }],
        "selection_lock": {
            "selection_frozen": True,
            "candidate_set_sha256": sha256_file(args.candidate_set),
            "oos_selection_allowed": False,
        },
        "execution_coverage": {
            "profile": "heavy-search-v1",
            "expected_counts": {
                "candidate_count": args.expected_candidate_count,
                "shard_count": args.expected_shard_count,
                "acquisition_unit_count": len(expected_unit_ids),
                "feature_count": len(feature_records),
            },
            "observed_counts": {
                "candidate_count": len(candidate_set),
                "shard_count": len(normalized_shards),
                "acquisition_unit_count": len(acquisition_records),
                "feature_count": len(feature_records),
            },
            "candidate_set_path": candidate_rel,
            "candidate_set_sha256": sha256_file(args.candidate_set),
            "expected_acquisition_ledger_path": expected_acq_rel,
            "expected_acquisition_ledger_sha256": sha256_file(args.expected_acquisition_ledger),
            "observed_acquisition_ledger_path": observed_acq_rel,
            "observed_acquisition_ledger_sha256": sha256_file(args.acquisition_ledger),
            "acquisition_units": acquisition_records,
            "feature_digests": feature_records,
            "shards": normalized_shards,
            "reducer": {
                **reducer,
                "result_path": reducer_result_rel,
                "reducer_metadata_path": reducer_metadata_rel,
                "reducer_metadata_sha256": sha256_file(args.reducer_metadata),
                "reducer_code_path": reducer_code_rel,
                "reducer_code_sha256": reducer_code_sha,
                "input_shards_digest": shard_digest,
            },
        },
        "result_files": result_file_list,
        "artifact_digest": {
            "algorithm": "sha256",
            "excluded_paths": ["terminal_manifest.json"],
            "digest_basis": "sorted path + NUL + sha256 + NUL + size",
        },
    }

    h = hashlib.sha256()
    for item in manifest["inputs"] + manifest["result_files"]:
        path = Path(str(item["path"])).as_posix()
        file_path = root / path
        h.update(path.encode())
        h.update(b"\0")
        h.update(sha256_file(file_path).encode())
        h.update(b"\0")
        h.update(str(file_path.stat().st_size).encode())
    manifest["artifact_digest"]["sha256"] = h.hexdigest()

    (root / "terminal_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
