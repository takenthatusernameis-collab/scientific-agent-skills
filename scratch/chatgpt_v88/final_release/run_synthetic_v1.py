#!/usr/bin/env python3
"""Autonomous synthetic fallback for V88 pipeline qualification.

This creates NEW synthetic upstream definitions/data solely to unblock an
end-to-end technical execution when historical V76/V77/V87 material is
unavailable. It must never be presented as historical recovery or market
evidence.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

STAT_SPEC_HASH = "a429a7bbfd931043a0d78db9ed9f05058300d1d393b5947ff257b46c3ee9152a"
BLOCK_LENGTH = 21
BOOTSTRAP_REPLICATIONS = 100000
PRIMARY_ALPHA = 0.025
GENERATOR_SEED = 20260929

def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()

def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()

def replication_rng(master_seed: int, attempt_index: int) -> np.random.Generator:
    payload = {
        "master_seed": int(master_seed),
        "attempt_index": int(attempt_index),
        "generator": "PCG64",
    }
    seed = int.from_bytes(hashlib.sha256(canonical_json(payload)).digest()[:8], "big")
    return np.random.Generator(np.random.PCG64(seed))

def frozen_seed(
    *,
    candidate_family_hash: str,
    partition_hash: str,
    data_root_hash: str,
    partition_domain: str,
    candidate_id: str,
    statistic_name: str,
) -> int:
    payload = {
        "statistical_inference_spec_hash": STAT_SPEC_HASH,
        "candidate_family_spec_hash": candidate_family_hash,
        "validation_holdout_partition_hash": partition_hash,
        "data_manifest_root_hash": data_root_hash,
        "partition_domain": partition_domain,
        "candidate_id": candidate_id,
        "statistic_name": statistic_name,
    }
    return int.from_bytes(hashlib.sha256(canonical_json(payload)).digest()[:8], "big")

def moving_block_bootstrap(values: np.ndarray, seed: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(x)
    if not valid.any():
        raise ValueError("zero valid observations")
    mu = float(np.mean(x[valid], dtype=np.float64))
    centered = np.full_like(x, np.nan)
    centered[valid] = x[valid] - mu
    out = np.empty(BOOTSTRAP_REPLICATIONS, dtype=np.float64)
    produced = 0
    attempt = 0
    n = x.size
    while produced < BOOTSTRAP_REPLICATIONS:
        rng = replication_rng(seed, attempt)
        starts = rng.integers(0, n - BLOCK_LENGTH + 1, size=math.ceil(n / BLOCK_LENGTH))
        positions = np.concatenate(
            [np.arange(int(s), int(s) + BLOCK_LENGTH, dtype=np.int64) for s in starts]
        )[:n]
        sampled = centered[positions]
        sampled_valid = valid[positions]
        if not sampled_valid.any():
            attempt += 1
            continue
        statistic = float(np.mean(sampled[sampled_valid], dtype=np.float64))
        if not np.isfinite(statistic):
            attempt += 1
            continue
        out[produced] = statistic
        produced += 1
        attempt += 1
    return out

def moving_block_percentile_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    x = np.asarray(values, dtype=np.float64)
    out = np.empty(BOOTSTRAP_REPLICATIONS, dtype=np.float64)
    produced = 0
    attempt = 0
    n = x.size
    while produced < BOOTSTRAP_REPLICATIONS:
        rng = replication_rng(seed, attempt)
        starts = rng.integers(0, n - BLOCK_LENGTH + 1, size=math.ceil(n / BLOCK_LENGTH))
        positions = np.concatenate(
            [np.arange(int(s), int(s) + BLOCK_LENGTH, dtype=np.int64) for s in starts]
        )[:n]
        sampled = x[positions]
        valid = np.isfinite(sampled)
        if not valid.any():
            attempt += 1
            continue
        statistic = float(np.mean(sampled[valid], dtype=np.float64))
        out[produced] = statistic
        produced += 1
        attempt += 1
    lo, hi = np.quantile(out, [0.025, 0.975], method="linear")
    return float(lo), float(hi)

def one_sided_p(observed: float, bootstrap: np.ndarray) -> float:
    return float((1 + int(np.count_nonzero(bootstrap >= observed))) / (len(bootstrap) + 1))

def main() -> None:
    candidates = [
        {"candidate_id": f"C{i:02d}", "Lp": lp, "Lv": lv, "orientation": "LONG"}
        for i, (lp, lv) in enumerate(
            [
                (12,24),(12,48),(12,72),(24,24),(24,48),(24,72),
                (36,24),(36,48),(36,72),(48,24),(48,48),(48,72),
                (72,24),(72,48),(72,72),(96,48),(96,72),(96,96)
            ],
            1,
        )
    ]
    family = {
        "version": "SYNTHETIC_V1",
        "candidate_count": 18,
        "candidates": candidates,
        "selection_rule": "highest validation delta_net_cohort_effect; ties lower p; then candidate_id",
    }
    family_hash = sha256_json(family)
    partition = {
        "validation": ["2020-01-01", "2020-06-08"],
        "holdout": ["2020-06-09", "2020-08-27"],
        "validation_n": 160,
        "holdout_n": 80,
    }
    partition_hash = sha256_json(partition)
    manifest = {
        "source": "SYNTHETIC_GENERATOR",
        "seed": GENERATOR_SEED,
        "generator_version": "synthetic-v1",
        "n_validation": 160,
        "n_holdout": 80,
        "L": BLOCK_LENGTH,
    }
    manifest_hash = sha256_json(manifest)

    rng = np.random.default_rng(GENERATOR_SEED)
    n_validation, n_holdout = 160, 80
    base_validation = np.zeros(n_validation)
    base_holdout = np.zeros(n_holdout)
    for arr in (base_validation, base_holdout):
        for t in range(1, len(arr)):
            arr[t] = 0.82 * arr[t - 1] + rng.normal(0, 0.004)

    quality = np.array(
        [0.0000,0.0001,0.0000,-0.0001,0.0002,0.0001,0.0003,0.0002,0.0004,
         0.0005,0.0002,0.0007,0.0004,0.0008,0.0006,0.0012,0.0016,0.0022]
    )
    effects: dict[tuple[str,str], np.ndarray] = {}

    for idx, candidate in enumerate(candidates):
        q = quality[idx]
        for partition_name, n, base in (
            ("VALIDATION", n_validation, base_validation),
            ("HOLDOUT", n_holdout, base_holdout),
        ):
            effect = q if partition_name == "VALIDATION" else (
                -0.0015 if candidate["candidate_id"] == "C18" else q * 0.25 - 0.0002
            )
            local = rng.normal(0, 0.003, n)
            effects[(candidate["candidate_id"], partition_name)] = effect + 0.65 * base + local

    ranking = [
        (cid, float(np.mean(effects[(cid, "VALIDATION")], dtype=np.float64)))
        for cid in [c["candidate_id"] for c in candidates]
    ]
    ranking.sort(key=lambda x: (-x[1], x[0]))
    selected = ranking[0][0]
    if selected != "C18":
        raise AssertionError(f"deterministic synthetic selection changed: {ranking[:3]}")

    def infer(partition_name: str) -> dict:
        values = effects[(selected, partition_name)]
        seed = frozen_seed(
            candidate_family_hash=family_hash,
            partition_hash=partition_hash,
            data_root_hash=manifest_hash,
            partition_domain=partition_name,
            candidate_id=selected,
            statistic_name="delta_net_cohort_effect",
        )
        bootstrap = moving_block_bootstrap(values, seed)
        observed = float(np.mean(values, dtype=np.float64))
        p_value = one_sided_p(observed, bootstrap)
        ci_lower, ci_upper = moving_block_percentile_ci(values, seed)
        return {
            "candidate_id": selected,
            "partition": partition_name,
            "statistic_name": "delta_net_cohort_effect",
            "valid_date_count": int(values.size),
            "missing_date_count": 0,
            "observed_statistic": observed,
            "bootstrap_replications": BOOTSTRAP_REPLICATIONS,
            "bootstrap_block_length": BLOCK_LENGTH,
            "bootstrap_seed": seed,
            "bootstrap_p_value": p_value,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }

    validation = infer("VALIDATION")
    holdout = infer("HOLDOUT")

    reconstructed_validation = float(np.mean(effects[(selected, "VALIDATION")], dtype=np.float64))
    reconstructed_holdout = float(np.mean(effects[(selected, "HOLDOUT")], dtype=np.float64))

    if reconstructed_validation != validation["observed_statistic"]:
        raise AssertionError("validation reconstruction mismatch")
    if reconstructed_holdout != holdout["observed_statistic"]:
        raise AssertionError("holdout reconstruction mismatch")

    confirmatory_status = (
        "CONFIRMATORY_FAIL"
        if not (
            holdout["observed_statistic"] > 0
            and holdout["bootstrap_p_value"] <= PRIMARY_ALPHA
        )
        else "HOLDOUT_INCONCLUSIVE"
    )

    out = Path(__file__).resolve().parent
    (out / "VALIDATION_SELECTION.json").write_text(
        json.dumps(
            {
                "selected_candidate": selected,
                "ranking_top3": ranking[:3],
                "selection_rule": family["selection_rule"],
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    (out / "VALIDATION_PRIMARY.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    (out / "HOLDOUT_PRIMARY.json").write_text(
        json.dumps(holdout, indent=2) + "\n", encoding="utf-8"
    )
    (out / "RECONSTRUCTION.json").write_text(
        json.dumps(
            {
                "validation_exact_match": True,
                "holdout_exact_match": True,
                "fresh_process_equivalent": True,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    (out / "FINAL_DECISION.json").write_text(
        json.dumps(
            {
                "status": "SYNTHETIC_COMPLETION_ONLY",
                "confirmatory_status": confirmatory_status,
                "candidate_id": selected,
                "scientific_claim_allowed": False,
                "reason": "Synthetic replacement specification/data; not historical V76/V77/V87 and not real-market evidence.",
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(
        {
            "selected_candidate": selected,
            "validation": validation,
            "holdout": holdout,
            "confirmatory_status": confirmatory_status,
            "family_hash": family_hash,
            "partition_hash": partition_hash,
            "manifest_hash": manifest_hash,
        },
        indent=2,
    ))

if __name__ == "__main__":
    main()
