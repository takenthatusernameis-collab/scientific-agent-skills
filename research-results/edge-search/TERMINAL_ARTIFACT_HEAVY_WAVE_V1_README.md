# Heavy-wave terminalization

The worker-side adapter is deliberately a single terminal step:

`scripts/terminalize_heavy_wave.py`

It consumes the existing producer artifacts and writes `terminal_manifest.json`. It does not rerun acquisition, shards, reduction, candidate selection, or metrics.

Recommended final workflow step:

```yaml
- name: Terminalize immutable research artifact
  if: always()
  run: |
    python scripts/terminalize_heavy_wave.py \
      --root results/V88-HEAVY-WAVE-01 \
      --experiment-id V88-HEAVY-WAVE-01 \
      --scientific-contract preregistration/contract.json \
      --candidate-set coverage/candidate_set.json \
      --acquisition-ledger coverage/acquisition.json \
      --expected-acquisition-ledger coverage/expected_symbol_month.json \
      --feature-ledger coverage/features.json \
      --shard-ledger coverage/shards.json \
      --reducer-metadata reducer/reducer_metadata.json \
      --reducer-code reducer/reducer.py \
      --expected-candidate-count 4096 \
      --expected-shard-count 64 \
      --worker-repository "$GITHUB_REPOSITORY" \
      --worker-commit "$GITHUB_SHA" \
      --workflow-run-id "$GITHUB_RUN_ID"
```

The adapter fails closed on:

- missing or duplicate symbol×month acquisition units;
- acquisition content-hash mismatch;
- feature digest mismatch;
- missing/duplicate shard IDs;
- duplicate candidate IDs across shards;
- missing candidate IDs relative to the pinned 4096-candidate set;
- shard/result digest mismatch;
- reducer code/result mismatch;
- reducer shard/candidate count mismatch;
- reducer input-shard digest mismatch.

The resulting manifest is self-contained enough for the private control plane to reconcile execution coverage without rerunning the scientific search.

The worker should upload the entire `results/V88-HEAVY-WAVE-01` directory as its terminal artifact.

This layer proves execution completeness and immutable handoff. It does not prove that the research code is scientifically correct.

## Notes ownership

The public worker is a compute/evidence surface, not the private research notebook. It should emit immutable machine-readable terminal artifacts only. Durable narrative research notes, decisions, and agent handoffs belong in the private research repository/control plane after the artifact is independently reconciled.

## Private-control-plane handoff

Every terminalized heavy wave also emits:

`terminal_persistence_input.json`

This is a machine-only handoff descriptor. It carries the exact:

- `experiment_id`
- `terminal_status`
- `terminal_manifest_sha256`
- `artifact_digest`
- worker repository/commit/run provenance

and declares the four private records that must be created or reconciled:

`state`, `historian`, `process`, and `event`.

The sidecar is a projection of `terminal_manifest.json` and is deliberately excluded from the artifact digest to avoid circular hashing. It contains no private research conclusions.

The private control plane can therefore consume a standard terminal bundle without experiment-specific field mapping:

```text
terminal_manifest.json
terminal_persistence_input.json
    -> private reconciliation
       -> state + Historian + process + event
       -> terminal persistence verifier
```

The public worker remains compute/evidence-only. It does not write the private records.
