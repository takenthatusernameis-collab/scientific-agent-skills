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
      --reducer-result reducer/FINAL_RESULT.json \
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
