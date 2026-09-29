# Terminal Artifact v1 — Worker Manifest

This template is intentionally provider-neutral and contains no private control-plane details.

The worker should materialize `terminal_manifest.json` only after the declared result files exist and their SHA-256 digests have been computed.

The private Quantitative-Laboratory verifier can then reconcile the bundle without rerunning the scientific search.

Important distinctions:

- A valid manifest proves artifact integrity and declared provenance, not trading performance.
- `selection_lock.oos_selection_allowed` must remain false for confirmatory OOS evidence.
- `shards` and `reducer` are optional for single-run experiments but provide the same reconciliation path for sharded heavy searches.
- Synthetic data may be represented only when the surrounding experiment explicitly classifies them as technical-only; synthetic data must never be presented as empirical market evidence.
