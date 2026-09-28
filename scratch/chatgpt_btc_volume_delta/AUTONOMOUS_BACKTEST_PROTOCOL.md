# Autonomous Backtest Completion Protocol — V2

## Current target
Execute the BTC volume-delta V2 controlled historical-data backtest on the disposable scientific-agent-skills branch.

## Completion artifact
A verified real-data result artifact containing:
- actual Binance/official archive-derived market data;
- dataset hashes;
- frozen 2024-06-01 to 2026-09-20 UTC window;
- training/OOS separation;
- selected parameters from training only;
- one-shot OOS result;
- higher-cost OOS stress result;
- trade count and provenance;
- reproducible artifact hashes.

## State machine
1. NOT_STARTED -> activate the backtest immediately.
2. RUNNING -> inspect the run and repair only the failed layer.
3. SUCCEEDED -> download/verify artifacts and record result.
4. FAILED -> classify the failure and retry through the deterministic fallback route.
5. TECHNICAL_BLOCK -> synthetic fixtures may validate plumbing, but the empirical backtest remains uncompleted until real historical data are processed.

## No-manual-input rule
Never wait for workflow_dispatch when a push-triggered disposable execution route is available.
Never ask the user to select a branch, workflow, retry, or connector route.

## Backtest-vs-synthetic rule
Synthetic data may replace only execution plumbing, schemas, parser fixtures, or other technical dependencies.
Synthetic data may never satisfy the real-data backtest completion condition.

## Anti-drift rule
Do not redirect to V88 statistical qualification, BTC baseline re-runs, AKE/Pionex experiments, or unrelated research. Those are reference material only.

## Scientific gates
Do not change the frozen strategy, universe, costs, dates, train/OOS boundary, candidate grid, or selection rule merely to make execution pass.

## Final status
Use:
- EMPIRICAL_BACKTEST_SUCCESS
- SYNTHETIC_TECHNICAL_QUALIFICATION_ONLY
- BACKTEST_BLOCKED_MISSING_REAL_DATA
- EXECUTION_FAILURE
