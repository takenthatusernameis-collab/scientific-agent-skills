# Real Historical Market-Data Backtest Activation

This branch is a disposable execution surface for the frozen BTC taker-buy
quote-volume delta baseline.

## Activation requirement

A run counts as activated only when:

- historical market data are fetched from Binance's public market-data API or
  Binance's official static historical archives;
- at least four pre-specified non-blue-chip altcoins are usable;
- all usable datasets carry SHA-256 digests;
- common 1-hour observations are complete after alignment;
- no synthetic market observations are used;
- the backtest produces a non-empty trade series;
- the output and trade artifacts are hashed and archived.

Synthetic data may be used only for software unit tests. It must never satisfy
the real-data activation gate.

## Frozen run

- period: 2024-06-01 through 2026-09-20 UTC
- signal: BTC taker-buy quote-volume delta z-score
- lookback: 48 bars
- threshold: ±2.0
- holding period: 6 bars
- fee: 5 bps per side
- slippage: 2 bps per side
- alternate basket: JTO, PYTH, WIF, SEI, TIA, ENA, ORDI, JUP
- Parquet dependency: none

This is a frozen baseline/activation execution, not parameter optimization.
