#!/usr/bin/env python3
"""
Android-triggerable, non-Parquet Bitcoin Volume Delta research runner.

The strategy is deliberately frozen for this execution harness:
- 1h Binance spot klines.
- BTCUSDT taker-buy quote volume is converted to bar delta:
    delta = 2 * taker_buy_quote_volume - total_quote_volume
- Signal is the current BTC delta z-score over the previous LOOKBACK bars.
- A +THRESHOLD signal opens an equal-weight long basket of pre-specified
  non-blue-chip altcoins on the next bar open.
- A -THRESHOLD signal opens an equal-weight short basket.
- Position is held for HOLD_BARS bars.
- One position at a time; no overlapping trades.
- Costs are applied on both entry and exit.

This is an execution-path test and a concrete baseline backtest, not a claim
that the parameterization is optimal. No parameter search is performed.
No Parquet dependency exists.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict
from datetime import datetime, timezone


API_CANDIDATES = [
    "https://data-api.binance.vision/api/v3/klines",
    "https://api1.binance.com/api/v3/klines",
    "https://api2.binance.com/api/v3/klines",
    "https://api3.binance.com/api/v3/klines",
    "https://api4.binance.com/api/v3/klines",
    "https://api-gcp.binance.com/api/v3/klines",
    "https://api.binance.com/api/v3/klines",
]
ALTCOINS = [
    "JTOUSDT",
    "PYTHUSDT",
    "WIFUSDT",
    "SEIUSDT",
    "TIAUSDT",
    "ENAUSDT",
    "ORDIUSDT",
    "JUPUSDT",
]


@dataclass(frozen=True)
class Bar:
    ts: int
    open: float
    high: float
    low: float
    close: float
    quote_volume: float
    taker_buy_quote_volume: float

    @property
    def delta(self) -> float:
        return 2.0 * self.taker_buy_quote_volume - self.quote_volume


def get_json(url: str, retries: int = 4) -> object:
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "ChatGPT-BTC-Volume-Delta-Runner/1.0"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            last_exc = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries} attempts: {url} :: {last_exc}")


def ms(dt: str) -> int:
    return int(datetime.fromisoformat(dt.replace("Z", "+00:00")).timestamp() * 1000)


def fetch_klines(symbol: str, start_ms: int, end_ms: int, interval: str = "1h") -> tuple[list[Bar], str]:
    rows: list[Bar] = []
    cursor = start_ms
    raw_chunks: list[bytes] = []
    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": 1000,
        }
        endpoint_errors: list[str] = []
        payload = None
        for base in API_CANDIDATES:
            url = base + "?" + urllib.parse.urlencode(params)
            try:
                payload = get_json(url, retries=1)
                break
            except Exception as exc:
                endpoint_errors.append(f"{base}: {exc}")
        if payload is None:
            raise RuntimeError(f"{symbol}: all Binance public market-data endpoints failed; " + " | ".join(endpoint_errors))
        if not isinstance(payload, list):
            raise RuntimeError(f"{symbol}: unexpected API response")
        if not payload:
            break

        raw_bytes = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        raw_chunks.append(raw_bytes)

        for r in payload:
            rows.append(
                Bar(
                    ts=int(r[0]),
                    open=float(r[1]),
                    high=float(r[2]),
                    low=float(r[3]),
                    close=float(r[4]),
                    quote_volume=float(r[7]),
                    taker_buy_quote_volume=float(r[10]),
                )
            )

        last_ts = rows[-1].ts
        next_cursor = last_ts + 60 * 60 * 1000
        if next_cursor <= cursor:
            raise RuntimeError(f"{symbol}: pagination failed at {cursor}")
        cursor = next_cursor

        if len(payload) < 1000:
            break
        time.sleep(0.15)

    dedup: dict[int, Bar] = {b.ts: b for b in rows}
    ordered = [dedup[k] for k in sorted(dedup)]
    filtered = [b for b in ordered if start_ms <= b.ts < end_ms]

    if len(filtered) < 200:
        raise RuntimeError(f"{symbol}: only {len(filtered)} hourly bars fetched")
    gaps = [
        filtered[i].ts - filtered[i - 1].ts
        for i in range(1, len(filtered))
        if filtered[i].ts - filtered[i - 1].ts != 60 * 60 * 1000
    ]
    if gaps:
        raise RuntimeError(f"{symbol}: {len(gaps)} hourly cadence gaps detected")
    digest = hashlib.sha256(b"".join(raw_chunks)).hexdigest()
    return filtered, digest


def zscore(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    mean = statistics.fmean(values)
    stdev = statistics.pstdev(values)
    if stdev <= 0:
        return 0.0
    return (values[-1] - mean) / stdev


def metrics(trade_returns: list[float], start_ts: int, end_ts: int) -> dict[str, float | int | None]:
    if not trade_returns:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "sharpe_trade_level": None,
            "sortino_trade_level": None,
            "max_drawdown": 0.0,
            "profit_factor": None,
            "trades": 0,
            "win_rate": None,
            "avg_trade": None,
        }

    equity = 1.0
    curve = [equity]
    wins: list[float] = []
    losses: list[float] = []
    for r in trade_returns:
        equity *= 1.0 + r
        curve.append(equity)
        (wins if r > 0 else losses).append(r)

    total_return = equity - 1.0
    years = max((end_ts - start_ts) / (365.25 * 24 * 3600), 1 / 365.25)
    cagr = equity ** (1.0 / years) - 1.0
    mean = statistics.fmean(trade_returns)
    stdev = statistics.pstdev(trade_returns)
    trades_per_year = len(trade_returns) / years
    sharpe = None if stdev == 0 else mean / stdev * math.sqrt(max(trades_per_year, 1.0))

    downside = [min(r, 0.0) for r in trade_returns]
    down_stdev = statistics.pstdev(downside)
    sortino = None if down_stdev == 0 else mean / down_stdev * math.sqrt(max(trades_per_year, 1.0))

    peak = curve[0]
    max_dd = 0.0
    for value in curve:
        peak = max(peak, value)
        max_dd = min(max_dd, value / peak - 1.0)

    gross_profit = sum(wins)
    gross_loss = -sum(losses)
    pf = None if gross_loss == 0 else gross_profit / gross_loss

    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe_trade_level": sharpe,
        "sortino_trade_level": sortino,
        "max_drawdown": max_dd,
        "profit_factor": pf,
        "trades": len(trade_returns),
        "win_rate": sum(1 for r in trade_returns if r > 0) / len(trade_returns),
        "avg_trade": mean,
    }


def run(
    start: str,
    end: str,
    lookback: int,
    threshold: float,
    hold_bars: int,
    fee_bps: float,
    slippage_bps: float,
) -> dict:
    start_ms, end_ms = ms(start), ms(end)
    all_symbols = ["BTCUSDT"] + ALTCOINS
    data: dict[str, list[Bar]] = {}
    hashes: dict[str, str] = {}
    failures: dict[str, str] = {}

    for symbol in all_symbols:
        try:
            bars, digest = fetch_klines(symbol, start_ms, end_ms)
            data[symbol] = bars
            hashes[symbol] = digest
        except Exception as exc:
            failures[symbol] = str(exc)

    if "BTCUSDT" not in data:
        raise RuntimeError(f"BTCUSDT fetch failed: {failures.get('BTCUSDT', 'unknown error')}")

    usable_alts = [s for s in ALTCOINS if s in data]
    if len(usable_alts) < 4:
        raise RuntimeError(f"Fewer than 4 altcoins were usable: {usable_alts}; failures={failures}")

    btc = {b.ts: b for b in data["BTCUSDT"]}
    alt_by_ts = {s: {b.ts: b for b in data[s]} for s in usable_alts}
    common_ts = sorted(set(btc).intersection(*(set(alt_by_ts[s]) for s in usable_alts)))
    if len(common_ts) < lookback + hold_bars + 20:
        raise RuntimeError("Insufficient common timestamps after alignment")

    cost_one_way = (fee_bps + slippage_bps) / 10000.0
    round_trip = 2.0 * cost_one_way

    trades: list[dict] = []
    trade_returns: list[float] = []
    next_free_index = lookback + 1

    for i in range(lookback, len(common_ts) - hold_bars - 1):
        if i < next_free_index:
            continue
        ts = common_ts[i]
        hist = [btc[common_ts[j]].delta for j in range(i - lookback + 1, i + 1)]
        z = zscore(hist)
        side = 1 if z >= threshold else (-1 if z <= -threshold else 0)
        if side == 0:
            continue

        entry_i = i + 1
        exit_i = entry_i + hold_bars
        entry_ts = common_ts[entry_i]
        exit_ts = common_ts[exit_i]

        per_asset: list[float] = []
        for sym in usable_alts:
            entry = alt_by_ts[sym][entry_ts]
            exit_bar = alt_by_ts[sym][exit_ts]
            gross = exit_bar.close / entry.open - 1.0
            per_asset.append(side * gross - round_trip)

        ret = statistics.fmean(per_asset)
        trade_returns.append(ret)
        trades.append({
            "signal_timestamp": datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
            "entry_timestamp": datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat(),
            "exit_timestamp": datetime.fromtimestamp(exit_ts / 1000, tz=timezone.utc).isoformat(),
            "btc_delta_z": z,
            "side": "LONG" if side == 1 else "SHORT",
            "basket_return_net": ret,
            "assets": ",".join(usable_alts),
        })
        next_free_index = exit_i + 1

    result = {
        "experiment": {
            "name": "BTC volume-delta directional alt-basket baseline",
            "execution_mode": "GitHub Actions public runner",
            "parquet_dependency": False,
            "data_source": "Binance public spot kline REST API with endpoint fallback ladder",
            "interval": "1h",
            "start": start,
            "end": end,
            "btc_signal": "z-score of Binance taker-buy quote volume delta",
            "lookback_bars": lookback,
            "threshold": threshold,
            "hold_bars": hold_bars,
            "fee_bps_one_way": fee_bps,
            "slippage_bps_one_way": slippage_bps,
            "alt_universe": usable_alts,
            "excluded_blue_chip_symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"],
        },
        "data": {
            "bars_common": len(common_ts),
            "hashes": hashes,
            "fetch_failures": failures,
            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        },
        "strategy_metrics": metrics(trade_returns, common_ts[0], common_ts[-1]),
        "trade_count_long": sum(1 for t in trades if t["side"] == "LONG"),
        "trade_count_short": sum(1 for t in trades if t["side"] == "SHORT"),
        "trades": trades,
    }
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2024-06-01T00:00:00Z")
    p.add_argument("--end", default="2026-09-20T00:00:00Z")
    p.add_argument("--lookback", type=int, default=48)
    p.add_argument("--threshold", type=float, default=2.0)
    p.add_argument("--hold-bars", type=int, default=6)
    p.add_argument("--fee-bps", type=float, default=5.0)
    p.add_argument("--slippage-bps", type=float, default=2.0)
    p.add_argument("--output", default="results/btc_volume_delta_result.json")
    args = p.parse_args()

    result = run(
        start=args.start,
        end=args.end,
        lookback=args.lookback,
        threshold=args.threshold,
        hold_bars=args.hold_bars,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
    )
    out = args.output
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    trade_csv = out.replace(".json", "_trades.csv")
    with open(trade_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=result["trades"][0].keys() if result["trades"] else ["signal_timestamp"])
        writer.writeheader()
        writer.writerows(result["trades"])

    print(json.dumps({
        "status": "SUCCESS",
        "result_file": out,
        "trade_file": trade_csv,
        "strategy_metrics": result["strategy_metrics"],
        "trade_count_long": result["trade_count_long"],
        "trade_count_short": result["trade_count_short"],
        "usable_altcoins": result["experiment"]["alt_universe"],
        "fetch_failures": result["data"]["fetch_failures"],
        "parquet_dependency": False,
    }, indent=2))


if __name__ == "__main__":
    main()
