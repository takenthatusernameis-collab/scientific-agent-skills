#!/usr/bin/env python3
"""
Controlled V2 optimization harness for the preserved BTC volume-delta baseline.

The original scratch/backtest.py is intentionally imported unchanged.
This file adds only:
- fixed train/OOS split with a flat reset at the boundary
- small pre-registered grid over lookback, threshold, hold, and signal polarity
- deterministic training selection with a minimum trade count
- one-shot OOS evaluation of the selected parameters
- cost-stress checks without re-optimizing
- explicit provenance and protocol metadata

No parameter is selected using OOS results.
No universe expansion is performed.
No Parquet dependency exists.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

from backtest import ALTCOINS, fetch_klines, metrics, ms, zscore


LOOKBACKS = (24, 48, 72)
THRESHOLDS = (1.5, 2.0, 2.5)
HOLD_BARS = (3, 6, 12)
POLARITIES = (1, -1)  # +1 continuation, -1 contrarian
MIN_TRAIN_TRADES = 100

START_DEFAULT = "2024-06-01T00:00:00Z"
TRAIN_END_DEFAULT = "2025-12-01T00:00:00Z"
END_DEFAULT = "2026-09-20T00:00:00Z"


def load_aligned_data(start: str, end: str) -> tuple[dict, dict, list[int], list[str], dict]:
    start_ms, end_ms = ms(start), ms(end)
    symbols = ["BTCUSDT"] + list(ALTCOINS)
    data: dict[str, list] = {}
    hashes: dict[str, str] = {}
    failures: dict[str, str] = {}

    for symbol in symbols:
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
    common_ts = sorted(
        set(btc).intersection(*(set(alt_by_ts[s]) for s in usable_alts))
    )

    if len(common_ts) < 1000:
        raise RuntimeError("Insufficient common timestamps after alignment")

    return data, hashes, common_ts, usable_alts, failures


def select_side(z: float, polarity: int, threshold: float) -> int:
    if z >= threshold:
        return polarity
    if z <= -threshold:
        return -polarity
    return 0


def eval_segment(
    btc: dict,
    alt_by_ts: dict,
    common_ts: list[int],
    usable_alts: list[str],
    *,
    segment_start: str,
    segment_end: str,
    lookback: int,
    threshold: float,
    hold_bars: int,
    polarity: int,
    fee_bps: float,
    slippage_bps: float,
) -> tuple[dict, list[dict]]:
    seg_start = ms(segment_start)
    seg_end = ms(segment_end)
    cost_one_way = (fee_bps + slippage_bps) / 10000.0
    round_trip = 2.0 * cost_one_way

    returns: list[float] = []
    trades: list[dict] = []
    next_free_index = lookback + 1

    for i in range(lookback, len(common_ts) - hold_bars - 1):
        if i < next_free_index:
            continue

        ts = common_ts[i]
        entry_i = i + 1
        exit_i = entry_i + hold_bars
        entry_ts = common_ts[entry_i]
        exit_ts = common_ts[exit_i]

        # The segment is a hard evaluation boundary; positions crossing into
        # the segment are not carried in, so the OOS evaluation starts flat.
        if entry_ts < seg_start:
            continue
        if exit_ts >= seg_end:
            break

        hist = [
            btc[common_ts[j]].delta
            for j in range(i - lookback + 1, i + 1)
        ]
        signal_z = zscore(hist)
        side = select_side(signal_z, polarity, threshold)
        if side == 0:
            continue

        per_asset: list[float] = []
        for sym in usable_alts:
            entry = alt_by_ts[sym][entry_ts]
            exit_bar = alt_by_ts[sym][exit_ts]
            gross = exit_bar.close / entry.open - 1.0
            per_asset.append(side * gross - round_trip)

        ret = statistics.fmean(per_asset)
        returns.append(ret)
        trades.append(
            {
                "signal_timestamp": datetime.fromtimestamp(
                    ts / 1000, tz=timezone.utc
                ).isoformat(),
                "entry_timestamp": datetime.fromtimestamp(
                    entry_ts / 1000, tz=timezone.utc
                ).isoformat(),
                "exit_timestamp": datetime.fromtimestamp(
                    exit_ts / 1000, tz=timezone.utc
                ).isoformat(),
                "btc_delta_z": signal_z,
                "side": "LONG" if side == 1 else "SHORT",
                "basket_return_net": ret,
                "assets": ",".join(usable_alts),
            }
        )
        next_free_index = exit_i + 1

    start_for_metrics = max(seg_start, common_ts[0])
    end_for_metrics = min(seg_end, common_ts[-1])
    return metrics(returns, start_for_metrics, end_for_metrics), trades


def candidate_key(row: dict):
    m = row["metrics"]
    sharpe = m["sharpe_trade_level"]
    pf = m["profit_factor"]
    dd = m["max_drawdown"]
    return (
        -float("-inf" if sharpe is None else sharpe),
        -float("-inf" if pf is None else pf),
        float("inf" if dd is None else abs(dd)),
        row["lookback"],
        row["threshold"],
        row["hold_bars"],
        row["polarity"],
    )


def polarity_name(polarity: int) -> str:
    return "continuation" if polarity == 1 else "contrarian"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default=START_DEFAULT)
    p.add_argument("--train-end", default=TRAIN_END_DEFAULT)
    p.add_argument("--end", default=END_DEFAULT)
    p.add_argument("--fee-bps", type=float, default=5.0)
    p.add_argument("--slippage-bps", type=float, default=2.0)
    p.add_argument("--stress-fee-bps", type=float, default=8.0)
    p.add_argument("--stress-slippage-bps", type=float, default=4.0)
    p.add_argument("--output", default="results/btc_volume_delta_v2_result.json")
    args = p.parse_args()

    data, hashes, common_ts, usable_alts, failures = load_aligned_data(
        args.start, args.end
    )

    btc = {b.ts: b for b in data["BTCUSDT"]}
    alt_by_ts = {s: {b.ts: b for b in data[s]} for s in usable_alts}

    candidates: list[dict] = []
    for lookback in LOOKBACKS:
        for threshold in THRESHOLDS:
            for hold_bars in HOLD_BARS:
                for polarity in POLARITIES:
                    m, _ = eval_segment(
                        btc,
                        alt_by_ts,
                        common_ts,
                        usable_alts,
                        segment_start=args.start,
                        segment_end=args.train_end,
                        lookback=lookback,
                        threshold=threshold,
                        hold_bars=hold_bars,
                        polarity=polarity,
                        fee_bps=args.fee_bps,
                        slippage_bps=args.slippage_bps,
                    )
                    candidates.append(
                        {
                            "lookback": lookback,
                            "threshold": threshold,
                            "hold_bars": hold_bars,
                            "polarity": polarity,
                            "polarity_name": polarity_name(polarity),
                            "metrics": m,
                            "eligible_for_selection": m["trades"] >= MIN_TRAIN_TRADES,
                        }
                    )

    eligible = [c for c in candidates if c["eligible_for_selection"]]
    if not eligible:
        raise RuntimeError(
            f"No training candidate met MIN_TRAIN_TRADES={MIN_TRAIN_TRADES}"
        )

    eligible.sort(key=candidate_key)
    selected = eligible[0]

    selected_oos_metrics, selected_oos_trades = eval_segment(
        btc,
        alt_by_ts,
        common_ts,
        usable_alts,
        segment_start=args.train_end,
        segment_end=args.end,
        lookback=selected["lookback"],
        threshold=selected["threshold"],
        hold_bars=selected["hold_bars"],
        polarity=selected["polarity"],
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
    )

    stress_metrics, _ = eval_segment(
        btc,
        alt_by_ts,
        common_ts,
        usable_alts,
        segment_start=args.train_end,
        segment_end=args.end,
        lookback=selected["lookback"],
        threshold=selected["threshold"],
        hold_bars=selected["hold_bars"],
        polarity=selected["polarity"],
        fee_bps=args.stress_fee_bps,
        slippage_bps=args.stress_slippage_bps,
    )

    selected_params = {
        "lookback": selected["lookback"],
        "threshold": selected["threshold"],
        "hold_bars": selected["hold_bars"],
        "polarity": selected["polarity"],
        "polarity_name": selected["polarity_name"],
    }

    result = {
        "protocol": {
            "name": "BTC volume-delta V2 controlled optimization",
            "baseline_fallback": "scratch/chatgpt_btc_volume_delta/backtest.py",
            "baseline_unchanged": True,
            "parquet_dependency": False,
            "data_source": "Binance public spot klines with official static-archive fallback",
            "interval": "1h",
            "start": args.start,
            "train_end": args.train_end,
            "end": args.end,
            "training_selection": "highest trade-level Sharpe among eligible candidates",
            "minimum_training_trades": MIN_TRAIN_TRADES,
            "oos_selection_used": False,
            "universe_frozen": usable_alts,
            "excluded_blue_chip_symbols": [
                "BTCUSDT",
                "ETHUSDT",
                "BNBUSDT",
                "SOLUSDT",
                "XRPUSDT",
            ],
            "grid": {
                "lookbacks": list(LOOKBACKS),
                "thresholds": list(THRESHOLDS),
                "hold_bars": list(HOLD_BARS),
                "polarities": {
                    "1": "continuation",
                    "-1": "contrarian",
                },
            },
        },
        "data": {
            "common_observations": len(common_ts),
            "hashes": hashes,
            "fetch_failures": failures,
            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        },
        "selection": {
            "selected_parameters": selected_params,
            "training_metrics": selected["metrics"],
            "oos_metrics": selected_oos_metrics,
            "stress_oos_metrics": stress_metrics,
            "stress_costs": {
                "fee_bps_one_way": args.stress_fee_bps,
                "slippage_bps_one_way": args.stress_slippage_bps,
            },
            "oos_trade_gate_100": selected_oos_metrics["trades"] >= 100,
        },
        "training_ranking_top_10": eligible[:10],
        "oos_trades": selected_oos_trades,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    csv_path = out.with_name(out.stem + "_trades.csv")
    import csv
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = selected_oos_trades[0].keys() if selected_oos_trades else [
            "signal_timestamp"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_oos_trades)

    print(
        json.dumps(
            {
                "status": "SUCCESS",
                "selected_parameters": selected_params,
                "training_metrics": selected["metrics"],
                "oos_metrics": selected_oos_metrics,
                "stress_oos_metrics": stress_metrics,
                "oos_trade_gate_100": selected_oos_metrics["trades"] >= 100,
                "common_observations": len(common_ts),
                "artifact": str(out),
                "trade_ledger": str(csv_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
