import csv, gzip, hashlib, json, math, os, zipfile
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent
DATA = R / "h5_data"
if not any(DATA.glob("*.csv.gz")) and (DATA / "h3_data").exists():
    DATA = DATA / "h3_data"

SYMBOLS = [
    "BTCUSDT","AAVEUSDT","ADAUSDT","ALGOUSDT","APTUSDT","ARBUSDT","ATOMUSDT",
    "AVAXUSDT","BCHUSDT","BNBUSDT","CRVUSDT","DOGEUSDT","DOTUSDT","ETCUSDT",
    "ETHUSDT","FILUSDT","HBARUSDT","ICPUSDT","INJUSDT","LINKUSDT","LTCUSDT",
    "MKRUSDT","NEARUSDT","OPUSDT","RUNEUSDT","SEIUSDT","SOLUSDT","STXUSDT",
    "SUIUSDT","TRXUSDT","UNIUSDT","XLMUSDT","XRPUSDT"
]
ALT_SYMBOLS = SYMBOLS[1:]

VAL_START = pd.Timestamp("2025-10-01", tz="UTC")
VAL_END = pd.Timestamp("2026-03-31 23:59:59", tz="UTC")
OOS_START = pd.Timestamp("2026-04-01", tz="UTC")
OOS_END = pd.Timestamp("2026-07-31 23:59:59", tz="UTC")
HOLD_START = pd.Timestamp("2026-08-01", tz="UTC")
HOLD_END = pd.Timestamp("2026-09-25 23:59:59", tz="UTC")

COST_RT = 0.0021
HOLD_BARS = 2
BARS_PER_DAY = 96

HYPOTHESES = {
    "H5": "INTRADAY_SHOCK_RELATIVE_REVERSAL",
    "H6": "OPENING_IMPULSE_CONTINUATION",
    "H7": "CROSS_SECTIONAL_BREADTH_EXTREMES",
    "H8": "SESSION_SHOCK_STATE_SWITCH",
}

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def load_data():
    paths = sorted(DATA.glob("*.csv.gz"))
    if not paths:
        raise RuntimeError("NO_H5_DATA_FILES")
    frames = {}
    manifest_hashes = {}
    for p in paths:
        symbol = p.name.removesuffix(".csv.gz")
        if symbol not in SYMBOLS:
            continue
        with gzip.open(p, "rt", newline="") as fh:
            df = pd.read_csv(fh, parse_dates=["timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        for col in ["open","high","low","close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["timestamp","open","close"]).sort_values("timestamp")
        df = df.drop_duplicates("timestamp", keep="last").set_index("timestamp")
        frames[symbol] = df
        manifest_hashes[symbol] = sha256_file(p)
    if "BTCUSDT" not in frames:
        raise RuntimeError("BTCUSDT_MISSING")
    return frames, manifest_hashes

def build_bar_panel(frames):
    panel = pd.DataFrame(index=sorted(set().union(*[set(x.index) for x in frames.values()])))
    for s, df in frames.items():
        panel[(s, "open")] = df["open"].reindex(panel.index)
        panel[(s, "close")] = df["close"].reindex(panel.index)
    panel.columns = pd.MultiIndex.from_tuples(panel.columns)
    return panel.sort_index()

def bar_returns(panel):
    out = {}
    for s in SYMBOLS:
        if (s,"open") not in panel.columns:
            continue
        out[s] = panel[(s,"close")] / panel[(s,"open")] - 1.0
    return pd.DataFrame(out)

def eligible_day_index(ret):
    return ret.index[(ret.index >= VAL_START) & (ret.index <= HOLD_END)]

def signal_for(hypothesis, i, idx, ret):
    t = idx[i]
    if t - pd.Timedelta(days=5) < idx[0]:
        return None
    if "BTCUSDT" not in ret.columns:
        return None
    btc = ret["BTCUSDT"].iloc[i]
    alts = ret[ALT_SYMBOLS].iloc[i].dropna()
    if len(alts) < 8 or not np.isfinite(btc):
        return None
    rel = (alts - btc).dropna()
    if hypothesis == "H5":
        # Reversion after a large BTC shock: bet against the largest residual mover.
        if abs(btc) < 0.004 or rel.empty:
            return None
        leader = rel.abs().idxmax()
        direction = -np.sign(rel[leader])
        if direction == 0:
            return None
        return {"weights": {leader: float(direction)}, "reason": "btc_shock_vs_relative_extreme"}

    if hypothesis == "H6":
        # Continuation: BTC intraday impulse plus broad confirmation, then strongest residual leader.
        breadth = float((alts > 0).mean())
        if btc >= 0.0015 and breadth >= 0.60:
            leader = rel.idxmax()
            return {"weights": {leader: 1.0}, "reason": "positive_btc_and_breadth"}
        if btc <= -0.0015 and breadth <= 0.40:
            leader = rel.idxmin()
            return {"weights": {leader: -1.0}, "reason": "negative_btc_and_breadth"}
        return None

    if hypothesis == "H7":
        # Extreme breadth reversal: oppose the whole cross-section.
        breadth = float((alts > 0).mean())
        if breadth >= 0.80:
            return {"weights": {s: -1.0 / len(alts) for s in alts.index}, "reason": "positive_breadth_extreme"}
        if breadth <= 0.20:
            return {"weights": {s: 1.0 / len(alts) for s in alts.index}, "reason": "negative_breadth_extreme"}
        return None

    if hypothesis == "H8":
        # Prior-only high-shock regime: continuation of strongest residual leader.
        past = ret["BTCUSDT"].iloc[max(0, i-96):i]
        if len(past) < 48:
            return None
        sigma = float(past.std(ddof=1))
        if not np.isfinite(sigma) or sigma <= 0 or abs(btc) <= 2.0 * sigma:
            return None
        leader = rel.idxmax() if btc > 0 else rel.idxmin()
        return {"weights": {leader: 1.0 if btc > 0 else -1.0}, "reason": "prior_only_high_shock_regime"}
    return None

def trade_return(panel, idx, i, signal):
    entry_i = i + 1
    exit_i = entry_i + HOLD_BARS
    if exit_i >= len(idx):
        return None
    entry_ts = idx[entry_i]
    exit_ts = idx[exit_i]
    gross = 0.0
    cost_base = 0.0
    for s, w in signal["weights"].items():
        try:
            e = float(panel.loc[entry_ts, (s,"open")])
            x = float(panel.loc[exit_ts, (s,"open")])
        except KeyError:
            return None
        if not (np.isfinite(e) and np.isfinite(x) and e > 0):
            return None
        gross += w * math.log(x / e)
        cost_base += abs(w)
    return math.exp(gross) - 1.0 - COST_RT * cost_base

def run(hypothesis):
    frames, hashes = load_data()
    # Validation workers are physically cut off at the fresh validation endpoint.
    frames = {s: df.loc[df.index <= VAL_END].copy() for s, df in frames.items()}
    panel = build_bar_panel(frames)
    ret = bar_returns(panel)
    idx = ret.index[(ret.index >= VAL_START - pd.Timedelta(days=2)) & (ret.index <= VAL_END)]
    trades = []
    next_free = 0
    for i in range(len(idx) - HOLD_BARS - 2):
        if i < next_free:
            continue
        t = idx[i]
        if not (VAL_START <= t <= HOLD_END):
            continue
        sig = signal_for(hypothesis, i, idx, ret)
        if sig is None:
            continue
        tr = trade_return(panel, idx, i, sig)
        if tr is None or not np.isfinite(tr):
            continue
        exit_i = i + 1 + HOLD_BARS
        next_free = exit_i + 1
        trades.append({"signal_timestamp":t.isoformat(),"return":float(tr),"reason":sig["reason"]})

    df = pd.DataFrame(trades)
    if df.empty:
        return {
            "hypothesis": HYPOTHESES[hypothesis], "hypothesis_id": hypothesis,
            "real_data": True, "trade_count": 0, "mean": float("nan"),
            "PF": 0.0, "Sharpe": float("nan"), "MDD": float("nan"),
            "total_return": float("nan"), "data_manifest_sha256": hashlib.sha256(
                json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
            "trades":[]
        }
    df["signal_timestamp"] = pd.to_datetime(df["signal_timestamp"], utc=True)
    df["date"] = df["signal_timestamp"].dt.floor("D")
    daily = df.groupby("date")["return"].apply(lambda x: float(np.prod(1.0 + x) - 1.0))
    r = df["return"].to_numpy(dtype=float)
    mean = float(r.mean())
    wins = r[r > 0].sum()
    losses = -r[r < 0].sum()
    pf = float(wins / losses) if losses > 0 else float("inf")
    sd = float(daily.std(ddof=1)) if len(daily) > 1 else float("nan")
    sharpe = float(math.sqrt(252.0) * daily.mean() / sd) if sd > 0 else float("nan")
    eq = (1.0 + daily).cumprod()
    dd = eq / eq.cummax() - 1.0
    return {
        "hypothesis": HYPOTHESES[hypothesis],
        "hypothesis_id": hypothesis,
        "real_data": True,
        "trade_count": int(len(r)),
        "mean": mean,
        "PF": pf,
        "Sharpe": sharpe,
        "MDD": float(dd.min()) if len(dd) else float("nan"),
        "total_return": float(eq.iloc[-1] - 1.0) if len(eq) else float("nan"),
        "data_manifest_sha256": hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
        "trades": df.to_dict("records"),
    }

def main():
    hypothesis = os.environ["HYPOTHESIS"]
    if hypothesis not in HYPOTHESES:
        raise SystemExit(f"unknown hypothesis {hypothesis}")
    frames, hashes = load_data()
    # Re-run with the shared in-memory data path through a temporary compact copy.
    out_dir = DATA / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run(hypothesis)
    # Keep only validation-period evidence in the validation worker.
    all_trades = pd.DataFrame(result.pop("trades", []))
    if not all_trades.empty:
        all_trades["signal_timestamp"] = pd.to_datetime(all_trades["signal_timestamp"], utc=True)
    val = all_trades[(all_trades["signal_timestamp"] >= VAL_START) & (all_trades["signal_timestamp"] <= VAL_END)] if not all_trades.empty else all_trades
    if val.empty:
        metrics = {"n":0,"mean":float("nan"),"PF":0.0,"Sharpe":float("nan"),"MDD":float("nan"),"total_return":float("nan")}
    else:
        daily = val.assign(date=val["signal_timestamp"].dt.floor("D")).groupby("date")["return"].apply(lambda x: float(np.prod(1+x)-1))
        r=val["return"].to_numpy(float)
        wins=r[r>0].sum(); losses=-r[r<0].sum()
        metrics={
            "n":int(len(r)),
            "mean":float(r.mean()),
            "PF":float(wins/losses) if losses>0 else float("inf"),
            "Sharpe":float(math.sqrt(252)*daily.mean()/daily.std(ddof=1)) if len(daily)>1 and daily.std(ddof=1)>0 else float("nan"),
            "MDD":float((daily.add(1).cumprod()/daily.add(1).cumprod().cummax()-1).min()) if len(daily) else float("nan"),
            "total_return":float(daily.add(1).prod()-1) if len(daily) else float("nan")
        }
    output={
        "experiment_id":"V88-CYCLE2-H5-H8-INTRADAY",
        "prompt_version":"1.6.9",
        "hypothesis_id":hypothesis,
        "hypothesis":HYPOTHESES[hypothesis],
        "real_data":True,
        "validation_window":["2025-10-01","2026-03-31"],
        "future_data_hidden":True,
        "validation":metrics,
        "data_manifest_sha256":hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
        "data_file_hashes":hashes,
        "oos_seen":False,
        "holdout_seen":False
    }
    path=out_dir/f"{hypothesis}_VALIDATION.json"
    path.write_text(json.dumps(output,indent=2,default=str)+"\n")
    print(json.dumps(output,indent=2,default=str))

if __name__=="__main__":
    main()
