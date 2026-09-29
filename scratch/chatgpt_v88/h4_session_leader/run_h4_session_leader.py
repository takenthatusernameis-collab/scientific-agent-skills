import csv, io, json, os, hashlib, math, zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from datetime import timedelta

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent
DATA = R / "h4_data"
DATA.mkdir(parents=True, exist_ok=True)

SYMBOLS = ["BTCUSDT","JTOUSDT","PYTHUSDT","WIFUSDT","SEIUSDT","TIAUSDT","ENAUSDT","ORDIUSDT","JUPUSDT"]
ALT_SYMBOLS = SYMBOLS[1:]
START = pd.Timestamp("2024-01-01", tz="UTC")
END = pd.Timestamp("2026-09-25 23:59:59", tz="UTC")
VAL_END = pd.Timestamp("2025-09-30", tz="UTC")
OOS_END = pd.Timestamp("2026-03-31", tz="UTC")
ROUND_TRIP_COST = 0.0021
THRESHOLDS = [0.002, 0.003, 0.004, 0.005]
EXITS = {"15:30": "15:15", "15:45": "15:30", "16:00": "15:45"}
SESSION_LOOKBACK = "09:30_to_10:00"

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def month_range():
    cur = START.to_period("M")
    last = END.to_period("M")
    out = []
    while cur <= last:
        out.append(str(cur))
        cur = cur + 1
    return out

def fetch(url, out):
    if out.exists() and out.stat().st_size > 100:
        return out
    req = Request(url, headers={"User-Agent": "V88-H4/1.0"})
    last = None
    for attempt in range(3):
        try:
            with urlopen(req, timeout=60) as r:
                data = r.read()
            if len(data) < 100:
                raise RuntimeError(f"tiny response {len(data)} bytes")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
            return out
        except HTTPError as e:
            if e.code == 404:
                return None
            last = e
        except Exception as e:
            last = e
    raise RuntimeError(f"download failed: {url}: {last}")

def acquire():
    current_ym = END.strftime("%Y-%m")
    jobs = []
    for symbol in SYMBOLS:
        for ym in month_range():
            p = DATA / "zips" / symbol / f"{symbol}-15m-{ym}.zip"
            p.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://data.binance.vision/data/futures/um/monthly/klines/{symbol}/15m/{symbol}-15m-{ym}.zip"
            jobs.append((symbol, ym, url, p))
    failures = []
    missing_months_404 = []
    current_month_fallback = []

    def do_fetch(item):
        symbol, key, url, p = item
        try:
            return symbol, key, fetch(url, p), None
        except Exception as e:
            return symbol, key, None, str(e)

    with ThreadPoolExecutor(max_workers=8) as ex:
        fs = {ex.submit(do_fetch, item): item for item in jobs}
        for f in as_completed(fs):
            symbol, key, result, err = f.result()
            if err:
                failures.append({"symbol":symbol,"month":key,"error":err})
            elif result is None:
                if key == current_ym:
                    current_month_fallback.append(symbol)
                else:
                    missing_months_404.append({"symbol":symbol,"month":key})

    daily_jobs = []
    for symbol in sorted(set(current_month_fallback)):
        day = pd.Timestamp(current_ym + "-01", tz="UTC")
        while day <= END.normalize():
            ds = day.strftime("%Y-%m-%d")
            p = DATA / "zips" / symbol / f"{symbol}-15m-{ds}.zip"
            url = f"https://data.binance.vision/data/futures/um/daily/klines/{symbol}/15m/{symbol}-15m-{ds}.zip"
            daily_jobs.append((symbol, ds, url, p))
            day += timedelta(days=1)

    daily_404 = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        fs = {ex.submit(do_fetch, item): item for item in daily_jobs}
        for f in as_completed(fs):
            symbol, key, result, err = f.result()
            if err:
                failures.append({"symbol":symbol,"day":key,"error":err})
            elif result is None:
                daily_404.append({"symbol":symbol,"day":key})

    if failures:
        (DATA / "DOWNLOAD_FAILURES.json").write_text(json.dumps(failures, indent=2))
        raise RuntimeError(f"{len(failures)} non-404 archive downloads failed")

    features = {}
    for symbol in SYMBOLS:
        parts = []
        for z in sorted((DATA / "zips" / symbol).glob("*.zip")):
            with zipfile.ZipFile(z) as zz:
                names = zz.namelist()
                if not names:
                    continue
                with zz.open(names[0]) as fh:
                    df = pd.read_csv(fh, header=None, usecols=[0,4], names=["open_time","close"])
            if not np.issubdtype(df["open_time"].dtype, np.number):
                df = df.iloc[1:]
            df["open_time"] = pd.to_datetime(pd.to_numeric(df["open_time"]), unit="ms", utc=True)
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df = df.dropna().set_index("open_time")
            parts.append(df)

        if not parts:
            continue
        x = pd.concat(parts).sort_index()
        x = x.loc[(x.index >= START) & (x.index <= END)]
        local = x.tz_convert("America/New_York")
        local["ny_date"] = local.index.date
        local["hhmm"] = local.index.strftime("%H:%M")
        needed = ["09:15","09:45"] + list(EXITS.values())
        local = local[local["hhmm"].isin(needed)].copy()
        piv = local.pivot_table(index="ny_date", columns="hhmm", values="close", aggfunc="last")
        piv.columns = [str(v).replace(":","") for v in piv.columns]
        piv.index = pd.to_datetime(piv.index)
        piv["symbol"] = symbol
        features[symbol] = piv.reset_index().rename(columns={"ny_date":"date"})

    rows = []
    for symbol, df in features.items():
        keep = ["date","symbol"] + sorted([col for col in df.columns if col not in {"date","symbol"}])
        rows.append(df[keep])
    if not rows:
        raise RuntimeError("NO_US_SESSION_FEATURE_ROWS")
    feature_df = pd.concat(rows, ignore_index=True)
    feature_path = DATA / "H4_FEATURES.csv"
    feature_df.to_csv(feature_path, index=False)

    manifest = {
        "experiment_id": "V88-CYCLE2-H4-US-SESSION-LEADER",
        "real_data": True,
        "source": "Binance USD-M Public Data monthly 15m klines with daily current-month fallback",
        "symbols": SYMBOLS,
        "date_range_utc": [str(START), str(END)],
        "signal": SESSION_LOOKBACK,
        "decision_time_ny": "10:00",
        "entry": "10:00 NY at the close of the 09:45 15m bar",
        "leader": "single highest (alt_return - BTC_return) over 09:30-10:00 NY",
        "btc_gate": THRESHOLDS,
        "exits_ny": list(EXITS.keys()),
        "round_trip_cost": ROUND_TRIP_COST,
        "feature_sha256": sha256_file(feature_path),
        "monthly_download_count": len(jobs),
        "current_month_daily_fallback_count": len(daily_jobs),
        "missing_months_404": missing_months_404,
        "current_month_daily_404": daily_404,
        "route_repair": "MISSING_INCOMPLETE_MONTHS_USE_DAILY; OLDER_404_MONTHS_EXPLICITLY_RECORDED"
    }
    (DATA / "H4_DATA_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def load_features():
    df = pd.read_csv(DATA / "H4_FEATURES.csv", parse_dates=["date"])
    return df

def metrics(returns):
    r = np.asarray([x for x in returns if np.isfinite(x)], dtype=float)
    if len(r) == 0:
        return {"n": 0, "mean": float("nan"), "PF": 0.0, "Sharpe": float("nan"), "MDD": float("nan"), "total_return": float("nan")}
    mean = float(r.mean())
    gross_pos = float(r[r > 0].sum())
    gross_neg = float(-r[r < 0].sum())
    pf = gross_pos / gross_neg if gross_neg > 0 else float("inf")
    sd = float(r.std(ddof=1)) if len(r) > 1 else float("nan")
    sharpe = mean / sd * math.sqrt(252) if sd > 0 else float("nan")
    eq = np.cumprod(1.0 + r)
    dd = eq / np.maximum.accumulate(eq) - 1.0
    return {
        "n": int(len(r)),
        "mean": mean,
        "PF": pf,
        "Sharpe": float(sharpe),
        "MDD": float(dd.min()),
        "total_return": float(eq[-1] - 1.0),
    }

def run_config(df, threshold, exit_label, start, end):
    pivot = df.pivot(index="date", columns="symbol")
    dates = sorted(set(df["date"]))
    out = []
    for d in dates:
        if d < start or d > end:
            continue
        try:
            b = pivot.loc[d]
            btc_0915 = float(b.loc["0915","BTCUSDT"])
            btc_0945 = float(b.loc["0945","BTCUSDT"])
            btc_ret = btc_0945 / btc_0915 - 1.0
        except Exception:
            continue
        if not np.isfinite(btc_ret) or btc_ret < threshold:
            continue
        rel = {}
        for s in ALT_SYMBOLS:
            try:
                r = float(b.loc["0945",s]) / float(b.loc["0915",s]) - 1.0
                rel[s] = r - btc_ret
            except Exception:
                pass
        if not rel:
            continue
        leader = max(rel, key=rel.get)
        try:
            entry = float(b.loc["0945",leader])
            exit_col = EXITS[exit_label].replace(":","")
            exit_px = float(b.loc[exit_col,leader])
        except Exception:
            continue
        gross = exit_px / entry - 1.0
        net = gross - ROUND_TRIP_COST
        out.append({"date":str(d.date()),"leader":leader,"btc_return":btc_ret,"relative_score":rel[leader],"gross_return":gross,"net_return":net})
    return out

def validate():
    df = load_features()
    threshold = float(os.environ["H4_THRESHOLD"])
    rows = []
    for exit_label in EXITS:
        trades = run_config(df, threshold, exit_label, pd.Timestamp("2024-01-01"), pd.Timestamp("2025-09-30"))
        vals = [x["net_return"] for x in trades]
        m = metrics(vals)
        rows.append({
            "config_id": f"BTC{threshold:.4f}_EXIT{exit_label.replace(':','')}",
            "threshold": threshold,
            "exit": exit_label,
            "validation": m,
            "trades": trades,
        })
    out = {
        "experiment_id":"V88-CYCLE2-H4-US-SESSION-LEADER",
        "prompt_version":"1.6.8",
        "real_data":True,
        "validation_window":["2024-01-01","2025-09-30"],
        "candidate_configs":rows,
        "screen_rule":"n>=100, mean>0, PF>1",
    }
    path = DATA / f"H4_VALIDATION_T{threshold:.4f}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    for row in rows:
        print(json.dumps({"config_id":row["config_id"],"validation":row["validation"]}))

def confirm():
    df = load_features()
    threshold = float(os.environ["H4_THRESHOLD"])
    exit_label = os.environ["H4_EXIT"]
    oos_trades = run_config(df, threshold, exit_label, pd.Timestamp("2025-10-01"), pd.Timestamp("2026-03-31"))
    hold_trades = run_config(df, threshold, exit_label, pd.Timestamp("2026-04-01"), pd.Timestamp("2026-09-25"))
    oos = metrics([x["net_return"] for x in oos_trades])
    hold = metrics([x["net_return"] for x in hold_trades])
    out = {
        "experiment_id":"V88-CYCLE2-H4-CONFIRMATORY",
        "prompt_version":"1.6.8",
        "real_data":True,
        "selected_config":{"threshold":threshold,"exit":exit_label},
        "oos":oos,
        "holdout":hold,
        "oos_gate":bool(oos["n"]>=100 and oos["mean"]>0 and oos["PF"]>1 and oos["Sharpe"]>=1.0),
        "holdout_gate":bool(hold["n"]>=50 and hold["mean"]>0 and hold["PF"]>1 and hold["Sharpe"]>=1.0),
    }
    out["verified_edge"] = bool(out["oos_gate"] and out["holdout_gate"])
    (R/"H4_CONFIRMATORY.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))

mode = os.environ.get("H4_MODE","validate")
if mode == "acquire":
    acquire()
elif mode == "confirm":
    confirm()
else:
    validate()
