import json, os
from pathlib import Path
import numpy as np
import pandas as pd
from run_intraday_hypothesis import (
    HOLD_END, HOLD_START, OOS_END, OOS_START,
    VAL_START, VAL_END, run, load_data, HYPOTHESES
)

R = Path(__file__).resolve().parent
DATA = R / "h5_data"
out_dir = DATA / "results"
out_dir.mkdir(parents=True, exist_ok=True)

def metrics(rows):
    if not rows:
        return {"n":0,"mean":float("nan"),"PF":0.0,"Sharpe":float("nan"),"MDD":float("nan"),"total_return":float("nan")}
    df = pd.DataFrame(rows).copy()
    df["signal_timestamp"] = pd.to_datetime(df["signal_timestamp"], utc=True)
    r = df["return"].to_numpy(float)
    wins = r[r > 0].sum()
    losses = -r[r < 0].sum()
    daily = df.assign(date=df["signal_timestamp"].dt.floor("D")).groupby("date")["return"].apply(
        lambda x: float(np.prod(1.0 + x) - 1.0)
    )
    sd = daily.std(ddof=1) if len(daily) > 1 else float("nan")
    return {
        "n": int(len(r)),
        "mean": float(r.mean()),
        "PF": float(wins / losses) if losses > 0 else float("inf"),
        "Sharpe": float(np.sqrt(252.0) * daily.mean() / sd) if sd > 0 else float("nan"),
        "MDD": float((daily.add(1).cumprod() / daily.add(1).cumprod().cummax() - 1.0).min()),
        "total_return": float(daily.add(1).prod() - 1.0),
    }

def main():
    hypothesis = os.environ["HYPOTHESIS"]
    if hypothesis not in HYPOTHESES:
        raise SystemExit(f"unknown hypothesis {hypothesis}")
    # run() computes signals causally for the full fixed data range, but no validation
    # information is used in the calculation of OOS/holdout returns.
    frames, hashes = load_data()
    # Rebuild through the public run helper by temporarily collecting its implementation.
    # The helper itself only depends on prior bars for every signal.
    import run_intraday_hypothesis as mod
    panel = mod.build_bar_panel(frames)
    ret = mod.bar_returns(panel)
    idx = mod.eligible_day_index(ret)
    trades=[]
    next_free=0
    for i in range(len(idx)-mod.HOLD_BARS-2):
        if i < next_free:
            continue
        t=idx[i]
        if not (VAL_START <= t <= HOLD_END):
            continue
        sig=mod.signal_for(hypothesis,i,idx,ret)
        if sig is None:
            continue
        tr=mod.trade_return(panel,idx,i,sig)
        if tr is None or not np.isfinite(tr):
            continue
        next_free=i+1+mod.HOLD_BARS+1
        trades.append({"signal_timestamp":t.isoformat(),"return":float(tr),"reason":sig["reason"]})
    df=pd.DataFrame(trades)
    if not df.empty:
        df["signal_timestamp"]=pd.to_datetime(df["signal_timestamp"],utc=True)
    oos_rows=[] if df.empty else df[(df.signal_timestamp>=OOS_START)&(df.signal_timestamp<=OOS_END)].to_dict("records")
    hold_rows=[] if df.empty else df[(df.signal_timestamp>=HOLD_START)&(df.signal_timestamp<=HOLD_END)].to_dict("records")
    oos=metrics(oos_rows)
    hold=metrics(hold_rows)
    oos_gate=bool(oos["n"]>=100 and oos["mean"]>0 and oos["PF"]>1 and oos["Sharpe"]>=1.0)
    hold_gate=bool(hold["n"]>=50 and hold["mean"]>0 and hold["PF"]>1 and hold["Sharpe"]>=1.0)
    out={
        "experiment_id":"V88-CYCLE2-H5-H8-CONFIRMATORY",
        "prompt_version":"1.6.9",
        "real_data":True,
        "selected_hypothesis":hypothesis,
        "selected_hypothesis_label":HYPOTHESES[hypothesis],
        "selection_artifact":"H5_H8_CONFIG_LOCK.json",
        "oos_window":["2026-04-01","2026-07-31"],
        "holdout_window":["2026-08-01","2026-09-25"],
        "data_manifest_sha256":__import__("hashlib").sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
        "oos":oos,
        "holdout":hold,
        "oos_gate":oos_gate,
        "holdout_gate":hold_gate,
        "verified_edge":bool(oos_gate and hold_gate),
        "validation_seen":False,
        "sibling_oos_seen":False
    }
    (out_dir/"H5_H8_CONFIRMATORY.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out,indent=2))

if __name__=="__main__":
    main()
