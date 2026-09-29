import hashlib,json,math,os
from pathlib import Path
import numpy as np,pandas as pd

R=Path(__file__).resolve().parent; DATA=R/"data"
VS=pd.Timestamp("2023-01-01",tz="UTC"); VE=pd.Timestamp("2023-03-31 23:59:59",tz="UTC"); COST=0.0021
SYMS=["BTCUSDT","ETHUSDT","SOLUSDT","DOGEUSDT","INJUSDT","OPUSDT","STXUSDT","LINKUSDT"]
META={
"H24":("LIQUIDITY","open_interest_and_liquidations","liquidity_dislocation","single_asset_directional","perpetuals"),
"H25":("CROSS_ASSET_LEAD_LAG","open_interest_and_liquidations","lead_lag","cross_sectional_portfolio","derivatives_cross_section"),
"H26":("MICROSTRUCTURE_ORDER_FLOW","order_book_and_market_impact","mean_reversion","event_driven","perpetuals"),
"H27":("LIQUIDITY","order_book_and_market_impact","trend_continuation","conditional_portfolio","perpetuals")}

def load(prefix,sym):
    p=next(DATA.glob(f"{sym}_{prefix}*.parquet")); d=pd.read_parquet(p)
    return d

def daily_open(sym):
    d=load("klines_1h",sym); t=pd.to_datetime(d["open_time"],utc=True); d=d.assign(day=t.dt.floor("D"),open=pd.to_numeric(d["open"],errors="coerce"))
    return d.dropna(subset=["open"]).sort_values("open_time").groupby("day",as_index=False).first()[["day","open"]].set_index("day")["open"]

def metrics(sym):
    d=load("metrics",sym); d=d.assign(day=pd.to_datetime(d["create_time"],utc=True).dt.floor("D"))
    for c in ["sum_open_interest","sum_open_interest_value","sum_toptrader_long_short_ratio"]:
        if c in d: d[c]=pd.to_numeric(d[c],errors="coerce")
    return d.sort_values("create_time").groupby("day",as_index=False).last().set_index("day")

def ret_next(o,day):
    i=o.index.searchsorted(day)
    if i<0 or i+2>=len(o): return None
    return float(math.log(o.iloc[i+2]/o.iloc[i+1])-COST)

def metric(rows):
    if not rows:return {"n":0,"mean":None,"PF":None,"Sharpe":None,"MDD":None,"total_return":None}
    r=np.asarray(rows,float); wins=r[r>0].sum(); losses=-r[r<0].sum(); eq=np.cumprod(1+r); sd=r.std(ddof=1)
    return {"n":int(len(r)),"mean":float(r.mean()),"PF":float(wins/losses) if losses>0 else float("inf"),"Sharpe":float(np.sqrt(252)*r.mean()/sd) if sd>0 else None,"MDD":float((eq/np.maximum.accumulate(eq)-1).min()),"total_return":float(eq[-1]-1)}

def h24():
    m=metrics("BTCUSDT"); o=daily_open("BTCUSDT"); oi=pd.to_numeric(m["sum_open_interest_value"],errors="coerce").pct_change(); px=np.log(pd.to_numeric(o).pct_change()); rows=[]
    for day in m.index[(m.index>=VS)&(m.index<=VE)]:
        if np.isfinite(oi.get(day,np.nan)) and np.isfinite(px.get(day,np.nan)) and oi.loc[day]<=-0.10 and px.loc[day]<0:
            rr=ret_next(o,day)
            if rr is not None: rows.append(rr)
    return rows

def h25():
    rows=[]; allm={s:metrics(s) for s in SYMS}; opens={s:daily_open(s) for s in SYMS}; days=sorted(set().union(*[set(x.index) for x in allm.values()]))
    for day in days:
        if day<VS or day>VE: continue
        vals=[]
        for s in SYMS:
            x=allm[s].get("sum_toptrader_long_short_ratio",pd.Series(dtype=float))
            if day in x.index and np.isfinite(x.loc[day]): vals.append((s,float(x.loc[day])))
        if len(vals)<6: continue
        vals.sort(key=lambda z:z[1]); k=max(1,len(vals)//4); longs=[s for s,_ in vals[:k]]; shorts=[s for s,_ in vals[-k:]]
        lr=[ret_next(opens[s],day)+COST for s in longs if ret_next(opens[s],day) is not None]; sr=[ret_next(opens[s],day)+COST for s in shorts if ret_next(opens[s],day) is not None]
        if lr and sr: rows.append(float(np.mean(lr)-np.mean(sr)-COST))
    return rows

def book(sym):
    d=load("bookDepth",sym); d["timestamp"]=pd.to_datetime(d["timestamp"],utc=True)
    for c in ["percentage","depth","notional"]: d[c]=pd.to_numeric(d[c],errors="coerce")
    return d.dropna(subset=["timestamp","percentage","depth","notional"]).assign(day=d["timestamp"].dt.floor("D"))

def h26():
    rows=[]
    for s in ["BTCUSDT","ETHUSDT"]:
        d=book(s)
        if not ((d.percentage<0).any() and (d.percentage>0).any()): raise RuntimeError(f"H26 BLOCKED_DATA_SCHEMA {s}")
        x=d.assign(signed=np.where(d.percentage<0,d.depth,-d.depth)).groupby("day").signed.sum().sort_index(); z=(x-x.shift(1).rolling(30).mean())/x.shift(1).rolling(30).std(ddof=1); o=daily_open(s)
        for day in x.index[(x.index>=VS)&(x.index<=VE)]:
            if np.isfinite(z.get(day,np.nan)) and abs(z.loc[day])>=2:
                rr=ret_next(o,day)
                if rr is not None: rows.append(float(-np.sign(z.loc[day])*(rr+COST)-COST))
    return rows

def h27():
    rows=[]
    for s in ["BTCUSDT","ETHUSDT"]:
        d=book(s); x=d.groupby("day").depth.sum().sort_index(); ch=x.pct_change(); o=daily_open(s); prior=np.log(pd.to_numeric(o).pct_change())
        for day in x.index[(x.index>=VS)&(x.index<=VE)]:
            if np.isfinite(ch.get(day,np.nan)) and ch.loc[day]<=-0.20 and np.isfinite(prior.get(day,np.nan)):
                rr=ret_next(o,day)
                if rr is not None: rows.append(float(np.sign(prior.loc[day])*(rr+COST)-COST))
    return rows

FUN={"H24":h24,"H25":h25,"H26":h26,"H27":h27}
def main():
    h=os.environ["HYPOTHESIS"]; meta=META[h]
    try: rows=FUN[h]()
    except RuntimeError as e: out={"experiment_id":"V88-CYCLE5-H24-H27","prompt_version":"1.9.5","hypothesis_id":h,"strategy_class":meta[0],"information_source_family":meta[1],"economic_mechanism":meta[2],"position_construction":meta[3],"data_domain":meta[4],"real_data":True,"future_data_hidden":True,"status":"BLOCKED_DATA_SCHEMA","error":str(e),"validation":None}
    else: out={"experiment_id":"V88-CYCLE5-H24-H27","prompt_version":"1.9.5","hypothesis_id":h,"strategy_class":meta[0],"information_source_family":meta[1],"economic_mechanism":meta[2],"position_construction":meta[3],"data_domain":meta[4],"real_data":True,"future_data_hidden":True,"status":"VALIDATION_COMPLETE","validation":metric(rows),"cost_model":{"round_trip":COST},"data_manifest_sha256":hashlib.sha256((R/"DATA_MANIFEST.json").read_bytes()).hexdigest()}
    (DATA/"results").mkdir(parents=True,exist_ok=True); (DATA/"results"/f"{h}_VALIDATION.json").write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out,indent=2))
if __name__=="__main__": main()
