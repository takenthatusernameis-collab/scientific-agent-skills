import hashlib, json, math, os
from pathlib import Path
import numpy as np, pandas as pd

R=Path(__file__).resolve().parent
DATA=R/"h20_h23_data"
VAL_START=pd.Timestamp("2023-01-01",tz="UTC"); VAL_END=pd.Timestamp("2023-03-31 23:59:59",tz="UTC")
OOS_START=pd.Timestamp("2023-04-01",tz="UTC"); OOS_END=pd.Timestamp("2023-06-30 23:59:59",tz="UTC")
HOLD_START=pd.Timestamp("2023-07-01",tz="UTC"); HOLD_END=pd.Timestamp("2023-12-31 23:59:59",tz="UTC")
COST=0.0021
PERPS=["SOLUSDT","APTUSDT","OPUSDT","STXUSDT","INJUSDT","DOGEUSDT","LINKUSDT","MATICUSDT"]

META={
"H20":{"class":"CARRY_FUNDING","source":"funding_basis_and_carry","mechanism":"carry","construction":"cross_sectional_portfolio","domain":"derivatives_cross_section","name":"FUNDING_DISPERSION_CARRY"},
"H21":{"class":"RELATIVE_VALUE_MEAN_REVERSION","source":"cross_asset_or_cross_market","mechanism":"mean_reversion","construction":"market_neutral_spread","domain":"spot_plus_perpetuals","name":"SPOT_PERP_BASIS_MEAN_REVERSION"},
"H22":{"class":"MICROSTRUCTURE_ORDER_FLOW","source":"volume_and_trade_flow","mechanism":"lead_lag","construction":"cross_sectional_portfolio","domain":"multi_asset_perpetuals","name":"FLOW_DIVERGENCE_PORTFOLIO"},
"H23":{"class":"CALENDAR_SEASONALITY","source":"calendar_or_event_time","mechanism":"seasonality","construction":"event_driven","domain":"multi_asset_perpetuals","name":"WEEKEND_EVENT_REVERSAL"}}

def load():
    frames={}; funding={}; hashes={}
    for p in sorted(DATA.glob("*.csv.gz")):
        hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
        name=p.name
        if name.startswith("funding_") and name.endswith(".csv.gz"):
            sym=name[len("funding_"):-len(".csv.gz")]
            d=pd.read_csv(p)
            d["fundingTime"]=pd.to_datetime(d["fundingTime"],utc=True,format="mixed")
            d["fundingRate"]=pd.to_numeric(d["fundingRate"],errors="coerce")
            funding[sym]=d.dropna().set_index("fundingTime").sort_index()
            continue
        if name.startswith("perp_") and name.endswith(".csv.gz"):
            sym=name[len("perp_"):-len(".csv.gz")]
            key="perp_"+sym
        elif name.startswith("spot_") and name.endswith(".csv.gz"):
            sym=name[len("spot_"):-len(".csv.gz")]
            key="spot_"+sym
        else:
            continue
        d=pd.read_csv(p)
        d["timestamp"]=pd.to_datetime(d["timestamp"],utc=True,format="mixed")
        for col in ["open","high","low","close","volume","taker_buy_base_volume"]:
            if col in d.columns:
                d[col]=pd.to_numeric(d[col],errors="coerce")
        frames[key]=d.dropna(subset=["timestamp","open","close"]).drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
    return frames,funding,hashes

def metric(rows):
    if not rows:return {"n":0,"mean":float("nan"),"PF":0.0,"Sharpe":float("nan"),"MDD":float("nan"),"total_return":float("nan")}
    d=pd.DataFrame(rows); r=d.ret.to_numpy(float)
    wins=r[r>0].sum(); losses=-r[r<0].sum()
    daily=d.assign(day=pd.to_datetime(d.ts,utc=True).dt.floor("D")).groupby("day").ret.apply(lambda x: float(np.prod(1+x)-1))
    sd=daily.std(ddof=1); eq=(1+daily).cumprod()
    return {"n":int(len(r)),"mean":float(r.mean()),"PF":float(wins/losses) if losses>0 else float("inf"),"Sharpe":float(np.sqrt(252)*daily.mean()/sd) if sd>0 else float("nan"),"MDD":float((eq/eq.cummax()-1).min()),"total_return":float(eq.iloc[-1]-1)}

def hold_return(d,start_i,end_i,direction=1):
    return direction*math.log(float(d.iloc[end_i].open)/float(d.iloc[start_i].open))

def h20(frames,funding,start,end):
    idx=pd.date_range(start,end,freq="8h",tz="UTC")
    rows=[]
    fund_df=pd.DataFrame({s:funding[s].fundingRate for s in PERPS}).sort_index()
    fund_df=fund_df.reindex(fund_df.index.union(idx)).sort_index().ffill().reindex(idx)
    for ts,row in fund_df.iterrows():
        vals=row.dropna()
        if len(vals)<6: continue
        lows=list(vals.nsmallest(2).index); highs=list(vals.nlargest(2).index)
        perps=[frames["perp_"+s] for s in lows+highs]
        anchor=frames["perp_"+lows[0]].index.searchsorted(ts,side="right")
        if anchor+32>=len(frames["perp_"+lows[0]]): continue
        rets=[]
        for s in lows:
            d=frames["perp_"+s]; i=d.index.searchsorted(ts,side="right"); 
            if i+32<len(d): rets.append(hold_return(d,i,i+32,1))
        for s in highs:
            d=frames["perp_"+s]; i=d.index.searchsorted(ts,side="right");
            if i+32<len(d): rets.append(hold_return(d,i,i+32,-1))
        if len(rets)>=3: rows.append({"ts":ts,"ret":float(np.mean(rets)-COST)})
    return rows

def h21(frames,start,end):
    rows=[]
    for sym in ["BTCUSDT","ETHUSDT"]:
        p=frames["perp_"+sym]; s=frames["spot_"+sym]
        df=p[["open","close"]].join(s[["open","close"]],lsuffix="_p",rsuffix="_s",how="inner")
        basis=df.close_p/df.close_s-1
        z=(basis-basis.rolling(96).mean().shift(1))/basis.rolling(96).std(ddof=1).shift(1)
        idx=df.index[(df.index>=start)&(df.index<=end)]; i=96
        while i+16<len(idx):
            ts=idx[i]; zz=z.loc[ts]
            if np.isfinite(zz) and abs(zz)>=1.5:
                a=idx[i+1]; b=idx[i+16]
                ret_sp=math.log(df.loc[b,"open_s"]/df.loc[a,"open_s"]); ret_pp=math.log(df.loc[b,"open_p"]/df.loc[a,"open_p"])
                direction=1 if zz<0 else -1
                rows.append({"ts":ts,"ret":float(direction*(ret_sp-ret_pp)-COST)})
                i+=16
            else: i+=1
    return rows

def h22(frames,start,end):
    idx=frames["perp_"+PERPS[0]].index
    idx=idx[(idx>=start)&(idx<=end)]
    rows=[]; step=4; hold=4
    for i in range(32,len(idx)-hold,step):
        ts=idx[i]; vals={}
        for s in PERPS:
            d=frames["perp_"+s]
            if ts in d.index:
                vol=float(d.loc[ts,"volume"]); tb=float(d.loc[ts,"taker_buy_base_volume"])
                if vol>0: vals[s]=2*tb/vol-1
        if len(vals)<6: continue
        high=list(pd.Series(vals).nlargest(2).index); low=list(pd.Series(vals).nsmallest(2).index)
        rets=[]
        for s in high:
            d=frames["perp_"+s]; a=d.index.searchsorted(ts,side="right"); 
            if a+hold<len(d): rets.append(hold_return(d,a,a+hold,1))
        for s in low:
            d=frames["perp_"+s]; a=d.index.searchsorted(ts,side="right");
            if a+hold<len(d): rets.append(hold_return(d,a,a+hold,-1))
        if len(rets)>=3: rows.append({"ts":ts,"ret":float(np.mean(rets)-COST)})
    return rows

def h23(frames,start,end):
    rows=[]; btc=frames["perp_BTCUSDT"]; alt=[frames["perp_"+s] for s in PERPS]
    idx=btc.index[(btc.index>=start)&(btc.index<=end)]
    for ts in idx[(idx.weekday==4)&(idx.hour==18)&(idx.minute==0)]:
        j=btc.index.searchsorted(ts,side="left")
        if j<96 or j+96>=len(btc.index): continue
        prev=math.log(float(btc.iloc[j].close)/float(btc.iloc[j-96].close))
        direction=-1 if prev>0 else 1
        rets=[]; end_ts=ts+pd.Timedelta(hours=24)
        for d in alt:
            a=d.index.searchsorted(ts,side="right"); b=d.index.searchsorted(end_ts,side="right")-1
            if a<b<len(d): rets.append(hold_return(d,a,b,direction))
        if len(rets)>=6: rows.append({"ts":ts,"ret":float(np.mean(rets)-COST)})
    return rows

RUN={"H20":h20,"H21":h21,"H22":h22,"H23":h23}

def main():
    h=os.environ["HYPOTHESIS"]; frames,funding,hashes=load()
    rows=RUN[h](frames,funding,VAL_START,VAL_END) if h=="H20" else RUN[h](frames,VAL_START,VAL_END)
    meta=META[h]
    out={"experiment_id":"V88-CYCLE4-H20-H23","prompt_version":"1.8.8","hypothesis_id":h,"strategy_class":meta["class"],"information_source_family":meta["source"],"economic_mechanism":meta["mechanism"],"position_construction":meta["construction"],"data_domain":meta["domain"],"real_data":True,"validation_window":["2023-01-01","2023-03-31"],"oos_window":["2023-04-01","2023-06-30"],"holdout_window":["2023-07-01","2023-12-31"],"future_data_hidden":True,"validation":metric(rows),"data_manifest_sha256":hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),"oos_seen":False,"holdout_seen":False,"new_idea":True}
    p=DATA/"results"; p.mkdir(parents=True,exist_ok=True); (p/f"{h}_VALIDATION.json").write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out,indent=2))

if __name__=="__main__": main()
