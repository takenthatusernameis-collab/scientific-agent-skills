import json,math
from pathlib import Path
import numpy as np,pandas as pd

R=Path("scratch/chatgpt_v88/deep_edge_117")
F=pd.read_parquet(R/"features/FEATURES.parquet")
C=json.loads((R/"inputs/candidate_matrix.json").read_text())
CD={x["candidate_id"]:x for x in C}
SYMS=["BTCUSDT","DOGEUSDT","LINKUSDT","AVAXUSDT","NEARUSDT","FILUSDT","ATOMUSDT","UNIUSDT","AAVEUSDT","INJUSDT","OPUSDT","STXUSDT","ARBUSDT","APTUSDT","SUIUSDT"]
OIX={"BTCUSDT","DOGEUSDT","LINKUSDT","AVAXUSDT","NEARUSDT","INJUSDT","OPUSDT","STXUSDT"}
COST_BASE=0.0021
COST_STRESS=0.0030
COST_HARD=0.0042

def mat(prefix,field):
 return pd.DataFrame({s:F[f"{s}|{field}"] for s in SYMS},index=F.index)

CLOSE=mat("","close"); VOL=mat("","vol20"); OI=mat("","oi"); RET1=mat("","ret3") # overwritten per candidate

def signal(spec):
 fam=spec["family"];L=int(spec["lookback"]);T=float(spec["threshold"])
 ret=mat("","ret"+str(L))
 if fam=="TS_MOM": z=ret
 elif fam=="TS_REV": z=-ret
 elif fam=="XS_MOM": z=ret.sub(ret.mean(axis=1),axis=0)
 elif fam=="XS_REV": z=-ret.sub(ret.mean(axis=1),axis=0)
 elif fam=="BREAKOUT":
  hi=CLOSE.shift(1).rolling(L,min_periods=L).max();z=CLOSE/hi-1
 elif fam=="BREAKDOWN":
  lo=CLOSE.shift(1).rolling(L,min_periods=L).min();z=-(CLOSE/lo-1)
 elif fam=="VOL_MOM": z=ret/(mat("","vol"+str(L)).abs()+1e-9)
 elif fam=="VOL_CONTRA": z=-ret/(mat("","vol"+str(L)).abs()+1e-9)
 elif fam=="OI_SHOCK_CONT": z=(ret/(ret.abs().rolling(L,min_periods=max(3,L//2)).std()+1e-9))*mat("","oiz"+str(L))
 elif fam=="OI_SHOCK_REV": z=-(ret/(ret.abs().rolling(L,min_periods=max(3,L//2)).std()+1e-9))*mat("","oiz"+str(L))
 elif fam=="OI_PRICE_DIV": z=-(ret*mat("","oichg"+str(L)))
 elif fam=="OI_XS": z=mat("","oichg"+str(L)).sub(mat("","oichg"+str(L)).mean(axis=1),axis=0)
 elif fam=="BREADTH_MOM":
  breadth=(ret>0).mean(axis=1)-0.5;z=ret.where(breadth.abs()>=max(0.05,T/10))
 elif fam=="BREADTH_REV":
  breadth=(ret>0).mean(axis=1)-0.5;z=-ret.where(breadth.abs()>=max(0.05,T/10))
 elif fam=="BTC_REL_MOM":
  btc=ret["BTCUSDT"];z=ret.sub(btc,axis=0)
 elif fam=="VOL_BREAKOUT":
  v=mat("","vol"+str(L));z=ret/(v.abs()+1e-9)
 else: raise ValueError(fam)
 # threshold gate uses prior-only cross-sectional signal scale; no result-derived tuning.
 rowstd=z.median(axis=1).abs()+z.std(axis=1).abs()+1e-9
 z=z.where(z.abs()>=T*rowstd)
 return z

def evaluate(cid,start,end,cost=COST_BASE,override_L=None,universe_mode=None):
 spec=dict(CD[cid])
 if override_L is not None: spec["lookback"]=int(max(3,override_L))
 z=signal(spec)
 if universe_mode=="alts":z=z.drop(columns=["BTCUSDT"])
 if universe_mode=="liquid":z=z.drop(columns=[c for c in ["APTUSDT","SUIUSDT","ARBUSDT"] if c in z.columns])
 q=float(spec["quantile"]);h=int(spec["hold"])
 sub=z.loc[(z.index>=pd.Timestamp(start,tz="UTC"))&(z.index<=pd.Timestamp(end,tz="UTC"))]
 fwd=CLOSE.shift(-h)/CLOSE-1
 fwd=fwd.reindex(sub.index)
 ranks=sub.rank(axis=1,pct=True,method="average")
 longmask=ranks>=1-q;shortmask=ranks<=q
 longn=longmask.sum(axis=1);shortn=shortmask.sum(axis=1)
 longret=(fwd.where(longmask).sum(axis=1)/(longn.replace(0,np.nan))).fillna(0)
 shortret=(fwd.where(shortmask).sum(axis=1)/(shortn.replace(0,np.nan))).fillna(0)
 # Market-neutral by construction when both legs exist; one-sided fallback is forbidden for this wave.
 valid=(longn>0)&(shortn>0)
 trade=(0.5*longret-0.5*shortret).where(valid).dropna()
 turnover=(valid.astype(float))
 trade=trade-cost*turnover.reindex(trade.index)
 n=int(len(trade))
 if n<2:return {"candidate_id":cid,"n":n,"mean":None,"pf":None,"sharpe":None,"mdd":None,"turnover":None}
 gains=trade[trade>0].sum();loss=-trade[trade<0].sum();pf=float(gains/loss) if loss>0 else float("inf")
 sd=float(trade.std(ddof=1));sh=float(trade.mean()/sd*math.sqrt(n)) if sd>0 else 0.0
 eq=(1+trade).cumprod();dd=eq/eq.cummax()-1
 return {"candidate_id":cid,"n":n,"mean":float(trade.mean()),"pf":pf,"sharpe":sh,"mdd":float(dd.min()),"turnover":float(turnover.reindex(trade.index).mean()),"start":start,"end":end,"cost":cost}


import os,json
from pathlib import Path
summary=json.loads((R/"FINAL_EDGE_WAVE_SUMMARY.json").read_text());ids=[x["candidate_id"] for x in summary["verified_candidates"]];shard=int(os.getenv("SHARD_ID","0"));mine=ids[shard::4]
rows=[]
for cid in mine:
 b=evaluate(cid,"2026-01-01","2026-09-20",COST_BASE);s=evaluate(cid,"2026-01-01","2026-09-20",COST_STRESS)
 rows.append({"candidate_id":cid,"rerun_base":b,"rerun_stress":s,"exact_reconciliation_candidate":True})
out=R/"rerun";out.mkdir(exist_ok=True);(out/f"shard_{shard}.json").write_text(json.dumps({"rows":rows,"stage":"INDEPENDENT_RERUN"},indent=2));print(json.dumps({"shard":shard,"count":len(rows)},indent=2))
