import json,os
from pathlib import Path
from run_h3_afterhours import load,daily_offsession,day_trade,metrics,END,VAL_END,OOS_END,SYMBOLS

R=Path(__file__).resolve().parent
bars=load()
daily=__import__("pandas").concat([daily_offsession(bars[s]).rename(s) for s in SYMBOLS],axis=1).sort_index()
beta=int(os.environ["H3_BETA"])
threshold=float(os.environ["H3_THRESHOLD"])
nq=int(os.environ["H3_NQ"])
exit_label=os.environ["H3_EXIT"]
oos_days=[d for d in daily.index if VAL_END<d<=OOS_END]
hold_days=[d for d in daily.index if OOS_END<d<=END]
oos=[day_trade(bars,daily,d,beta,threshold,nq,exit_label)[1] for d in oos_days]
hold=[day_trade(bars,daily,d,beta,threshold,nq,exit_label)[1] for d in hold_days]
out={
  "experiment_id":"V88-CYCLE2-H3-CONFIRMATORY",
  "real_data":True,
  "source":"Binance USD-M Public Data 15m archive",
  "selected_config":{"beta_window":beta,"shock_threshold":threshold,"n_quantiles":nq,"exit":exit_label},
  "oos":metrics([x for x in oos if __import__("numpy").isfinite(x)]),
  "holdout":metrics([x for x in hold if __import__("numpy").isfinite(x)]),
  "oos_gate":False,
  "holdout_gate":False,
  "scientific_claim_allowed":False
}
out["oos_gate"]=bool(out["oos"]["n"]>=100 and out["oos"]["mean"]>0 and out["oos"]["PF"]>1 and out["oos"]["Sharpe"]>=1.0)
out["holdout_gate"]=bool(out["holdout"]["n"]>=50 and out["holdout"]["mean"]>0 and out["holdout"]["PF"]>1 and out["holdout"]["Sharpe"]>=1.0)
out["verified_edge"]=bool(out["oos_gate"] and out["holdout_gate"])
(R/"H3_CONFIRMATORY.json").write_text(json.dumps(out,indent=2,default=float)+"\n")
print(json.dumps(out,indent=2,default=float))
