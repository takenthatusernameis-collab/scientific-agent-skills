import json,os,importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location("base","scratch/chatgpt_v88/h20_h23/run_h20_h23.py")
base=importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
h=os.environ["HYPOTHESIS"]; frames,funding,hashes=base.load()
rows_o=base.RUN[h](frames,funding,base.OOS_START,base.OOS_END) if h=="H20" else base.RUN[h](frames,base.OOS_START,base.OOS_END)
rows_h=base.RUN[h](frames,funding,base.HOLD_START,base.HOLD_END) if h=="H20" else base.RUN[h](frames,base.HOLD_START,base.HOLD_END)
out={"experiment_id":"V88-CYCLE4-H20-H23-CONFIRMATORY","prompt_version":"1.8.8","hypothesis_id":h,"real_data":True,"oos":base.metric(rows_o),"holdout":base.metric(rows_h),"validation_seen_only_for_lock":True,"sibling_oos_seen":False}
out["oos_gate"]=bool(out["oos"]["n"]>=100 and out["oos"]["mean"]>0 and out["oos"]["PF"]>1 and out["oos"]["Sharpe"]>=1)
out["holdout_gate"]=bool(out["holdout"]["n"]>=50 and out["holdout"]["mean"]>0 and out["holdout"]["PF"]>1 and out["holdout"]["Sharpe"]>=1)
out["verified_edge"]=bool(out["oos_gate"] and out["holdout_gate"])
p=base.DATA/"results"; p.mkdir(parents=True,exist_ok=True); (p/"H20_H23_CONFIRMATORY.json").write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out,indent=2))
