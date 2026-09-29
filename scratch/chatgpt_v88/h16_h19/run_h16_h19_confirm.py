import json,os,importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location("base","scratch/chatgpt_v88/h16_h19/run_h16_h19.py")
base=importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
h=os.environ["HYPOTHESIS"]
frames,funding,hashes=base.load()
rows_o=base.RUN[h](frames,funding,base.OOS_START,base.OOS_END) if h=="H16" else base.RUN[h](frames,base.OOS_START,base.OOS_END)
rows_h=base.RUN[h](frames,funding,base.HOLD_START,base.HOLD_END) if h=="H16" else base.RUN[h](frames,base.HOLD_START,base.HOLD_END)
out={"experiment_id":"V88-CYCLE3-H16-H19-CONFIRMATORY","prompt_version":base.PROMPT_VERSION,"prompt_sha":base.PROMPT_SHA,"code_sha":os.environ.get("GITHUB_SHA","UNKNOWN"),"workflow_run_id":os.environ.get("GITHUB_RUN_ID","UNKNOWN"),"hypothesis_id":h,"real_data":True,"oos":base.metric(rows_o),"holdout":base.metric(rows_h),"validation_seen_only_for_lock":True,"sibling_oos_seen":False}
out["oos_gate"]=bool(out["oos"]["n"]>=100 and out["oos"]["mean"]>0 and out["oos"]["PF"]>1 and out["oos"]["Sharpe"]>=1)
out["holdout_gate"]=bool(out["holdout"]["n"]>=50 and out["holdout"]["mean"]>0 and out["holdout"]["PF"]>1 and out["holdout"]["Sharpe"]>=1)
out["verified_edge"]=bool(out["oos_gate"] and out["holdout_gate"])
data_root=base.DATA; p=data_root/"results"; p.mkdir(parents=True,exist_ok=True); (p/"H16_H19_CONFIRMATORY.json").write_text(json.dumps(out,indent=2,default=str)+"
"); print(json.dumps(out,indent=2,default=str))
