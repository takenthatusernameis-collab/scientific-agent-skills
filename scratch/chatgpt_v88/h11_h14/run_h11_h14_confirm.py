import gzip,json,math,os,hashlib
from pathlib import Path
import numpy as np,pandas as pd
import importlib.util
spec=importlib.util.spec_from_file_location("base","scratch/chatgpt_v88/h11_h14/run_h11_h14.py")
base=importlib.util.module_from_spec(spec); spec.loader.exec_module(base)

def main():
    h=os.environ["HYPOTHESIS"]
    data_root=Path("scratch/chatgpt_v88/h11_h14_data")
    cands=list(data_root.glob("**/BTCUSDT.csv.gz"))
    if cands: data_root=cands[0].parent
    base.DATA=data_root
    frames,hashes=base.load()
    rows=[]
    for segment,start,end in [
      ("oos",base.OOS_START,base.OOS_END),
      ("holdout",base.HOLD_START,base.HOLD_END)
    ]:
        r=base.RUNNERS[h](frames,start,end)
        m=base.metrics(r)
        rows.append((segment,m))
    out=dict(
      experiment_id="V88-CYCLE2-H11-H14-CONFIRMATORY",
      prompt_version="1.8.1",
      hypothesis_id=h,
      real_data=True,
      selection_seen_from_validation_only=True,
      sibling_oos_seen=False,
      oos=rows[0][1],
      holdout=rows[1][1]
    )
    out["oos_gate"]=bool(out["oos"]["n"]>=100 and out["oos"]["mean"]>0 and out["oos"]["PF"]>1 and out["oos"]["Sharpe"]>=1)
    out["holdout_gate"]=bool(out["holdout"]["n"]>=50 and out["holdout"]["mean"]>0 and out["holdout"]["PF"]>1 and out["holdout"]["Sharpe"]>=1)
    out["verified_edge"]=bool(out["oos_gate"] and out["holdout_gate"])
    p=data_root/"results"; p.mkdir(parents=True,exist_ok=True)
    (p/"H11_H14_CONFIRMATORY.json").write_text(json.dumps(out,indent=2,default=str)+"\n")
    print(json.dumps(out,indent=2,default=str))

if __name__=="__main__":
    main()
