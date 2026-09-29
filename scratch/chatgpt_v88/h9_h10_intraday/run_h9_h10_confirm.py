import gzip,hashlib,json,math,os
from pathlib import Path
import numpy as np,pandas as pd
import importlib.util
spec=importlib.util.spec_from_file_location("base","scratch/chatgpt_v88/h9_h10_intraday/run_h9_h10.py")
base=importlib.util.module_from_spec(spec); spec.loader.exec_module(base)

def run_confirm(h,data_root):
    base.DATA=data_root
    frames,hashes=base.load()
    idx_all=sorted(set().union(*[set(x.index) for x in frames.values()]))
    p=base.panel(frames); r=base.returns(p)
    idx=r.index
    def evaluate(start,end):
        subidx=idx[(idx>=start)&(idx<=end)]
        rows=[]; nxt=0
        for i in range(len(subidx)-base.HOLD_BARS-2):
            if i<nxt: continue
            # map sub-index timestamp back to global positional index for prior-history signals
            ts=subidx[i]
            gi=idx.get_loc(ts)
            w=base.signal(h,gi,idx,r)
            if w is None: continue
            tr=base.trade(p,idx,gi,w)
            if tr is None or not np.isfinite(tr): continue
            nxt=i+base.HOLD_BARS+1
            rows.append({"ts":ts,"return":float(tr)})
        df=pd.DataFrame(rows)
        if df.empty: return base.metrics(df)
        df["ts"]=pd.to_datetime(df["ts"],utc=True)
        return base.metrics(df)
    oos=evaluate(base.OOS_START,base.OOS_END)
    hold=evaluate(base.HOLD_START,base.HOLD_END)
    out={"experiment_id":"V88-CYCLE2-H9-H10-CONFIRMATORY","prompt_version":"1.7.0","hypothesis_id":h,"real_data":True,"oos":oos,"holdout":hold,
         "oos_gate":bool(oos["n"]>=100 and oos["mean"]>0 and oos["PF"]>1 and oos["Sharpe"]>=1),
         "holdout_gate":bool(hold["n"]>=50 and hold["mean"]>0 and hold["PF"]>1 and hold["Sharpe"]>=1)}
    out["verified_edge"]=bool(out["oos_gate"] and out["holdout_gate"])
    out["validation_seen"]=False
    out["sibling_oos_seen"]=False
    p=Path(data_root)
    candidates=list(p.glob("**/BTCUSDT.csv.gz"))
    if candidates:
        p=candidates[0].parent
    p=p/"results"; p.mkdir(parents=True,exist_ok=True)
    (p/"H9_H10_CONFIRMATORY.json").write_text(json.dumps(out,indent=2,default=str)+"
")
    print(json.dumps(out,indent=2))
