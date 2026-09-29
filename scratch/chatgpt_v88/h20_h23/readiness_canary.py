import importlib.util, json
spec=importlib.util.spec_from_file_location("runner","scratch/chatgpt_v88/h20_h23/run_h20_h23.py")
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
frames,funding,hashes=mod.load()
expected_perp=["perp_"+s for s in ["SOLUSDT","APTUSDT","OPUSDT","STXUSDT","INJUSDT","DOGEUSDT","LINKUSDT","MATICUSDT","BTCUSDT","ETHUSDT"]]
expected_spot=["spot_BTCUSDT","spot_ETHUSDT"]
expected_funding=["SOLUSDT","APTUSDT","OPUSDT","STXUSDT","INJUSDT","DOGEUSDT","LINKUSDT","MATICUSDT"]
missing=[k for k in expected_perp+expected_spot if k not in frames] + [k for k in expected_funding if k not in funding]
if missing:
    raise SystemExit("READINESS_MISSING_KEYS:"+",".join(missing))
checks={
    "perp_BTCUSDT_rows":len(frames["perp_BTCUSDT"]),
    "perp_SOLUSDT_rows":len(frames["perp_SOLUSDT"]),
    "spot_BTCUSDT_rows":len(frames["spot_BTCUSDT"]),
    "funding_SOLUSDT_rows":len(funding["SOLUSDT"]),
}
if any(v<=0 for v in checks.values()):
    raise SystemExit("READINESS_EMPTY_REQUIRED_ARTIFACT")
print(json.dumps({"readiness_status":"PASS","checks":checks,"artifact_file_count":len(hashes)},indent=2))
