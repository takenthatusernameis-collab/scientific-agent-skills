from pathlib import Path
p=Path(__file__).with_name("run_h1_real.py")
s=p.read_text()
old="        d[0]=pd.to_datetime(d[0],unit='ms',utc=True)"
new="        if str(d.iloc[0,0]).strip().lower() in {'open time','open_time'}: d=d.iloc[1:].reset_index(drop=True)\n        d[0]=pd.to_datetime(d[0],unit='ms',utc=True)"
if old not in s:
    raise SystemExit("H1_PARSER_PATCH_TARGET_NOT_FOUND")
p.write_text(s.replace(old,new,1))
print("H1_PARSER_PATCH_APPLIED")
