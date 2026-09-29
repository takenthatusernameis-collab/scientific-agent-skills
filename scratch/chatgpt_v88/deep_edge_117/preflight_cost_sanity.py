BASE_COST=0.0021
LONG_WEIGHT=0.5
SHORT_WEIGHT=0.5
GROSS_NOTIONAL=LONG_WEIGHT+SHORT_WEIGHT
charged=BASE_COST*GROSS_NOTIONAL
assert abs(charged-BASE_COST)<1e-12, (charged, BASE_COST)
print({"round_trip_cost":BASE_COST,"gross_notional":GROSS_NOTIONAL,"charged_cost":charged,"status":"PASS"})
