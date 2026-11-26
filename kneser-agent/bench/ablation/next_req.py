import json, sys, time
from pathlib import Path
d = Path(sys.argv[1]); t0 = time.time()
while time.time() - t0 < 540:
    pend = sorted(p for p in d.glob("req_*.json") if not (d / p.name.replace("req_", "rep_")).exists())
    if pend:
        print(pend[0].read_text()); sys.exit(0)
    if (d / "DONE").exists():
        print("DONE"); sys.exit(0)
    time.sleep(1)
print("WAIT (no request yet; run this command again)")
