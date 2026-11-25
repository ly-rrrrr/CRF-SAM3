import json,sys,math
from pathlib import Path
rows=[json.loads(l) for l in Path("runs/noisy_channel/scores.jsonl").read_text().splitlines() if l.strip()]
assert len({r["key"] for r in rows})>=int(sys.argv[1]), f"Incomplete: {len(rows)} / {sys.argv[1]}"
for r in rows:
 assert set(r["s"])=={"gv2","base"}
 for mode in r["s"].values():
  assert len(mode)==2
  assert all(math.isfinite(v) for scores in mode.values() for v in scores.values())
print(f"[validated] {len(rows)} rows, both modes, finite scores")
