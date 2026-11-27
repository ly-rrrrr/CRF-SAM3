import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from v2.gt import TrainSet
from v2.validation import validate_cache
p=argparse.ArgumentParser()
for key in ("root","splits","dataset","part","video-datas"): p.add_argument("--"+key,required=True)
a=p.parse_args()
ds=TrainSet(a.dataset,a.video_datas)
counts=validate_cache(a.root,a.splits,a.dataset,a.part,ds.videos)
Path(a.root,f"COMPLETE_{a.part}.json").write_text(json.dumps(counts,indent=2))
print(counts)
