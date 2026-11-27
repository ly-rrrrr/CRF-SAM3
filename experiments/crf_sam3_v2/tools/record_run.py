"""Record commands, package hashes and environment without reading secrets."""
import hashlib,json,os,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
       for sub in ('v2','tools','scripts') for p in (root/sub).rglob('*')
       if p.is_file() and p.suffix in ('.py','.sh')}
for p in root.glob('*.sh'): files[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
rec={'time':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'argv':sys.argv[2:],'source_sha256':files,
     'env':{k:os.environ.get(k) for k in ('GPUS','REPO','MODEL','VD','GROUND_EPOCHS','GROUND_MAX_STEPS','VERIFY_MAX_STEPS','LIMIT','ONLY')}}
with (out/'commands.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
