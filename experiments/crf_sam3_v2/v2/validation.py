"""Fail-closed validation for reusable stage outputs."""
import json
from pathlib import Path
import numpy as np

def validate_heat(path):
    with np.load(path, allow_pickle=False) as z:
        required = {"responses", "image_grid_thw", "frame_stems", "merge"}
        if not required <= set(z.files) or not np.isfinite(z["responses"]).all():
            raise ValueError(f"Invalid heatmap: {path}")

def validate_cache(root, split_manifest, dataset, part, meta_videos):
    root=Path(root)
    sp=json.loads(Path(split_manifest).read_text())["datasets"][dataset]
    expected={f"{v}/{e}" for v in sp[part] for e in meta_videos[v]["expressions"]}
    records={}
    for p in root.glob(f"done_{part}_*.jsonl"):
        for l in p.read_text().splitlines():
            row=json.loads(l)
            if row["key"] in records:
                raise ValueError(f"Duplicate cache completion record: {row['key']}")
            records[row["key"]]=row
    if set(records)!=expected:
        raise ValueError(f"Incomplete cache {dataset}/{part}: {len(records)}/{len(expected)}")
    for key,row in records.items():
        status=row["status"]
        if status not in {"ok","no_gt","no_positive","no_candidate","oom"}:
            raise ValueError(f"Unknown completion status {status}")
        if status!="ok":
            continue
        p=root/f"{part}_root"/"train"/key/"meta.json"
        m=json.loads(p.read_text())
        ids=m["obj_ids"]
        if part=="train" and not any(m["iou_map"][str(i)]>=0.5 for i in ids):
            raise ValueError(f"No usable positive: {key}")
        for i in ids:
            frames=m["frame_indices"][str(i)]
            if not frames or not all((p.parent/"masks"/f"obj_{i}"/f"{t:05d}.png").is_file() for t in frames):
                raise ValueError(f"Incomplete candidate masks: {key}/{i}")
    return {s:sum(r["status"]==s for r in records.values()) for s in sorted({r["status"] for r in records.values()})}
