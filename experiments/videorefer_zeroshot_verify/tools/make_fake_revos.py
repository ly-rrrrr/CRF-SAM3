#!/usr/bin/env python
"""Build a tiny fake ReVOS-like dataset + historical files for a CPU smoke test.

python tools/make_fake_revos.py --out /tmp/fake_revos
Then run scripts/smoke_test.sh (uses step1 --mock_recall and step2 --dry_run).
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.io_utils import decode_rle as dec, encode_rle  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    root = a.out
    H, W = 48, 64
    videos, mask_dict, scores, official, results = {}, {}, [], [], {}
    anno = 1000
    for vi in range(3):
        vid = f"FAKE/train/v{vi}"
        T = 10 + vi * 3
        os.makedirs(os.path.join(root, vid), exist_ok=True)
        frames = [f"img_{t:07d}" for t in range(T)]
        for t in range(T):
            Image.fromarray(rng.integers(0, 255, (H, W, 3), dtype=np.uint8)).save(
                os.path.join(root, vid, frames[t] + ".jpg"))
        # two GT objects moving horizontally
        objs = []
        for oi in range(2):
            seq = []
            for t in range(T):
                m = np.zeros((H, W), dtype=np.uint8)
                x0 = 5 + oi * 30 + t % 5
                m[10 + oi * 5: 25 + oi * 5, x0: x0 + 12] = 1
                if vi == 2 and oi == 1 and t < 3:
                    m[:] = 0  # object absent in some frames
                seq.append(encode_rle(m) if m.any() else None)
            mask_dict[str(anno)] = seq
            objs.append(anno)
            anno += 1
        exps = {}
        for ei in range(3):
            target = [objs[ei % 2]] if ei < 2 else objs  # last expression: both objects
            exps[str(ei)] = {"exp": f"fake expression {ei} of video {vi}", "obj_id": [0], "anno_id": target,
                             "type_id": ei % 2}
            # historical: candidates 0,1 (GT objs) and 2 (distractor); select by qwen score
            k = len(target)
            qs = {"0": float(rng.random()), "1": float(rng.random()), "2": float(rng.random())}
            sel = sorted(sorted(qs, key=lambda x: -qs[x])[:k], key=int)
            scores.append({"video_id": vid, "exp_id": str(ei), "expression": exps[str(ei)]["exp"],
                           "category": "thing", "primary_category": "thing", "used_category_query": "thing",
                           "category_override_used": False, "selected_candidates": [int(x) for x in sel],
                           "all_candidates": [0, 1, 2],
                           "scores": {i: {"score": s, "prob_yes": s, "prob_no": 1 - s,
                                          "prediction": "Yes" if s > .5 else "No"} for i, s in qs.items()}})
            pm = []
            for t in range(T):
                u = np.zeros((H, W), dtype=np.uint8)
                for c in sel:
                    c = int(c)
                    if c < 2:
                        r = mask_dict[str(objs[c])][t]
                        if r is not None:
                            u |= dec(r)
                    else:
                        r = mask_dict[str(objs[0])][t]
                        u |= np.roll(dec(r), 7, axis=1)
                pm.append(encode_rle(u))
            results.setdefault(vid, {})[str(ei)] = {"video_id": vid, "exp_id": str(ei), "frames": frames,
                                                    "prediction_masks": pm}
            official.append({"video_name": vid, "exp_id": str(ei), "exp": exps[str(ei)]["exp"],
                             "videxp": f"{vid}_{ei}", "J": 50.0, "F": 50.0, "JF": 50.0, "A": 0, "R": 0,
                             "type_id": ei % 2})
        # one forced-choice expression (1 candidate)
        exps["9"] = {"exp": "forced", "obj_id": [0], "anno_id": [objs[0]], "type_id": 0}
        scores.append({"video_id": vid, "exp_id": "9", "expression": "forced", "category": "thing",
                       "used_category_query": "thing", "selected_candidates": [0], "all_candidates": [0],
                       "scores": {"0": {"score": 0.9}}})
        official.append({"video_name": vid, "exp_id": "9", "exp": "forced", "videxp": f"{vid}_9",
                         "J": 90.0, "F": 90.0, "JF": 90.0, "A": 0, "R": 0, "type_id": 0})
        videos[vid] = {"frames": frames, "height": H, "width": W, "expressions": exps}
    json.dump({"videos": videos}, open(os.path.join(root, "meta_expressions_valid_.json"), "w"))
    json.dump(mask_dict, open(os.path.join(root, "mask_dict.json"), "w"))
    os.makedirs(os.path.join(root, "hist"), exist_ok=True)
    json.dump(scores, open(os.path.join(root, "hist", "scores.json"), "w"))
    json.dump(results, open(os.path.join(root, "hist", "results.json"), "w"))
    with open(os.path.join(root, "hist", "revos_valid_official.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(official[0].keys()))
        wr.writeheader()
        wr.writerows(official)
    print("fake data at", root)


if __name__ == "__main__":
    main()
