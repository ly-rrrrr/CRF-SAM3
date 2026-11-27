"""CPU test of tools/audit_trainsets.py on a fake tree.  python tests/test_audit_trainsets.py

Fake Ref-YTVOS: one video with 12 expressions (each a different object), so the Sa2VA order
("0","1","10","11","2",...) differs from json order for exp 2..11. mask_dict is written in Sa2VA
order; the audit must find that from the PNGs, flag the Verify cache / Ground jsonl built with
json order, and see a train-vs-eval duplicate first frame.
"""
import json
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as cm

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "tools"))
import audit_trainsets as au  # noqa: E402

H, W, NOBJ = 16, 48, 12
FRAMES = ["00000", "00005", "00010"]


def rle(m):
    r = cm.encode(np.asfortranarray(m.astype(np.uint8)))
    r["counts"] = r["counts"].decode()
    return r


def obj_mask(k):
    m = np.zeros((H, W), np.uint8)
    m[:, (k - 1) * 4: k * 4] = 1
    return m


def w(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))


def noise(seed):
    return Image.fromarray((np.random.RandomState(seed).rand(H, W, 3) * 255).astype(np.uint8))


def build(d):
    vd, repo = d / "vd", d / "repo"
    yt = vd / "Ref-Youtube-VOS" / "train"
    exps = {str(e): {"exp": f"object {e + 1}", "obj_id": str(e + 1)} for e in range(NOBJ)}
    meta = {"videos": {"v1": {"frames": FRAMES, "expressions": exps},
                       "v2": {"frames": FRAMES, "expressions": {"0": {"exp": "thing", "obj_id": "1"}}}}}
    w(yt / "meta_expressions.json", meta)
    w(yt / "meta.json", {"videos": {"v1": {"objects": {str(k): {"category": "cat"} for k in range(1, NOBJ + 1)}},
                                    "v2": {"objects": {"1": {"category": "dog"}}}}})
    ids = np.zeros((H, W), np.uint8)
    for k in range(1, NOBJ + 1):
        ids[obj_mask(k) > 0] = k
    for v in ("v1", "v2"):
        for i, f in enumerate(FRAMES):
            (yt / "JPEGImages" / v).mkdir(parents=True, exist_ok=True)
            noise(100 + i if v == "v1" else 200 + i).save(yt / "JPEGImages" / v / f"{f}.jpg", quality=95)
            (yt / "Annotations" / v).mkdir(parents=True, exist_ok=True)
            Image.fromarray(ids, mode="L").save(yt / "Annotations" / v / f"{f}.png")
    # mask_dict in Sa2VA order: video insertion order, expression ids sorted as strings
    md, n = {}, 0
    for v, dv in meta["videos"].items():
        for e in sorted(dv["expressions"]):
            k = int(dv["expressions"][e]["obj_id"])
            md[str(n)] = [rle(obj_mask(k)) for _ in FRAMES]
            n += 1
    with open(yt / "mask_dict.pkl", "wb") as f:
        pickle.dump(md, f)

    # DAVIS: anno_id keyed, matches PNG
    dv = vd / "davis17"
    w(dv / "meta_expressions" / "train" / "meta_expressions.json", {"videos": {"bear": {
        "frames": ["00000", "00001"], "expressions": {"0": {"exp": "a bear", "obj_id": 1, "anno_id": [7]}}}}})
    dmd = {"7": [rle(obj_mask(1)), rle(obj_mask(1))]}
    (dv / "train").mkdir(parents=True, exist_ok=True)
    with open(dv / "train" / "mask_dict.pkl", "wb") as f:
        pickle.dump(dmd, f)
    for i, f in enumerate(["00000", "00001"]):
        (dv / "train" / "JPEGImages" / "bear").mkdir(parents=True, exist_ok=True)
        noise(300 + i).save(dv / "train" / "JPEGImages" / "bear" / f"{f}.jpg", quality=95)
        (dv / "train" / "Annotations" / "bear").mkdir(parents=True, exist_ok=True)
        Image.fromarray((obj_mask(1)).astype(np.uint8), mode="L").save(dv / "train" / "Annotations" / "bear" / f"{f}.png")

    # eval set: Ref-YTVOS valid video whose first frame is DAVIS train "bear" (cross-dataset leak)
    yv = vd / "Ref-Youtube-VOS" / "valid"
    w(yv / "meta_expressions.json", {"videos": {"leak": {"frames": ["00000"], "expressions": {}}}})
    (yv / "JPEGImages" / "leak").mkdir(parents=True, exist_ok=True)
    noise(300).save(yv / "JPEGImages" / "leak" / "00000.jpg", quality=95)

    # Verify cache for refytros, iou_map computed the way the current code does (json order)
    cache = repo / au.VERIFY_CACHES["refytros"][0] / "train"
    ins = au.expr_orders(meta)["insertion"]
    inv = {i: k for k, i in au.expr_orders(meta)["sa2va"].items()}
    for e in range(NOBJ):
        own = obj_mask(e + 1)                      # the candidate is the right object
        gt_used = int(inv[ins[("v1", e)]][1]) + 1  # object whose mask the json-order index really holds
        j = 1.0 if gt_used == e + 1 else 0.0
        ed = cache / "v1" / str(e)
        for i in range(3):
            (ed / "masks" / "obj_0").mkdir(parents=True, exist_ok=True)
            Image.fromarray(own * 255).save(ed / "masks" / "obj_0" / f"{i:05d}.png")
        w(ed / "meta.json", {"video_id": "v1", "exp_id": e, "expression": f"object {e + 1}", "category": "cat",
                             "sam3_prompt_source": "category", "frame_pad": 5, "obj_ids": [0],
                             "iou_map": {"0": j}, "frame_indices": {"0": [0, 1, 2]},
                             "video_path": str(yt / "JPEGImages" / "v1")})

    # Ground v1 jsonl, global_index counted in json order
    g = repo / "category_grounding_mvp/data/refytvos_query_grounding"
    g.mkdir(parents=True, exist_ok=True)
    with open(g / "train.jsonl", "w") as f:
        for e in range(NOBJ):
            f.write(json.dumps({"video_id": "v1", "exp_id": e, "global_index": ins[("v1", e)], "category": "cat",
                                "frame_paths": [str(yt / "JPEGImages" / "v1" / f"{x}.jpg") for x in FRAMES],
                                "mask_frame_indices": [0, 1, 2]}) + "\n")
    with open(g / "val.jsonl", "w") as f:
        f.write(json.dumps({"video_id": "v2", "exp_id": 0, "global_index": 12, "category": "dog",
                            "frame_paths": [str(yt / "JPEGImages" / "v2" / f"{x}.jpg") for x in FRAMES],
                            "mask_frame_indices": [0, 1, 2]}) + "\n")
    return vd, repo


def test_audit():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        vd, repo = build(d)
        o = d / "audit"
        au.LINES.clear(), au.GATES.clear(), au.SUMMARY.clear()
        assert au.main(["--repo", str(repo), "--video-datas", str(vd), "--revos-root", str(vd / "revos"),
                        "--gv2", str(d / "nogv2"), "--datasets", "refytvos,davis17", "--leak",
                        "--out", str(o)]) == 0
        t = (o / "trainsets.md").read_text()
        s = json.loads((o / "trainsets.json").read_text())
        assert "这一节出错" not in t, t
        b = s["B_index"]["refytvos"]
        assert b["mode"] == "expression_index" and b["insertion_vs_sa2va_differ"] == 10, b
        assert b["correct_order"] == "sa2va", b
        assert any(g[0] == "FAIL" and "sa2va 顺序" in g[1] for g in s["gates"]), s["gates"]
        assert any(g[0] == "PASS" and "davis17：mask_dict 和 PNG" in g[1] for g in s["gates"]), s["gates"]
        assert any(g[0] == "PASS" and "A refytvos 帧对齐：0 / 2" in g[1] for g in s["gates"]), s["gates"]
        e = s["E_caches"]["refytvos"]
        assert e["meta"] == 12 and e["affected"] == 10, e
        assert e["recompute"]["reproduce_err_max"] == 0.0 and e["recompute"]["label_flip_by_order"] == 10, e
        assert e["force_pos_noise"] == 10 and e["missing_mask_files"] == 0, e
        dg = [v for k, v in s["D_ground"].items() if k.endswith("refytvos_query_grounding/train.jsonl")][0]
        assert dg["index_affected"] == 10 and dg["missing_paths"] == 0 and dg["name_mismatch"] == 0, dg
        assert "train/val 视频重叠 0" in t
        assert s["F_leak"]["pairs"]["refytvos/valid|davis17"] == 1, s["F_leak"]
        aff = json.loads((o / "affected_index_order_refytvos.json").read_text())
        assert len(aff["samples"]) == 10 and aff["correct_order"] == "sa2va"


if __name__ == "__main__":
    test_audit()
    print("audit test passed")
