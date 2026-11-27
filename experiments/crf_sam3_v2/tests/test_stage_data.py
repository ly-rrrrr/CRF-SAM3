"""CPU test of v2/gt.py, tools/make_splits.py, v2/build_ground.py on a fake tree.
    python tests/test_stage_data.py

Ref-YTVOS: 12 videos; video y0 has 12 expressions (one object each) and mask_dict is written in Sa2VA
order, so json order would give wrong GT for exp 2..11. ReVOS: anno-keyed mask_dict with a 2-target
expression, None frames and an expression without API category. MeViS / DAVIS for splits only.
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tools"))
from v2 import gt  # noqa: E402
from v2 import build_ground  # noqa: E402
from v2.mix import repeats  # noqa: E402
import make_splits  # noqa: E402

H, W = 16, 48


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


def jpg(p, seed):
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.random.RandomState(seed).rand(H, W, 3) * 255).astype(np.uint8)).save(p, quality=90)


def build(vd):
    # ---------------- Ref-YTVOS
    yt = vd / "Ref-Youtube-VOS" / "train"
    frames = ["00000", "00005", "00010", "00015"]
    videos, objects = {}, {}
    for v in range(12):
        vid = f"y{v}"
        n = 12 if v == 0 else 2
        videos[vid] = {"frames": frames,
                       "expressions": {str(e): {"exp": f"{vid} object {e + 1}", "obj_id": str(e + 1)} for e in range(n)}}
        objects[vid] = {"objects": {str(k): {"category": "cat" if k % 2 else "dog"} for k in range(1, n + 1)}}
        ids = np.zeros((H, W), np.uint8)
        for k in range(1, n + 1):
            ids[obj_mask(k) > 0] = k
        for i, f in enumerate(frames):
            jpg(yt / "JPEGImages" / vid / f"{f}.jpg", v * 10 + i)
            p = yt / "Annotations" / vid / f"{f}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(ids, mode="L").save(p)
    meta = {"videos": videos}
    w(yt / "meta_expressions.json", meta)
    w(yt / "meta.json", {"videos": objects})
    md, n = {}, 0
    for vid, d in meta["videos"].items():
        for e in sorted(d["expressions"]):  # Sa2VA order
            k = int(d["expressions"][e]["obj_id"])
            md[str(n)] = [rle(obj_mask(k)) for _ in frames]
            n += 1
    with open(yt / "mask_dict.pkl", "wb") as f:
        pickle.dump(md, f)

    # ---------------- ReVOS (revos_prepared/train)
    rv = vd / "revos_prepared" / "train"
    rframes = [f"{i:05d}" for i in range(6)]
    rvideos, rmd, api, aid = {}, {}, [], 0
    for v in range(10):
        vid = f"r{v}"
        a1, a2, a3 = aid, aid + 1, aid + 2
        aid += 3
        rmd[str(a1)] = [rle(obj_mask(1)) if i < 4 else None for i in range(6)]   # absent in frames 4, 5
        rmd[str(a2)] = [rle(obj_mask(2)) for _ in range(6)]
        rmd[str(a3)] = [rle(obj_mask(3)) for _ in range(6)]
        rvideos[vid] = {"frames": rframes, "expressions": {
            "0": {"exp": "the man", "anno_id": [a1]},
            "1": {"exp": "the horse", "anno_id": [a2]},
            "2": {"exp": "two things", "anno_id": [a2, a3]},
            "3": {"exp": "the other man", "anno_id": [a3]},
            "4": {"exp": "no category here", "anno_id": [a1]}}}
        for i, f in enumerate(rframes):
            jpg(rv / "JPEGImages" / vid / f"{f}.jpg", 500 + v * 10 + i)
        api += [{"video_id": vid, "exp_id": 0, "category": "person"},
                {"video_id": vid, "exp_id": 1, "category": "horse"},
                {"video_id": vid, "exp_id": 2, "category": "horse"},
                {"video_id": vid, "exp_id": 3, "category": "person"}]
    w(rv / "meta_expressions.json", {"videos": rvideos})
    w(rv / "mask_dict.json", rmd)
    cats = vd / "api_cats.json"
    w(cats, {"records": api})

    # ---------------- v1 Ground splits (kept): YTVOS y1 is v1 val, y0/y2..y9 v1 train, y10/y11 unseen;
    # ReVOS r0..r6 had categories (r6 v1 val), r7..r9 had none
    repo = vd.parent / "repo"
    g = repo / "category_grounding_mvp/data"
    for sub, tr, va in (("refytvos_query_grounding", ["y0"] + [f"y{i}" for i in range(2, 10)], ["y1"]),
                        ("revos_query_grounding_api_20260717", [f"r{i}" for i in range(6)], ["r6"])):
        for split, vids in (("train", tr), ("val", va)):
            p = g / sub / f"{split}.jsonl"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("".join(json.dumps({"video_id": v, "exp_id": 0}) + "\n" for v in vids))

    # ---------------- MeViS / DAVIS (splits only)
    for sub, metap in (("mevis", "mevis/train/meta_expressions.json"),
                       ("davis17", "davis17/meta_expressions/train/meta_expressions.json")):
        vids = {}
        for v in range(20):
            vid = f"{sub[0]}{v}"
            vids[vid] = {"frames": ["00000", "00001"], "expressions": {"0": {"exp": "x", "anno_id": [v], "obj_id": 1}}}
            for i, f in enumerate(["00000", "00001"]):
                jpg(vd / sub / "train" / "JPEGImages" / vid / f"{f}.jpg", 900 + v)
        w(vd / metap, {"videos": vids})
    return cats


def test_all():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        vd = d / "vd"
        cats = build(vd)

        # gt: Sa2VA order
        ds = gt.TrainSet("refytvos", str(vd))
        assert ds.mode == "expr"
        for eid in ("2", "10", "11"):
            t = ds.targets("y0", eid)
            assert len(t) == 1 and (t[0][0] == obj_mask(int(eid) + 1)).all(), eid
        # j_scores: eval-style, empty-vs-empty = 1
        rs = gt.TrainSet("revos", str(vd))
        tg = rs.targets("r0", "0")
        cand = [obj_mask(1)] * 4 + [None, None]
        ju, jb = gt.j_scores(cand, tg)
        assert abs(ju - 1.0) < 1e-9 and abs(jb - 1.0) < 1e-9
        ju, jb = gt.j_scores([obj_mask(1)] * 6, tg)          # follows something in the absent frames
        assert abs(jb - 4 / 6) < 1e-9, jb
        two = rs.targets("r0", "2")
        ju, jb = gt.j_scores([obj_mask(2)] * 6, two)
        assert abs(jb - 1.0) < 1e-9 and abs(ju - 0.5) < 1e-9, (ju, jb)
        assert gt.label_of(1.0) == 1 and gt.label_of(0.05) == 0 and gt.label_of(0.3) == -1

        # splits
        sp = d / "splits.json"
        assert make_splits.main(["--video-datas", str(vd), "--repo", str(d / "repo"), "--out", str(sp)]) == 0
        s = json.loads(sp.read_text())
        for n, info in s["datasets"].items():
            assert not set(info["holdout"]) & set(info["train"]), n
        for n in ("mevis", "davis17"):
            assert len(s["datasets"][n]["holdout"]) == 2, n
        assert s["datasets"]["refytvos"]["holdout"] == ["y1"]            # v1 val kept; 2 unseen -> round(0.2)=0 new
        assert set(s["datasets"]["refytvos"]["train"]) >= {"y0", "y2", "y10", "y11"}
        rh = s["datasets"]["revos"]["holdout"]
        assert "r6" in rh and len(rh) == 1 + round(3 * 0.1) and not set(rh) & {f"r{i}" for i in range(6)}, rh
        assert set(s["datasets"]["refytvos"]["fold"].values()) == {0, 1}
        assert s["datasets"]["mevis"]["fold"] == {}

        # ground
        out = d / "ground"
        rc = build_ground.main(["--video-datas", str(vd), "--revos-categories", str(cats),
                                "--gv2", str(HERE.parent / "ground_v2"), "--splits", str(sp), "--out", str(out)])
        assert rc == 0
        summ = json.loads((out / "summary.json").read_text())
        assert summ["png_check_refytvos"]["below_0.9"] == 0 and summ["png_check_refytvos"]["frames_checked"] > 0
        assert summ["stats"]["revos"]["skip_no_category"] == 10
        rep = summ["repeats"]
        assert min(r["r"] for r in rep.values()) == 1
        full = [json.loads(x) for x in (out / "full/train.jsonl").read_text().splitlines()]
        cf0 = [json.loads(x) for x in (out / "cf0/train.jsonl").read_text().splitlines()]
        cf1 = [json.loads(x) for x in (out / "cf1/train.jsonl").read_text().splitlines()]
        val = [json.loads(x) for x in (out / "val.jsonl").read_text().splitlines()]
        assert len(full) == len(cf0) + len(cf1)
        assert all(r["fold"] == 1 for r in cf0) and all(r["fold"] == 0 for r in cf1)
        hold = {(n, v) for n, i in s["datasets"].items() for v in i["holdout"]}
        assert all((r["dataset"], r["video_id"]) in hold for r in val)
        assert not any((r["dataset"], r["video_id"]) in hold for r in full)
        for n in ("revos", "refytvos"):
            got = Counter_(r for r in full if r["dataset"] == n)
            assert set(got.values()) == {rep[n]["r"]}, (n, set(got.values()))
        # y0 exp 10: target is object 11 in every frame, neg = dog objects (even ids), ign = other cats
        y = [r for r in full + val if r["dataset"] == "refytvos" and r["video_id"] == "y0" and r["exp_id"] == 10]
        for r in y:
            for m in r["mask_rles"]:
                assert (gt.decode(m) == obj_mask(11)).all()
            negm = gt.decode(r["neg_rles"][0])
            assert negm[:, 4:8].all() and not negm[:, 40:44].any()      # obj 2 (dog) neg, obj 11 (cat) not
            ignm = gt.decode(r["ign_rles"][0])
            assert ignm[:, 0:4].all() and not ignm[:, 4:8].any()        # obj 1 (cat) ign
        # repeat copies get different frames when there is room
        rv = [r for r in full if r["dataset"] == "revos" and r["video_id"] == next(
            r["video_id"] for r in full if r["dataset"] == "revos") and r["exp_id"] == 1]
        if len(rv) > 1:
            assert len({tuple(r["mask_frame_indices"]) for r in rv}) > 1
        # ReVOS: exp 0 (person, absent in frames 4-5) -> absent frames, neg = horse objects, ign = person obj a3
        r0 = next(r for r in full + val if r["dataset"] == "revos" and r["exp_id"] == 0)
        assert {a["frame_index"] for a in r0["absent_frames"]} == {4, 5}
        n0 = gt.decode(r0["neg_rles"][0])
        assert n0[:, 4:8].all() and not n0[:, 8:12].any()   # a2 horse neg; a3 person -> ign
        assert gt.decode(r0["ign_rles"][0])[:, 8:12].all()
        assert all(set(r["mask_frame_indices"]) <= {0, 1, 2, 3} for r in full + val if r["dataset"] == "revos" and r["exp_id"] == 0)
        # gv2 collator field contract
        for r in full[:5]:
            assert len(r["frame_paths"]) == len(r["mask_rles"]) == len(r["neg_rles"]) == len(r["mask_frame_indices"])
            assert all(Path(p).is_file() for p in r["frame_paths"])


def test_wrong_order_fails():
    """mask_dict in json order (not Sa2VA's): the PNG gate must stop the build."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        vd = d / "vd"
        cats = build(vd)
        yt = vd / "Ref-Youtube-VOS" / "train"
        meta = json.loads((yt / "meta_expressions.json").read_text())
        md, n = {}, 0
        for vid, dv in meta["videos"].items():
            for e in dv["expressions"]:  # json order
                md[str(n)] = [rle(obj_mask(int(dv["expressions"][e]["obj_id"]))) for _ in range(4)]
                n += 1
        with open(yt / "mask_dict.pkl", "wb") as f:
            pickle.dump(md, f)
        sp = d / "splits.json"
        make_splits.main(["--video-datas", str(vd), "--repo", str(d / "repo"), "--out", str(sp)])
        rc = build_ground.main(["--video-datas", str(vd), "--revos-categories", str(cats),
                                "--gv2", str(HERE.parent / "ground_v2"), "--splits", str(sp), "--out", str(d / "g")])
        assert rc == 1


def Counter_(recs):
    from collections import Counter
    return Counter((r["video_id"], r["exp_id"]) for r in recs)


if __name__ == "__main__":
    test_all()
    test_wrong_order_fails()
    # repeats with the inventory numbers (full ReVOS)
    print(" ".join(f"{k}={v['r']}" for k, v in repeats({"mevis": (1662, 23051), "davis17": (60, 572),
                                                         "refytvos": (3471, 12913), "revos": (626, 29252)}).items()))
    print("stage data test passed")
