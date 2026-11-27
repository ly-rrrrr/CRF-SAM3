"""CPU test of v2.verify_cache with the fake repo of test_mainline.  python tests/test_verify_cache.py

Ref-YTVOS-like train video v (frames 00000, 00005, ...), mask_dict in Sa2VA order.
exp 0: GT = the RIGHT object (moving). Text recall finds LEFT (negative), the heatmap seed adds RIGHT
       (positive, J = 1) -> written, labels from J_best, masks named by frame position.
exp 1: GT = a box nobody recalls -> no positive -> left out of the training cache, recorded.
The Ground word comes from the cross-fitted model (fold 0 -> cf0). Verify scoring must not run.
"""
import json
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "tests"))
import test_mainline as tm  # noqa: E402  (fake repo, fake SAM3 session; puts p1a on sys.path)
from p1a import sam3_ops  # noqa: E402
from v2 import verify_cache  # noqa: E402
from pycocotools import mask as mu  # noqa: E402

EXPRS = {"0": "the right cup", "1": "the cup in the corner"}


def rle(m):
    r = mu.encode(np.asfortranarray(m.astype(np.uint8)))
    r["counts"] = r["counts"].decode()
    return r


def corner():
    m = np.zeros((tm.H, tm.W), bool)
    m[0:10, 0:10] = True
    return m


def words(d: Path, model: str, vid: str):
    import cv2
    ht, wt = 12, 20
    tok = cv2.resize(tm.RIGHT.astype(np.float32), (wt, ht), interpolation=cv2.INTER_AREA)
    base = np.clip(tok * 0.95 + 0.02, 0, 1).reshape(-1)
    recs = []
    for eid, e in EXPRS.items():
        recs.append({"video_id": vid, "exp_id": eid, "expression": e, "pred_category": "cup"})
        hd = d / model / "npz" / vid
        hd.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(hd / f"{eid}.npz", responses=np.concatenate([base] * 3)[None].astype(np.float16),
                            image_grid_thw=np.array([[1, ht * 2, wt * 2]] * 3), merge=np.array(2),
                            frame_indices=np.array([0, 3, 7]), frame_stems=np.array([tm.FR[0], tm.FR[3], tm.FR[7]]),
                            orig_hw=np.array([tm.H, tm.W]), expression=np.array(e))
    (d / model / "categories.json").write_text(json.dumps({"records": recs}))


def test_cache():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        repo = tm.fake_repo(d)
        vd = d / "vd"
        yt = vd / "Ref-Youtube-VOS" / "train"
        tm.write_video(yt / "JPEGImages" / "v")
        (yt / "meta_expressions.json").write_text(json.dumps({"videos": {"v": {
            "frames": tm.FR, "expressions": {k: {"exp": v, "obj_id": str(int(k) + 1)} for k, v in EXPRS.items()}}}}))
        with open(yt / "mask_dict.pkl", "wb") as f:
            pickle.dump({"0": [rle(tm.right_at(t)) for t in range(tm.T)], "1": [rle(corner()) for _ in range(tm.T)]}, f)
        sp = d / "splits.json"
        sp.write_text(json.dumps({"datasets": {"refytvos": {"train": ["v"], "holdout": [], "fold": {"v": 0}}}}))
        mem = d / "members.json"
        mem.write_text(json.dumps({"refytvos": {"v": 0}}))
        wd = d / "words"
        words(wd, "cf0", "v")
        sam3_ops.SAM3Session = tm.FakeSession
        out = d / "cache"
        argv = ["--dataset", "refytvos", "--video-datas", str(vd), "--splits", str(sp), "--members", str(mem),
                "--words", str(wd), "--out", str(out), "--repo", str(repo), "--config", str(repo / "cfg.yaml")]
        assert tm.call_main(verify_cache.main, argv) == 0
        done = [json.loads(x) for x in (out / "done_train_0of1.jsonl").read_text().splitlines()]
        st = {r["exp_id"]: r for r in done}
        assert st[0]["status"] == "ok" and st[0]["ground_model"] == "cf0" and st[0]["n_seed"] == 1, st[0]
        assert st[1]["status"] == "no_positive", st[1]
        root = out / "train_root" / "train" / "v"
        assert not (root / "1" / "meta.json").exists()
        meta = json.loads((root / "0" / "meta.json").read_text())
        assert meta["source"] == {"0": "word", "1000": "seed"}
        assert meta["iou_map"]["1000"] == 1.0 and meta["iou_map"]["0"] == 0.0, meta["iou_map"]
        assert meta["frame_indices"]["1000"] == list(range(tm.T)) and meta["frame_pad"] == 5
        for t in range(tm.T):  # position-keyed names: 00001.png is frame 00005
            m = np.array(Image.open(root / "0" / "masks" / "obj_1000" / f"{t:05d}.png")) > 0
            assert (m == tm.right_at(t)).all(), t
        # re-run: nothing left to do
        assert tm.call_main(verify_cache.main, argv) == 0
        assert len((out / "done_train_0of1.jsonl").read_text().splitlines()) == 2


if __name__ == "__main__":
    test_cache()
    print("verify cache test passed")
