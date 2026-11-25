"""CPU test of gv2.build_refytvos (Ref-YouTube-VOS -> Ground v2 training records).  python tests/test_build_refytvos.py

Video v, frames 00000..00015 (step 5): object 1 person (target, gone in the last two frames), object 2 person,
object 3 ape. Expect: ape -> negative, the other person -> ignored, absent frames = the last two,
v1 target masks agree with the PNGs (IoU 1)."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gv2 import build_refytvos  # noqa: E402
from gv2.common import decode_rle, normalize_rle  # noqa: E402

H = W = 32
FR = [f"{5 * i:05d}" for i in range(4)]


def ids_frame(t):
    a = np.zeros((H, W), np.uint8)
    if t < 2:
        a[0:8, 0:8] = 1          # target person
    a[8:16, 8:16] = 2            # other person
    a[16:32, 16:32] = 3          # ape
    return a


def test_build_refytvos():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "train"
        for t, f in enumerate(FR):
            (root / "JPEGImages" / "v").mkdir(parents=True, exist_ok=True)
            (root / "Annotations" / "v").mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (W, H)).save(root / "JPEGImages" / "v" / f"{f}.jpg")
            im = Image.fromarray(ids_frame(t), mode="P")
            im.putpalette([0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255] + [0] * (256 * 3 - 12))
            im.save(root / "Annotations" / "v" / f"{f}.png")
        (root / "meta.json").write_text(json.dumps({"videos": {"v": {"objects": {
            "1": {"category": "person"}, "2": {"category": "person"}, "3": {"category": "ape"}}}}}))
        (root / "meta_expressions.json").write_text(json.dumps({"videos": {"v": {"frames": FR, "expressions": {
            "0": {"exp": "a man on the left", "obj_id": "1"}}}}}))
        tgt = [normalize_rle(mu.encode(np.asfortranarray((ids_frame(t) == 1).astype(np.uint8)))) for t in range(2)]
        rec = {"video_id": "v", "exp_id": 0, "obj_id": "1", "category": "person", "expression": "a man on the left",
               "frames": FR[:2], "frame_paths": [str(root / "JPEGImages" / "v" / f"{f}.jpg") for f in FR[:2]],
               "mask_rles": tgt, "mask_frame_indices": [0, 1]}
        (Path(d) / "v1.jsonl").write_text(json.dumps(rec) + "\n")
        out = Path(d) / "out" / "train.jsonl"
        build_refytvos.main(["--v1-jsonl", str(Path(d) / "v1.jsonl"), "--ytvos-train", str(root), "--out", str(out)])
        r = json.loads(out.read_text().splitlines()[0])
        assert r["target_group"] == "person"
        assert decode_rle(r["neg_rles"][0]).sum() == 256 and decode_rle(r["ign_rles"][0]).sum() == 64
        assert sorted(a["frame_index"] for a in r["absent_frames"]) == [2, 3]
        assert decode_rle(r["absent_frames"][0]["neg_rle"]).sum() == 256
        s = json.loads((out.parent / "summary.json").read_text())
        assert s["records_with_neg"] == 1 and s["records_with_ign"] == 1 and s["target_png_iou_mean"] == 1.0, s


if __name__ == "__main__":
    test_build_refytvos()
    print("build_refytvos test passed")
