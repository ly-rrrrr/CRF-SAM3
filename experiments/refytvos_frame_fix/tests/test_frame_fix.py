"""CPU test: a fake Ref-YTVOS video with frames 00000, 00005, ... reproduces the repo writer's shift,
check_frame_index detects it, and run_fixed.save_masks_by_position writes every frame correctly."""
import json, os, sys, types
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import check_frame_index as cfi
import run_fixed

N, H, W = 12, 20, 30
FR = [f"{5 * i:05d}" for i in range(N)]


def mask_at(p):  # object moves right one column-block per position
    m = np.zeros((H, W), bool); m[5:15, 2 * p:2 * p + 4] = True; return m


def repo_writer(masks_dict, frames):  # the repo's lookup: int(name) first, then position
    out = {}
    for pos, name in enumerate(frames):
        key = next((k for k in ([int(name)] if name.isdigit() else []) + [pos] if k in masks_dict), None)
        out[name] = masks_dict.get(key, {}).get(0, np.zeros((H, W), bool))
    return out


def test_all(tmp_path):
    assert cfi.affected_positions(FR) == [(1, 5), (2, 10)]
    assert cfi.affected_positions([f"{i:05d}" for i in range(N)]) == []
    root = tmp_path / "ytvos"
    (root / "valid" / "JPEGImages" / "v").mkdir(parents=True)
    gtd = root / "valid" / "Annotations" / "v" / "0"; gtd.mkdir(parents=True)
    for p, f in enumerate(FR):
        Image.fromarray(np.zeros((H, W, 3), np.uint8)).save(root / "valid" / "JPEGImages" / "v" / f"{f}.jpg")
        Image.fromarray(mask_at(p).astype(np.uint8) * 255).save(gtd / f"{f}.png")
    (root / "valid" / "meta_expressions_challenge.json").write_text(json.dumps(
        {"videos": {"v": {"frames": FR, "expressions": {"0": {"exp": "x"}}}}}))
    masks = {p: {0: mask_at(p)} for p in range(N)}
    bad = tmp_path / "bad" / "Annotations" / "v" / "0"; bad.mkdir(parents=True)
    for f, m in repo_writer(masks, FR).items():
        Image.fromarray(m.astype(np.uint8) * 50).save(bad / f"{f}.png")
    # fixed writer through run_fixed with a stand-in for run_crf_sam3_dataset._make_instance_mask
    fake = types.ModuleType("run_crf_sam3_dataset")
    fake._make_instance_mask = lambda fm, h, w: (sum((np.asarray(m) > 0).astype(np.uint8) for m in fm.values()) * 50).astype(np.uint8) if fm else np.zeros((h, w), np.uint8)
    sys.modules["run_crf_sam3_dataset"] = fake
    run_fixed.save_masks_by_position(masks, str(tmp_path / "good"), "v", "0", FR, (H, W))
    rep = cfi.main(["--refytvos-root", str(root), "--pred-root", str(tmp_path / "bad"),
                    "--pred-root", str(tmp_path / "good"), "--out", str(tmp_path / "out")])
    r = json.loads((tmp_path / "out" / "frame_index_check.json").read_text())
    assert r["refytvos_meta"]["frames_affected"] == 2
    assert r["refytvos_jpeg_layout"]["videos_jpeg_equals_meta"] == 1
    b, g = r["predictions"][str(tmp_path / "bad")], r["predictions"][str(tmp_path / "good")]
    assert b["affected_pairs_identical"] == "2/2" and b["frame_J_affected"] == 0.0 and b["frame_J_unaffected"] == 100.0
    assert g["affected_pairs_identical"] == "0/2" and g["frame_J_affected"] == 100.0
    assert g["expression_J_as_is"] == 100.0 and b["expression_J_as_is"] < 100.0
