#!/usr/bin/env python3
"""CPU check: are predicted masks written to the right frames?

SAM3 and the visual tubes index frames by POSITION in the sorted JPEG folder (0, 1, 2, ...).
run_crf_sam3_dataset.save_masks (Ref-YouTube-VOS, MeViS, DAVIS runners) and
run_crf_sam3_revos_dataset._build_revos_rles look a frame up with int(frame_name) FIRST and only then
with its position. That is harmless when names are 00000, 00001, 00002, ... (name == position), but
Ref-YouTube-VOS frames are every 5th frame (00000, 00005, 00010, ...): the frame named 00005 is
position 1, yet the lookup hits key 5, i.e. the mask of position 5 (frame 00025). Every frame whose
int(name) < number of frames (except 00000) gets the mask of a later frame.

This script needs no GPU and reads only meta, JPEG folder listings, GT PNGs and an existing prediction:
  1. how many frames / expressions the name-vs-position rule affects (per dataset meta you pass)
  2. proof on an existing prediction: the PNG of an affected frame is identical to the PNG of the
     frame at the position it was looked up from (impossible by chance)
  3. per-frame IoU with GT on affected vs unaffected frames of that prediction (how much it costs)

  python check_frame_index.py --refytvos-root .../Ref-Youtube-VOS --split valid \
      --pred-root $REPO/crf_sam3_refytros_valid_query_grounding_mainline_20260701 \
      --revos-meta .../revos/meta_expressions_valid_.json --out runs/check
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


def affected_positions(frames: List[str]) -> List[Tuple[int, int]]:
    """(position, looked-up position) for frames that the int(name)-first rule maps to another frame.

    Assumes every position has a mask (the common case); a position without one falls back to itself."""
    n = len(frames)
    out = []
    for pos, name in enumerate(frames):
        if name.isdigit():
            k = int(name)
            if k != pos and k < n:
                out.append((pos, k))
    return out


def meta_stats(meta_path: str, sort_frames: bool) -> Dict:
    videos = json.loads(Path(meta_path).read_text(encoding="utf-8"))["videos"]
    nv = nf = af = ne = ae = 0
    for vid, v in videos.items():
        frames = [str(x) for x in v.get("frames", [])]
        if sort_frames:
            frames = sorted(frames)
        a = affected_positions(frames)
        e = len(v.get("expressions", {}))
        nv += 1
        nf += len(frames) * e
        af += len(a) * e
        ne += e
        ae += e if a else 0
    return {"videos": nv, "expressions": ne, "expressions_affected": ae, "frames": nf, "frames_affected": af,
            "frame_share": af / max(nf, 1)}


def jpeg_layout(root: str, split: str, videos: Dict) -> Dict:
    """Does the JPEG folder hold exactly the meta frames (so position = index in the sorted meta list)?"""
    same = differ = 0
    examples = []
    for vid, v in videos.items():
        d = Path(root) / split / "JPEGImages" / vid
        if not d.is_dir():
            continue
        stems = sorted(p.stem for p in d.glob("*.jpg"))
        frames = sorted(str(x) for x in v.get("frames", []))
        if stems == frames:
            same += 1
        else:
            differ += 1
            if len(examples) < 3:
                examples.append({"video": vid, "jpgs": len(stems), "meta_frames": len(frames),
                                 "first_jpgs": stems[:4], "first_meta": frames[:4]})
    return {"videos_jpeg_equals_meta": same, "videos_jpeg_differs": differ, "examples": examples}


def _bin(path: Path, hw=None):
    if not path.is_file():
        return None
    a = np.array(Image.open(path))
    if a.ndim == 3:
        a = a[..., 0]
    a = a > 0
    if hw is not None and a.shape != hw:
        a = np.array(Image.fromarray(a.astype(np.uint8) * 255).resize((hw[1], hw[0]), Image.NEAREST)) > 0
    return a


def _iou(g: np.ndarray, p: np.ndarray) -> float:
    u = np.logical_or(g, p).sum()
    return 1.0 if u == 0 else float(np.logical_and(g, p).sum() / u)  # same convention as db_eval_iou


def _pred_dir(pred_root: Path, vid: str, eid: str) -> Path:
    d = pred_root / "Annotations" / vid / eid
    return d if d.is_dir() else pred_root / vid / eid


def pred_check(root: str, split: str, videos: Dict, pred_root: str) -> Dict:
    pr = Path(pred_root)
    same_pairs = pairs = ctrl_same = ctrl = 0
    iou_aff, iou_un, iou_un_no0 = [], [], []
    exp_j, exp_j_unaff_only = [], []
    for vid, v in videos.items():
        frames = sorted(str(x) for x in v.get("frames", []))
        aff = dict(affected_positions(frames))
        for eid in v.get("expressions", {}):
            pd = _pred_dir(pr, vid, str(eid))
            gd = Path(root) / split / "Annotations" / vid / str(eid)
            if not pd.is_dir():
                continue
            preds, gts = [], []
            for f in frames:
                g = _bin(gd / f"{f}.png")
                preds.append(_bin(pd / f"{f}.png", None if g is None else g.shape))
                gts.append(g)
            for pos, k in aff.items():
                if preds[pos] is not None and preds[k] is not None:
                    pairs += 1
                    same_pairs += bool(np.array_equal(preds[pos], preds[k]))
            # control: neighbouring unaffected frames are almost never pixel-identical
            un = [p for p in range(len(frames)) if p not in aff]
            for a, b in zip(un, un[1:]):
                if preds[a] is not None and preds[b] is not None and (preds[a].any() or preds[b].any()):
                    ctrl += 1
                    ctrl_same += bool(np.array_equal(preds[a], preds[b]))
            js, ju = [], []
            for p, (g, q) in enumerate(zip(gts, preds)):
                if g is None:
                    continue
                q = np.zeros_like(g) if q is None else q
                j = _iou(g, q)
                js.append(j)
                if p in aff:
                    iou_aff.append(j)
                else:
                    iou_un.append(j)
                    ju.append(j)
                    if p != 0:
                        iou_un_no0.append(j)
            if js:
                exp_j.append(np.mean(js))
                exp_j_unaff_only.append(np.mean(ju) if ju else np.mean(js))
    m = lambda x: round(100 * float(np.mean(x)), 2) if x else None
    return {"affected_pairs_identical": f"{same_pairs}/{pairs}",
            "control_neighbour_pairs_identical": f"{ctrl_same}/{ctrl}",
            "frame_J_affected": m(iou_aff), "frame_J_unaffected": m(iou_un), "frame_J_unaffected_excl_first": m(iou_un_no0),
            "n_affected_frames": len(iou_aff), "n_unaffected_frames": len(iou_un),
            "expression_J_as_is": m(exp_j), "expression_J_on_unaffected_frames_only": m(exp_j_unaff_only)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refytvos-root", required=True)
    ap.add_argument("--split", default="valid")
    ap.add_argument("--meta-file", default="meta_expressions_challenge.json")
    ap.add_argument("--pred-root", action="append", default=[], help="existing Ref-YTVOS prediction dir(s)")
    ap.add_argument("--revos-meta", default="", help="also count ReVOS frames hit by the same rule")
    ap.add_argument("--extra-meta", action="append", default=[], help="name=path of other meta_expressions files")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = os.path.join(a.refytvos_root, a.split, a.meta_file)
    videos = json.loads(Path(meta_path).read_text(encoding="utf-8"))["videos"]
    rep = {"refytvos_meta": meta_stats(meta_path, sort_frames=True),
           "refytvos_jpeg_layout": jpeg_layout(a.refytvos_root, a.split, videos), "predictions": {}}
    if a.revos_meta:
        rep["revos_meta"] = meta_stats(a.revos_meta, sort_frames=False)
    for item in a.extra_meta:
        name, path = item.split("=", 1)
        rep[f"{name}_meta"] = meta_stats(path, sort_frames=True)
    for p in a.pred_root:
        print(f"[check] scanning {p} ...", flush=True)
        rep["predictions"][p] = pred_check(a.refytvos_root, a.split, videos, p)
    (out / "frame_index_check.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")

    r = rep["refytvos_meta"]
    L = ["# 帧对齐检查（name vs position）", "",
         f"- Ref-YTVOS {a.split}：{r['expressions_affected']}/{r['expressions']} 条表达式受影响，"
         f"{r['frames_affected']}/{r['frames']} 帧（{100 * r['frame_share']:.1f}%）会被写成别的帧的掩码。",
         f"- JPEG 文件夹与 meta 帧列表一致的视频：{rep['refytvos_jpeg_layout']['videos_jpeg_equals_meta']}，"
         f"不一致：{rep['refytvos_jpeg_layout']['videos_jpeg_differs']}（不一致时先把这一行发给我）。"]
    for k in [x for x in rep if x.endswith("_meta") and x != "refytvos_meta"]:
        s = rep[k]
        L.append(f"- {k[:-5]}：{s['expressions_affected']}/{s['expressions']} 条表达式、{100 * s['frame_share']:.2f}% 的帧受影响。")
    for p, s in rep["predictions"].items():
        L += ["", f"## 已有预测 `{p}`", "",
              f"- 受影响帧与其被查到的那一帧 PNG 完全相同：{s['affected_pairs_identical']}"
              f"（对照：相邻正常帧完全相同 {s['control_neighbour_pairs_identical']}）。前者接近 100% 即证实错位。",
              f"- 逐帧 J：受影响帧 {s['frame_J_affected']}，正常帧 {s['frame_J_unaffected']}"
              f"（不含第 0 帧 {s['frame_J_unaffected_excl_first']}）。",
              f"- 表达式级 J：现状 {s['expression_J_as_is']}；只算正常帧 {s['expression_J_on_unaffected_frames_only']}"
              f"（修复后大致能到的水平，粗估）。"]
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
