"""The one place that reads training-set GT. Every builder in this package goes through it.

Conventions (training_set_spec.md, sections 2 and 5):
  * frame position = index in the SORTED frame list of the video (meta frames sorted, which audit A
    checked equals the sorted JPEG list). Never int(frame_name): Ref-YTVOS names go 00000, 00005, ...
  * mask_dict keys, mirroring Sa2VA's loader projects/sa2va/datasets/sa2va_data_03_refvos.py
    (json_file_preprocess):
      - "expr"  (Ref-YTVOS, Ref-DAVIS17): key = global expression counter, counted over videos in json
        order and, inside a video, expression ids sorted AS STRINGS ("10" before "2"). This is Sa2VA's
        dataset_type="refytvos" path; audit B confirmed it against the palette PNGs.
      - "anno"  (MeViS, ReVOS): key = each anno_id of the expression (Sa2VA's dataset_type="default").
  * a missing / None per-frame entry means "target not in this frame" (empty mask).
  * J is the official eval J over ALL frames, IoU(empty, empty) = 1. For multi-target expressions
    J_union (candidate vs union of targets, for reporting) and J_best (max over single targets, for labels).
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")
POS_J, NEG_J = 0.5, 0.1          # same pair for all four datasets

# name -> (dir under video_datas, mask_dict file, key mode, has palette PNG)
DATASETS = {
    "mevis": ("mevis", "mask_dict.json", "anno", False),
    "davis17": ("davis17", "mask_dict.pkl", "expr", True),
    "refytvos": ("Ref-Youtube-VOS", "mask_dict.pkl", "expr", True),
    "revos": ("revos_prepared", "mask_dict.json", "anno", False),
}
GROUND_SETS = ("revos", "refytvos")     # datasets with object categories
VERIFY_SETS = ("mevis", "davis17", "refytvos", "revos")


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def find_meta(split_root: Path) -> Optional[Path]:
    """Same lookup order as crf_sam3_attn/dataset.py::_load_meta_expressions."""
    for p in (split_root / "meta_expressions.json",
              split_root.parent / "meta_expressions" / split_root.name / "meta_expressions.json",
              split_root.parent / "meta_expressions.json"):
        if p.is_file():
            return p
    return None


def ids_of(x) -> List[int]:
    if x is None or x == "":
        return []
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    return [int(v) for v in str(x).split(",") if str(v).strip() != ""]


def normalize_rle(rle) -> Optional[dict]:
    if not isinstance(rle, dict) or not rle.get("counts"):
        return None
    c = rle["counts"]
    return {"size": [int(v) for v in rle["size"]], "counts": c.decode("ascii") if isinstance(c, bytes) else c}


def decode(rle) -> Optional[np.ndarray]:
    """RLE (or list of RLEs, merged) -> uint8 HxW, or None for an empty / missing entry."""
    if rle is None or rle == "" or rle == []:
        return None
    if isinstance(rle, (list, tuple)):
        ms = [m for m in (decode(r) for r in rle) if m is not None]
        if not ms:
            return None
        o = ms[0].astype(bool)
        for m in ms[1:]:
            o |= m.astype(bool)
        return o.astype(np.uint8)
    from pycocotools import mask as cm
    r = normalize_rle(rle)
    if r is None:
        return None
    m = cm.decode(dict(r, counts=r["counts"].encode("ascii")))
    if m.ndim == 3:
        m = m.any(axis=2)
    return m.astype(np.uint8)


def encode(mask: Optional[np.ndarray]) -> Optional[dict]:
    if mask is None or not mask.any():
        return None
    from pycocotools import mask as cm
    return normalize_rle(cm.encode(np.asfortranarray(mask.astype(np.uint8))))


def fit(m: Optional[np.ndarray], shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """Nearest-neighbour resize of a GT mask to the prediction size (as the evals do)."""
    if m is None or m.shape == tuple(shape):
        return m
    from PIL import Image
    return np.array(Image.fromarray(m.astype(np.uint8) * 255).resize((shape[1], shape[0]), Image.NEAREST)) > 127


def iou(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    a_any = a is not None and bool(a.any())
    b_any = b is not None and bool(b.any())
    if not a_any and not b_any:
        return 1.0
    if not a_any or not b_any:
        return 0.0
    a, b = a.astype(bool), b.astype(bool)
    return float((a & b).sum()) / float((a | b).sum())


def j_scores(cand: Sequence[Optional[np.ndarray]], targets: List[List[Optional[np.ndarray]]],
             shape: Optional[Tuple[int, int]] = None) -> Tuple[float, float]:
    """cand: per-frame masks over ALL frames of the video (None = no mask in that frame).
    targets: per target, per-frame decoded GT (None = absent).  Returns (J_union, J_best)."""
    T = len(cand)
    if T == 0 or not targets:
        return 0.0, 0.0
    if shape is None:
        shape = next((m.shape for m in cand if m is not None), None) or \
            next((m.shape for t in targets for m in t if m is not None), (1, 1))
    tg = [[fit(t[f] if f < len(t) else None, shape) for f in range(T)] for t in targets]
    per_t = [float(np.mean([iou(cand[f], t[f]) for f in range(T)])) for t in tg]
    if len(tg) == 1:
        return per_t[0], per_t[0]
    uni = []
    for f in range(T):
        ms = [t[f] for t in tg if t[f] is not None]
        uni.append(np.logical_or.reduce(ms) if ms else None)
    j_union = float(np.mean([iou(cand[f], uni[f]) for f in range(T)]))
    return j_union, max(per_t)


def label_of(j_best: float) -> int:
    """1 positive, 0 negative, -1 ignored (between the thresholds)."""
    return 1 if j_best >= POS_J else (0 if j_best <= NEG_J else -1)


def list_stems(d) -> Optional[List[str]]:
    if not os.path.isdir(d):
        return None
    return sorted(os.path.splitext(n)[0] for n in os.listdir(d) if n.lower().endswith(IMG_EXT))


class TrainSet:
    """Official train split of one dataset: frames, expressions in Sa2VA order, per-target GT."""

    def __init__(self, name: str, video_datas: str, split: str = "train") -> None:
        if name not in DATASETS:
            raise KeyError(f"unknown dataset {name}; known: {sorted(DATASETS)}")
        self.name = name
        sub, md, self.mode, self.has_png = DATASETS[name]
        self.root = Path(video_datas) / sub / split
        self.meta_path = find_meta(self.root)
        if self.meta_path is None:
            raise FileNotFoundError(f"{name}: no meta_expressions.json under {self.root}")
        self.videos: Dict[str, dict] = load_json(self.meta_path)["videos"]
        self.md_path = self.root / md
        self._md = None
        self._key: Optional[Dict[Tuple[str, str], int]] = None
        self._objects = None

    # ---------------------------------------------------------------- frames
    def frames(self, vid: str) -> List[str]:
        return sorted(str(f) for f in self.videos[vid].get("frames", []))

    def frame_paths(self, vid: str) -> List[str]:
        d = self.root / "JPEGImages" / vid
        out = []
        for f in self.frames(vid):
            p = next((d / f"{f}{e}" for e in IMG_EXT if (d / f"{f}{e}").exists()), d / f"{f}.jpg")
            out.append(str(p))
        return out

    def check_frames(self, vid: str) -> Optional[str]:
        """None if meta frames == sorted JPEG stems, else the reason (audit A, per video)."""
        st = list_stems(self.root / "JPEGImages" / vid)
        if st is None:
            return "missing_jpeg_dir"
        return None if st == self.frames(vid) else f"jpeg {len(st)} vs meta {len(self.frames(vid))}"

    # ---------------------------------------------------------------- expressions
    def expressions(self) -> Iterator[Tuple[str, str, dict]]:
        """(video_id, exp_id, exp_dict) in Sa2VA order: videos in json order, exp ids string-sorted."""
        for vid, d in self.videos.items():
            exps = d.get("expressions", {})
            for eid in sorted(exps):
                yield vid, str(eid), exps[eid]

    def expr_key(self, vid: str, eid: str) -> int:
        if self._key is None:
            self._key = {(v, e): i for i, (v, e, _) in enumerate(self.expressions())}
        return self._key[(vid, str(eid))]

    # ---------------------------------------------------------------- GT
    @property
    def mask_dict(self) -> dict:
        if self._md is None:
            if str(self.md_path).endswith(".pkl"):
                with open(self.md_path, "rb") as f:
                    self._md = pickle.load(f)
            else:
                self._md = load_json(self.md_path)
            if self.mode == "expr":
                n = sum(len(d.get("expressions", {})) for d in self.videos.values())
                keys = sorted(int(k) for k in self._md)
                if keys != list(range(n)):
                    raise ValueError(f"{self.name}: mask_dict keys are not 0..{n - 1}; expected one entry per expression")
        return self._md

    def free(self) -> None:
        self._md = None

    def target_ids(self, vid: str, eid: str) -> List[int]:
        e = self.videos[vid]["expressions"][str(eid)]
        if self.mode == "expr":
            return [self.expr_key(vid, eid)]
        return ids_of(e.get("anno_id"))

    def target_rles(self, vid: str, eid: str) -> List[List[Optional[dict]]]:
        """Per target, per frame (all frames, sorted order) RLE or None. [] if the expression has no mask."""
        md, T = self.mask_dict, len(self.frames(vid))
        out = []
        for k in self.target_ids(vid, eid):
            seq = md.get(str(k), md.get(k))
            if seq is None:
                continue
            if len(seq) != T:
                raise ValueError(f"{self.name}/{vid}/{eid}: GT length {len(seq)} != frames {T}")
            seq = [normalize_rle(r) for r in seq]
            out.append(seq + [None] * (T - len(seq)))
        return out

    def targets(self, vid: str, eid: str) -> List[List[Optional[np.ndarray]]]:
        return [[decode(r) for r in seq] for seq in self.target_rles(vid, eid)]

    @staticmethod
    def union_rles(target_rles: List[List[Optional[dict]]]) -> List[Optional[dict]]:
        T = max((len(s) for s in target_rles), default=0)
        out = []
        for f in range(T):
            rs = [s[f] for s in target_rles if f < len(s) and s[f] is not None]
            if not rs:
                out.append(None)
            elif len(rs) == 1:
                out.append(rs[0])
            else:
                from pycocotools import mask as cm
                m = cm.merge([dict(r, counts=r["counts"].encode("ascii")) for r in rs])
                out.append(normalize_rle(m) if int(cm.area(m)) > 0 else None)
        return out

    # ---------------------------------------------------------------- objects (Ground)
    def png(self, vid: str, frame: str) -> Optional[np.ndarray]:
        p = self.root / "Annotations" / vid / f"{frame}.png"
        if not p.is_file():
            return None
        from PIL import Image
        with Image.open(p) as im:
            return np.array(im)

    def object_categories(self, vid: str) -> Dict[int, str]:
        """Ref-YTVOS: official category of every annotated object (train/meta.json)."""
        if self._objects is None:
            p = self.root / "meta.json"
            self._objects = load_json(p)["videos"] if p.is_file() else {}
        objs = self._objects.get(vid, {}).get("objects", {})
        return {int(o): str(d.get("category", "")).strip() for o, d in objs.items()}

    def n_expressions(self) -> int:
        return sum(len(d.get("expressions", {})) for d in self.videos.values())
