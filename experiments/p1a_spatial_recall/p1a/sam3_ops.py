"""Thin wrapper over the SAM3-DMS video predictor (frozen, no code changes).

Prompt semantics, as implemented in SAM3-DMS/sam3/model/sam3_video_inference.py:

* ``add_prompt(text=...)``                  -> concept prompt for ALL frames
  (reset_state first; detector runs on every frame during propagation).
* ``add_prompt(text=..., bounding_boxes=[b])`` on frame t
  -> one *semantic* query: text on every frame + box exemplar on frame t only
  (``per_frame_geometric_prompt[t]``). The exemplar means "objects that look
  like this"; it is not instance-specific. It also resets the session.
* ``add_prompt(bounding_boxes=[b])`` without text -> exemplar-only, and new
  detections are allowed only on frame t (``allow_new_detections``), then the
  tracker carries them.
* ``add_prompt(points=..., point_labels=..., obj_id=k)`` -> tracker instance
  prompt (SAM2-style PVS). Labels 1/0 = positive/negative click, 2/3 = box
  corners (sam/prompt_encoder.py). No detector involvement.
* New tracks are spawned only from detections scoring >= ``new_det_thresh``
  (0.7 in model_builder.py), which is why some visible targets never become
  candidates. ``detect_on_frame(..., thresh=0.2)`` exposes them on one frame.

One ``SAM3Session`` == one loaded video; operations reset the session between
calls so the decoded frames / feature cache are reused.
"""

from __future__ import annotations

import contextlib
import sys
import uuid
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

Tube = Dict[int, np.ndarray]
# whether tracker prompts need a text-warmed session; learned on the first track_seed call
STATE: Dict[str, Optional[bool]] = {"needs_warm": None}


@contextlib.contextmanager
def _prepend(path: str):
    sys.path.insert(0, path)
    try:
        yield
    finally:
        try:
            sys.path.remove(path)
        except ValueError:
            pass


def build_predictor(dms_repo: str, checkpoint: str, gpu: int = 0):
    with _prepend(dms_repo):
        from sam3.model_builder import build_sam3_video_predictor

        return build_sam3_video_predictor(gpus_to_use=[gpu], checkpoint_path=checkpoint)


def _np(x):
    try:
        import torch

        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
    except ImportError:  # pragma: no cover
        pass
    return np.asarray(x)


def _mask(m) -> np.ndarray:
    a = _np(m)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    return a > 0


class SAM3Session:
    def __init__(self, predictor, video_path: str, offload_video_to_cpu: bool = False):
        self.p = predictor
        self.video_path = video_path
        self.warm_text: Optional[str] = None
        resp = self.p.handle_request(
            dict(
                type="start_session",
                resource_path=video_path,
                session_id=f"p1a_{uuid.uuid4().hex}",
                offload_video_to_cpu=offload_video_to_cpu,
            )
        )
        self.sid = resp["session_id"]

    # ------------------------------------------------------------------
    def close(self):
        self.p.handle_request(dict(type="close_session", session_id=self.sid))
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def reset(self):
        self.p.handle_request(dict(type="reset_session", session_id=self.sid))

    @contextlib.contextmanager
    def _model_attrs(self, **attrs):
        model = self.p.model
        old = {k: getattr(model, k) for k in attrs if hasattr(model, k)}
        for k, v in attrs.items():
            if v is not None and hasattr(model, k):
                setattr(model, k, v)
        try:
            yield
        finally:
            for k, v in old.items():
                setattr(model, k, v)

    def _propagate(self, direction: str = "both", start: Optional[int] = None) -> Tuple[Dict[int, Tube], Dict[int, float]]:
        req = dict(type="propagate_in_video", session_id=self.sid, propagation_direction=direction)
        if start is not None:
            req["start_frame_index"] = int(start)
        tubes: Dict[int, Tube] = {}
        score_sum: Dict[int, float] = {}
        for resp in self.p.handle_stream_request(req):
            t = int(resp["frame_index"])
            out = resp.get("outputs") or {}
            ids = _np(out.get("out_obj_ids", [])).reshape(-1).tolist()
            masks = out.get("out_binary_masks")
            probs = out.get("out_probs")
            if masks is None or not ids:
                continue
            probs = None if probs is None else _np(probs).reshape(-1).tolist()
            for i, oid in enumerate(ids):
                m = _mask(masks[i])
                if not m.any():
                    continue
                tubes.setdefault(int(oid), {})[t] = m
                if probs is not None and i < len(probs):
                    score_sum[int(oid)] = score_sum.get(int(oid), 0.0) + float(probs[i])
        return tubes, score_sum

    # ------------------------------------------------------------------
    # 1) the paper's text recall (for samples without an offline cache)
    # ------------------------------------------------------------------
    def text_recall(self, text: str, recall_top_k: int = 24, keep_top_k: int = 7,
                    overrides: Optional[Dict[str, float]] = None) -> Dict[int, Tube]:
        """Same protocol as DMSConceptRecall (no reverse reconditioning).

        ``overrides`` temporarily sets SAM3 video heuristics, e.g.
        {"new_det_thresh": 0.5, "det_nms_thresh": 0.3, "hotstart_delay": 0,
         "hotstart_unmatch_thresh": 0, "hotstart_dup_thresh": 0,
         "suppress_overlapping_based_on_recent_occlusion_threshold": 0.0,
         "recondition_every_nth_frame": 8}
        (attribute names from sam3_video_base.py / model_builder.py).
        """
        self.reset()
        attrs = dict(overrides or {})
        attrs["max_num_objects"] = recall_top_k
        with self._model_attrs(**attrs):
            self.p.handle_request(dict(type="add_prompt", session_id=self.sid, frame_index=0, text=text))
            tubes, score = self._propagate("both")
        if keep_top_k and len(tubes) > keep_top_k:
            keep = sorted(tubes, key=lambda k: (-score.get(k, 0.0), k))[:keep_top_k]
            tubes = {k: tubes[k] for k in keep}
        return tubes

    # ------------------------------------------------------------------
    # 2) low-threshold detection on one frame (no tracking)
    # ------------------------------------------------------------------
    def detect_on_frame(self, frame_idx: int, text: Optional[str], thresh: float = 0.2,
                        exemplar_box_xyxy: Optional[Sequence[float]] = None,
                        max_dets: int = 50) -> List[Dict]:
        """All SAM3 detections on ``frame_idx`` with score >= ``thresh``.

        Lowers ``score_threshold_detection`` and ``new_det_thresh`` only for this
        single-frame call, then restores them. Returns masks, xyxy boxes
        (normalised) and scores.
        """
        self.reset()
        req = dict(type="add_prompt", session_id=self.sid, frame_index=int(frame_idx))
        if text:
            req["text"] = text
        if exemplar_box_xyxy is not None:
            x0, y0, x1, y1 = [float(v) for v in exemplar_box_xyxy]
            req["bounding_boxes"] = [[x0, y0, max(x1 - x0, 1e-3), max(y1 - y0, 1e-3)]]
            req["bounding_box_labels"] = [1]
        with self._model_attrs(score_threshold_detection=thresh, new_det_thresh=thresh, max_num_objects=max_dets):
            resp = self.p.handle_request(req)
        out = resp.get("outputs") or {}
        ids = _np(out.get("out_obj_ids", [])).reshape(-1).tolist()
        masks = out.get("out_binary_masks")
        probs = _np(out.get("out_probs", np.zeros(len(ids)))).reshape(-1).tolist()
        dets = []
        for i, oid in enumerate(ids):
            m = _mask(masks[i])
            if not m.any():
                continue
            ys, xs = np.where(m)
            h, w = m.shape
            dets.append(
                dict(
                    obj_id=int(oid),
                    score=float(probs[i]) if i < len(probs) else float("nan"),
                    mask=m,
                    box=[xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h],
                )
            )
        self.reset()
        return dets

    # ------------------------------------------------------------------
    # 3) tracker instance seed -> one tube
    # ------------------------------------------------------------------
    def track_seed(self, frame_idx: int, points: Sequence[Sequence[float]] = (), labels: Sequence[int] = (),
                   box_xyxy: Optional[Sequence[float]] = None, obj_id: int = 1000) -> Tube:
        """SAM2-style instance prompt on ``frame_idx`` and bidirectional propagation.

        ``box_xyxy`` is passed as corner points with labels 2/3 (first, as SAM
        requires); ``points``/``labels`` are positive (1) / negative (0) clicks.
        Coordinates are normalised.

        Some SAM3-DMS versions refuse a tracker prompt in a fresh session
        ("No cached outputs found. Ensure normal propagation has run first").
        Then the session is first warmed with one text propagation
        (``self.warm_text``, the sample's word, or "object"), the point object is
        added with a separate obj_id, and only that object's tube is returned.
        The first failure switches the whole run to the warm path.
        """
        import torch

        pts: List[List[float]] = []
        lbs: List[int] = []
        if box_xyxy is not None:
            x0, y0, x1, y1 = [float(v) for v in box_xyxy]
            pts += [[x0, y0], [x1, y1]]
            lbs += [2, 3]
        pts += [[float(x), float(y)] for x, y in points]
        lbs += [int(l) for l in labels]
        if not pts:
            raise ValueError("track_seed needs a box or at least one point")
        req = dict(type="add_prompt", session_id=self.sid, frame_index=int(frame_idx),
                   points=torch.tensor(pts, dtype=torch.float32),
                   point_labels=torch.tensor(lbs, dtype=torch.int32), obj_id=int(obj_id))
        if STATE["needs_warm"] is not True:
            self.reset()
            try:
                self.p.handle_request(dict(req))
                tubes, _ = self._propagate("both", start=frame_idx)
                STATE["needs_warm"] = False
                self.reset()
                return tubes.get(int(obj_id), {})
            except AssertionError as e:
                if "cached outputs" not in str(e):
                    raise
                STATE["needs_warm"] = True
                print("[sam3_ops] tracker prompt needs a prior propagation; using text-warmed sessions", flush=True)
        self.reset()
        self.p.handle_request(dict(type="add_prompt", session_id=self.sid, frame_index=int(frame_idx),
                                   text=(self.warm_text or "object")))
        self._propagate("both", start=frame_idx)
        self.p.handle_request(dict(req))
        tubes, _ = self._propagate("both", start=frame_idx)
        self.reset()
        return tubes.get(int(obj_id), {})

    # ------------------------------------------------------------------
    # 4) text + exemplar box on one frame (full propagation)
    # ------------------------------------------------------------------
    def exemplar_recall(self, frame_idx: int, box_xyxy: Sequence[float], text: Optional[str],
                        recall_top_k: int = 24) -> Dict[int, Tube]:
        self.reset()
        x0, y0, x1, y1 = [float(v) for v in box_xyxy]
        req = dict(
            type="add_prompt",
            session_id=self.sid,
            frame_index=int(frame_idx),
            bounding_boxes=[[x0, y0, max(x1 - x0, 1e-3), max(y1 - y0, 1e-3)]],
            bounding_box_labels=[1],
        )
        if text:
            req["text"] = text
        with self._model_attrs(max_num_objects=recall_top_k):
            self.p.handle_request(req)
            tubes, _ = self._propagate("both", start=frame_idx)
        self.reset()
        return tubes
