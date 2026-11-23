"""track_seed falls back to a text-warmed session when the tracker refuses a prompt without cached outputs."""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from p1a import sam3_ops
RealSession = sam3_ops.SAM3Session  # other tests monkeypatch sam3_ops.SAM3Session

class FakePredictor:
    """Mimics SAM3-DMS: point prompts need a prior propagation; text objects get ids 0.., the point object its obj_id."""
    model = object()
    def __init__(self): self.log = []; self.cached = False; self.objs = []
    def handle_request(self, r):
        self.log.append(r["type"] + ("/text" if "text" in r else "/points" if "points" in r else ""))
        if r["type"] == "start_session": return {"session_id": "s"}
        if r["type"] == "reset_session": self.cached = False; self.objs = []
        if r["type"] == "add_prompt":
            if "points" in r:
                assert self.cached, "No cached outputs found. Ensure normal propagation has run first to populate the cache."
                self.objs.append(r["obj_id"])
            if "text" in r: self.objs.append(0)
        return {}
    def handle_stream_request(self, r):
        self.cached = True
        for t in range(4):
            m = np.zeros((1, 8, 8), bool); m[0, 2:5, 2:5] = True
            yield {"frame_index": t, "outputs": {"out_obj_ids": np.array(self.objs), "out_binary_masks": np.stack([m] * len(self.objs)) if self.objs else None,
                                                 "out_probs": np.ones(len(self.objs))}}

def test_warm_fallback():
    sam3_ops.STATE['needs_warm'] = None
    p = FakePredictor()
    s = RealSession(p, "v"); s.warm_text = "cat"
    tube = s.track_seed(1, points=[[0.3, 0.3]], labels=[1], obj_id=1000)
    assert len(tube) == 4 and sam3_ops.STATE['needs_warm'] is True
    p.log.clear()
    tube = s.track_seed(2, box_xyxy=[0.1, 0.1, 0.5, 0.5], obj_id=1001)
    assert len(tube) == 4 and p.log.count("add_prompt/text") == 1, p.log
    sam3_ops.STATE['needs_warm'] = None

if __name__ == "__main__":
    test_warm_fallback(); print("OK")
