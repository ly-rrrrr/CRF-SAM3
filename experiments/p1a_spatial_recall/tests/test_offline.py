"""CPU smoke test with synthetic data and a fake SAM3 session (no GPU, no weights).

Scene: two same-category objects (left = distractor already recalled by text,
right = target missed by text). The Ground map lights up BOTH (category-level),
so the right blob is "uncovered evidence"; rescue/point should recover it.
"""
import json, sys, tempfile
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from p1a import run_p1a, sam3_ops, common
from p1a.seeds import select_seeds

H, W, T = 120, 200, 10
def box_mask(x0, y0, x1, y1):
    m = np.zeros((H, W), bool); m[y0:y1, x0:x1] = True; return m
LEFT, RIGHT = box_mask(20, 40, 70, 100), box_mask(130, 40, 180, 100)

def build(tmp: Path):
    vd = tmp / "JPEGImages" / "vid"; vd.mkdir(parents=True)
    ad = tmp / "Annotations" / "vid"; ad.mkdir(parents=True)
    for t in range(T):
        Image.fromarray(np.full((H, W, 3), 128, np.uint8)).save(vd / f"{t*5:05d}.jpg")
        pal = np.zeros((H, W), np.uint8); pal[LEFT] = 1; pal[RIGHT] = 2
        Image.fromarray(pal).save(ad / f"{t*5:05d}.png")
    meta = {"videos": {"vid": {"expressions": {"0": {"exp": "the right box", "obj_id": "2"}}}}}
    (tmp / "meta.json").write_text(json.dumps(meta))
    # cache: text recall found only LEFT
    md = tmp / "cache" / "train" / "vid" / "0" / "masks" / "obj_0"; md.mkdir(parents=True)
    for t in range(T):
        Image.fromarray(LEFT.astype(np.uint8) * 255).save(md / f"{t:05d}.png")
    (md.parent.parent / "meta.json").write_text(json.dumps({"video_id": "vid", "exp_id": 0, "category": "box",
                                                           "sam3_prompt_text": "box", "expression": "the right box"}))
    # heatmap: grid 2*ht x 2*wt, category-level (both boxes bright)
    ht, wt = 12, 20
    def tok(m):
        import cv2
        return cv2.resize(m.astype(np.float32), (wt, ht), interpolation=cv2.INTER_AREA)
    base = np.clip(tok(LEFT | RIGHT) * 0.95 + 0.02, 0, 1).reshape(-1)
    M = 3
    resp = np.stack([np.concatenate([base] * M)] * 8).astype(np.float16)
    hd = tmp / "heat" / "vid"; hd.mkdir(parents=True)
    np.savez_compressed(hd / "0.npz", responses=resp, image_grid_thw=np.array([[1, ht * 2, wt * 2]] * M),
                        merge=np.array(2), frame_indices=np.array(common.sample_frame_indices(T, M)),
                        frame_stems=np.array(["x"] * M), orig_hw=np.array([H, W]), expression=np.array("the right box"))
    (tmp / "samples.json").write_text(json.dumps({"samples": [{"video_id": "vid", "exp_id": 0}]}))

class FakeSession:
    def __init__(self, predictor, video_path, offload=False): pass
    def __enter__(self): return self
    def __exit__(self, *e): pass
    def text_recall(self, *a, **k): return {0: {t: LEFT for t in range(T)}}
    def detect_on_frame(self, t, text, thresh=0.2, **k):
        return [dict(obj_id=1, score=0.9, mask=LEFT, box=[20/W, 40/H, 70/W, 100/H]),
                dict(obj_id=2, score=0.35, mask=RIGHT, box=[130/W, 40/H, 180/W, 100/H])]
    def track_seed(self, t, points=(), labels=(), box_xyxy=None, obj_id=1000):
        x = box_xyxy[0] if box_xyxy is not None else points[0][0]
        return {tt: (RIGHT if x > 0.5 else LEFT) for tt in range(T)}
    def exemplar_recall(self, t, box, text, k=24): return {0: {tt: LEFT for tt in range(T)}, 1: {tt: RIGHT for tt in range(T)}}

def test_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); build(tmp)
        sam3_ops.build_predictor = lambda *a, **k: None
        sam3_ops.SAM3Session = FakeSession
        argv = ["run_p1a", "--samples", str(tmp / "samples.json"), "--video-root", str(tmp / "JPEGImages"),
                "--heat-dir", str(tmp / "heat"), "--cache-root", str(tmp / "cache"), "--split", "train",
                "--gt", "palette", "--gt-kw", json.dumps({"annotation_root": str(tmp / "Annotations"), "meta_expressions": str(tmp / "meta.json")}),
                "--dms-repo", "x", "--sam3-checkpoint", "x", "--modes", "rescue,point,exemplar,oracle_gt", "--out", str(tmp / "out")]
        sys.argv = argv
        run_p1a.main()
        rec = json.loads((tmp / "out" / "results.jsonl").read_text().splitlines()[0])
        print(json.dumps({k: v for k, v in rec.items() if k != "seeds"}, indent=1)[:1500])
        assert rec["j_old"] < 0.01
        assert rec["n_uncovered"] >= 1, "right blob must be uncovered"
        for m in ("rescue", "point", "exemplar", "oracle_gt"):
            assert rec[m]["j_star"] > 0.99, m
        assert rec["rescue"]["added"] == 1

class FakeSessionDiag(FakeSession):
    """main word 'box' only sees LEFT; alt word 'crate' sees RIGHT at score 0.6."""
    def text_recall(self, text, *a, **k):
        return {0: {t: (RIGHT if text == "crate" else LEFT) for t in range(T)}}
    def detect_on_frame(self, t, text, thresh=0.2, **k):
        if text == "crate":
            return [dict(obj_id=2, score=0.6, mask=RIGHT, box=[130/W, 40/H, 180/W, 100/H])]
        return [dict(obj_id=1, score=0.9, mask=LEFT, box=[20/W, 40/H, 70/W, 100/H])]

def test_diagnostics_without_heatmap():
    from p1a import analyze
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); build(tmp)
        (tmp / "alt.json").write_text(json.dumps({"vid/0": "crate"}))
        sam3_ops.build_predictor = lambda *a, **k: None
        sam3_ops.SAM3Session = FakeSessionDiag
        sys.argv = ["run_p1a", "--samples", str(tmp / "samples.json"), "--video-root", str(tmp / "JPEGImages"),
                    "--cache-root", str(tmp / "cache"), "--split", "train",
                    "--gt", "palette", "--gt-kw", json.dumps({"annotation_root": str(tmp / "Annotations"), "meta_expressions": str(tmp / "meta.json")}),
                    "--dms-repo", "x", "--sam3-checkpoint", "x", "--modes", "det_probe,text_cfg,text_word,oracle_gt",
                    "--alt-word-json", str(tmp / "alt.json"), "--sam3-overrides", '{"new_det_thresh": 0.5}', "--out", str(tmp / "out")]
        run_p1a.main()
        rec = json.loads((tmp / "out" / "results.jsonl").read_text().splitlines()[0])
        assert rec["text_cfg"]["j_star"] < 0.01
        assert rec["text_word"]["info"][0]["j_star_replace"] > 0.99
        main = [p for p in rec["det_probe"]["info"] if p["word"] == "box"]
        assert main and all(p["best_iou"] < 0.01 for p in main)
        assert analyze.diagnose(rec, 0.5) == "word"
        rec2 = dict(rec, det_probe={"info": [dict(main[0], best_iou=0.8, best_score=0.4)]})
        assert analyze.diagnose(rec2, 0.5) == "below_threshold"
        # repropagate: a cached tube present on frames 0-3 only; re-seeding fills the rest
        md = tmp / "cache" / "train" / "vid" / "0" / "masks" / "obj_0"
        for t in range(4, T):
            (md / f"{t:05d}.png").unlink()
        Image.fromarray(np.zeros((H, W), np.uint8)).save(md / "00001.png")
        sam3_ops.SAM3Session = FakeSession
        sys.argv = sys.argv[:sys.argv.index("--modes") + 1] + ["repropagate", "--out", str(tmp / "out2")]
        run_p1a.main()
        rec3 = json.loads((tmp / "out2" / "results.jsonl").read_text().splitlines()[0])
        assert rec3["repropagate"]["added"] >= 1 and any(i["kind"] == "filled" for i in rec3["repropagate"]["info"])
        sys.argv = ["analyze", str(tmp / "out" / "results.jsonl"), "--out", str(tmp / "out" / "s.md")]
        analyze.main()
        assert "word" in (tmp / "out" / "s.causes.jsonl").read_text()

if __name__ == "__main__":
    test_end_to_end(); test_diagnostics_without_heatmap(); print("OK")
