"""CPU test of p1a.refyt_point with a fake repo (no models).

Video v (frames 00000, 00005, ..., sparse like Ref-YTVOS): text recall finds only LEFT; GT and heatmap are RIGHT.
The point seed adds a RIGHT tube, the fake Verify prefers it -> row 'point' is correct, row 'base' is not,
and every PNG sits on its own frame (position-keyed)."""
import json, os, sys, textwrap
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from p1a import refyt_point, sam3_ops

H, W, T = 120, 200, 8
FR = [f"{5 * i:05d}" for i in range(T)]
def box(x0, x1):
    m = np.zeros((H, W), bool); m[40:100, x0:x1] = True; return m
LEFT, RIGHT = box(20, 70), box(130, 180)
def right_at(t):  # moves 2 px per frame, so a frame mix-up is visible in the PNGs
    m = np.zeros((H, W), bool); m[40:100, 130 + 2 * t:180 + 2 * t] = True; return m

DEMO = textwrap.dedent('''
    import numpy as np
    LEFT = np.zeros((120, 200), bool); LEFT[40:100, 20:70] = True
    def build_sam3_recall_backend(*a, **k): return None
    def contains_color_attribute(e): return "red" in e
    class _CE:
        def extract_category(self, expr, video_path=None):
            return "cup", 1, {"backup_categories": ["mug"]}
    class _TB:
        visualization_style = "mask"
        def build_tubes(self, video_path, all_masks, expr):
            ids = sorted({o for fm in all_masks.values() for o in fm})
            return {o: {"m": [all_masks[t][o] for t in sorted(all_masks) if o in all_masks[t]]} for o in ids}
    class _D:
        def score_and_filter(self, tubes, expr, k, return_details=False, visualization_style=None):
            det = {o: {"score": float(np.where(np.any(tb["m"][0], 0))[0].mean()) / 200} for o, tb in tubes.items()}
            return sorted(det, key=lambda o: -det[o]["score"])[:k], det
    class _Recall:
        predictor = "PRED"
    class CRFSAM3Pipeline:
        def __init__(self, **kw):
            self._original_visualization_style = kw["visualization_style"]
            self.concept_extractor, self.tube_builder, self.discriminator, self.sam3_recall = _CE(), _TB(), _D(), _Recall()
        def predict(self, video_path, expr, return_details=True):
            c, k, r = self.concept_extractor.extract_category(expr, video_path)
            assert r["backup_categories"] == []
            all_masks = {t: {0: LEFT.astype(np.uint8)} for t in range(8)}
            sel, det = self.discriminator.score_and_filter(self.tube_builder.build_tubes(video_path, all_masks, expr), expr, k, True)
            final = {t: {o: m for o, m in fm.items() if o in sel} for t, fm in all_masks.items()}
            return final, None, {"all_masks": all_masks, "num_targets": k, "category": c, "scores": det, "selected_candidates": sel}
''')


class FakeSession:
    def __init__(self, predictor, video_path, offload=False):
        assert predictor == "PRED"
    def __enter__(self): return self
    def __exit__(self, *e): pass
    def track_seed(self, t, points=(), labels=(), box_xyxy=None, obj_id=1000):
        assert points[0][0] > 0.5
        return {tt: right_at(tt) for tt in range(T)}


def test_refyt_point(tmp_path, monkeypatch):
    for _m in ("crf_sam3_demo", "run_crf_sam3_dataset", "run_crf_sam3_refytros_dataset", "run_crf_sam3_revos_dataset"):
        sys.modules.pop(_m, None)  # fake repos of other tests
    repo = tmp_path / "repo"; repo.mkdir()
    (repo / "crf_sam3_demo.py").write_text(DEMO)
    (repo / "run_crf_sam3_dataset.py").write_text(textwrap.dedent('''
        import numpy as np, yaml
        def load_config(p): return yaml.safe_load(open(p))
        def _infer_video_hw(vp, frames): return (120, 200)
        def _make_instance_mask(fm, h, w):
            out = np.zeros((h, w), np.uint8)
            for o, m in fm.items(): out |= (np.asarray(m) > 0).astype(np.uint8) * 50
            return out
    '''))
    (repo / "run_crf_sam3_refytros_dataset.py").write_text("def _resolve_prompt_template(t): return 'x'\n")
    (repo / "cfg.yaml").write_text("model: {model_type: qwen3, mllm_path: m, sam3_checkpoint: s}\n"
                                   "pipeline: {num_tube_frames: 8}\nvisual_tube: {visualization_style: mask}\n"
                                   "discrimination: {backend: mllm, prompt_type: yesno}\n")
    root = tmp_path / "ytvos"
    vd = root / "valid" / "JPEGImages" / "v"; vd.mkdir(parents=True)
    gd = root / "valid" / "Annotations" / "v" / "0"; gd.mkdir(parents=True)
    for t, f in enumerate(FR):
        Image.fromarray(np.zeros((H, W, 3), np.uint8)).save(vd / f"{f}.jpg")
        Image.fromarray(right_at(t).astype(np.uint8) * 255).save(gd / f"{f}.png")
    (root / "valid" / "meta_expressions_challenge.json").write_text(json.dumps(
        {"videos": {"v": {"frames": FR, "expressions": {"0": {"exp": "the right cup"}}}}}))
    import cv2
    ht, wt = 12, 20
    tok = cv2.resize(RIGHT.astype(np.float32), (wt, ht), interpolation=cv2.INTER_AREA)
    base = np.clip(tok * 0.95 + 0.02, 0, 1).reshape(-1)
    hd = tmp_path / "heat" / "v"; hd.mkdir(parents=True)
    np.savez_compressed(hd / "0.npz", responses=np.concatenate([base] * 3)[None].astype(np.float16),
                        image_grid_thw=np.array([[1, ht * 2, wt * 2]] * 3), merge=np.array(2),
                        frame_indices=np.array([0, 3, 7]), frame_stems=np.array([FR[0], FR[3], FR[7]]),
                        orig_hw=np.array([H, W]), expression=np.array("the right cup"))
    monkeypatch.setattr(sam3_ops, "SAM3Session", FakeSession)
    out = tmp_path / "out"
    cwd = os.getcwd()
    try:
        refyt_point.main(["--repo", str(repo), "--config", str(repo / "cfg.yaml"), "--data-root", str(root),
                          "--heat-dir", str(tmp_path / "heat"), "--out", str(out), "--gt"])
    finally:
        os.chdir(cwd)
    r = json.loads((out / "scores.jsonl").read_text().splitlines()[0])
    assert r["base_selected"] == [0] and r["point_ids"] == [1000] and r["point_selected"] == [1000]
    assert r["pool_j_base"] == 0.0 and r["pool_j_point"] == 1.0
    for t, f in enumerate(FR):  # position-keyed: frame 00005 holds position 1, not position 5
        m = np.array(Image.open(out / "point" / "Annotations" / "v" / "0" / f"{f}.png")) > 0
        assert (m == right_at(t)).all(), f
        b = np.array(Image.open(out / "base" / "Annotations" / "v" / "0" / f"{f}.png")) > 0
        assert (b == LEFT).all()
    refyt_point.summary(["--out", str(out)])
    assert "Recall@0.5：0.00% → 100.00%" in (out / "summary.md").read_text()
