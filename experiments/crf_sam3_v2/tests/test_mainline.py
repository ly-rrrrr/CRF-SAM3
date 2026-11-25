"""CPU tests of v2.mainline with a fake repo (no models, no GPU).  python tests/test_mainline.py

Video v (frames 00000, 00005, ... sparse like Ref-YTVOS): text recall finds only LEFT; GT and heatmap are RIGHT.
The point seed adds a RIGHT tube and the fake Verify prefers it, so row 'final' is right and row 'words' is not.
Checked: Ground v2 words reach the pipeline (category_override_cache_path), backup words are off, every mask
sits on its own frame (position-keyed) for both writers (Ref-YTVOS PNG, ReVOS RLE json), the SAM3 recall cache
runs SAM3 once per (video, word), and a missing Ground v2 word stops the run."""
import json
import os
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
P1A = Path(os.environ.get("P1A", HERE.parent / "p1a_spatial_recall")).resolve()
sys.path[:0] = [str(HERE), str(P1A)]
from p1a import sam3_ops  # noqa: E402
from v2 import mainline  # noqa: E402

H, W, T = 120, 200, 8
FR = [f"{5 * i:05d}" for i in range(T)]


def box(x0, x1):
    m = np.zeros((H, W), bool)
    m[40:100, x0:x1] = True
    return m


LEFT, RIGHT = box(20, 70), box(130, 180)


def right_at(t):  # moves 2 px per frame, so a frame mix-up is visible
    m = np.zeros((H, W), bool)
    m[40:100, 130 + 2 * t:180 + 2 * t] = True
    return m


DEMO = textwrap.dedent('''
    import json, numpy as np
    LEFT = np.zeros((120, 200), bool); LEFT[40:100, 20:70] = True
    CALLS = {"recall": 0}
    def build_sam3_recall_backend(*a, **k): return None
    def contains_color_attribute(e): return "red" in e
    class _CE:
        def __init__(self, cfg):
            recs = json.load(open(cfg["category_override_cache_path"]))["records"]
            self.words = {(r["video_id"], r["expression"]): r["pred_category"] for r in recs}
        def extract_category(self, expr, video_path=None):
            parts = video_path.rstrip("/").split("/")
            for i in range(len(parts) - 1, -1, -1):  # same suffix lookup as the real ConceptExtractor
                if ("/".join(parts[i:]), expr) in self.words:
                    return self.words[("/".join(parts[i:]), expr)], 1, {"backup_categories": ["mug"]}
            return "WRONG", 1, {"backup_categories": ["mug"]}
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
        def recall_all_instances_stream(self, video_path, category, candidate_top_k=None, candidate_keep_top_k=None):
            CALLS["recall"] += 1
            for t in range(8):
                yield t, {0: LEFT.astype(np.uint8)}, {0: 0.9}
    class CRFSAM3Pipeline:
        def __init__(self, **kw):
            self._original_visualization_style = kw["visualization_style"]
            self.concept_extractor, self.tube_builder, self.discriminator = _CE(kw["concept_config"]), _TB(), _D()
            self.sam3_recall = _Recall()
        def predict(self, video_path, expr, return_details=True):
            c, k, r = self.concept_extractor.extract_category(expr, video_path)
            assert r["backup_categories"] == [], "backup words must be off"
            assert c == "cup", c
            all_masks = {}
            for t, fm, s in self.sam3_recall.recall_all_instances_stream(video_path, c, candidate_top_k=24, candidate_keep_top_k=7):
                all_masks[t] = fm
            sel, det = self.discriminator.score_and_filter(self.tube_builder.build_tubes(video_path, all_masks, expr), expr, k, True)
            final = {t: {o: m for o, m in fm.items() if o in sel} for t, fm in all_masks.items()}
            return final, None, {"all_masks": all_masks, "num_targets": k, "category": c, "scores": det, "selected_candidates": sel}
''')

RUNNER = textwrap.dedent('''
    import numpy as np, yaml
    def load_config(p): return yaml.safe_load(open(p))
    def _infer_video_hw(vp, frames): return (120, 200)
    def _make_instance_mask(fm, h, w):
        out = np.zeros((h, w), np.uint8)
        for o, m in fm.items(): out |= (np.asarray(m) > 0).astype(np.uint8) * 50
        return out
''')

REVOS_RUNNER = textwrap.dedent('''
    import numpy as np
    from pycocotools import mask as mu
    def _compose_binary_mask(fm, h, w):
        out = np.zeros((h, w), bool)
        for m in fm.values(): out |= np.asarray(m) > 0
        return out
    def _encode_mask_to_rle(b):
        r = mu.encode(np.asfortranarray(b.astype(np.uint8))); r["counts"] = r["counts"].decode(); return r
''')


class FakeSession:
    def __init__(self, predictor, video_path, offload=False):
        assert predictor == "PRED"

    def __enter__(self):
        return self

    def __exit__(self, *e):
        pass

    def track_seed(self, t, points=(), labels=(), box_xyxy=None, obj_id=1000):
        assert points[0][0] > 0.5
        return {tt: right_at(tt) for tt in range(T)}


def fake_repo(d: Path) -> Path:
    for m in ("crf_sam3_demo", "run_crf_sam3_dataset", "run_crf_sam3_refytros_dataset", "run_crf_sam3_revos_dataset"):
        sys.modules.pop(m, None)
    repo = d / "repo"
    repo.mkdir()
    (repo / "crf_sam3_demo.py").write_text(DEMO)
    (repo / "run_crf_sam3_dataset.py").write_text(RUNNER)
    (repo / "run_crf_sam3_refytros_dataset.py").write_text("def _resolve_prompt_template(t): return 'x'\n")
    (repo / "run_crf_sam3_revos_dataset.py").write_text(REVOS_RUNNER)
    (repo / "cfg.yaml").write_text("model: {model_type: qwen3, mllm_path: m, sam3_checkpoint: s}\n"
                                   "pipeline: {num_tube_frames: 8}\nvisual_tube: {visualization_style: mask}\n"
                                   "discrimination: {backend: mllm, prompt_type: yesno}\n"
                                   "concept_extraction: {category_override_cache_path: OLD_V1_WORDS.json}\n")
    return repo


def write_video(vd: Path):
    vd.mkdir(parents=True)
    for f in FR:
        Image.fromarray(np.zeros((H, W, 3), np.uint8)).save(vd / f"{f}.jpg")


def write_ground(d: Path, vid: str, exprs, missing=()):
    import cv2

    ht, wt = 12, 20
    tok = cv2.resize(RIGHT.astype(np.float32), (wt, ht), interpolation=cv2.INTER_AREA)
    base = np.clip(tok * 0.95 + 0.02, 0, 1).reshape(-1)
    recs = []
    for eid, e in exprs.items():
        if eid in missing:
            continue
        recs.append({"video_id": vid, "exp_id": eid, "expression": e, "pred_category": "cup"})
        hd = d / "heat" / vid
        hd.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(hd / f"{eid}.npz", responses=np.concatenate([base] * 3)[None].astype(np.float16),
                            image_grid_thw=np.array([[1, ht * 2, wt * 2]] * 3), merge=np.array(2),
                            frame_indices=np.array([0, 3, 7]), frame_stems=np.array([FR[0], FR[3], FR[7]]),
                            orig_hw=np.array([H, W]), expression=np.array(e))
    (d / "cats.json").write_text(json.dumps({"records": recs}))


def call(argv):
    cwd = os.getcwd()
    try:
        return mainline.main(argv)
    finally:
        os.chdir(cwd)


EXPRS = {"0": "the right cup", "1": "cup on the right side"}


def test_refytvos():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        repo = fake_repo(d)
        root = d / "ytvos"
        write_video(root / "valid" / "JPEGImages" / "v")
        for eid in EXPRS:
            gd = root / "valid" / "Annotations" / "v" / eid
            gd.mkdir(parents=True)
            for t, f in enumerate(FR):
                Image.fromarray(right_at(t).astype(np.uint8) * 255).save(gd / f"{f}.png")
        (root / "valid" / "meta_expressions_challenge.json").write_text(json.dumps(
            {"videos": {"v": {"frames": FR, "expressions": {k: {"exp": v} for k, v in EXPRS.items()}}}}))
        write_ground(d, "v", EXPRS)
        sam3_ops.SAM3Session = FakeSession
        out = d / "out"
        call(["run", "--dataset", "refytvos", "--repo", str(repo), "--config", str(repo / "cfg.yaml"),
              "--data-root", str(root), "--categories", str(d / "cats.json"), "--heat-dir", str(d / "heat"),
              "--out", str(out), "--gt"])
        import crf_sam3_demo
        assert crf_sam3_demo.CALLS["recall"] == 1, "same video + same word -> SAM3 recall runs once"
        rs = [json.loads(l) for l in (out / "scores.jsonl").read_text().splitlines()]
        assert len(rs) == 2
        r = rs[0]
        assert r["category"] == "cup" and r["words_selected"] == [0] and r["point_ids"] == [1000]
        assert r["final_selected"] == [1000] and r["pool_j_words"] == 0.0 and r["pool_j_final"] == 1.0
        for t, f in enumerate(FR):  # position-keyed: frame 00005 holds position 1, not position 5
            m = np.array(Image.open(out / "final" / "Annotations" / "v" / "0" / f"{f}.png")) > 0
            assert (m == right_at(t)).all(), f
            b = np.array(Image.open(out / "words" / "Annotations" / "v" / "0" / f"{f}.png")) > 0
            assert (b == LEFT).all()
        try:
            call(["export", "--dataset", "refytvos", "--data-root", str(root), "--out", str(out)])
        except SystemExit as e:
            assert e.code == 0, e.code
        for row, name in (("words", "a"), ("final", "b")):
            (out / row / "evaluation_results_refytros_png.json").write_text(json.dumps(
                {"overall_expression_level": {"J": 1, "F": 2, "J&F": 1.5}, "overall_video_level": {"J&F": 1.5}}))
        call(["summary", "--dataset", "refytvos", "--out", str(out), "--ref", "paper=/nope.json"])
        s = (out / "summary.md").read_text()
        assert "Recall@0.5：0.00% → 100.00%" in s and "新主线" in s and "未找到" in s, s


def test_revos_nested_and_export():
    from pycocotools import mask as mu

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        repo = fake_repo(d)
        root = d / "revos"
        vid = "DS/train/v"
        write_video(root / vid)

        def rle(m):
            r = mu.encode(np.asfortranarray(m.astype(np.uint8)))
            r["counts"] = r["counts"].decode()
            return r

        (root / "mask_dict.json").write_text(json.dumps({"7": [rle(right_at(t)) for t in range(T)]}))
        meta = {"videos": {vid: {"frames": FR, "height": H, "width": W,
                                 "expressions": {k: {"exp": v, "anno_id": [7]} for k, v in EXPRS.items()}}}}
        (root / "meta_expressions_valid_.json").write_text(json.dumps(meta))
        write_ground(d, vid, EXPRS)
        sam3_ops.SAM3Session = FakeSession
        out = d / "out"
        call(["run", "--dataset", "revos", "--repo", str(repo), "--config", str(repo / "cfg.yaml"),
              "--data-root", str(root), "--categories", str(d / "cats.json"), "--heat-dir", str(d / "heat"),
              "--out", str(out), "--gt", "--limit", "1"])
        r = json.loads((out / "scores.jsonl").read_text().splitlines()[0])
        assert r["final_selected"] == [1000] and r["pool_j_final"] == 1.0
        try:
            call(["export", "--dataset", "revos", "--data-root", str(root), "--out", str(out)])
        except SystemExit as e:
            assert e.code == 0, e.code
        res = json.loads((out / "final" / "results.json").read_text())
        pm = res[vid]["0"]["prediction_masks"]
        for t in range(T):
            m = mu.decode(dict(pm[t], counts=pm[t]["counts"].encode())) > 0
            assert (m == right_at(t)).all(), t
        assert res[vid]["0"]["frames"] == FR


def test_missing_word_stops():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        repo = fake_repo(d)
        root = d / "ytvos"
        write_video(root / "valid" / "JPEGImages" / "v")
        (root / "valid" / "meta_expressions_challenge.json").write_text(json.dumps(
            {"videos": {"v": {"frames": FR, "expressions": {k: {"exp": v} for k, v in EXPRS.items()}}}}))
        write_ground(d, "v", EXPRS, missing=("1",))
        try:
            call(["run", "--dataset", "refytvos", "--repo", str(repo), "--config", str(repo / "cfg.yaml"),
                  "--data-root", str(root), "--categories", str(d / "cats.json"), "--heat-dir", str(d / "heat"),
                  "--out", str(d / "out")])
        except SystemExit as e:
            assert "no Ground v2 word" in str(e.code), e.code
        else:
            raise AssertionError("expected SystemExit")


def test_recall_cache_limit():
    class R:
        n = 0

        def recall_all_instances_stream(self, video_path, category, **kw):
            R.n += 1
            for t in range(3):
                yield t, {0: np.ones((10, 10), np.uint8)}, {0: 1.0}

    r = R()
    mainline.install_recall_cache(r, max_bytes=150)   # 3 x 100 bytes > limit -> not kept
    for _ in range(2):
        assert len(list(r.recall_all_instances_stream("v", "cup", session_id="x"))) == 3
    assert R.n == 2
    r2 = R()
    R.n = 0
    mainline.install_recall_cache(r2, max_bytes=1e6)
    list(r2.recall_all_instances_stream("v", "cup", session_id="a"))
    list(r2.recall_all_instances_stream("v", "cup", session_id="b"))
    list(r2.recall_all_instances_stream("v", "dog"))
    list(r2.recall_all_instances_stream("w", "cup"))
    assert R.n == 3


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("all mainline tests passed")
