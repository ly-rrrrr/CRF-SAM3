"""CPU test of p1a.verify_pool with a fake CRF-SAM3 repo (no model).

The fake scorer prefers masks on the RIGHT. Video a: old word recalled LEFT only, v2 word recalled LEFT + RIGHT;
video b: v2 word got 0 candidates, the point tube is RIGHT. So the 'old' row misses, 'v2' hits a, 'v2_point' hits both.
With CRF_REPO=<real Sa2VA-main> the real run_crf_sam3_revos_dataset._build_revos_rles and tools/eval/eval_revos.py are used.
"""
import csv, gzip, json, os, shutil, subprocess, sys, textwrap
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from p1a import common, verify_pool

H, W, T = 40, 60, 5
def box(x0, x1):
    m = np.zeros((H, W), bool); m[10:30, x0:x1] = True; return m
LEFT, RIGHT = box(5, 25), box(35, 55)

FAKE_DEMO = textwrap.dedent('''
    import numpy as np
    def build_sam3_recall_backend(*a, **k):
        raise AssertionError("SAM3 must not be loaded")
    def contains_color_attribute(e):
        return "red" in e
    class _CE:
        def extract_category(self, expr, video_path):
            return "thing", (2 if "both" in expr else 1), {"backup_categories": ["never used"]}
    class _TB:
        def __init__(self): self.visualization_style = "mask"; self.calls = []
        def build_tubes(self, video_path, all_masks, expr):
            self.calls.append(self.visualization_style)
            ids = sorted({o for fm in all_masks.values() for o in fm})
            return {o: {"frame_paths": ["x"], "masks": [all_masks[t][o] for t in sorted(all_masks) if o in all_masks[t]]} for o in ids}
    class _D:
        def score_and_filter(self, tubes, expr, k, return_details=False, visualization_style=None):
            det = {}
            for o, tb in tubes.items():
                xs = np.where(np.any(tb["masks"][0], axis=0))[0]
                det[o] = {"score": float(xs.mean()) / 100.0}
            return sorted(det, key=lambda o: -det[o]["score"])[:k], det
    class CRFSAM3Pipeline:
        def __init__(self, **kw):
            import crf_sam3_demo as me
            me.build_sam3_recall_backend()
            self._original_visualization_style = kw["visualization_style"]
            self.concept_extractor, self.tube_builder, self.discriminator = _CE(), _TB(), _D()
            me.LAST = self
''')

FAKE_DATASET = textwrap.dedent('''
    import json, os, numpy as np
    from PIL import Image
    def load_config(p):
        import yaml
        return yaml.safe_load(open(p))
    def setup_logging(*a, **k): pass
    def clear_cuda_cache(*a, **k): pass
    def save_masks(*a, **k): pass
    def _infer_video_hw(video_path, frames):
        a = np.array(Image.open(os.path.join(video_path, frames[0] + ".jpg")))
        return a.shape[0], a.shape[1]
''')

FAKE_REVOS = textwrap.dedent('''
    import numpy as np
    from pycocotools import mask as cocomask
    def _resolve_prompt_template(t): return "configs/x.txt"
    def _build_revos_rles(masks_dict, frame_names, h, w):
        out = []
        for i, fn in enumerate(frame_names):
            m = np.zeros((h, w), np.uint8)
            for o, mk in masks_dict.get(i, {}).items():
                m = np.maximum(m, (np.asarray(mk) > 0).astype(np.uint8))
            r = cocomask.encode(np.asfortranarray(m)); r["counts"] = r["counts"].decode("utf-8"); out.append(r)
        return out
''')


def build(tmp: Path):
    repo = tmp / "repo"; repo.mkdir()
    (repo / "crf_sam3_demo.py").write_text(FAKE_DEMO)
    (repo / "run_crf_sam3_dataset.py").write_text(FAKE_DATASET)
    real = os.environ.get("CRF_REPO")
    if real:
        shutil.copy(Path(real) / "run_crf_sam3_revos_dataset.py", repo)
        (repo / "tools/eval").mkdir(parents=True)
        shutil.copy(Path(real) / "tools/eval/eval_revos.py", repo / "tools/eval")
        shutil.copytree(Path(real) / "third_parts/revos/utils", repo / "third_parts/revos/utils")
    else:
        (repo / "run_crf_sam3_revos_dataset.py").write_text(FAKE_REVOS)
    (repo / "cfg.yaml").write_text("model: {model_type: qwen3, mllm_path: m, sam3_checkpoint: s}\n"
                                   "pipeline: {num_tube_frames: 4}\nvisual_tube: {visualization_style: mask}\n"
                                   "discrimination: {backend: mllm, prompt_type: yesno_images}\n")
    root = tmp / "revos"
    meta = {"videos": {}}
    md, fg = {}, {}
    exps = {"S/a": {"0": ("the right one", 0), "1": ("the red right one", 1)}, "S/b": {"0": ("both right", 0)}}
    for v, ex in exps.items():
        vd = root / v; vd.mkdir(parents=True)
        for t in range(T):
            Image.fromarray(np.full((H, W, 3), 100, np.uint8)).save(vd / f"{t:05d}.jpg")
        meta["videos"][v] = {"frames": [f"{t:05d}" for t in range(T)], "height": H, "width": W,
                             "expressions": {e: {"exp": x, "anno_id": [v], "type_id": ty} for e, (x, ty) in ex.items()}}
        md[v] = [common.encode_rle(RIGHT)] * T
        fg[v] = {"masks_rle": [common.encode_rle(np.ones((H, W), bool))] * T}
    (root / "meta.json").write_text(json.dumps(meta))
    (root / "mask_dict.json").write_text(json.dumps(md))
    (root / "fg.json").write_text(json.dumps(fg))
    pools = tmp / "pools"; pools.mkdir()
    def pool(v, word, tubes):
        rles = {str(k): [common.encode_rle(m) for m in seq] for k, seq in tubes.items()}
        with gzip.open(common.pool_file(pools, v, word), "wt") as f:
            json.dump({"video_id": v, "word": word, "rles": rles}, f)
    pool("S/a", "cup", {3: [LEFT] * T})
    pool("S/a", "mug", {0: [LEFT] * T, 1: [RIGHT] * T})
    pool("S/b", "box", {0: [LEFT] * T})
    pool("S/b", "giraffe", {})
    rows = [("S/a", "0", "cup", "mug"), ("S/a", "1", "cup", "mug"), ("S/b", "0", "box", "giraffe")]
    with open(tmp / "per_expression.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["video_id", "exp_id", "type_id", "old_word", "new_word"])
        for r in rows:
            w.writerow([r[0], r[1], 0, r[2], r[3]])
    pts = tmp / "point_all"
    for v, e in (("S/a", "0"), ("S/a", "1"), ("S/b", "0")):
        d = pts / "tubes" / v; d.mkdir(parents=True, exist_ok=True)
        if v == "S/b":
            (d / f"{e}.json").write_text(json.dumps({"point": [common.tube_to_rle({t: RIGHT for t in range(T)})]}))
    (pts / "results.jsonl").write_text("".join(json.dumps({"key": f"{v}/{e}"}) + "\n" for v, e, *_ in rows))
    return repo, root, pools, pts


def test_verify_pool(tmp_path):
    for _m in ("crf_sam3_demo", "run_crf_sam3_dataset", "run_crf_sam3_refytros_dataset", "run_crf_sam3_revos_dataset"):
        sys.modules.pop(_m, None)  # fake repos of other tests
    repo, root, pools, pts = build(tmp_path)
    out = tmp_path / "out"
    cwd = os.getcwd()
    try:
        verify_pool.main(["predict", "--repo", str(repo), "--config", str(repo / "cfg.yaml"), "--revos-root", str(root),
                          "--meta", str(root / "meta.json"), "--per-expression", str(tmp_path / "per_expression.csv"),
                          "--pool-dir", str(pools), "--point-dir", str(pts), "--out", str(out)])
    finally:
        os.chdir(cwd)
    import crf_sam3_demo
    assert crf_sam3_demo.LAST.tube_builder.calls == ["mask", "bbox", "mask"]   # colour expression -> bbox, then restored
    assert crf_sam3_demo.LAST.tube_builder.visualization_style == "mask"
    recs = {r["key"]: r for r in map(json.loads, (out / "scores.jsonl").read_text().splitlines())}
    a0, b0 = recs["S/a/0"], recs["S/b/0"]
    assert a0["ids"] == {"old": [503], "v2": [0, 1], "point": []}
    assert a0["selected"] == {"old": [503], "v2": [1], "v2_point": [1]}
    assert b0["num_targets"] == 2 and b0["ids"]["v2"] == [] and b0["ids"]["point"] == [1000]
    assert b0["selected"] == {"old": [500], "v2": [], "v2_point": [1000]}
    # resumable: a second run does nothing
    verify_pool.main(["predict", "--repo", str(repo), "--config", str(repo / "cfg.yaml"), "--revos-root", str(root),
                      "--meta", str(root / "meta.json"), "--per-expression", str(tmp_path / "per_expression.csv"),
                      "--pool-dir", str(pools), "--point-dir", str(pts), "--out", str(out)])
    os.chdir(cwd)
    assert len((out / "scores.jsonl").read_text().splitlines()) == 3
    try:
        verify_pool.main(["merge", "--meta", str(root / "meta.json"), "--out", str(out)])
    except SystemExit as e:
        assert e.code == 0
    res = json.loads((out / "v2_point" / "results.json").read_text())
    m = common.decode_rle(res["S/b"]["0"]["prediction_masks"][2])
    assert (m == RIGHT).all()
    m = common.decode_rle(json.loads((out / "v2" / "results.json").read_text())["S/b"]["0"]["prediction_masks"][2])
    assert m is None or not m.any()

    if os.environ.get("CRF_REPO"):  # the real official evaluation on the three rows
        for row in ("old", "v2", "v2_point"):
            subprocess.run([sys.executable, "tools/eval/eval_revos.py", str(out / row / "results.json"),
                            "--exp_path", str(root / "meta.json"), "--mask_path", str(root / "mask_dict.json"),
                            "--foreground_mask_path", str(root / "fg.json"), "--workers", "1",
                            "--save_json_name", "revos_valid_official.json"], cwd=repo, check=True,
                           capture_output=True)
        verify_pool.main(["compare", "--meta", str(root / "meta.json"), "--out", str(out)])
        jf = {row: json.loads((out / row / "revos_valid_official.json").read_text())["overall"]["JF"]
              for row in ("old", "v2", "v2_point")}
        assert jf["old"] < jf["v2"] < jf["v2_point"]
        assert "v2 词池 + 热力图点种子" in (out / "compare.md").read_text()


def test_fidelity(tmp_path):
    out = tmp_path / "o"; out.mkdir()
    rec = {"key": "S/a/0", "old_word": "cup", "ids": {"old": [503, 504], "v2": [], "point": []},
           "scores": {"503": 0.9, "504": 0.2}, "selected": {"old": [503]}}
    (out / "scores.jsonl").write_text(json.dumps(rec) + "\n")
    hist = [{"video_id": "S/a", "exp_id": "0", "used_category_query": "cup", "selected_candidates": [3],
             "scores": {"3": {"score": 0.9}, "4": {"score": 0.2}}}]
    (tmp_path / "s.json").write_text(json.dumps(hist))
    assert verify_pool.fidelity(verify_pool.argparse.Namespace(out=str(out), paper_scores=str(tmp_path / "s.json"))) == 0
    # one bf16 step near a tie flips the choice -> still OK
    hist[0]["scores"] = {"3": {"score": 0.4378}, "4": {"score": 0.4073}}
    rec.update(scores={"503": 0.4073, "504": 0.4378}, selected={"old": [504]})
    (out / "scores.jsonl").write_text(json.dumps(rec) + "\n")
    (tmp_path / "s.json").write_text(json.dumps(hist))
    assert verify_pool.fidelity(verify_pool.argparse.Namespace(out=str(out), paper_scores=str(tmp_path / "s.json"))) == 0
    # a real disagreement (clear winner in the paper, other choice here) -> MISMATCH
    hist[0]["scores"] = {"3": {"score": 0.9}, "4": {"score": 0.2}}
    rec.update(scores={"503": 0.2, "504": 0.9}, selected={"old": [504]})
    (out / "scores.jsonl").write_text(json.dumps(rec) + "\n")
    (tmp_path / "s.json").write_text(json.dumps(hist))
    assert verify_pool.fidelity(verify_pool.argparse.Namespace(out=str(out), paper_scores=str(tmp_path / "s.json"))) == 1
