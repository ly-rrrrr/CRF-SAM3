"""CPU test of the ReVOS v2 path: pools from gv2.recall_pool + v2 heatmap npz + ReVOS GT, fake SAM3.

Video A: the v2 word recalled only the distractor (LEFT); target RIGHT is lit by the heatmap -> 'fail' subset.
Video B: the v2 word got 0 candidates; the heatmap lights the target -> 'zero' subset (word-free point seed).
"""
import csv, gzip, json, sys, tempfile
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from p1a import run_p1a, sam3_ops, common, revos_subsets, revos_summary

H, W, T = 120, 200, 6
def box(x0, y0, x1, y1):
    m = np.zeros((H, W), bool); m[y0:y1, x0:x1] = True; return m
LEFT, RIGHT = box(20, 40, 70, 100), box(130, 40, 180, 100)
opened = []

class FakeSession:
    def __init__(self, predictor, video_path, offload=False): opened.append(video_path)
    def __enter__(self): return self
    def __exit__(self, *e): pass
    def detect_on_frame(self, t, text, thresh=0.2, **k):
        return [dict(obj_id=2, score=0.3, mask=RIGHT, box=[130/W, 40/H, 180/W, 100/H])]
    def track_seed(self, t, points=(), labels=(), box_xyxy=None, obj_id=1000):
        x = box_xyxy[0] if box_xyxy is not None else points[0][0]
        return {tt: (RIGHT if x > 0.5 else LEFT) for tt in range(T)}

def build(tmp):
    root = tmp / "revos"
    meta = {"videos": {}}
    md = {}
    for v, aid in (("S/a", 1), ("S/b", 2)):
        vd = root / v; vd.mkdir(parents=True)
        for t in range(T):
            Image.fromarray(np.full((H, W, 3), 128, np.uint8)).save(vd / f"{t:05d}.jpg")
        meta["videos"][v] = {"frames": [f"{t:05d}" for t in range(T)], "height": H, "width": W,
                             "expressions": {"0": {"exp": "the right one", "anno_id": [aid]}, "1": {"exp": "x", "anno_id": [aid]}}}
        md[str(aid)] = [common.encode_rle(RIGHT)] * T
    (root / "meta_expressions_valid_.json").write_text(json.dumps(meta))
    (root / "mask_dict.json").write_text(json.dumps(md))
    pools = tmp / "pools"; pools.mkdir()
    def write_pool(v, word, tubes):
        rles = {str(k): [common.encode_rle(m) for m in seq] for k, seq in tubes.items()}
        with gzip.open(common.pool_file(pools, v, word), "wt") as f:
            json.dump({"video_id": v, "word": word, "rles": rles}, f)
    write_pool("S/a", "cup", {0: [LEFT] * T})
    write_pool("S/b", "giraffe", {})
    heat = tmp / "heat"
    ht, wt = 12, 20
    import cv2
    tok = lambda m: cv2.resize(m.astype(np.float32), (wt, ht), interpolation=cv2.INTER_AREA)
    for v in ("S/a", "S/b"):
        for e in ("0", "1"):
            base = np.clip(tok(RIGHT) * 0.95 + 0.02, 0, 1).reshape(-1)
            d = heat / v; d.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(d / f"{e}.npz", responses=np.concatenate([base] * 3)[None].astype(np.float16),
                                image_grid_thw=np.array([[1, ht * 2, wt * 2]] * 3), merge=np.array(2),
                                frame_indices=np.array([0, 2, 5]), frame_stems=np.array(["a"] * 3),
                                orig_hw=np.array([H, W]), expression=np.array("the right one"))
    (tmp / "cats.json").write_text(json.dumps({"records": [
        {"video_id": "S/a", "exp_id": e, "pred_category": "Cup"} for e in ("0", "1")] + [
        {"video_id": "S/b", "exp_id": e, "pred_category": "giraffe"} for e in ("0", "1")]}))
    with open(tmp / "per_expression.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "exp_id", "type_id", "new_word", "new_n", "new_oracle_j", "expression"])
        for e in ("0", "1"):
            w.writerow(["S/a", e, 0, "cup", 1, 0.0, "the right one"])
            w.writerow(["S/b", e, 1, "giraffe", 0, 0.0, "the right one"])
        w.writerow(["S/c", "0", 0, "dog", 2, 0.9, "ok one"])
    return root

def test_revos_v2_path():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); root = build(tmp)
        revos_subsets.main(["--per-expression", str(tmp / "per_expression.csv"), "--out", str(tmp / "runs")])
        assert len(json.loads((tmp / "runs/samples_zero.json").read_text())["samples"]) == 2
        sam3_ops.build_predictor = lambda *a, **k: None
        sam3_ops.SAM3Session = FakeSession
        gtkw = json.dumps({"mask_dict": str(root / "mask_dict.json"), "meta_expressions": str(root / "meta_expressions_valid_.json")})
        for sub in ("zero", "fail"):
            sys.argv = ["run_p1a", "--samples", str(tmp / f"runs/samples_{sub}.json"), "--video-root", str(root),
                        "--heat-dir", str(tmp / "heat"), "--pool-dir", str(tmp / "pools"), "--category-json", str(tmp / "cats.json"),
                        "--gt", "revos", "--gt-kw", gtkw, "--dms-repo", "x", "--sam3-checkpoint", "x",
                        "--modes", "rescue,point,oracle_gt", "--out", str(tmp / f"runs/run_{sub}")]
            opened.clear()
            run_p1a.main()
            assert len(opened) == 1, "one SAM3 session per video"
            recs = [json.loads(l) for l in (tmp / f"runs/run_{sub}/results.jsonl").read_text().splitlines()]
            assert len(recs) == 2
            for r in recs:
                assert r["j_old"] < 0.01 and r["triggered"]
                assert r["point"]["j_star"] > 0.99 and r["rescue"]["j_star"] > 0.99, r
            assert recs[0]["num_candidates"] == (0 if sub == "zero" else 1)
        revos_summary.main(["--per-expression", str(tmp / "per_expression.csv"), "--runs", str(tmp / "runs"),
                            "--out", str(tmp / "runs/summary.md")])
        txt = (tmp / "runs/summary.md").read_text()
        assert "| All | 5 | 20.00% | 100.00% | 100.00% | 100.00% |" in txt, txt

if __name__ == "__main__":
    test_revos_v2_path(); print("OK")
