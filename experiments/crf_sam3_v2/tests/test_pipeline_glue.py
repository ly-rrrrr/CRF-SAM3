"""CPU test of the glue between stages: tools/ground_infer_plan.py and tools/make_verify_config.py.
    python tests/test_pipeline_glue.py
"""
import json
import sys
import tempfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE / "tools"), str(HERE / "tests")]
import test_stage_data as tsd  # noqa: E402
import make_splits  # noqa: E402
import ground_infer_plan as gip  # noqa: E402
import make_verify_config as mvc  # noqa: E402
from v2 import build_ground  # noqa: E402


def test_plan_and_config():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        vd = d / "vd"
        cats = tsd.build(vd)
        sp = d / "splits.json"
        assert make_splits.main(["--video-datas", str(vd), "--repo", str(d / "repo"), "--out", str(sp)]) == 0
        g = d / "ground"
        assert build_ground.main(["--video-datas", str(vd), "--revos-categories", str(cats), "--gv2",
                                  str(HERE.parent / "ground_v2"), "--splits", str(sp), "--out", str(g)]) == 0
        out = d / "gi"
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert gip.main(["plan", "--video-datas", str(vd), "--splits", str(sp), "--members", str(g / "members.json"),
                             "--out", str(out), "--parts", "2"]) == 0
        jobs = [l.split() for l in buf.getvalue().splitlines() if l.strip()]
        s = json.loads(sp.read_text())["datasets"]
        mem = json.loads((g / "members.json").read_text())
        seen = {}
        for name, model, part, root, meta, outj, heat, n in jobs:
            for vid in json.loads(Path(meta).read_text())["videos"]:
                assert vid not in seen.get(name, {}), (name, vid)
                seen.setdefault(name, {})[vid] = model
        for name in ("mevis", "davis17", "refytvos", "revos"):
            assert set(seen[name]) == set(s[name]["train"]) | set(s[name]["holdout"]), name
            for vid, model in seen[name].items():
                f = mem.get(name, {}).get(vid)
                assert model == ("full" if f is None else f"cf{f}"), (name, vid, model, f)
        for v in s["refytvos"]["holdout"]:
            assert seen["refytvos"][v] == "full"
        assert {"cf0", "cf1", "full"} <= set(seen["refytvos"].values())
        # merge: fake gv2.predict outputs
        for name, model, part, root, meta, outj, heat, n in jobs:
            vids = json.loads(Path(meta).read_text())["videos"]
            recs = [{"video_id": v, "exp_id": e, "expression": x["exp"], "pred_category": "cat"}
                    for v, dv in vids.items() for e, x in dv["expressions"].items()]
            Path(outj).write_text(json.dumps({"records": recs}))
            import numpy as np
            for rec in recs:
                hp=Path(heat)/rec["video_id"]/f"{rec['exp_id']}.npz"
                hp.parent.mkdir(parents=True,exist_ok=True)
                np.savez(hp,responses=np.ones(1),image_grid_thw=np.ones((1,3)),frame_stems=np.array(["00000"]),merge=2)
        assert gip.main(["merge", "--out", str(out)]) == 0
        assert (out / "refytvos" / "cf0" / "categories.json").is_file()

        # make_verify_config: fake caches + repo recipe
        repo = d / "repo"
        (repo / "crf_sam3_attn").mkdir(parents=True, exist_ok=True)
        (repo / mvc.RECIPE).write_text(yaml.safe_dump({
            "model": {"listwise_force_pos": True, "lora": {"r": 16}, "loss_type": "feature"},
            "train": {"lr": 1e-5, "epochs": 1, "grad_accum": 1, "output_dir": "x"}}))
        cache = d / "cache"
        counts = {"mevis": (4, 3), "davis17": (2, 2), "refytvos": (6, 1), "revos": (2, 10)}  # (videos, exprs per video)
        for ds, (nv, ne) in counts.items():
            for v in range(nv):
                for e in range(ne):
                    p = cache / ds / "train_root" / "train" / (f"OVIS/train/v{v}" if ds == "revos" else f"v{v}") / str(e) / "meta.json"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(json.dumps({"video_id": f"v{v}", "obj_ids": list(range(9 if ds == "revos" else 7))}))
        cfg_path = d / "train.yaml"
        assert mvc.main(["--repo", str(repo), "--cache", str(cache), "--out", str(cfg_path)]) == 0
        cfg = yaml.safe_load(cfg_path.read_text())
        assert cfg["model"]["listwise_force_pos"] is False
        assert cfg["data"]["num_frames"] == 4 and cfg["data"]["listwise_max_candidates"] >= 9
        rep = {x["name"]: x["repeats"] for x in cfg["data"]["datasets"]}
        assert min(rep.values()) == 1 and rep["refytros"] > rep["revos"], rep
        assert all(x["pos_iou"] == 0.5 and x["neg_iou"] == 0.1 for x in cfg["data"]["datasets"])


if __name__ == "__main__":
    test_plan_and_config()
    print("pipeline glue test passed")
