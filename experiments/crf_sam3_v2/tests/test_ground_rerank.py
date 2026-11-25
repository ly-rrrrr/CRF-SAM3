"""CPU test of v2.ground_rerank (fake model).  python tests/test_ground_rerank.py"""
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parent / "ground_v2")]
from gv2.common import CategoryNormalizer  # noqa: E402
from v2 import ground_rerank as gr  # noqa: E402


class Fake:
    calls = 0

    def __init__(self, a):
        pass

    def __call__(self, rec, norm):
        Fake.calls += 1
        beams = {"0": ["person", "watch"], "1": ["dog", "puppy"]}[rec["exp_id"]]
        cands = gr.distinct_candidates(rec["pred_category"], beams, 2, norm)
        if len(cands) < 2:
            return rec["pred_category"], [{"word": w} for w in cands], beams
        sc = [{"word": w, "rev": -5.0 if w == "watch" else -9.0} for w in cands]
        return max(sc, key=lambda s: s["rev"])["word"], sc, beams


def test_rerank():
    n = CategoryNormalizer()
    assert gr.distinct_candidates("dog", ["dog", "puppy", "cat"], 2, n) == ["dog", "cat"]
    assert gr.distinct_candidates("person", ["man", "skier"], 3, n) == ["person"]
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "c.json"
        recs = [{"sample_id": f"s{i}", "video_id": "v", "exp_id": str(i), "expression": e, "pred_category": w,
                 "frame_paths": []} for i, (e, w) in enumerate([("a watch worn by a person", "person"), ("a dog", "dog")])]
        src.write_text(json.dumps({"summary": {}, "records": recs}))
        gr.Reranker = Fake
        out = Path(d) / "k2.json"
        args = ["--model-path", "m", "--ckpt", "c", "--categories", str(src), "--out", str(out), "--k", "2"]
        gr.main(args)
        r = json.loads(out.read_text())
        assert [x["pred_category"] for x in r["records"]] == ["watch", "dog"], r
        assert r["records"][0]["changed"] and r["records"][0]["greedy_category"] == "person"
        assert r["summary"]["changed"] == 1 and len(r["records"][1]["candidates"]) == 1
        gr.main(args)  # resume: nothing recomputed
        assert Fake.calls == 2


if __name__ == "__main__":
    test_rerank()
    print("ground_rerank test passed")
