"""CPU test of tools/inventory.py on a fake server tree.  python tests/test_inventory.py"""
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "tools"))
import inventory  # noqa: E402


def w(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj) if not isinstance(obj, str) else obj)


def test_inventory():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        vd, repo, gv2 = d / "vd", d / "repo", d / "ground_v2"
        w(vd / "Ref-Youtube-VOS" / "train" / "meta_expressions.json", {"videos": {
            "a": {"frames": ["0", "5"], "expressions": {"0": {"exp": "x", "obj_id": "1"}, "1": {"exp": "y", "obj_id": "2"}}},
            "b": {"frames": ["0"], "expressions": {"0": {"exp": "z", "obj_id": "1"}}}}})
        w(vd / "Ref-Youtube-VOS" / "train" / "mask_dict.pkl", "xx")
        w(vd / "mevis" / "train" / "meta_expressions.json", {"videos": {
            "m": {"frames": ["0"], "expressions": {"0": {"exp": "two", "obj_id": [1, 2], "anno_id": [3, 4]}}}}})
        w(vd / "revos" / "meta_expressions_train_.json", {"videos": {
            "DS/train/v": {"expressions": {"0": {"exp": "q", "obj_id": [1]}, "1": {"exp": "r", "obj_id": [1]}}}}})
        w(repo / inventory.REVOS_API_CATS, {"records": [{"video_id": "DS/train/v", "exp_id": 0, "category": "cat"}]})
        w(repo / "category_grounding_mvp/data/refytvos_query_grounding/summary.json",
          {"train_records": 2, "val_records": 1, "train_videos": 1, "val_videos": 1})
        w(repo / "category_grounding_mvp/data/refytvos_query_grounding/train.jsonl", "{}\n{}\n")
        c = repo / "crf_sam3_attn/offline_cache_revos_api_categories_recall24_keep7_20260717/train"
        w(c / "DS/train/v/0/meta.json", {"iou_map": {"1": 0.8, "2": 0.1}, "frame_indices": [0, 1, 2, 3]})
        w(c / "DS/train/v/1/meta.json", {"iou_map": {"1": 0.2}, "frame_indices": [0, 1, 2, 3]})
        w(repo / "crf_sam3_attn/cache_reports/revos_train_api_categories_recall24_keep7_max_j_lt_0p5_20260718.json",
          {"summary": {"num_samples": 1}, "samples": []})
        (repo / "crf_sam3_attn_outputs_joint_train_poswin/final").mkdir(parents=True)
        (repo / "crf_sam3_attn_outputs_joint_train_poswin/checkpoint_1000").mkdir(parents=True)
        (repo / "category_grounding_mvp/outputs/g1/checkpoint-900").mkdir(parents=True)
        w(gv2 / "runs/gv2_revos/gv2_config.json", {})
        w(gv2 / "runs/gv2_revos/eval_val.json", {"word_acc": 0.835})
        w(gv2 / "data/refytvos_v2/summary_train.json", {"records": 2, "target_png_iou_mean": 1.0})
        rep = d / "inv.md"
        inventory.LINES.clear()
        assert inventory.main(["--repo", str(repo), "--video-datas", str(vd), "--revos-root", str(vd / "revos"),
                               "--gv2", str(gv2), "--p1a", str(d / "none"), "--out", str(rep)]) == 0
        t = rep.read_text()
        assert "这一节出错" not in t, t
        assert "| Ref-YTVOS | train | `train/meta_expressions.json` | 2 | 3 | 1.5（2） | 0 / 3 / 0 | 2 B |" in t, t
        assert "| MeViS | train |" in t and "0 / 0 / 1" in t
        assert "覆盖 `revos` train 表达式 1 / 2（50.0%）" in t, t
        assert "| revos | `offline_cache_revos_api_categories_recall24_keep7_20260717` | train | 2 | 1 | 1.5 | 50.0% | 0.5 | 1 |" in t, t
        assert "jsonl 实际行数 {'train': 2}" in t
        assert "`crf_sam3_attn_outputs_joint_train_poswin`：final，1 个 checkpoint，最后 checkpoint_1000" in t
        assert "`g1`：checkpoint-900" in t and "word_acc=0.835" in t and "summary_train.json` | 2 |" in t
        assert "缺失" in t  # davis 缓存等不存在


if __name__ == "__main__":
    test_inventory()
    print("inventory test passed")
