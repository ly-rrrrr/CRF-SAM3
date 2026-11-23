"""CPU tests with a tiny random Qwen3-VL (no weights, no network).

1. attention captured through the RoPE hook == eager attention weights
2. end-to-end run_probe with a fake processor and a fixed answer: word-token
   rows, image-token columns, frame layout, metrics, figures, summary.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration  # noqa: E402

from probe import attn_capture, run_probe  # noqa: E402

IMG, VS, VE = 151655, 151652, 151653
PATCH, MERGE, TP = 16, 2, 2


def tiny_model():
    cfg = Qwen3VLConfig(
        text_config=dict(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                         rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [2, 3, 3]}),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, patch_size=PATCH,
                           spatial_merge_size=MERGE, temporal_patch_size=TP, out_hidden_size=64,
                           num_position_embeddings=64, deepstack_visual_indexes=[]),
        image_token_id=IMG, vision_start_token_id=VS, vision_end_token_id=VE,
    )
    torch.manual_seed(0)
    return Qwen3VLForConditionalGeneration(cfg).eval()


class Tok:
    eos_token_id = 1
    pad_token_id = 0

    def encode(self, s):
        return [1000 + ord(c) for c in s]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i - 1000) for i in ids if i >= 1000)


class FakeImageProcessor:
    merge_size = MERGE


class FakeProcessor:
    tokenizer = Tok()
    image_processor = FakeImageProcessor()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        parts = []
        for m in messages:
            for c in m["content"]:
                parts.append("<IMG>" if c["type"] == "image" else c["text"])
        return "|".join(parts)

    def __call__(self, text, images, return_tensors="pt"):
        segs = text[0].split("<IMG>")
        ids, pix, grids = [], [], []
        for k, seg in enumerate(segs):
            ids += self.tokenizer.encode(seg)
            if k < len(images):
                w, h = images[k].size
                gh, gw = h // PATCH, w // PATCH
                n = gh * gw // (MERGE * MERGE)
                ids += [VS] + [IMG] * n + [VE]
                pix.append(torch.randn(gh * gw, 3 * TP * PATCH * PATCH))
                grids.append([1, gh, gw])
        ids = torch.tensor([ids])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "mm_token_type_ids": (ids == IMG).long(),
                "pixel_values": torch.cat(pix), "image_grid_thw": torch.tensor(grids)}


def test_capture_matches_eager():
    model = tiny_model()
    model.config._attn_implementation = "eager"
    for m in model.modules():
        if hasattr(m, "config") and hasattr(m.config, "_attn_implementation"):
            m.config._attn_implementation = "eager"
    from PIL import Image
    proc = FakeProcessor()
    imgs = [Image.new("RGB", (64, 32)), Image.new("RGB", (32, 64))]
    inp = proc(["hello<IMG>x<IMG>which one"], imgs)
    L = inp["input_ids"].shape[1]
    cols = torch.nonzero(inp["input_ids"][0] == IMG).flatten()
    rows = [L - 3, L - 1]
    got = attn_capture.layer_maps(model, inp, rows, cols)
    with torch.inference_mode():
        ref = model(**inp, output_attentions=True, use_cache=False).attentions
    assert got.shape[0] == len(ref) == 3, (got.shape, len(ref))
    for ly, a in enumerate(ref):
        exp = a[0][:, rows][:, :, cols].mean(0)
        assert torch.allclose(got[ly], exp.float(), atol=1e-5), (ly, (got[ly] - exp).abs().max())


def test_end_to_end():
    from PIL import Image
    from pycocotools import mask as mask_utils

    model = tiny_model()
    answer = '{"category": "deer"}'
    ans_ids = Tok().encode(answer)

    def fake_generate(**kw):
        return torch.cat([kw["input_ids"], torch.tensor([ans_ids + [1]])], dim=1)

    model.generate = fake_generate
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        frames = []
        for k in range(2):
            p = d / f"__tmp__f{k}.jpg"
            Image.fromarray((np.random.rand(64, 128, 3) * 255).astype(np.uint8)).save(p)
            frames.append(str(p))
        gt = np.zeros((64, 128), np.uint8)
        gt[16:48, 64:128] = 1
        rle = mask_utils.encode(np.asfortranarray(gt))
        rle["counts"] = rle["counts"].decode()
        rec = {"sample_id": "t:0", "expression": "a small brown deer", "frame_paths": frames,
               "mask_rles": [rle, rle], "pred_category": "zebra"}
        (d / "in.jsonl").write_text(json.dumps(rec) + "\n")
        # Ground v1 npz in the failure-package format
        np.savez(d / "case_00.npz", aggregate=np.random.rand(2 * 2 * 4).astype(np.float32),
                 gt_token_mask=np.tile(np.array([0, 0, 1, 1, 0, 0, 1, 1], bool), 2),
                 image_grid_thw=np.array([[1, 4, 8], [1, 4, 8]]))

        import transformers
        transformers.AutoProcessor.from_pretrained = staticmethod(lambda *a, **k: FakeProcessor())
        Qwen3VLForConditionalGeneration.from_pretrained = staticmethod(lambda *a, **k: model)
        sys.argv = ["run_probe", "--model-path", "x", "--input-jsonl", str(d / "in.jsonl"), "--device", "cpu",
                    "--dtype", "float32", "--ground-npz-dir", str(d), "--out", str(d / "out")]
        run_probe.main()
        r = json.loads((d / "out" / "results.jsonl").read_text().splitlines()[0])
        assert r["zero_shot_word"] == "deer"
        assert r["grid"] == [[2, 4], [2, 4]], r["grid"]
        assert "Ground v1 MAX (336)" in r["metrics"]
        assert "word/generic·middle layers" in r["metrics"]
        assert len(r["per_layer_peak_hit"]) == 3
        assert (d / "out" / "fig" / "00.png").is_file()
        assert "frozen zero-shot" in (d / "out" / "summary.md").read_text()
        assert "word·all layers, sinks removed" in r["metrics"]
        # words-only mode -> categories.json usable by p1a run_p1a --category-json
        sys.argv = sys.argv[:-1] + [str(d / "out_w"), "--words-only"]
        run_probe.main()
        cats = json.loads((d / "out_w" / "categories.json").read_text())["records"]
        assert cats[0]["pred_category"] == "deer" and cats[0]["ground_v1_category"] == "zebra"
        # word rows must be exactly the 4 tokens of "deer"
        idx, w = run_probe.word_token_offsets(Tok(), ans_ids, answer)
        assert w == "deer" and len(idx) == 4 and answer[idx[0]: idx[-1] + 1] == "deer"


if __name__ == "__main__":
    test_capture_matches_eager()
    print("capture == eager: OK")
    test_end_to_end()
    print("end-to-end: OK")
