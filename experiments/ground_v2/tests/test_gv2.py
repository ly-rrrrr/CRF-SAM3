"""CPU tests for Ground v2 with a tiny random Qwen3-VL (no weights, no GPU, no network).

  python tests/test_gv2.py

Checks the design claims in the README, not only that the code runs:
  1. labels: target > other-category (weight 3) > same-category (weight 0) > background
  2. mask loss reaches LoRA only in layers <= L; category loss does NOT reach the heatmap
     heads (detach), but does reach the summary MLPs; gradient checkpointing keeps the graph
  3. the hook adds the summaries to the two <TGT> states and only fires on the full sequence
  4. inference: heat from generate()'s prefill == heat from a plain forward
  5. build_data: negatives / ignore / absent frames from meta + mask_dict
  6. save -> load_heads round trip (+ LoRA adapter files)
"""

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pycocotools import mask as mask_utils  # noqa: E402
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration  # noqa: E402

from gv2 import build_data, train  # noqa: E402
from gv2.common import frame_labels, normalize_rle  # noqa: E402
from gv2.data import GroundV2Collator  # noqa: E402
from gv2.model import GroundV2, text_layers  # noqa: E402

IMG, VS, VE, IM_START, EOS = 151655, 151652, 151653, 151644, 151645
PATCH, MERGE, TP = 16, 2, 2
NLAYERS = 4


def tiny_qwen():
    cfg = Qwen3VLConfig(
        text_config=dict(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=NLAYERS,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                         rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [2, 3, 3]},
                         rope_scaling={"rope_type": "default", "mrope_section": [2, 3, 3]}),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, patch_size=PATCH,
                           spatial_merge_size=MERGE, temporal_patch_size=TP, out_hidden_size=64,
                           num_position_embeddings=64, deepstack_visual_indexes=[]),
        image_token_id=IMG, vision_start_token_id=VS, vision_end_token_id=VE,
    )
    torch.manual_seed(0)
    return Qwen3VLForConditionalGeneration(cfg)


class Tok:
    pad_token_id = 0
    eos_token_id = EOS

    def __init__(self):
        self.vocab = {"<|im_start|>": IM_START}

    def get_vocab(self):
        return dict(self.vocab)

    def add_special_tokens(self, d):
        for t in d["additional_special_tokens"]:
            self.vocab.setdefault(t, 151700)

    def convert_tokens_to_ids(self, t):
        return self.vocab[t]

    def __len__(self):
        return 151701

    def encode(self, s):
        out, i = [], 0
        while i < len(s):
            if s.startswith("<TGT>", i):
                out.append(self.vocab["<TGT>"])
                i += 5
            else:
                out.append(1000 + ord(s[i]) % 5000)
                i += 1
        return out

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(int(i) - 1000) for i in ids if 1000 <= int(i) < 6000)

    def save_pretrained(self, d):
        pass


class Proc:
    def __init__(self):
        self.tokenizer = Tok()

    def save_pretrained(self, d):
        pass

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors="pt"):
        from PIL import Image

        ids, pix, grids = [], [], []
        for m in messages:
            ids += [IM_START] + self.tokenizer.encode(m["role"] + "\n")
            content = m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"]}]
            for c in content:
                if c["type"] == "text":
                    ids += self.tokenizer.encode(c["text"])
                else:
                    w, h = Image.open(c["image"]).size
                    gh, gw = h // PATCH, w // PATCH
                    ids += [VS] + [IMG] * (gh * gw // MERGE ** 2) + [VE]
                    g = torch.Generator().manual_seed(gh * 1000 + gw)
                    pix.append(torch.randn(gh * gw, 3 * TP * PATCH * PATCH, generator=g))
                    grids.append([1, gh, gw])
            if m["role"] == "assistant":
                ids += [EOS]
        if add_generation_prompt:
            ids += [IM_START] + self.tokenizer.encode("assistant\n")
        t = torch.tensor([ids])
        return {"input_ids": t, "attention_mask": torch.ones_like(t), "mm_token_type_ids": (t == IMG).long(),
                "pixel_values": torch.cat(pix), "image_grid_thw": torch.tensor(grids)}


def rle(m):
    r = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
    return normalize_rle(r)


def boxes(h, w, *rects):
    m = np.zeros((h, w), bool)
    for y0, y1, x0, x1 in rects:
        m[y0:y1, x0:x1] = True
    return m


def make_record(d: Path):
    from PIL import Image

    paths = []
    for k in range(2):
        p = d / f"f{k}.jpg"
        Image.fromarray((np.random.RandomState(k).rand(64, 128, 3) * 255).astype(np.uint8)).save(p)
        paths.append(str(p))
    tgt = boxes(64, 128, (0, 32, 0, 32))
    neg = boxes(64, 128, (32, 64, 64, 96))
    ign = boxes(64, 128, (0, 32, 96, 128))
    return {"sample_id": "t:0", "expression": "a man giving a hand to a monkey", "category": "person",
            "frame_paths": paths, "mask_rles": [rle(tgt)] * 2, "mask_frame_indices": [0, 5],
            "neg_rles": [rle(neg)] * 2, "ign_rles": [rle(ign)] * 2, "absent_frames": []}


def setup(d, **over):
    args = train.parse_args(["--model-path", "x", "--train-jsonl", "x", "--output-dir", str(d), "--feature-layer", "2",
                             "--query-dim", "32", "--num-heads", "4", "--mask-dim", "16", "--gradient-checkpointing"])
    for k, v in over.items():
        setattr(args, k, v)
    proc = Proc()
    model, merge = train.build(args, proc, tiny_qwen())
    col = GroundV2Collator(proc, merge, IMG, neg_weight=3.0, absent_prob=0.0)
    return args, proc, model, col


def test_labels():
    tgt = boxes(64, 64, (0, 32, 0, 32))
    neg = boxes(64, 64, (0, 64, 16, 64))   # overlaps target -> target wins
    ign = boxes(64, 64, (32, 64, 0, 16))
    lab, w = frame_labels(rle(tgt), rle(neg), rle(ign), 4, 4, 2, 0.05, 3.0)
    lab, w = lab.view(2, 2), w.view(2, 2)
    assert lab.tolist() == [[1, 0], [0, 0]], lab
    assert w.tolist() == [[1, 3], [0, 3]], w


def lora_grads(model, layer):
    gs = [p.grad for n, p in model.named_parameters() if f"layers.{layer}." in n and "lora_B" in n]
    assert gs, f"no LoRA params found in layer {layer}"
    return sum(0.0 if g is None else float(g.abs().sum()) for g in gs)


def head_grad(mod):
    return sum(0.0 if p.grad is None else float(p.grad.abs().sum()) for p in mod.parameters())


def test_gradient_routing():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        rec = make_record(d)
        # mask loss only
        _, _, model, col = setup(d, lambda_cat=0.0, summary_drop=0.0)
        model.lambda_cat = 0.0
        model.train()
        batch = col([rec])
        assert int((batch["token_weights"] == 3).sum()) > 0 and int((batch["token_weights"] == 0).sum()) > 0
        out = model(**batch)
        assert torch.isfinite(out["loss"])
        out["loss"].backward()
        L = model.feature_layer  # hook on layers[L-1]
        assert lora_grads(model, 0) > 0 and lora_grads(model, L - 1) > 0, "mask loss must reach LoRA in layers <= L"
        for ly in range(L, NLAYERS):
            assert lora_grads(model, ly) == 0, f"mask loss leaked into layer {ly} (> L)"
        assert head_grad(model.decoder) > 0 and head_grad(model.mask_q) > 0
        # category loss only: heat heads untouched (p detached), summary MLP trained, LoRA everywhere
        _, _, model, col = setup(d, lambda_mask=0.0, summary_drop=0.0)
        model.lambda_mask = 0.0
        model.train()
        batch = col([rec])
        out = model(**{k: v for k, v in batch.items() if k not in ("token_labels", "token_weights", "token_valid")})
        out["loss"].backward()
        assert head_grad(model.mask_q) == 0 and head_grad(model.mask_v) == 0 and head_grad(model.decoder) == 0, \
            "category loss must not reach the heatmap heads"
        assert head_grad(model.summary_mean) > 0 and head_grad(model.summary_topk) > 0
        assert lora_grads(model, NLAYERS - 1) > 0


def test_hook_writes_tgt_and_generate_prefill():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        rec = make_record(d)
        _, proc, model, col = setup(d, summary_drop=0.0)
        for m in (model.summary_mean, model.summary_topk):  # make the summaries non-zero
            torch.nn.init.normal_(m.net[-1].weight, std=0.5)
        model.eval()
        from gv2.common import build_messages

        inp = proc.apply_chat_template(build_messages(rec, False), add_generation_prompt=True)
        layers = text_layers(model.qwen)
        cap = {}
        h1 = layers[model.feature_layer - 1].register_forward_hook(lambda m, a, o: cap.update(raw_out=o.clone()) if "raw_out" not in cap else None, prepend=True)
        h2 = layers[model.feature_layer].register_forward_pre_hook(lambda m, a, kw: cap.update(next_in=(a[0] if a else kw["hidden_states"]).clone()) if "next_in" not in cap else None, with_kwargs=True)
        with torch.no_grad():
            out = model(**inp)
        h1.remove(), h2.remove()
        pos = model.positions(inp["input_ids"][0])
        diff = (cap["next_in"][0] - cap["raw_out"][0]).abs().sum(-1)
        changed = torch.nonzero(diff > 1e-6).flatten().tolist()
        assert changed == pos["tgt"].tolist(), (changed, pos["tgt"])
        # generate(): hook fires once on the prefill, heat identical to the forward pass
        from gv2.predict import predict_one

        heat_fwd = torch.sigmoid(out["heat_logits"][0].float()).numpy()
        calls = {"n": 0}
        orig = model.heat_logits
        model.heat_logits = lambda *a: (calls.__setitem__("n", calls["n"] + 1), orig(*a))[1]
        pred, raw, heat, grid = predict_one(proc, model, rec, 4, 0, str(d / "cache"))
        assert calls["n"] == 1, calls
        assert np.allclose(heat.astype(np.float32), heat_fwd, atol=2e-3), np.abs(heat - heat_fwd).max()
        assert grid.tolist() == [[1, 4, 8], [1, 4, 8]]


def test_build_data():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        from PIL import Image

        (d / "img" / "v").mkdir(parents=True)
        for f in range(4):
            Image.new("RGB", (32, 32)).save(d / "img" / "v" / f"{f:05d}.jpg")
        H = W = 32
        person1 = [rle(boxes(H, W, (0, 8, 0, 8)))] * 2 + [None, None]          # target, absent in frames 2-3
        person2 = [rle(boxes(H, W, (8, 16, 8, 16)))] * 4
        monkey = [rle(boxes(H, W, (16, 32, 16, 32)))] * 4
        mask_dict = {"1": person1, "2": person2, "3": monkey}
        meta = {"videos": {"v": {"frames": [f"{f:05d}" for f in range(4)], "expressions": {
            "0": {"exp": "a man giving a hand to a monkey", "anno_id": [1]},
            "1": {"exp": "the woman on the right", "anno_id": [2]},
            "2": {"exp": "a monkey", "anno_id": [3]}}}}}
        recs = [{"video_id": "v", "exp_id": 0, "obj_id": "1", "category": "man", "expression": "a man ...",
                 "frame_paths": ["a", "b"], "mask_rles": person1[:2], "mask_frame_indices": [0, 1]},
                {"video_id": "v", "exp_id": 1, "obj_id": "2", "category": "woman", "expression": "w",
                 "frame_paths": ["a"], "mask_rles": person2[:1], "mask_frame_indices": [0]},
                {"video_id": "v", "exp_id": 2, "obj_id": "3", "category": "monkeys", "expression": "m",
                 "frame_paths": ["a"], "mask_rles": monkey[:1], "mask_frame_indices": [0]}]
        (d / "in.jsonl").write_text("\n".join(json.dumps(r) for r in recs))
        (d / "meta.json").write_text(json.dumps(meta))
        (d / "md.json").write_text(json.dumps(mask_dict))
        sys.argv = ["x", "--train-jsonl", str(d / "in.jsonl"), "--meta-expressions", str(d / "meta.json"),
                    "--mask-dict", str(d / "md.json"), "--image-root", str(d / "img"), "--out", str(d / "out.jsonl")]
        build_data.main()
        out = [json.loads(x) for x in (d / "out.jsonl").read_text().splitlines()]
        r0 = out[0]
        assert r0["target_group"] == "person"
        # woman == person group -> ignored ; monkey -> negative
        assert mask_utils.area(dict(r0["ign_rles"][0], counts=r0["ign_rles"][0]["counts"].encode())) == 64
        assert mask_utils.area(dict(r0["neg_rles"][0], counts=r0["neg_rles"][0]["counts"].encode())) == 256
        assert sorted(a["frame_index"] for a in r0["absent_frames"]) == [2, 3]
        s = json.loads((d / "summary.json").read_text())
        assert s["records_with_neg"] == 3 and s["records_with_ign"] == 2, s


def test_save_load():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _, proc, model, _ = setup(d)
        torch.nn.init.normal_(model.query_bias)
        model.save(d / "ck")
        assert (d / "ck" / "gv2_heads.pt").is_file() and (d / "ck" / "lora" / "adapter_config.json").is_file()
        hp = json.loads((d / "ck" / "gv2_config.json").read_text())["hparams"]
        other = GroundV2(tiny_qwen(), tgt_token_id=hp["tgt_token_id"], im_start_id=hp["im_start_id"],
                         feature_layer=hp["feature_layer"], query_dim=hp["query_dim"], num_heads=hp["num_heads"],
                         mask_dim=hp["mask_dim"])
        other.load_heads(d / "ck")
        assert torch.equal(other.query_bias, model.query_bias.detach())


def test_cat_loss_matches_hf():
    """Answer-only logits (logits_to_keep) give the same CE as HF's full-sequence loss."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        rec = make_record(d)
        _, _, model, col = setup(d, summary_drop=0.0)
        model.eval()
        batch = col([rec])
        with torch.no_grad():
            ours = model(**{k: v for k, v in batch.items() if k not in ("token_labels", "token_weights", "token_valid")})["loss"]
            model.set_context(batch["input_ids"])
            ref = model.qwen(**{k: v for k, v in batch.items() if k not in ("token_labels", "token_weights", "token_valid")},
                             use_cache=False).loss
            model.clear_context()
        assert torch.allclose(ours, model.lambda_cat * ref.float(), atol=1e-5), (ours, ref)


def test_eval_metrics():
    from gv2.evaluate import frame_token_masks, heat_metrics

    rec = {"frame_paths": ["a"], "mask_rles": [rle(boxes(64, 64, (0, 32, 0, 32)))],
           "neg_rles": [rle(boxes(64, 64, (32, 64, 32, 64)))]}
    grid = np.array([[1, 4, 4]])
    gts, negs = frame_token_masks(rec, grid, 2, 0.05)
    m = heat_metrics(np.array([3.0, 0.0, 0.0, 1.0]), grid, 2, gts, negs)
    assert m == {"mass_in_gt": 0.75, "mass_on_other_cat": 0.25, "peak_in_gt": 1.0, "mass_on_other_excl": 0.25,
                 "gt_area": 0.25, "other_excl_area": 0.25}, m
    # "other" object that contains the target (person vs. their pants): a perfect heatmap must not count as a leak
    rec["neg_rles"] = [rle(boxes(64, 64, (0, 64, 0, 32)))]
    gts, negs = frame_token_masks(rec, grid, 2, 0.05)
    m = heat_metrics(np.array([1.0, 0.0, 0.0, 0.0]), grid, 2, gts, negs)
    assert m["mass_on_other_cat"] == 1.0 and m["mass_on_other_excl"] == 0.0 and m["other_excl_area"] == 0.25, m


def test_trainer_smoke():
    """Two optimiser steps through the HF Trainer, then the checkpoint layout."""
    from transformers import TrainingArguments

    from gv2.data import JsonlDataset

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        rec = make_record(d)
        (d / "tr.jsonl").write_text(json.dumps(rec) + "\n" + json.dumps(rec) + "\n")
        _, proc, model, col = setup(d)
        before = model.query_bias.detach().clone()
        targs = TrainingArguments(output_dir=str(d / "run"), per_device_train_batch_size=1, max_steps=2,
                                  learning_rate=1e-2, logging_steps=1, save_strategy="no", report_to="none",
                                  remove_unused_columns=False, use_cpu=True, prediction_loss_only=True)
        tr = train.GV2Trainer(model=model, args=targs, train_dataset=JsonlDataset(str(d / "tr.jsonl")),
                              data_collator=col, processing_class=proc)
        tr.train()
        tr._save(str(d / "run"))
        assert not torch.equal(before, model.query_bias.detach()), "heads did not train"
        assert (d / "run" / "gv2_heads.pt").is_file() and (d / "run" / "lora" / "adapter_config.json").is_file()


def test_logging_and_nan_guard():
    """Logged rows carry averaged loss parts + step and gv2.plot_log draws them; a non-finite
    loss stops training after that step and cancels its checkpoint."""
    import contextlib
    import io

    from transformers import TrainingArguments

    from gv2 import plot_log
    from gv2.data import JsonlDataset

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        rec = make_record(d)
        (d / "tr.jsonl").write_text((json.dumps(rec) + "\n") * 4)
        _, proc, model, col = setup(d)
        targs = TrainingArguments(output_dir=str(d / "run"), per_device_train_batch_size=1, max_steps=4,
                                  learning_rate=1e-3, logging_steps=1, save_strategy="steps", save_steps=1,
                                  report_to="none", remove_unused_columns=False, use_cpu=True,
                                  prediction_loss_only=True)
        tr = train.GV2Trainer(model=model, args=targs, train_dataset=JsonlDataset(str(d / "tr.jsonl")),
                              data_collator=col, processing_class=proc)
        tr.add_callback(train.NanGuard(tr))
        real_forward, calls = model.forward, {"n": 0}

        def forward(*a, **k):
            out = real_forward(*a, **k)
            calls["n"] += 1
            if calls["n"] == 2:                      # second optimiser step blows up
                out["loss"] = out["loss"] * float("nan")
            return out

        model.forward = forward
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            tr.train()
        log = buf.getvalue()
        assert tr.nonfinite and tr.nonfinite_step == 2, (tr.nonfinite, tr.nonfinite_step)
        assert tr.state.global_step == 2, f"should stop right after the bad step, got {tr.state.global_step}"
        assert (d / "run" / "checkpoint-1").is_dir() and not (d / "run" / "checkpoint-2").exists(), \
            "the step that saw NaN must not be checkpointed"
        rows = [json.loads(x[6:]) for x in log.splitlines() if x.startswith("[gv2] {")]
        first = [r for r in rows if "loss" in r and r["step"] == 1][0]
        assert {"loss_mask", "loss_cat", "loss_bce", "loss_dice"} <= set(first), first
        (d / "run.log").write_text(log)
        plot_log.main([str(d / "run.log"), "--out", str(d / "c.png")])
        assert (d / "c.png").stat().st_size > 10_000


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: OK", flush=True)
