"""Ground v2 选词 + 反向打分重排（noisy-channel），固定 K。

对每条表达式：
  1. Ground v2 用 beam search 给出前 K 个词（K 固定，默认 2）；第 1 个候选始终是原来贪心解码的词
     （读 gv2.predict 写好的 categories.json），保证没有分歧时结果和原来完全一样。
  2. 同一类别的不同叫法（dog / puppy、ape / gorilla，按 gv2.common.CategoryNormalizer 归组）只留第一个，
     重排只在「指不同物体」的候选之间进行。只剩 1 个候选就直接用它，不多算。
  3. 每个候选 c 算 log p(表达式 | 同样 3 帧, 目标是 c)：teacher forcing 一次前向，不生成文字，
     用的是 Ground v2 自己的权重（带 LoRA）。选分数最高的词。

输出和 gv2.predict 同格式的 categories json（records[].pred_category 换成重排后的词），另加
greedy_category / candidates / changed 字段，主线直接读。热力图不受影响（它不依赖词）。

  cd ground_v2 的上级目录下运行，PYTHONPATH 需包含 ground_v2/：
  python -m v2.ground_rerank --model-path $MODEL --ckpt ../ground_v2/runs/gv2_revos \
      --categories runs/refytvos_gv2revos/ground/categories.json --out runs/x/ground/categories_k2.json --k 2
"""
import argparse
import json
import sys
import time
from pathlib import Path

REV_SYSTEM = "You write referring expressions for objects in videos."
REV_USER = ("The frames are from the same video. The target object is: {word}.\n"
            "Write one referring expression that uniquely describes this target object.")


YN_SYSTEM = "You check which object a referring expression refers to."
YN_USER = ("Referring expression:\n{expr}\n\nThe frames are from the same video.\n"
           "Question: Is the {word} the object that this expression refers to (its referent), "
           "not just something mentioned in it? Answer Yes or No.")


def distinct_candidates(greedy, beams, k, norm):
    """greedy 词放第一位；按类别组去重，最多 k 个。"""
    out, groups = [], set()
    for w in [greedy] + list(beams):
        w = str(w or "").strip()
        g = norm(w)
        if not w or g in groups:
            continue
        out.append(w)
        groups.add(g)
        if len(out) >= k:
            break
    return out


def rev_messages(frames, word, expression):
    content = [{"type": "image", "image": p} for p in frames]
    content.append({"type": "text", "text": REV_USER.format(word=word)})
    msgs = [{"role": "system", "content": [{"type": "text", "text": REV_SYSTEM}]},
            {"role": "user", "content": content}]
    if expression is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": expression}]})
    return msgs


class Reranker:
    def __init__(self, a):
        import torch
        from gv2.model import load_for_inference
        self.torch = torch
        self.proc, self.model = load_for_inference(a.model_path, a.ckpt, torch.bfloat16, "", "cuda")
        self.k, self.max_side, self.cache, self.mode = a.k, a.max_image_side, a.frame_cache, a.score
        tok = self.proc.tokenizer
        self.yn_ids = (tok("Yes", add_special_tokens=False)["input_ids"][0], tok("No", add_special_tokens=False)["input_ids"][0])
        self.dev = self.model.qwen.get_input_embeddings().weight.device

    def _to(self, inp):
        return {k: (v.to(self.dev) if self.torch.is_tensor(v) else v) for k, v in inp.items()}

    def beams(self, rec):
        """Ground v2 原样的提示和 <TGT> 摘要，beam search 返回 K 个词。"""
        from gv2.common import build_messages
        from gv2.predict import parse_category
        inp = self._to(self.proc.apply_chat_template(build_messages(rec, False), tokenize=True,
                                                     add_generation_prompt=True, return_dict=True,
                                                     return_tensors="pt"))
        n = inp["input_ids"].shape[1]
        # beam search 会把输入扩成 K 份再做 prefill；上下文也给 K 份，保证每个 beam 都加上 <TGT> 摘要
        self.model.set_context(inp["input_ids"].repeat(self.k, 1))
        try:
            with self.torch.inference_mode():
                gen = self.model.qwen.generate(**inp, max_new_tokens=32, do_sample=False, num_beams=self.k,
                                               num_return_sequences=self.k, early_stopping=True)
        finally:
            self.model.clear_context()
        return [parse_category(self.proc.tokenizer.decode(g[n:], skip_special_tokens=True)) for g in gen]

    def rev_logp(self, frames, word, expression):
        torch = self.torch
        p = self.proc.apply_chat_template(rev_messages(frames, word, None), tokenize=True,
                                          add_generation_prompt=True, return_dict=True, return_tensors="pt")
        full = self.proc.apply_chat_template(rev_messages(frames, word, expression), tokenize=True,
                                             add_generation_prompt=False, return_dict=True, return_tensors="pt")
        n0 = p["input_ids"].shape[1]
        ids = full["input_ids"][0]
        n = len(self.proc.tokenizer(expression, add_special_tokens=False)["input_ids"])
        with torch.inference_mode():
            logits = self.model.qwen(**self._to(full), logits_to_keep=ids.shape[0] - n0 + 1).logits[0]
            off = ids.shape[0] - logits.shape[0]
            lp = torch.log_softmax(logits.float(), -1)
            pos = torch.arange(n0 - 1, n0 - 1 + n, device=lp.device) - off
            return float(lp[pos, ids[n0:n0 + n].to(lp.device)].sum())

    def yes_no(self, frames, word, expression):
        """直接问：这个词是不是表达式的指代对象？返回 log p(Yes) - log p(No)。"""
        torch = self.torch
        content = [{"type": "image", "image": p} for p in frames]
        content.append({"type": "text", "text": YN_USER.format(expr=expression, word=word)})
        msgs = [{"role": "system", "content": [{"type": "text", "text": YN_SYSTEM}]}, {"role": "user", "content": content}]
        inp = self._to(self.proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                                     return_dict=True, return_tensors="pt"))
        with torch.inference_mode():
            lp = torch.log_softmax(self.model.qwen(**inp, logits_to_keep=1).logits[0, -1].float(), -1)
        return float(lp[self.yn_ids[0]] - lp[self.yn_ids[1]])

    def __call__(self, rec, norm):
        from gv2.data import resize_cached
        r = dict(rec)
        r["frame_paths"] = [resize_cached(p, self.max_side, self.cache) for p in rec["frame_paths"]]
        greedy = rec.get("pred_category", "")
        try:
            bw = self.beams(r)
        except Exception as e:  # beam search 出问题时只用贪心词，不影响主线
            print(f"[rerank] beam 失败 {rec.get('sample_id')}: {type(e).__name__}: {e}", flush=True)
            bw = []
        cands = distinct_candidates(greedy, bw, self.k, norm)
        if len(cands) < 2:
            return greedy, [{"word": w} for w in cands], bw
        fn = self.yes_no if self.mode == "yesno" else self.rev_logp
        scores = [{"word": w, self.mode: fn(r["frame_paths"], w, rec["expression"])} for w in cands]
        best = max(scores, key=lambda s: s[self.mode])["word"]
        return best, scores, bw


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--categories", required=True, help="gv2.predict 写的 categories.json（贪心词 + 帧路径）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--score", choices=["yesno", "rev"], default="yesno",
                    help="yesno：直接问是不是指代对象；rev：反向打分 log p(表达式 | 词)")
    ap.add_argument("--max-image-side", type=int, default=0, help="和 gv2.predict 一致（0 = 原分辨率）")
    ap.add_argument("--frame-cache", default="runs/_frame_cache")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=25)
    a = ap.parse_args(argv)
    from gv2.common import CategoryNormalizer

    norm = CategoryNormalizer()
    src = json.loads(Path(a.categories).read_text(encoding="utf-8"))
    recs = src["records"][: a.limit] if a.limit else src["records"]
    out_p = Path(a.out)
    done = {}
    if out_p.is_file():  # 断点续跑
        done = {r["sample_id"]: r for r in json.loads(out_p.read_text(encoding="utf-8"))["records"] if "candidates" in r}
    rr = None
    out, t0, changed = [], time.time(), 0
    for n, rec in enumerate(recs, 1):
        if rec["sample_id"] in done:
            out.append(done[rec["sample_id"]])
        else:
            rr = rr or Reranker(a)
            best, cands, bw = rr(rec, norm)
            o = dict(rec)
            o.update(greedy_category=rec.get("pred_category", ""), pred_category=best, candidates=cands,
                     beam_words=bw, changed=best != rec.get("pred_category", ""))
            out.append(o)
        changed += bool(out[-1].get("changed"))
        if n % a.save_every == 0 or n == len(recs):
            payload = {"summary": dict(src.get("summary", {}), rerank_k=a.k, rerank=a.score,
                                       total=len(out), changed=changed), "records": out}
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[rerank {n}/{len(recs)}] 改词 {changed} 条，{(time.time() - t0) / n:.2f}s/条", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
