"""反向打分小验证（noisy-channel）：对每条表达式的候选词 c，算 log p(表达式 | 画面, 目标=c)，看能不能选对主语。

不生成文字，只做 teacher-forcing 前向，读表达式 token 的对数概率。画面和 Ground 一样取 3 帧均匀帧、原分辨率。
候选词：v1 的词（论文主线 scores.json 的 used_category_query）和 v2 的词（新主线 scores.jsonl 的 category）。
两个词不同的表达式才需要选；相同的直接跳过。

同一个模型打两遍分：
  base : 原版 Qwen3-VL-4B（关掉 LoRA）
  gv2  : 加上 Ground v2 的 LoRA（--gckpt 给了才有）
另外算一个正向先验 log p(词 | 画面, 表达式)，用来试 p(c|e) ∝ p(e|c)·p(c)。

  python tools/noisy_channel.py --model-path $MODEL --gckpt ../ground_v2/runs/gv2_revos \
      --data-root $REFYT/valid/JPEGImages --meta $REFYT/valid/meta_expressions_challenge.json \
      --v1-scores $REPO/crf_sam3_refytros_valid_query_grounding_mainline_20260701/scores.json \
      --v2-scores runs/refytvos_gv2revos/scores.jsonl \
      --jf-v1 $REPO/crf_sam3_refytros_valid_query_grounding_mainline_20260701/evaluation_results_refytros_png.json \
      --jf-v2 runs/refytvos_gv2revos/words/evaluation_results_refytros_png.json \
      --out runs/noisy_channel

  只重算报告（不用 GPU）：加 --report-only
"""
import argparse
import json
import math
import statistics as st
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
FWD_SYSTEM = "You identify the target object of a referring expression in a video."
FWD_USER = ("Referring expression:\n{expr}\n\nThe frames are from the same video. "
            "What is the category of the target object? Answer with the category name only.")


def sample_idx(n, k=3):
    if n <= k:
        return list(range(n))
    out = []
    for i in range(k):
        j = round(i * (n - 1) / max(k - 1, 1))
        if j not in out:
            out.append(j)
    return out


def norm(w):
    return " ".join(str(w or "").strip().lower().split())


def load_words(a):
    v1 = {}
    for r in json.loads(Path(a.v1_scores).read_text(encoding="utf-8")):
        w = r.get("used_category_query") or r.get("category")
        if w:
            v1[f"{r['video_id']}_{r['exp_id']}"] = str(w).strip()
    v2 = {}
    with open(a.v2_scores, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            v2[f"{r['video_id']}_{r['exp_id']}"] = str(r["category"]).strip()
    return v1, v2


def build_items(a, v1, v2):
    videos = json.loads(Path(a.meta).read_text(encoding="utf-8"))["videos"]
    items = []
    for vid in sorted(videos):
        v = videos[vid]
        frames = [str(x) for x in v.get("frames", [])]
        paths = [str(Path(a.data_root) / vid / f"{frames[i]}.jpg") for i in sample_idx(len(frames))]
        for eid, e in v.get("expressions", {}).items():
            k = f"{vid}_{eid}"
            if k not in v1 or k not in v2:
                continue
            words = []
            for w in (v1[k], v2[k]):
                if norm(w) not in [norm(x) for x in words]:
                    words.append(w)
            if len(words) < 2 and not a.all:
                continue
            items.append({"key": k, "expression": str(e.get("exp", "")).strip(), "frames": paths,
                          "v1": v1[k], "v2": v2[k], "words": words})
    return items


class Scorer:
    def __init__(self, a):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.torch = torch
        src = a.gckpt if a.gckpt else a.model_path
        self.proc = AutoProcessor.from_pretrained(src, trust_remote_code=True)
        m = Qwen3VLForConditionalGeneration.from_pretrained(a.model_path, dtype=torch.bfloat16, device_map=None).to("cuda")
        if len(self.proc.tokenizer) > m.get_input_embeddings().weight.shape[0]:
            m.resize_token_embeddings(len(self.proc.tokenizer))
        self.has_lora = False
        if a.gckpt and (Path(a.gckpt) / "lora").is_dir():
            from peft import PeftModel
            m = PeftModel.from_pretrained(m, str(Path(a.gckpt) / "lora"))
            self.has_lora = True
        self.m = m.eval()

    def _msgs(self, system, frames, user, answer):
        content = [{"type": "image", "image": p} for p in frames] + [{"type": "text", "text": user}]
        msgs = [{"role": "system", "content": [{"type": "text", "text": system}]},
                {"role": "user", "content": content}]
        if answer is not None:
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
        return msgs

    def logp(self, system, frames, user, answer):
        """sum and mean log p(answer tokens | prompt); answer ends before <|im_end|>."""
        torch = self.torch
        p = self.proc.apply_chat_template(self._msgs(system, frames, user, None), tokenize=True,
                                          add_generation_prompt=True, return_dict=True, return_tensors="pt")
        full = self.proc.apply_chat_template(self._msgs(system, frames, user, answer), tokenize=True,
                                             add_generation_prompt=False, return_dict=True, return_tensors="pt")
        n0 = p["input_ids"].shape[1]
        ids = full["input_ids"][0]
        ans = self.proc.tokenizer(answer, add_special_tokens=False)["input_ids"]
        if ids[n0:n0 + len(ans)].tolist() != ans:  # 模板差异时退回到按长度对齐
            ans = ids[n0:n0 + len(ans)].tolist()
        n = len(ans)
        inp = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in full.items()}
        with torch.inference_mode():
            try:
                logits = self.m(**inp, logits_to_keep=ids.shape[0] - n0 + 1).logits[0]
                off = ids.shape[0] - logits.shape[0]
            except TypeError:
                logits = self.m(**inp).logits[0]
                off = 0
            lp = torch.log_softmax(logits.float(), -1)
            tgt = ids[n0:n0 + n].to("cuda")
            pos = torch.arange(n0 - 1, n0 - 1 + n, device="cuda") - off
            vals = lp[pos, tgt]
        s = float(vals.sum())
        return s, s / max(1, n)

    def yes_no(self, frames, expr, word):
        """log p(Yes) - log p(No) for the first answer token."""
        torch = self.torch
        if not hasattr(self, "_yn_ids"):
            tok = self.proc.tokenizer
            self._yn_ids = (tok("Yes", add_special_tokens=False)["input_ids"][0],
                            tok("No", add_special_tokens=False)["input_ids"][0])
        p = self.proc.apply_chat_template(self._msgs(YN_SYSTEM, frames, YN_USER.format(expr=expr, word=word), None),
                                          tokenize=True, add_generation_prompt=True, return_dict=True,
                                          return_tensors="pt")
        inp = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in p.items()}
        with torch.inference_mode():
            try:
                lg = self.m(**inp, logits_to_keep=1).logits[0, -1]
            except TypeError:
                lg = self.m(**inp).logits[0, -1]
            lp = torch.log_softmax(lg.float(), -1)
        y, n = self._yn_ids
        return float(lp[y] - lp[n])

    def score(self, it):
        out = {}
        modes = [("gv2", True), ("base", False)] if self.has_lora else [("base", False)]
        for name, use_lora in modes:
            ctx = self.m.disable_adapter() if (self.has_lora and not use_lora) else _null()
            with ctx:
                for w in it["words"]:
                    rs, rm = self.logp(REV_SYSTEM, it["frames"], REV_USER.format(word=w), it["expression"])
                    fs, fm = self.logp(FWD_SYSTEM, it["frames"], FWD_USER.format(expr=it["expression"]), w)
                    yn = self.yes_no(it["frames"], it["expression"], w)
                    out.setdefault(name, {})[w] = {"rev": rs, "rev_mean": rm, "fwd": fs, "fwd_mean": fm, "yn": yn}
        return out


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def report(a):
    rows = [json.loads(l) for l in open(Path(a.out) / "scores.jsonl", encoding="utf-8")]
    jf1 = json.loads(Path(a.jf_v1).read_text(encoding="utf-8"))["per_expression"] if a.jf_v1 else {}
    jf2 = json.loads(Path(a.jf_v2).read_text(encoding="utf-8"))["per_expression"] if a.jf_v2 else {}
    rows = [r for r in rows if len(r["words"]) == 2 and r["key"] in jf1 and r["key"] in jf2]
    L = ["# 反向打分验证（noisy-channel）", "",
         f"候选词两个不同、两边都有 J&F 的表达式：{len(rows)} 条。v1 一列的 J&F 来自 `{a.jf_v1}`，"
         f"v2 一列来自 `{a.jf_v2}`。", ""]
    if not rows:
        Path(a.out, "report.md").write_text("\n".join(L + ["没有可比较的样本。"]), encoding="utf-8")
        return
    m = lambda xs: 100 * st.mean(xs)
    J = lambda r, w: (jf1 if norm(w) == norm(r["v1"]) else jf2)[r["key"]]["J&F"]
    L += ["| 选法 | J&F |", "|---|---|",
          f"| 总用 v1 的词 | {m([jf1[r['key']]['J&F'] for r in rows]):.2f} |",
          f"| 总用 v2 的词 | {m([jf2[r['key']]['J&F'] for r in rows]):.2f} |"]
    modes = sorted({k for r in rows for k in r["s"]})
    crit = {"直接问是不是指代对象（Yes−No）": lambda d: d.get("yn", float("-inf")),
            "反向（句子总概率）": lambda d: d["rev"], "反向（每 token 平均）": lambda d: d["rev_mean"],
            "正向（只看词）": lambda d: d["fwd"], "反向 + 正向": lambda d: d["rev"] + d["fwd"]}
    best = {}
    for md in modes:
        for cn, f in crit.items():
            picks = [max(r["words"], key=lambda w: f(r["s"][md][w])) for r in rows]
            v = m([J(r, w) for r, w in zip(rows, picks)])
            best[(md, cn)] = picks
            L.append(f"| {md}：{cn} | {v:.2f} |")
    L.append(f"| 理想上限（每条都选对） | {m([max(jf1[r['key']]['J&F'], jf2[r['key']]['J&F']) for r in rows]):.2f} |")
    clear = [i for i, r in enumerate(rows) if abs(jf1[r["key"]]["J&F"] - jf2[r["key"]]["J&F"]) > 0.2]
    L += ["", f"两个词的 J&F 差 > 0.2 的明显样本 {len(clear)} 条，各准则选对的条数：", ""]
    for (md, cn), picks in best.items():
        ok = sum(J(rows[i], picks[i]) >= max(J(rows[i], w) for w in rows[i]["words"]) for i in clear)
        L.append(f"- {md}：{cn}  {ok} / {len(clear)}")
    md0 = "gv2" if "gv2" in modes else modes[0]
    L += ["", f"## 明显样本（{md0}）", "",
          "| 表达式 | v1 词 | v2 词 | 是不是指代对象 v1 / v2 | 反向 logp v1 / v2 | J&F v1 / v2 |", "|---|---|---|---|---|---|"]
    for i in clear[:60]:
        r = rows[i]
        d1, d2 = r["s"][md0][r["words"][0]], r["s"][md0][r["words"][1]]
        yn = f"{d1['yn']:.1f} / {d2['yn']:.1f}" if "yn" in d1 else "-"
        L.append(f"| {r['expression'][:70]} | {r['v1']} | {r['v2']} | {yn} | {d1['rev']:.1f} / {d2['rev']:.1f} | "
                 f"{jf1[r['key']]['J&F']:.2f} / {jf2[r['key']]['J&F']:.2f} |")
    Path(a.out, "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:30]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", default="")
    ap.add_argument("--gckpt", default="", help="Ground v2 目录（有 lora/ 就同时算 gv2 和 base 两种）")
    ap.add_argument("--data-root", default="", help="Ref-YTVOS valid/JPEGImages")
    ap.add_argument("--meta", default="")
    ap.add_argument("--v1-scores", default="")
    ap.add_argument("--v2-scores", default="")
    ap.add_argument("--jf-v1", default="")
    ap.add_argument("--jf-v2", default="")
    ap.add_argument("--all", action="store_true", help="词相同的表达式也算（默认只算词不同的）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="runs/noisy_channel")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args(argv)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if not a.report_only:
        v1, v2 = load_words(a)
        items = build_items(a, v1, v2)
        if a.limit:
            items = items[: a.limit]
        sp = Path(a.out) / "scores.jsonl"
        done = {json.loads(l)["key"] for l in open(sp, encoding="utf-8")} if sp.is_file() else set()
        todo = [it for it in items if it["key"] not in done]
        print(f"[nc] {len(items)} 条，已有 {len(done)}，待算 {len(todo)}", flush=True)
        if todo:
            sc = Scorer(a)
            t0 = time.time()
            with open(sp, "a", encoding="utf-8") as f:
                for n, it in enumerate(todo, 1):
                    try:
                        it["s"] = sc.score(it)
                    except Exception as e:  # 单条失败不影响其他条，重跑会补
                        print(f"[error] {it['key']}: {type(e).__name__}: {e}", flush=True)
                        continue
                    it.pop("frames")
                    f.write(json.dumps(it, ensure_ascii=False) + "\n")
                    f.flush()
                    if n % 20 == 0 or n == len(todo):
                        print(f"[{n}/{len(todo)}] {(time.time() - t0) / n:.2f}s/条", flush=True)
    report(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
