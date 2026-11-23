# ground_v2_probe：Ground 重构第 0 步（免训练）

要回答的问题：**一个完全冻结的 Qwen3-VL，读完整句表达式后说出目标是什么，同时它看向的位置，是否已经比现在的 Ground 热力图更准？**

- 如果是，《Ground重构方案》里"热力图应来自 LLM 生成目标时的内部状态"的方向就成立，这个免训练版本也可以先接给补召回用。
- 如果不是，就需要按方案用 LoRA 加实例对比损失去训练。

不训练，不改仓库，不加载 Ground 的 query 模块，只用原版 `Qwen3-VL-4B-Instruct`。帧用**原分辨率**。

```
probe/
  run_probe.py     主脚本：零样本出词 → 强制同一答案前向 → 逐层取"类别词 token → 图像 token"的注意力 → 指标与作图
  attn_capture.py  在 Qwen3-VL 文本层的 RoPE 之后取 q/k，只为指定的几行计算注意力（不换 sdpa，显存几乎不增加）
  metrics.py       GT 解码、token 级指标、作图
tests/test_probe.py  用随机小模型在 CPU 上测试：(1) 取出的注意力与 eager 注意力逐元素一致；(2) 全流程跑通
```

## 运行（8 例失败包）

```bash
cd ground_v2_probe
PY=/mnt/csip-500/share/kongjiawei/Sa2VA/.venv/bin/python
REPO=/mnt/csip-500/share/kongjiawei/Sa2VA-main
PKG=$REPO/paper_assets/ground_response_recall_failures_20261119      # 你 8 例失败包在服务器上的目录

CUDA_VISIBLE_DEVICES=0 $PY -m probe.run_probe \
  --model-path $REPO/Qwen3-VL-4B-Instruct \
  --input-jsonl $PKG/input_8_failures.jsonl \
  --video-root /mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/revos \
  --ground-npz-dir $PKG \
  --out runs/probe_8cases
```

- `--video-root`：从 `<root>/<video_id>/<frame>.jpg` 读原图。不传时，脚本会从 `side336` 缓存文件名还原原图路径；还原不了就退回缓存，并打印警告。
- `--ground-npz-dir`：按顺序读取 `case_00.npz` 到 `case_07.npz`，在同样的帧上给出现有 Ground 的 MAX 图作为对照。
- 先加 `--limit 1` 试一例。日志会打印 `grid=(23, 40)` 之类的网格，确认是原分辨率。
- 显存：原分辨率 3 帧约 2800 个视觉 token，4B bf16 预计十几 GB。不够可以用 `--max-side 960`。
- 每例 3 次前向：生成、强制答案、通用提示。不需要第 3 次可以加 `--no-generic`。

## 只出词：全量验证集零样本类别词（给 SAM3 召回用）

8 例结果显示，冻结模型的零样本词明显好于现有 Ground。是否值得换词，要在全量 ReVOS 验证集上用 SAM3 召回来判断。`--words-only` 只做生成，不取注意力，每条约 1–3 秒：

```bash
CUDA_VISIBLE_DEVICES=0 $PY -m probe.run_probe --model-path $REPO/Qwen3-VL-4B-Instruct \
  --input-jsonl <Ground 在 ReVOS 验证集上的预测 jsonl（与 input_8_failures.jsonl 同格式，含 video_id / exp_id / expression / frames 或 frame_paths）> \
  --video-root /mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/revos \
  --words-only --out runs/zeroshot_words_revos_valid
```

输出 `categories.json`，每条记录带 `pred_category`（零样本词）和 `ground_v1_category`。它和 p1a 包里 `run_p1a --recompute-text --category-json` 的输入格式一致，也可以直接替换你现有召回流程的类别词文件。

## 输出

- `summary.md`：
  - 每例的词：现有 Ground 和冻结模型零样本，各说了什么。
  - 各种热力图在全部 GT 可见帧上的指标：峰值是否在 GT 上、top-3 / top-5 命中、GT 上的注意力占比、lift（注意力占比 ÷ GT 面积占比；1 表示和随机一样）。
  - 逐层的峰值命中率：哪几层最会"看"。
- `fig/XX.png`：每例三帧。依次是原图加 GT，以及各种注意力图（青色 × 为峰值位置）。
- `results.jsonl`、`maps/XX.npz`：逐帧指标和逐层原始注意力，可做后续分析。

## 几种热力图

| 名称 | 含义 |
|---|---|
| word·all layers | 生成的类别词 token 对图像 token 的注意力，所有层、所有头平均（F-LMM 的做法） |
| word·all layers, sinks removed | 同上，但把孤立的极高值 token（注意力汇聚点 / sink，常出现在第一帧顶部）换成邻域均值。8 例里这一版最好 |
| word·middle layers | 只取中间一半的层。经验上中间层的定位最好，逐层表会验证这一点 |
| pre-word·middle layers | "即将说出类别词"那个位置的注意力 |
| word/generic·middle layers | 同一个词放在通用提示（"Describe these frames."）后的注意力作分母（"MLLMs Know Where to Look", ICLR 2025）。8 例上无效，信号被噪声放大，可以加 `--no-generic` 省掉这次前向 |
| Ground v1 MAX (336) | 现有 Ground 的 MAX 图，只在传了 `--ground-npz-dir` 时出现 |

## 怎么读结果

- **零样本的词：** 如果冻结模型在 00 / 02 / 05 上说出了 greenery、deer、car 一类的词，说明"视觉压过文字"是现有 Ground 的结构造成的，不是 Qwen3-VL 本身的问题。
- **定位：** 如果 word / generic 那一行的峰值命中和 lift 明显高于 Ground v1，方向就成立。
- **逐层表：** 命中率最高的几层，可以直接作为 Ground v2 里取视觉隐状态的层号。

建议 8 例看完后，再按同样的 jsonl 格式（expression、frames、mask_rles、pred_category）导出约 100 条 ReVOS 失败样本跑一次，结论才稳。

## 测试

```bash
python tests/test_probe.py     # 需要 transformers 里有 Qwen3-VL；不需要权重和 GPU
```
