# VideoRefer-7B 零样本 Verify 可行性实验（CRF-SAM3 独立实验包）

本包只做一件事：**在不改 Ground、不改 SAM3、不训练任何模型的前提下**，把 CRF-SAM3 在 ReVOS valid 上的真实候选 Mask Tube 和指代表达式直接交给现成的 VideoRefer-7B 做候选判别，并在**同一批候选、同一个目标数 k** 上与当前 Qwen3-VL(+LoRA) Verify 对照。

```
Step 1  step1_dump_candidates.py   CRF-SAM3 环境，GPU   冻结 SAM3-DMS 重放，导出每条表达式的候选 Tube（RLE）并审计
Step 2  step2_score_videorefer.py  VideoRefer 环境，GPU 对每个候选打分（Yes/No 或多选）
Step 3  step3_evaluate.py          任意环境，CPU        与历史 Qwen Verify 对照：J/F/J&F、Top-1、MRR、AUC、互补性
```

三个步骤分开运行，SAM3 与 VideoRefer 不会同时占用显存。

---

## 0. 先说清楚一个关键事实：为什么需要 Step 1

历史 ReVOS valid 运行（`crf_sam3_revos_valid_revos_query_ranker_20260806`，配置里 `save_visual_tubes: false`）保存了：

- `scores.json`：每条表达式的召回词 `used_category_query`、全部候选 id、每个候选的 Qwen P(Yes)、选中 id；
- `results.json`：最终预测 RLE；
- `revos_valid_official.csv`：官方逐表达式 J/F。

但**没有保存未被选中候选的 mask**（`visual_tubes_temp/obj_*` 跨表达式复用，不能用）。所以 Step 1 用**冻结的** SAM3-DMS、记录下来的召回词和同样的 recall24→keep7 预算重新跑一遍召回，把候选 Tube 存下来。做法与 `paper_assets/revos_qualitative_20261008/scripts/replay_recall.py` 完全一致（那次在 A/B/C 三个案例上候选 id 集合与历史一致，选中候选与历史最终 mask 逐帧逐像素相同）。

Step 1 会对每条表达式自动审计：

- `ids_match`：重放保留的候选 id 是否等于历史 `all_candidates`；
- `selected_exact_frames`：把历史选中的 id 在重放结果里取并集，是否与历史 `results.json` 每一帧逐像素相同（需要提供 `--historical_results`）。

默认只评测 `ids_match=true` 的表达式；加 `--require_pixel_exact` 可只保留逐像素一致的表达式（最严格）。Step 3 还会报告"Qwen 历史选择在重放候选上的 J&F"与官方 CSV 的差，作为重放保真度。

**只需重放"有竞争"的表达式**：候选数 > 选中数（scores.json 统计：5,822 条中 3,728 条，涉及 359 个视频、652 个 (视频, 召回词) 对）。其余 2,094 条（0 候选或候选数 ≤ k）无论用哪个 Verify 结果都一样，Step 3 直接沿用历史官方逐表达式指标，因此全量 ReVOS 数字中 Qwen 一行应当正好复现 59.74。

---

## 1. 需要准备的路径

以下用原服务器路径举例（来自仓库中的配置），按实际情况替换：

| 变量 | 含义 | 示例 |
|---|---|---|
| `CRF_ROOT` | 含 `crf_sam3_demo.py` 的目录 | `/mnt/csip-500/share/kongjiawei/Sa2VA-main` |
| `SAM3_CKPT` | SAM3 权重 | `/mnt/csip-500/share/kongjiawei/sam3-main/sam3-main/checkpoints/sam3/sam3.pt` |
| `REVOS` | ReVOS 根目录，含 `meta_expressions_valid_.json`、`mask_dict.json` 和各视频帧目录 | `/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/revos` |
| `HIST` | 历史运行目录 | `$CRF_ROOT/crf_sam3_revos_valid_revos_query_ranker_20260806` |
| `VR_REPO` | VideoRefer 仓库中含 `videorefer/` 包的目录 | `/path/to/VideoRefer/VideoRefer` |
| `VR_CKPT` | VideoRefer-7B 权重 | `/path/to/VideoRefer-7B` |
| `SIGLIP` | siglip-so400m-patch14-384（离线服务器才需要） | `/path/to/siglip-so400m-patch14-384` |
| `WORK` | 本实验输出目录 | `/path/to/vr_zeroshot_work` |

`HIST` 下需要 `scores.json`、`revos_valid_official.csv`（GitHub 仓库里已有），以及服务器上的 `results.json`（用于逐像素审计和导出，可选但强烈建议）。

`scripts/paths.example.sh` 是一个可直接改的模板。

---

## 2. 环境

### Step 1：直接用 CRF-SAM3 现有环境

```bash
# 例：原服务器上的环境
source /mnt/csip-500/share/kongjiawei/Sa2VA/.venv/bin/activate
pip install pycocotools tqdm   # 一般已装
```

### Step 2：单独建 VideoRefer 环境（不要装进 CRF-SAM3 环境，transformers 版本冲突）

```bash
git clone https://github.com/DAMO-NLP-SG/VideoRefer
conda create -n videorefer python=3.10 -y && conda activate videorefer
# 只装推理需要的包（官方 requirements.txt 里还有 gradio/deepspeed 等训练和 demo 依赖，可不装）
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.46.3 tokenizers==0.20.3 accelerate==1.0.1 timm==1.0.3 \
    decord==0.6.0 imageio==2.34.0 imageio-ffmpeg==0.4.9 moviepy==1.0.3 opencv-python==4.6.0.66 \
    einops==0.6.1 einops-exts==0.0.4 sentencepiece==0.1.99 pycocotools==2.0.8 numpy==1.24.4 \
    pillow tqdm requests
pip install bitsandbytes==0.43.3   # 只有用 --quant 8bit/4bit 时需要
# flash-attn 可选；不装就用默认的 --attn sdpa
```

权重下载（服务器能联网可以跳过，脚本会按 HF id 自动下载）：

```bash
huggingface-cli download DAMO-NLP-SG/VideoRefer-7B --local-dir $VR_CKPT
huggingface-cli download google/siglip-so400m-patch14-384 --local-dir $SIGLIP
```

VideoRefer-7B 的 `config.json` 里视觉塔写的是 HF id `google/siglip-so400m-patch14-384`。离线服务器用 `--vision_tower $SIGLIP` 覆盖即可，**目录名里必须包含 `siglip`**（VideoRefer 按名字判断视觉塔类型）。

### Step 3：任意有 `numpy opencv-python scikit-image pycocotools` 的 Python 环境（CRF-SAM3 环境即可），纯 CPU。

---

## 3. 运行

### 3.0 先跑本地冒烟测试（CPU，30 秒，不需要任何权重）

```bash
bash scripts/smoke_test.sh            # 用 PY=/path/to/python 指定解释器
```

它会生成一个假的 ReVOS 小数据集，用假召回和随机打分走通 Step 1→2→3，确认依赖和路径逻辑没问题。

### 3.1 Step 1：导出候选（CRF-SAM3 环境）

先用 20 个 (视频, 召回词) 对做试跑：

```bash
python step1_dump_candidates.py \
  --crf_root $CRF_ROOT --sam3_checkpoint $SAM3_CKPT \
  --revos_root $REVOS \
  --historical_scores $HIST/scores.json \
  --historical_results $HIST/results.json \
  --out_dir $WORK --max_pairs 20
cat $WORK/replay_audit_summary.json
```

看 `contested_ids_match` / `contested_selected_pixel_exact` 是否接近 `contested_replayed`。没问题后去掉 `--max_pairs` 跑全量（可断点续跑，已完成的对会跳过）。

多卡并行：

```bash
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python step1_dump_candidates.py ...同上... \
     --num_shards 4 --shard_id $i --gpu 0 &
done; wait
python step1_dump_candidates.py --revos_root $REVOS --historical_scores $HIST/scores.json \
  --out_dir $WORK --manifest_only
```

显存：与原推理配置相同（CPU offload 打开）。共享显卡可加 `--memory_fraction 0.5`。某个长视频 OOM 只会记为该对失败，不会中断。

输出：`$WORK/candidates/**.json.gz`（候选 RLE）、`$WORK/pair_status*.jsonl`、`$WORK/manifest.jsonl`、`$WORK/replay_audit_summary.json`。

### 3.2 Step 2：VideoRefer 打分（VideoRefer 环境）

主实验（逐候选 Yes/No，与当前 Verify 的 P(Yes) 形式一致）：

```bash
python step2_score_videorefer.py \
  --manifest $WORK/manifest.jsonl --image_root $REVOS \
  --videorefer_repo $VR_REPO --model_path $VR_CKPT --vision_tower $SIGLIP \
  --mode yesno --prompt_id p0 --mask_frames 8 \
  --out $WORK/vr_yesno_p0_m8.jsonl
```

建议补充的几组（每组一个输出文件，Step 3 一起比较）：

| 目的 | 改动 | 输出名示例 |
|---|---|---|
| 与 Qwen 看到的 4 帧完全相同 | `--mask_frames 4` | `vr_yesno_p0_m4.jsonl` |
| 更多目标帧 | `--mask_frames 16` | `vr_yesno_p0_m16.jsonl` |
| 提示词敏感性 | `--prompt_id p1` / `p2` | `vr_yesno_p1_m8.jsonl` |
| 列表式多选（所有候选同时比较） | `--mode choice`（默认正序+倒序平均以消除位置偏置） | `vr_choice_c0_m8.jsonl` |

先用 `--max_expressions 50` 试跑，确认 `top_token` 大多是 Yes/No（或字母），再跑全量。支持断点续跑和 `--num_shards/--shard_id` 多卡分片（按视频分片）。

**显存建议**

| 显存 | 参数 |
|---|---|
| ≥ 24 GB | 默认 fp16（权重约 16 GB） |
| 16 GB | `--quant 8bit` |
| 12 GB | `--quant 4bit`；仍紧张可再加 `--mask_max_side 640` 或 `--mask_frames 4` |

量化时视觉塔、projector、region encoder 和 lm_head 保持 fp16，只量化语言模型主体。量化会带来少量数值差异，论文里如使用请注明。

**输入构造（与官方 VideoRefer-Bench 推理一致）**

- 全局视频：在**整段视频**上均匀采样 `--video_frames` 帧（默认取 VideoRefer 自带的 `NUM_FRAMES`），不画 mask，交给 VideoRefer 的视频分支；每个视频只编码一次并缓存。
- 目标指代：候选 mask 放在该候选出现过的帧中均匀采样的 `--mask_frames` 帧上，走 VideoRefer 的 region encoder（`[<region>]` token）。采样规则与 CRF-SAM3 4 帧视觉 Tube 完全相同，`--mask_frames 4` 时与 Qwen Verify 看到的帧一致。
- 只做一次前向（prefill），读下一个 token 的 logits：`score = P(Yes) / (P(Yes)+P(No))`；多选模式取选项字母的 softmax。不生成文本，不改任何权重。

### 3.3 Step 3：评测与对照（CPU）

```bash
python step3_evaluate.py \
  --manifest $WORK/manifest.jsonl --revos_root $REVOS \
  --official_csv $HIST/revos_valid_official.csv \
  --vr yesno_p0_m8=$WORK/vr_yesno_p0_m8.jsonl \
  --vr yesno_p0_m4=$WORK/vr_yesno_p0_m4.jsonl \
  --vr choice_c0_m8=$WORK/vr_choice_c0_m8.jsonl \
  --out_dir $WORK/eval --workers 16
```

可选：`--export_results --historical_results $HIST/results.json` 会为每个方法导出完整的 ReVOS `results.json`，再用仓库里的官方评测脚本复核：

```bash
cd $CRF_ROOT && python tools/eval/eval_revos.py $WORK/eval/results_json/vr_yesno_p0_m8/results.json \
  --exp_path $REVOS/meta_expressions_valid_.json --mask_path $REVOS/mask_dict.json \
  --foreground_mask_path $REVOS/mask_dict_foreground.json
```

---

## 4. 输出怎么读

`$WORK/eval/report.md`（中文表格）、`summary.json`、`per_expression.csv`。

对照的方法（全部在同一批候选、同一个 k 上）：

| 方法 | 含义 |
|---|---|
| `qwen` | 当前 Qwen3-VL+LoRA Verify 的历史选择 |
| `vr:<label>` | VideoRefer 零样本打分取前 k |
| `fuse:<label>` | 0.5·P_qwen + 0.5·P_videorefer，固定权重，仅看互补性 |
| `sam3_native` | 按 SAM3 原生分数和取前 k（相当于不做 Verify） |
| `random` | 随机选 k 个的期望 |
| `oracle` | 按候选与 GT 的 J&F 取前 k（Verify 的上界；k>1 时为近似） |

报告内容：

1. **有竞争子集**的 referring / reasoning / overall J&F（overall 按 ReVOS 官方口径＝两类平均）；
2. **全量 ReVOS valid**：未评测的表达式沿用历史官方逐表达式指标，`qwen_official` 应等于 59.74；
3. **候选判别**（k=1 且最佳候选 J ≥ 0.5 的表达式）：Top-1 准确率、MRR、成对 AUC；
4. **Qwen 与 VideoRefer 的一致性与互补性**：选择相同的比例、各自更好的条数、逐条取二者更优的上界；
5. 按表达式类型、候选数、来源数据集分组。

判读建议：VideoRefer 是否"能迁移"主要看 3（判别能力）和 1 的 reasoning 列（时序、推理类表达式是否受益）；`best_of_two` 明显高于两者时说明二者互补，可作为后续方向的依据。

---

## 5. 注意事项

- **数据泄漏**：VideoRefer-700K 的 QA 数据使用了 MeViS、A2D、YouTube-VOS 视频。本包默认只在 ReVOS 上评测；若以后扩展到 MeViS / Ref-YouTube-VOS，需要说明 VideoRefer 可能见过这些视频。
- **公平性**：Qwen Verify 经过 LoRA 训练（含 ReVOS 训练集），VideoRefer 完全零样本。结果只说明"现成对象级 Video LLM 的迁移能力"，不是同等训练条件下的模型对比。
- **颜色词**：历史 Qwen Verify 遇到颜色词时会改用 bbox 可视化；VideoRefer 始终使用 mask 区域特征，这是两者输入形式上的固有差异。
- **候选 id**：只在 `ids_match=true` 的表达式上比较，Qwen 分数与重放候选一一对应；`--include_id_mismatch` 可放宽但不建议用于正式数字。
- **VideoRefer-VideoLLaMA3-7B**：官方后续版本接口不同（HF `trust_remote_code`），本包未包含；如需要可在 `step2` 里按同样的打分接口再加一个后端。

## 6. 文件清单

```
step1_dump_candidates.py   冻结 SAM3-DMS 重放 + 审计 + manifest
step2_score_videorefer.py  VideoRefer-7B 零样本打分（yesno / choice，含 --dry_run）
step3_evaluate.py          对照评测与报告
common/io_utils.py         RLE、候选文件、帧采样（与 CRF-SAM3 / VideoRefer 规则一致）
common/prompts.py          提示词模板（p0/p1/p2/c0）
common/revos_metrics.py    ReVOS 官方 J/F 实现（拷贝自仓库 third_parts/revos/utils/metircs.py）
tools/make_fake_revos.py   冒烟测试用假数据
scripts/smoke_test.sh      CPU 冒烟测试
scripts/paths.example.sh   路径模板
scripts/run_all.example.sh 全流程示例
```
