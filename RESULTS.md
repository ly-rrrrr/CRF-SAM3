# RESULTS：关键结果汇总（截至 2026-11-25）

论文主结果（v19 稿件依据）见 [README.md](README.md#recorded-results) 与 [PAPER_EVIDENCE_MAP.md](PAPER_EVIDENCE_MAP.md)。本文件汇总 2026-11 之后围绕“候选池质量”的后续实验，代码在 [experiments/](experiments/)。

## 0. 论文主结果（已记录）

| 数据集 | J&F |
|---|---:|
| MeViS valid_u | 71.45 |
| Ref-DAVIS17 valid | 76.39 |
| Ref-YouTube-VOS valid（overlay） | 66.47 |
| ReVOS valid（官方 5,822 条） | 59.74（J 57.20 / F 62.28） |

> 注：v19 稿件中 Ref-YTVOS / Ref-DAVIS17 主表因去掉备用类别词逻辑而暂为 “--”，待重跑。

## 1. Ground v2（`experiments/ground_v2/`）

设置：Qwen3-VL-4B-Instruct，全 36 层 LoRA（可训练 81.59M），第 12 层特征出热力图，summary detach；原分辨率；2 epoch ≈ 2,344 step，2×GPU 约 3h36m，无 NaN。损失：mask 0.75→0.16，dice 0.85→0.10，cat 0.14→0.03。

ReVOS 内部验证集 1,041 条（A = 旧 Ground，E = v2，E′ = v2 推理时去掉 summary，与 E 一致）：

| 指标 | A 旧 Ground | E v2 |
|---|---:|---:|
| 类别词正确率 | 0.786 | **0.835** |
| 多名词子集正确率（n=66） | 0.864 | 0.849 |
| 热力图质量落在 GT（mass_in_gt） | 0.225 | **0.696** |
| 峰值落在 GT（peak_in_gt） | 0.477 | **0.793** |
| 落在别类物体（扣掉 GT） | 0.336 | **0.157** |
| 相对平图倍数：GT / 别类 | 1.88× / 1.60× | 5.82× / 0.75× |

来源：`experiments/ground_v2/runs/eval_v2metric/compare.md`。

## 2. 候选池召回（ReVOS valid，官方 Ref+Rea 5,605 条，Recall@0.5 = 池中至少一个候选 J ≥ 0.5）

| 候选池 | Referring | Reasoning | Official |
|---|---:|---:|---:|
| 旧 Ground 词 + SAM3 文本召回 | 80.83 | 62.22 | 72.61 |
| v2 词 + SAM3 文本召回 | 79.55 | 65.09 | 73.17 |
| v2 词 + 热力图 rescue（低阈值检测） | 81.88 | 67.76 | 75.65 |
| **v2 词 + 热力图点种子（point）** | **85.72** | **73.33** | **80.25** |
| v2 + point ∪ rescue | – | – | 80.75 |
| 上限：GT 框做种子（oracle_gt，非方法） | 91.37 | 84.28 | 88.24 |

- 换词：1,531/5,822 条的词变了；救回 304、变差 273（其中 171 条是 v2 词在视频中不存在导致 0 候选）；0 候选率 1.93% → 7.05%。
- point：救回 zero 子集 180/419、fail 子集 217/1,244；平均每条多 0.70 个候选（Verify 需承受的干扰）。池 oracle J 65.18 → 71.15。
- 剩余失败（fail 1,244 条）：614 条即使给 GT 种子 SAM3 也失败（SAM3 上限）；277 条种子落在错误物体；305 条未触发（热力图落在已有的错误候选上）。
- 3 条 OVIS/train/a3f0cb28 样本因 Triton CUDA 错误未跑出，全量数字按下限计。

来源：`experiments/ground_v2/runs/pool_revos/report.md`、`experiments/p1a_spatial_recall/runs/revos_v2/summary.md`。

## 3. 最终 J&F：扩展候选池上的 Verify（ReVOS valid，官方 eval_revos.py）

| 候选池 | Referring J&F | Reasoning J&F | Overall J | Overall F | Overall J&F |
|---|---:|---:|---:|---:|---:|
| 论文结果（原流程，含 backup 词） | 64.92 | 54.56 | 57.20 | 62.28 | 59.74 |
| 旧词池（复现，无 backup 词） | 65.05 | 54.36 | 57.18 | 62.23 | 59.71 |
| v2 词池 | 64.54 | 56.63 | 58.34 | 62.83 | 60.59 |
| **v2 词池 + 热力图点种子** | **66.17** | **58.58** | **59.86** | **64.90** | **62.38** |

- 5,822 条全部打分，无缺失；v2_point 与 v2 选择不同 908 条，均选中点种子轨迹。
- 6 条 OVIS/train/a3f0cb28 表达式原先因 SAM3 Triton 连通域 CUDA 错误失败，已在 `p1a/sam3_ops.py` 加该错误的 CPU 回退后补齐。

来源：`experiments/p1a_spatial_recall/runs/verify_v2/compare.md`。

## 4. Ref-YouTube-VOS：帧错位修复 + 新主线（`experiments/refytvos_frame_fix/`、`experiments/crf_sam3_v2/`）

帧错位：旧写法按文件名写掩码，732/834 条表达式、3,516/22,046 帧（15.9%）被写成别帧的掩码；受影响帧逐帧 J 31.25 vs 正常帧 71.07。ReVOS 仅 0.61% 帧受影响。

新主线（`run_v2.sh refytvos`，Ground v2 借用 ReVOS 训练的权重，跨域）：

| 结果 | J | F | J&F（表达式级） | J&F（视频级） |
|---|---:|---:|---:|---:|
| 论文主线（帧错位） | 64.27 | 66.57 | 65.42 | 68.17 |
| Ground v2 词（无点种子） | 68.10 | 71.77 | 69.94 | 72.62 |
| **Ground v2 词 + 热力图点种子** | **69.96** | **74.05** | **72.01** | **75.07** |

- 候选池 Recall@0.5 85.85% → 92.93%；池内最好候选平均 J 75.46 → 81.28（实际选出结果的 J 为 69.96，差距在 Verify）。
- 拆解：大部分提升来自帧对齐修复；与 v1 换了词的 241 条上，跨域 v2 词（55.62）不如 v1 词（57.93），点种子救回到 65.28。帧修复后的 v1 基线估计约 72–73，因此跨域 Ground v2 本身几乎没有净增益，需按数据集训练。
- 241 条换词样本上的候选词选择：反向打分（noisy-channel）各准则 56–58，理想上限 67.95（`crf_sam3_v2/runs/noisy_channel/report.md`）。

来源：`experiments/crf_sam3_v2/runs/refytvos_gv2revos/summary.md`、`experiments/refytvos_frame_fix/runs/check/report.md`。

## 5. 进行中 / 未完成

- Ref-YTVOS 按数据集训练的 Ground v2（`ground_v2/gv2/build_refytvos.py`）及新主线重跑。
- Ground 候选词重排（`crf_sam3_v2/v2/ground_rerank.py`，从 top-K 候选词中选一个，`WORD_SCORE=yesno|rev`）。
- **VideoRefer-7B 零样本 Verify**：`experiments/videorefer_zeroshot_verify/`，只做过 CPU 冒烟测试。
