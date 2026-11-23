# RESULTS：关键结果汇总（截至 2026-11-23）

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

## 3. 进行中 / 未完成

- **最终 J&F（扩展池上的 Verify）**：`experiments/p1a_spatial_recall/scripts/verify_v2.sh`，行 old / v2 / v2_point。20 条检查：19/20 与论文选择一致，概率最大差 0.093（bf16 logit 量化噪声）。全量待服务器运行。
- **VideoRefer-7B 零样本 Verify**：`experiments/videorefer_zeroshot_verify/`，只做过 CPU 冒烟测试。

