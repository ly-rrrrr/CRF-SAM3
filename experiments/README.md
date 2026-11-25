# experiments/：2026-11 后续实验包（独立于主流程）

这里的每个目录都是**独立实验包**，不修改 `Sa2VA-main/` 里的论文主流程代码；运行时通过各包 `scripts/env.sh`（或 README 里的变量）指向服务器上的 `Sa2VA-main` 仓库、模型和数据。关键结果汇总见仓库根目录 [RESULTS.md](../RESULTS.md)。

| 目录 | 做什么 | 状态（2026-11-23） |
|---|---|---|
| [`ground_v2_probe/`](ground_v2_probe/) | Ground 重构第 0 步：冻结 Qwen3-VL，读取“类别词 → 图像 token”注意力，看它是否比旧 Ground 热力图更准（免训练） | 代码已交付（8 例失败样本的探针脚本） |
| [`ground_v2/`](ground_v2/) | Ground v2：Qwen3-VL-4B 全 36 层 LoRA，第 12 层特征出热力图（summary detach），原分辨率训练/推理；含数据构建、训练（含 NaN 保护、loss 曲线）、评估、ReVOS valid 出词、`recall_pool`（用新词重跑 SAM3 召回） | 已训练、已评估（代码为服务器上适配 transformers 4.57.1 / DDP 后的版本） |
| [`p1a_spatial_recall/`](p1a_spatial_recall/) | 用 v2 热力图给 SAM3 补点种子，扩展候选池（point / rescue / oracle_gt）；`verify_pool.py` + `scripts/verify_v2.sh` 在扩展后的池子上重跑 Verify 得最终 J&F | 已完成：ReVOS 最终 J&F 62.38（见 `runs/verify_v2/compare.md`） |
| [`crf_sam3_v2/`](crf_sam3_v2/) | **新主线，一个入口跑完**：`run_v2.sh <dataset>`；Ground v2 出词 → 仓库原版 CRFSAM3Pipeline（无 backup 词）→ 热力图点种子补轨迹 → 同一个 Verify 选前 num_targets；另含 Ground 候选词重排（`--k`、`WORD_SCORE=yesno|rev`）与反向打分验证工具 | Ref-YTVOS（借 ReVOS 训练的 Ground v2）已跑完；按数据集训练 Ground v2 与词重排在进行中 |
| [`refytvos_frame_fix/`](refytvos_frame_fix/) | 修复 Ref-YTVOS 预测掩码按文件名而非帧位置写出导致的帧错位，并检查旧预测受影响程度 | 检查完成，`crf_sam3_v2` 已按帧位置写掩码 |
| [`ring_flow/`](ring_flow/) | 与 CRF-SAM3 无关的视频原生范式探索（固定圆心环形滤波 + 流匹配），纯合成数据、无预训练模型 | CPU 小规模参考结果（`runs_cpu_reference/`） |
| [`videorefer_zeroshot_verify/`](videorefer_zeroshot_verify/) | VideoRefer-7B 零样本替换 Verify 的可行性对照（不改 Ground/SAM3、不训练） | 仅 CPU 冒烟测试，未在 GPU 上真实运行 |

## 运行顺序

新主线直接用 `crf_sam3_v2/run_v2.sh refytvos|revos [all|ground|heat|run|eval]`（见其 README）。下面是 2026-11-23 之前分步实验的顺序：

1. `ground_v2`：`scripts/01_build_data.sh` → `03_train.sh` → `06_check_and_eval.sh` → `05_predict_revos_valid.sh` → `07_recall_pool.sh`
2. `p1a_spatial_recall`：`scripts/revos_v2.sh check|zero|fail|ok|summary`
3. `p1a_spatial_recall`：`scripts/verify_v2.sh check|fidelity|points|predict|eval`

## 没有放进 git 的内容

模型权重 / LoRA checkpoint、生成的训练数据（`ground_v2/data/`）、热力图 npz、候选池 RLE、`runs/` 下的逐样本预测与日志、各类 zip。`runs/` 里只保留了小的汇总文件（`compare.md`、`report.md`、`summary.md`、`STATUS_*.txt`）。这些大文件仍在服务器与本机 `Desktop/论文/` 下。
