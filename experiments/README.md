# experiments/：2026-11 新主线实现代码（独立于原主流程）

每个目录都是独立的包，不修改 `Sa2VA-main/` 里的原主流程代码；运行时通过各包 `scripts/env.sh`（或 README 里的变量）指向服务器上的 `Sa2VA-main` 仓库、模型和数据。

| 目录 | 内容 |
|---|---|
| [`crf_sam3_v2/`](crf_sam3_v2/) | **新主线入口** `run_v2.sh <dataset>`：Ground v2 出词（可选 top-K 候选词重排，`WORD_SCORE=yesno\|rev`）→ 仓库原版 CRFSAM3Pipeline（无 backup 词）→ 热力图点种子补轨迹 → 同一个 Verify 选前 num_targets |
| [`ground_v2/`](ground_v2/) | Ground v2：Qwen3-VL-4B 全 36 层 LoRA，第 12 层特征出热力图（summary detach），原分辨率；数据构建（ReVOS、Ref-YTVOS）、训练、评估、出词、`recall_pool` |
| [`p1a_spatial_recall/`](p1a_spatial_recall/) | 热力图点种子 → SAM3 追踪补候选（ReVOS / Ref-YTVOS），以及扩展候选池上的 Verify 打分（`verify_pool.py`） |
| [`refytvos_frame_fix/`](refytvos_frame_fix/) | Ref-YTVOS 掩码按帧位置写出（修复按文件名写导致的帧错位）及检查工具 |

## 运行

新主线：`bash crf_sam3_v2/run_v2.sh refytvos|revos [all|ground|heat|run|eval]`，单元测试 `bash crf_sam3_v2/run_v2.sh test`，详见各包 README。

## 不放进 git 的内容

实验结果与汇总、探索性代码、模型权重 / LoRA checkpoint、生成的训练数据、热力图 npz、候选池、逐样本预测与日志、zip。
