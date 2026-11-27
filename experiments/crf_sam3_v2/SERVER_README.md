# 服务器适配版 · 2026-11-26（c：在服务器版上再改两处）

## c 版相对服务器版的改动（见 CHANGES_FROM_server.patch）

1. Verify 训练遇到 CUDA OOM：两张卡在同一步一起丢弃这一步（梯度清零、不更新），样本记到
   `<ckpt>/oom_skips_rank*.jsonl`，累计次数存进 training_state.pt 和 final/run_stats.json；超过
   `--max-oom-skips`（默认 50，约占 10.6 万步的 0.05%）就停。原因：服务器版遇 OOM 直接停，而续训时采样顺序
   完全确定，会在同一个样本上再次 OOM，变成死循环。非有限 loss / 梯度仍然直接停。
2. Verify 试训（VERIFY_MAX_STEPS）不再要求缓存完整，可以在 LIMIT 缓存上先跑。正式训练仍要求完整。


基于 `crf_sam3_v2_20261126b.zip`，独立目录 `crf_sam3_v2_server_20261126`。
本次仅改此新目录，未改原 `crf_sam3_v2/`、`ground_v2/`、`p1a_spatial_recall/`、仓库训练代码、数据或权重。
新目录完整保留原包的阶段工具，正式流程入口为 `run_full.sh`。原 README/docs 是设计历史，遇到冲突以本文件为准。

## 服务器位置与依赖

- 工作仓库：`/mnt/csip-500/share/kongjiawei/Sa2VA-main`
- Python：`/mnt/csip-500/share/kongjiawei/Sa2VA/.venv/bin/python`
- 基础模型：`$REPO/Qwen3-VL-4B-Instruct`
- 数据集：`/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas`
- SAM3：`/mnt/csip-500/share/kongjiawei/sam3-main/sam3-main/checkpoints/sam3/sam3.pt`
- 代码依赖：同级 `ground_v2/`、`p1a_spatial_recall/`，以及 `$REPO/crf_sam3_attn/`、主线 Python、配置与 `SAM3-DMS`。
- 默认两卡 GPU 2,3；CPU 检查不占用 GPU。

配置集中在 `scripts/server_env.sh`，支持用环境变量覆盖 PY / REPO / MODEL / REVOS_ROOT。
GV2 / P1A / VD / REFYT / SAM3_CKPT / GPUS / RUN_ROOT 也可覆盖；路径应使用绝对路径。
缓存和训练配置依赖仓库原文件，因此这不是脱离基础仓库、模型和数据即可运行的单文件项目。
`server_reference/` 是服务器依赖源码及已有审计报告的只读快照，供本地核验，不在运行时自动导入，也不要直接覆盖你的 GitHub 仓库。
`SERVER_MANIFEST.json` 列出快照来源、SHA256 和服务器软件版本，不包含模型权重和数据集。

## 修正内容

1. ReVOS 多层视频 ID 缓存递归扫描，避免漏掉 OVIS/train/...。
2. Ground 20 步试训写入 runs/smoke/ground；Verify 试训写入 runs/smoke/verify，不污染正式训练。
3. Ground 数据校验成功后才写可复用训练产物及 COMPLETE；划分校验失败不提交 splits。
4. Verify 使用 threshold 负样本策略：J≥0.5 正，J≤0.1 负，中间不进损失，force_pos 关闭。
5. 类别记录按 (video_id, exp_id) 精确核对、拒绝重复；热力图须存在且能读取，响应不得 NaN/Inf。
6. Verify 开训前逐表达式检查缓存完成状态；有排除记录的 no_gt/no_positive/no_candidate/oom 单独计数，不伪装为可训练记录。
7. 子进程失败向上传递，最终预测导出不完整不能继续宣称评测完成。
8. Verify 修正 checkpoint 步数；保存每 rank RNG；保存到临时目录后提交；非有限 loss/梯度停止；OOM 停止而非悄悄改 batch。
9. Verify 训练使用与推理同一颜色词判定，颜色表达式用框、其余用 mask。通过新包内子类注入，不改旧 dataset.py。
10. 正式训练权重要求 COMPLETE 标记，不能拿旧 final 目录掩盖失败。
11. 单卡评测顺序执行，双卡分配 ReVOS 与其他三个域；Ground 推理 ONLY 过滤与汇总保持一致。
12. 保存阶段命令和代码 SHA256 至 runs/provenance/commands.jsonl。

## 推荐运行顺序（本次未执行训练、缓存生成或数据重建）

所有命令在新目录下执行。CPU 测试使用假数据；真实 GPU 前向/训练仍需下一步冒烟验证。

```bash
bash run_full.sh test
bash run_full.sh splits
bash run_full.sh ground-data
GROUND_MAX_STEPS=20 bash run_full.sh ground-train full
bash run_full.sh ground-train
bash run_full.sh ground-infer
LIMIT=2 bash run_full.sh cache
VERIFY_MAX_STEPS=20 bash run_full.sh verify-train   # 在部分缓存上试训，约 40 小时的完整缓存之前先发现训练器问题
bash run_full.sh cache
bash run_full.sh verify-train
bash run_full.sh eval
```

- Ground 默认沿用本地 b 包的 **1 epoch**，不是此前 ReVOS 的 2 epoch。要 2 epoch 必须在正式训练前设 `GROUND_EPOCHS=2`，不能把两者写成同一实验。
- Ground：lr=1e-4、等效 batch=8、原训练器 warmup/linear 调度。中断后不支持自动恢复；脚本会拒绝复用未完成目录，需要检查后归档该目录再重新开始。
- Verify：两卡各 batch=1，grad_accum=1（等效 batch=2），保持原配方恒定 lr=1e-5；新增 grad norm=1 裁剪与非有限值防护。此适配版明确不接受其他 grad_accum，避免未经验证的累积恢复行为。
- Verify 自动从完整 checkpoint_* 恢复；final 的存在本身不代表流程通过。所有权重路径均在新目录 runs 下。
- LIMIT 缓存试跑可被正式缓存续用。正式 Verify 训练必须通过完整性检查；Verify 试训（VERIFY_MAX_STEPS）允许在部分缓存上跑，产物在 runs/smoke/verify。
- 修改数据版本、划分、超参或模型后必须使用新的 RUN_ROOT；不要在已有完成产物上变参续跑。
- OOM 缓存跳过会记录在 done JSONL，正式训练前完整性检查输出各类条数。是否可接受须按论文数据协议判断，不等于补齐了这些样本。

## 已知研究边界

跨数据集及 ReVOS 内部重复视频目前只有首帧候选筛查，尚未逐对确认。本版没有擅自删除官方训练数据，也没有证明无泄漏。相关审计随包保留，正式论文应准确说明最终协议。
历史 J&F 保持原样，不把修帧收益归给 Ground；新协议尚无新成绩。
两折交叉拟合是按数据集视频 ID 划分，尚未跨数据集做真实视频实体去重。
Ground 的 ReVOS 标签仍仅覆盖现有 API 标注的 175 个视频，不宣称使用全部 626 个训练视频。

## 本地拿回哪些内容

整个发布 zip 即可：包括代码、修改差异、CPU 检查日志、服务器路径/依赖清单、依赖源码参考快照、旧审计报告。
不含模型权重、视频帧或大规模预测缓存。正式 GPU 验证前不要把“CPU 测试通过”写成“全流程已验证”。
