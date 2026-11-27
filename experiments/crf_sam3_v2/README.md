# crf_sam3_v2：CRF-SAM3 主线和完整训练流程

两个入口：
- `run_full.sh`：**完整流程**，从官方原始数据开始，依次构造训练集、训练两段模型、在四个验证集上评测（2026-11-26 新增，见下一节）。
- `run_v2.sh`：单个数据集的主线推理，以及只读的审计和盘点（之前的用法，见后文）。

## 完整流程（run_full.sh）

只读 `video_datas/` 和 `$REPO`，从不修改它们。所有产物都写在 `crf_sam3_v2/runs/` 下。规则见 `docs/training_set_spec.md`。

| 阶段 | 命令 | 做什么 | 资源 |
|---|---|---|---|
| 0 | `bash run_full.sh test` | 假数据自检（7 组） | CPU |
| 1 | `bash run_full.sh splits` | holdout：v1 Ground 的 val 视频原样保留，其余视频按 10%（seed 42）补齐；ReVOS / Ref-YTVOS 分两折 | CPU |
| 2a | `bash run_full.sh ground-data` | Ground 训练数据（ReVOS + Ref-YTVOS），全量和两折各一份，Sa2VA 式 repeats | CPU |
| 2b | `bash run_full.sh ground-train` | 训练三份 Ground v2：`full`、`cf0`（没见过第 0 折）、`cf1`；配方同 ReVOS v2 | 2 卡 |
| 3 | `bash run_full.sh ground-infer` | 每个视频用没见过它的那份 Ground 出词和热力图（训练集、holdout、四个验证集） | 2 卡并行 |
| 4 | `bash run_full.sh cache` | 四个数据集的 Verify 训练缓存，候选池和推理一样（DMS 24/7 + 点种子），J 用评测口径 | 2 卡分片 |
| 5 | `bash run_full.sh verify-train` | Verify 四数据集联合训练，一份权重，Sa2VA 式 repeats | 2 卡 |
| 6 | `bash run_full.sh eval` | 四个验证集推理 + 官方评测，结果在 `runs/eval/<数据集>/summary.md` | 2 卡 |

- 每个阶段都能断点续跑，重跑同一条命令会跳过已完成的部分。Verify 训练会从最近的 checkpoint 接着训。
- 默认用 GPU 2、3（`GPUS=2,3`）。`ONLY=refytvos,revos` 只跑部分数据集。
- 试跑：`LIMIT=2 bash run_full.sh cache` 让每个分片只做 2 个视频；`GROUND_MAX_STEPS=20`、`VERIFY_MAX_STEPS=20` 让训练只跑 20 步。
- 第 2a 步有两道门槛，不过就停下、不训练：
  - Ref-YTVOS 目标和 PNG 比对，IoU < 0.9 的帧超过 0.5%；
  - ReVOS 的 API 类别记录和 meta 的表达式文本对不上。

### 和论文旧流程不同的地方（都是为了训练和推理同口径）

| 项 | 旧做法 | 现在 |
|---|---|---|
| mask_dict 表达式顺序 | json 原顺序（DAVIS 298 条、Ref-YTVOS 40 条取错 GT） | Sa2VA 加载器的顺序，用 `v2/gt.py` 统一取 |
| 候选的 J | GT 为空的帧不进分母 | 评测口径：所有帧都算，空对空记 1 |
| 正负阈值 | MeViS 0.2/0.05，其余 0.5/0.1 | 四个数据集统一 0.5/0.1；多目标按「和任一单个目标的 J」打标签 |
| 没有正样本的表达式 | force_pos 硬选一个当正样本 | 不进训练缓存，记在 `done_*.jsonl` 里（status = no_positive） |
| 缓存里的词 | prompt 提取器的词、ReVOS 的 API 类别 | 交叉拟合的 Ground v2 词 + 点种子，和推理一样 |
| Verify 提示词 | 推理时只有 ReVOS 用训练时的提示词 | 四个数据集都用 `yesno_images` + train_aligned 提示词，F = 4 |
| 联合配比 | MeViS 1 / DAVIS 4 / Ref-YTVOS 2，没有 ReVOS | 按 Sa2VA 换算，由实际条数算出（`v2/mix.py`） |
| 帧数上限 | 部分数据集跳过 300 帧以上的视频 | 不设上限；显存不够的记 status = oom |

### 代码

| 阶段 | 文件 |
|---|---|
| 取 GT、算 J、打标签 | `v2/gt.py` |
| 划分 | `tools/make_splits.py` |
| Ground 数据 | `v2/build_ground.py`，repeats 在 `v2/mix.py` |
| Ground 推理任务分配 | `tools/ground_infer_plan.py`（训练用 `../ground_v2/gv2/train.py`，推理用 `gv2/predict.py`，都用服务器上的版本） |
| 统一的主线 yaml | `tools/make_mainline_yaml.py` |
| Verify 缓存 | `v2/verify_cache.py`（点种子和推理共用 `v2/mainline.py::point_seeds`） |
| Verify 训练配置 | `tools/make_verify_config.py` |
| Verify 双卡训练 | `v2/train_verify_ddp.py`（包装仓库的 `crf_sam3_attn` 模型和数据集，不改仓库） |
| 推理和评测 | `v2/mainline.py`（四个数据集） |

---

# 单数据集主线（run_v2.sh）


之前的实验是一段一段拼起来的：先导热力图，再单独跑点种子，再单独打分，Ref-YTVOS 还借用了 ReVOS 训练的 Ground v2。
这个包把它们合成一条主线，**所有数据集用同一份代码**，每个数据集用**自己训练的** Ground v2。

## 流程（每条表达式）

1. **Ground v2**（按数据集训练）在 3 帧上输出类别词和热力图。
2. **仓库原版 CRFSAM3Pipeline**（读该数据集主线 yaml）：
   - Qwen3-VL 抽取器给出 num_targets；
   - 用 Ground v2 的词做 SAM3 文本召回（recall 24 / keep 7）；
   - Verify 打分，取前 num_targets 个。
   - 不用 backup 词。

   这一步的结果写成 `words` 行（消融用）。
3. **热力图点种子**：
   - 在热力图上找出候选掩码没盖住的高响应区域；
   - 按质量取前 2 个，用它们的峰值点在 SAM3 上点击，追踪出整段轨迹；
   - 新轨迹最多 3 条，和已有候选重合（IoU > 0.7）的丢掉。
4. **同一个 Verify** 给新轨迹打分，在「原候选 + 新轨迹」里取前 num_targets 个。这一行写成 `final`，就是**新主线结果**。
5. 掩码一律**按 SAM3 帧位置**写出，不再用 `int(帧名)` 查。这修掉了 Ref-YTVOS 上的帧错位问题。

代码位置：

| 内容 | 文件 |
|---|---|
| 主线推理（上面第 2–5 步） | `v2/mainline.py` |
| 入口脚本 | `run_v2.sh` |
| Ref-YTVOS 的 Ground v2 训练数据 | `../ground_v2/gv2/build_refytvos.py`（新增） |

点种子和 SAM3 调用复用 `../p1a_spatial_recall/p1a/`（seeds.py、sam3_ops.py）。Ground v2 的训练和推理复用 `../ground_v2/gv2/`。这些都用你服务器上已打过补丁的版本，本包不覆盖它们。

## 放法

`crf_sam3_v2/`、`ground_v2/`、`p1a_spatial_recall/` 三个目录放在同一个上级目录下，就是你现在「论文」目录的结构。

需要上传到服务器的只有新文件：
- `crf_sam3_v2/` 整个目录；
- `ground_v2/gv2/build_refytvos.py`；
- `ground_v2/tests/test_build_refytvos.py`。

服务器上已有的其他文件不要覆盖。路径沿用 `ground_v2/scripts/env.sh`，Ref-YTVOS 和 SAM3 的路径已按你之前的脚本写好。

## 命令（在 `crf_sam3_v2/` 下运行）

```bash
bash run_v2.sh test                      # 1. CPU 自检，约 1 分钟
bash run_v2.sh inventory                 # 只读盘点：各数据集训练集规模、Ground/Verify 实际用了多少、已有权重和磁盘，结果在 runs/inventory.md
bash run_v2.sh audit                     # 只读审计已构造的训练集：帧对齐、mask_dict 表达式顺序（PNG 判定）、Ground jsonl、Verify 缓存标签；AUDIT_ARGS=--leak 加跨数据集泄漏检查。结果在 runs/audit/trainsets.md

# Ref-YouTube-VOS
GPU=2,3 bash run_v2.sh refytvos ground   # 2. 训练 Ref-YTVOS 的 Ground v2（两卡，和 ReVOS 那次同一套超参）；只有一张卡就写 GPU=2
GPU=2 LIMIT=3 bash run_v2.sh refytvos    # 3. 试跑前 3 个视频（几分钟），确认能跑通
GPU=2 bash run_v2.sh refytvos            # 4. 全量：热力图约 30 分钟，主线约 2 小时，然后评测

# ReVOS（Ground v2 已训好、热力图已导出，会直接复用）
GPU=2 bash run_v2.sh revos               # 主线 10 小时以上，然后评测
```

- 训练用的超参和你训 ReVOS 版时的 `launch_revos_gpu23.sh` 完全相同，等效 batch 都是 8（两卡各累积 4 步，单卡累积 8 步）。Ref-YTVOS 训练集 11,619 条，ReVOS 是 9,371 条，所以训练时间约是 ReVOS 那次的 1.2 倍，适合放一晚上。
- 每个数据集的全流程按 ground → heat → run → eval 顺序执行，已完成的步骤会自动跳过。
- 也可以只跑其中一步：`bash run_v2.sh refytvos heat|run|eval`。
- 中断后重跑同一条命令会接着跑。**训练例外**：训练中断后会从头开始。

## 第 2 步会先检查一个数

数据构建结束后，脚本会打印「v1 目标掩码与 Annotations PNG 的平均 IoU」，正常应接近 1。

低于 0.9 时脚本会自动停下。这说明 v1 训练清单（mask_dict.pkl 按表达式顺序取掩码）和 PNG 标注对不上目标，这时把 `ground_v2/data/refytvos_v2/summary_train.json` 发给我。

另外两个值得看的数：
- `records_with_neg_ratio`：有别类负样本的记录占比。Ref-YTVOS 每个物体都有官方类别，所以应明显高于 ReVOS。
- 训练后的 `runs/gv2_refytvos/eval_val.json`：留出视频上的词正确率和热力图命中率。

## 输出

```
runs/<数据集>/
  ground/            Ground v2 在 valid 上的词（categories.json）和热力图（npz/）；ReVOS 直接复用已有的
  scores.jsonl       每条表达式：词、num_targets、候选、点种子、Verify 分数、两行各自的选择、候选池 J
  words/  final/     两行预测（Ref-YTVOS：Annotations/ 下的 PNG；ReVOS：results.json）和评测结果
  summary.md         结果表，最后把它发给我
```

`summary.md` 中的对照行：
- **Ref-YTVOS**：论文主线（原写法，帧错位）；如果 `../refytvos_frame_fix` 跑过，还有「只修帧错位」一行。
- **ReVOS**：论文结果 59.74，以及上一轮拼接实验的 62.38。

## 和上一轮拼接实验的区别（ReVOS）

上一轮 62.38 的候选池是「旧词池 ∪ v2 词池 + 点种子」，两套词的候选都在里面。新主线只用 Ground v2 的词，这是论文里要写的干净设定，所以数字可能和 62.38 有小差别。

`p1a_spatial_recall/scripts/refyt_v2.sh` 已被本包取代，不用再跑。

## 选词重排（可选，WORD_K）

`WORD_K=2 bash run_v2.sh refytvos` 会在热力图之后多一步 `rerank`：Ground v2 用 beam search 取前 K 个词，
同一类别的不同叫法只留一个，然后对每个候选直接问「它是不是这句话的指代对象」，取 log p(Yes) − log p(No)，选最高的（`WORD_SCORE=rev` 改用反向打分 log p(表达式 | 3 帧, 目标是这个词)）。
没有分歧（只剩 1 个候选）的表达式不多算，结果和原来一样。K 固定（2 或 3），没有其他参数。
重排后的词存在 `ground/categories_k<K>_<打分方式>.json`，每条带 `greedy_category`、`candidates`、`changed`。
代码在 `v2/ground_rerank.py`。
