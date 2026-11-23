# ground_v2：重构后的 Ground 训练包

对应文档《Ground 模块重构方案（v2）》。独立包，不改仓库；只在评测现有 Ground（消融表 A 行）时读取仓库代码。
输出格式和现有 Ground 一致：类别词缓存 JSON 可直接给 CRFSAM3Pipeline 读，热力图 npz 可直接给 p1a 补召回读。

## 1. 改了什么（对照现有 `train_query_grounding_qwen3vl.py`）

| | 现有 Ground | v2（本包） | 代码位置 |
|---|---|---|---|
| 热力图的视觉特征 | 进 LLM 之前的 `inputs_embeds`（第 522 行） | LLM 第 L 层输出（默认 L=12），已经读过整句表达式 | `model.py` `_layer_hook` |
| 表达式特征 | 词表查表 + 从零训练的 2 层编码器（第 526 行） | 同一次前向里第 L 层的表达式 token 状态 | `model.py` `positions` |
| 查询 | 8 个，只监督最好的一个，推理取 MAX | 1 个，初值取帧后第一个 `<TGT>` 前那个 token 在第 L 层的状态 | `model.py` `heat_logits` |
| 负样本 | 所有非目标 token 权重相同 | 别类物体权重 3；同类其他实例不计入损失；其余背景权重 1 | `common.py` `frame_labels` |
| 帧 | 只用目标可见帧 | 20% 概率换入一张目标不可见帧（标签全 0） | `data.py` `frames_of` |
| 热力图影响类别词的方式 | 调制器缩放全部视觉 token 和 deepstack 特征 | 两个 `<TGT>` 摘要 token：热力图加权平均、top-16 平均，经零初始化 MLP 加到 `<TGT>` 上；p 截断梯度；训练时 30% 换成空向量 | `model.py` `_layer_hook` |
| LLM | 完全冻结 | LoRA（r=16，只加在语言模型的线性层，视觉塔不动） | `model.py` `attach_lora` |
| 分辨率 | 训练用原图；ReVOS 推理用 336 | 训练、推理都用原图（`--max-image-side 0`） | `data.py` / `predict.py` |

**一次前向**：表达式 → 帧 → `Target evidence: <TGT><TGT>` → 输出要求 →（训练时）答案。第 L 层的 hook 算出热力图，把摘要写进两个 `<TGT>`，后面的层照常运行并生成 `{"category": ...}`。

**损失**：`1.0 × (加权 BCE + Dice) + 0.5 × 类别词交叉熵`。

## 2. 测试验证了什么

`bash scripts/00_test.sh`，CPU 上约 1 分钟，不需要权重。测试用的是随机初始化的小 Qwen3-VL，检查的是设计本身，不只是代码能跑：

| 测试 | 验证的设计点 |
|---|---|
| `test_labels` | 标签优先级：目标 > 别类（权重 3）> 同类（权重 0）> 背景 |
| `test_gradient_routing` | 只用 mask 损失时，梯度到达第 1 到 L 层的 LoRA，第 L+1 层以后为 0。只用类别损失时，热力图头梯度为 0（detach 生效），摘要 MLP 有梯度。两种情况都开着梯度检查点 |
| `test_hook_writes_tgt_and_generate_prefill` | hook 只改了两个 `<TGT>` 位置；`generate()` 时 hook 只在预填充阶段算一次热力图，结果和普通前向一致 |
| `test_build_data` | 别类负样本、同类忽略区、目标缺席帧的构造（同义词 woman 归为 person） |
| `test_save_load` / `test_trainer_smoke` | 经过 HF Trainer 训练两步、保存、加载 |
| `test_logging_and_nan_guard` | 日志里有各项损失的平均值并能画图；人为制造 NaN 后，训练在那一步停下，那一步不存 checkpoint |

本地环境：transformers 5.23、torch 2.15、peft 0.21。如果服务器的 transformers 是 4.x，decoder 层返回 tuple，hook 已经兼容；`00_test.sh` 能通过就说明没问题。

## 3. 在服务器上的步骤

先改 `scripts/env.sh` 里的路径（已经按你之前的脚本填好）。所有命令都在 `ground_v2/` 目录下运行。

```bash
bash scripts/00_test.sh            # 1. CPU 测试
bash scripts/01_build_data.sh      # 2. 生成 v2 数据（CPU，几分钟）
bash scripts/02_train_smoke.sh     # 3. 冒烟：20 步，看显存和 loss
bash scripts/03_train.sh           # 4. 正式训练（消融表 E 行），2 个 epoch；另开终端随时 bash scripts/plot.sh 看曲线
bash scripts/04_eval.sh v1                     # 5a. 消融表 A 行：现有 Ground
bash scripts/04_eval.sh runs/gv2_revos         # 5b. E 行
SUFFIX=_nosummary bash scripts/04_eval.sh runs/gv2_revos --drop-summary   # 5c. E' 行
bash scripts/05_predict_revos_valid.sh runs/gv2_revos   # 6. ReVOS valid 全量推理，接入下游
```

**第 2 步要先看一个数**：`data/revos_v2/summary_train.json` 里的 `records_with_neg_ratio`。它是"有别类负样本的记录"所占比例。
- 接近 0：说明你的数据里同一视频的其他物体拿不到类别，改动二的别类负样本不起作用。
- 物体的类别来自同一视频里指向这个物体的单目标表达式（API 给的类别词）；没有这类表达式的物体按普通背景处理。
- 同义词不够时，用 `--synonyms my_syn.json`（格式 `{"person": ["man", "woman", ...]}`）补充，内置表在 `common.py` 的 `DEFAULT_SYNONYMS`。

**第 3 步看什么**：
- 日志里 `[gv2] {...}` 一行包含 `loss_mask`、`loss_bce`、`loss_dice`、`loss_cat`。前 20 步里 `loss_mask` 应该下降。
- 用 `nvidia-smi` 看峰值显存。

## 4. 消融表每一行怎么跑

单查询在所有行里都打开，因为 v2 没有重新实现 8 查询。

| 行 | 含义 | 训练时追加的参数 | 输出目录 |
|---|---|---|---|
| A | 现有 Ground | 不训练，`04_eval.sh v1` | — |
| B | 第 L 层特征，LLM 冻结 | `--no-lora --neg-weight 1 --absent-prob 0 --no-summary` | `OUT=runs/gv2_B` |
| C | + LoRA | `--neg-weight 1 --absent-prob 0 --no-summary` | `OUT=runs/gv2_C` |
| D | + 别类加权、缺席帧 | `--no-summary` | `OUT=runs/gv2_D` |
| E | + 目标摘要 token（完整 v2） | 无 | `runs/gv2_revos` |
| E′ | E 推理时去掉摘要 | 不训练，`04_eval.sh runs/gv2_revos --drop-summary` | — |

例：`OUT=runs/gv2_C bash scripts/03_train.sh --neg-weight 1 --absent-prob 0 --no-summary`，然后 `bash scripts/04_eval.sh runs/gv2_C`。

B 行 LLM 冻结且没有摘要 token，所以它的类别词就是冻结模型的零样本词，只有热力图被训练。

`04_eval.sh` 输出的 `summary` 字段对应表中的前五列：
- `word_acc`：类别词正确率
- `word_acc_multinoun`：多名词子集上的类别词正确率。多名词子集指表达式里出现 2 个及以上已知类别词的样本，例如 "a man ... a monkey"
- `mass_in_gt`：热力图落在 GT 上的占比
- `mass_on_other_cat`：热力图落在别类物体上的占比
- `peak_in_gt`：热力图峰值是否落在 GT 上

最后一列"补召回后候选池上限"：用第 6 步的两个输出跑你现有的 SAM3 召回和 p1a 补召回。

**关于 A 行的分辨率**：按 `build_revos_query_grounding_from_category_cache.py`，v1 训练用的是原图；ReVOS 推理脚本却用了 `--max-image-side 336`。所以 A 行按原图评测，才是 v1 自己的训练设置。如果你想同时报 336 的结果，加 `--max-image-side 336` 再跑一次。

## 5. 接入下游

- **类别词**：`runs/gv2_revos/revos_valid_categories.json`。格式和 `predict_query_grounding_revos_meta.py` 的输出一致，在 CRF-SAM3 配置里替换原来的 category cache 路径即可。
- **热力图**：`runs/gv2_revos/heat_revos_valid/<video>/<exp>.npz`。字段和 `p1a.heatmap_dump` 一致：
  - `responses` 的形状是 (1, N)，只有一张图，所以对它取 MAX 也就是它本身
  - 网格约为每帧 23×40，旧的是 6×11
  - **p1a 里的阈值和峰值个数是按旧图定的，需要重新调**
- **Verify 不用改代码**，但候选变了，要在新候选上重新评测。

## 6. 显存与多卡

单卡 5090（32G）预计峰值 16–20 GB（估算，未实测）。类别损失只对答案位置计算 logits，不再为整条序列算全词表。
两张卡做数据并行（每卡一份完整模型，时间约减半）：
`$PY -m torch.distributed.run --nproc_per_node 2 -m gv2.train ...（参数同 03_train.sh）--gradient-accumulation-steps 4`

显存不够时（按影响从小到大）：

1. `--lora-layers 0-11,30-35`：LoRA 只加在前 L 层和最后几层。
2. `--max-frames 2`：每条训练样本最多 2 帧（推理仍用 3 帧，训练和推理会有轻微差异）。
3. `--max-image-side 960`：训练和推理都要设成同一个值。

## 6.5 训练监控（防训崩）

**训练时会记录什么**：每 `--logging-steps`（默认 10）步打印一行 `[gv2] {...}`，`03_train.sh` 同时写进 `runs/gv2_revos.log`。每行包括：

| 字段 | 含义 | 正常的样子 |
|---|---|---|
| `loss` | 总损失 = 1.0×loss_mask + 0.5×loss_cat | 前几百步明显下降，之后缓慢下降 |
| `loss_mask`（`loss_bce`、`loss_dice`） | 热力图损失 | 稳步下降；Dice 一直贴近 1 说明热力图没学到形状 |
| `loss_cat` | 类别词交叉熵 | v1 已经会说词，起点就不高，很快降到低位 |
| `grad_norm` | 裁剪前的梯度范数（`--max-grad-norm 1.0` 裁剪） | 大致平稳；偶尔尖峰正常，持续升高要警惕 |
| `learning_rate` | 3% warmup 后线性下降到 0 | |
| `eval_loss`、`eval_loss_mask`、`eval_loss_cat` | 每 500 步在 200 条验证集上算一次 | 跟训练损失一起降；训练降、验证升就是过拟合 |

这些数都是两次打印之间所有 micro-batch 的平均值，不是单个样本。

**看曲线**：训练进行中随时运行

```bash
bash scripts/plot.sh                 # 读 runs/gv2_revos.log，画到 runs/gv2_revos.png，并打印健康检查
```

健康检查会给出每个损失「前 20% 均值 → 后 20% 均值」及趋势（down / flat / UP）、grad_norm 中位数和尖峰个数、每次 eval_loss 和最好的一步。也可以把几行消融放在一张图上比：`bash scripts/plot.sh runs/gv2_revos.log runs/gv2_rowB.log --labels E B --out runs/E_vs_B.png`。

想用 TensorBoard 的话，在 `03_train.sh` 后面加 `--report-to tensorboard`（服务器需要装 tensorboard），日志写在输出目录的 `runs/` 下。

**自动防护**：任何一步训练损失出现 NaN/Inf，训练在这一步结束后立刻停止，这一步的 checkpoint 和最终保存都取消，磁盘上已有的 checkpoint 都是出问题之前的。日志里会有 `[gv2] WARNING non-finite ...` 和 `[gv2] STOPPED ...`，进程以退出码 2 结束。

**出问题怎么办**：

| 现象 | 处理 |
|---|---|
| NaN 自动停止 | 用更小的学习率重跑：`bash scripts/03_train.sh --learning-rate 5e-5`；仍然出现再加 `--max-grad-norm 0.5` |
| grad_norm 持续变大、loss 开始上升 | 先停掉，同上降学习率 |
| loss_mask 几乎不动 | 检查 `01_build_data.sh` 的 summary；试 `--feature-layer 8` |
| loss_cat 升高（类别词变差） | 调高 `--lambda-cat`（如 1.0） |
| 训练损失降、eval_loss 升 | 用 eval_loss 最低那一步附近的 checkpoint（`runs/gv2_revos/checkpoint-*`） |

步数参考：9,371 条训练样本，batch 1 × 梯度累积 8，每个 epoch 约 1,170 步，2 个 epoch 约 2,340 步；默认每 500 步存一次 checkpoint、评一次验证集。

## 7. 主要参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--feature-layer` | 12 | 从第几层输出取特征（从 1 开始数，等于 HF `hidden_states[L]`）。探针里第 8–13 层最好，可以试 8、12、16 |
| `--neg-weight` | 3.0 | 别类物体的权重 |
| `--absent-prob` | 0.2 | 换入目标缺席帧的概率 |
| `--summary-drop` | 0.3 | 训练时把摘要换成空向量的概率 |
| `--topk` | 16 | 第二个摘要取热力图最高的多少个 token |
| `--lambda-mask / --lambda-cat` | 1.0 / 0.5 | 沿用 v1。类别词仍弱时调高 `lambda-cat` |
| `--lora-r / --lora-alpha` | 16 / 32 | |
| `--num-train-epochs` | 2 | v1 不到 1 个 epoch |
| `--learning-rate` | 1e-4 | LoRA 和热力图头共用；训崩时先降到 5e-5 |
| `--max-grad-norm` | 1.0 | 梯度裁剪 |
| `--logging-steps / --eval-steps / --save-steps` | 10 / 500 / 500 | 打印、验证、存 checkpoint 的间隔 |
| `--report-to` | none | 设为 tensorboard 时额外写 TensorBoard 日志 |

## 8. 已知限制

- 这是 CPU 小模型上验证过的实现，**还没有在真实权重上跑过**。第 3 步冒烟是第一次真实运行。
- 类别标签来自 API 伪标注；同义词表决定"同类"和"别类"怎么划分，会直接影响负样本。
- 只覆盖 ReVOS（数据构造依赖 `meta_expressions` 和 `mask_dict`）。Ref-YTVOS、MeViS 要等 ReVOS 上消融表结论成立后再扩展。
- 训练要求 batch size 为 1 或很小：hook 里按样本逐个计算热力图，batch 大了不会出错，但也不会更快。
