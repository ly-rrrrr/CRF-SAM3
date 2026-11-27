# CRF-SAM3 训练集构造规范

Nov 26, 2026 · @yea

## 原则与范围

四个数据集共用一套构造规则。训练前先跑只读审计，FAIL 项修完才进训练。

- 样本单位是一条表达式，不是 Sa2VA 的「一个视频 + 5 条表达式」。
- Ground 训练用 ReVOS 和 Ref-YTVOS，因为它们有物体类别标签。MeViS 没有类别，不进。DAVIS 看审计 C 节，有类别才进。
- Verify 训练用 MeViS、DAVIS、Ref-YTVOS、ReVOS 四个数据集，训出一份权重。
- 只用官方 train。valid 和 test 只做评测。
- 每份产物（jsonl、缓存、排除表）都带 manifest。规则改了就整份重建，不在旧产物上打补丁。
- 训练缓存和推理同口径：同样的词来源、召回设置、点种子。
- 训练用两张 5090（GPU2、GPU3）。

## 数据源与索引约定

帧的位置一律是「排序后 JPEG 列表里的下标」，任何地方都不能用 int(帧名) 当位置。路径都在 `video_datas/` 下。

| 数据集 | 路径 | meta | GT 存法 | mask 的 key | 帧名 | 调色板 PNG | 物体类别 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MeViS | `mevis/train` | `train/meta_expressions.json` | `mask_dict.json` | anno\_id（全局唯一） | 连续 | 无 | 无 |
| Ref-DAVIS17 | `davis17/train` | `meta_expressions/train/meta_expressions.json` | `mask_dict.pkl` | anno\_id，没有就退回 obj\_id（待审计 B） | 连续 | 有 | 待审计 C |
| Ref-YTVOS | `Ref-Youtube-VOS/train` | `train/meta_expressions.json` + `train/meta.json` | `mask_dict.pkl` | 全局表达式序号 | 每 5 帧一张：00000、00005… | 有 | `meta.json` |
| ReVOS（Verify） | `revos_prepared/train`（软链） | `train/meta_expressions.json` | `mask_dict.json` | anno\_id | 待审计 A | 无 | API 类别缓存 |
| ReVOS（Ground） | `revos/` | `meta_expressions_train_.json` | `mask_dict.json` | anno\_id | 待审计 A | 无 | API 类别缓存 |

- meta 的 frames 必须和 JPEG 列表一一相等（个数、名字、顺序），否则 SAM3 输出和 GT 对不上帧。
- SAM3 输出、候选 mask 文件名、GT 序列都按这个位置。
- 多目标表达式（MeViS、ReVOS 的 anno\_id 列表）保留两份 GT：逐帧并集，和每个目标各自的 mask（第 5 节打标签要用）。
- 所有构造脚本共用一个取 GT 的函数，不再各写各的。

## 审查现有构造代码发现的问题

逐行读了 Ground v1/v2 的构造脚本、`crf_sam3_attn/dataset.py` 的缓存生成和读取、损失里的正样本处理，以及 Sa2VA 的加载器。第 1 条最要紧，但还只是推断，要审计 B 节用 PNG 判定。

| # | 问题 | 影响 | 规范里的处理 |
| --- | --- | --- | --- |
| 1 | Ref-YTVOS 的 mask\_dict 按全局表达式序号存。我们三处代码（v1 Ground 构造、Verify 缓存、继承 v1 的 Ground v2）按 json 原顺序数序号。Sa2VA 加载器按「视频原顺序 + 表达式 id 字符串排序」数，"10" 排在 "2" 前面。mask\_dict.pkl 来自 Sa2VA 数据，大概率是后一种 | 只影响 ≥11 条表达式的视频里 id≥2 的表达式。若成立，它们拿到同视频另一条表达式的 GT，常常是另一个物体：Ground 学错目标，Verify 正负标签颠倒 | 审计 B 用 Annotations PNG 判定哪种顺序对，写进唯一的取 GT 函数。build\_refytvos 的 target\_png\_iou\_mean 只看均值，少量错条会被平均掉，门槛改看 below\_0.9 的条数 |
| 2 | `crf_sam3_demo.py::load_gt_masks` 用 int(帧名) 当帧位置，和已修的评测错帧是同一类 | 现在的配置都走 mask\_dict，没触发；一旦有人改用 gt\_masks\_root，Ref-YTVOS 会整体错帧 | 新包不用这条路径 |
| 3 | 缓存算候选 J 时，GT 为空值的帧（目标不在画面）不进分母 | 候选在目标消失的帧里跟着别的物体不扣分，评测时这些帧记 0，标签和评测不一致 | 统一用评测口径（第 5 节）；审计 E 报告会翻多少标签 |
| 4 | force\_pos：没有候选达到正样本阈值时，训练把 J 最高的硬当正样本，哪怕 J 接近 0 | DAVIS/YTVOS/ReVOS 靠排除表挡住；MeViS 没有排除表。本地缓存样例「Goat moving forward」6 只羊 J 全是 0，仍会被硬选一只 | 没有正样本的表达式不进训练，记进清单；关掉 force\_pos |
| 5 | 排除表（DAVIS 123、YTVOS 1,149、ReVOS 1,763 条）是按旧词、无点种子的候选池算的 | 换成新池后哪些表达式没有正样本会变 | 新缓存生成时自己算，不沿用旧表 |
| 6 | 正负阈值各不相同：MeViS 0.2/0.05，其余 0.5/0.1。MeViS 用低阈值是因为多目标：单个候选对并集的 J 天然低 | 论文里要解释两套阈值，而且 0.2 会把半对的候选当正样本 | 多目标按「和任一单个目标的 J 的最大值」打标签，四个数据集统一 0.5/0.1 |
| 7 | 长视频处理不一致：MeViS/DAVIS/YTVOS 缓存超过 300 帧整条跳过，ReVOS 不限 | 长视频在评测里有，训练里没有 | 统一不限；显存不够的记进跳过清单并报告条数 |
| 8 | listwise\_max\_candidates=10 时按 GT IoU 排序截断候选 | 现在保留 7 个，不触发。加点种子后池子可能超过 10，会按标签挑候选，推理没有这一步 | 上限设为池子最大值；必须截断时按 SAM3 分数和 |
| 9 | 训练 F=4；推理 MeViS/DAVIS/YTVOS 用 8 帧，ReVOS 用 4 帧 | 训练和推理看到的帧数不同 | 统一 F=4（已定）。帧选法两边已经一致：每个候选在自己出现的帧上均匀取 |
| 10 | 缓存的词来自 prompt 提取器（ReVOS 用 API 类别，num\_targets 固定为 1），推理用 Ground 词 | 训练和推理的候选池来源不同 | 缓存一律用交叉拟合的 Ground v2 词；num\_targets 来源和推理一致 |
| 11 | 跨数据集泄漏没查过：ReVOS 里有 MOSE 等来源的视频，MeViS 也来自 MOSE | 联合训练后评测视频可能在别的训练集里出现过。Sa2VA 也是这样联合训练的，但我们要知道数字 | 审计 F 列出首帧几乎相同的视频对，确认后排除 |

另外两点不算问题，但要知道：Ref-YTVOS 同一物体常有多条表达式，候选池完全相同（本地缓存 003234408d 的 exp 0 和 1）；Sa2VA 每个样本抽 5 条表达式是有放回的。

## 划分与泄漏

一份 splits manifest 管全流程，按视频切，不按表达式切。

- train\_holdout：每个数据集官方 train 按视频切出 10%（seed 42）。Ground 的 val、Verify 调参、各种阈值都只在它上面定。v1 的 jsonl 已经是按视频 val\_ratio 0.1、seed 42 切的，ReVOS 和 Ref-YTVOS 直接沿用它的 val 视频；MeViS 和 DAVIS 用同样规则新切。
- 交叉拟合：去掉 holdout 后，Ground 训练集按视频切两折。Verify 缓存里 ReVOS 和 Ref-YTVOS 的训练视频，词和热力图来自没见过它的那一折。
- MeViS（及没有类别的 DAVIS）不进 Ground 训练，缓存直接用全量联合 Ground 的输出，和推理同口径。
- holdout 视频也生成缓存，只用来调参，不进训练。
- 任何评测集的视频都不能出现在任何训练集里。审计 F 列出首帧几乎相同的视频对，人眼确认后写进排除清单。

## GT 与 J 的统一定义

候选的 J 和官方评测的 J 用同一个定义：所有标注帧都算，目标不在画面时 GT 是空 mask。

```latex
J(M, G) = \frac{1}{T} \sum_{t=1}^{T} \mathrm{IoU}(M_t, G_t), \qquad \mathrm{IoU}(\varnothing, \varnothing) = 1
```

- T = 这个视频的全部标注帧。候选在某帧没有 mask 就记空。
- GT 和预测尺寸不同时，GT 最近邻缩放到预测尺寸（和现在一样）。
- 多目标表达式算两个数：J\_union 是候选对目标并集，只用于报告；J\_best 是候选对每个单个目标的 J 取最大，用于打标签。单目标时两者相等。
- 标签：J\_best ≥ 0.5 为正，≤ 0.1 为负，中间不进损失。四个数据集同一对阈值。
- 最终输出取前 num\_targets 个候选，所以多目标时「对上其中一个目标」就该是正样本，和推理一致。

## Ground 训练集

Ground 的 jsonl 沿用 v2 的字段，只是 GT 改由唯一的取 GT 函数提供，并把抽查门槛写死。

| 字段 | 规则 |
| --- | --- |
| 数据集 | ReVOS、Ref-YTVOS；DAVIS 有类别才进；MeViS 不进 |
| category | Ref-YTVOS 用 `train/meta.json`；ReVOS 用 API 类别缓存；都过同一个 CategoryNormalizer 归组 |
| 帧 | 目标可见的帧（面积 ≥ min\_area）上均匀取 3 帧，原分辨率 |
| 目标 mask | 多目标取并集，类别用 API 给的那个词（v1 现状） |
| neg | 同视频里类别组不同的其他物体的并集 |
| ign | 同类别组的其他物体，不进损失 |
| absent\_frames | 目标不在画面的帧，最多 4 帧 |
| ReVOS 物体类别 | 指向它的单目标表达式的类别多数票；没有的记 unknown，当普通背景 |
| 划分 | train / val（= holdout 视频），另出 fold0 / fold1 的视频清单 |
| 抽查 | 有 PNG 的数据集逐条比 target IoU；IoU < 0.9 的条数占比 ≤ 0.5%，否则停下 |

ReVOS 的 Ground 数据现在只覆盖 157 个训练视频（API 类别缓存的范围），配比时按实际进数据的视频数算。

## Verify 训练缓存

训练缓存的候选池和推理走同一条流程，只是多算每个候选的 J 和标签。旧缓存和旧排除表只作对照，不进新训练。

1. 取词：交叉拟合的 Ground v2 词（MeViS 用全量联合 Ground）。不加备选词，不做语义扩展。指代检查（WORD\_K）默认关。
2. 召回：SAM3-DMS 召回 24 个，按逐帧 SAM3 分数之和保留 7 个。num\_targets 来源和推理一致。
3. 点种子：用同一折 Ground 的热力图，按推理同样的触发规则加种子候选。
4. 存盘：每个候选逐帧 mask 按帧位置命名；meta 里记 J\_union、J\_best、来源（word / seed）、SAM3 分数和。
5. 标签：按第 5 节。没有正样本的表达式不进训练，原因记 no\_positive；不再用 force\_pos。
6. 跳过：缺视频、GT 为空、显存不够，全部记原因进 manifest。不设帧数上限。

训练读取时：F=4，每个候选在自己出现的帧上均匀取；listwise 上限不小于池子最大值，必须截断时按 SAM3 分数和。同一 (视频, 词) 的 SAM3 召回可以共用缓存，只省时间，不改结果。

## 联合配比：参考 Sa2VA

现在的 repeats 确实不正常。Verify 联合配置是 MeViS 1 / DAVIS 4 / Ref-YTVOS 2，没有 ReVOS（ReVOS 是另起的 warmstart 训练）；Ground v1 的联合训练是直接拼接，没有 repeats。Sa2VA 的数字也不能照抄，因为采样单位不同。

|  | Sa2VA | 我们 |
| --- | --- | --- |
| 一个样本 | 1 个视频，有放回抽 5 条表达式、5 帧 | 1 条表达式 |
| 一个 epoch 的样本数 | 视频数 × repeats | 表达式数 × repeats |
| repeats | ReVOS 10、MeViS 4、Ref-YTVOS 4、Ref-SAV 4 | 按下式换算 |
| DAVIS | 不训练，零样本评测 | 进 Verify 训练（已定） |
| 训练长度 | 1 epoch | 1 epoch |

来源：上游 `projects/sa2va/configs/sa2va_in30_8b.py` 和本地 `sa2va_qwen3_4b.py`，数据集类 `sa2va_data_03_refvos.py`（`__len__ = 视频数 × repeats`）。

ReVOS 得到 ×10 是因为它视频少、每个视频表达式多。换算规则：让每个数据集在一个 epoch 里被看到的表达式次数占比和 Sa2VA 相同，再整体缩放到最小的为 1 并取整。V 是视频数，E 是表达式数，R 是 Sa2VA 的 repeats。

```latex
r_d = \mathrm{round}\left( c \cdot \frac{5\, V_d\, R_d}{E_d} \right), \qquad c = \left( \min_{d} \frac{5\, V_d\, R_d}{E_d} \right)^{-1}
```

- DAVIS 不在 Sa2VA 里，取 R = 4（Sa2VA 给 MeViS 和 Ref-YTVOS 的值）。这是我们自己的选择，论文里要写明。
- Ground 用同一公式，只对进 Ground 的数据集算，V、E 按实际进数据的条数。
- 例：Ref-YTVOS 约 3,471 个视频、12,913 条表达式，5 × 3,471 × 4 / 12,913 ≈ 5.4。它每个视频表达式少，换到表达式级会得到较高的 repeats。
- 其余 V、E 等 inventory 的数字到了再填。一个 epoch 的总样本数 = Σ E\_d · r\_d，按它估计两张 5090 上的时间。
- 只照搬 Sa2VA 的组织方式，超参不照搬：Ground 用 ReVOS v2 的配方，Verify 用现在的配方。

## 清单与审计门槛

每份产物旁边放一个 manifest.json，训练前跑审计，下表任一项不过就不训练。

manifest 记：源文件路径、大小和修改时间；构造脚本名、参数和包的日期；用哪份 splits；每个数据集的条数和各跳过原因的计数；词和热力图来自哪一折的 Ground 权重。

| 检查 | 门槛 | 审计节 |
| --- | --- | --- |
| meta frames 和 JPEG 对齐 | 不一致的视频 0 | A |
| mask\_dict 表达式顺序 | PNG 判定的顺序 = 取 GT 函数用的顺序 | B |
| mask\_dict 对 PNG | DAVIS、Ref-YTVOS mean IoU ≥ 0.95 | B |
| Ground jsonl 帧路径和帧名 | 错 0 条 | D |
| train 和 holdout 视频重叠 | 0 | D |
| Verify 缓存字段和 mask 文件 | 错 0 条 | E |
| 重算 J 对缓存 | 最大误差 ≤ 0.02 | E |
| 没有正样本却进了训练 | 0 条 | E |
| 跨数据集泄漏 | 人眼确认后 0 | F |

运行：`bash run_v2.sh audit`，加 `AUDIT_ARGS=--leak` 做泄漏检查；或者直接 `python tools/audit_trainsets.py --repo $REPO --revos-root $REVOS_ROOT --gv2 ../ground_v2`。报告在 `runs/audit/trainsets.md`，受影响的条目清单在同目录的 json 里。

## 下一步

1. 你在服务器跑审计和 inventory（都只读、只用 CPU），把 `runs/audit/trainsets.md` 和 `runs/inventory.md` 发我。MeViS 的 mask\_dict 加载最慢。
2. 按审计 B 节的结论定唯一的取 GT 函数，写进新包的构造脚本。
3. 用 inventory 的数字算出各数据集的 r\_d，填进第 8 节，并估算两张 5090 的训练时间。
4. 再写分阶段的完整脚本：splits → Ground 两折 + 全量 → Verify 缓存 → Verify 联合训练 → 四个数据集推理和评测。

开放问题：DAVIS 的 meta 里有没有物体类别（决定它进不进 Ground），等审计 C 节。

## 落地时的补充决定（11-26）

写完整脚本（`run_full.sh`）时定下或改动的几处，以这一节为准：

- **划分**：不重切。Ref-YTVOS 和 ReVOS 的 holdout 就是 v1 Ground 的 val 视频，原样保留。v1 没覆盖的视频（ReVOS 约 451 个没有类别的视频，MeViS、DAVIS 的全部视频）再按 10%（seed 42）补进 holdout。
- **原始数据不动**：只重建我们自己生成的 Ground jsonl 和 Verify 缓存。取 GT 的索引规则照搬 Sa2VA 加载器 `sa2va_data_03_refvos.py::json_file_preprocess`，复现的人按 Sa2VA 的说明下载数据即可。
- **第 12 个问题：Verify 提示词训练和推理不一致**。训练用 `crf_sam3_attn/model.py::_build_prompt`，帧逐张作为图片输入，对应推理的 `yesno_images` 加 `discrimination_prompt_yesno_train_aligned_20260806.txt`。论文旧结果里只有 ReVOS 是这样推理的，Ref-YTVOS、DAVIS、MeViS 用的是另一句提示词，帧当成视频输入。现在四个数据集统一用 `yesno_images`，F = 4（`tools/make_mainline_yaml.py`）。
- **v2 主线的一个 bug**：`v2/mainline.py` 原来用 Ref-YTVOS 运行器查提示词模板，那张表里没有 `yesno_images`，ReVOS 会退回 CoT 模板。已改为用 ReVOS 运行器的表（它是前者的超集）。
- **Ground 的 repeats**：一条表达式的 r 份拷贝各自抽不同的 3 帧（第 0 份均匀取，其余在均匀分箱里随机取），对应 Sa2VA 每次随机抽帧。
- **Verify 缓存怎么落到训练里**：缓存的 `iou_map` 直接写 J_best（评测口径），所以训练器自带的 pos_iou 0.5 / neg_iou 0.1 就是规范的标签。没有正样本的表达式不写 meta.json，训练器就不会采到它，force_pos 也就不会触发，同时在配置里关掉。原因记在 `done_*.jsonl` 里。holdout 的缓存保留全部表达式，只用于调参。
- **交叉拟合用哪份 Ground**：进过 Ground 训练的视频用没见过它那一折的模型（cf0 / cf1），其余视频（holdout、没有类别的 ReVOS 视频、MeViS、DAVIS、四个验证集）都用 full。
- **Verify 训练**：仓库的 `train_resume.py` 是单卡的，新包里的 `v2/train_verify_ddp.py` 做两卡数据并行：每卡一份完整模型，LoRA 梯度在 optimizer step 前取平均，checkpoint 格式和 `train_resume.py` 相同，可以断点续训。默认等效 batch 是 2（两卡各 1），lr 沿用旧配方的 1e-5。
