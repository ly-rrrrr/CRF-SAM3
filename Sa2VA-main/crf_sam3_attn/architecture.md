# CRF-SAM3 + Qwen3-VL 完整架构设计书（当前实现对齐版）

- 文档版本: v2.0
- 更新日期: 2026-04-02
- 适用目录: `Sa2VA-main/crf_sam3_attn`
- 用途: 记录当前方案、代码结构、训练配置与实验演进，便于后续接续与复现。

---

## 1. 问题定义与目标

### 1.1 任务定义
输入视频序列与自然语言指代表达式，系统首先召回一组对象候选 tube，再在候选集合内完成语义重排，输出最终目标的全视频掩码序列，并以 `J / F / J&F` 进行评估。

### 1.2 当前关注的核心困难
1. 背景捷径:
   - VLM 容易根据整场景语义而不是候选本体给分。
2. 候选同质化:
   - 多个候选常属于同类对象，细粒度差异小。
3. 表达式的锚点歧义:
   - 模型可能把“场景 somewhere 满足表达式”误当成“该候选满足表达式”。
4. 多数据集联合训练的分布冲突:
   - 单一训练目标容易在提升泛化时牺牲单数据集专项能力。
5. 工程稳定性:
   - 候选数、帧数、在线 SAM3 调用都会带来时延和显存波动。

### 1.3 当前方案目标
1. 训练-推理同一评分口径。
2. 让正样本在候选集合内稳定胜出。
3. 通过 mask-aware 软约束抑制背景捷径。
4. 通过锚点式 prompt 强化“候选即主语”的判别语义。
5. 通过离线缓存与联合训练增强稳定性和泛化。

---

## 2. 系统总体架构

系统由三层组成：

1. 候选生成层
   - 概念抽取
   - SAM3 召回
2. 候选重排层
   - Qwen3-VL 判别器
   - LoRA 微调
   - Listwise 排序
   - Mask-aware 对齐
   - 违规负样本惩罚
3. 推理评估层
   - 批量数据集 runner
   - 结果导出
   - 指标评估

当前全流程是：

1. 从表达式抽取召回语义
2. 用 SAM3 召回候选 tube
3. 候选与 GT 对齐生成 IoU 监督
4. 基于离线缓存训练候选重排器
5. 导出 LoRA 或 merged 模型
6. 批量推理与评测

---

## 3. 核心代码模块与职责

### 3.1 `crf_sam3_attn/dataset.py`
职责：
1. 从表达式元信息构建训练样本
2. 支持 online 候选生成与 offline 缓存读取
3. 读取并组织候选 frames / masks / IoU
4. 构造 listwise 训练单元
5. 注入数据集特定的监督参数

当前关键行为：
1. 支持 `pos_iou / neg_iou`
2. 支持 `listwise_neg_policy`
3. 支持 `hard_neg_weight / hard_neg_margin / hard_neg_pos_strategy`
4. 对训练集缺 PNG GT 的情况保留回退逻辑

### 3.2 `crf_sam3_attn/model.py`
职责：
1. 加载 Qwen3-VL 与 Processor / Tokenizer
2. 注入 LoRA
3. 冻结视觉编码器（默认）
4. 统一锚点式 Yes/No prompt
5. 计算候选分数
6. 计算 `L_cls + L_rank + L_align`

当前关键点：
1. 推理主分数使用 `P(Yes)`
2. 难负样本排序约束内部使用 `yes_logit - no_logit`
3. `hard_neg_topk` 已经删除，不再采用 top-k 负样本

### 3.3 `crf_sam3_attn/model_resume.py`
职责：
1. 与 `model.py` 保持同一训练逻辑
2. 支持从已有 LoRA checkpoint 恢复训练
3. 用于 `train_resume.py` 的 true resume 路径

### 3.4 `crf_sam3_attn/train.py`
职责：
1. 读取训练配置
2. 在 `offline_prepare=true` 时生成缓存并退出
3. 构造训练数据集与 repeats 软平衡
4. 训练 LoRA 判别器
5. 打印 step 日志与 debug 监督信息
6. 保存 checkpoint / final / merged

### 3.5 `crf_sam3_attn/train_resume.py`
职责：
1. 从带 `training_state.pt` 的 checkpoint 真正续训
2. 恢复 optimizer state 与 RNG state
3. 保持与 `train.py` 一致的 debug 打印和保存逻辑

说明：
1. 只有带 `training_state.pt` 的目录才是 true resume checkpoint
2. 仅有 adapter 文件的目录只能 warm-start，不是严格 resume

### 3.6 `crf_sam3_attn/train_data.py`
职责：
1. 构建多数据集联合训练集
2. 支持 `datasets:` 方式混合多个数据源
3. 引入 `RepeatDataset` 做软平衡

### 3.7 数据集 runner
相关脚本包括：
1. `run_crf_sam3_dataset.py`（MeViS）
2. `run_crf_sam3_davis_dataset.py`
3. `run_crf_sam3_refytros_dataset.py`

当前特点：
1. 支持更完整的过程记录文件输出
2. DAVIS / MeViS 结果导出已补齐 `scores.json` 与 `expression_details`
3. 部分 runner 已支持单基座双模式推理

---

## 4. 候选生成阶段

### 4.1 概念抽取
候选生成前先从表达式与视频中提取：

1. `category`
2. `backup_categories`
3. `num_targets`

当前概念抽取的约束强调：
1. `category` 优先对应表达式中的主体名词
2. `backup_categories` 是安全回退机制，而不是随意属性短语
3. 显式出现的 generic object noun 不应随意改写为其描绘语义类别

### 4.2 SAM3 候选召回
用 `category` 作为召回提示词，得到：
1. 每帧实例掩码
2. 对象 ID
3. 候选 tube

### 4.3 候选监督构建
用 GT 掩码与候选 tube 对齐，得到：
1. `iou_map`
2. `pos_obj_ids`
3. `neg_obj_ids`

这些对象级监督进入离线缓存并用于后续训练。

---

## 5. 输入表示与锚点式判别语义

### 5.1 候选可视化输入
每个候选对象由若干采样帧表示。每帧将候选 mask 叠加到原图，构成“该对象已被显式标出”的视觉输入。

### 5.2 当前训练/推理 prompt
当前训练 prompt 已明确切换为锚点式表达：

1. 将被 mask / highlight 的对象视为表达式锚点主语
2. 判断当表达式 grounded 在该对象上时，整句是否成立

这比“候选是否匹配表达式”的弱定义更接近任务本质。

### 5.3 为什么这一步重要
很多错误并不是模型看不到目标，而是模型把：
1. 整个场景中的相关线索
2. 某个真正候选附近的线索
3. 当前被标出的候选

这三件事混在一起。锚点式 prompt 的目的就是减弱这种混淆。

---

## 6. 打分定义

### 6.1 推理主分数
当前推理输出分数为：

`s_i = P(Yes | expression, candidate_i)`

这是最终排序和结果记录使用的主分数。

### 6.2 排序辅助分数
为了让排序约束更稳定，难负样本损失内部使用：

`r_i = yes_logit - no_logit`

原因：
1. 概率容易饱和
2. logit gap 更适合表达相对胜负关系

### 6.3 训练-推理一致性
当前保持：
1. 推理输出仍是 `P(Yes)`
2. 训练主排序项 `L_cls` 也基于推理分数口径
3. 只有难负样本辅助项使用 logit gap

这是一个“主口径一致 + 辅助排序更稳”的折中设计。

---

## 7. 当前损失函数设计

### 7.1 Listwise 主分类损失
对同一表达式下全部候选一起计算：

`L_cls = logsumexp(s_j / tau, j in C) - logsumexp(s_j / tau, j in P)`

其中：
1. `C` 是全部候选
2. `P` 是正样本集合
3. `tau` 是温度系数

作用：
1. 优化候选集合内的相对排序
2. 不再把任务简化成独立 pointwise 分类

### 7.2 Mask-aware 软对齐损失
当前主用的是 feature-based 软对齐约束：

1. 提取文本 token 与视觉 token 表征
2. 计算文本与视觉 token 的相关性
3. 将候选 mask 下采样到 token 网格
4. 最大化语义质量落在 mask 内的比例

该损失的作用是：
1. 减少背景捷径
2. 强化“围绕候选读取视觉证据”
3. 保留必要上下文，不做硬裁剪

### 7.3 当前难负样本损失
这是近期最重要的改动之一。

当前逻辑：

1. 先根据正样本集合得到 `pos_ref`
   - 由 `hard_neg_pos_strategy` 控制
   - 例如 `max` 或 `mean`
2. 将所有满足
   - `neg > pos_ref - margin`
   的负样本视为违规负样本
3. 对所有违规负样本统一计算惩罚

对应思想是：
1. 不再只打 top-k hardest negative
2. 而是要求正样本整体上稳定胜出
3. `margin` 控制安全间隔

### 7.4 总损失

`L_total = L_cls + lambda_rank * L_rank + lambda_align * L_align`

其中：
1. `L_cls`：组内相对排序
2. `L_rank`：所有违规负样本惩罚
3. `L_align`：mask-aware 软对齐

---

## 8. 正负样本构造与多数据集差异

### 8.1 基本规则
候选监督由 IoU 构造：
1. `IoU >= pos_iou` -> 正样本
2. `IoU <= neg_iou` -> 负样本
3. 中间带候选保留但不一定作为显式负样本

### 8.2 `listwise_force_pos`
若当前批次没有正样本且开启 `listwise_force_pos`：
1. 用最大 IoU 候选兜底为正

这能避免整组训练样本失效。

### 8.3 多数据集特定设置
当前联合训练中，监督参数会按数据集下沉：

1. `MeViS`
   - `pos_iou` 更宽松
   - `listwise_neg_policy = threshold`
   - `hard_neg_pos_strategy = mean`
2. `DAVIS`
   - `pos_iou` 更严格
   - `listwise_neg_policy = all`
   - `hard_neg_pos_strategy = max`
3. `Ref-YTROS`
   - 与 DAVIS 类似，但保留自身分布适配

原因是：
1. `MeViS` 存在多目标表达
2. `DAVIS / Ref-YTROS` 更偏单目标细粒度区分

---

## 9. 参数高效微调策略

### 9.1 当前更新范围
当前仍以 LoRA 为主，仅更新语言侧 / 融合侧投影层：

1. `q_proj`
2. `k_proj`
3. `v_proj`
4. `o_proj`
5. `gate_proj`
6. `up_proj`
7. `down_proj`

### 9.2 冻结视觉编码器
默认 `freeze_visual_encoder = true`。

含义：
1. 不重训底层视觉表征
2. 主要学习“如何根据文本围绕锚点读取已有视觉 token”

优点：
1. 稳定
2. 成本低
3. 适合当前候选级重排任务

局限：
1. 局部细粒度视觉能力提升有限

---

## 10. 联合训练策略

### 10.1 为什么从串行微调转向联合训练
串行微调会导致：
1. 不同数据集的优化轨迹断裂
2. 后一阶段覆盖前一阶段偏好

联合训练则让多个数据集在同一轮优化中共同约束模型。

### 10.2 软平衡采样
当前不是 sampler 级严格平衡，而是采用 `repeats` 做软平衡：

1. 每个数据集可设置不同 `repeats`
2. 通过扩大有效样本长度调节采样比例
3. 再在总数据集上统一 shuffle

这是当前实现中最接近 `sa2va` 风格的联合训练方案。

### 10.3 当前 joint config 的特点
当前联合训练配置包含：
1. 多数据集 `datasets:` 拼接
2. 各数据集独立的 IoU 与 hard negative 策略
3. `debug` 详细监督打印
4. `save_merged` 导出完整模型

---

## 11. 训练流程

### 11.1 两阶段离线训练
当前训练通常分为：

1. Offline Prepare
   - 在线调用概念抽取与 SAM3
   - 生成离线缓存
2. Offline Train
   - 只读取缓存训练判别器

### 11.2 为什么采用离线缓存
优点：
1. 避免训练时同时常驻 SAM3 与 VLM
2. 显著降低时延波动
3. 更利于复现与对照实验

### 11.3 当前训练循环中已有的稳定性机制
1. Gradient checkpointing
2. `device_map=auto` 支持
3. OOM batch skip
4. 更详细的 step 监督打印

### 11.4 当前 debug 打印
训练时可打印：
1. 当前表达式
2. `pos / neg / middle` 候选分布
3. 候选 `obj_id / IoU / score`
4. top scores
5. 当前训练 prompt

这对诊断 hard negative 是否真的在压高分负样本非常关键。

---

## 12. true resume 与 warm-start 区别

### 12.1 true resume
`train_resume.py` 依赖：
1. adapter checkpoint
2. `training_state.pt`

这样才能恢复：
1. optimizer state
2. RNG state
3. epoch / step / batch 位置

### 12.2 warm-start
如果目录里只有 adapter 权重，没有 `training_state.pt`：
1. 可以拿来继续初始化训练
2. 但不属于严格 resume

当前代码明确区分这两种情况。

---

## 13. 推理与部署形态

### 13.1 推理流程
1. 概念抽取得到召回语义
2. SAM3 召回候选
3. 判别器对候选算 `P(Yes)`
4. 按分数排序
5. 输出掩码序列

### 13.2 单基座双模式
当前在主 pipeline 中已经支持共享 base 模型实例：

1. 概念抽取可用 base 行为
2. 判别打分可用 base + adapter 行为

这样可以避免同时部署两份完整模型。

### 13.3 结果记录
当前 DAVIS / MeViS / Ref-YTROS 的 runner 已尽量统一输出：
1. `scores.json`
2. `expression_details/...`
3. 相关 category / coverage / summary 文件

这使得 bad case 分析更系统。

---

## 14. 评估口径

当前核心指标仍为：

1. `J`
2. `F`
3. `J&F = (J + F) / 2`

常见统计口径：
1. expression-level
2. video-level

当前对比实验中，expression-level 仍是主要主表口径。

---

## 15. 当前方案相较早期版本的主要改进

相较最初的“listwise + soft align”版本，当前新增或明确化的改进包括：

1. 锚点式 Yes/No prompt
2. 违规负样本全量惩罚取代 top-k hardest negative
3. `margin` 作为安全间隔重新解释
4. 多数据集联合训练
5. `repeats` 软平衡
6. 更细的 debug 打印
7. true resume 脚本
8. 更统一的过程记录与评测输出

---

## 16. 当前已知局限与实验解释框架

### 16.1 已知局限
1. 候选召回质量仍决定性能上限
2. 冻结视觉端限制了局部细节表征改进空间
3. 联合训练可能牺牲 DAVIS 这类单目标专项能力
4. 锚点式语义虽然更清楚，但仍不能完全消除复杂关系表达中的歧义

### 16.2 当前结果分析建议
当某次实验退化时，应优先区分：
1. category recall 是否退化
2. 候选集是否仍覆盖正确目标
3. discriminator 是否在同类候选中选错
4. 是 joint training 造成的稀释，还是 hard negative 约束过强

---

## 17. 后续优化建议

1. 更系统地做 DAVIS-only / joint 的 clean ablation
2. 分析锚点式 prompt 对不同数据集的收益差异
3. 继续强化“候选为主语、表达式是否成立”的判别建模
4. 在不重训整套视觉 backbone 的前提下，探索更强的局部对齐增强
