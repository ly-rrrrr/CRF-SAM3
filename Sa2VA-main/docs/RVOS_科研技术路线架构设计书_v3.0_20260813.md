# RVOS 科研技术路线架构设计书（v3.0）

- 更新日期：2026-08-13
- 文档性质：论文写作与后续实验设计底稿
- 文档定位：独立、透明的科研方法说明，不依赖特定代码仓库或运行目录
- 本版原则：严格区分“已实现并验证”“已实现但仅探索”“拟议但未实现”

---

## 1. 一页式结论

当前主线已经由早期的“LLM 概念抽取—SAM3 候选召回—VLM 候选打分”扩展为一套面向 ReVOS 的表达式条件两阶段系统：

```text
表达式 X + 少量原始视频帧 Vq
        │
        ▼
表达式条件类别查询模块（Qwen3-VL 冻结，轻量 Query Grounding 可训练）
        │  输出开放词汇类别提示 p
        ▼
SAM3-DMS 全视频候选召回（recall 24，keep 7）
        │  输出候选 tube 集合 C={C1,...,Cn}
        ▼
候选视觉 tube 构建（4 帧、全图保留、候选 mask 高亮）
        │
        ▼
Qwen3-VL 候选打分器（3域 LoRA → ReVOS warm-start）
        │  输出 P(Yes | X,Ci)
        ▼
Top-1 候选的全视频 mask
```

从物理意义上看，系统完成两次不同粒度的“指代收缩”：

1. 类别查询模块把复杂表达式收缩为 SAM3 能接受的开放词汇语义入口，解决“应该召回哪一类物体”。
2. 候选打分模块在 SAM3 给出的实例集合中完成目标选择，解决“这类物体中的哪一个满足表达式”。

当前完整 ReVOS valid 评估已覆盖 5,822/5,822 条表达式，官方 type-balanced 结果为：

| 子任务 | J | F | J&F |
|---|---:|---:|---:|
| Referring | 62.511 | 67.337 | 64.924 |
| Reasoning | 51.897 | 57.223 | 54.560 |
| Overall | 57.204 | 62.280 | 59.742 |

当前最清楚的瓶颈不是 mask 表示方式，而是离散类别提示这一信息瓶颈：提示词必须既准确指向表达式目标，又落在 SAM3 可响应的词汇区域。后续最有科研价值的方向不是继续堆叠独立校验器，而是让类别查询端形成的连续“指代状态”跨过 SAM3，与现有候选打分端共享语义依据。

---

## 2. 研究问题与科学抽象

### 2.1 任务定义

给定视频帧序列 \(V=\{I_t\}_{t=1}^{T}\) 和自然语言指代表达式 \(X\)，预测每一帧中被指代对象的二值掩码：

\[
\hat{M}=\{\hat{M}_t\}_{t=1}^{T}=f(V,X).
\]

ReVOS 同时包含两类输入：

- Referring：表达式通常显式描述目标的外观、位置、动作或关系。
- Reasoning：表达式可能以功能、常识、问句、无显式主语或非典型形式间接指向目标。

因此，模型不能只做名词抽取；它必须先完成“表达式究竟指向什么实体”的语义消歧，再在视频中建立实例级对应。

### 2.2 当前方法的科学抽象

当前系统将端到端 RVOS 分解为两个条件概率问题：

\[
p^*=\arg\max_p P(p\mid X,V_q),
\]

\[
C^*=\arg\max_{C_i\in \mathcal{R}(V,p^*)}P(C_i\mid X,V_{C_i}).
\]

其中：

- \(V_q\) 是用于语义查询的少量视频帧；
- \(p^*\) 是开放词汇类别提示；
- \(\mathcal{R}\) 是冻结的 SAM3-DMS 候选召回器；
- \(V_{C_i}\) 是候选 \(C_i\) 的高亮视觉 tube。

这不是把分割任务简单改写成分类任务，而是把强分割基础模型视为“可调用的候选生成算子”，把学习能力集中在两个最不确定的位置：召回语义形成和候选指代判别。

### 2.3 三个核心矛盾

1. **语义正确不等于可召回。** 人类认为正确的名词不一定处于 SAM3 最敏感的文本提示区域。
2. **可召回不等于已指代。** `person` 可以召回多个实例，但表达式只指向其中一个。
3. **少帧理解与全视频分割存在尺度差异。** VLM 只读少量帧形成语义，SAM3 则需要把提示传播到完整视频。

当前架构通过“少帧语义解析 + 全视频候选传播 + 候选级少帧判别”降低计算量，但尚未从根本上解决长视频信息缺失与离散提示丢失连续指代语义的问题。

---

## 3. 技术演进与当前状态

| 阶段 | 主要内容 | 当前状态 | 论文定位 |
|---|---|---|---|
| 早期两阶段主线 | LLM 概念抽取 → SAM3 → Qwen3-VL 候选打分 | 已实现 | 基础方法与对照组 |
| 三域打分器 | MeViS + DAVIS17 + Ref-YTROS 联合 LoRA | 已完成 | 通用候选判别初始化 |
| 类别查询 MVP | 表达式条件 query、mask token 监督、类别生成 | 已实现并在 ReVOS 训练 | 当前方法核心模块 |
| ReVOS 打分器适配 | 从三域 LoRA 低学习率 warm-start | 已完成 | 域适配策略 |
| ReVOS 完整评估 | Referring + Reasoning，5,822 条 | 已完成 | 当前主结果 |
| SAM3-native 词库扩展 | SAM3 文本原型库与语义近邻 prompt 扩展 | 已实现过探索 | 可作为补充实验，不是当前最终主线 |
| 共享指代状态桥 | 类别查询与打分器共享连续 referent state | 尚未实现 | 下一阶段核心研究假设 |
| CESTR 长视频编码器 | 表达式驱动、实时全视频状态编码 | 尚未实现 | 长期独立研究方向，不应写成现有方法 |

---

## 4. 当前已实现总体架构

### 4.1 训练与推理的模块边界

当前存在两个分别训练的模块，中间由冻结的 SAM3 候选召回隔开：

```text
模块 A：Expression-Conditioned Category Query Grounding
  训练：ReVOS RGB帧 + 表达式 + GT mask监督 + 伪类别标签
  推理：ReVOS RGB帧 + 表达式 → category prompt

冻结接口：SAM3-DMS
  category prompt + 全视频 → candidate tubes

模块 B：Candidate Referent Scorer
  训练：表达式 + 候选高亮帧 + candidate/GT IoU监督
  推理：表达式 + 每个候选高亮帧 → Yes概率 → 排序
```

两者没有联合反向传播。SAM3 输出的是离散候选集合，因而当前类别查询端形成的连续视觉—语言状态没有传给打分器。

### 4.2 为什么仍称为“两阶段”

工程执行可写成“类别推理—候选召回—候选打分”三步，但从可学习决策看只有两个阶段：

- 阶段 I：生成候选空间，即决定 SAM3 应围绕什么概念找目标。
- 阶段 II：在候选空间内识别指代实例。

SAM3 是二者之间的非训练接口，而不是第三个需要学习的语义决策器。

---

## 5. 阶段 I：表达式条件类别查询模块

### 5.1 输入、输出与训练监督

输入：

- 原始 RGB 视频帧，训练和推理均不输入 GT mask 图；
- 原始指代表达式；
- 训练期 GT mask，仅用于视觉 token 响应监督；
- 训练期伪类别标签，用于 category JSON 生成监督。

输出：

```json
{"category": "washing machine"}
```

这里的类别是开放字符串，而不是固定类别分类头，因此能覆盖 ReVOS 中的长尾和功能性表达。

### 5.2 ReVOS 训练数据

类别监督来自 Qwen3.7+ 基于三视图与目标 bbox 生成的伪类别标注。

数据统计：

- 可用记录：10,412；
- 视频：175；
- 归一化类别：735 个；
- type 0：5,677；type 1：4,735；
- `surface_form`：9,007；`person_ontology`：1,405；
- 当前均按单目标记录构建。

需要在论文中明确称为“伪类别标注”，不能写成数据集提供的真实类别标签。GT mask 是真实空间监督，伪类别则负责确定最终应生成的语义词。

经视频级划分后，query-grounding 数据为：

- train：9,371 条，157 个视频；
- validation：1,041 条，18 个视频；
- 每条样本使用目标可见的 3 帧。

### 5.3 网络结构

基座为冻结的 Qwen3-VL-4B-Instruct。可训练部分包括：

1. `ExpressionConditioner`；
2. `ExpressionConditionedQueryDecoder`；
3. `visual_to_query` 投影；
4. `MaskResponseHead`；
5. `TargetTokenModulator`。

当前关键超参数：

| 参数 | 数值 |
|---|---:|
| target queries | 8 |
| query dimension | 1024 |
| expression conditioner layers | 2 |
| query decoder layers | 2 |
| attention heads | 8 |
| mask projection dimension | 256 |
| mask logit scale | 10.0 |
| modulator bottleneck | 512 |
| \(\lambda_{mask}\) | 1.0 |
| \(\lambda_{cat}\) | 0.5 |

### 5.4 表达式条件查询的物理过程

#### 第一步：表达式形成条件上下文

表达式 token 先使用 Qwen3-VL 的词嵌入，再经过带 summary token 的双向 Transformer Encoder：

\[
T=E_{text}(X).
\]

这里的 \(T\) 不是一个预先规定“主语、关系、背景”分工的手工 token 集合，而是由整条表达式共同形成的条件状态。疑问句、功能描述或省略主语的输入仍以完整 token 序列进入该模块，不依赖固定语法槽位。

#### 第二步：多个查询形成目标假设

8 个可学习查询先做 self-attention，再依次读取表达式状态和视觉 token：

\[
Q^{l+1}=\operatorname{FFN}(\operatorname{CA}_{V}(\operatorname{CA}_{T}(\operatorname{SA}(Q^l),T),V)).
\]

物理含义是：每个 query 都是一个尚未命名的“可能目标解释”；它先被表达式规定搜索意图，再到图像 token 中寻找支持该解释的区域。

#### 第三步：查询产生 token 级目标响应

query 与视觉 token 投影后计算归一化相似度：

\[
R_{k,n}=\gamma\left\langle
\frac{W_qQ_k}{\|W_qQ_k\|},
\frac{W_vV_n}{\|W_vV_n\|}
\right\rangle .
\]

每个查询得到一张 token 级响应图。GT mask 被下采样到 Qwen3-VL 视觉 token 网格，只作为监督目标。

#### 第四步：训练期查询匹配

每个 query 与 GT token mask 计算 BCE 与 Dice 代价：

\[
\mathcal{C}_k=\mathcal{L}_{BCE}(R_k,M)+\mathcal{L}_{Dice}(R_k,M),
\qquad
k^*=\arg\min_k\mathcal{C}_k.
\]

最低代价的 query 负责该样本的 mask 学习。这允许多个 query 自组织为不同的视觉目标假设，而不预设固定语义角色。

#### 第五步：目标响应回写视觉 token

选中响应以残差形式调制视觉表示：

\[
V'_n=V_n+\alpha\,\sigma(R_{k^*,n})\,A(V_n).
\]

调制器最后一层零初始化，因此训练从原 Qwen3-VL 视觉表示出发，不会在初始阶段剧烈破坏基座分布。

#### 第六步：生成开放词汇类别

调制后的视觉 token 与表达式共同进入冻结的语言模型，生成类别 JSON。总损失为：

\[
\mathcal{L}_{query}=\lambda_{mask}\mathcal{L}_{mask}+\lambda_{cat}\mathcal{L}_{cat}.
\]

其中 \(\mathcal{L}_{cat}\) 只作用在 assistant 的类别输出 token 上。

### 5.5 无 GT 推理

推理时只输入表达式和均匀采样的 3 帧 RGB 图像，不输入 mask。当前代码将所有 query 的响应按 token 取最大值：

\[
R_n^{infer}=\max_k \sigma(R_{k,n}),
\]

再调制视觉 token 并贪心生成类别。

这一点存在明确的训练—推理差异：训练时只用最佳匹配 query，推理时聚合全部 query。它是后续应做消融的真实风险，不能在论文中忽略。

### 5.6 已完成训练与模型选择

训练配置：1 epoch、batch size 1、gradient accumulation 8、学习率 \(10^{-4}\)、bf16、gradient checkpointing；共完成 1,172 个 optimizer step。

已评估 checkpoint：

| Checkpoint | Eval loss | Mask soft Dice |
|---:|---:|---:|
| 300 | 0.7633 | 0.6557 |
| 600 | 0.7136 | 0.7218 |
| 900 | 0.6911 | 0.7172 |

当前完整验证流程使用 checkpoint-900，依据是其在已评估 checkpoint 中具有最低 eval loss；不能仅凭该表声称其 category accuracy 最优，因为这里尚缺少人工类别正确率与 SAM3 prompt recall 的独立评测。

ReVOS valid 推理生成 5,822 条类别缓存：

- 正常 JSON 解析：5,749；
- 从简短纯文本恢复：73；
- 最终有效：5,822/5,822。

---

## 6. 中间接口：SAM3-DMS 候选召回

### 6.1 接口定义

类别提示 \(p\) 被送入 SAM3-DMS，在完整视频上得到对象级 tube 候选：

\[
\mathcal{C}=\mathcal{R}_{SAM3}(V,p).
\]

当前 ReVOS 配置：

- backend：DMS；
- candidate recall top-k：24；
- candidate keep top-k：7；
- 完整视频跟踪，不设置最大跟踪帧数；
- 视频与输出可卸载到 CPU 以控制显存。

### 6.2 物理意义

类别提示不是最终答案，而是启动 SAM3 内部视觉对象记忆的“检索钥匙”。SAM3 负责把少帧语义决策扩展成全视频空间候选，因此系统无需让 Qwen3-VL 逐帧编码整段视频。

### 6.3 性能上限

若 GT 对象不在 \(\mathcal{C}\) 中，任何后续排序都无法恢复它。因此最终性能满足一个直观上限：

\[
\text{Final J\&F}\leq \text{Candidate Oracle J\&F}.
\]

这也是当前研究重点转向“尽最大可能获得准确 SAM3 提示词”的原因。

### 6.4 不采用的训练路径

当前不允许把 SAM3 缓存或 SAM3 是否召回成功作为类别提示模块的训练监督。原因是：

- SAM3 的失败可能来自其自身开放词汇边界，而非提示语义错误；
- 以 SAM3 反馈训练提示词会把基础模型偏差固化为教师信号；
- 它无法证明学到的是目标语义，而不是迎合某个冻结分割器的词汇漏洞。

SAM3 输出可用于推理、评估与误差归因，但不应被描述为类别查询模块的语义教师。

---

## 7. 阶段 II：候选指代打分模块

### 7.1 输入表示

每个候选单独构造视觉 tube，当前均匀采样 4 帧。实际 `mask` 可视化保留完整 RGB 画面，并以约 25% 透明度的绿色区域高亮候选 mask；它不是把候选裁成脱离背景的孤立目标。

因此打分器同时能看到：

- 被高亮的候选本体；
- 候选周围的场景、关系对象和动作上下文；
- 同一候选在多个时刻的状态。

但不同候选仍是逐个前向打分，候选之间不会在同一次 attention 中直接交互；它们仅通过 listwise 损失在训练目标上发生组内竞争。

### 7.2 锚点式 Yes/No 判别

打分 prompt 要求模型把高亮对象视作表达式的锚点主语，并判断表达式落在该对象上是否成立。候选分数为：

\[
s_i=P(\text{Yes}\mid X,V_{C_i}).
\]

最终选择：

\[
i^*=\arg\max_i s_i.
\]

### 7.3 训练监督

SAM3 候选与 GT mask 计算 tube IoU：

- IoU \(\ge 0.5\)：正候选；
- IoU \(\le 0.1\)：负候选；
- 中间区间保留为候选，但不等同于明确负样本；
- 若组内无正样本，可由 `listwise_force_pos` 以最大 IoU 候选兜底。

### 7.4 损失函数

#### Listwise 候选集合损失

\[
\mathcal{L}_{list}
=\log\sum_{j\in\mathcal{C}}\exp(s_j/\tau)
-\log\sum_{j\in\mathcal{P}}\exp(s_j/\tau).
\]

它直接优化同一表达式下正候选相对全部候选的排序，而不是把每个候选当作互不相关的二分类样本。

#### 违规负样本间隔损失

以 \(r_i=l_i^{Yes}-l_i^{No}\) 作为更稳定的排序量。所有超过安全边界的负候选都受到惩罚，而不是只处理固定 top-k hardest negatives。

#### Mask-aware 特征对齐

从 Qwen3-VL 隐藏状态计算表达式—视觉 token 相似性，使语义质量更多落在高亮候选 mask 内，同时保留关系判断所需的全图上下文。

总损失写为：

\[
\mathcal{L}_{score}
=\mathcal{L}_{list}
+\lambda_{rank}\mathcal{L}_{rank}
+\lambda_{align}\mathcal{L}_{align}.
\]

当前 ReVOS warm-start 运行中有效 `hard_neg_weight` 由 data 配置覆盖为 0.3，margin 为 0.1，`attn_loss_weight` 为 0.3。

### 7.5 参数高效微调

Qwen3-VL 视觉编码器被冻结，LoRA 作用于：

- `q_proj / k_proj / v_proj / o_proj`；
- `gate_proj / up_proj / down_proj`。

该设计主要学习“如何围绕候选锚点读取既有视觉 token”，而不是重训视觉 backbone。

### 7.6 从三域模型继续训练，而非重新联合训练

初始化模型已在 MeViS、DAVIS17、Ref-YTROS 三域联合训练。ReVOS 训练采用 fresh optimizer 的 warm-start：

- 1 epoch；
- batch size 1；
- gradient accumulation 4；
- 学习率 \(2\times10^{-6}\)；
- weight decay 0；
- 视觉编码器冻结；
- 4 帧视觉 tube。

这不是严格 resume，因为优化器重新初始化；但它保留三域 LoRA 的候选判别能力，避免重新进行耗时的四数据集联合训练。

### 7.7 ReVOS 打分训练样本审计

10,412 条伪类别记录不等于 10,412 条可用于打分训练的候选组：

| 环节 | 样本数 | 说明 |
|---|---:|---|
| 伪类别缓存 | 10,412 | 类别查询训练入口 |
| 可用 SAM3 离线候选缓存 | 9,545 | 867 条未形成当前可用候选缓存 |
| 最大候选 J < 0.5 的组 | 1,763 | 无可靠正候选，按配置排除 |
| 最终打分训练组 | 7,782 | 9,545 - 1,763 |

这体现了两种不同的数据需求：类别查询可以使用全部 10,412 条表达式；打分器训练则要求候选集合中至少存在足够可靠的正候选。

---

## 8. ReVOS 完整验证结果与正确解释

### 8.1 最终配置

```text
ReVOS query checkpoint-900
    + SAM3-DMS recall24/keep7
    + ReVOS warm-start scorer final
    + 4-frame candidate tube
    + Yes/No ranking
```

覆盖情况：

- 输入表达式：5,822；
- 有效预测：5,822；
- Referring：3,130；
- Reasoning：2,475；
- type 2：217，未计入官方 referring/reasoning 平衡总分；
- 官方平衡统计样本：5,605。

### 8.2 主结果

| 子任务 | J | F | A | R | J&F |
|---|---:|---:|---:|---:|---:|
| Referring | 62.511 | 67.337 | 96.018 | 91.670 | 64.924 |
| Reasoning | 51.897 | 57.223 | 94.614 | 91.243 | 54.560 |
| Overall | 57.204 | 62.280 | 95.316 | 91.457 | 59.742 |

### 8.3 与旧 2,322 条进度快照的关系

旧跨域实验显示的 2,322/5,822 是运行中的非随机前缀快照，不是完整验证集结果。旧结果文件后来实际积累到 2,390 条。直接用旧前缀 J&F 约 60.1 与新完整集 59.742 比较，会混入样本难度和覆盖范围差异。

在完全相同的旧 2,390 条子集上重算：

| 系统 | Referring J&F | Reasoning J&F | Balanced J&F |
|---|---:|---:|---:|
| 旧：RefYTROS query 跨域流程 | 65.909 | 54.261 | 60.085 |
| 新：ReVOS query + ReVOS warm-start scorer | 68.404 | 57.272 | 62.838 |
| 差值 | +2.495 | +3.012 | +2.753 |

因此，当前证据不支持“继续训练一定劣于未继续训练”的结论。更准确的结论是：新系统在旧共同子集上更好，而新增覆盖的 3,432 条样本更难，使完整集均值下降。

同时，这个对照不能单独归因于类别查询或打分器，因为两端都发生了变化，且历史流程的帧数与 prompt 设置也不完全一致。论文必须补充受控消融。

### 8.4 误差结构

按来源数据集的 J&F：

| 来源 | Overall | Referring | Reasoning |
|---|---:|---:|---:|
| LV-VIS | 63.710 | 66.567 | 60.673 |
| MOSE | 62.512 | 66.956 | 55.146 |
| OVIS | 62.034 | 68.178 | 51.203 |
| UVO | 56.695 | 61.463 | 51.300 |

Reasoning 始终明显弱于 Referring，说明功能、常识和隐式指代仍是主要语义难点。

候选数量与 J&F 的观察值：

| 保留候选数 | 样本数 | J&F |
|---:|---:|---:|
| 0 | 124 | 17.724 |
| 1 | 1,754 | 61.844 |
| 2 | 1,954 | 64.048 |
| 3 | 912 | 58.143 |
| 4 | 520 | 52.932 |
| 5 | 200 | 53.330 |
| 6 | 129 | 38.495 |
| 7 | 229 | 43.389 |

这说明 0 候选是严重召回失败，多候选样本也更容易失败；但不能据此断言“候选越多必然导致指标下降”，因为多候选往往同时意味着提示更宽、场景更拥挤或目标更难，存在明显混杂变量。

打分器 top-1 与 top-2 的分数间隔具有很强诊断意义：

- margin < 0.05：918 条，J&F 34.673；
- margin > 0.6：3,018 条，J&F 74.741。

它说明低 margin 是可靠的不确定性信号，但目前只适合用于审计和分层分析，不应在未验证前写成新的推理规则。

---

## 9. 当前方法的逻辑自洽性

### 9.1 自洽之处

1. 复杂语言首先被映射到开放词汇提示，符合 SAM3 的接口约束。
2. GT mask 只监督类别查询的视觉 grounding，不作为推理输入，训练—部署输入形式基本一致。
3. SAM3 读取完整视频并负责候选传播，使 VLM 不必逐帧处理长视频。
4. 候选打分器读取实例高亮与完整上下文，适合处理同类实例消歧和关系表达。
5. 三域打分器 warm-start 到 ReVOS，符合当前计算资源与防遗忘需求。

### 9.2 尚未完全闭合之处

1. 类别 query 在训练时以 GT 选 \(k^*\)，推理时对所有 query 取 max，存在 query 聚合差异。
2. 类别字符串是窄离散通道，会丢失“哪个人、与谁交互、在何时做什么”等连续指代信息。
3. 类别查询和候选打分由相互独立的参数形成语义判断，可能产生前后不一致。
4. 3 帧类别推理和 4 帧候选判别仍可能错过只在短时段出现的关键证据。
5. 当前最终结果没有分离“类别查询改进”和“打分器 ReVOS 适配”的独立贡献。

因此，现有系统在工程和概率分解上是自洽的，但在跨模块语义连续性上仍存在一个明确断点。

---

## 10. 下一阶段拟议核心：共享指代状态桥

> 状态：研究设计，尚未实现，不能写成当前实验已采用的方法。

### 10.1 动机

当前类别查询模块内部已经产生表达式条件 target queries 和 token 响应，但最终只导出一个类别字符串。SAM3 用这个字符串召回候选后，打分器又从头根据表达式判断候选。两端在解决同一个“表达式指向谁”的问题，却没有共享连续表示。

应把类别词视为共享指代状态的一个读出，而不是完整状态本身：

\[
Z_{ref}=F_{ref}(X,V_q),
\qquad
p=H_{prompt}(Z_{ref}),
\qquad
s_i=H_{score}(Z_{ref},X,V_{C_i}).
\]

### 10.2 latent referent state 的物理意义

\(Z_{ref}\) 表示“模型在看完表达式和少量视频证据后，对待分割实体形成的连续内部假设”。它不被人为拆成主体 token、背景 token、关系 token，也不要求输出 bbox、ID 或 mask。

它应该隐含：

- 目标可能属于什么视觉概念；
- 表达式中哪些属性和关系是辨认目标的证据；
- 当前视频里哪些区域支持这一实体假设；
- 哪些信息需要保留到候选实例消歧阶段。

类别词只是 \(Z_{ref}\) 面向 SAM3 的语言投影，候选得分则是 \(Z_{ref}\) 面向实例判断的视觉投影。

### 10.3 最小可实现结构

不改 SAM3、不改变当前候选打分流程的主体，只增加一个轻量桥：

```text
ExpressionConditioner + QueryDecoder
              │
              ├── Prompt Readout → category string → SAM3
              │
              └── Referent Pooler → Zref → Bridge Projector → latent prefix
                                                        │
expression + candidate highlighted frames + latent prefix
                                                        │
                                           existing Yes/No scorer
```

推荐从 query states 与表达式 summary 中池化，而不是直接传最终类别 token：

\[
Z_{ref}=\operatorname{LN}\big(W_z[\operatorname{Pool}(Q);T_{summary}]\big).
\]

通过 2—8 个 prefix token 投影到打分器隐藏维度，插入现有 Yes/No prompt 前部。候选 mask、4 帧视觉 tube、listwise 损失和最终排序规则保持不变。

### 10.4 两种训练边界

#### 稳健版：停止梯度

\[
\tilde Z_{ref}=\operatorname{sg}(Z_{ref}).
\]

只训练 bridge projector 和现有 scorer LoRA。优点是不会让候选打分损失反向污染类别查询，适合作为第一版可归因实验。

#### 共享版：有限联合更新

允许打分损失以较小系数更新 query decoder，但类别查询仍由 mask 与类别损失锚定：

\[
\mathcal{L}
=\mathcal{L}_{query}
+\lambda_s\mathcal{L}_{score}
+\lambda_c\|\operatorname{sg}(Z_{ref}^{query})-Z_{ref}^{score}\|_2^2.
\]

这里的状态一致性不是人为构造提示词正负样本的对比学习，而是要求同一实体假设在“生成召回词”和“判断候选”两个读出中保持连续。

### 10.5 必须保持的约束

1. 不用 SAM3 召回成功与否监督 prompt 正误。
2. 不由 SAM3 缓存构造类别查询训练标签。
3. 不改变现有候选生成和 Yes/No 排序的主要逻辑。
4. 不手工规定多个 token 的固定语义职责。
5. 不把 bridge 的效果与更多候选、更换 scorer 或更换帧数混在同一实验中。

### 10.6 主要风险

- `Z_ref` 可能退化为类别标签编码，未携带实例关系信息；
- scorer 可能忽略 prefix，只依赖高亮图像和原表达式；
- 联合反传可能使 query 模块过拟合打分训练中“已有正候选”的子集；
- 类别查询有 10,412 条，而 scorer 只有 7,782 个可靠候选组，监督覆盖不对称；
- 训练期 target query 选择与推理期 max 聚合问题仍需先处理。

因此建议先做 stop-gradient bridge，再决定是否联合训练。

---

## 11. 论文实验设计

### 11.1 必做主消融

所有对照必须在同一 5,822 条 ReVOS valid、同一 SAM3-DMS 参数、同一 4 帧 scorer 输入上运行。

| 实验 | Category prompt | Scorer | 目的 |
|---|---|---|---|
| A | 旧 RefYTROS query | 三域 scorer | 跨域基线 |
| B | ReVOS query | 三域 scorer | 隔离类别查询贡献 |
| C | 旧 RefYTROS query | ReVOS warm-start scorer | 隔离 scorer 适配贡献 |
| D | ReVOS query | ReVOS warm-start scorer | 当前完整系统 |

若实现共享状态桥，再增加：

| 实验 | Bridge | Query更新 | 目的 |
|---|---|---|---|
| E | 无 | 否 | D 的复现基线 |
| F | latent prefix | stop-gradient | 验证共享状态本身 |
| G | latent prefix | 小学习率联合 | 验证双读出协同 |

### 11.2 类别查询模块独立评估

不能只用最终 J&F 判断类别查询好坏，至少应报告：

1. 人工抽样的 prompt 语义正确率；
2. Referring 与 Reasoning 分开统计；
3. exact/同义词/上位词/过宽词/错误实体/空输出的错误类型；
4. 在冻结 SAM3 下的 candidate oracle J&F，仅作为召回评估，不作为训练监督；
5. query 推理聚合方式：max、learned query confidence、soft aggregation。

### 11.3 候选级评估

建议将最终误差拆为：

```text
Prompt failure：提示导致 0 候选或 oracle 很低
Recall failure：有候选但 GT 未进入 keep-7
Ranking failure：GT 候选存在但 scorer 未选中
Propagation failure：选中正确对象但全视频 mask 质量不足
```

关键指标：

- candidate recall@1/2/4/7；
- oracle J&F@K；
- scorer top-1 accuracy（仅在 oracle 候选存在时）；
- top1-top2 margin 分层 J&F；
- 0 候选率与候选数分布。

### 11.4 帧数消融

旧评估曾使用 8 帧，而当前三域训练与 ReVOS warm-start 都使用 4 帧。应固定模型后单独比较 4/8 帧，不能把帧数变化与模型训练变化一起解释。

类别 query 当前用 3 帧，也应比较 3/5/自适应关键帧，但第一轮论文主表宜保持 3 帧以匹配已完成模型。

---

## 12. 可主张的创新与不可过度主张的内容

### 12.1 当前代码与结果支持的表述

1. 提出一种 mask-supervised expression-conditioned target-query 模块，用少量 RGB 视频帧和完整表达式产生开放词汇 SAM3 提示。
2. GT mask 只在训练期监督视觉 token 响应，推理不依赖 mask 提示。
3. 使用冻结 SAM3-DMS 将少帧语义提示扩展为全视频候选 tube，再以候选锚点式 VLM 排序完成实例选择。
4. 通过三域打分器到 ReVOS 的低学习率 warm-start 完成域适配，并在 ReVOS valid 全量评估 referring 与 reasoning。

### 12.2 当前不能直接写入论文结论的表述

1. “类别查询模块单独带来 +2.753 J&F”——该提升来自两端共同变化。
2. “候选越多必然使性能下降”——当前只有相关性，没有因果控制。
3. “4 帧优于 8 帧”——尚未做同模型、同样本受控比较。
4. “类别提示准确率达到某数值”——尚无独立人工语义标注评测。
5. “共享指代状态已经解决模块割裂”——该模块尚未实现。
6. “模型读完所有视频帧”——当前 VLM 只读少量采样帧；全视频传播由 SAM3 完成。

---

## 13. 论文方法章节建议结构

### 3.1 Problem Formulation

定义视频、表达式、类别提示、候选 tube 与最终 mask，并说明候选空间分解。

### 3.2 Expression-Conditioned Category Query Grounding

依次描述 expression conditioner、target query decoder、mask response head、token modulator 和类别生成损失。

### 3.3 Frozen Foundation Segmenter as Candidate Interface

说明 SAM3-DMS 如何接收开放词汇提示、在全视频上召回和筛选 tube，并强调冻结且不提供训练监督。

### 3.4 Candidate-Anchored Referent Scoring

描述高亮视觉 tube、Yes/No 分数、listwise 排序、违规负样本和 mask-aware 对齐。

### 3.5 Domain Adaptation and Inference

说明三域 scorer warm-start、ReVOS 查询模块训练、完整推理及复杂度。

若共享指代状态桥完成并获得受控实验支持，可把它升级为 3.3，在其后再介绍 SAM3 接口。

---

## 14. 复现所需信息

为了使方法可以独立复现，而不依赖原始工程目录，实验发布时至少应提供以下内容。

### 14.1 数据说明

- ReVOS 原始训练与验证划分；
- 10,412 条伪类别标注及其生成协议；
- 9,371/1,041 的视频级训练—验证划分；
- 训练期 RGB 帧、GT mask 与表达式的对应关系；
- 9,545 条可用候选缓存和 1,763 条低召回排除记录的统计规则。

### 14.2 模型说明

- Qwen3-VL-4B-Instruct 类别查询基座；
- 8-query grounding 模块的完整结构和权重；
- SAM3-DMS 的模型版本与 recall24/keep7 参数；
- MeViS、DAVIS17、Ref-YTROS 三域初始化打分器；
- ReVOS warm-start 后的候选打分器。

### 14.3 训练说明

- 类别查询模块的损失、学习率、batch、梯度累积、帧数和 checkpoint 选择标准；
- 候选正负样本的 IoU 阈值与排除策略；
- listwise、违规负样本和 mask-aware 对齐损失的系数；
- warm-start 与严格 resume 的区别；
- 所有随机种子、精度模式和视觉编码器冻结范围。

### 14.4 评估说明

- ReVOS 5,822 条表达式的完整预测；
- Referring、Reasoning 和 type 2 的计数及总体指标计算方式；
- 逐表达式 J、F、J&F 和候选审计记录；
- 旧跨域系统与新系统在共同 2,390 条子集上的对齐结果；
- 四格 A/B/C/D 受控消融的统一评估协议。

---

## 15. 下一步执行优先级

1. **先补齐四格受控消融 A/B/C/D。** 这是判断类别查询与 scorer 各自贡献的必要证据。
2. **建立 prompt—candidate—ranking 三层错误审计。** 不改变推理流程，只形成可量化瓶颈。
3. **解决 query 训练/推理聚合差异。** 比较 max aggregation 与可学习 query confidence。
4. **实现 stop-gradient 共享指代状态桥。** 保持 SAM3 和 scorer 主流程不变，先验证连续状态是否有增益。
5. **仅在 bridge 有稳定增益后尝试有限联合训练。** 避免过早让 7,782 个候选组反向主导 10,412 条类别语义学习。
6. **最后再讨论表达式驱动的全视频实时编码器。** 该方向与当前工作有关，但规模和研究问题更大，宜作为独立工作推进。

---

## 16. 本版最终定位

当前成果已经形成一条可复现、可评估的 RVOS 技术主线：

> 用表达式条件 target queries 将复杂语言与少量视觉证据压缩为开放词汇召回提示；用冻结 SAM3 在完整视频中形成候选；再用候选锚点式多模态打分器完成实例消歧。

这条路线的已验证价值是把“语义解析、全视频传播、实例选择”放到各自擅长的模型中。它的核心未解问题是：类别提示作为离散接口过窄，使召回前形成的连续指代状态无法帮助召回后的候选判断。

因此，论文近期应以现有两阶段系统和 ReVOS 全量结果为事实基础，以受控消融补足归因；下一阶段创新则围绕“一个表达式驱动的连续指代状态，两个任务读出，中间经过冻结 SAM3”展开。
