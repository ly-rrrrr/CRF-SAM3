# v19 修改说明（2026-11-19，基于 v18 + 仓库源码）

v18 原稿未改动。v19 只修改了 `sections/3_method.tex`、`sections/4_experiments.tex` 的 4.1 节和 `references.bib`；摘要、引言、相关工作、结论、4.2 节以后的结果与分析均未改动。逐行差异见 `v18_to_v19.diff`。

## Method（3_method.tex）

| 位置 | v18 写法 | v19 写法（依据） |
|---|---|---|
| 3.1 总体公式 | 输出单个 tube $\hat\tau$ | 输出被选中的 tube 集合 $\hat{\mathcal S}$，最终 mask 为其并集（`crf_sam3_demo.py` 第 2365、3475 行） |
| 3.2 视觉/文本表示 | 笼统描述 | $V$ 是 Qwen3-VL 插入占位符位置的 merged visual token；表达式单独分词，经冻结的词嵌入层进入 conditioner |
| 3.2 Query decoder | 无 pre-norm 描述 | pre-norm 残差结构；明确 decoder 和 response 使用投影后的 $\bar V$ |
| 3.2 Token 标签 | 映射到 token 网格 | GT 缩放到 patch 网格后，在 2×2 merge 窗口内求平均，覆盖率 ≥ ρ 记为正（`token_mask_from_rle`） |
| 3.2 Soft modulation | 轻量 adapter | A = LN–Linear–SiLU–Linear，末层零初始化（起始为恒等映射），α 可学习；**同一个** gate 和 adapter 也作用于 DeepStack |
| 3.2 Concept 生成 | category-generation supervision | 冻结的 decoder 生成 JSON；ℒ_cat 是该 JSON 的交叉熵；**梯度经 m 回传到 query decoder**（`selected_response` 未 detach） |
| 3.3 Recall | 只写 SAM3 | 写明使用 SAM3-DMS（新增引用 arXiv 2601.09699）；先按检测分数召回 $K_r$ 个实例，再保留 $K_{keep}$ 个；零候选时输出空 mask |
| 3.3 训练缓存 | 未说明 concept 来源 | 训练缓存使用训练期 concept，而不是 grounder 的输出（详见 4.1） |
| 3.4 Verifier 输入 | mask-conditioned 可视化 | 在候选存在的帧中均匀采样 F 帧，用半透明 mask 叠加标出候选；**推理时表达式含颜色词，改用 bbox 标记**（`crf_sam3_demo.py` 第 3211 行） |
| 3.4 打分 | 首 token Yes/No | 写出 $s_i$ 的 softmax 公式；训练和推理使用同一个分数 |
| 3.4 ℒ_list | 对 $\mathcal C$ 求和 | 对 $\mathcal P\cup\mathcal N$ 求和（MeViS 只取 J≤θ_neg 的负样本）；温度改记为 κ，避免与 tube 的 τ 冲突 |
| 3.4 ℒ_hard | — | $r^+$ 取 max 或 mean（按数据集）；违规负样本属于 $\mathcal N$ |
| 3.4 ℒ_align | 文本 token 平均；对所有 token 做 softmax | 使用最后一层 hidden state；文本向量是全部**非视觉 prompt token**（指令 + 表达式）的平均；**softmax 按帧计算**，再对帧取平均；**对正负候选都计算**，并给出理由 |
| 新 3.5 推理与选择 | 选最高分的一个 | 用冻结的 Qwen3-VL（关闭 LoRA）回答计数 prompt 得到 $\hat N$，取前 min($\hat N$, $N_c$) 个候选，逐帧取并集；单目标时退化为 top-1 |
| 已删除 | — | 不出现 backup categories 的任何描述 |

## 实验设置（4_experiments.tex 4.1）

- 新增 `\label{sec:setup}`。将 "Expression-conditioned grounding" 和 "Candidate recall and verification" 两段改写为 "Ground stage" 和 "Recall and verification"，所有超参数均与配置文件一一核对。
- 写明 grounder 训练帧取自目标有标注的帧，推理帧则均匀采样，并说明这样做的原因。
- 写明 ℒ_cat 的类别标签来源：Ref-YTVOS 用数据集自带的类别标注；ReVOS 用 Qwen3.7-Plus 根据表达式和 3 帧带 GT 框的图离线生成，再做归一化，仅用于训练。
- 写明 verifier 训练缓存的 concept 来源（三个数据集用 prompt-based 抽取器，ReVOS 用上述类别标签）、三数据集采样比 1:4:2，以及 ReVOS 的 warm-start。
- 新增表 `tab:verifier_cfg`，列出每个训练集的 θ_pos / θ_neg / 负样本策略 / λ_h / r⁺。

## 第二轮修改（按作者 2026-11-19 的意见）

- 全文统一使用 target-query grounder：删除了表 1 的 † 脚注，以及 4.1 节的 TODO 注释。
- 4.1 节写明：ReVOS 使用在 ReVOS 上训练的 grounder，其余三个数据集使用在 Ref-YTVOS 上训练的 grounder。**请核对 MeViS 是否属实。**
- 正式主实验已取消备选词逻辑：Ref-YTVOS 和 Ref-DAVIS17 的主结果（表 2、4.2 节正文、摘要）先改为 "--"。删除了依赖这两个数字的对比句（"DAVIS 强于 YTVOS"），结论改为只提 MeViS。正文不出现备选词或语义扩展。MeViS、ReVOS 的数字和全部分析实验均保持不变。
- ReVOS 类别标签（Qwen3.7-Plus + GT 框）和 verifier 训练缓存的 concept 来源，已从正文移到补充材料。`qualitative_supplement.tex` 新增 "Training labels and candidate caches" 一节，标题改为 "Supplementary Material"。正文只保留 "见补充材料"。

## 尚需作者处理

1. 重跑后，把 Ref-YTVOS 和 Ref-DAVIS17 的数字填回摘要、表 2 和 4.2 节（全文搜索 `--` 即可定位）。
2. 本环境无法编译 LaTeX，只做了静态检查（引用、标签、括号和环境配对均无问题），请在 Overleaf 上编译确认。
