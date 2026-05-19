# CRF-SAM3 + Qwen3-VL 候选判别方法论（论文版）

## 1. 任务定义
给定视频序列 $V=\{I_t\}_{t=1}^{T}$ 与自然语言指代表达式 $q$，目标是在候选对象集合 $\mathcal{C}=\{c_i\}_{i=1}^{N}$ 中选出与 $q$ 最匹配的目标对象，并输出其时序掩码。本文方法采用“候选召回 + 语义重排”范式：

1. 候选召回：基于语义类别提示的 SAM3 全视频传播。
2. 候选重排：基于 Qwen3-VL 的 listwise 匹配评分与软掩码对齐约束。

---

## 2. 总体框架

### 2.1 阶段 A：概念抽取（Concept Extraction）
输入表达式与视频关键帧，使用 Qwen3-VL 生成结构化结果（类别词、目标数量、子条件）。
类别词（如 `rabbit`）作为后续 SAM3 的文本提示。

### 2.2 阶段 B：候选召回（SAM3 Concept Recall）
将类别词输入 SAM3，在整段视频上执行传播推理，得到每帧的实例掩码集合：
\[
\mathcal{M}_t = \{M_t^{(k)}\}_{k=1}^{K_t}
\]
并按实例 ID 组织为对象 tube。

### 2.3 阶段 C：候选判别（Qwen3-VL Semantic Discrimination）
针对同一表达式下所有候选对象 tube，构造“掩码可视化帧 + Yes/No 提示”的多模态输入，使用 Qwen3-VL 输出匹配分数并排序选择 Top-$K$。

---

## 3. 离线缓存与训练样本构建

### 3.1 离线缓存动机
在线训练会同时占用 SAM3 与 VLM 显存，且长视频候选生成开销大。本文将候选召回与训练解耦：

1. 离线阶段仅运行召回与 IoU 计算；
2. 训练阶段仅读取离线缓存并更新判别模型。

### 3.2 缓存内容
对每个 `(video_id, exp_id)` 保存：

1. `meta.json`：表达式文本、候选 ID、候选 IoU 映射、每候选可用帧索引、视频路径等；
2. `masks/obj_{id}/xxxxx.png`：候选对象逐帧二值掩码。

### 3.3 GT 读取策略
训练数据支持两类 GT 来源：

1. PNG 目录：`gt_masks_root/{video_id}/{exp_id}/*.png`；
2. RLE 回退：当 PNG 缺失时，从 `{split}/mask_dict.json` 解码（train split 适配）。

### 3.4 IoU 与正负样本定义
对每个候选对象 $c_i$，按视频帧计算与 GT 的平均 IoU：
\[
\text{IoU}(c_i)=\frac{1}{|\Omega_i|}\sum_{t\in\Omega_i}\text{IoU}(M_t^{(i)},G_t)
\]
其中 $\Omega_i$ 为候选和 GT 同时存在的帧集合。

根据阈值划分：
\[
\mathcal{P}=\{i\mid \text{IoU}(c_i)\ge \tau_{pos}\},\quad
\mathcal{N}=\{i\mid \text{IoU}(c_i)\le \tau_{neg}\}
\]
当前默认阈值：$\tau_{pos}=0.5, \tau_{neg}=0.1$。

---

## 4. 候选判别模型

### 4.1 基座模型与参数高效微调
基座为 `Qwen3-VL-4B`，采用 LoRA 微调语言主干关键投影层（`q/k/v/o_proj`, `gate/up/down_proj`）。
视觉编码器默认冻结，仅更新 LoRA 参数，以降低训练开销与过拟合风险。

### 4.2 输入构造
对每个候选对象，采样若干关键帧（默认均匀采样），将掩码叠加到原图（`mask` 或 `bbox` 可选），构造多模态消息：

1. 多帧图像 token；
2. 文本提示：
   `Given the referring expression, judge whether the masked object in the frames matches it. Answer Yes or No.`

### 4.3 训练-推理一致打分
模型在回答首 token 位置输出词表 logits，提取 `Yes`/`No` 两个 token：
\[
\ell^{(i)}_{yes},\ \ell^{(i)}_{no}
\]
采用概率分数：
\[
s_i=P(\text{Yes}\mid q,c_i)=\text{Softmax}([\ell^{(i)}_{yes},\ell^{(i)}_{no}])_{yes}
\]
该设计确保训练目标与推理排序规则一致（避免训练/推理目标错位）。

---

## 5. Listwise 排序目标

### 5.1 动机
RVOS 末端本质是“同表达式下候选排序”，而非独立二分类。点式损失（pointwise）容易在 hard negative 上产生高置信误判。listwise 将同组候选联合优化，直接约束“正确候选应在排序中胜出”。

### 5.2 损失定义
设同表达式候选分数为 $\{s_i\}_{i\in\mathcal{C}}$，正样本集合为 $\mathcal{P}$，温度为 $\tau$：
\[
\mathcal{L}_{cls}
=\log\sum_{j\in\mathcal{C}}\exp(s_j/\tau)
-\log\sum_{j\in\mathcal{P}}\exp(s_j/\tau)
\]
优化目标是提高正样本在全候选中的概率质量。

### 5.3 无正样本兜底
若阈值下没有正样本，使用当前 batch 内 IoU 最大候选作为伪正样本（`listwise_force_pos=true`），避免损失退化。

### 5.4 负样本策略
支持两种策略：

1. `all`：所有非正候选都参与竞争；
2. `threshold`：仅保留显式负样本集合 $\mathcal{N}$。

---

## 6. 软掩码对齐约束（Mask-Aware Soft Constraint）

### 6.1 核心思想
不改模型结构，仅对中间表征施加软约束：文本相关性应主要集中在候选掩码区域，抑制“背景语义捷径”。

### 6.2 表征提取
在指定 Transformer 层注册 hook（默认最后层）：

1. 提取文本 token 特征均值 $t$；
2. 提取图像 token 特征 $\{h_j\}$。

### 6.3 特征对齐损失（当前主用）
相似度（cosine 或 dot）：
\[
a_j=\text{sim}(h_j,t)
\]
softmax 后得到注意分布：
\[
p_j=\frac{\exp(a_j/\tau_a)}{\sum_k\exp(a_k/\tau_a)}
\]
将候选掩码下采样到 token 网格，记为 $m_j\in\{0,1\}$，则帧级损失为：
\[
\mathcal{L}_{align}^{frame}=1-\sum_j p_j m_j
\]
多帧平均得到 $\mathcal{L}_{align}$。

### 6.4 可选注意力能量损失
若 `loss_type=attn`，可使用跨模态注意力权重替代特征相似度，优化 mask 内注意力占比：
\[
\mathcal{L}_{attn}^{frame}=1-\frac{\sum_j A_j m_j}{\sum_j A_j+\epsilon}
\]

### 6.5 总损失
\[
\mathcal{L}=\mathcal{L}_{cls}+\lambda\mathcal{L}_{aux}
\]
其中 $\mathcal{L}_{aux}$ 为 $\mathcal{L}_{align}$ 或 $\mathcal{L}_{attn}$，当前默认 `loss_type=feature`，`\lambda=0.3`。

---

## 7. 训练流程（实现级）

1. 加载配置与随机种子；
2. 若 `offline_prepare=true`，仅生成离线缓存并退出；
3. 若 `offline_mode=true`，从 `offline_cache/{split}` 读取样本；
4. DataLoader 输出 batch（当前固定 `batch_size=1`，每个 batch 为一个表达式的候选集）；
5. 前向计算 listwise 分数与辅助损失；
6. 反向传播并更新 LoRA 参数（AdamW）；
7. 定期打印 `loss/cls/attn/grad_norm/scores_top` 调试信息；
8. 训练结束保存 LoRA 结果，并可 `merge_and_unload` 导出完整推理模型。

工程稳定性补充：训练循环已包含 OOM batch 跳过机制（捕获 CUDA OOM 后清理缓存并继续），降低长训练中断风险。

---

## 8. 推理流程

1. 对每个表达式运行概念抽取与候选召回，得到候选 tube；
2. 对每个候选构造多帧输入并计算 $P(Yes)$；
3. 依据分数降序排序，选择 Top-$K$（$K$ 来自目标数估计或固定设置）；
4. 输出所选候选的逐帧掩码。

推理端 `yesno` 判别同样使用首 token logits 计算 `P(Yes)`，与训练分数定义一致。

---

## 9. 评测协议

评测脚本基于 GT 的 `mask_dict.json` 与 `meta_expressions.json` 计算：

1. `J`：区域 IoU（mask 重叠）；
2. `F`：边界 F-score（轮廓质量）；
3. `J&F`：二者平均。

并提供两种统计口径：

1. Expression-level：对所有表达式直接平均；
2. Video-level：先视频内平均，再跨视频平均。

---

## 10. 方法优势与局限

### 10.1 优势
1. 与任务目标一致：listwise 直接优化候选重排。
2. 与推理一致：训练与推理都使用 `P(Yes)` 排序。
3. 结构无侵入：不改 Qwen3-VL 结构，仅加 LoRA 与软约束。
4. 工程可落地：离线缓存显著降低训练资源波动。

### 10.2 局限
1. listwise 候选数过大时显存峰值仍高，需限制候选上限或帧数。
2. 类别召回错误会将误差传递至判别阶段。
3. 当前辅助损失依赖候选掩码质量，若召回掩码粗糙会影响对齐信号。

---

## 11. 关键实现文件（复现实验）
1. `crf_sam3_attn/dataset.py`：样本构建、离线缓存、GT 回退（mask_dict 解码）。
2. `crf_sam3_attn/model.py`：Qwen3-VL + LoRA、listwise 损失、软掩码对齐。
3. `crf_sam3_attn/train.py`：训练循环、日志、checkpoint、merge 导出。
4. `run_crf_sam3_dataset.py`：全数据集推理。
5. `eval_mapp_sam3.py`：J/F/J&F 指标评测。

