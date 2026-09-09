# Referring Video Object Segmentation (RVOS) 判别器微调需求文档

## 1. 业务背景

### 1.1 任务定义
**Referring Video Object Segmentation (RVOS)** 是指根据自然语言表达式，在视频中定位并分割特定目标对象的任务。

### 1.2 当前系统架构：CRF-SAM3

```
Stage 1: Concept Recall (SAM3)
   ↓ 召回所有候选轨迹（过召回）
Stage 2: Visual Tube Construction
   ↓ 构建时空视觉表示
Stage 3: Semantic Discrimination (MLLM) ← 当前瓶颈
   ↓ 从候选中筛选出正确轨迹
Output: 最终分割结果
```

### 1.3 核心痛点

**Stage 3 判别器** 使用通用视觉语言大模型（Qwen2.5-VL / Qwen3-VL），存在以下问题：

| 问题 | 具体表现 | 影响 |
|------|---------|------|
| **判别能力不足** | 所有候选得分都偏低（<0.4） | 难以区分正确/错误轨迹 |
| **过于保守** | 倾向输出 "No"，不敢判断 "Yes" | 可能漏选正确目标 |
| **缺乏时序理解** | 只看8帧静态画面，难以理解动作 | "eats the food first" 等动作表达式失败 |
| **缺乏对比能力** | 无法理解 "the one vs. the other" | 多目标对比场景失败 |
| **泛化性过强** | 通用模型对 RVOS 任务不够专注 | 响应冗长，逻辑复杂 |

---

## 2. 任务需求

### 2.1 输入格式

**单个样本输入：**
```python
{
    "visual_tube": [
        "frame_00000.jpg",  # 8帧标注图像（带有目标轮廓/mask）
        "frame_00025.jpg",
        ...
        "frame_00150.jpg"
    ],
    "frame_indices": [0, 25, 50, 75, 100, 125, 150, 175],  # 真实帧索引
    "referring_expression": "the one who eats the food first",
    "candidate_id": 2,
    "label": "Yes"  # 或 "Partial" / "No"
}
```

**关键特点：**
- Visual Tube 包含 **8 帧**带标注的视频帧（uniform 采样）
- 标注方式：彩虹轮廓 / 绿色半透明 mask / 红色边界框 / 黄色发光轮廓
- 每个样本对应一个候选轨迹
- 需要判断该轨迹是否匹配表达式

### 2.2 输出要求

**Three-Level 判别（推荐）：**
```json
{
    "answer": "Yes",  // "Yes" / "Partial" / "No"
    "confidence": 0.95,
    "reasoning": "The lizard in the visual tube clearly eats the food at frame 50, earlier than others."
}
```

**或 Fine-Grained 评分：**
```json
{
    "confidence_score": 85,  // 0-100
    "visual_summary": "A lizard moving toward food and eating it",
    "text_requirements": "eats the food first",
    "alignment_analysis": "Strong match: the lizard eats at frame 50, first among all candidates"
}
```

### 2.3 性能指标

| 指标 | 定义 | 目标值 |
|------|------|--------|
| **Top-1 Accuracy** | 选出的最高分候选是否正确 | ≥ 75% |
| **J&F Score** | 最终分割质量（IoU 指标） | ≥ 60% |
| **区分度** | max(score) - mean(score) | ≥ 0.3 |
| **置信度** | 正确样本的平均得分 | ≥ 0.7 |

---

## 3. 数据集需求

### 3.1 训练数据构建

**来源：MeViS, Ref-YouTube-VOS, Ref-DAVIS**

**样本生成流程：**
```
1. 对每个视频-表达式对，运行 SAM3 Recall
   ↓ 得到 N 个候选轨迹
2. 构建 Visual Tubes（每个候选8帧）
   ↓
3. 标注每个候选：
   - "Yes": 与 Ground Truth IoU ≥ 0.7
   - "Partial": 0.4 ≤ IoU < 0.7
   - "No": IoU < 0.4
   ↓
4. 每个样本：(Visual Tube, Expression, Label)
```

**数据规模估算：**
- MeViS train: ~300 videos × 平均 6 candidates = **1,800 样本**
- Ref-YouTube-VOS: ~3,000 videos × 6 = **18,000 样本**
- Ref-DAVIS: ~90 videos × 6 = **540 样本**

**总计：约 20,000 训练样本**

### 3.2 数据分布

**表达式类型分布：**
```
- 动作类 (40%): "elephant walking behind", "lizard eating food"
- 关系类 (20%): "hand holding lizard", "person sitting with phone"
- 状态类 (10%): "stationary truck", "crouched monkey"
- 空间类 (15%): "front elephant", "lizard in the distance"
- 对比类 (15%): "the one who eats first", "the other moving away"
```

**候选标签分布（需要平衡）：**
```
- Yes: 30% (正样本)
- Partial: 20% (边界样本)
- No: 50% (负样本)
```

---

## 4. 微调方案设计

### 4.1 监督学习微调（Baseline）

#### 4.1.1 LoRA 微调

**优点：** 参数效率高，快速收敛
**方法：**
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                    # LoRA rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # 注意力层
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(base_model, lora_config)
```

**训练设置：**
```yaml
learning_rate: 1e-4
batch_size: 8  # 梯度累积 4 步 → 有效 BS=32
epochs: 3
warmup_steps: 500
optimizer: AdamW
loss: CrossEntropyLoss (3-class) or MSELoss (score)
```

**损失函数：**
```python
# Three-Level 判别
loss = CrossEntropyLoss(logits_for_[Yes, Partial, No], label)

# 或 Fine-Grained 评分
loss = MSELoss(predicted_score / 100.0, gt_score / 100.0)
```

#### 4.1.2 Full Fine-Tuning（可选）

仅当 LoRA 效果不佳时使用，需要更多计算资源。

---

### 4.2 强化学习微调（RLHF/RLAIF）⭐ 推荐

#### 4.2.1 为什么需要强化学习？

**核心问题：** RVOS 判别任务的真实目标是 **最终分割质量（J&F Score）**，而非单纯的分类准确率。

**监督学习的局限：**
- 标签是基于 IoU 阈值的离散划分（Yes/Partial/No）
- 无法反映 "选错候选" 对最终分割的影响
- 难以学习候选之间的相对排序关系

**强化学习的优势：**
- 直接优化最终目标（J&F Score）
- 学习在多个候选中做出最优选择
- 自动发现难以监督的判别策略

#### 4.2.2 RLHF 方案设计

**Step 1: 监督微调（SFT）** - 作为 baseline
```
使用 4.1 的 LoRA 方法，得到 SFT 模型
```

**Step 2: Reward Model 训练**

**方法 A: 直接奖励（无需 Reward Model）**
```python
def compute_reward(selected_candidate, ground_truth_mask, all_candidates):
    """
    直接使用分割质量作为奖励
    """
    # 1. IoU-based reward
    iou = compute_iou(selected_candidate.mask, ground_truth_mask)
    reward_iou = iou  # 0-1 范围

    # 2. Ranking reward（鼓励选出最优候选）
    all_ious = [compute_iou(c.mask, ground_truth_mask) for c in all_candidates]
    rank = sorted(all_ious, reverse=True).index(iou) + 1
    reward_rank = 1.0 / rank  # 排名越高奖励越大

    # 3. Binary reward（简化版）
    reward_binary = 1.0 if iou >= 0.7 else -0.5

    # 综合奖励
    reward = 0.5 * reward_iou + 0.3 * reward_rank + 0.2 * reward_binary
    return reward
```

**方法 B: 学习 Reward Model（复杂但更强大）**
```python
# 训练一个独立的奖励模型
class RewardModel(nn.Module):
    def __init__(self, base_vl_model):
        # 基于轻量 VL 模型（如 CLIP + Temporal Encoder）
        self.encoder = base_vl_model
        self.reward_head = nn.Linear(hidden_dim, 1)

    def forward(self, visual_tube, expression, candidate_id):
        features = self.encoder(visual_tube, expression)
        reward = self.reward_head(features)  # 预测 IoU
        return reward

# 训练数据：(Visual Tube, Expression, Candidate) → Ground Truth IoU
```

**Step 3: PPO/DPO 强化学习**

**PPO (Proximal Policy Optimization):**
```python
from trl import PPOTrainer, PPOConfig

ppo_config = PPOConfig(
    model_name="qwen3-vl-rvos-sft",
    learning_rate=1e-5,
    batch_size=16,
    mini_batch_size=4,
    gradient_accumulation_steps=4,
    ppo_epochs=4,
    max_grad_norm=0.5,
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=sft_model,
    ref_model=None,  # 或使用 SFT 模型作为参考
    tokenizer=tokenizer,
    reward_model=reward_model,  # 如果使用方法 B
)

# 训练循环
for batch in dataloader:
    # 1. 模型生成输出（选择候选）
    query_tensors = batch["input_ids"]
    response_tensors = ppo_trainer.generate(query_tensors)

    # 2. 计算奖励
    rewards = [compute_reward(response, gt) for response, gt in zip(responses, batch["ground_truth"])]

    # 3. PPO 更新
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
```

**DPO (Direct Preference Optimization) - 更简单：**
```python
from trl import DPOTrainer, DPOConfig

# 构建偏好对数据
preference_data = []
for sample in dataset:
    candidates = sample["all_candidates"]
    ious = [compute_iou(c, sample["gt"]) for c in candidates]

    # 选择最好和最差的候选作为偏好对
    best_idx = np.argmax(ious)
    worst_idx = np.argmin(ious)

    preference_data.append({
        "prompt": f"Visual Tube + Expression: {sample['expression']}",
        "chosen": candidates[best_idx],  # 更好的候选
        "rejected": candidates[worst_idx],  # 更差的候选
    })

dpo_config = DPOConfig(
    beta=0.1,  # KL 惩罚系数
    learning_rate=5e-6,
    num_train_epochs=1,
)

dpo_trainer = DPOTrainer(
    model=sft_model,
    ref_model=sft_model,  # 参考模型（冻结）
    args=dpo_config,
    train_dataset=preference_dataset,
)

dpo_trainer.train()
```

#### 4.2.3 RLAIF (RL from AI Feedback) - 无需标注

**核心思想：** 使用更强大的 MLLM（如 GPT-4V）作为"评委"，生成偏好数据。

```python
def generate_ai_feedback(visual_tube, expression, candidate_a, candidate_b):
    """
    让 GPT-4V 比较两个候选的匹配程度
    """
    prompt = f"""
    Given the expression: "{expression}"

    Candidate A: [Visual Tube A with 8 annotated frames]
    Candidate B: [Visual Tube B with 8 annotated frames]

    Which candidate better matches the expression? Answer: A or B
    """

    response = gpt4v.generate(prompt, images=[visual_tube_a, visual_tube_b])
    return "A" if "A" in response else "B"

# 生成偏好对数据
for sample in unlabeled_dataset:
    candidates = sample["all_candidates"]

    # 随机采样候选对
    for _ in range(5):
        a, b = random.sample(candidates, 2)
        winner = generate_ai_feedback(visual_tube, expression, a, b)

        preference_data.append({
            "chosen": a if winner == "A" else b,
            "rejected": b if winner == "A" else a,
        })

# 使用 DPO 训练
dpo_trainer.train(preference_data)
```

---

## 5. 实施路线图

### Phase 1: 数据准备（1-2周）
```
1. 运行 SAM3 在 MeViS/Ref-YouTube-VOS 上生成候选
2. 计算每个候选的 IoU，生成标签
3. 构建训练集：20,000 样本（70% train, 15% val, 15% test）
4. 数据增强：改变可视化风格、帧采样策略
```

### Phase 2: 监督学习 Baseline（1周）
```
1. LoRA 微调 Qwen3-VL-8B（或 Qwen2.5-VL-7B）
2. 评估 Top-1 Accuracy 和 J&F Score
3. 错误分析：哪些表达式类型失败？
```

### Phase 3: 强化学习优化（2-3周）
```
Option A: 直接奖励 PPO
1. 使用 Phase 2 的 SFT 模型
2. 定义 IoU-based reward
3. PPO 训练 1-2 epochs

Option B: DPO（推荐）
1. 构建偏好对数据（基于 IoU 排序）
2. DPO 训练
3. 对比 SFT vs. DPO 性能

Option C: RLAIF（探索性）
1. 使用 GPT-4V 生成偏好数据
2. 在无标注数据上扩展训练
```

### Phase 4: 评估与部署（1周）
```
1. 在 MeViS valid_u 上评估完整流程
2. 消融实验：SFT vs. PPO vs. DPO
3. 集成到 CRF-SAM3 Pipeline
```

---

## 6. 预期效果

| 方法 | Top-1 Acc | J&F Score | 训练时间 | 复杂度 |
|------|-----------|-----------|----------|--------|
| **Zero-Shot (当前)** | ~40% | ~45% | - | - |
| **LoRA SFT** | ~65% | ~58% | 8小时 | ⭐ |
| **LoRA + PPO** | ~70% | ~62% | 16小时 | ⭐⭐⭐ |
| **LoRA + DPO** | ~72% | ~64% | 12小时 | ⭐⭐ (推荐) |
| **RLAIF** | ~75% | ~66% | 20小时 | ⭐⭐⭐⭐ |

---

## 7. 技术风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 训练数据不足 | 中 | 高 | 使用数据增强；RLAIF 扩展 |
| 过拟合 | 中 | 中 | Early stopping；LoRA dropout |
| RL 训练不稳定 | 高 | 高 | 使用 DPO 代替 PPO；降低学习率 |
| 计算资源不足 | 低 | 高 | 使用 LoRA；gradient checkpointing |

---

## 8. 参考资源

### 8.1 相关论文
- **RLHF**: InstructGPT (OpenAI, 2022)
- **DPO**: Direct Preference Optimization (Rafailov et al., 2023)
- **PPO**: Proximal Policy Optimization (Schulman et al., 2017)
- **RLAIF**: Constitutional AI (Anthropic, 2022)

### 8.2 工具库
- **Hugging Face TRL**: https://github.com/huggingface/trl (PPO/DPO)
- **PEFT**: https://github.com/huggingface/peft (LoRA)
- **DeepSpeed**: 训练加速

---

## 9. 总结

### 核心建议

**🎯 最优方案：监督学习（LoRA SFT）+ DPO 强化学习**

**原因：**
1. ✅ **LoRA SFT** 快速建立 baseline，参数效率高
2. ✅ **DPO** 比 PPO 更稳定，更易收敛
3. ✅ 直接优化排序能力（选出最优候选）
4. ✅ 训练成本可控（单卡 A100 可训练）

**实施优先级：**
1. **P0**: 数据准备 + LoRA SFT
2. **P1**: DPO 强化学习
3. **P2**: RLAIF 探索（如果效果不理想）

**预期时间：4-6周完成完整流程**
