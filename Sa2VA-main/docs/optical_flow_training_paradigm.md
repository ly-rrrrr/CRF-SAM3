# 光流增强的训练-推理不对称范式

## 📋 目录

- [概述](#概述)
- [核心思想](#核心思想)
- [方案 A: 多模态输入蒸馏（推荐）](#方案-a-多模态输入蒸馏推荐)
- [方案 B: 双分支知识蒸馏](#方案-b-双分支知识蒸馏)
- [方案 C: 渐进式光流退火](#方案-c-渐进式光流退火)
- [实现代码](#实现代码)
- [实验设计](#实验设计)

---

## 概述

### 问题背景

在视频理解任务中，**光流（Optical Flow）**能提供显式的运动信息，但在实际推理场景中：
- ❌ 计算光流耗时（实时性差）
- ❌ 部署成本高（需要额外的光流估计模型）
- ❌ 某些场景无法获取光流（如单帧推理）

### 解决方案

采用**训练-推理不对称范式**：
- **训练阶段**：提供 RGB + 光流，让模型学习运动模式
- **推理阶段**：仅使用 RGB，模型从帧序列推断运动

这是一种**特权信息学习（Learning Using Privileged Information, LUPI）**思路。

---

## 核心思想

### 训练-推理流程对比

```
训练阶段：
┌─────────────────┐
│  RGB Frame 1    │ ──┐
├─────────────────┤   │
│  Flow 1→2       │   │
├─────────────────┤   ├──> VL Model (Trainable) ──> Score
│  RGB Frame 2    │   │
├─────────────────┤   │
│  Flow 2→3       │   │
└─────────────────┘ ──┘

推理阶段：
┌─────────────────┐
│  RGB Frame 1    │ ──┐
├─────────────────┤   │
│  RGB Frame 2    │   ├──> VL Model (Frozen) ──> Score
├─────────────────┤   │
│  RGB Frame 3    │   │
└─────────────────┘ ──┘
```

### 理论依据

1. **RGB 序列隐含运动信息**：连续帧的差异本身就包含运动线索
2. **VL 模型的泛化能力**：大模型能学习"从显式运动预测隐式运动"的映射
3. **知识蒸馏**：训练时的额外信息帮助模型形成更好的特征表示

---

## 方案 A: 多模态输入蒸馏（推荐）

### ⭐ 特点

- **难度**：★☆☆☆☆（最简单）
- **改动**：最小，只需修改数据加载和 prompt
- **效果**：基础但有效
- **适用**：快速验证光流是否有帮助

---

### 1. 数据构造

#### 1.1 训练数据格式

**输入序列结构**：RGB 和 Flow 帧交替排列

```
[RGB_1, Flow_1→2, RGB_2, Flow_2→3, RGB_3, Flow_3→4, ..., RGB_8]
```

**总帧数**：8 个 RGB 帧 + 7 个光流帧 = **15 帧**

#### 1.2 光流可视化方法

**Middlebury 色轮编码**（推荐）：

```python
import cv2
import numpy as np

def visualize_optical_flow(flow):
    """
    将光流转换为 HSV 色轮可视化

    Args:
        flow: (H, W, 2) 光流数组，flow[:,:,0]=x方向, flow[:,:,1]=y方向

    Returns:
        bgr: (H, W, 3) BGR 彩色图像

    颜色含义：
        - 色相 (Hue): 运动方向
          * 红色 (0°) = 向右
          * 黄色 (60°) = 右下
          * 绿色 (120°) = 向下
          * 青色 (180°) = 向左
          * 蓝色 (240°) = 左上
          * 品红 (300°) = 向上
        - 饱和度 (Saturation): 固定为最大（255）
        - 明度 (Value): 运动速度（越亮越快）
    """
    # 转换为极坐标
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # 创建 HSV 图像
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = angle * 180 / np.pi / 2  # 角度 -> 色相 [0, 180]
    hsv[..., 1] = 255  # 饱和度固定
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # 幅度 -> 明度

    # 转换为 BGR
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def visualize_optical_flow_masked(flow, mask):
    """
    仅可视化对象区域的光流

    Args:
        flow: (H, W, 2) 光流数组
        mask: (H, W) 二值掩码

    Returns:
        bgr: (H, W, 3) 仅对象区域有颜色的光流图
    """
    flow_vis = visualize_optical_flow(flow)

    # 非对象区域显示为黑色
    result = np.where(mask[..., None] > 0, flow_vis, 0)

    return result.astype(np.uint8)
```

#### 1.3 完整数据生成代码

```python
def create_training_visual_tube_with_flow(
    video_path: str,
    all_masks: Dict[int, Dict[int, np.ndarray]],  # {frame_idx: {obj_id: mask}}
    obj_id: int,
    num_keyframes: int = 8,
    visualization_style: str = "bbox"
) -> List[str]:
    """
    创建带光流的训练 Visual Tube

    Args:
        video_path: 视频帧目录路径
        all_masks: 所有帧的对象掩码
        obj_id: 目标对象 ID
        num_keyframes: RGB 关键帧数量
        visualization_style: 可视化风格 ('bbox', 'mask', etc.)

    Returns:
        frame_paths: 保存的帧路径列表 [RGB, Flow, RGB, Flow, ...]
    """
    import os
    import tempfile

    # 获取所有帧文件
    frame_files = sorted(glob.glob(os.path.join(video_path, "*.jpg")))

    # 获取对象的生命周期
    lifespan_frames = [idx for idx in all_masks.keys() if obj_id in all_masks[idx]]
    if len(lifespan_frames) == 0:
        return []

    start_idx, end_idx = min(lifespan_frames), max(lifespan_frames)

    # 均匀采样关键帧
    if len(lifespan_frames) <= num_keyframes:
        keyframe_indices = lifespan_frames
    else:
        step = len(lifespan_frames) / num_keyframes
        keyframe_indices = [lifespan_frames[int(i * step)] for i in range(num_keyframes)]

    # 创建临时目录保存帧
    temp_dir = tempfile.mkdtemp(prefix=f"visual_tube_flow_{obj_id}_")
    frame_paths = []

    for i, frame_idx in enumerate(keyframe_indices):
        # === 1. RGB 帧（带标注） ===
        rgb_frame = cv2.imread(frame_files[frame_idx])
        mask = all_masks[frame_idx][obj_id]

        # 根据 visualization_style 标注
        if visualization_style == "bbox":
            annotated_rgb = draw_bbox_on_frame(rgb_frame, mask)
        elif visualization_style == "mask":
            annotated_rgb = overlay_mask_on_frame(rgb_frame, mask)
        # ... 其他样式

        # 保存 RGB 帧
        rgb_path = os.path.join(temp_dir, f"frame_{i*2:03d}_rgb.jpg")
        cv2.imwrite(rgb_path, annotated_rgb)
        frame_paths.append(rgb_path)

        # === 2. 光流帧（如果不是最后一帧） ===
        if i < len(keyframe_indices) - 1:
            next_frame_idx = keyframe_indices[i + 1]

            # 计算光流
            flow = compute_optical_flow(
                frame_files[frame_idx],
                frame_files[next_frame_idx]
            )

            # 可视化光流（仅对象区域）
            flow_vis = visualize_optical_flow_masked(flow, mask)

            # 添加边框（可选，保持一致性）
            if visualization_style == "bbox":
                flow_vis = draw_bbox_on_frame(flow_vis, mask)

            # 保存光流帧
            flow_path = os.path.join(temp_dir, f"frame_{i*2+1:03d}_flow.jpg")
            cv2.imwrite(flow_path, flow_vis)
            frame_paths.append(flow_path)

    return frame_paths


def compute_optical_flow(frame_path_1: str, frame_path_2: str) -> np.ndarray:
    """
    计算两帧之间的光流

    Args:
        frame_path_1: 第一帧路径
        frame_path_2: 第二帧路径

    Returns:
        flow: (H, W, 2) 光流数组
    """
    import cv2

    frame1 = cv2.imread(frame_path_1)
    frame2 = cv2.imread(frame_path_2)

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 使用 Farneback 算法（CPU 友好）
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5,      # 金字塔缩放
        levels=3,           # 金字塔层数
        winsize=15,         # 窗口大小
        iterations=3,       # 迭代次数
        poly_n=5,           # 多项式展开邻域大小
        poly_sigma=1.2,     # 高斯标准差
        flags=0
    )

    return flow
```

---

### 2. Prompt 设计

#### 2.1 训练阶段 Prompt

```python
TRAINING_PROMPT_WITH_FLOW = """
Please observe these video frames showing an object's appearance and motion.

**Frame Organization:**
- ODD-numbered frames (1st, 3rd, 5th, ...): RGB images with bounding box showing the object
- EVEN-numbered frames (2nd, 4th, 6th, ...): MOTION FLOW visualization between consecutive RGB frames

**How to Read Motion Flow:**
- COLOR indicates DIRECTION of movement:
  * Red/Orange = Moving RIGHT
  * Blue/Cyan = Moving LEFT
  * Green = Moving DOWN
  * Yellow = Moving UP
  * Other colors = Diagonal movements
- BRIGHTNESS indicates SPEED:
  * Brighter = Faster movement
  * Darker = Slower movement
- Black areas = No movement or outside object region

**Your Task:**
Use BOTH the object's appearance (from RGB frames) AND its motion pattern (from flow frames) to make your judgment.

Referring Expression: "{referring_expression}"

Question: Does this object match the referring expression in BOTH appearance AND motion behavior?

Please strictly follow the following reasoning principles:

1. Focus on the overall intent of the referring expression.
2. Grasp the main points of the referring expression, including MOTION cues (e.g., "walking", "jumping", "to the left").
3. Logical verification centered on the object inside the bounding box.

Rating Criteria:
- "Yes" = Perfect Match. The object matches the expression in appearance AND motion.
- "Partial" = Partial Match. Matches appearance but motion is unclear or partially correct.
- "No" = Mismatch. Clear contradiction in appearance or motion.

**Answer with one word only: Yes, Partial, or No:**
"""
```

#### 2.2 推理阶段 Prompt（不变）

```python
INFERENCE_PROMPT = """
Please observe the following consecutive video frames.
The object marked by the bounding box is the **protagonist** (focus subject), while everything outside the bounding box is the background/context.

Referring Expression: "{referring_expression}"

Question: Does the protagonist inside the bounding box exactly match the referring expression?

Please strictly follow the following reasoning principles:

1. Focus on the overall intent of the referring expression.
2. Grasp the main points of the referring expression.
3. Logical verification centered on the object inside the bounding box.

Rating Criteria:
- "Yes" = Perfect Match. The protagonist is the **true object** described by the referring expression.
- "Partial" = Distractor/Bystander. The protagonist matches only **part** of the description but does not fit the complete description.
- "No" = Mismatch. Clearly a wrong object that contradicts the facts described in the referring expression.

**Answer with one word only: Yes, Partial, or No:**
"""
```

---

### 3. 训练流程

#### 3.1 数据准备

**数据集结构**：

```
training_data/
├── video_1/
│   ├── frames/
│   │   ├── 00000.jpg
│   │   ├── 00001.jpg
│   │   └── ...
│   └── annotations.json  # {obj_id: {exp: "...", label: "Yes/Partial/No"}}
├── video_2/
└── ...
```

**annotations.json 格式**：

```json
{
  "video_id": "abc123",
  "objects": [
    {
      "obj_id": 0,
      "expression": "person walking to the left",
      "label": "Yes",
      "frames": [0, 5, 10, 15, 20, 25, 30, 35],
      "masks": {...}
    },
    {
      "obj_id": 1,
      "expression": "person walking to the left",
      "label": "No",
      "frames": [0, 5, 10, 15, 20, 25, 30, 35],
      "masks": {...}
    }
  ]
}
```

#### 3.2 数据加载器

```python
import torch
from torch.utils.data import Dataset, DataLoader

class RVOSFlowDataset(Dataset):
    """RVOS 数据集（带光流）"""

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        use_flow: bool = True,
        transform=None
    ):
        self.data_root = data_root
        self.split = split
        self.use_flow = use_flow
        self.transform = transform

        # 加载标注
        self.samples = self._load_annotations()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        video_path = os.path.join(self.data_root, sample['video_id'], 'frames')

        if self.use_flow:
            # 训练模式：生成 RGB + Flow
            frame_paths = create_training_visual_tube_with_flow(
                video_path=video_path,
                all_masks=sample['masks'],
                obj_id=sample['obj_id'],
                num_keyframes=8,
                visualization_style="bbox"
            )
        else:
            # 推理模式：仅 RGB
            frame_paths = create_visual_tube_rgb_only(
                video_path=video_path,
                all_masks=sample['masks'],
                obj_id=sample['obj_id'],
                num_keyframes=8,
                visualization_style="bbox"
            )

        # 加载图像
        frames = [Image.open(p).convert('RGB') for p in frame_paths]

        if self.transform:
            frames = [self.transform(f) for f in frames]

        return {
            'frames': frames,
            'expression': sample['expression'],
            'label': sample['label'],  # "Yes" / "Partial" / "No"
            'video_id': sample['video_id'],
            'obj_id': sample['obj_id']
        }
```

#### 3.3 训练脚本

```python
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from torch.utils.data import DataLoader

def train_with_optical_flow(
    model_path: str,
    data_root: str,
    output_dir: str,
    num_epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 1e-5
):
    """
    训练带光流的 VL 模型
    """
    # 1. 加载模型
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype="auto",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)

    # 2. 准备数据
    train_dataset = RVOSFlowDataset(
        data_root=data_root,
        split="train",
        use_flow=True  # 训练时使用光流
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    # 3. 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # 4. 训练循环
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            frames = batch['frames']  # [B, N, 3, H, W]
            expressions = batch['expression']
            labels = batch['label']

            # 构造 prompt
            prompts = [
                TRAINING_PROMPT_WITH_FLOW.format(referring_expression=expr)
                for expr in expressions
            ]

            # 准备输入
            inputs = processor(
                text=prompts,
                images=frames,
                return_tensors="pt",
                padding=True
            ).to(model.device)

            # 准备标签（tokenize "Yes"/"Partial"/"No"）
            label_ids = processor.tokenizer(
                labels,
                return_tensors="pt",
                padding=True
            ).input_ids.to(model.device)

            # 前向传播
            outputs = model(**inputs, labels=label_ids)
            loss = outputs.loss

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished, Average Loss: {avg_loss:.4f}")

        # 保存 checkpoint
        model.save_pretrained(os.path.join(output_dir, f"checkpoint-epoch-{epoch+1}"))

    # 5. 保存最终模型
    model.save_pretrained(os.path.join(output_dir, "final"))
    print(f"Training completed! Model saved to {output_dir}/final")


# 使用示例
if __name__ == "__main__":
    train_with_optical_flow(
        model_path="/path/to/Qwen3-VL-8B-Instruct",
        data_root="/path/to/mevis_training_data",
        output_dir="./trained_models/qwen3_with_flow",
        num_epochs=10,
        batch_size=4,
        learning_rate=1e-5
    )
```

#### 3.4 推理脚本（使用训练好的模型）

```python
def inference_without_optical_flow(
    trained_model_path: str,
    video_path: str,
    referring_expression: str,
    all_masks: Dict,
    obj_id: int
):
    """
    推理时不使用光流（仅 RGB）
    """
    # 1. 加载训练好的模型
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        trained_model_path,
        dtype="auto",
        device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(trained_model_path)

    # 2. 准备 RGB-only visual tube
    frame_paths = create_visual_tube_rgb_only(
        video_path=video_path,
        all_masks=all_masks,
        obj_id=obj_id,
        num_keyframes=8,
        visualization_style="bbox"
    )

    # 3. 构造推理 prompt（不提及光流）
    prompt = INFERENCE_PROMPT.format(referring_expression=referring_expression)

    # 4. 准备输入
    frames = [Image.open(p).convert('RGB') for p in frame_paths]
    inputs = processor(
        text=[prompt],
        images=[frames],
        return_tensors="pt"
    ).to(model.device)

    # 5. 生成预测
    with torch.no_grad():
        # 获取 "Yes", "Partial", "No" 的 logits
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            output_scores=True,
            return_dict_in_generate=True
        )

        # 解析输出
        response = processor.decode(outputs.sequences[0], skip_special_tokens=True)

    return response
```

---

### 4. 训练技巧

#### 4.1 数据增强

```python
# 针对光流的数据增强
def augment_flow_and_rgb(rgb_frame, flow, augmentation):
    """
    对 RGB 和光流同步应用数据增强
    """
    if augmentation == "horizontal_flip":
        rgb_frame = cv2.flip(rgb_frame, 1)
        flow = cv2.flip(flow, 1)
        flow[:, :, 0] = -flow[:, :, 0]  # x 方向取反

    elif augmentation == "rotation":
        angle = np.random.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        rgb_frame = cv2.warpAffine(rgb_frame, M, (w, h))
        # 光流也需要旋转和方向调整
        flow_rotated = rotate_flow(flow, angle)

    return rgb_frame, flow_rotated
```

#### 4.2 课程学习（可选）

```python
# 从简单到困难的运动模式
def curriculum_sampling(epoch, total_epochs):
    """
    早期训练：选择运动明显的样本（快速运动）
    后期训练：加入运动微弱的样本（慢速、静止）
    """
    if epoch < total_epochs * 0.3:
        # 早期：快速运动
        min_motion_threshold = 5.0
    elif epoch < total_epochs * 0.7:
        # 中期：中速运动
        min_motion_threshold = 2.0
    else:
        # 后期：所有运动
        min_motion_threshold = 0.0

    return min_motion_threshold
```

#### 4.3 LoRA 微调（推荐）

```python
from peft import LoraConfig, get_peft_model

def train_with_lora(model_path, ...):
    """
    使用 LoRA 高效微调，避免过拟合
    """
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_path)

    # 配置 LoRA
    lora_config = LoraConfig(
        r=16,                  # LoRA rank
        lora_alpha=32,         # LoRA alpha
        target_modules=["q_proj", "v_proj"],  # 目标层
        lora_dropout=0.1,
        bias="none"
    )

    model = get_peft_model(model, lora_config)

    # 训练...
```

---

### 5. 评估指标

```python
def evaluate_motion_understanding(model, test_dataset):
    """
    评估模型对运动的理解能力
    """
    # 1. 整体准确率
    overall_acc = compute_accuracy(model, test_dataset)

    # 2. 运动相关表达式的准确率
    motion_expressions = filter_motion_expressions(test_dataset)
    motion_acc = compute_accuracy(model, motion_expressions)

    # 3. 按运动类型分类
    metrics = {
        'fast_motion': compute_accuracy(model, filter_by_speed(test_dataset, "fast")),
        'slow_motion': compute_accuracy(model, filter_by_speed(test_dataset, "slow")),
        'directional': compute_accuracy(model, filter_by_direction(test_dataset)),
        'action_verbs': compute_accuracy(model, filter_by_action(test_dataset))
    }

    return overall_acc, motion_acc, metrics
```

---

## 方案 B: 双分支知识蒸馏

### ⭐ 特点

- **难度**：★★★☆☆
- **改动**：需要两阶段训练
- **效果**：更强的监督信号
- **适用**：追求极致性能

---

### 1. 架构设计

```
Teacher Model (RGB + Flow):
    - Frozen after pre-training
    - Provides soft targets

Student Model (RGB only):
    - Trained to mimic teacher
    - Can be deployed efficiently
```

### 2. 训练流程

#### Phase 1: 训练 Teacher Model

```python
# 使用 RGB + Flow 训练一个强模型
teacher = train_model(
    data=rgb_flow_pairs,
    architecture="Qwen3-VL-8B",
    epochs=20
)
teacher.eval()
teacher.requires_grad_(False)
```

#### Phase 2: 知识蒸馏到 Student

```python
def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
    alpha: float = 0.5
):
    """
    知识蒸馏损失

    Args:
        student_logits: 学生模型输出 (RGB only)
        teacher_logits: 教师模型输出 (RGB + Flow)
        labels: 真实标签
        temperature: 蒸馏温度（软化分布）
        alpha: 任务损失与蒸馏损失的权重

    Returns:
        总损失
    """
    # 任务损失（硬标签）
    L_task = F.cross_entropy(student_logits, labels)

    # 蒸馏损失（软标签）
    L_kd = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction='batchmean'
    ) * (temperature ** 2)

    # 总损失
    loss = alpha * L_task + (1 - alpha) * L_kd

    return loss


def train_student_with_distillation(
    teacher_model,
    student_model,
    train_loader,
    num_epochs=10
):
    """训练学生模型"""
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-5)

    for epoch in range(num_epochs):
        for batch in train_loader:
            rgb_frames = batch['rgb_frames']
            flow_frames = batch['flow_frames']
            labels = batch['labels']

            # Teacher 前向（RGB + Flow）
            with torch.no_grad():
                teacher_logits = teacher_model(rgb_frames, flow_frames)

            # Student 前向（RGB only）
            student_logits = student_model(rgb_frames)

            # 计算蒸馏损失
            loss = distillation_loss(
                student_logits,
                teacher_logits,
                labels,
                temperature=2.0,
                alpha=0.5
            )

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### 3. 优势

- ✅ 显式地将光流知识"压缩"到 RGB 分支
- ✅ 软标签提供更丰富的监督信号
- ✅ 推理时完全不依赖光流

---

## 方案 C: 渐进式光流退火

### ⭐ 特点

- **难度**：★★☆☆☆
- **改动**：中等，需要动态调整输入
- **效果**：平滑过渡，减小 train-test gap
- **适用**：单阶段训练，简化流程

---

### 1. 核心思想

训练过程中**逐渐减少光流的可见性**，让模型平滑地从"依赖光流"过渡到"仅用 RGB"。

```
Epoch 1-5:   RGB + 100% Flow  (学习运动模式)
Epoch 6-10:  RGB + 75% Flow   (开始依赖 RGB)
Epoch 11-15: RGB + 50% Flow   (逐渐独立)
Epoch 16-20: RGB + 25% Flow   (几乎独立)
Epoch 21+:   RGB only          (完全泛化)
```

### 2. 实现

#### 2.1 退火调度器

```python
def get_flow_alpha(epoch: int, total_epochs: int, schedule: str = "linear"):
    """
    计算当前 epoch 的光流可见度

    Args:
        epoch: 当前 epoch（从 0 开始）
        total_epochs: 总 epoch 数
        schedule: 调度策略 ("linear", "cosine", "step")

    Returns:
        alpha: 光流权重 [0, 1]，1 表示完全可见
    """
    progress = epoch / total_epochs

    if schedule == "linear":
        # 线性衰减
        alpha = max(0.0, 1.0 - progress)

    elif schedule == "cosine":
        # 余弦衰减（更平滑）
        import math
        alpha = 0.5 * (1 + math.cos(math.pi * progress))

    elif schedule == "step":
        # 阶梯衰减
        if progress < 0.25:
            alpha = 1.0
        elif progress < 0.5:
            alpha = 0.75
        elif progress < 0.75:
            alpha = 0.5
        else:
            alpha = 0.25

    return alpha
```

#### 2.2 动态混合光流和 RGB

```python
def create_curriculum_visual_tube(
    rgb_frames: List[np.ndarray],
    flow_frames: List[np.ndarray],
    flow_alpha: float
) -> List[np.ndarray]:
    """
    根据 alpha 动态混合 RGB 和光流

    Args:
        rgb_frames: RGB 帧列表
        flow_frames: 光流可视化帧列表
        flow_alpha: 光流权重 [0, 1]

    Returns:
        mixed_frames: 混合后的帧列表
    """
    mixed_frames = []

    for rgb, flow_vis in zip(rgb_frames, flow_frames):
        if flow_alpha > 0:
            # 加权混合
            mixed = cv2.addWeighted(
                rgb,
                1.0,
                flow_vis,
                flow_alpha,
                0
            )
        else:
            # 完全是 RGB
            mixed = rgb

        mixed_frames.append(mixed)

    return mixed_frames
```

#### 2.3 训练脚本

```python
def train_with_curriculum(
    model_path: str,
    data_root: str,
    output_dir: str,
    num_epochs: int = 20,
    schedule: str = "cosine"
):
    """使用课程学习训练"""
    model = load_model(model_path)
    train_loader = create_dataloader(data_root)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    for epoch in range(num_epochs):
        # 计算当前 epoch 的光流权重
        flow_alpha = get_flow_alpha(epoch, num_epochs, schedule)

        print(f"Epoch {epoch+1}/{num_epochs}, Flow Alpha: {flow_alpha:.2f}")

        for batch in train_loader:
            rgb_frames = batch['rgb_frames']
            flow_frames = batch['flow_frames']

            # 动态混合
            mixed_frames = create_curriculum_visual_tube(
                rgb_frames,
                flow_frames,
                flow_alpha
            )

            # 训练...
            loss = model(mixed_frames, batch['labels'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 保存 checkpoint
        save_checkpoint(model, output_dir, epoch, flow_alpha)
```

### 3. Prompt 设计

动态调整 prompt 中对光流的描述：

```python
def get_curriculum_prompt(referring_expression: str, flow_alpha: float):
    """根据光流可见度调整 prompt"""

    if flow_alpha > 0.8:
        # 早期：强调光流
        prompt = f"""
        RGB frames and MOTION FLOW frames are provided.
        Use both to judge: "{referring_expression}"
        """
    elif flow_alpha > 0.3:
        # 中期：弱化光流
        prompt = f"""
        Video frames with motion hints are provided.
        Judge: "{referring_expression}"
        """
    else:
        # 后期：不提光流
        prompt = f"""
        Video frames are provided.
        Judge: "{referring_expression}"
        """

    return prompt
```

---

## 实现代码

### 完整的 VisualTubeBuilder 修改

```python
class VisualTubeBuilder:
    """Visual Tube 构建器（支持光流）"""

    def __init__(
        self,
        num_tube_frames: int = 8,
        visualization_style: str = "bbox",
        use_optical_flow: bool = False,
        flow_alpha: float = 1.0,
        flow_method: str = "farneback"
    ):
        """
        Args:
            num_tube_frames: 关键帧数量
            visualization_style: 可视化风格
            use_optical_flow: 是否使用光流
            flow_alpha: 光流可见度 [0, 1]
            flow_method: 光流计算方法 ("farneback", "raft")
        """
        self.num_tube_frames = num_tube_frames
        self.visualization_style = visualization_style
        self.use_optical_flow = use_optical_flow
        self.flow_alpha = flow_alpha
        self.flow_method = flow_method

    def build_tubes(
        self,
        video_path: str,
        all_masks: Dict[int, Dict[int, np.ndarray]],
        referring_expression: str
    ) -> Dict[int, Dict]:
        """
        构建 Visual Tubes（可选光流增强）
        """
        visual_tubes = {}

        for obj_id in self._get_all_obj_ids(all_masks):
            if self.use_optical_flow:
                # 带光流的 tube
                tube_data = self._build_tube_with_flow(
                    video_path,
                    all_masks,
                    obj_id
                )
            else:
                # 纯 RGB tube
                tube_data = self._build_tube_rgb_only(
                    video_path,
                    all_masks,
                    obj_id
                )

            visual_tubes[obj_id] = tube_data

        return visual_tubes

    def _build_tube_with_flow(
        self,
        video_path: str,
        all_masks: Dict,
        obj_id: int
    ) -> Dict:
        """构建带光流的 tube"""
        frame_files = sorted(glob.glob(os.path.join(video_path, "*.jpg")))
        keyframe_indices = self._sample_keyframes(all_masks, obj_id)

        temp_dir = tempfile.mkdtemp()
        frame_paths = []

        for i, frame_idx in enumerate(keyframe_indices):
            # RGB frame
            rgb = cv2.imread(frame_files[frame_idx])
            mask = all_masks[frame_idx][obj_id]
            rgb_annotated = self._annotate_frame(rgb, mask)

            rgb_path = os.path.join(temp_dir, f"{i*2:03d}_rgb.jpg")
            cv2.imwrite(rgb_path, rgb_annotated)
            frame_paths.append(rgb_path)

            # Flow frame (if not last)
            if i < len(keyframe_indices) - 1:
                flow = self._compute_flow(
                    frame_files[frame_idx],
                    frame_files[keyframe_indices[i+1]]
                )

                flow_vis = visualize_optical_flow_masked(flow, mask)

                # Apply alpha blending (for curriculum)
                if self.flow_alpha < 1.0:
                    flow_vis = cv2.addWeighted(
                        rgb, 1 - self.flow_alpha,
                        flow_vis, self.flow_alpha,
                        0
                    )

                flow_path = os.path.join(temp_dir, f"{i*2+1:03d}_flow.jpg")
                cv2.imwrite(flow_path, flow_vis)
                frame_paths.append(flow_path)

        return {
            'frame_paths': frame_paths,
            'num_frames': len(frame_paths),
            'has_flow': True
        }

    def _compute_flow(self, frame_path_1: str, frame_path_2: str):
        """计算光流"""
        if self.flow_method == "farneback":
            return compute_optical_flow(frame_path_1, frame_path_2)
        elif self.flow_method == "raft":
            # TODO: 集成 RAFT
            raise NotImplementedError("RAFT not implemented yet")
```

---

## 实验设计

### 消融实验

| 实验组 | RGB | Flow (Train) | Flow (Test) | 预期效果 |
|-------|-----|--------------|-------------|---------|
| Baseline | ✓ | ✗ | ✗ | 基准性能 |
| Oracle | ✓ | ✓ | ✓ | 性能上界 |
| **方案 A** | ✓ | ✓ | ✗ | 主要对比 |
| **方案 B** | ✓ | ✓ (KD) | ✗ | 更强监督 |
| **方案 C** | ✓ | ✓→✗ | ✗ | 平滑过渡 |

### 评估维度

1. **整体准确率**：所有表达式的平均准确率
2. **运动相关**：包含动作动词的表达式（"walking", "jumping", "running"）
3. **方向相关**：包含方向描述的表达式（"to the left", "backwards", "upward"）
4. **速度相关**：包含速度描述的表达式（"fast", "slowly", "quickly"）
5. **静态对象**：不包含运动的表达式（对照组）

---

## 总结

### 方案对比

| 方案 | 复杂度 | 训练时间 | 性能 | 推荐度 |
|------|--------|---------|------|--------|
| A: 多模态输入 | 低 | 1x | 中 | ⭐⭐⭐⭐⭐ |
| B: 双分支蒸馏 | 高 | 2x | 高 | ⭐⭐⭐⭐ |
| C: 渐进退火 | 中 | 1x | 中-高 | ⭐⭐⭐⭐ |

### 实施建议

1. **快速验证**：先实施方案 A，验证光流是否真的有帮助
2. **性能优化**：如果有效，尝试方案 C 减小 train-test gap
3. **极致性能**：如果需要最佳性能，实施方案 B

### 关键成功因素

- ✅ 光流可视化质量（色轮编码 vs 箭头）
- ✅ Prompt 设计（明确告知模型如何理解光流）
- ✅ 数据多样性（覆盖各种运动模式）
- ✅ 训练策略（LoRA、课程学习）

---

**参考文献**：

1. Vapnik, V., & Vashist, A. (2009). A new learning paradigm: Learning using privileged information.
2. Hinton, G., et al. (2015). Distilling the knowledge in a neural network.
3. Sun, D., et al. (2018). PWC-Net: CNNs for optical flow using pyramid, warping, and cost volume.
