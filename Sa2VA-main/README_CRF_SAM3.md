# CRF-SAM3: Concept Recall & Spatio-Temporal Filtering with SAM3

[![Paper](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

> A novel paradigm for Referring Video Object Segmentation (RVOS) that shifts from **generative localization** to **discriminative filtering**.

---

## 📖 Overview

**CRF-SAM3** tackles the core challenge in RVOS: complex referring expressions that involve temporal actions (e.g., *"the dog that stops running and sits down"*). Unlike traditional methods that ask MLLMs to directly localize objects from limited frames, our approach:

1. **Recalls ALL candidates** of the target category using SAM3's powerful segmentation
2. **Constructs spatio-temporal representations** (Visual Tubes) for each candidate
3. **Discriminates and filters** using MLLM's semantic understanding with full temporal context

### Key Advantages

| Aspect | Traditional MAPP-SAM3 | **CRF-SAM3** |
|--------|----------------------|--------------|
| **Fault Tolerance** | MLLM must find correct object immediately | Recalls all candidates, filters later |
| **Action Understanding** | Limited frames, hard to infer actions | Visual Tubes provide complete motion trajectory |
| **Prompt Engineering** | Complex phrase generation with semantic gap | Simple category words, no semantic gap |
| **Robustness** | Fails if MLLM makes wrong initial choice | Multiple opportunities to identify correct target |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CRF-SAM3 Pipeline                        │
└─────────────────────────────────────────────────────────────────┘

Input: Complex Referring Expression
       "the elephant that turns around and walks away"
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: Concept-Level Recall                                  │
│  ─────────────────────────────────────────────────────────      │
│  1a. ConceptExtractor (MLLM)                                    │
│      Expression → Category: "elephant", Count: 1                │
│                                                                  │
│  1b. SAM3ConceptRecall                                          │
│      Text: "elephant" → SAM3 → Detect ALL elephants (A, B, C)  │
│      Output: {Elephant A, Elephant B, Elephant C}               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 2: Visual Tube Construction                              │
│  ─────────────────────────────────────────────────────────      │
│  VisualTubeBuilder:                                             │
│    For each candidate (A, B, C):                                │
│      • Sample K keyframes across object's lifespan              │
│      • Draw red bounding boxes (SoM - Set of Mark)              │
│      • Preserve spatial context (no cropping)                   │
│                                                                  │
│  Output: Visual Tubes with temporal motion patterns             │
│    Tube_A: [frame_0, frame_5, ..., frame_40] (with red boxes)  │
│    Tube_B: [frame_0, frame_5, ..., frame_40] (with red boxes)  │
│    Tube_C: [frame_0, frame_5, ..., frame_40] (with red boxes)  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 3: Semantic Discrimination & Filtering                   │
│  ─────────────────────────────────────────────────────────      │
│  SemanticDiscriminator (MLLM):                                  │
│    For each Visual Tube:                                        │
│      MLLM observes red-boxed object across K frames             │
│      Question: "Does this match 'turns around and walks away'?" │
│                                                                  │
│  Scores:                                                         │
│    Elephant A: 0.95 ✓ (faces camera → turns → walks away)      │
│    Elephant B: 0.12 ✗ (static throughout)                      │
│    Elephant C: 0.08 ✗ (always facing away)                     │
│                                                                  │
│  Output: Top-K objects (Elephant A)                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    Final Segmentation Masks
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
cd /path/to/Sa2VA-main

# Install dependencies
pip install torch torchvision transformers Pillow numpy pyyaml tqdm

# Install SAM3 (if not already installed)
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
```

### Download Models

```bash
# Download Qwen2.5-VL-7B-Instruct (MLLM)
# Already at: /mnt/csip-200/share/kongjiawei/Qwen2.5-VL-7B-Instruct

# Download SAM3 checkpoint
# Already at: /mnt/csip-200/share/kongjiawei/sam2/checkpoints/sam2.1_hiera_large.pt
```

### Basic Usage

```bash
# Process single video (debugging)
python run_crf_sam3_dataset.py \
    --config configs/crf_sam3_config.yaml \
    --max_videos 1

# Process entire dataset
python run_crf_sam3_dataset.py \
    --config configs/crf_sam3_config.yaml

# Custom configuration
python run_crf_sam3_dataset.py \
    --config configs/crf_sam3_config.yaml \
    --num_tube_frames 12 \
    --output_dir my_results
```

### Python API

```python
from crf_sam3_demo import CRFSAM3Pipeline, load_video_frames

# Initialize pipeline
pipeline = CRFSAM3Pipeline(
    mllm_path="/path/to/Qwen2.5-VL-7B-Instruct",
    sam3_checkpoint="/path/to/sam2.1_hiera_large.pt",
    num_tube_frames=8
)

# Load video
video_frames = load_video_frames("path/to/video/frames")

# Run inference
referring_expression = "the elephant that turns around and walks away"
masks = pipeline.predict(video_frames, referring_expression)

# masks: Dict[frame_idx][obj_id] -> binary mask (H, W)
```

---

## ⚙️ Configuration

All parameters can be configured via `configs/crf_sam3_config.yaml`:

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pipeline.num_tube_frames` | 8 | Number of frames per Visual Tube (↑ for longer actions) |
| `visual_tube.sampling_strategy` | `"uniform"` | Keyframe sampling: `"uniform"` or `"adaptive"` |
| `discrimination.scoring_method` | `"binary"` | Scoring: `"binary"` (Yes/No) or `"logits"` (probability) |
| `discrimination.selection_strategy` | `"count_based"` | Top-K selection or threshold-based |
| `dataset.max_videos` | `null` | Limit videos for debugging (set to int) |

### Example: High-Quality Mode

```yaml
# configs/crf_sam3_high_quality.yaml
pipeline:
  num_tube_frames: 16  # More frames for complex actions

visual_tube:
  sampling_strategy: "adaptive"  # Denser sampling in high-motion regions

discrimination:
  scoring_method: "logits"  # Use probability instead of binary
  selection_strategy: "threshold_based"
  threshold: 0.7  # Higher threshold for precision
```

---

## 📊 Evaluation

### Run Evaluation

```bash
# Generate predictions
python run_crf_sam3_dataset.py --config configs/crf_sam3_config.yaml

# Evaluate on MeViS
python eval_mapp_sam3.py \
    --pred_dir crf_sam3_results \
    --gt_dir /path/to/mevis/valid_u/Annotations \
    --meta_file /path/to/mevis/valid_u/meta_expressions.json \
    --output eval_crf_sam3.json
```

### Expected Results

| Method | J&F | J | F |
|--------|-----|---|---|
| MAPP-SAM3 (baseline) | XX.X | XX.X | XX.X |
| **CRF-SAM3** | **XX.X** | **XX.X** | **XX.X** |

*(Results to be updated after full evaluation)*

---

## 🔧 Advanced Usage

### Custom MLLM

Replace Qwen2.5-VL with other video-capable MLLMs:

```python
# In crf_sam3_demo.py, modify ConceptExtractor and SemanticDiscriminator
from transformers import YourMLLM, YourProcessor

class ConceptExtractor:
    def __init__(self, mllm_path):
        self.mllm = YourMLLM.from_pretrained(mllm_path)
        self.processor = YourProcessor.from_pretrained(mllm_path)
        # ... adapt API calls
```

### Adaptive Visual Tube Sampling

Enable adaptive sampling for high-motion regions:

```yaml
# configs/crf_sam3_config.yaml
visual_tube:
  sampling_strategy: "adaptive"
```

Implementation in `VisualTubeBuilder`:
```python
# Compute optical flow to detect high-motion regions
# Sample more keyframes from these regions
```

### Logits-Based Scoring

Extract "Yes" token probability for more accurate scoring:

```python
# In SemanticDiscriminator._score_tube()
with torch.no_grad():
    outputs = self.mllm.generate(
        **inputs,
        return_dict_in_generate=True,
        output_scores=True
    )

# Extract logits for "Yes" token
yes_token_id = self.processor.tokenizer.encode("Yes")[0]
score = outputs.scores[0][0, yes_token_id].softmax(dim=-1).item()
```

---

## 📂 Project Structure

```
Sa2VA-main/
├── crf_sam3_demo.py              # Core pipeline implementation
├── run_crf_sam3_dataset.py       # Batch inference script
├── eval_mapp_sam3.py             # Evaluation script (reusable)
├── configs/
│   ├── crf_sam3_config.yaml      # Default configuration
│   └── crf_sam3_high_quality.yaml # High-quality preset
├── README_CRF_SAM3.md            # This file
└── crf_sam3_results/             # Output directory
    └── <video_id>/
        └── <exp_id>/
            ├── 00000.png
            ├── 00001.png
            └── ...
```

---

## 🧪 Debugging & Visualization

### Save Visual Tubes

```yaml
# configs/crf_sam3_config.yaml
output:
  save_visual_tubes: true
```

This saves annotated frames with red boxes to `{output_dir}/visual_tubes/{video_id}/{obj_id}/`.

### Print MLLM Responses

```yaml
logging:
  print_mllm_responses: true
```

Example output:
```
[ConceptExtractor] Raw response:
{"category": "elephant", "num_targets": 1, "reasoning": "Singular 'the elephant'"}

[Discriminator] Object 0: Response='Yes', Score=0.950
[Discriminator] Object 1: Response='No', Score=0.120
```

---

## 🐛 Troubleshooting

### Issue: SAM3 detects too many candidates

**Solution:** SAM3's Presence Head is highly sensitive. This is expected behavior. The Discriminator stage will filter out false positives.

### Issue: CUDA OOM during MLLM inference

**Solution 1:** Reduce `num_tube_frames`:
```yaml
pipeline:
  num_tube_frames: 4  # Reduce from 8
```

**Solution 2:** Use gradient checkpointing:
```python
self.mllm.gradient_checkpointing_enable()
```

### Issue: Low discrimination scores for all candidates

**Possible causes:**
1. Visual Tubes don't cover the action period → Increase `num_tube_frames`
2. MLLM doesn't understand the expression → Check prompt formatting
3. Red boxes are too small/large → Adjust `box_width` in config

---

## 📝 Citation

If you use CRF-SAM3 in your research, please cite:

```bibtex
@article{crfsam3,
  title={CRF-SAM3: Concept Recall and Spatio-Temporal Filtering for Referring Video Object Segmentation},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

---

## 🙏 Acknowledgments

- **SAM3**: [Segment Anything Model 3](https://github.com/facebookresearch/sam2)
- **Qwen2.5-VL**: [Qwen Vision-Language Model](https://github.com/QwenLM/Qwen2-VL)
- **MeViS Dataset**: [Moving Expression Video Segmentation](https://henghuiding.github.io/MeViS/)

---

## 📄 License

This project is released under the Apache 2.0 License. See [LICENSE](LICENSE) for details.

---

## 📧 Contact

For questions or collaboration:
- GitHub Issues: [https://github.com/your-repo/issues](https://github.com/your-repo/issues)
- Email: your.email@example.com

---

**Happy Segmenting! 🎯**
