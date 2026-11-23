# p1a_spatial_recall — Ground 响应图门控的 SAM3 召回补全（独立实验包）

不修改仓库，不训练，只读调用 `Sa2VA-main/category_grounding_mvp`（Ground）和 `Sa2VA-main/SAM3-DMS`（冻结 SAM3）。
设计依据见同目录 `设计说明.md`。

```
p1a/
  common.py          帧顺序 / 候选缓存读取 / GT 读取（palette, png_dir, revos）/ RLE / J
  heatmap_dump.py    阶段1（GPU，仅 Qwen3-VL）：导出 Ground 8 个 slot 的 response map
  heatmap_diag.py    CPU：热力图能否当“召回门控”（GT 只用于评分）
  seeds.py           MAX 图 → 连通块 → 与已有候选的覆盖率 → 未覆盖证据
  sam3_ops.py        SAM3 会话封装：text_recall / detect_on_frame(低阈值) / track_seed / exemplar_recall
  check_sam3_api.py  服务器上 1 分钟自检 4 个 SAM3 调用
  run_p1a.py         阶段2（GPU，仅 SAM3）：补召回 + oracle J*
  analyze.py         汇总成 Markdown 表；有 det_probe 时给每条失败判定原因
  failure_taxonomy.py  CPU：只读现有候选缓存 + GT，把 J*<0.5 的失败分成 7 类
tests/test_offline.py   CPU 合成数据 + 假 SAM3 的端到端测试（已在开发环境通过）
tests/test_taxonomy.py  失败分型的单元测试（已通过）
```

两阶段分开跑，Qwen3-VL 与 SAM3 不同时占显存（阶段1约 10 GB bf16；阶段2 SAM3 约 4–8 GB，可加 `--offload-video-to-cpu`）。

## 0. 环境

用原服务器的 venv 即可（需要 torch、transformers、pycocotools、scipy、opencv）。在本包根目录运行 `python -m p1a.xxx`。

```bash
REPO=/mnt/csip-500/share/kongjiawei/Sa2VA-main
PY=/mnt/csip-500/share/kongjiawei/Sa2VA/.venv/bin/python
DATA=/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas
```

## 1. 先自检 SAM3 调用（必须）

```bash
$PY -m p1a.check_sam3_api --dms-repo $REPO/SAM3-DMS --sam3-checkpoint <sam3.pt> \
   --video $DATA/davis17/train/JPEGImages/tuk-tuk --text girl
```
看三件事：`detect_on_frame` 在 0.7 → 0.2 时检测数是否增加；`track_seed` 返回帧数是否接近视频长度；`exemplar_recall` 能否跑通。
若 `track_seed` 返回 0 帧，说明“只有 tracker 点提示、没有文本”的会话在这个 SAM3-DMS 版本下不传播，把输出贴给我，我改成在文本会话里追加 obj_id 的写法。

## 2. 阶段1：导出热力图（DAVIS train 失败子集为例）

```bash
CUDA_VISIBLE_DEVICES=0 $PY -m p1a.heatmap_dump --repo $REPO \
  --model-path $REPO/Qwen3-VL-4B-Instruct \
  --query-module-dir <DAVIS/YTVOS 用的 Ground checkpoint> \
  --samples $REPO/crf_sam3_attn/cache_reports/davis17_train_dms_recall24_keep7_20260620_max_j_lt_0p5.json \
  --video-root $DATA/davis17/train/JPEGImages \
  --cache-root $REPO/crf_sam3_attn/offline_cache_davis17_dms_recall24_keep7_20260620 --split train \
  --frames-per-sample 3 --out runs/heat_davis_fail --bf16
```
ReVOS 加 `--max-image-side 448`（与 `predict_query_grounding_revos_meta.py` 一致）。`--frames-per-sample` 先保持与 Ground 推理一致（3），之后可做 5/8 的消融。

## 3. CPU 诊断：热力图能否当门控

```bash
$PY -m p1a.heatmap_diag --heat-dir runs/heat_davis_fail --video-root $DATA/davis17/train/JPEGImages \
  --gt palette --gt-kw '{"annotation_root": "'$DATA'/davis17/train/Annotations", "meta_expressions": "'$DATA'/davis17/meta_expressions/train/meta_expressions.json"}' \
  --cache-root $REPO/crf_sam3_attn/offline_cache_davis17_dms_recall24_keep7_20260620 --out runs/diag_davis_fail
```
关键输出：`frac_frames_gt_found`（GT 区域落在某个连通块里的帧比例）、`gt_blob_pur`（实例级还是类别级）、`uncovered`（失败样本里，GT 所在块未被现有候选覆盖的比例，也就是门控能否触发）。
用你 11-12 那批 4 例 npz 也能直接跑：`--diagnostic-dir <.../results>`。

## 4. 阶段2：补召回

```bash
CUDA_VISIBLE_DEVICES=0 $PY -m p1a.run_p1a \
  --samples <同上失败清单> --video-root $DATA/davis17/train/JPEGImages \
  --heat-dir runs/heat_davis_fail \
  --cache-root $REPO/crf_sam3_attn/offline_cache_davis17_dms_recall24_keep7_20260620 --split train \
  --gt palette --gt-kw '<同上>' \
  --dms-repo $REPO/SAM3-DMS --sam3-checkpoint <sam3.pt> \
  --modes rescue,point,oracle_gt --det-thresh 0.2 --budget 2 --max-new 3 \
  --out runs/p1a_davis_fail
$PY -m p1a.analyze runs/p1a_davis_fail/results.jsonl --total-n 553 --total-recall 73.06 --out runs/p1a_davis_fail/summary.md
```
- 可断点续跑（已写入 results.jsonl 的样本会跳过）。
- `exemplar` 模式是对照组（文本 + 热力图框作为 exemplar），默认不跑，需要时加进 `--modes`。
- **误触发代价**：再从 J\*≥0.5 的样本里随机抽 200 条跑一次（只需 `rescue,point`），看平均新增候选数。这决定 Verify 端要多承受多少干扰。
- 没有离线缓存的验证集：去掉 `--cache-root`，加 `--recompute-text --category-json <Ground 预测 JSON>`，会按论文协议（recall24/keep7）先跑文本召回。

## 4b. 诊断：失败到底是"词"还是"SAM3"（不需要热力图）

对应 `调研/候选质量不足_原因与对策_20261119.md`。建议按这个顺序跑，每步都只需要已有缓存。

**第 1 步（CPU，几分钟）：失败分型**
```bash
GTKW_DAVIS='{"annotation_root": "'$DATA'/davis17/train/Annotations", "meta_expressions": "'$DATA'/davis17/meta_expressions/train/meta_expressions.json"}'
$PY -m p1a.failure_taxonomy --cache-root $REPO/crf_sam3_attn/offline_cache_davis17_dms_recall24_keep7_20260620 --split train \
  --video-root $DATA/davis17/train/JPEGImages --gt palette --gt-kw "$GTKW_DAVIS" --out runs/taxonomy_davis
```
Ref-YTVOS 换成对应缓存、`$DATA/ref_youtube_vos/train/JPEGImages` 与它的 Annotations / meta_expressions。输出 `taxonomy_summary.json`（各类占比）和逐条 `taxonomy.csv`。
加 `--all` 会把成功样本也写进 csv，方便比较目标面积等。

**第 2 步（GPU，只用 SAM3）：检测探针 + oracle 种子 + 放宽启发式 + 锚帧重传播**
```bash
CUDA_VISIBLE_DEVICES=0 $PY -m p1a.run_p1a \
  --samples $REPO/crf_sam3_attn/cache_reports/davis17_train_dms_recall24_keep7_20260620_max_j_lt_0p5.json \
  --video-root $DATA/davis17/train/JPEGImages \
  --cache-root $REPO/crf_sam3_attn/offline_cache_davis17_dms_recall24_keep7_20260620 --split train \
  --gt palette --gt-kw "$GTKW_DAVIS" \
  --dms-repo $REPO/SAM3-DMS --sam3-checkpoint <sam3.pt> \
  --modes det_probe,oracle_gt,repropagate,text_cfg \
  --sam3-overrides '{"new_det_thresh":0.5,"det_nms_thresh":0.3,"hotstart_delay":0,"hotstart_unmatch_thresh":0,"hotstart_dup_thresh":0,"suppress_overlapping_based_on_recent_occlusion_threshold":0.0,"recondition_every_nth_frame":8}' \
  --out runs/diag_davis_fail
$PY -m p1a.analyze runs/diag_davis_fail/results.jsonl --total-n 542 --total-recall 73.06 --out runs/diag_davis_fail/summary.md
```
- `det_probe`：在 3 帧（与 Ground 同样的取帧方式）上把检测阈值降到 0.05，记录与 GT 最匹配的检测的 IoU、分数和名次。不产生新候选，每条约几秒。
- `oracle_gt`：GT 框当 tracker 种子，是"SAM3 被告知是哪一个之后能做到多好"的上限。
- `repropagate`：可部署。每个已有候选取面积最大的一帧当锚点，重新用 tracker 双向传播，补上被启发式提前杀掉或晚出现的帧。
- `text_cfg`：同一个词、放宽 SAM3 视频启发式后重跑一次文本召回。`hotstart_unmatch_thresh` / `hotstart_dup_thresh` 必须 ≤ `hotstart_delay`，所以三者一起置 0。
  报告里同时给"并集"和"单独替换"两个 J\*，后者才是能不能直接换配置的依据。
- keep7 截断损失：再跑一次 `--modes text_cfg --sam3-overrides '{}' --keep-top-k 24`（默认启发式，只是不截断到 7 个），`j_star_replace` 和原 J\* 的差就是"按分数和只留 7 个"丢掉的部分。
- 失败清单请用你这次统计（542 条 / 146 条失败）对应的那份；上面的 `_max_j_lt_0p5.json` 是 6 月 overlay 补跑前的旧清单（166 条）。
- 放宽启发式会增加误检，所以 `text_cfg` 和 `repropagate` 都要在 200 条成功样本上再跑一次，看平均新增候选数和"单独替换"时丢了几条。

**第 3 步（可选）：换词探针**
准备一个 `{"<video_id>/<exp_id>": "<另一个词>"}` 的 JSON（例如 Ref-YTVOS 训练集里 YouTube-VOS `meta.json` 的对象类别，或人工把 girl/boy/man 统一成 person），加
`--alt-word-json alt.json --modes det_probe,text_word`。这只是诊断用，不是论文方法。

`analyze` 有 `det_probe` 时会给每条失败判一个原因，写到 `summary.causes.jsonl`：

| 原因 | 判定 | 含义 |
|---|---|---|
| lost_after_detection | 主词检测到目标（IoU≥0.5）且分数≥0.7 | 检测器看得见，后面的跟踪 / 启发式 / keep7 把它丢了 |
| below_threshold | 主词能检测到，但分数<0.7 | 被 0.5 / 0.7 阈值挡掉 |
| word | 主词检测不到，换词能检测到 | 词的问题 |
| detector_blind | 两个词都检测不到，但 GT 种子能跟踪好 | 文本检测器认不出，分割和跟踪没问题 |
| seg_or_track_limit | GT 种子也做不好 | SAM3 分割或跟踪能力本身不够 |

## 5. 输出字段

`results.jsonl` 每行：`j_old`（原候选 J*），`n_uncovered`，`triggered`，以及每个模式的 `added` / `j_new_tubes` / `j_star` / `info`（种子帧、检测分数、框或点）。`--save-tubes` 会把新 tube 存成 RLE，后续可直接接 Verify。

J 的口径：GT 有标注的帧上逐帧 IoU 取平均（两者都空记 1，预测缺帧记空）。新旧 J* 用同一脚本计算，可直接比较；绝对值可能和缓存 meta 里的 `max_j` 有细微差别。

## 6. ReVOS valid：v2 热力图辅助扩展候选池（2026-11-22）

接 ground_v2 第 4 步的结果，不重跑文本召回：候选直接读 `ground_v2/runs/pool_revos/pools/`（v2 词的召回池），
热力图读 `ground_v2/runs/gv2_revos/heat_revos_valid/`，GT 用 ReVOS 的 `mask_dict.json`。
把本目录放在 `ground_v2/` 旁边，路径都从 `ground_v2/scripts/env.sh` 取，其余用环境变量覆盖（`SAM3_CKPT`、`POOL`、`HEAT`、`CATS`、`OUT`）。

```bash
GPU=2 bash scripts/revos_v2.sh check          # 必须先跑：SAM3 点/框跟踪接口自检
GPU=2 LIMIT=10 bash scripts/revos_v2.sh zero  # 试跑
GPU=2 bash scripts/revos_v2.sh zero           # v2 词 0 候选（约 420 条）
GPU=2 bash scripts/revos_v2.sh ok             # 300 条本来可用的：误触发代价
GPU=2 bash scripts/revos_v2.sh fail           # 有候选但不可用（约 1,250 条）
bash scripts/revos_v2.sh summary              # runs/revos_v2/summary.md
```

- 子集由 `p1a.revos_subsets` 从 `per_expression.csv` 划分：zero（0 候选）/ fail（有候选，J* < 0.5）/ ok（抽 300 条）。
- 方式：`rescue`（低阈值 0.2 检测，限定在未被候选覆盖的热区里）、`point`（热区峰值点直接做 tracker 种子，不依赖词，0 候选时主要靠它）、
  `oracle_gt`（GT 框做种子的上限，只作参照）。
- `p1a.revos_summary`：各子集救回条数、新增候选数，以及全量 Recall@0.5（未跑的样本保留第 4 步的值），并自检与第 4 步的 J* 一致。
- 同一视频的样本共用一个 SAM3 会话（帧只解码一次）；单条出错不会中断，重跑同一命令会补上。

## 7. ReVOS valid：最终 J&F（Verify 复评，2026-11-23）

用论文原版 Verify（同一份 yaml：Qwen3-VL 抽 num_targets、同样的视觉管、同一个 yes/no 打分器）给三种候选池打分，
再用官方 `tools/eval/eval_revos.py` 出 J&F。三行：

| 行 | 候选 | 作用 |
|---|---|---|
| old | 旧词的文本召回池 | 复现论文（不含 backup 词），对照 59.74 |
| v2 | v2 词的文本召回池 | 只换词 |
| v2_point | v2 词池 + 热力图点种子轨迹 | 本节方法 |

```bash
GPU=2 bash scripts/verify_v2.sh check     # 约 5 分钟：20 条试跑，并和论文那次的 scores.json 逐候选对分数
GPU=2 bash scripts/verify_v2.sh all       # points（约 3–4 h，SAM3）→ predict（约 4–6 h，Qwen3-VL）→ eval（CPU）
# 也可以分步：points / predict / eval；都能断点续跑
```

- 三行共用一次打分：并集里每个候选只打一次分，再在各自子集里取前 num_targets（打分器逐候选独立打分，所以和分别跑等价）。
- SAM3 不加载（`build_sam3_recall_backend` 被替换）；不用 backup 词和语义扩展，0 候选就输出空掩码。
- `points` 在全部 5,822 条上跑 point 模式并保存轨迹到 `runs/revos_v2/point_all/tubes/`；没有未覆盖热区的样本在 CPU 上直接跳过，不开 SAM3。
- 输出：`runs/verify_v2/scores.jsonl`（每条的候选、分数、各行选择）、`runs/verify_v2/<行>/results.json` 与官方评测结果、`runs/verify_v2/compare.md`。
