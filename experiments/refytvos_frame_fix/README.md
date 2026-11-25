# Ref-YouTube-VOS 帧错位检查与修复（2026-11-24）

**问题**：SAM3 和视觉管都按"排序后第几张 JPEG"给帧编号（0, 1, 2, …），但仓库写掩码时
（`run_crf_sam3_dataset.save_masks`，ReVOS 是 `_build_revos_rles`）先用 `int(帧名)` 查，查不到才用位置。
Ref-YTVOS 的帧是每 5 帧取 1 帧（00000, 00005, 00010, …），于是名为 00005 的帧（位置 1）拿到的是位置 5（00025）的掩码；
所有 `int(帧名) < 帧数` 的帧（除 00000）都被写成更靠后的帧的掩码。MeViS、DAVIS 的帧名是连续的，不受影响。

```bash
bash scripts/run.sh check          # CPU 约 2 分钟：受影响帧数；用主线已有预测证明错位（受影响帧与被查到的帧 PNG 完全相同），并比较两类帧的 J
GPU=2 bash scripts/run.sh rerun    # 主线配置原样重跑（词、SAM3、Verify 都不变），只把掩码按帧位置写出，再用 eval_refytros_png.py 评测
```

- `run_fixed.py`：只替换写掩码的函数，其余全部调用仓库原代码；`refytros` / `revos` 两种 runner 都支持。
- `check_frame_index.py`：只读 meta、JPEG 目录、GT PNG 和已有预测；同时统计 ReVOS meta 里有没有同样的情况。
- 路径默认是服务器上的路径，可用 `PY` `REPO` `REFYT` `REVOS_ROOT` `MAINLINE` `OUT` 覆盖。
