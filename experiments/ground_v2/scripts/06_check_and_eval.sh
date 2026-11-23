#!/usr/bin/env bash
# 第 1、2 步一次跑完：检查训练结果 -> 在 val（1,041 条）上评测 A / E / E′ -> 打印对比表
#
# 用法（在 ground_v2/ 目录下，单卡即可，评测不需要多卡）：
#   GPU=2 bash scripts/06_check_and_eval.sh                  # A、E、E′
#   GPU=2 EXTRA=1 bash scripts/06_check_and_eval.sh          # 另外加跑：A 按 336（ReVOS 实际推理设置）、E 的 checkpoint-1500
#   GPU=2 MAX=50 bash scripts/06_check_and_eval.sh           # 先只跑 50 条试通流程
#   GPU=2 ROWS="A E" bash scripts/06_check_and_eval.sh       # 只跑指定的行（A E E_prime A336 E1500）
#
# 所有输出放在 runs/eval_<时间>/：steps.log、health.txt、train_curves.png、各行的 log 和 json、compare.md
# 已经跑完的行会跳过（json 已存在），中断后重跑同一命令即可续上（用 OUTDIR=runs/eval_xxx 指定同一目录）
set -uo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${GPU:-${CUDA_VISIBLE_DEVICES:-0}}
source scripts/env.sh
OUTDIR=${OUTDIR:-runs/eval_$(date +%Y%m%d_%H%M)}
mkdir -p "$OUTDIR"
MAXARG=(); [ -n "${MAX:-}" ] && MAXARG=(--max-samples "$MAX")
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUTDIR/steps.log"; }

# ---------------------------------------------------------------- 第 1 步：训练结果检查
log "== 第 1 步：检查训练结果 ($RUN)"
if grep -qaE "\[gv2\] (WARNING|STOPPED)" "$RUN.log" 2>/dev/null; then
  log "训练日志里有 NaN 警告或提前停止，先不要评测，把 $RUN.log 发给我"; exit 1
fi
for f in gv2_heads.pt lora/adapter_config.json; do
  [ -e "$RUN/$f" ] || { log "缺少 $RUN/$f：训练没有正常保存"; exit 1; }
done
log "checkpoint: $(ls -d "$RUN"/checkpoint-* 2>/dev/null | xargs -n1 basename | tr '\n' ' ')"
$PY -m gv2.plot_log "$RUN.log" --out "$OUTDIR/train_curves.png" > "$OUTDIR/health.txt" 2>&1
cat "$OUTDIR/health.txt" | tee -a "$OUTDIR/steps.log"

# 上次评测失败是因为进程看不到 GPU（CUDA driver initialization failed），这里先确认
$PY - <<'EOF' 2>&1 | tee -a "$OUTDIR/steps.log"
import sys, torch
ok = torch.cuda.is_available()
print("cuda available:", ok, "| devices:", torch.cuda.device_count(), "| CUDA_VISIBLE_DEVICES =", __import__("os").environ.get("CUDA_VISIBLE_DEVICES"))
sys.exit(0 if ok else 3)
EOF
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  log "当前进程看不到 GPU。请在有 GPU 的节点/会话里运行，并用 GPU=<卡号> 指定空闲的卡"; exit 3
fi

# ---------------------------------------------------------------- 第 2 步：评测
run_eval() {   # run_eval <行名> <输出 json> <命令...>
  local name=$1 out=$2; shift 2
  if [ -n "${ROWS:-}" ] && [[ " $ROWS " != *" $name "* ]]; then return 0; fi
  if [ -s "$out" ]; then log "$name 已有结果，跳过：$out"; return 0; fi
  log "评测 $name ..."
  "$@" ${MAXARG[@]+"${MAXARG[@]}"} > "$OUTDIR/eval_$name.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ] || [ ! -s "$out" ]; then
    log "$name 失败（退出码 $rc），最后 20 行："; tail -20 "$OUTDIR/eval_$name.log" | tee -a "$OUTDIR/steps.log"; return 1
  fi
  log "$name 完成"
}
EV=(--model-path "$MODEL" --jsonl "$DATA/val.jsonl" --bf16 --device-map '')
V1=(--v1-repo "$REPO" --v1-query-module-dir "$V1_CKPT")

run_eval A        "$OUTDIR/A.json"       $PY -u -m gv2.evaluate "${EV[@]}" "${V1[@]}" --max-image-side 0 --out "$OUTDIR/A.json"
run_eval E        "$OUTDIR/E.json"       $PY -u -m gv2.evaluate "${EV[@]}" --ckpt "$RUN" --out "$OUTDIR/E.json"
run_eval E_prime  "$OUTDIR/E_prime.json" $PY -u -m gv2.evaluate "${EV[@]}" --ckpt "$RUN" --drop-summary --out "$OUTDIR/E_prime.json"
if [ -n "${EXTRA:-}" ]; then
  run_eval A336   "$OUTDIR/A336.json"    $PY -u -m gv2.evaluate "${EV[@]}" "${V1[@]}" --max-image-side 336 --out "$OUTDIR/A336.json"
  [ -d "$RUN/checkpoint-1500" ] && \
  run_eval E1500  "$OUTDIR/E1500.json"   $PY -u -m gv2.evaluate "${EV[@]}" --ckpt "$RUN/checkpoint-1500" --out "$OUTDIR/E1500.json"
fi

# ---------------------------------------------------------------- 对比表
$PY - "$OUTDIR" <<'EOF' | tee "$OUTDIR/compare.md" | tee -a "$OUTDIR/steps.log"
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
rows = [("A", "旧 Ground（原图）"), ("A336", "旧 Ground（336，ReVOS 实际推理设置）"),
        ("E", "v2"), ("E_prime", "v2 推理去掉摘要（E′）"), ("E1500", "v2 checkpoint-1500")]
keys = [("word_acc", "类别词正确率", +1), ("word_acc_multinoun", "多名词子集正确率", +1),
        ("mass_in_gt", "热力图落在 GT 的占比", +1), ("mass_on_other_excl", "落在别类物体（扣掉 GT）的占比", -1),
        ("peak_in_gt", "峰值落在 GT", +1), ("mass_on_other_cat", "落在别类物体（旧口径，含 GT 重叠）", 0)]
res = {k: json.loads((d / f"{k}.json").read_text())["summary"] for k, _ in rows if (d / f"{k}.json").exists()}
if not res:
    print("没有任何评测结果"); sys.exit(0)
print("| 行 | n | " + " | ".join(n for _, n, _ in keys) + " |")
print("|---|---|" + "---|" * len(keys))
for k, name in rows:
    if k in res:
        s = res[k]
        print(f"| {k} {name} | {s['n']} | " + " | ".join("-" if s.get(m) is None else f"{s[m]:.4f}" for m, _, _ in keys) + " |")
if "A" in res and "E" in res:
    a, e = res["A"], res["E"]
    print("\nE 相对 A：")
    for m, n, sign in keys:
        if a.get(m) is None or e.get(m) is None:
            continue
        diff = e[m] - a[m]
        verdict = "仅供参考" if sign == 0 else ("更好" if diff * sign > 0.005 else ("更差" if diff * sign < -0.005 else "持平"))
        print(f"- {n}: {a[m]:.4f} -> {e[m]:.4f} ({diff:+.4f}) {verdict}")
    print(f"- 多名词子集样本数: {e.get('n_multinoun')}")
for k, _ in rows:   # 相对“完全平的热力图”的倍数：>1 说明更集中在该区域
    s = res.get(k)
    if s and s.get("gt_area") and s.get("other_excl_area"):
        print(f"- {k}: GT 倍数 {s['mass_in_gt'] / s['gt_area']:.2f}x，别类(扣 GT)倍数 {s['mass_on_other_excl'] / s['other_excl_area']:.2f}x"
              f"（平图都是 1x；GT 面积 {s['gt_area']:.3f}，别类面积 {s['other_excl_area']:.3f}）")
EOF
log "全部完成。把 $OUTDIR 整个目录（不含大文件）发给我：steps.log、health.txt、train_curves.png、compare.md、*.json"
