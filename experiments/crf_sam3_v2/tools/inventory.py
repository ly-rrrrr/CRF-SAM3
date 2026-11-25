"""只读盘点：数据集、Ground 训练数据、Verify 缓存、权重、磁盘。只用 CPU，不写任何数据文件（只写一份报告）。

    python tools/inventory.py --repo $REPO --gv2 ../ground_v2 --p1a ../p1a_spatial_recall --out runs/inventory.md

或者 `bash run_v2.sh inventory`。每一节独立 try，缺什么就在报告里写「缺失」，不会中途停下。
报告最后把它发给我，用来排完整脚本的数据量和时间。
"""
import argparse
import glob
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

VD = "/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas"

# (显示名, 数据根, [(split, meta 文件相对数据根的候选)], mask_dict 文件名)
DATASETS = [
    ("MeViS", "mevis", ["train", "valid_u", "valid"], "mask_dict.json"),
    ("Ref-DAVIS17", "davis17", ["train", "valid"], "mask_dict.pkl"),
    ("Ref-YTVOS", "Ref-Youtube-VOS", ["train", "valid"], "mask_dict.pkl"),
    ("ReVOS（Verify 用的 prepared）", "revos_prepared", ["train", "valid"], "mask_dict.json"),
]

# 当前论文结果用到的 Verify 缓存和排除表（相对 REPO）
VERIFY_CACHES = {
    "mevis": ("crf_sam3_attn/offline_cache_mevis", None),
    "davis17": ("crf_sam3_attn/offline_cache_davis17_dms_recall24_keep7_20260620",
                "crf_sam3_attn/cache_reports/davis17_train_exclude_max_j_lt_0p5_score_sum_keep7_20260623.json"),
    "refytros": ("crf_sam3_attn/offline_cache_refytros_dms_recall24_keep7_20260620",
                 "crf_sam3_attn/cache_reports/refytros_train_exclude_max_j_lt_0p5_score_sum_keep7_20260623.json"),
    "revos": ("crf_sam3_attn/offline_cache_revos_api_categories_recall24_keep7_20260717",
              "crf_sam3_attn/cache_reports/revos_train_api_categories_recall24_keep7_max_j_lt_0p5_20260718.json"),
}
REVOS_API_CATS = "category_grounding_mvp/inferera_category_api/results/revos_train_qwen37plus_categories_normalized_20260717.json"
GROUND_V1_DATA = ["category_grounding_mvp/data/refytvos_query_grounding",
                  "category_grounding_mvp/data/revos_query_grounding_api_20260717",
                  "category_grounding_mvp/data/refytvos_revos_query_grounding_joint_api_20260717"]
PAPER_VERIFY = {
    "MeViS 71.45": "crf_sam3_attn_outputs_joint_train_poswin/final",
    "DAVIS 76.39 / YTVOS 65.42(ckpt 39000)": "crf_sam3_attn_outputs_joint_train_poswin_dms_recall24_keep7_filter_failures_20260623",
    "ReVOS 59.74": "crf_sam3_attn_outputs_revos_warmstart_3domain_api_categories_recall24_keep7_20260805/final",
}

LINES = []


def out(s=""):
    LINES.append(s)


def section(title):
    def deco(fn):
        def run(*a, **k):
            out(f"\n## {title}\n")
            t = time.time()
            try:
                fn(*a, **k)
            except Exception as e:  # 一节失败不影响其他节
                out(f"**这一节出错**：`{type(e).__name__}: {e}`")
            print(f"[inventory] {title}  {time.time() - t:.1f}s", flush=True)
        return run
    return deco


def size_str(n):
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024 or u == "TB":
            return f"{n:.1f} {u}" if u != "B" else f"{n} B"
        n /= 1024


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def find_meta(root, split):
    """和 crf_sam3_attn/dataset.py::_load_meta_expressions 同样的查找顺序，再补 ReVOS 的命名。"""
    for c in [os.path.join(root, split, "meta_expressions.json"),
              os.path.join(root, "meta_expressions", split, "meta_expressions.json"),
              os.path.join(root, split, f"meta_expressions_{split}.json"),
              os.path.join(root, f"meta_expressions_{split}_.json"),
              os.path.join(root, f"meta_expressions_{split}.json")]:
        if os.path.isfile(c):
            return c
    return None


def obj_count(e):
    for k in ("obj_id", "anno_id"):
        v = e.get(k)
        if isinstance(v, list):
            return len(v)
        if v is not None:
            return 1
    return 0


def meta_stats(path):
    vids = load_json(path)["videos"]
    n_exp, frames, nobj = 0, [], Counter()
    for v in vids.values():
        ex = v.get("expressions", {})
        n_exp += len(ex)
        if "frames" in v:
            frames.append(len(v["frames"]))
        for e in ex.values():
            c = obj_count(e)
            nobj["0" if c == 0 else "1" if c == 1 else "≥2"] += 1
    return {"videos": len(vids), "expressions": n_exp,
            "frames_mean": round(statistics.mean(frames), 1) if frames else None,
            "frames_max": max(frames) if frames else None, "targets": dict(nobj), "_vids": vids}


@section("1. 数据集（来自 meta_expressions，掩码文件只看大小不读取）")
def datasets(a, keep):
    out("| 数据集 | split | meta 文件 | 视频 | 表达式 | 平均帧数（最大） | 目标数 0 / 1 / ≥2 | mask_dict |")
    out("|---|---|---|---|---|---|---|---|")
    for name, sub, splits, md in DATASETS:
        root = os.path.join(a.video_datas, sub)
        for sp in splits:
            m = find_meta(root, sp)
            if not m:
                if os.path.isdir(os.path.join(root, sp)):
                    out(f"| {name} | {sp} | 缺失 | | | | | |")
                continue
            s = meta_stats(m)
            keep[(sub, sp)] = s
            mdp = os.path.join(root, sp, md)
            mds = size_str(os.path.getsize(mdp)) if os.path.isfile(mdp) else "无"
            t = s["targets"]
            out(f"| {name} | {sp} | `{os.path.relpath(m, root)}` | {s['videos']} | {s['expressions']} | "
                f"{s['frames_mean']}（{s['frames_max']}） | {t.get('0', 0)} / {t.get('1', 0)} / {t.get('≥2', 0)} | {mds} |")
    # 原版 ReVOS（Ground 和 API 类别用的是这份）
    for sp in ["train", "valid"]:
        m = find_meta(a.revos_root, sp)
        if m:
            s = meta_stats(m)
            keep[("revos", sp)] = s
            t = s["targets"]
            out(f"| ReVOS（原版） | {sp} | `{os.path.basename(m)}` | {s['videos']} | {s['expressions']} | "
                f"{s['frames_mean']}（{s['frames_max']}） | {t.get('0', 0)} / {t.get('1', 0)} / {t.get('≥2', 0)} | |")
    out("\n目标数是每条表达式指向的物体个数（obj_id / anno_id 的长度），用来看多目标表达式占比。")


@section("2. Ground 训练数据覆盖了多少训练集")
def ground_data(a, keep):
    out("| 数据 | 训练记录 | 验证记录 | 训练视频 | 验证视频 | 备注 |")
    out("|---|---|---|---|---|---|")
    for rel in GROUND_V1_DATA:
        p = os.path.join(a.repo, rel, "summary.json")
        if not os.path.isfile(p):
            out(f"| v1 `{rel.split('/')[-1]}` | 缺失 | | | | |")
            continue
        s = load_json(p)
        lines = {}
        for sp in ("train", "val"):
            j = os.path.join(a.repo, rel, f"{sp}.jsonl")
            if os.path.isfile(j):
                with open(j, encoding="utf-8") as f:
                    lines[sp] = sum(1 for _ in f)
        note = f"jsonl 实际行数 {lines}" if lines else "没有 jsonl"
        out(f"| v1 `{rel.split('/')[-1]}` | {s.get('train_records')} | {s.get('val_records')} | "
            f"{s.get('train_videos')} | {s.get('val_videos')} | {note} |")
    for p in sorted(glob.glob(os.path.join(a.gv2, "data", "*", "summary*.json"))):
        s = load_json(p)
        rec = s.get("records") or s.get("num_records") or s.get("n")
        keys = ", ".join(f"{k}={s[k]}" for k in list(s)[:8] if not isinstance(s[k], (dict, list)))
        out(f"| v2 `{Path(p).parent.name}/{Path(p).name}` | {rec} | | | | {keys} |")
    out("")
    for (sub, sp), label in [(("revos", "train"), "ReVOS 原版 train"), (("Ref-Youtube-VOS", "train"), "Ref-YTVOS train")]:
        s = keep.get((sub, sp))
        if s:
            out(f"- {label}：{s['videos']} 个视频，{s['expressions']} 条表达式。和上表比，就是 Ground 训练实际用了多少。")


@section("3. ReVOS API 类别缓存（Verify 训练时 ReVOS 的词来源）")
def api_cats(a, keep):
    p = os.path.join(a.repo, REVOS_API_CATS)
    if not os.path.isfile(p):
        out(f"缺失：`{REVOS_API_CATS}`")
        return
    payload = load_json(p)
    recs = payload.get("records", payload) if isinstance(payload, dict) else payload
    keys = {(str(r.get("video_id")), int(r.get("exp_id"))) for r in recs
            if isinstance(r, dict) and r.get("video_id") is not None and r.get("exp_id") is not None and r.get("category")}
    out(f"- 可用记录 {len(keys)} 条，涉及 {len({v for v, _ in keys})} 个视频。")
    for sub in ("revos", "revos_prepared"):
        s = keep.get((sub, "train"))
        if not s:
            continue
        allk = {(v, int(e)) for v, d in s["_vids"].items() for e in d.get("expressions", {})}
        out(f"- 覆盖 `{sub}` train 表达式 {len(allk & keys)} / {len(allk)}"
            f"（{100 * len(allk & keys) / max(1, len(allk)):.1f}%）。")


def walk_cache(split_root, limit):
    n, vids, cands, hit5, maxj, frames, keys = 0, set(), [], 0, [], Counter(), Counter()
    for dirpath, dirnames, filenames in os.walk(split_root):
        if "meta.json" not in filenames:
            continue
        dirnames[:] = []
        n += 1
        rel = os.path.relpath(dirpath, split_root)
        vids.add(os.path.dirname(rel))
        if n > limit:
            continue
        try:
            m = load_json(os.path.join(dirpath, "meta.json"))
        except Exception:
            keys["<坏文件>"] += 1
            continue
        keys.update(m.keys())
        im = {k: float(v) for k, v in (m.get("iou_map") or {}).items()}
        cands.append(len(im))
        mj = max(im.values()) if im else 0.0
        maxj.append(mj)
        hit5 += mj >= 0.5
        frames[len(m.get("frame_indices") or [])] += 1
    read = min(n, limit)
    return {"n": n, "videos": len(vids), "read": read,
            "cand_mean": round(statistics.mean(cands), 2) if cands else None,
            "recall05": round(100 * hit5 / read, 1) if read else None,
            "maxj_mean": round(statistics.mean(maxj), 3) if maxj else None,
            "frames": dict(frames.most_common(3)), "keys": [k for k, _ in keys.most_common(12)]}


@section("4. Verify 训练缓存（现在论文用的这几份）")
def verify_caches(a, keep):
    out("| 数据集 | 缓存目录 | split | 表达式 | 视频 | 平均候选数 | max J≥0.5 占比 | 平均 max J | 排除表条数 |")
    out("|---|---|---|---|---|---|---|---|---|")
    seen = set()
    for ds, (rel, exc) in VERIFY_CACHES.items():
        root = os.path.join(a.repo, rel)
        seen.add(os.path.normpath(root))
        nexc = "无"
        if exc:
            pe = os.path.join(a.repo, exc)
            nexc = load_json(pe).get("summary", {}).get("num_samples", "?") if os.path.isfile(pe) else "缺失"
        if not os.path.isdir(root):
            out(f"| {ds} | `{Path(rel).name}` | | 缺失 | | | | | {nexc} |")
            continue
        for sp in sorted(os.listdir(root)):
            if not os.path.isdir(os.path.join(root, sp)):
                continue
            s = walk_cache(os.path.join(root, sp), a.max_meta)
            keep.setdefault("cache_keys", s["keys"])
            part = "" if s["read"] == s["n"] else f"（抽 {s['read']}）"
            out(f"| {ds} | `{Path(rel).name}` | {sp} | {s['n']} | {s['videos']} | {s['cand_mean']} | "
                f"{s['recall05']}%{part} | {s['maxj_mean']} | {nexc} |")
    others = [p for p in sorted(glob.glob(os.path.join(a.repo, "crf_sam3_attn", "offline_cache*")))
              if os.path.normpath(p) not in seen and os.path.isdir(p)]
    if others:
        out("\n其他缓存目录（论文没用，只列名字和 split）：")
        for p in others:
            sps = [d for d in sorted(os.listdir(p)) if os.path.isdir(os.path.join(p, d))]
            out(f"- `{Path(p).name}`：{', '.join(sps) or '空'}")
    if keep.get("cache_keys"):
        out(f"\nmeta.json 里的字段：`{', '.join(keep['cache_keys'])}`")
    out("\n表达式数和第 1 节的 train 表达式数对比，就是 Verify 实际用了多少训练集。")


@section("5. 权重")
def checkpoints(a, keep):
    out("**论文结果用的 Verify 权重**\n")
    for k, rel in PAPER_VERIFY.items():
        p = os.path.join(a.repo, rel)
        if os.path.isdir(p) and not os.path.basename(rel).startswith(("final", "checkpoint")):
            sub = sorted(os.listdir(p))
            out(f"- {k}：`{rel}` 存在，里面有 {', '.join(sub[:12])}{' …' if len(sub) > 12 else ''}")
        else:
            out(f"- {k}：`{rel}` {'存在' if os.path.isdir(p) else '**缺失**'}")
    out("\n**所有 Verify 输出目录**（`crf_sam3_attn_outputs*`，只列 final 和最后一个 checkpoint）\n")
    for p in sorted(glob.glob(os.path.join(a.repo, "crf_sam3_attn_outputs*"))):
        if not os.path.isdir(p) or p.endswith("_merged"):
            continue
        sub = os.listdir(p)
        ck = sorted((d for d in sub if d.startswith("checkpoint")), key=lambda d: int("".join(c for c in d if c.isdigit()) or 0))
        has_final = "final" in sub
        out(f"- `{Path(p).name}`：{'final，' if has_final else ''}{len(ck)} 个 checkpoint{('，最后 ' + ck[-1]) if ck else ''}")
    out("\n**Ground v1**（`category_grounding_mvp/outputs` 下带 checkpoint 的目录）\n")
    for p in sorted(glob.glob(os.path.join(a.repo, "category_grounding_mvp", "outputs", "*"))):
        if os.path.isdir(p):
            ck = [d for d in os.listdir(p) if d.startswith("checkpoint")]
            if ck:
                out(f"- `{Path(p).name}`：{', '.join(sorted(ck, key=lambda d: int(d.split('-')[-1]) if d.split('-')[-1].isdigit() else 0)[-3:])}")
    out("\n**Ground v2**（`ground_v2/runs`）\n")
    for p in sorted(glob.glob(os.path.join(a.gv2, "runs", "*"))):
        if not os.path.isdir(p):
            continue
        cfg = os.path.isfile(os.path.join(p, "gv2_config.json"))
        ev = os.path.join(p, "eval_val.json")
        evs = ""
        if os.path.isfile(ev):
            e = load_json(ev)
            evs = "，eval_val: " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                             for k, v in list(e.items())[:6] if not isinstance(v, (dict, list)))
        out(f"- `{Path(p).name}`：{'训练完成（有 gv2_config.json）' if cfg else '没有 gv2_config.json'}{evs}")


@section("6. 已有的新主线 / 实验输出")
def runs(a, keep):
    for base in [os.path.join(a.here, "runs"), os.path.join(a.p1a, "runs")]:
        if not os.path.isdir(base):
            continue
        for p in sorted(os.listdir(base)):
            d = os.path.join(base, p)
            if not os.path.isdir(d):
                continue
            marks = [f for f in ("summary.md", "scores.jsonl") if os.path.isfile(os.path.join(d, f))]
            evs = glob.glob(os.path.join(d, "*", "evaluation_results*.json")) + glob.glob(os.path.join(d, "*", "revos_valid_official.json"))
            out(f"- `{os.path.relpath(d, os.path.dirname(a.here))}`：{', '.join(marks) or '无汇总'}；评测文件 {len(evs)} 个")


@section("7. 磁盘和显卡")
def disk_gpu(a, keep):
    for p in [a.repo, a.here, a.video_datas]:
        if os.path.isdir(p):
            u = shutil.disk_usage(p)
            out(f"- `{p}`：剩余 {size_str(u.free)} / 共 {size_str(u.total)}")
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        if r.returncode == 0:
            out("\n```\n" + r.stdout.strip() + "\n```")
    except Exception:
        out("- 没有 nvidia-smi")


def main(argv=None):
    here = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.environ.get("REPO", "/mnt/csip-500/share/kongjiawei/Sa2VA-main"))
    ap.add_argument("--video-datas", default=VD)
    ap.add_argument("--revos-root", default=os.environ.get("REVOS_ROOT", os.path.join(VD, "revos")))
    ap.add_argument("--gv2", default=str(here.parent / "ground_v2"))
    ap.add_argument("--p1a", default=str(here.parent / "p1a_spatial_recall"))
    ap.add_argument("--max-meta", type=int, default=200000, help="每个缓存 split 最多读多少个 meta.json（计数不受限）")
    ap.add_argument("--out", default=str(here / "runs" / "inventory.md"))
    a = ap.parse_args(argv)
    a.here = str(here)
    t0 = time.time()
    out("# CRF-SAM3 盘点")
    out(f"\n生成时间 {time.strftime('%Y-%m-%d %H:%M')}，REPO=`{a.repo}`")
    keep = {}
    for fn in (datasets, ground_data, api_cats, verify_caches, checkpoints, runs, disk_gpu):
        fn(a, keep)
    out(f"\n---\n用时 {time.time() - t0:.0f} 秒")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print(f"[inventory] 写好了：{a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
