"""只读审计：服务器上已经构造好的训练集（数据集本身、Ground jsonl、Verify 缓存）是否对得上。只用 CPU。

    python tools/audit_trainsets.py --repo $REPO --revos-root $REVOS_ROOT --gv2 ../ground_v2 --out runs/audit
    或者 `bash run_v2.sh audit`

不改任何数据，只写 runs/audit/ 下的报告：trainsets.md（给人看）、trainsets.json（汇总数字）、
affected_*.json（受影响的 (video_id, exp_id) 清单，之后重建数据时直接用）。

逐项检查（每一节独立，缺文件就写「缺失」继续）：
  A 帧对齐：meta_expressions 的 frames 和 JPEGImages 里的文件是否一一对应（个数、名字、顺序）。
    所有代码都按「排序后的位置」取帧和 mask，两者不一致就会错帧。
  B mask_dict 索引：Ref-YTVOS 的 mask_dict 按「全局表达式序号」存。我们的代码按 json 原顺序数序号，
    Sa2VA 按「视频原顺序 + 表达式 id 字符串排序」数（一个视频 ≥11 条表达式时两者不同，"10" 排在 "2" 前）。
    这里用 Annotations 的调色板 PNG（像素值 = 物体 id）当裁判，看哪种顺序取出来的 mask 才是 obj_id 那个物体。
    DAVIS 等按 anno_id/obj_id 存的也用 PNG 对一遍。同时统计目标不在画面的帧占比（影响 J 的定义）。
  C 类别标签：各数据集有没有 Ground 训练需要的物体类别。
  D Ground jsonl：条数、重复、train/val 视频重叠、帧路径存在、帧名和 meta 对得上、受 B 影响的条数。
  E Verify 缓存：meta 和表达式的覆盖、字段一致性、mask 文件、候选数和 max J 分布、
    「没有正样本却没被排除」会被 force_pos 硬选一个当正样本的条数、受 B 影响的条数；
    抽样重算 J（缓存口径 / 换正确顺序 / 按评测口径把目标不在的帧也算上），看正负标签会翻多少。
  F 跨数据集泄漏（--leak）：训练视频和各评测集视频首帧感知哈希比对（ReVOS 里有 MOSE 等来源的视频）。
"""
import argparse
import glob
import json
import os
import pickle
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from inventory import VD, VERIFY_CACHES, REVOS_API_CATS  # noqa: E402

LINES, SUMMARY, GATES = [], {}, []
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")

# 数据集：名字 → (数据根相对 video_datas, mask_dict 文件, 有没有调色板 PNG, Verify 缓存名)
TRAIN_SETS = {
    "mevis": ("mevis", "mask_dict.json", False, "mevis"),
    "davis17": ("davis17", "mask_dict.pkl", True, "davis17"),
    "refytvos": ("Ref-Youtube-VOS", "mask_dict.pkl", True, "refytros"),
    "revos": ("revos_prepared", "mask_dict.json", False, "revos"),
}
# 现有联合配置里的正样本阈值（force_pos 统计用）；ReVOS 用的 warmstart 配置是 0.5
POS_IOU = {"mevis": 0.2, "davis17": 0.5, "refytvos": 0.5, "revos": 0.5}
NEG_IOU = {"mevis": 0.05, "davis17": 0.1, "refytvos": 0.1, "revos": 0.1}


def out(s=""):
    LINES.append(s)


def gate(level, text):
    GATES.append((level, text))


def section(title):
    def deco(fn):
        def run(*a, **k):
            out(f"\n## {title}\n")
            t = time.time()
            try:
                return fn(*a, **k)
            except Exception as e:  # 一节失败不影响其他节
                import traceback
                out(f"**这一节出错**：`{type(e).__name__}: {e}`")
                out("```\n" + traceback.format_exc()[-1500:] + "\n```")
                gate("FAIL", f"{title}：脚本出错 {type(e).__name__}")
            finally:
                print(f"[audit] {title}  {time.time() - t:.1f}s", flush=True)
        return run
    return deco


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def pct(a, b):
    return f"{100.0 * a / b:.1f}%" if b else "-"


def find_meta(split_root):
    """和 crf_sam3_attn/dataset.py::_load_meta_expressions 同样的查找顺序。"""
    split_root = Path(split_root)
    for p in (split_root / "meta_expressions.json",
              split_root.parent / "meta_expressions" / split_root.name / "meta_expressions.json",
              split_root.parent / "meta_expressions.json"):
        if p.is_file():
            return p
    return None


def list_stems(d):
    if not os.path.isdir(d):
        return None
    return sorted(os.path.splitext(n)[0] for n in os.listdir(d) if n.lower().endswith(IMG_EXT))


def load_mask_dict(p):
    if str(p).endswith(".pkl"):
        with open(p, "rb") as f:
            return pickle.load(f)
    return load_json(p)


# ---------------------------------------------------------------- 表达式全局序号的几种数法
def expr_orders(meta):
    vids = meta.get("videos", {})
    orders = {
        "insertion": [(v, e) for v, d in vids.items() for e in d.get("expressions", {})],  # 我们的代码
        "sa2va": [(v, e) for v, d in vids.items() for e in sorted(d.get("expressions", {}))],  # Sa2VA 加载器
        "numeric": [(v, e) for v, d in vids.items()
                    for e in sorted(d.get("expressions", {}), key=lambda x: int(x) if str(x).isdigit() else 10 ** 9)],
    }
    return {k: {(v, int(e)): i for i, (v, e) in enumerate(lst)} for k, lst in orders.items()}


def decode(rle):
    if rle is None or rle == "" or rle == []:
        return None
    from pycocotools import mask as cm
    if isinstance(rle, (list, tuple)):
        ms = [decode(r) for r in rle]
        ms = [m for m in ms if m is not None]
        if not ms:
            return None
        o = ms[0].astype(bool)
        for m in ms[1:]:
            o |= m.astype(bool)
        return o.astype(np.uint8)
    if isinstance(rle, dict) and isinstance(rle.get("counts"), str):
        rle = dict(rle, counts=rle["counts"].encode("utf-8"))
    m = cm.decode(rle)
    if m.ndim == 3:
        m = m.any(axis=2)
    return m.astype(np.uint8)


def iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    u = (a | b).sum()
    return 1.0 if u == 0 else float((a & b).sum()) / float(u)


def read_png(p):
    from PIL import Image
    if not os.path.isfile(p):
        return None
    with Image.open(p) as im:
        return np.array(im)


def ids_of(x):
    if x is None or x == "":
        return []
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    return [int(v) for v in str(x).split(",") if str(v).strip() != ""]


class DS:
    """一个训练集：meta、mask_dict、取 GT 的几种方式。"""

    def __init__(self, name, vd, revos_root):
        self.name = name
        sub, md, self.has_png, self.cache_name = TRAIN_SETS[name]
        self.root = Path(vd) / sub / "train"
        self.meta_path = find_meta(self.root)
        self.meta = load_json(self.meta_path) if self.meta_path else {"videos": {}}
        self.videos = self.meta.get("videos", {})
        self.md_path = self.root / md
        self.md = None
        self.mode = None
        self.orders = None
        self.differ = set()
        self.correct_order = "insertion"

    def load_md(self):
        if self.md is None and self.md_path.is_file():
            self.md = load_mask_dict(self.md_path)
            n_expr = sum(len(d.get("expressions", {})) for d in self.videos.values())
            try:
                keys = [int(k) for k in self.md]
                self.mode = ("expression_index" if n_expr and len(keys) == n_expr
                             and min(keys) == 0 and max(keys) == n_expr - 1 else "obj_or_anno")
            except (TypeError, ValueError):
                self.mode = "obj_or_anno"
        return self.md

    def seq(self, vid, eid, order=None):
        """一条表达式逐帧的 RLE 列表（多目标已按帧合并成 list）。"""
        md = self.load_md()
        if md is None:
            return None
        if self.mode == "expression_index":
            i = self.orders[order or self.correct_order].get((vid, int(eid)))
            s = md.get(str(i), md.get(i)) if i is not None else None
            return list(s) if s is not None else None
        e = self.videos.get(vid, {}).get("expressions", {}).get(str(eid), {})
        ids = ids_of(e.get("anno_id")) or ids_of(e.get("obj_id"))
        seqs = [md.get(str(a), md.get(a)) for a in ids]
        seqs = [s for s in seqs if s is not None]
        if not seqs:
            return None
        n = max(len(s) for s in seqs)
        return [[s[f] for s in seqs if f < len(s) and s[f] is not None] or None for f in range(n)]


# ---------------------------------------------------------------- A 帧对齐
@section("A 帧对齐（meta frames ↔ JPEGImages）")
def check_frames(dss, revos_root):
    out("| 数据集 | 视频 | 缺目录 | 帧数不同 | 名字/顺序不同 | meta frames 未排序 | 帧名步长 | >300 帧 |")
    out("|---|---|---|---|---|---|---|---|")
    res = {}
    for ds in dss.values():
        c, steps, bad = Counter(), Counter(), []
        for vid, d in ds.videos.items():
            mf = [str(f) for f in d.get("frames", [])]
            st = list_stems(ds.root / "JPEGImages" / vid)
            c["videos"] += 1
            if mf != sorted(mf):
                c["unsorted"] += 1
            if len(mf) > 300:
                c["long"] += 1
            if len(mf) >= 2 and all(x.isdigit() for x in mf[:2]):
                steps[int(sorted(mf)[1]) - int(sorted(mf)[0])] += 1
            if st is None:
                c["missing"] += 1
                bad.append([vid, "missing_dir"])
                continue
            if len(st) != len(mf):
                c["count"] += 1
                bad.append([vid, f"jpg {len(st)} vs meta {len(mf)}"])
            elif st != sorted(mf):
                c["names"] += 1
                bad.append([vid, "names differ"])
        res[ds.name] = dict(c, steps=dict(steps.most_common(3)), examples=bad[:20])
        out(f"| {ds.name} | {c['videos']} | {c['missing']} | {c['count']} | {c['names']} | {c['unsorted']} | "
            f"{dict(steps.most_common(3))} | {c['long']} |")
        n_bad = c["missing"] + c["count"] + c["names"] + c["unsorted"]
        gate("PASS" if n_bad == 0 else "FAIL",
             f"A {ds.name} 帧对齐：{n_bad} / {c['videos']} 个视频不一致" + (f"（例：{bad[:3]}）" if bad else ""))
    # Ground 用的 ReVOS 原始布局：帧直接在 revos/<video_id>/ 下
    rr = Path(revos_root)
    mp = rr / "meta_expressions_train_.json"
    if mp.is_file():
        vids = load_json(mp).get("videos", {})
        c = Counter()
        for vid, d in vids.items():
            mf = [str(f) for f in d.get("frames", [])]
            st = list_stems(rr / vid)
            c["videos"] += 1
            if st is None:
                c["missing"] += 1
            elif st != sorted(mf):
                c["mismatch"] += 1
        res["revos_raw"] = dict(c)
        out(f"\nReVOS 原始布局（Ground 用）：{c['videos']} 个视频，缺目录 {c['missing']}，帧不一致 {c['mismatch']}。")
        gate("PASS" if c["missing"] + c["mismatch"] == 0 else "WARN",
             f"A revos 原始布局：{c['missing'] + c['mismatch']} 个视频帧不一致")
    out("\n帧名步长 5 表示每 5 帧一张（Ref-YTVOS），这时决不能用 int(帧名) 当位置。")
    SUMMARY["A_frames"] = res


# ---------------------------------------------------------------- B mask_dict 索引
@section("B mask_dict 索引与 GT")
def check_index(dss, png_frames, png_max, seed, outdir):
    rng = random.Random(seed)
    res = {}
    for ds in dss.values():
        if not ds.md_path.is_file():
            out(f"- {ds.name}：mask_dict 缺失（{ds.md_path}）")
            continue
        md = ds.load_md()
        ds.orders = expr_orders(ds.meta)
        r = {"mode": ds.mode, "entries": len(md),
             "expressions": len(ds.orders["insertion"])}
        exprs = list(ds.orders["insertion"])
        if ds.mode == "expression_index":
            ins, sa = ds.orders["insertion"], ds.orders["sa2va"]
            ds.differ = {k for k in exprs if ins[k] != sa[k]}
            r["insertion_vs_sa2va_differ"] = len(ds.differ)
            r["videos_ge_11_expr"] = sum(len(d.get("expressions", {})) >= 11 for d in ds.videos.values())
        else:
            no_anno = sum(1 for v, e in exprs
                          if not ids_of(ds.videos[v]["expressions"][str(e)].get("anno_id")))
            r["expr_without_anno_id"] = no_anno
            if no_anno:
                # 只有 obj_id（每个视频从 1 开始）却按全局 key 查 mask_dict：不同视频会撞 key
                gate("WARN", f"B {ds.name}：{no_anno} 条表达式没有 anno_id，代码会退回用 obj_id 查 mask_dict，"
                     "obj_id 在每个视频里从 1 开始，必须靠下面的 PNG 比对确认没撞 key")
        # 长度和 meta 帧数
        bad_len = 0
        absent, present, empty_expr = 0, 0, 0
        for v, e in exprs:
            s = ds.seq(v, e, "insertion" if ds.mode == "expression_index" else None)
            n = len(ds.videos[v].get("frames", []))
            if s is None:
                empty_expr += 1
                continue
            if len(s) != n:
                bad_len += 1
            a = sum(1 for x in s if not x)
            absent += a
            present += len(s) - a
        r.update(len_mismatch=bad_len, no_mask_expr=empty_expr,
                 absent_frame_ratio=round(absent / max(1, absent + present), 4))
        # PNG 裁判
        if ds.has_png and (ds.root / "Annotations").is_dir():
            pool = sorted(ds.differ)
            rng.shuffle(pool)
            ctrl = [k for k in exprs if k not in ds.differ]
            rng.shuffle(ctrl)
            test = pool[:png_max] + ctrl[: max(50, png_max // 4)]
            orders = ["insertion", "sa2va", "numeric"] if ds.mode == "expression_index" else [None]
            scores = {str(o): {"differ": [], "control": []} for o in orders}
            for v, e in test:
                d = ds.videos[v]
                tgt = ids_of(d["expressions"][str(e)].get("obj_id"))
                frames = [str(f) for f in d.get("frames", [])]
                fsel = []
                for fi in range(len(frames)):
                    png = read_png(ds.root / "Annotations" / v / f"{frames[fi]}.png")
                    if png is not None and np.isin(png, tgt).any():
                        fsel.append((fi, png))
                    if len(fsel) >= png_frames:
                        break
                if not fsel:
                    continue
                grp = "differ" if (v, e) in ds.differ else "control"
                for o in orders:
                    s = ds.seq(v, e, o)
                    vals = []
                    for fi, png in fsel:
                        m = decode(s[fi]) if s is not None and fi < len(s) else None
                        g = np.isin(png, tgt)
                        vals.append(iou(m, g) if m is not None and m.shape == g.shape else 0.0)
                    scores[str(o)][grp].append(float(np.mean(vals)))
            out(f"\n**{ds.name}** PNG 比对（mask_dict 取出的 mask vs Annotations 里 obj_id 的像素）：")
            out("| 顺序 | 受影响表达式 mean IoU（条数） | IoU≥0.9 | 对照组 mean IoU（条数） |")
            out("|---|---|---|---|")
            best, best_v = None, -1
            for o, sc in scores.items():
                dm = float(np.mean(sc["differ"])) if sc["differ"] else float("nan")
                cm_ = float(np.mean(sc["control"])) if sc["control"] else float("nan")
                hi = pct(sum(x >= 0.9 for x in sc["differ"]), len(sc["differ"]))
                out(f"| {o if o != 'None' else 'anno_id / obj_id'} | {dm:.3f}（{len(sc['differ'])}） | {hi} | {cm_:.3f}（{len(sc['control'])}） |")
                key = dm if sc["differ"] else cm_
                if key == key and key > best_v:
                    best, best_v = o, key
            r["png_check"] = {o: {g: (round(float(np.mean(x)), 4) if x else None, len(x)) for g, x in sc.items()}
                              for o, sc in scores.items()}
            if ds.mode == "expression_index":
                ds.correct_order = best or "insertion"
                r["correct_order"] = ds.correct_order
                if not ds.differ:
                    gate("PASS", f"B {ds.name}：没有 ≥11 条表达式的视频，几种顺序一致")
                elif ds.correct_order == "insertion":
                    gate("PASS", f"B {ds.name}：mask_dict 和我们代码的顺序一致（{len(ds.differ)} 条在两种顺序下不同，PNG 判定 insertion 对）")
                else:
                    gate("FAIL", f"B {ds.name}：mask_dict 是 {ds.correct_order} 顺序，我们的代码用 insertion 顺序，"
                         f"{len(ds.differ)} 条表达式拿到的是同视频另一条表达式的 GT")
            else:
                ctl = scores["None"]["control"]
                mean_ctl = float(np.mean(ctl)) if ctl else float("nan")
                gate("PASS" if mean_ctl >= 0.9 else "FAIL",
                     f"B {ds.name}：mask_dict 和 PNG 的 mean IoU {mean_ctl:.3f}（{len(ctl)} 条）")
        elif ds.mode == "expression_index" and ds.differ:
            gate("WARN", f"B {ds.name}：{len(ds.differ)} 条表达式在两种顺序下序号不同，但没有 PNG 可判定")
        if ds.differ:
            (Path(outdir) / f"affected_index_order_{ds.name}.json").write_text(json.dumps(
                {"dataset": ds.name, "correct_order": ds.correct_order,
                 "samples": [{"video_id": v, "exp_id": e} for v, e in sorted(ds.differ)]}, indent=1))
        res[ds.name] = r
        ds.md = None  # 一个一个数据集来，MeViS 的 mask_dict 很大，不同时放在内存里
    out("\n| 数据集 | 存法 | 条目 | 表达式 | 序号受影响 | 帧数≠meta | 无 mask 表达式 | GT 为空(None)的帧占比 |")
    out("|---|---|---|---|---|---|---|---|")
    for n, r in res.items():
        out(f"| {n} | {r['mode']} | {r['entries']} | {r['expressions']} | {r.get('insertion_vs_sa2va_differ', '-')} | "
            f"{r['len_mismatch']} | {r['no_mask_expr']} | {r['absent_frame_ratio']:.1%} |")
        if r["len_mismatch"]:
            gate("FAIL", f"B {n}：{r['len_mismatch']} 条表达式 mask 序列长度 ≠ meta 帧数")
    out("\nGT 为 None 的帧（目标不在画面，且 mask_dict 里存的是空值而不是空 mask）：现有缓存算 J 时跳过这些帧，"
        "不进分母；评测时这些帧照样算（预测也为空记 1，否则记 0）。占比越高，两种口径的正负标签差得越多，见 E 节重算。")
    SUMMARY["B_index"] = res


# ---------------------------------------------------------------- C 类别标签
@section("C 类别标签（Ground 训练需要）")
def check_categories(dss, vd, repo):
    for ds in dss.values():
        has_cat = 0
        n_obj = 0
        mj = ds.root / "meta.json"
        if mj.is_file():
            for v in load_json(mj).get("videos", {}).values():
                for o in v.get("objects", {}).values():
                    n_obj += 1
                    has_cat += bool(str(o.get("category", "")).strip())
        in_expr = sum(1 for d in ds.videos.values() for e in d.get("expressions", {}).values() if e.get("category"))
        out(f"- {ds.name}：train/meta.json {'有' if mj.is_file() else '无'}（物体 {n_obj}，带类别 {has_cat}）；"
            f"表达式里带 category 字段 {in_expr} 条")
        SUMMARY.setdefault("C_categories", {})[ds.name] = {"meta_json_objects": n_obj, "with_category": has_cat,
                                                         "expr_with_category": in_expr}
    p = Path(repo) / REVOS_API_CATS
    if p.is_file():
        recs = load_json(p).get("records", [])
        out(f"- ReVOS API 类别缓存：{len(recs)} 条")
        SUMMARY["C_categories"]["revos_api"] = len(recs)
    out("\nDAVIS 若既没有 meta.json 类别也没有表达式类别，就和 MeViS 一样不进 Ground 训练，只进 Verify。")


# ---------------------------------------------------------------- D Ground jsonl
def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


@section("D Ground 训练数据（jsonl）")
def check_ground(yt, revos_root, files, outdir):
    rv_meta = Path(revos_root) / "meta_expressions_train_.json"
    rv = load_json(rv_meta).get("videos", {}) if rv_meta.is_file() else {}
    out("| 文件 | 条数 | 唯一表达式 | 重复 | 视频 | 帧路径缺失 | 帧名≠meta | 空类别 | 序号受影响 | neg/ign/absent 覆盖 |")
    out("|---|---|---|---|---|---|---|---|---|---|")
    vids_by_file, res = {}, {}
    for f in files:
        if not Path(f).is_file():
            out(f"| `{f}` | 缺失 |||||||||")
            continue
        recs = read_jsonl(f)
        keys = Counter((str(r.get("video_id")), int(r.get("exp_id", -1))) for r in recs)
        miss = name_bad = empty_cat = affected = 0
        cov = Counter()
        for r in recs:
            vid, eid = str(r.get("video_id")), int(r.get("exp_id", -1))
            for p in r.get("frame_paths", []):
                if not os.path.exists(p):
                    miss += 1
            if not str(r.get("category", "")).strip():
                empty_cat += 1
            frames = (yt.videos.get(vid) or rv.get(vid) or {}).get("frames", [])
            for k, fi in enumerate(r.get("mask_frame_indices") or []):
                fp = r.get("frame_paths", [])
                if k < len(fp) and (fi >= len(frames) or os.path.splitext(os.path.basename(fp[k]))[0] != str(frames[fi])):
                    name_bad += 1
            if "global_index" in r and yt.orders and (vid, eid) in yt.orders[yt.correct_order]:
                affected += int(r["global_index"]) != yt.orders[yt.correct_order][(vid, eid)]
            for fld in ("neg_rles", "ign_rles"):
                if any(x is not None for x in r.get(fld) or []):
                    cov[fld] += 1
            cov["absent"] += bool(r.get("absent_frames"))
        vids_by_file[f] = {str(r.get("video_id")) for r in recs}
        dup = sum(c - 1 for c in keys.values() if c > 1)
        covs = "/".join(pct(cov[x], len(recs)) for x in ("neg_rles", "ign_rles", "absent")) if cov else "-"
        out(f"| `{Path(f).parent.name}/{Path(f).name}` | {len(recs)} | {len(keys)} | {dup} | {len(vids_by_file[f])} | "
            f"{miss} | {name_bad} | {empty_cat} | {affected} | {covs} |")
        res[f] = dict(records=len(recs), unique=len(keys), dup=dup, missing_paths=miss, name_mismatch=name_bad,
                      empty_category=empty_cat, index_affected=affected)
        if miss or name_bad:
            gate("FAIL", f"D `{Path(f).parent.name}/{Path(f).name}`：帧路径缺失 {miss}，帧名和 meta 不符 {name_bad}")
        if affected:
            gate("FAIL", f"D `{Path(f).parent.name}/{Path(f).name}`：{affected} 条按错的表达式序号取了 GT mask")
    # train / val 视频重叠
    for f in files:
        if f.endswith("train.jsonl"):
            g = f[: -len("train.jsonl")] + "val.jsonl"
            if f in vids_by_file and g in vids_by_file:
                ov = vids_by_file[f] & vids_by_file[g]
                out(f"\n`{Path(f).parent.name}`：train/val 视频重叠 {len(ov)}")
                gate("PASS" if not ov else "FAIL", f"D `{Path(f).parent.name}` train/val 视频重叠 {len(ov)}")
    SUMMARY["D_ground"] = res


# ---------------------------------------------------------------- E Verify 缓存
def iter_metas(split_root):
    for d, _, fs in os.walk(split_root):
        if "meta.json" in fs:
            yield os.path.join(d, "meta.json")


def fit(g, shape):
    """和 dataset.py 一样：GT 尺寸和预测不同就最近邻缩放到预测尺寸。"""
    if g.shape == shape:
        return g
    from PIL import Image
    return np.array(Image.fromarray(g.astype(np.uint8)).resize((shape[1], shape[0]), Image.NEAREST))


def j_cache_style(cand, gt_seq):
    """现有缓存口径：只在 GT 非空的帧上平均，候选缺帧记 0。"""
    vals = []
    for f, rle in enumerate(gt_seq):
        g = decode(rle) if rle else None
        if g is None:
            continue
        m = cand.get(f)
        vals.append(iou(m, fit(g, m.shape)) if m is not None else 0.0)
    return float(np.mean(vals)) if vals else 0.0


def j_eval_style(cand, gt_seq, shape):
    """评测口径：所有标注帧都算，目标不在画面时 GT 为空（预测也空记 1）。"""
    vals = []
    for f, rle in enumerate(gt_seq):
        g = decode(rle) if rle else None
        m = cand.get(f)
        if g is None:
            g = np.zeros(shape, np.uint8)
        g = fit(g, shape)
        if m is None:
            m = np.zeros(shape, np.uint8)
        vals.append(iou(m, g))
    return float(np.mean(vals)) if vals else 0.0


def label(j, ds):
    return "pos" if j >= POS_IOU[ds] else ("neg" if j <= NEG_IOU[ds] else "ign")


@section("E Verify 训练缓存")
def check_caches(dss, repo, caches, deep, recompute, seed, outdir):
    rng = random.Random(seed)
    from PIL import Image
    res = {}
    out("| 数据集 | 缓存 | meta | 覆盖表达式 | 字段不一致 | 帧号越界 | 排除表 | 候选数 均值/中位/最大 | max J≥0.5 | max J=0 | 没正样本也没排除（force_pos 硬选） | 序号受影响 |")
    out("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, ds in dss.items():
        cache_rel, excl_rel = caches.get(ds.cache_name, (None, None))
        if not cache_rel:
            continue
        root = Path(cache_rel) if os.path.isabs(cache_rel) else Path(repo) / cache_rel
        split_root = root / "train"
        if not split_root.is_dir():
            out(f"| {name} | `{root.name}` | 缺失 |||||||||||")
            continue
        excl = set()
        if excl_rel:
            ep = Path(excl_rel) if os.path.isabs(excl_rel) else Path(repo) / excl_rel
            if ep.is_file():
                excl = {(str(s["video_id"]), int(s["exp_id"])) for s in load_json(ep).get("samples", [])}
        c, ncand, maxj, src, bad_ex, noise = Counter(), [], [], Counter(), [], []
        metas = []
        for mp in iter_metas(split_root):
            try:
                m = load_json(mp)
            except Exception:
                c["unreadable"] += 1
                continue
            vid, eid = str(m.get("video_id")), int(m.get("exp_id", -1))
            k = (vid, eid)
            c["meta"] += 1
            metas.append((mp, m))
            iou_keys = {int(x) for x in m.get("iou_map", {})}
            obj = {int(x) for x in m.get("obj_ids") or iou_keys}
            fi = {int(x): v for x, v in (m.get("frame_indices") or {}).items()}
            if obj != iou_keys or not obj <= set(fi):
                c["field_mismatch"] += 1
                bad_ex.append([vid, eid, "fields"])
            nf = len(ds.videos.get(vid, {}).get("frames", []))
            if fi and nf and max(max(v) for v in fi.values() if v) >= nf:
                c["frame_oob"] += 1
                bad_ex.append([vid, eid, "frame_oob"])
            if vid not in ds.videos or str(eid) not in ds.videos.get(vid, {}).get("expressions", {}):
                c["not_in_meta"] += 1
            src[m.get("sam3_prompt_source", "?")] += 1
            ncand.append(len(obj))
            mj = max(m.get("iou_map", {}).values(), default=0.0)
            maxj.append(mj)
            if mj < POS_IOU[name] and k not in excl:
                noise.append(k)
            c["affected"] += k in ds.differ
            c["excluded"] += k in excl
        n_expr = len(ds.orders["insertion"]) if ds.orders else sum(len(d.get("expressions", {})) for d in ds.videos.values())
        arr = np.array(maxj) if maxj else np.zeros(1)
        nc = np.array(ncand) if ncand else np.zeros(1)
        out(f"| {name} | `{root.name}` | {c['meta']} | {pct(c['meta'], n_expr)} | {c['field_mismatch']} | {c['frame_oob']} | "
            f"{len(excl)}（命中 {c['excluded']}） | {nc.mean():.1f}/{np.median(nc):.0f}/{nc.max():.0f} | "
            f"{pct((arr >= 0.5).sum(), len(maxj))} | {pct((arr == 0).sum(), len(maxj))} | "
            f"{len(noise)}（{pct(len(noise), c['meta'])}） | {c['affected']} |")
        r = dict(c, prompt_source=dict(src), expressions=n_expr, excluded_list=len(excl),
                 force_pos_noise=len(noise), examples=bad_ex[:20])
        if c["field_mismatch"] or c["frame_oob"]:
            gate("FAIL", f"E {name}：字段不一致 {c['field_mismatch']}，帧号越界 {c['frame_oob']}")
        if noise:
            gate("WARN", f"E {name}：{len(noise)} 条没有 J≥{POS_IOU[name]} 的候选也没被排除，训练时 force_pos 会把 J 最高的"
                 "（可能接近 0）硬当正样本")
            (Path(outdir) / f"force_pos_noise_{name}.json").write_text(json.dumps(
                {"dataset": name, "pos_iou": POS_IOU[name], "samples": [{"video_id": v, "exp_id": e} for v, e in noise]}, indent=1))
        if c["affected"]:
            gate("FAIL" if ds.correct_order != "insertion" else "PASS",
                 f"E {name}：{c['affected']} 条缓存的 iou_map 是用 insertion 顺序的 GT 算的（正确顺序判定为 {ds.correct_order}）")
        # 深查：mask 文件
        rng.shuffle(metas)
        for mp, m in metas[:deep]:
            d = Path(mp).parent
            pad = int(m.get("frame_pad", 0))
            vp = m.get("video_path") or str(ds.root / "JPEGImages" / str(m.get("video_id")))
            stems = list_stems(vp) or []
            if stems and pad and len(stems[0]) != pad:
                c["pad_mismatch"] += 1
            sized = False
            for o, fl in (m.get("frame_indices") or {}).items():
                od = d / "masks" / f"obj_{o}"
                have = {os.path.splitext(x)[0] for x in os.listdir(od)} if od.is_dir() else set()
                want = {f"{int(x):0{pad}d}" if pad else str(int(x)) for x in fl}
                c["missing_mask_files"] += len(want - have)
                if have and stems and not sized:
                    sized = True
                    with Image.open(od / f"{sorted(have)[0]}.png") as im, Image.open(
                            glob.glob(os.path.join(vp, stems[0] + ".*"))[0]) as fr:
                        c["mask_size_mismatch"] += im.size != fr.size
        r.update(pad_mismatch=c["pad_mismatch"], missing_mask_files=c["missing_mask_files"],
                 mask_size_mismatch=c["mask_size_mismatch"], deep_checked=min(deep, len(metas)))
        if c["missing_mask_files"] or c["mask_size_mismatch"] or c["pad_mismatch"]:
            gate("FAIL", f"E {name}：抽查 {min(deep, len(metas))} 条，缺 mask 文件 {c['missing_mask_files']}，"
                 f"尺寸不符 {c['mask_size_mismatch']}，frame_pad 不符 {c['pad_mismatch']}")
        # 重算 J
        if recompute and ds.load_md() is not None:
            pick = [x for x in metas if (str(x[1]["video_id"]), int(x[1]["exp_id"])) in ds.differ][: recompute // 2]
            seen = {x[0] for x in pick}
            pick += [x for x in metas if x[0] not in seen][: recompute - len(pick)]
            err, flips_order, flips_def, n_c = [], 0, 0, 0
            for mp, m in pick:
                vid, eid = str(m["video_id"]), int(m["exp_id"])
                pad = int(m.get("frame_pad", 0))
                d = Path(mp).parent
                g_ins = ds.seq(vid, eid, "insertion" if ds.mode == "expression_index" else None)
                g_ok = ds.seq(vid, eid, ds.correct_order if ds.mode == "expression_index" else None)
                if g_ins is None:
                    continue
                for o, fl in (m.get("frame_indices") or {}).items():
                    cand = {}
                    for f in fl:
                        p = d / "masks" / f"obj_{o}" / (f"{int(f):0{pad}d}.png" if pad else f"{int(f)}.png")
                        a = read_png(p)
                        if a is not None:
                            cand[int(f)] = (a > 0).astype(np.uint8)
                    if not cand:
                        continue
                    shape = next(iter(cand.values())).shape
                    j0 = j_cache_style(cand, g_ins)
                    j1 = j_cache_style(cand, g_ok) if g_ok is not None else 0.0
                    j2 = j_eval_style(cand, g_ok, shape) if g_ok is not None else 0.0
                    err.append(abs(j0 - float(m["iou_map"].get(str(o), m["iou_map"].get(o, 0.0)))))
                    flips_order += label(j0, name) != label(j1, name)
                    flips_def += label(j1, name) != label(j2, name)
                    n_c += 1
            ds.md = None
            r["recompute"] = dict(candidates=n_c, reproduce_err_max=round(max(err), 4) if err else None,
                                  label_flip_by_order=flips_order, label_flip_by_j_definition=flips_def)
            out(f"\n{name} 重算 {n_c} 个候选：缓存口径复现误差最大 {r['recompute']['reproduce_err_max']}；"
                f"换正确顺序后正/负/忽略标签变的 {flips_order}（{pct(flips_order, n_c)}）；"
                f"改成评测口径（目标不在的帧也算）后变的 {flips_def}（{pct(flips_def, n_c)}）。")
            if err and max(err) > 0.02:
                gate("WARN", f"E {name}：重算 J 和缓存 iou_map 对不上（最大误差 {max(err):.3f}），缓存可能不是用现在的 GT 算的")
        res[name] = r
        if src:
            out(f"\n{name} SAM3 提示来源：{dict(src)}")
    SUMMARY["E_caches"] = res


# ---------------------------------------------------------------- F 泄漏
def dhash(path):
    from PIL import Image
    with Image.open(path) as im:
        a = np.asarray(im.convert("L").resize((9, 8)), dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int(sum(1 << i for i, b in enumerate(bits) if b))


def popcount64(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int64)


def first_frames(meta_path, frame_root):
    res = {}
    if not meta_path or not Path(meta_path).is_file():
        return res
    for vid, d in load_json(meta_path).get("videos", {}).items():
        fr = sorted(str(f) for f in d.get("frames", []))
        if not fr:
            continue
        for ext in IMG_EXT:
            p = Path(frame_root) / vid / f"{fr[0]}{ext}"
            if p.is_file():
                res[vid] = str(p)
                break
    return res


@section("F 跨数据集泄漏（训练视频 vs 评测视频，首帧 dHash）")
def check_leak(vd, revos_root, thr):
    vd, rr = Path(vd), Path(revos_root)
    train = {
        "mevis": first_frames(find_meta(vd / "mevis/train"), vd / "mevis/train/JPEGImages"),
        "davis17": first_frames(find_meta(vd / "davis17/train"), vd / "davis17/train/JPEGImages"),
        "refytvos": first_frames(find_meta(vd / "Ref-Youtube-VOS/train"), vd / "Ref-Youtube-VOS/train/JPEGImages"),
        "revos": first_frames(rr / "meta_expressions_train_.json", rr),
    }
    evals = {
        "mevis/valid_u": first_frames(find_meta(vd / "mevis/valid_u"), vd / "mevis/valid_u/JPEGImages"),
        "mevis/valid": first_frames(find_meta(vd / "mevis/valid"), vd / "mevis/valid/JPEGImages"),
        "davis17/valid": first_frames(find_meta(vd / "davis17/valid"), vd / "davis17/valid/JPEGImages"),
        "refytvos/valid": first_frames(find_meta(vd / "Ref-Youtube-VOS/valid"), vd / "Ref-Youtube-VOS/valid/JPEGImages"),
        "revos/valid": first_frames(rr / "meta_expressions_valid_.json", rr),
    }

    def hashes(d):
        ks, hs = [], []
        for k, p in d.items():
            try:
                h = dhash(p)
            except Exception:
                continue
            if bin(h).count("1") in (0, 64):  # 纯色帧，不比
                continue
            ks.append(k)
            hs.append(h)
        return ks, np.array(hs, dtype=np.uint64)

    th = {k: hashes(v) for k, v in train.items()}
    out("| 评测集 | 视频 | " + " | ".join(f"∩ {k} train" for k in th) + " |")
    out("|---|---|" + "---|" * len(th))
    res, examples = {}, []
    for en, ed in evals.items():
        ek, eh = hashes(ed)
        row = []
        for tn, (tk, thh) in th.items():
            hit = 0
            if len(ek) and len(tk):
                for i in range(len(ek)):
                    dist = popcount64(np.bitwise_xor(thh, eh[i]))
                    j = np.where(dist <= thr)[0]
                    if len(j):
                        hit += 1
                        if len(examples) < 30:
                            examples.append([en, ek[i], tn, tk[int(j[0])], int(dist[j[0]])])
            row.append(hit)
            res[f"{en}|{tn}"] = hit
        out(f"| {en} | {len(ek)} | " + " | ".join(str(x) for x in row) + " |")
    out(f"\n阈值：64 位 dHash 汉明距离 ≤ {thr}。命中只是「首帧几乎一样」的候选，要人眼看例子确认：")
    for e in examples[:15]:
        out(f"- {e[0]} `{e[1]}` ≈ {e[2]} train `{e[3]}`（距离 {e[4]}）")
    SUMMARY["F_leak"] = {"pairs": res, "examples": examples}
    cross = {k: v for k, v in res.items() if v and k.split("/")[0] != k.split("|")[1]}
    gate("PASS" if not cross else "WARN",
         f"F 跨数据集泄漏候选：{sum(cross.values())} 个评测视频首帧和别的数据集训练视频几乎一样" + (f" {cross}" if cross else ""))


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="/mnt/csip-500/share/kongjiawei/Sa2VA-main")
    ap.add_argument("--video-datas", default=VD)
    ap.add_argument("--revos-root", default=os.path.join(VD, "revos"))
    ap.add_argument("--gv2", default=str(HERE.parent.parent / "ground_v2"))
    ap.add_argument("--datasets", default="mevis,davis17,refytvos,revos")
    ap.add_argument("--ground-jsonl", nargs="*", default=None, help="默认：v1 两个 + v2 两个 train/val")
    ap.add_argument("--png-frames", type=int, default=3, help="B：每条表达式比对的帧数")
    ap.add_argument("--png-max", type=int, default=800, help="B：最多比对多少条受影响的表达式")
    ap.add_argument("--deep", type=int, default=300, help="E：抽查 mask 文件的 meta 条数")
    ap.add_argument("--recompute", type=int, default=150, help="E：每个缓存重算 J 的表达式条数（0 关闭）")
    ap.add_argument("--leak", action="store_true", help="F：跨数据集泄漏检查（读每个视频首帧，几分钟）")
    ap.add_argument("--leak-thr", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(HERE.parent / "runs" / "audit"))
    a = ap.parse_args(argv)
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    names = [n for n in a.datasets.split(",") if n in TRAIN_SETS]
    dss = {}
    for n in names:
        try:
            dss[n] = DS(n, a.video_datas, a.revos_root)
        except Exception as e:
            out(f"- {n} 读取失败：{type(e).__name__}: {e}")
    check_frames(dss, a.revos_root)
    check_index(dss, a.png_frames, a.png_max, a.seed, od)
    check_categories(dss, a.video_datas, a.repo)
    files = a.ground_jsonl
    if files is None:
        files = []
        for d in (Path(a.repo) / "category_grounding_mvp/data/refytvos_query_grounding",
                  Path(a.repo) / "category_grounding_mvp/data/revos_query_grounding_api_20260717",
                  Path(a.gv2) / "data/revos_v2", Path(a.gv2) / "data/refytvos_v2"):
            files += [str(d / "train.jsonl"), str(d / "val.jsonl")]
    yt = dss.get("refytvos") or DS("refytvos", a.video_datas, a.revos_root)  # 只用它的 meta 和顺序判定
    check_ground(yt, a.revos_root, files, od)
    check_caches(dss, a.repo, VERIFY_CACHES, a.deep, a.recompute, a.seed, od)
    if a.leak:
        check_leak(a.video_datas, a.revos_root, a.leak_thr)

    order = {"FAIL": 0, "WARN": 1, "PASS": 2}
    head = ["# 训练集审计", "",
            f"生成于 {time.strftime('%Y-%m-%d %H:%M')}，用时 {time.time() - t0:.0f}s。只读，没有改任何数据。", "",
            "## 结论（FAIL 必须先修再训练，WARN 要看一眼）", ""]
    for lv, t in sorted(GATES, key=lambda g: order[g[0]]):
        head.append(f"- **{lv}** {t}")
    (od / "trainsets.md").write_text("\n".join(head + LINES) + "\n", encoding="utf-8")
    SUMMARY["gates"] = GATES
    (od / "trainsets.json").write_text(json.dumps(SUMMARY, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"[audit] 报告：{od / 'trainsets.md'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
