#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import Counter
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as cocomask

HTML = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>ReVOS Low Recall Viewer</title>
<style>
:root{--ink:#20231f;--muted:#687066;--paper:#f4efe5;--panel:#fffaf0;--line:#d8cbb7;--accent:#176f5d;--hot:#bb3f32;--blue:#2f6f9f}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:Inter,Arial,sans-serif;font-size:14px}header{position:sticky;top:0;z-index:5;background:rgba(255,250,240,.96);border-bottom:1px solid var(--line);padding:14px 18px}h1{margin:0 0 4px;font-size:22px}.sub,.note{color:var(--muted);line-height:1.45}.layout{display:grid;grid-template-columns:450px 1fr;min-height:calc(100vh - 70px)}aside{border-right:1px solid var(--line);padding:14px;overflow:auto;max-height:calc(100vh - 70px)}main{padding:16px 20px 32px;overflow:auto}input,select,button{border:1px solid var(--line);background:var(--panel);border-radius:7px;padding:8px 9px;font:inherit;color:var(--ink)}button{cursor:pointer}.active{background:var(--accent);color:#fff;border-color:var(--accent)}.filters{display:grid;gap:8px;margin-bottom:10px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}.tabs{display:flex;flex-wrap:wrap;gap:6px}.tabs button{font-size:12px;padding:6px 8px}.sample{border:1px solid var(--line);background:rgba(255,250,240,.82);border-radius:8px;margin:8px 0;padding:10px;cursor:pointer}.sample:hover{border-color:var(--accent)}.sample.sel{border-color:var(--hot);box-shadow:0 0 0 2px rgba(187,63,50,.12)}.row{display:flex;align-items:center;justify-content:space-between;gap:8px}.id{font-weight:700}.expr{margin-top:6px;color:var(--muted);font-size:13px}.chips{display:flex;flex-wrap:wrap;gap:5px;margin-top:7px}.chip{border:1px solid var(--line);background:#fffdf8;border-radius:999px;padding:3px 7px;font-size:12px}.bad{color:var(--hot);border-color:rgba(187,63,50,.42)}.good{color:var(--accent);border-color:rgba(23,111,93,.38)}.blue{color:var(--blue);border-color:rgba(47,111,159,.38)}.panel{background:rgba(255,250,240,.76);border:1px solid var(--line);border-radius:8px;padding:13px;margin-bottom:13px}.metrics{display:grid;grid-template-columns:repeat(7,minmax(96px,1fr));gap:9px;margin-top:12px}.metric{border:1px solid var(--line);border-radius:8px;background:var(--panel);padding:9px}.metric b{display:block;font-size:18px;word-break:break-word}.metric span{font-size:12px;color:var(--muted)}.viewer-controls{display:grid;grid-template-columns:minmax(240px,1fr) auto auto auto;gap:10px;align-items:center;margin:10px 0 12px}.range{display:grid;grid-template-columns:64px 1fr 92px;gap:8px;align-items:center}.range input{width:100%;padding:0}.toggle{display:inline-flex;gap:6px;align-items:center;white-space:nowrap}.panels{display:grid;grid-template-columns:repeat(3,minmax(240px,1fr));gap:12px}figure{margin:0;border:1px solid var(--line);border-radius:8px;background:#171916;overflow:hidden}figure img{display:block;width:100%;height:42vh;object-fit:contain;background:#171916}figcaption{display:flex;justify-content:space-between;gap:8px;background:var(--panel);padding:8px 10px;font-size:13px}.candidate-strip{display:flex;flex-wrap:wrap;gap:7px;margin:10px 0}.candidate-strip button{font-size:12px;padding:6px 8px}table{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--line);border-radius:8px;overflow:hidden}th,td{border-bottom:1px solid var(--line);padding:7px 8px;text-align:left;font-size:13px}th{color:var(--muted)}.pager{display:flex;align-items:center;gap:8px;margin:10px 0}.pager button{padding:6px 9px}@media(max-width:1150px){.layout{grid-template-columns:1fr}aside{max-height:440px;border-right:0;border-bottom:1px solid var(--line)}.panels{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(2,1fr)}}
</style>
</head>
<body>
<header><h1 id="pageTitle">ReVOS Low Recall Viewer</h1><div class="sub">max_j < 0.5 的候选召回失败样本。三栏：Raw / Candidate / GT；按需加载帧与掩码。</div></header>
<div class="layout"><aside>
<div class="filters"><input id="q" placeholder="video / exp / expression / category" /><div class="grid"><select id="category"></select><select id="sort"><option value="delta">Sort: delta desc</option><option value="new_max_j">Sort: low new J</option><option value="old_max_j">Sort: low old J</option><option value="video_id">Sort: video</option></select></div><div class="tabs"><button data-bucket="all" class="active">All</button><button data-bucket="improved">improved</button><button data-bucket="worse">worse</button><button data-bucket="same">same</button><button data-bucket="still_failed">new &lt;0.5</button><button data-bucket="rescued">new >=0.5</button><button data-bucket="zero">new =0</button></div></div>
<div id="summary" class="note"></div><div class="pager"><button id="prev">Prev</button><span id="pageInfo"></span><button id="next">Next</button></div><div id="list"></div>
</aside><main><section id="empty" class="panel"><h2>选择左侧样本</h2><p class="note">建议先看 worse 和 still failed，再看 rescued 样本确认候选是否确实合理。</p></section><section id="detail" style="display:none"><div class="panel"><div class="row"><h2 id="title"></h2><span id="frameBadge" class="chip"></span></div><div id="expr" class="note"></div><div id="metrics" class="metrics"></div></div><div class="viewer-controls"><div class="range"><span>Frame</span><input id="frame" type="range" min="0" value="0" /><span id="frameText"></span></div><label class="toggle"><input id="showMask" type="checkbox" checked />Mask</label><button id="play">Play</button><button id="gtOnly">GT frame</button></div><div id="candidateButtons" class="candidate-strip"></div><div class="panels"><figure><img id="rawImg"><figcaption><b>1. Raw</b><span>原图</span></figcaption></figure><figure><img id="candImg"><figcaption><b id="candTitle">2. Candidate</b><span>候选 mask</span></figcaption></figure><figure><img id="gtImg"><figcaption><b>3. GT</b><span id="gtRangeText">表达式 GT</span></figcaption></figure></div><h3>Candidate table</h3><div id="candTable"></div></section></main></div>
<script>
let state={page:1,pageSize:50,bucket:'all',total:0,items:[]};let active=null,activeCandidate=null,timer=null,masksVisible=true;const $=id=>document.getElementById(id);function fmt(x,n=3){return x===undefined||x===null||x===''?'-':Number(x).toFixed(n)}function params(extra={}){return new URLSearchParams({page:state.page,page_size:state.pageSize,q:q.value,category:category.value,sort:sort.value,bucket:state.bucket,...extra})}
async function loadSummary(){const s=await(await fetch('/api/summary')).json();pageTitle.textContent=`${s.name} ReVOS Low Recall Viewer`;summary.innerHTML=`samples: <b>${s.num_samples}</b> · improved ${s.num_improved} · worse ${s.num_worse} · same ${s.num_same}<br>old mean=${fmt(s.old_mean_max_j)} new mean=${fmt(s.new_mean_max_j_missing_as_zero)} · rescued=${s.new_num_ge_0p5_missing_as_zero} · zero=${s.new_num_eq_0_missing_as_zero}`;category.innerHTML='<option value="">All categories</option>'+s.categories.map(c=>`<option value="${c[0]}">${c[0]} (${c[1]})</option>`).join('')}
async function loadList(){const data=await(await fetch('/api/samples?'+params())).json();state.items=data.items;state.total=data.total;pageInfo.textContent=`${state.page}/${Math.max(1,Math.ceil(state.total/state.pageSize))} · ${state.total}`;renderList()}
function renderList(){list.innerHTML=state.items.map(r=>{const cls=r.status==='improved'?'good':(r.status==='worse'?'bad':'blue');return `<div class="sample ${active&&active.video_id===r.video_id&&String(active.exp_id)===String(r.exp_id)?'sel':''}" onclick="openSample('${r.video_id}','${r.exp_id}')"><div class="row"><span class="id">${r.video_id}/${r.exp_id}</span><span class="chip ${cls}">${r.status} Δ ${fmt(r.delta)}</span></div><div class="expr">${r.expression||''}</div><div class="chips"><span class="chip">old ${fmt(r.old_max_j)}</span><span class="chip ${Number(r.new_max_j_filled)>=.5?'good':'bad'}">new ${fmt(r.new_max_j_filled)}</span><span class="chip">${r.category||'-'}</span><span class="chip">N ${r.new_num_candidates??'-'}</span></div></div>`}).join('')}
function imageUrl(view,extra={}){const p=new URLSearchParams({video_id:active.video_id,exp_id:active.exp_id,frame_idx:frame.value,view,show_mask:masksVisible?'1':'0',...extra});return '/image?'+p+'&t='+Date.now()}
function updateImages(){if(!active)return;rawImg.src=imageUrl('raw');const cid=activeCandidate||active.best_obj_id;candImg.src=imageUrl('candidate',{obj_id:cid});gtImg.src=imageUrl('gt');candTitle.textContent=`2. Candidate obj_${cid}`;gtRangeText.textContent=`GT idx: ${active.gt_frame_index_range||'-'} | frame: ${active.gt_frame_range||'-'}`;frameText.textContent=`${Number(frame.value)+1}/${active.frames.length}`;frameBadge.textContent=`frame ${active.frames[frame.value]}`}
function renderDetail(){empty.style.display='none';detail.style.display='block';title.textContent=`${active.video_id}/${active.exp_id}`;expr.textContent=`${active.expression} | GT frames=${active.gt_frame_range||'-'} | old category=${active.old_category||active.category} | current category=${active.current_category||active.category} | prompt=${active.sam3_prompt_text} | status=${active.status}`;metrics.innerHTML=[['old J',active.old_max_j],['new J',active.new_max_j_filled],['delta',active.delta],['best obj',active.best_obj_id],['candidates',active.new_num_candidates],['GT idx',active.gt_frame_index_range],['filter',active.candidate_filter]].map(([k,v])=>`<div class="metric"><b>${typeof v==='number'?fmt(v):v}</b><span>${k}</span></div>`).join('');frame.max=Math.max(0,active.frames.length-1);if(!activeCandidate||!active.per_candidate[String(activeCandidate)]){activeCandidate=active.best_obj_id;frame.value=0}const ids=Object.keys(active.per_candidate||{}).sort((a,b)=>Number(a)-Number(b));candidateButtons.innerHTML=ids.map(id=>`<button class="${id===String(activeCandidate)?'active':''}" onclick="setCandidate('${id}')">obj_${id} J=${fmt(active.per_candidate[id].J)}</button>`).join('');candTable.innerHTML=`<table><thead><tr><th>obj</th><th>J</th><th>frames</th><th>idx range</th><th>frame range</th><th>SAM3 score sum</th></tr></thead><tbody>`+ids.map(id=>{const c=active.per_candidate[id];return `<tr><td>obj_${id}${id===String(active.best_obj_id)?' ★':''}</td><td>${fmt(c.J)}</td><td>${c.present_frames}</td><td>${c.frame_index_range||'-'}</td><td>${c.frame_range}</td><td>${c.score_sum==null?'-':fmt(c.score_sum)}</td></tr>`}).join('')+'</tbody></table>';updateImages()}
window.setCandidate=id=>{activeCandidate=String(id);renderDetail()};window.openSample=async(video_id,exp_id)=>{active=await(await fetch(`/api/sample?video_id=${video_id}&exp_id=${exp_id}`)).json();activeCandidate=active.best_obj_id;renderList();renderDetail()};q.oninput=()=>{state.page=1;loadList()};category.onchange=()=>{state.page=1;loadList()};sort.onchange=()=>{state.page=1;loadList()};prev.onclick=()=>{state.page=Math.max(1,state.page-1);loadList()};next.onclick=()=>{state.page+=1;loadList()};document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>{document.querySelectorAll('.tabs button').forEach(x=>x.classList.remove('active'));b.classList.add('active');state.bucket=b.dataset.bucket;state.page=1;loadList()});frame.oninput=updateImages;showMask.onchange=()=>{masksVisible=showMask.checked;updateImages()};play.onclick=()=>{if(timer){clearInterval(timer);timer=null;play.textContent='Play';return}play.textContent='Pause';timer=setInterval(()=>{if(active){frame.value=(Number(frame.value)+1)%active.frames.length;updateImages()}},220)};gtOnly.onclick=()=>{if(!active)return;frame.value=active.gt_first_frame_idx||0;updateImages()};loadSummary().then(loadList)
</script>
</body>
</html>
"""

COLORS = {"gt": (38, 160, 87), "candidate": (225, 144, 35)}


def _json_response(handler, payload, status=HTTPStatus.OK):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers(); handler.wfile.write(data)


def _html_response(handler, html):
    data = html.encode("utf-8")
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers(); handler.wfile.write(data)


def _png_response(handler, image):
    buf = BytesIO(); image.save(buf, format="PNG"); data = buf.getvalue()
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", "image/png")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers(); handler.wfile.write(data)


def _overlay(base, mask, color, alpha=0.52):
    base = base.convert("RGBA"); mask = mask > 0
    if mask.shape != (base.height, base.width):
        mask = np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize((base.width, base.height), Image.NEAREST)) > 0
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0)); arr = np.array(layer)
    arr[mask] = (color[0], color[1], color[2], int(255 * alpha))
    return Image.alpha_composite(base, Image.fromarray(arr)).convert("RGB")


def _draw_label(img, text):
    out = img.convert("RGB"); draw = ImageDraw.Draw(out); pad = 8
    bbox = draw.textbbox((0, 0), text)
    draw.rectangle((0, 0, bbox[2] + pad * 2, bbox[3] + pad * 2), fill=(255, 250, 240))
    draw.text((pad, pad), text, fill=(29, 33, 28))
    return out


def _compact(indices, frames):
    vals = sorted({int(i) for i in indices if 0 <= int(i) < len(frames)})
    if not vals: return "-"
    ranges = []; start = prev = vals[0]
    for idx in vals[1:]:
        if idx == prev + 1: prev = idx
        else: ranges.append((start, prev)); start = prev = idx
    ranges.append((start, prev))
    return ", ".join(frames[a] if a == b else f"{frames[a]}-{frames[b]}" for a, b in ranges)


def _compact_local(indices):
    vals = sorted({int(i) for i in indices})
    if not vals: return "-"
    ranges = []; start = prev = vals[0]
    for idx in vals[1:]:
        if idx == prev + 1: prev = idx
        else: ranges.append((start, prev)); start = prev = idx
    ranges.append((start, prev))
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in ranges)


def _load_binary_png(path, hw):
    if not os.path.exists(path): return np.zeros(hw, dtype=np.uint8)
    arr = np.array(Image.open(path).convert("L")) > 0
    if arr.shape != hw:
        arr = np.array(Image.fromarray(arr.astype(np.uint8) * 255).resize((hw[1], hw[0]), Image.NEAREST)) > 0
    return arr.astype(np.uint8)


def _decode_rle(mask_rle):
    if not mask_rle: return None
    if isinstance(mask_rle, (list, tuple)):
        merged = None
        for item in mask_rle:
            dec = _decode_rle(item)
            if dec is None: continue
            merged = dec.astype(np.uint8) if merged is None else np.logical_or(merged > 0, dec > 0).astype(np.uint8)
        return merged
    if isinstance(mask_rle, dict) and isinstance(mask_rle.get("counts"), str):
        mask_rle = dict(mask_rle); mask_rle["counts"] = mask_rle["counts"].encode("utf-8")
    dec = cocomask.decode(mask_rle)
    if dec.ndim == 3: dec = dec.any(axis=2).astype(np.uint8)
    return dec.astype(np.uint8)


class State:
    def __init__(self, args):
        self.args = args
        payload = json.load(open(args.report_json, "r", encoding="utf-8"))
        self.summary = payload.get("summary", {})
        self.samples = payload.get("samples", [])
        self.by_key = {(str(r["video_id"]), str(r["exp_id"])): r for r in self.samples}
        self.meta = json.load(open(args.meta_path, "r", encoding="utf-8")).get("videos", {})
        self.mask_dict = self._load_mask_dict(args.mask_dict_path)
        self.mask_mode = self._infer_mask_dict_mode()
        self.expr_global_index = self._build_expression_global_index() if self.mask_mode == "expression_index" else {}
        self.categories = sorted(Counter(str(r.get("category", "")) for r in self.samples).items(), key=lambda x: (-x[1], x[0]))

    def _load_mask_dict(self, path):
        if str(path).lower().endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        with open(path, "rb") as f:
            return pickle.load(f)

    def _build_expression_global_index(self):
        out = {}; idx = 0
        for video_id, video_meta in self.meta.items():
            for exp_id in video_meta.get("expressions", {}).keys():
                out[(str(video_id), int(exp_id))] = idx; idx += 1
        return out

    def _infer_mask_dict_mode(self):
        total_expr = sum(len(v.get("expressions", {})) for v in self.meta.values())
        keys = []
        try:
            for k in self.mask_dict.keys(): keys.append(int(k))
        except Exception:
            return "obj_or_anno"
        if total_expr > 0 and len(keys) == total_expr and min(keys) == 0 and max(keys) == total_expr - 1:
            return "expression_index"
        return "obj_or_anno"

    def frames(self, video_id):
        frames = [str(x) for x in self.meta.get(str(video_id), {}).get("frames", [])]
        if frames: return frames
        image_dir = os.path.join(self.args.image_root, str(video_id))
        return sorted(os.path.splitext(x)[0] for x in os.listdir(image_dir) if x.lower().endswith((".jpg", ".png", ".jpeg")))

    def cache_dir(self, video_id, exp_id):
        return os.path.join(self.args.cache_root, self.args.split, str(video_id), str(exp_id))

    def image_path(self, video_id, frame):
        for ext in (".jpg", ".png", ".jpeg"):
            p = os.path.join(self.args.image_root, str(video_id), f"{frame}{ext}")
            if os.path.exists(p): return p
        return os.path.join(self.args.image_root, str(video_id), f"{frame}.jpg")

    def gt_mask(self, video_id, exp_id, frame_idx, hw):
        video_id = str(video_id); exp_id = int(exp_id)
        seq = None
        if self.mask_mode == "expression_index":
            expr_idx = self.expr_global_index.get((video_id, exp_id))
            seq = self.mask_dict.get(str(expr_idx)) if expr_idx is not None else None
            if seq is None and expr_idx is not None: seq = self.mask_dict.get(expr_idx)
        else:
            exp = self.meta[video_id]["expressions"][str(exp_id)]
            ids = exp.get("anno_id") or exp.get("obj_id") or []
            if isinstance(ids, (int, str)): ids = [ids]
            merged = None
            for oid in ids:
                cur = self.mask_dict.get(str(oid)) or self.mask_dict.get(int(oid))
                if cur is None or frame_idx >= len(cur): continue
                dec = _decode_rle(cur[frame_idx])
                if dec is None: continue
                merged = dec if merged is None else np.logical_or(merged > 0, dec > 0).astype(np.uint8)
            if merged is not None:
                return self._fit_mask(merged, hw)
        if seq is None or frame_idx >= len(seq):
            return np.zeros(hw or (1, 1), dtype=np.uint8)
        dec = _decode_rle(seq[frame_idx])
        return self._fit_mask(dec if dec is not None else np.zeros(hw or (1, 1), dtype=np.uint8), hw)

    def _fit_mask(self, mask, hw):
        arr = (np.asarray(mask) > 0).astype(np.uint8)
        if hw is not None and arr.shape != hw:
            arr = np.array(Image.fromarray(arr * 255).resize((hw[1], hw[0]), Image.NEAREST)) > 0
            arr = arr.astype(np.uint8)
        return arr

    def candidate_mask(self, video_id, exp_id, obj_id, frame_idx, frame_name, hw):
        obj_dir = os.path.join(self.cache_dir(video_id, exp_id), "masks", f"obj_{obj_id}")
        names = [f"{int(frame_idx):0{len(str(frame_name))}d}.png", f"{int(frame_idx)}.png", f"{frame_name}.png"]
        for name in names:
            p = os.path.join(obj_dir, name)
            if os.path.exists(p): return _load_binary_png(p, hw)
        return np.zeros(hw, dtype=np.uint8)

    def sample_detail(self, video_id, exp_id):
        base = dict(self.by_key[(str(video_id), str(exp_id))])
        meta_path = os.path.join(self.cache_dir(video_id, exp_id), "meta.json")
        meta = json.load(open(meta_path, "r", encoding="utf-8"))
        frames = self.frames(video_id)
        iou_map = {str(k): float(v) for k, v in (meta.get("iou_map") or {}).items()}
        frame_indices = {str(k): v for k, v in (meta.get("frame_indices") or {}).items()}
        score_sums = {str(k): float(v) for k, v in (meta.get("candidate_score_sums") or {}).items()}
        per = {}
        for obj_id in sorted(set(iou_map) | set(frame_indices), key=lambda x: int(x) if str(x).isdigit() else str(x)):
            inds = [int(x) for x in frame_indices.get(obj_id, [])]
            per[obj_id] = {"J": float(iou_map.get(obj_id, 0.0)), "present_frames": len(set(inds)), "frame_index_range": _compact_local(inds), "frame_range": _compact(inds, frames), "score_sum": score_sums.get(str(obj_id))}
        best = max(per, key=lambda k: per[k]["J"]) if per else "none"
        gt_present = []
        for i, _ in enumerate(frames):
            # Preserve native GT resolution for presence tests; resizing to 1x1
            # can erase a valid small target.
            if np.any(self.gt_mask(video_id, exp_id, i, None)):
                gt_present.append(i)
        old_category = base.get("category", "")
        current_category = meta.get("category", old_category)
        base.update({
            "frames": frames,
            "old_category": old_category,
            "current_category": current_category,
            "category": current_category,
            "best_obj_id": best,
            "per_candidate": per,
            "gt_present_frames": len(gt_present),
            "gt_frame_index_range": _compact_local(gt_present),
            "gt_frame_range": _compact(gt_present, frames),
            "gt_first_frame_idx": gt_present[0] if gt_present else 0,
            "sam3_prompt_text": meta.get("sam3_prompt_text", base.get("sam3_prompt_text", "")),
            "sam3_prompt_source": meta.get("sam3_prompt_source", base.get("sam3_prompt_source", "")),
            "candidate_filter": meta.get("candidate_filter", base.get("candidate_filter", "")),
            "new_num_candidates": int(meta.get("num_candidates", len(iou_map))),
        })
        return base

    def filtered(self, qs):
        q = str(qs.get("q", "")).lower().strip(); cat = str(qs.get("category", "")).strip(); bucket = str(qs.get("bucket", "all"))
        items = self.samples
        if cat: items = [r for r in items if str(r.get("category", "")) == cat]
        if q: items = [r for r in items if q in f"{r.get('video_id')} {r.get('exp_id')} {r.get('expression')} {r.get('category')}".lower()]
        if bucket in {"improved", "worse", "same"}: items = [r for r in items if str(r.get("status")) == bucket]
        elif bucket == "still_failed": items = [r for r in items if float(r.get("new_max_j_filled", 0.0)) < 0.5]
        elif bucket == "rescued": items = [r for r in items if float(r.get("new_max_j_filled", 0.0)) >= 0.5]
        elif bucket == "zero": items = [r for r in items if float(r.get("new_max_j_filled", 0.0)) <= 1e-12]
        sort = str(qs.get("sort", "delta"))
        if sort == "new_max_j": items = sorted(items, key=lambda r: (float(r.get("new_max_j_filled", 0.0)), str(r.get("video_id")), int(r.get("exp_id", 0))))
        elif sort == "old_max_j": items = sorted(items, key=lambda r: (float(r.get("old_max_j", 0.0)), str(r.get("video_id")), int(r.get("exp_id", 0))))
        elif sort == "video_id": items = sorted(items, key=lambda r: (str(r.get("video_id")), int(r.get("exp_id", 0))))
        else: items = sorted(items, key=lambda r: (float(r.get("delta", 0.0)), float(r.get("new_max_j_filled", 0.0))), reverse=True)
        return items


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args): sys.stderr.write("[rerun-viewer] " + fmt % args + "\n")
        def do_GET(self):
            parsed = urlparse(self.path); qs = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            try:
                if parsed.path == "/": return _html_response(self, HTML)
                if parsed.path == "/api/summary":
                    s = dict(state.summary); s["categories"] = state.categories; return _json_response(self, s)
                if parsed.path == "/api/samples":
                    items = state.filtered(qs); page = max(1, int(qs.get("page", "1"))); page_size = min(200, max(10, int(qs.get("page_size", "50"))))
                    start = (page - 1) * page_size; return _json_response(self, {"total": len(items), "page": page, "page_size": page_size, "items": items[start:start + page_size]})
                if parsed.path == "/api/sample": return _json_response(self, state.sample_detail(qs["video_id"], qs["exp_id"]))
                if parsed.path == "/image": return self._image(qs)
                return _json_response(self, {"error": "not found"}, HTTPStatus.NOT_FOUND)
            except Exception as exc:
                return _json_response(self, {"error": repr(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
        def _image(self, qs):
            video_id, exp_id = qs["video_id"], qs["exp_id"]
            sample = state.sample_detail(video_id, exp_id); frames = sample["frames"]
            idx = max(0, min(int(qs.get("frame_idx", "0")), len(frames) - 1)); frame = frames[idx]
            base = Image.open(state.image_path(video_id, frame)).convert("RGB"); hw = (base.height, base.width)
            view = qs.get("view", "raw"); show = qs.get("show_mask", "1") not in {"0", "false", "False", "off"}
            if view == "raw": return _png_response(self, _draw_label(base, f"{video_id}/{frame}"))
            if view == "gt":
                img = _overlay(base, state.gt_mask(video_id, exp_id, idx, hw), COLORS["gt"]) if show else base
                return _png_response(self, _draw_label(img, "GT" if show else "GT hidden"))
            if view == "candidate":
                obj_id = qs.get("obj_id", sample["best_obj_id"])
                img = _overlay(base, state.candidate_mask(video_id, exp_id, obj_id, idx, frame, hw), COLORS["candidate"]) if show else base
                return _png_response(self, _draw_label(img, f"Candidate obj_{obj_id}" if show else f"Candidate obj_{obj_id} hidden"))
            return _json_response(self, {"error": f"unknown view {view}"}, HTTPStatus.BAD_REQUEST)
    return Handler


def main():
    p = argparse.ArgumentParser(description="Failure rerun three-panel viewer")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--report-json", required=True)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--meta-path", required=True)
    p.add_argument("--image-root", required=True)
    p.add_argument("--mask-dict-path", required=True)
    args = p.parse_args()
    state = State(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print(f"Serving rerun viewer at http://{args.host}:{args.port}")
    print(f"Loaded {len(state.samples)} samples from {args.report_json}")
    print(f"mask_dict mode={state.mask_mode}")
    server.serve_forever()

if __name__ == "__main__":
    main()
