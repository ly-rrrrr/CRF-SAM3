#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image, ImageDraw

HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Ref-YT Train Low Recall Viewer</title>
  <style>
    :root{--ink:#1f2320;--muted:#687068;--paper:#f6f1e7;--panel:#fffaf0;--line:#d7cbb8;--accent:#176f5d;--hot:#bd3f2f;--blue:#286c9c;}
    *{box-sizing:border-box} body{margin:0;background:var(--paper);color:var(--ink);font-family:Inter,Arial,sans-serif;font-size:14px}
    header{position:sticky;top:0;z-index:5;background:rgba(255,250,240,.94);border-bottom:1px solid var(--line);padding:14px 18px}
    h1{margin:0 0 4px;font-size:22px} .sub,.note{color:var(--muted);line-height:1.45}.layout{display:grid;grid-template-columns:430px 1fr;min-height:calc(100vh - 70px)}
    aside{border-right:1px solid var(--line);padding:14px;overflow:auto;max-height:calc(100vh - 70px)} main{padding:16px 20px 32px;overflow:auto}
    input,select,button{border:1px solid var(--line);background:var(--panel);border-radius:7px;padding:8px 9px;font:inherit;color:var(--ink)}button{cursor:pointer}.active{background:var(--accent);color:white;border-color:var(--accent)}
    .grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}.filters{display:grid;gap:8px;margin-bottom:10px}.row{display:flex;align-items:center;justify-content:space-between;gap:8px}.tabs{display:flex;flex-wrap:wrap;gap:6px}.tabs button{font-size:12px;padding:6px 8px}
    .sample{border:1px solid var(--line);background:rgba(255,250,240,.82);border-radius:8px;margin:8px 0;padding:10px;cursor:pointer}.sample:hover{border-color:var(--accent)}.sample.sel{border-color:var(--hot);box-shadow:0 0 0 2px rgba(189,63,47,.12)}
    .id{font-weight:700}.expr{margin-top:6px;color:var(--muted);font-size:13px}.chips{display:flex;flex-wrap:wrap;gap:5px;margin-top:7px}.chip{border:1px solid var(--line);background:#fffdf8;border-radius:999px;padding:3px 7px;font-size:12px}.bad{color:var(--hot);border-color:rgba(189,63,47,.4)}.good{color:var(--accent);border-color:rgba(23,111,93,.35)}
    .panel{background:rgba(255,250,240,.76);border:1px solid var(--line);border-radius:8px;padding:13px;margin-bottom:13px}.metrics{display:grid;grid-template-columns:repeat(6,minmax(105px,1fr));gap:9px;margin-top:12px}.metric{border:1px solid var(--line);border-radius:8px;background:var(--panel);padding:9px}.metric b{display:block;font-size:19px;word-break:break-word}.metric span{font-size:12px;color:var(--muted)}
    .viewer-controls{display:grid;grid-template-columns:minmax(240px,1fr) auto auto auto;gap:10px;align-items:center;margin:10px 0 12px}.range{display:grid;grid-template-columns:64px 1fr 84px;gap:8px;align-items:center}.range input{width:100%;padding:0}.toggle{display:inline-flex;gap:6px;align-items:center;white-space:nowrap}
    .panels{display:grid;grid-template-columns:repeat(3,minmax(240px,1fr));gap:12px}figure{margin:0;border:1px solid var(--line);border-radius:8px;background:#171916;overflow:hidden}figure img{display:block;width:100%;height:42vh;object-fit:contain;background:#171916}figcaption{display:flex;justify-content:space-between;gap:8px;background:var(--panel);padding:8px 10px;font-size:13px}
    .candidate-strip{display:flex;flex-wrap:wrap;gap:7px;margin:10px 0}.candidate-strip button{font-size:12px;padding:6px 8px}table{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--line);border-radius:8px;overflow:hidden}th,td{border-bottom:1px solid var(--line);padding:7px 8px;text-align:left;font-size:13px}th{color:var(--muted)}
    .pager{display:flex;align-items:center;gap:8px;margin:10px 0}.pager button{padding:6px 9px}@media(max-width:1150px){.layout{grid-template-columns:1fr}aside{max-height:440px;border-right:0;border-bottom:1px solid var(--line)}.panels{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(2,1fr)}}
  </style>
</head>
<body>
<header><h1>Ref-Youtube-VOS Train Low Recall Viewer</h1><div class="sub">Offline cache low-recall samples. 三栏：Raw / Candidate / GT；支持分页、筛选、mask 显隐。</div></header>
<div class="layout">
<aside>
  <div class="filters">
    <input id="q" placeholder="video / exp / expression / prompt" />
    <div class="grid"><select id="category"></select><select id="sort"><option value="max_j">Sort: low max_j</option><option value="num_candidates">Sort: candidates</option><option value="video_id">Sort: video</option></select></div>
    <div class="tabs"><button data-bucket="all" class="active">All</button><button data-bucket="zero">max_j=0</button><button data-bucket="lt01">&lt;0.1</button><button data-bucket="top7">N=7</button><button data-bucket="backup">backup prompt</button></div>
  </div>
  <div id="summary" class="note"></div><div class="pager"><button id="prev">Prev</button><span id="pageInfo"></span><button id="next">Next</button></div><div id="list"></div>
</aside>
<main>
  <section id="empty" class="panel"><h2>选择左侧样本</h2><p class="note">建议先看 max_j=0 和 N=7 的拥挤类别样本，区分“完全错召回”和“top-k 饱和”。</p></section>
  <section id="detail" style="display:none">
    <div class="panel"><div class="row"><h2 id="title"></h2><span id="frameBadge" class="chip"></span></div><div id="expr" class="note"></div><div id="metrics" class="metrics"></div></div>
    <div class="viewer-controls"><div class="range"><span>Frame</span><input id="frame" type="range" min="0" value="0" /><span id="frameText"></span></div><label class="toggle"><input id="showMask" type="checkbox" checked />Mask</label><button id="play">Play</button><button id="gtOnly">GT frame</button></div>
    <div id="candidateButtons" class="candidate-strip"></div>
    <div class="panels"><figure><img id="rawImg"><figcaption><b>1. Raw</b><span>原图</span></figcaption></figure><figure><img id="candImg"><figcaption><b id="candTitle">2. Candidate</b><span>候选 mask</span></figcaption></figure><figure><img id="gtImg"><figcaption><b>3. GT</b><span id="gtRangeText">表达式目标 GT</span></figcaption></figure></div>
    <h3>Candidate table</h3><div id="candTable"></div>
  </section>
</main>
</div>
<script>
let state={page:1,pageSize:50,bucket:'all',total:0,items:[]}; let active=null, activeCandidate=null, timer=null, masksVisible=true;
const $=id=>document.getElementById(id); function fmt(x,n=3){return x===undefined||x===null?'-':Number(x).toFixed(n)}
function params(extra={}){const p=new URLSearchParams({page:state.page,page_size:state.pageSize,q:q.value,category:category.value,sort:sort.value,bucket:state.bucket,...extra});return p}
async function loadSummary(){const s=await (await fetch('/api/summary')).json(); summary.innerHTML=`samples max_j&lt;0.5: <b>${s.num_max_j_lt_0p5}</b> / ${s.num_generated_meta} (${fmt(100*s.rate_max_j_lt_0p5,2)}%)<br>mean=${fmt(s.mean_max_j)} median=${fmt(s.median_max_j)} zero≈${s.distribution_count_max_j_lt['1e-12']||0}`; category.innerHTML='<option value="">All categories</option>'+s.categories.map(c=>`<option value="${c[0]}">${c[0]} (${c[1]})</option>`).join('')}
async function loadList(){const data=await (await fetch('/api/samples?'+params())).json(); state.items=data.items; state.total=data.total; pageInfo.textContent=`${state.page}/${Math.max(1,Math.ceil(state.total/state.pageSize))} · ${state.total}`; renderList()}
function renderList(){list.innerHTML=state.items.map(r=>`<div class="sample ${active&&active.video_id===r.video_id&&String(active.exp_id)===String(r.exp_id)?'sel':''}" onclick="openSample('${r.video_id}','${r.exp_id}')"><div class="row"><span class="id">${r.video_id}/${r.exp_id}</span><span class="chip ${r.max_j<.1?'bad':'good'}">max_j ${fmt(r.max_j)}</span></div><div class="expr">${r.expression||''}</div><div class="chips"><span class="chip">${r.category||'-'}</span><span class="chip">N ${r.num_candidates}</span><span class="chip">${r.sam3_prompt_source||'-'}</span><span class="chip">prompt=${r.sam3_prompt_text||'-'}</span></div></div>`).join('')}
function imageUrl(view, extra={}){const p=new URLSearchParams({video_id:active.video_id,exp_id:active.exp_id,frame_idx:frame.value,view,show_mask:masksVisible?'1':'0',...extra});return '/image?'+p+'&t='+Date.now()}
function updateImages(){if(!active)return; rawImg.src=imageUrl('raw'); const cid=activeCandidate||active.best_obj_id; candImg.src=imageUrl('candidate',{obj_id:cid}); gtImg.src=imageUrl('gt'); candTitle.textContent=`2. Candidate obj_${cid}`; gtRangeText.textContent=`GT idx: ${active.gt_frame_index_range||'-'} | frame: ${active.gt_frame_range||'-'}`; frameText.textContent=`${Number(frame.value)+1}/${active.frames.length}`; frameBadge.textContent=`frame ${active.frames[frame.value]}`}
function renderDetail(){empty.style.display='none'; detail.style.display='block'; title.textContent=`${active.video_id}/${active.exp_id}`; expr.textContent=`${active.expression} | category=${active.category} | prompt=${active.sam3_prompt_text} | GT obj=${active.gt_obj_id}`; metrics.innerHTML=[['max_j',active.max_j],['best obj',active.best_obj_id],['candidates',active.num_candidates],['GT frames',active.gt_present_frames],['GT idx',active.gt_frame_index_range],['GT frames range',active.gt_frame_range],['prompt src',active.sam3_prompt_source],['frame count',active.frames.length]].map(([k,v])=>`<div class="metric"><b>${typeof v==='number'?fmt(v):v}</b><span>${k}</span></div>`).join(''); frame.max=Math.max(0,active.frames.length-1); if(!activeCandidate||!active.per_candidate[String(activeCandidate)]){activeCandidate=active.best_obj_id; frame.value=0} const ids=Object.keys(active.per_candidate||{}).sort((a,b)=>Number(a)-Number(b)); candidateButtons.innerHTML=ids.map(id=>`<button class="${id===String(activeCandidate)?'active':''}" onclick="setCandidate('${id}')">obj_${id} J=${fmt(active.per_candidate[id].J)}</button>`).join(''); candTable.innerHTML=`<table><thead><tr><th>obj</th><th>J</th><th>frames</th><th>idx range</th><th>frame range</th></tr></thead><tbody>`+ids.map(id=>{const c=active.per_candidate[id];return `<tr><td>obj_${id}${id===String(active.best_obj_id)?' ★':''}</td><td>${fmt(c.J)}</td><td>${c.present_frames}</td><td>${c.frame_index_range||'-'}</td><td>${c.frame_range}</td></tr>`}).join('')+'</tbody></table>'; updateImages()}
window.setCandidate=id=>{activeCandidate=String(id);renderDetail()}; window.openSample=async(video_id,exp_id)=>{active=await (await fetch(`/api/sample?video_id=${video_id}&exp_id=${exp_id}`)).json(); activeCandidate=active.best_obj_id; renderList(); renderDetail()}
q.oninput=()=>{state.page=1;loadList()}; category.onchange=()=>{state.page=1;loadList()}; sort.onchange=()=>{state.page=1;loadList()}; prev.onclick=()=>{state.page=Math.max(1,state.page-1);loadList()}; next.onclick=()=>{state.page+=1;loadList()}; document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>{document.querySelectorAll('.tabs button').forEach(x=>x.classList.remove('active'));b.classList.add('active');state.bucket=b.dataset.bucket;state.page=1;loadList()}); frame.oninput=updateImages; showMask.onchange=()=>{masksVisible=showMask.checked;updateImages()}; play.onclick=()=>{if(timer){clearInterval(timer);timer=null;play.textContent='Play';return}play.textContent='Pause';timer=setInterval(()=>{if(active){frame.value=(Number(frame.value)+1)%active.frames.length;updateImages()}},220)}; gtOnly.onclick=()=>{if(!active)return; const idx=active.gt_first_frame_idx||0; frame.value=idx; updateImages()}
loadSummary().then(loadList)
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
    handler.end_headers()
    handler.wfile.write(data)


def _html_response(handler, html):
    data = html.encode("utf-8")
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _png_response(handler, image):
    buf = BytesIO()
    image.save(buf, format="PNG")
    data = buf.getvalue()
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", "image/png")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _overlay(base, mask, color, alpha=0.52):
    base = base.convert("RGBA")
    mask = mask > 0
    if mask.shape != (base.height, base.width):
        mask = np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize((base.width, base.height), Image.NEAREST)) > 0
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    arr = np.array(layer)
    arr[mask] = (color[0], color[1], color[2], int(255 * alpha))
    return Image.alpha_composite(base, Image.fromarray(arr)).convert("RGB")


def _draw_label(img, text):
    out = img.convert("RGB")
    draw = ImageDraw.Draw(out)
    pad = 8
    bbox = draw.textbbox((0, 0), text)
    draw.rectangle((0, 0, bbox[2] + pad * 2, bbox[3] + pad * 2), fill=(255, 250, 240))
    draw.text((pad, pad), text, fill=(29, 33, 28))
    return out


def _compact(indices, frames):
    vals = sorted({int(i) for i in indices if 0 <= int(i) < len(frames)})
    if not vals:
        return "-"
    ranges = []
    start = prev = vals[0]
    for idx in vals[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            ranges.append((start, prev)); start = prev = idx
    ranges.append((start, prev))
    return ", ".join(frames[a] if a == b else f"{frames[a]}-{frames[b]}" for a, b in ranges)




def _compact_local(indices):
    vals = sorted({int(i) for i in indices})
    if not vals:
        return "-"
    ranges = []
    start = prev = vals[0]
    for idx in vals[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            ranges.append((start, prev))
            start = prev = idx
    ranges.append((start, prev))
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in ranges)

def _load_binary_png(path, hw):
    if not os.path.exists(path):
        return np.zeros(hw, dtype=np.uint8)
    arr = np.array(Image.open(path).convert("L")) > 0
    if arr.shape != hw:
        arr = np.array(Image.fromarray(arr.astype(np.uint8) * 255).resize((hw[1], hw[0]), Image.NEAREST)) > 0
    return arr.astype(np.uint8)


class State:
    def __init__(self, args):
        self.args = args
        payload = json.load(open(args.report_json, "r", encoding="utf-8"))
        self.summary = payload.get("summary", {})
        self.samples = payload.get("samples", [])
        self.by_key = {(str(r["video_id"]), str(r["exp_id"])): r for r in self.samples}
        self.meta = json.load(open(args.meta_path, "r", encoding="utf-8")).get("videos", {})
        self.categories = sorted(__import__('collections').Counter(str(r.get("category", "")) for r in self.samples).items(), key=lambda x: (-x[1], x[0]))

    def frames(self, video_id):
        frames = [str(x) for x in self.meta.get(str(video_id), {}).get("frames", [])]
        if frames:
            return frames
        image_dir = os.path.join(self.args.image_root, str(video_id))
        return sorted(os.path.splitext(x)[0] for x in os.listdir(image_dir) if x.lower().endswith((".jpg", ".png", ".jpeg")))

    def cache_dir(self, video_id, exp_id):
        return os.path.join(self.args.cache_root, self.args.split, str(video_id), str(exp_id))

    def sample_detail(self, video_id, exp_id):
        base = dict(self.by_key[(str(video_id), str(exp_id))])
        meta_path = os.path.join(self.cache_dir(video_id, exp_id), "meta.json")
        meta = json.load(open(meta_path, "r", encoding="utf-8"))
        frames = self.frames(video_id)
        expr = self.meta[str(video_id)]["expressions"][str(exp_id)]
        gt_obj_id = str(expr.get("obj_id", ""))
        iou_map = {str(k): float(v) for k, v in (meta.get("iou_map") or {}).items()}
        frame_indices = {str(k): v for k, v in (meta.get("frame_indices") or {}).items()}
        per = {}
        for obj_id in sorted(set(iou_map) | set(frame_indices), key=lambda x: int(x) if str(x).isdigit() else str(x)):
            inds = [int(x) for x in frame_indices.get(obj_id, [])]
            per[obj_id] = {"J": float(iou_map.get(obj_id, 0.0)), "present_frames": len(set(inds)), "frame_index_range": _compact_local(inds), "frame_range": _compact(inds, frames)}
        best = max(per, key=lambda k: per[k]["J"]) if per else "none"
        gt_present = []
        for i, frame in enumerate(frames):
            path = os.path.join(self.args.annotation_root, str(video_id), f"{frame}.png")
            if os.path.exists(path):
                ann = np.array(Image.open(path))
                if np.any(ann == int(gt_obj_id)):
                    gt_present.append(i)
        base.update({"frames": frames, "gt_obj_id": gt_obj_id, "best_obj_id": best, "per_candidate": per, "gt_present_frames": len(gt_present), "gt_frame_index_range": _compact_local(gt_present), "gt_frame_range": _compact(gt_present, frames), "gt_first_frame_idx": gt_present[0] if gt_present else 0})
        return base

    def image_path(self, video_id, frame):
        return os.path.join(self.args.image_root, str(video_id), f"{frame}.jpg")

    def gt_mask(self, video_id, exp_id, frame, hw):
        expr = self.meta[str(video_id)]["expressions"][str(exp_id)]
        obj_id = int(expr.get("obj_id"))
        path = os.path.join(self.args.annotation_root, str(video_id), f"{frame}.png")
        if not os.path.exists(path):
            return np.zeros(hw, dtype=np.uint8)
        ann = np.array(Image.open(path))
        if ann.shape != hw:
            ann = np.array(Image.fromarray(ann.astype(np.uint8)).resize((hw[1], hw[0]), Image.NEAREST))
        return (ann == obj_id).astype(np.uint8)

    def candidate_mask(self, video_id, exp_id, obj_id, frame_idx, frame_name, hw):
        obj_dir = os.path.join(self.cache_dir(video_id, exp_id), "masks", f"obj_{obj_id}")
        candidates = []
        if frame_name:
            candidates.append(os.path.join(obj_dir, f"{int(frame_idx):0{len(str(frame_name))}d}.png"))
        candidates.append(os.path.join(obj_dir, f"{int(frame_idx)}.png"))
        if frame_name:
            candidates.append(os.path.join(obj_dir, f"{frame_name}.png"))
        for path in candidates:
            if os.path.exists(path):
                return _load_binary_png(path, hw)
        return np.zeros(hw, dtype=np.uint8)

    def filtered(self, qs):
        q = str(qs.get("q", "")).lower().strip()
        cat = str(qs.get("category", "")).strip()
        bucket = str(qs.get("bucket", "all"))
        items = self.samples
        if cat:
            items = [r for r in items if str(r.get("category", "")) == cat]
        if q:
            items = [r for r in items if q in f"{r.get('video_id')} {r.get('exp_id')} {r.get('expression')} {r.get('sam3_prompt_text')} {r.get('category')}".lower()]
        if bucket == "zero":
            items = [r for r in items if float(r.get("max_j", 0.0)) <= 1e-12]
        elif bucket == "lt01":
            items = [r for r in items if float(r.get("max_j", 0.0)) < 0.1]
        elif bucket == "top7":
            items = [r for r in items if int(r.get("num_candidates", 0)) >= 7]
        elif bucket == "backup":
            items = [r for r in items if str(r.get("sam3_prompt_source", "")) == "backup_category"]
        sort = str(qs.get("sort", "max_j"))
        if sort == "num_candidates":
            items = sorted(items, key=lambda r: (-int(r.get("num_candidates", 0)), float(r.get("max_j", 0.0))))
        elif sort == "video_id":
            items = sorted(items, key=lambda r: (str(r.get("video_id")), int(r.get("exp_id", 0))))
        else:
            items = sorted(items, key=lambda r: (float(r.get("max_j", 0.0)), str(r.get("video_id")), int(r.get("exp_id", 0))))
        return items


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            sys.stderr.write("[low-recall-viewer] " + fmt % args + "\n")
        def do_GET(self):
            parsed = urlparse(self.path); qs = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            try:
                if parsed.path == "/": return _html_response(self, HTML)
                if parsed.path == "/api/summary":
                    s = dict(state.summary); s["categories"] = state.categories; return _json_response(self, s)
                if parsed.path == "/api/samples":
                    items = state.filtered(qs); page = max(1, int(qs.get("page", "1"))); page_size = min(200, max(10, int(qs.get("page_size", "50"))))
                    start = (page - 1) * page_size; slim = items[start:start + page_size]
                    return _json_response(self, {"total": len(items), "page": page, "page_size": page_size, "items": slim})
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
                img = _overlay(base, state.gt_mask(video_id, exp_id, frame, hw), COLORS["gt"]) if show else base
                return _png_response(self, _draw_label(img, "GT" if show else "GT hidden"))
            if view == "candidate":
                obj_id = qs.get("obj_id", sample["best_obj_id"])
                img = _overlay(base, state.candidate_mask(video_id, exp_id, obj_id, idx, frame, hw), COLORS["candidate"]) if show else base
                return _png_response(self, _draw_label(img, f"Candidate obj_{obj_id}" if show else f"Candidate obj_{obj_id} hidden"))
            return _json_response(self, {"error": f"unknown view {view}"}, HTTPStatus.BAD_REQUEST)
    return Handler


def main():
    p = argparse.ArgumentParser(description="Ref-Youtube-VOS train low-recall offline-cache viewer")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8793)
    p.add_argument("--report-json", default="/mnt/csip-500/share/kongjiawei/Sa2VA-main/crf_sam3_attn/cache_reports/refytros_train_dms_topk7_20260618_max_j_lt_0p5.json")
    p.add_argument("--cache-root", default="/mnt/csip-500/share/kongjiawei/Sa2VA-main/crf_sam3_attn/offline_cache_refytros_dms_topk7_20260618")
    p.add_argument("--split", default="train")
    p.add_argument("--meta-path", default="/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/Ref-Youtube-VOS/train/meta_expressions.json")
    p.add_argument("--image-root", default="/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/Ref-Youtube-VOS/train/JPEGImages")
    p.add_argument("--annotation-root", default="/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/Ref-Youtube-VOS/train/Annotations")
    args = p.parse_args()
    state = State(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print(f"Serving low-recall viewer at http://{args.host}:{args.port}")
    print(f"Loaded {len(state.samples)} samples from {args.report_json}")
    server.serve_forever()

if __name__ == "__main__":
    main()
