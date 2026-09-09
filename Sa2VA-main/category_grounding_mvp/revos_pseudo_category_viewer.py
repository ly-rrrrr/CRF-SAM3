#!/usr/bin/env python3
"""Viewer for ReVOS mask-crop pseudo category labels."""

from __future__ import annotations

import argparse
import io
import json
import math
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

HTML = r"""
<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>ReVOS Pseudo Category Viewer</title>
<style>
:root { --bg:#f5f7fb; --panel:#fff; --line:#d8dee9; --text:#111827; --muted:#667085; --blue:#175cd3; --bad:#b42318; --ok:#167a3a; }
* { box-sizing:border-box; }
body { margin:0; font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:var(--text); background:var(--bg); }
header { position:sticky; top:0; z-index:10; padding:12px 16px; background:rgba(255,255,255,.97); border-bottom:1px solid var(--line); }
h1 { font-size:18px; margin:0 0 8px; }
.summary { display:flex; flex-wrap:wrap; gap:10px; color:var(--muted); font-size:13px; }
.summary b { color:var(--text); }
.toolbar { display:grid; grid-template-columns: minmax(220px,1fr) 190px 130px 120px auto auto; gap:8px; margin-top:10px; }
input,select,button { height:34px; border:1px solid var(--line); border-radius:6px; padding:0 10px; background:#fff; font:inherit; font-size:13px; }
button { cursor:pointer; font-weight:600; }
main { display:grid; grid-template-columns:420px minmax(0,1fr); gap:12px; padding:12px; }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:8px; overflow:hidden; min-height:calc(100vh - 128px); }
.list-head { padding:10px 12px; display:flex; justify-content:space-between; color:var(--muted); font-size:13px; border-bottom:1px solid var(--line); }
.items { height:calc(100vh - 184px); overflow:auto; }
.item { display:block; width:100%; border:0; border-bottom:1px solid #edf0f5; background:#fff; text-align:left; border-radius:0; height:auto; min-height:94px; padding:10px 12px; }
.item:hover { background:#f8fbff; }
.item.active { background:#eef5ff; box-shadow:inset 3px 0 0 var(--blue); }
.row { display:flex; gap:8px; align-items:center; justify-content:space-between; }
.id { font-weight:700; font-size:13px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.badge { font-size:11px; padding:2px 7px; border:1px solid var(--line); border-radius:999px; background:#fff; color:#344054; white-space:nowrap; }
.badge.ok { color:var(--ok); border-color:#abefc6; background:#ecfdf3; }
.badge.bad { color:var(--bad); border-color:#fecdca; background:#fef3f2; }
.expr { margin-top:6px; color:#344054; font-size:12px; line-height:1.35; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; }
.pred { margin-top:6px; color:var(--muted); font-size:12px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.detail { padding:14px; overflow:auto; }
.empty { color:var(--muted); padding:18px; }
.title { display:flex; justify-content:space-between; align-items:flex-start; gap:12px; padding-bottom:12px; border-bottom:1px solid var(--line); margin-bottom:12px; }
.title h2 { margin:0; font-size:18px; }
.title p { margin:6px 0 0; line-height:1.45; color:#344054; }
.metrics { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin-bottom:12px; }
.metric { border:1px solid var(--line); border-radius:8px; background:#fff; padding:10px; }
.metric .label { font-size:12px; color:var(--muted); margin-bottom:4px; }
.metric .value { font-size:16px; font-weight:750; overflow-wrap:anywhere; }
.images { display:grid; grid-template-columns:repeat(2,minmax(260px,1fr)); gap:10px; margin-bottom:12px; }
figure { margin:0; border:1px solid var(--line); border-radius:8px; background:#fff; overflow:hidden; }
figure img { width:100%; display:block; object-fit:contain; aspect-ratio:16/10; background:#111827; }
figcaption { padding:8px 9px; font-size:12px; color:#475467; border-top:1px solid var(--line); }
table { width:100%; border-collapse:collapse; background:#fff; border:1px solid var(--line); border-radius:8px; overflow:hidden; font-size:12px; }
th,td { padding:8px; border-bottom:1px solid #edf0f5; text-align:left; vertical-align:top; }
th { background:#f8fafc; color:#475467; font-weight:650; }
pre { white-space:pre-wrap; word-break:break-word; background:#f8fafc; border:1px solid var(--line); border-radius:8px; padding:10px; font-size:12px; line-height:1.45; }
@media (max-width:1050px) { main { grid-template-columns:1fr; } .toolbar { grid-template-columns:1fr 1fr; } .items { height:360px; } .metrics { grid-template-columns:repeat(2,1fr); } .images { grid-template-columns:1fr; } }
</style>
</head>
<body>
<header>
  <h1>ReVOS Pseudo Category Viewer</h1>
  <div id="summary" class="summary">Loading...</div>
  <div class="toolbar">
    <input id="q" placeholder="搜索 video / exp / 表达式 / 类别" />
    <select id="category"><option value="">全部类别</option></select>
    <select id="status"><option value="all">全部状态</option><option value="ok">ok</option><option value="skipped">skipped</option></select>
    <select id="pageSize"><option>25</option><option selected>50</option><option>100</option></select>
    <button id="prev">上一页</button><button id="next">下一页</button>
  </div>
</header>
<main>
  <section class="panel"><div class="list-head"><span id="count">0</span><span id="page">page</span></div><div id="items" class="items"></div></section>
  <section id="detail" class="panel detail"><div class="empty">选择左侧样本查看 mask 和 crop。</div></section>
</main>
<script>
let state={page:1,pageSize:50,total:0,records:[],selected:null};
const $=id=>document.getElementById(id);
function esc(s){return String(s??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));}
function media(id,type,i=0){return `/media?id=${encodeURIComponent(id)}&type=${type}&frame=${i}&t=${Date.now()}`;}
async function meta(){const r=await fetch('/api/meta'); const m=await r.json(); $('summary').innerHTML=`records <b>${m.records}</b> · ok <b>${m.ok}</b> · skipped <b>${m.skipped}</b> · categories <b>${m.categories.length}</b> · source <b>${esc(m.jsonl_name)}</b>`; const c=$('category'); c.innerHTML='<option value="">全部类别</option>'; for(const x of m.categories.slice(0,500)){let o=document.createElement('option'); o.value=x; o.textContent=x; c.appendChild(o);} }
async function load(first=false){const p=new URLSearchParams({page:state.page,page_size:$('pageSize').value,q:$('q').value.trim(),category:$('category').value,status:$('status').value}); const r=await fetch('/api/records?'+p); const d=await r.json(); state.records=d.records; state.total=d.total; state.page=d.page; state.pageSize=d.page_size; renderList(); if(first&&state.records.length) selectRec(state.records[0].sample_id);}
function renderList(){const pages=Math.max(1,Math.ceil(state.total/state.pageSize)); $('count').textContent=`${state.total} records`; $('page').textContent=`${state.page}/${pages}`; $('items').innerHTML=state.records.map(r=>`<button class="item ${state.selected===r.sample_id?'active':''}" onclick="selectRec('${esc(r.sample_id)}')"><div class="row"><span class="id">${esc(r.video_id)}/${esc(r.exp_id)}</span><span class="badge ${r.status==='ok'?'ok':'bad'}">${esc(r.status)}</span></div><div class="expr">${esc(r.expression)}</div><div class="pred">${esc(r.pseudo_category||r.skip_reason||'-')} · score ${fmt(r.score)} · margin ${fmt(r.margin_top1_top2)}</div></button>`).join('')||'<div class="empty">没有记录。</div>';}
function fmt(x){return x===null||x===undefined?'-':Number(x).toFixed(4);}
async function selectRec(id){state.selected=id; renderList(); const r=await (await fetch('/api/record?id='+encodeURIComponent(id))).json(); const top=(r.top_categories||[]).map((x,i)=>`<tr><td>${i+1}</td><td><b>${esc(x.category_name||'')}</b><br><span style="color:#667085">retrieval: ${esc(x.display_prompt_text||x.variant_text||'')} · sam3: ${esc(x.sam3_prompt_text||x.category_prompt||'')}</span></td><td>${fmt(x.score)}</td><td>${esc((x.parent_names||[]).join(', '))}</td><td>${esc((x.prompt_candidates||[]).join(' / '))}</td></tr>`).join(''); const imgs=(r.selected_frames||[]).map((f,i)=>`<div class="images"><figure><img src="${media(r.sample_id,'overlay',i)}"/><figcaption>mask overlay · frame ${esc(f)} · idx ${esc((r.selected_frame_indices||[])[i])}</figcaption></figure><figure><img src="${media(r.sample_id,'crop',i)}"/><figcaption>bbox crop · area ${esc((r.selected_mask_areas||[])[i])} · quality ${fmt((r.selected_visibility_scores||[])[i])}</figcaption></figure></div>`).join(''); $('detail').innerHTML=`<div class="title"><div><h2>${esc(r.video_id)}/${esc(r.exp_id)}</h2><p>${esc(r.expression)}</p></div><span class="badge ${r.status==='ok'?'ok':'bad'}">${esc(r.status)}</span></div><div class="metrics"><div class="metric"><div class="label">OpenImages 600 pseudo label</div><div class="value">${esc(r.pseudo_category||'-')}</div></div><div class="metric"><div class="label">retrieval text</div><div class="value">${esc(r.pseudo_category_text||r.pseudo_variant||'-')}</div></div><div class="metric"><div class="label">SAM3 prompt separate</div><div class="value">${esc(r.pseudo_sam3_prompt||'-')}</div></div><div class="metric"><div class="label">score</div><div class="value">${fmt(r.score)}</div></div><div class="metric"><div class="label">margin</div><div class="value">${fmt(r.margin_top1_top2)}</div></div></div>${imgs||'<div class="empty">没有可显示帧。</div>'}<h3>Top categories</h3><table><thead><tr><th>#</th><th>category</th><th>score</th><th>parents</th><th>prompt candidates</th></tr></thead><tbody>${top}</tbody></table><h3>Raw record</h3><pre>${esc(JSON.stringify(r,null,2))}</pre>`;}
let timer; for(const id of ['category','status','pageSize']) $(id).addEventListener('change',()=>{state.page=1;load(true)}); $('q').addEventListener('input',()=>{clearTimeout(timer);timer=setTimeout(()=>{state.page=1;load(true)},250)}); $('prev').onclick=()=>{if(state.page>1){state.page--;load(true)}}; $('next').onclick=()=>{if(state.page*state.pageSize<state.total){state.page++;load(true)}}; meta().then(()=>load(true));
</script>
</body>
</html>
"""


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def decode_rle(rle: Dict[str, Any] | None) -> np.ndarray | None:
    if not rle:
        return None
    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = decoded.any(axis=2)
    return decoded.astype(bool)


def merge_mask(mask_dict: Dict[str, Any], anno_ids: Sequence[int], frame_idx: int) -> np.ndarray | None:
    merged = None
    for anno_id in anno_ids:
        seq = mask_dict.get(str(anno_id))
        if seq is None:
            seq = mask_dict.get(anno_id)
        if seq is None or frame_idx >= len(seq):
            continue
        m = decode_rle(seq[frame_idx])
        if m is None:
            continue
        merged = m if merged is None else np.logical_or(merged, m)
    return merged


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def frame_path(root: Path, video_id: str, frame_name: str) -> Path:
    video_dir = root / video_id
    for suffix in (".jpg", ".jpeg", ".png"):
        path = video_dir / f"{frame_name}{suffix}"
        if path.exists():
            return path
    return video_dir / f"{frame_name}.jpg"


def overlay_image(image: Image.Image, mask: np.ndarray) -> Image.Image:
    rgb = image.convert("RGB")
    arr = np.asarray(rgb).copy().astype(np.float32)
    color = np.array([255, 48, 48], dtype=np.float32)
    arr[mask] = arr[mask] * 0.45 + color * 0.55
    bbox = mask_bbox(mask)
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        arr[max(0, y0 - 2):min(arr.shape[0], y0 + 2), x0:x1] = [255, 255, 0]
        arr[max(0, y1 - 2):min(arr.shape[0], y1 + 2), x0:x1] = [255, 255, 0]
        arr[y0:y1, max(0, x0 - 2):min(arr.shape[1], x0 + 2)] = [255, 255, 0]
        arr[y0:y1, max(0, x1 - 2):min(arr.shape[1], x1 + 2)] = [255, 255, 0]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def crop_image(image: Image.Image, mask: np.ndarray, padding_ratio: float) -> Image.Image:
    rgb = image.convert("RGB")
    bbox = mask_bbox(mask)
    if bbox is None:
        return rgb
    w, h = rgb.size
    x0, y0, x1, y1 = bbox
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    pad = int(round(max(bw, bh) * padding_ratio))
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w, x1 + pad)
    y1 = min(h, y1 + pad)
    return rgb.crop((x0, y0, x1, y1))


class Store:
    def __init__(self, jsonl: Path, mask_dict: Path, image_root: Path, padding_ratio: float):
        self.jsonl = jsonl
        self.mask_dict = read_json(mask_dict)
        self.image_root = image_root
        self.padding_ratio = padding_ratio

    def iter_records(self):
        if not self.jsonl.exists():
            return
        with self.jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue

    def meta(self) -> Dict[str, Any]:
        total = ok = skipped = 0
        cats = set()
        for r in self.iter_records() or []:
            total += 1
            if r.get("status") == "ok":
                ok += 1
                if r.get("pseudo_category"):
                    cats.add(str(r.get("pseudo_category")))
            else:
                skipped += 1
        return {"records": total, "ok": ok, "skipped": skipped, "categories": sorted(cats), "jsonl_name": self.jsonl.name}

    def query(self, params: Dict[str, List[str]]) -> Dict[str, Any]:
        page = max(1, int((params.get("page") or ["1"])[0] or 1))
        page_size = max(1, min(200, int((params.get("page_size") or ["50"])[0] or 50)))
        q = ((params.get("q") or [""])[0] or "").lower().strip()
        category = (params.get("category") or [""])[0]
        status = (params.get("status") or ["all"])[0]
        rows = []
        for r in self.iter_records() or []:
            if status != "all" and r.get("status") != status:
                continue
            if category and r.get("pseudo_category") != category:
                continue
            if q:
                hay = " ".join(str(r.get(k, "")) for k in ["sample_id", "video_id", "exp_id", "expression", "pseudo_category", "pseudo_variant"]).lower()
                if q not in hay:
                    continue
            rows.append(self.slim(r))
        total = len(rows)
        start = (page - 1) * page_size
        return {"total": total, "page": page, "page_size": page_size, "records": rows[start:start + page_size]}

    @staticmethod
    def slim(r: Dict[str, Any]) -> Dict[str, Any]:
        return {k: r.get(k) for k in ["sample_id", "video_id", "exp_id", "expression", "status", "skip_reason", "pseudo_category", "pseudo_variant", "score", "margin_top1_top2"]}

    def get(self, sample_id: str) -> Dict[str, Any] | None:
        for r in self.iter_records() or []:
            if r.get("sample_id") == sample_id:
                return r
        return None

    def render_media(self, sample_id: str, kind: str, frame_slot: int) -> bytes:
        rec = self.get(sample_id)
        if not rec:
            raise FileNotFoundError(sample_id)
        frame_indices = rec.get("selected_frame_indices") or []
        frame_names = rec.get("selected_frames") or []
        if frame_slot < 0 or frame_slot >= len(frame_indices):
            raise FileNotFoundError(str(frame_slot))
        frame_idx = int(frame_indices[frame_slot])
        frame_name = str(frame_names[frame_slot]) if frame_slot < len(frame_names) else f"{frame_idx:05d}"
        mask = merge_mask(self.mask_dict, [int(x) for x in rec.get("anno_ids", [])], frame_idx)
        if mask is None:
            raise FileNotFoundError("mask")
        path = frame_path(self.image_root, str(rec.get("video_id")), frame_name)
        img = Image.open(path)
        out = crop_image(img, mask, self.padding_ratio) if kind == "crop" else overlay_image(img, mask)
        bio = io.BytesIO()
        out.save(bio, format="JPEG", quality=92)
        return bio.getvalue()


class Handler(BaseHTTPRequestHandler):
    store: Store

    def send_json(self, payload: Any):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        try:
            if parsed.path == "/":
                data = HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif parsed.path == "/api/meta":
                self.send_json(self.store.meta())
            elif parsed.path == "/api/records":
                self.send_json(self.store.query(params))
            elif parsed.path == "/api/record":
                rec = self.store.get((params.get("id") or [""])[0])
                self.send_json(rec or {"error": "not found"})
            elif parsed.path == "/media":
                data = self.store.render_media((params.get("id") or [""])[0], (params.get("type") or ["overlay"])[0], int((params.get("frame") or ["0"])[0]))
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_error(404)
        except Exception as exc:
            self.send_error(500, str(exc))

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="category_grounding_mvp/data/revos_siglip2_openimages_hierarchy_bestview_20260703/pseudo_categories.jsonl")
    ap.add_argument("--mask-dict", default="/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/revos/mask_dict.json")
    ap.add_argument("--image-root", default="/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/revos")
    ap.add_argument("--padding-ratio", type=float, default=0.15)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8803)
    args = ap.parse_args()
    Handler.store = Store(Path(args.jsonl), Path(args.mask_dict), Path(args.image_root), args.padding_ratio)
    print(json.dumps({"event": "start", "url": f"http://{args.host}:{args.port}", "jsonl": args.jsonl}, ensure_ascii=False), flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
