#!/usr/bin/env python3
"""Lightweight web viewer for category-grounding eval comparisons."""

from __future__ import annotations

import argparse
import html
import json
import mimetypes
import os
import posixpath
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List

HTML_PAGE = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Category Grounding Eval Viewer</title>
  <style>
    :root { color-scheme: light; --border:#d7dde8; --muted:#667085; --bg:#f6f8fb; --panel:#fff; --text:#111827; --ok:#167a3a; --bad:#b42318; --blue:#175cd3; }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--text); }
    header { position: sticky; top: 0; z-index: 5; background: rgba(255,255,255,.96); border-bottom: 1px solid var(--border); padding: 12px 18px; }
    h1 { margin: 0 0 8px; font-size: 18px; font-weight: 650; }
    .summary { display: flex; flex-wrap: wrap; gap: 8px; color: var(--muted); font-size: 13px; }
    .summary b { color: var(--text); }
    .toolbar { display: grid; grid-template-columns: 1.3fr 160px 170px 140px auto auto; gap: 8px; align-items: center; margin-top: 12px; }
    input, select, button { height: 34px; border: 1px solid var(--border); border-radius: 6px; background: #fff; color: var(--text); padding: 0 10px; font: inherit; font-size: 13px; }
    button { cursor: pointer; font-weight: 560; }
    main { display: grid; grid-template-columns: 420px minmax(0, 1fr); gap: 12px; padding: 12px; }
    .list, .detail { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; min-height: calc(100vh - 128px); overflow: hidden; }
    .list-head { display: flex; align-items: center; justify-content: space-between; padding: 10px 12px; border-bottom: 1px solid var(--border); font-size: 13px; color: var(--muted); }
    .items { height: calc(100vh - 184px); overflow: auto; }
    .item { display: block; width: 100%; text-align: left; border: 0; border-bottom: 1px solid #edf0f5; background: #fff; padding: 10px 12px; height: auto; min-height: 86px; border-radius: 0; }
    .item:hover { background: #f8fbff; }
    .item.active { background: #eef5ff; box-shadow: inset 3px 0 0 var(--blue); }
    .row1 { display:flex; gap:8px; align-items:center; justify-content:space-between; font-size:13px; }
    .id { font-weight:650; color:#182230; white-space:nowrap; }
    .tag { font-size:11px; padding:2px 6px; border-radius:999px; border:1px solid var(--border); color:#344054; background:#fff; white-space:nowrap; }
    .tag.ok { color:var(--ok); border-color:#abefc6; background:#ecfdf3; }
    .tag.bad { color:var(--bad); border-color:#fecdca; background:#fef3f2; }
    .expr { color:#344054; font-size:12px; margin-top:6px; line-height:1.35; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; }
    .pred { color:#667085; font-size:12px; margin-top:5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .detail { overflow:auto; padding: 14px; }
    .empty { color: var(--muted); padding: 18px; }
    .detail-title { display:flex; justify-content:space-between; gap:12px; align-items:flex-start; border-bottom:1px solid var(--border); padding-bottom:12px; margin-bottom:12px; }
    .detail-title h2 { margin:0; font-size:18px; }
    .detail-title p { margin:6px 0 0; color:#344054; line-height:1.45; }
    .grid { display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap:10px; margin: 12px 0; }
    .metric { border:1px solid var(--border); border-radius:8px; padding:10px; background:#fff; }
    .metric .label { font-size:12px; color:var(--muted); margin-bottom:4px; }
    .metric .value { font-size:18px; font-weight:700; overflow-wrap:anywhere; }
    .metric.ok .value { color:var(--ok); }
    .metric.bad .value { color:var(--bad); }
    .frames { display:grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap:10px; }
    figure { margin:0; border:1px solid var(--border); border-radius:8px; background:#fff; overflow:hidden; }
    figure img { display:block; width:100%; aspect-ratio: 16/9; object-fit: contain; background:#111827; }
    figcaption { padding:7px 9px; font-size:12px; color:#475467; border-top:1px solid var(--border); word-break:break-all; }
    pre { white-space:pre-wrap; word-break:break-word; background:#f8fafc; border:1px solid var(--border); padding:10px; border-radius:8px; font-size:12px; line-height:1.45; }
    @media (max-width: 980px) { main { grid-template-columns: 1fr; } .toolbar { grid-template-columns: 1fr 1fr; } .items { height: 360px; } }
  </style>
</head>
<body>
  <header>
    <h1>Category Grounding Eval Viewer</h1>
    <div class="summary" id="summary">Loading...</div>
    <div class="toolbar">
      <input id="q" placeholder="Search video / expression / category" />
      <select id="status">
        <option value="all">All</option>
        <option value="base_wrong_sft_right">Base wrong, SFT right</option>
        <option value="sft_wrong_base_right">SFT wrong, Base right</option>
        <option value="both_wrong">Both wrong</option>
        <option value="both_right">Both right</option>
        <option value="sft_wrong">SFT wrong</option>
      </select>
      <select id="category"><option value="">All categories</option></select>
      <select id="pageSize"><option>25</option><option selected>50</option><option>100</option></select>
      <button id="prev">Prev</button>
      <button id="next">Next</button>
    </div>
  </header>
  <main>
    <section class="list">
      <div class="list-head"><span id="count">0 records</span><span id="page">page 1</span></div>
      <div class="items" id="items"></div>
    </section>
    <section class="detail" id="detail"><div class="empty">Select a record.</div></section>
  </main>
<script>
let state = { page: 1, pageSize: 50, total: 0, records: [], selected: null };
const $ = (id) => document.getElementById(id);
function esc(s){ return String(s ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c])); }
function imageUrl(path){ return '/image?path=' + encodeURIComponent(path); }
async function loadMeta(){
  const res = await fetch('/api/meta');
  const meta = await res.json();
  $('summary').innerHTML = `Base <b>${meta.base_exact}/${meta.total} (${(meta.base_rate*100).toFixed(2)}%)</b> · SFT <b>${meta.sft_exact}/${meta.total} (${(meta.sft_rate*100).toFixed(2)}%)</b> · fixed <b>${meta.base_wrong_sft_right}</b> · regressed <b>${meta.sft_wrong_base_right}</b> · both wrong <b>${meta.both_wrong}</b> · video overlap <b>${meta.video_overlap}</b>`;
  const cat = $('category');
  for (const c of meta.categories) { const opt = document.createElement('option'); opt.value = c; opt.textContent = c; cat.appendChild(opt); }
}
async function loadRecords(selectFirst=false){
  const params = new URLSearchParams({
    page: state.page,
    page_size: $('pageSize').value,
    status: $('status').value,
    category: $('category').value,
    q: $('q').value.trim(),
  });
  const res = await fetch('/api/records?' + params);
  const data = await res.json();
  state.records = data.records; state.total = data.total; state.page = data.page; state.pageSize = data.page_size;
  renderList();
  if (selectFirst && state.records.length) selectRecord(state.records[0].id);
}
function statusTag(r){
  if (r.base_exact && r.sft_exact) return ['both right','ok'];
  if (!r.base_exact && r.sft_exact) return ['fixed','ok'];
  if (r.base_exact && !r.sft_exact) return ['regressed','bad'];
  return ['both wrong','bad'];
}
function renderList(){
  $('count').textContent = `${state.total} records`;
  const pages = Math.max(1, Math.ceil(state.total / state.pageSize));
  $('page').textContent = `page ${state.page} / ${pages}`;
  $('items').innerHTML = state.records.map(r => {
    const [label, cls] = statusTag(r);
    return `<button class="item ${state.selected===r.id?'active':''}" onclick="selectRecord('${esc(r.id)}')">
      <div class="row1"><span class="id">${esc(r.video_id)}/${esc(r.exp_id)}</span><span class="tag ${cls}">${label}</span></div>
      <div class="expr">${esc(r.expression)}</div>
      <div class="pred">GT ${esc(r.gt)} · Base ${esc(r.base_pred)} · SFT ${esc(r.sft_pred)}</div>
    </button>`;
  }).join('') || '<div class="empty">No records.</div>';
}
async function selectRecord(id){
  state.selected = id; renderList();
  const res = await fetch('/api/record?id=' + encodeURIComponent(id));
  const r = await res.json();
  const frames = (r.frame_paths || []).map((p, i) => `<figure><img src="${imageUrl(p)}" loading="lazy"/><figcaption>${i+1}. ${esc(p.split('/').slice(-2).join('/'))}</figcaption></figure>`).join('');
  $('detail').innerHTML = `<div class="detail-title"><div><h2>${esc(r.video_id)}/${esc(r.exp_id)}</h2><p>${esc(r.expression)}</p></div><span class="tag ${r.sft_exact?'ok':'bad'}">SFT ${r.sft_exact?'right':'wrong'}</span></div>
    <div class="grid">
      <div class="metric"><div class="label">GT category</div><div class="value">${esc(r.gt)}</div></div>
      <div class="metric ${r.base_exact?'ok':'bad'}"><div class="label">Base prediction</div><div class="value">${esc(r.base_pred)}</div></div>
      <div class="metric ${r.sft_exact?'ok':'bad'}"><div class="label">SFT prediction</div><div class="value">${esc(r.sft_pred)}</div></div>
    </div>
    <div class="frames">${frames}</div>
    <h3>Raw response</h3><pre>Base: ${esc(r.base_raw)}\n\nSFT: ${esc(r.sft_raw)}</pre>`;
}
let debounce;
for (const id of ['status','category','pageSize']) $(id).addEventListener('change', () => { state.page=1; loadRecords(true); });
$('q').addEventListener('input', () => { clearTimeout(debounce); debounce=setTimeout(()=>{ state.page=1; loadRecords(true); }, 250); });
$('prev').addEventListener('click', () => { if(state.page>1){ state.page--; loadRecords(true); } });
$('next').addEventListener('click', () => { if(state.page*state.pageSize<state.total){ state.page++; loadRecords(true); } });
loadMeta().then(()=>loadRecords(true));
</script>
</body>
</html>
"""


def norm(text: Any) -> str:
    return str(text or "").strip().lower().replace("_", " ")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class Store:
    def __init__(self, base_path: Path, sft_path: Path, train_jsonl: Path, val_jsonl: Path):
        self.base = load_json(base_path)
        self.sft = load_json(sft_path)
        base_map = {(r.get("video_id"), str(r.get("exp_id"))): r for r in self.base.get("records", [])}
        records: List[Dict[str, Any]] = []
        for s in self.sft.get("records", []):
            key = (s.get("video_id"), str(s.get("exp_id")))
            b = base_map.get(key, {})
            rec = {
                "id": f"{key[0]}::{key[1]}",
                "video_id": key[0],
                "exp_id": key[1],
                "expression": s.get("expression", ""),
                "gt": s.get("category", ""),
                "base_pred": b.get("pred_category", ""),
                "sft_pred": s.get("pred_category", ""),
                "base_exact": bool(b.get("category_exact")),
                "sft_exact": bool(s.get("category_exact")),
                "base_raw": b.get("raw_response", ""),
                "sft_raw": s.get("raw_response", ""),
                "frame_paths": s.get("frame_paths", []),
            }
            records.append(rec)
        self.records = records
        self.record_map = {r["id"]: r for r in records}
        self.categories = sorted({str(r["gt"]) for r in records if r.get("gt")})
        train_videos = self._videos_from_jsonl(train_jsonl)
        val_videos = self._videos_from_jsonl(val_jsonl)
        self.video_overlap = len(train_videos & val_videos)

    @staticmethod
    def _videos_from_jsonl(path: Path) -> set[str]:
        out = set()
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    out.add(str(json.loads(line).get("video_id", "")))
        return out

    def meta(self) -> Dict[str, Any]:
        total = len(self.records)
        base_exact = sum(r["base_exact"] for r in self.records)
        sft_exact = sum(r["sft_exact"] for r in self.records)
        return {
            "total": total,
            "base_exact": base_exact,
            "base_rate": base_exact / total if total else 0,
            "sft_exact": sft_exact,
            "sft_rate": sft_exact / total if total else 0,
            "base_wrong_sft_right": sum((not r["base_exact"]) and r["sft_exact"] for r in self.records),
            "sft_wrong_base_right": sum(r["base_exact"] and (not r["sft_exact"]) for r in self.records),
            "both_wrong": sum((not r["base_exact"]) and (not r["sft_exact"]) for r in self.records),
            "video_overlap": self.video_overlap,
            "categories": self.categories,
        }

    def query(self, params: Dict[str, List[str]]) -> Dict[str, Any]:
        status = params.get("status", ["all"])[0]
        category = params.get("category", [""])[0]
        q = params.get("q", [""])[0].strip().lower()
        page = max(1, int(params.get("page", ["1"])[0] or 1))
        page_size = max(1, min(200, int(params.get("page_size", ["50"])[0] or 50)))
        rows = self.records
        if status == "base_wrong_sft_right":
            rows = [r for r in rows if (not r["base_exact"]) and r["sft_exact"]]
        elif status == "sft_wrong_base_right":
            rows = [r for r in rows if r["base_exact"] and (not r["sft_exact"])]
        elif status == "both_wrong":
            rows = [r for r in rows if (not r["base_exact"]) and (not r["sft_exact"])]
        elif status == "both_right":
            rows = [r for r in rows if r["base_exact"] and r["sft_exact"]]
        elif status == "sft_wrong":
            rows = [r for r in rows if not r["sft_exact"]]
        if category:
            rows = [r for r in rows if r.get("gt") == category]
        if q:
            rows = [r for r in rows if q in " ".join([str(r.get("video_id")), str(r.get("exp_id")), str(r.get("expression")), str(r.get("gt")), str(r.get("base_pred")), str(r.get("sft_pred"))]).lower()]
        total = len(rows)
        start = (page - 1) * page_size
        return {"total": total, "page": page, "page_size": page_size, "records": rows[start : start + page_size]}


class Handler(BaseHTTPRequestHandler):
    store: Store

    def send_json(self, payload: Any) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        if parsed.path == "/":
            data = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if parsed.path == "/api/meta":
            self.send_json(self.store.meta())
            return
        if parsed.path == "/api/records":
            self.send_json(self.store.query(params))
            return
        if parsed.path == "/api/record":
            rec_id = params.get("id", [""])[0]
            rec = self.store.record_map.get(rec_id)
            if rec is None:
                self.send_error(404, "record not found")
            else:
                self.send_json(rec)
            return
        if parsed.path == "/image":
            raw = params.get("path", [""])[0]
            path = Path(raw)
            if not path.exists() or not path.is_file():
                self.send_error(404, "image not found")
                return
            ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_error(404, "not found")

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {self.address_string()} {fmt % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-json", default="category_grounding_mvp/outputs/eval/refytvos_val_category_base_qwen3vl.json")
    parser.add_argument("--sft-json", default="category_grounding_mvp/outputs/eval/refytvos_val_category_lora_stage1_final.json")
    parser.add_argument("--train-jsonl", default="category_grounding_mvp/data/refytvos_category_sft/train.jsonl")
    parser.add_argument("--val-jsonl", default="category_grounding_mvp/data/refytvos_category_sft/val.jsonl")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8799)
    args = parser.parse_args()

    Handler.store = Store(Path(args.base_json), Path(args.sft_json), Path(args.train_jsonl), Path(args.val_jsonl))
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[viewer] http://{args.host}:{args.port} records={len(Handler.store.records)}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
