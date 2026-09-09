#!/usr/bin/env python3
"""Small viewer for CLIP/SigLIP category retrieval diagnostics."""

from __future__ import annotations

import argparse
import json
import mimetypes
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List

HTML = r"""
<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Category Retrieval Score Viewer</title>
<style>
:root{--bg:#f6f8fb;--panel:#fff;--line:#d7dde8;--text:#111827;--muted:#667085;--blue:#175cd3;--bad:#b42318;--ok:#167a3a;}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;}
header{position:sticky;top:0;z-index:5;background:rgba(255,255,255,.96);border-bottom:1px solid var(--line);padding:12px 16px;} h1{font-size:18px;margin:0 0 6px}.sub{font-size:13px;color:var(--muted)}
main{padding:12px;display:grid;gap:14px}.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;overflow:hidden}.head{padding:12px 14px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:12px}.head h2{font-size:17px;margin:0}.expr{margin-top:5px;color:#344054;font-size:13px}.meta{color:var(--muted);font-size:12px;white-space:nowrap}.body{padding:12px 14px;display:grid;grid-template-columns:360px minmax(0,1fr);gap:14px}.imgs{display:grid;gap:10px}figure{margin:0;border:1px solid var(--line);border-radius:8px;overflow:hidden;background:#fff}img{display:block;width:100%;aspect-ratio:16/10;object-fit:contain;background:#111827}figcaption{font-size:12px;color:#475467;border-top:1px solid var(--line);padding:7px 9px}
table{width:100%;border-collapse:collapse;font-size:12px;background:#fff;border:1px solid var(--line);margin-bottom:12px}th,td{padding:7px 8px;border-bottom:1px solid #edf0f5;text-align:left;vertical-align:top}th{background:#f8fafc;color:#475467;font-weight:650}.rank{font-weight:700}.rank.bad{color:var(--bad)}.rank.ok{color:var(--ok)}.score{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.top{color:#344054}.method{font-weight:750;color:#182230;margin:2px 0 6px}.pill{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:2px 7px;background:#fff;color:#344054;font-size:11px;margin-left:6px}
@media(max-width:960px){.body{grid-template-columns:1fr}.meta{white-space:normal}}
</style>
</head>
<body>
<header><h1>类别词检索诊断分数</h1><div class="sub">显示 mask overlay、bbox crop、各检索方式的 top-k、关注类别排名和分数。</div></header>
<main id="root"></main>
<script>
function esc(s){return String(s??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));}
function fmt(x){return x===null||x===undefined?'-':Number(x).toFixed(6)}
function img(p){return '/image?path='+encodeURIComponent(p)}
function expected(sample){if(sample.includes('CiPki3XuVI'))return 'Person'; if(sample.includes('4539b907'))return 'Dog'; if(sample.includes('91b2f707'))return 'Motorcycle'; return ''}
function correctRank(method, name){return (method.watch_ranks||[]).find(x=>x.category_name===name)}
function methodTable(r, key, label){const m=r.methods[key]; const exp=expected(r.sample); const cr=correctRank(m, exp); const top=(m.top||[]).slice(0,10).map((x,i)=>`<tr><td>${i+1}</td><td class="top">${esc(x.category_name)}</td><td class="score">${fmt(x.score)}</td></tr>`).join(''); const cls=cr&&cr.rank<=10?'ok':'bad'; return `<div class="method">${esc(label)} <span class="pill">${esc(exp)} rank: <span class="rank ${cls}">${cr?cr.rank:'-'}</span></span> <span class="pill">score: <span class="score">${cr?fmt(cr.score):'-'}</span></span></div><table><thead><tr><th>#</th><th>top category</th><th>score</th></tr></thead><tbody>${top}</tbody></table>`}
function methodKeys(r){return Object.keys(r.methods||{});}
async function main(){const rows=await (await fetch('/api/results')).json(); const root=document.getElementById('root'); root.innerHTML=rows.map(r=>{const overlay=r.debug_overlay||r.overlay_path; const crop=r.debug_crop||r.crop_path; const tables=methodKeys(r).map(k=>methodTable(r,k,k)).join(''); return `<section class="card"><div class="head"><div><h2>${esc(r.sample)}</h2><div class="expr">${esc(r.expression)}</div></div><div class="meta">${esc(r.model||'')}<br>frame ${esc(r.selected_frame)} · bbox ${esc(JSON.stringify(r.bbox_xyxy))}<br>mask_ratio ${fmt(r.mask_area_ratio)} · visibility ${fmt(r.visibility_score)}</div></div><div class="body"><div class="imgs"><figure><img src="${img(overlay)}"><figcaption>mask overlay</figcaption></figure><figure><img src="${img(crop)}"><figcaption>bbox crop</figcaption></figure></div><div>${tables}</div></div></section>`}).join('')}
main();
</script>
</body>
</html>
"""

class Store:
    def __init__(self, result_root: Path):
        self.result_root = result_root

    def results(self) -> List[Dict[str, Any]]:
        rows = []
        for path in sorted(self.result_root.glob("*/result.json")):
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        return rows

class Handler(BaseHTTPRequestHandler):
    store: Store

    def send_bytes(self, data: bytes, content_type: str):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        try:
            if parsed.path == "/":
                self.send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif parsed.path == "/api/results":
                self.send_bytes(json.dumps(self.store.results(), ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
            elif parsed.path == "/image":
                path = Path((params.get("path") or [""])[0]).resolve()
                cwd = Path.cwd().resolve()
                if cwd not in path.parents and path != cwd:
                    self.send_error(403)
                    return
                ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
                self.send_bytes(path.read_bytes(), ctype)
            else:
                self.send_error(404)
        except Exception as exc:
            self.send_error(500, str(exc))

    def log_message(self, fmt: str, *args: Any) -> None:
        return

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-root", type=Path, default=Path("category_grounding_mvp/outputs/siglip_label_only_debug_20260706_rerun"))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8803)
    args = ap.parse_args()
    Handler.store = Store(args.result_root)
    print(json.dumps({"event":"start", "url":f"http://{args.host}:{args.port}", "result_root":str(args.result_root)}, ensure_ascii=False), flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()

if __name__ == "__main__":
    main()
