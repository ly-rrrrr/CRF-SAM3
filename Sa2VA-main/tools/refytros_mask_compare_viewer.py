#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

HTML = r'''
<!doctype html><html><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>RefYTROS Mask Compare</title>
<style>
:root{--bg:#f5f7fb;--panel:#fff;--line:#d9e0ea;--text:#111827;--muted:#667085;--blue:#175cd3;--green:#07823b;--red:#b42318}*{box-sizing:border-box}body{margin:0;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:var(--bg);color:var(--text)}header{position:sticky;top:0;z-index:4;background:#fff;border-bottom:1px solid var(--line);padding:10px 14px}h1{font-size:18px;margin:0 0 8px}.summary{font-size:13px;color:var(--muted);display:flex;gap:10px;flex-wrap:wrap}.summary b{color:var(--text)}.toolbar{display:grid;grid-template-columns:minmax(220px,1fr) 150px 150px 90px 90px 110px;gap:8px;margin-top:10px}input,select,button{height:34px;border:1px solid var(--line);border-radius:6px;background:#fff;padding:0 9px;font:inherit;font-size:13px}button{cursor:pointer;font-weight:560}main{display:grid;grid-template-columns:370px minmax(0,1fr);gap:10px;padding:10px}.list,.detail{background:var(--panel);border:1px solid var(--line);border-radius:8px;overflow:hidden}.listhead{display:flex;justify-content:space-between;padding:9px 11px;border-bottom:1px solid var(--line);font-size:12px;color:var(--muted)}.items{height:calc(100vh - 168px);overflow:auto}.item{display:block;width:100%;text-align:left;border:0;border-bottom:1px solid #edf0f5;background:#fff;height:auto;padding:9px 11px;border-radius:0}.item:hover{background:#f8fbff}.item.active{background:#eef5ff;box-shadow:inset 3px 0 0 var(--blue)}.row{display:flex;justify-content:space-between;gap:8px;align-items:center}.id{font-weight:700;font-size:13px}.chip{font-size:11px;border:1px solid var(--line);border-radius:999px;padding:2px 6px;background:#fff;color:#344054;white-space:nowrap}.good{color:var(--green);border-color:#9fe6b8;background:#edfcf2}.bad{color:var(--red);border-color:#f4b2ad;background:#fff1f0}.expr{font-size:12px;color:#344054;line-height:1.35;margin-top:5px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}.metrics{font-size:12px;color:#667085;margin-top:5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.detail{padding:12px;overflow:auto;height:calc(100vh - 118px)}.title{border-bottom:1px solid var(--line);padding-bottom:10px;margin-bottom:10px}.title h2{font-size:18px;margin:0}.title p{margin:6px 0 0;color:#344054}.metricgrid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px;margin-bottom:10px}.metric{border:1px solid var(--line);border-radius:8px;padding:8px;background:#fff}.metric span{display:block;color:var(--muted);font-size:11px}.metric b{font-size:16px}.viewerbar{display:flex;align-items:center;gap:10px;margin:8px 0 10px}.viewerbar input[type=range]{flex:1}.frames{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.card{border:1px solid var(--line);border-radius:8px;background:#fff;overflow:hidden}.card h3{margin:0;padding:8px 10px;font-size:13px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between}.card img{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:#111827}.cap{font-size:12px;color:#475467;padding:7px 9px;border-top:1px solid var(--line);word-break:break-all}.empty{padding:18px;color:var(--muted)}.scoregrid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin-top:10px}.scorecard{border:1px solid var(--line);border-radius:8px;background:#fff;overflow:hidden}.scorecard h3{margin:0;padding:8px 10px;font-size:13px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between}.scoremeta{padding:8px 10px;font-size:12px;color:#344054;line-height:1.55}.scoremeta b{color:#111827}.scoretable{width:100%;border-collapse:collapse;font-size:12px}.scoretable th,.scoretable td{border-top:1px solid #edf0f5;padding:6px 8px;text-align:left;white-space:nowrap}.scoretable th{color:#667085;font-weight:600;background:#f8fafc}.scoretable td.sel{font-weight:700;color:var(--green)}@media(max-width:1050px){main{grid-template-columns:1fr}.toolbar{grid-template-columns:1fr 1fr}.items{height:340px}.frames{grid-template-columns:1fr}.metricgrid{grid-template-columns:repeat(2,1fr)}}
</style></head><body>
<header><h1>RefYTROS Mask Compare</h1><div class="summary" id="summary">Loading...</div><div class="toolbar"><input id="q" placeholder="video / exp / expression"/><select id="bucket"><option value="all">All</option><option value="improved">New better</option><option value="worse">New worse</option><option value="old_wrong_new_right">old &lt;50, new >=50</option><option value="new_wrong">new J&F &lt;50</option></select><select id="sort"><option value="delta">Sort delta</option><option value="new_low">Sort new low</option><option value="old_low">Sort old low</option><option value="video">Sort video</option></select><button id="prev">Prev</button><button id="next">Next</button><label style="display:flex;align-items:center;gap:6px;font-size:13px"><input type="checkbox" id="show" checked/>Mask</label></div></header>
<main><section class="list"><div class="listhead"><span id="count">0</span><span id="page">page</span></div><div class="items" id="items"></div></section><section class="detail" id="detail"><div class="empty">Select a sample.</div></section></main>
<script>
let state={page:1,pageSize:50,total:0,records:[],active:null,sample:null,frame:0};const $=id=>document.getElementById(id);const esc=s=>String(s??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));const fmt=(x,n=2)=>x==null?'-':Number(x).toFixed(n);function cls(d){return d>0.01?'good':(d<-0.01?'bad':'')}async function loadSummary(){const m=await(await fetch('/api/summary')).json();summary.innerHTML=`old <b>${fmt(m.old.jf)}</b> · new <b>${fmt(m.new.jf)}</b> · Δ <b class="${cls(m.delta_jf)}">${fmt(m.delta_jf)}</b> · expressions <b>${m.num_expressions}</b>`}function params(){return new URLSearchParams({page:state.page,page_size:state.pageSize,q:q.value,bucket:bucket.value,sort:sort.value})}async function loadList(first=false){const d=await(await fetch('/api/list?'+params())).json();state.records=d.records;state.total=d.total;state.page=d.page;renderList();if(first&&d.records.length)openSample(d.records[0].video_id,d.records[0].exp_id)}function renderList(){count.textContent=`${state.total} records`;page.textContent=`page ${state.page} / ${Math.max(1,Math.ceil(state.total/state.pageSize))}`;items.innerHTML=state.records.map(r=>`<button class="item ${state.active===r.id?'active':''}" onclick="openSample('${esc(r.video_id)}','${esc(r.exp_id)}')"><div class="row"><span class="id">${esc(r.video_id)}/${esc(r.exp_id)}</span><span class="chip ${cls(r.delta_jf)}">Δ ${fmt(r.delta_jf)}</span></div><div class="expr">${esc(r.expression)}</div><div class="metrics">old ${fmt(r.old_jf)} · new ${fmt(r.new_jf)} · J ${fmt(r.old_j)}→${fmt(r.new_j)} · F ${fmt(r.old_f)}→${fmt(r.new_f)}</div></button>`).join('')||'<div class="empty">No records.</div>'}async function openSample(v,e){state.active=`${v}::${e}`;state.sample=await(await fetch(`/api/sample?video_id=${encodeURIComponent(v)}&exp_id=${encodeURIComponent(e)}`)).json();state.frame=0;renderList();renderDetail()}function img(kind){const s=state.sample;const fr=s.frames[state.frame];const showMask=document.getElementById('show').checked?'1':'0';return `/overlay?kind=${kind}&video_id=${encodeURIComponent(s.video_id)}&exp_id=${encodeURIComponent(s.exp_id)}&frame=${encodeURIComponent(fr)}&show=${showMask}`}
function scoreCard(label,d){d=d||{};const rows=(d.candidates||[]).map(c=>`<tr><td class="${c.is_selected?'sel':''}">obj_${esc(c.obj_id)}${c.is_selected?' *':''}</td><td>${fmt(c.score,4)}</td><td>${fmt(c.prob_yes,4)}</td><td>${fmt(c.prob_partial,4)}</td><td>${fmt(c.prob_no,4)}</td><td>${esc(c.prediction)}</td></tr>`).join('')||'<tr><td colspan="6">No candidates</td></tr>';return `<div class="scorecard"><h3>${label}<span>${esc(d.used_category_query||d.category||'-')}</span></h3><div class="scoremeta"><b>category</b>: ${esc(d.category||'-')} · <b>primary</b>: ${esc(d.primary_category||'-')} · <b>used query</b>: ${esc(d.used_category_query||'-')}<br/><b>num_targets</b>: ${esc(d.num_targets??'-')} · <b>candidates</b>: ${esc(d.num_candidates??'-')} · <b>selected</b>: ${esc((d.selected_obj_ids||[]).join(', ')||'-')}<br/><b>rule</b>: ${esc(d.final_rule_applied||'-')}</div><table class="scoretable"><thead><tr><th>obj</th><th>score</th><th>yes</th><th>partial</th><th>no</th><th>pred</th></tr></thead><tbody>${rows}</tbody></table></div>`}function renderDetail(){const s=state.sample;if(!s)return;const fr=s.frames[state.frame];detail.innerHTML=`<div class="title"><h2>${esc(s.video_id)}/${esc(s.exp_id)}</h2><p>${esc(s.expression)}</p></div><div class="metricgrid"><div class="metric"><span>old J&F</span><b>${fmt(s.old.JF)}</b></div><div class="metric"><span>new J&F</span><b>${fmt(s.new.JF)}</b></div><div class="metric"><span>delta</span><b class="${cls(s.delta_jf)}">${fmt(s.delta_jf)}</b></div><div class="metric"><span>old J/F</span><b>${fmt(s.old.J)} / ${fmt(s.old.F)}</b></div><div class="metric"><span>new J/F</span><b>${fmt(s.new.J)} / ${fmt(s.new.F)}</b></div></div><div class="viewerbar"><button onclick="step(-1)">Prev frame</button><input id="frame" type="range" min="0" max="${s.frames.length-1}" value="${state.frame}" oninput="state.frame=Number(this.value);updateImgs()"/><button onclick="step(1)">Next frame</button><span id="frlabel">${esc(fr)} (${state.frame+1}/${s.frames.length})</span></div><div class="frames"><div class="card"><h3>Old recall24+keep7 <span>${fmt(s.old.JF)}</span></h3><img id="oldimg" src="${img('old')}"/><div class="cap">${esc(s.old_dir)}</div></div><div class="card"><h3>New query-grounding <span>${fmt(s.new.JF)}</span></h3><img id="newimg" src="${img('new')}"/><div class="cap">${esc(s.new_dir)}</div></div><div class="card"><h3>GT</h3><img id="gtimg" src="${img('gt')}"/><div class="cap">${esc(s.gt_dir)}</div></div></div><div class="scoregrid">${scoreCard('Old category / scores',s.old_score)}${scoreCard('New category / scores',s.new_score)}</div>`}function updateImgs(){const s=state.sample;const fr=s.frames[state.frame];$('frlabel').textContent=`${fr} (${state.frame+1}/${s.frames.length})`;$('oldimg').src=img('old');$('newimg').src=img('new');$('gtimg').src=img('gt')}function step(d){if(!state.sample)return;state.frame=(state.frame+d+state.sample.frames.length)%state.sample.frames.length;renderDetail()}q.oninput=()=>{state.page=1;loadList(true)};bucket.onchange=()=>{state.page=1;loadList(true)};sort.onchange=()=>{state.page=1;loadList(true)};prev.onclick=()=>{state.page=Math.max(1,state.page-1);loadList(true)};next.onclick=()=>{if(state.page*state.pageSize<state.total){state.page++;loadList(true)}};show.onchange=()=>{if(state.sample)updateImgs()};loadSummary().then(()=>loadList(true));
</script></body></html>
'''


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def metric_key(video_id: str, exp_id: str) -> str:
    return f'{video_id}_{exp_id}'


class Store:
    def __init__(self, data_root: Path, split: str, old_root: Path, new_root: Path, meta_file: str):
        self.data_root = data_root.resolve(); self.split = split
        self.old_root = old_root.resolve(); self.new_root = new_root.resolve()
        self.meta_path = self.data_root / split / meta_file
        self.meta = load_json(self.meta_path)['videos']
        self.old_eval = load_json(self.old_root / 'evaluation_results_refytros_png.json')
        self.new_eval = load_json(self.new_root / 'evaluation_results_refytros_png.json')
        self.allowed = [self.data_root, self.old_root, self.new_root]
        self.old_scores = self._load_scores(self.old_root / 'scores.json')
        self.new_scores = self._load_scores(self.new_root / 'scores.json')
        self.records = self._build_records()
        self.map = {(r['video_id'], r['exp_id']): r for r in self.records}

    @staticmethod
    def _load_scores(path: Path) -> Dict[tuple[str, str], Dict[str, Any]]:
        if not path.exists():
            return {}
        data = load_json(path)
        if isinstance(data, dict):
            data = data.get('records', [])
        out: Dict[tuple[str, str], Dict[str, Any]] = {}
        for rec in data:
            if isinstance(rec, dict):
                out[(str(rec.get('video_id', '')), str(rec.get('exp_id', '')))] = rec
        return out

    def _build_records(self) -> List[Dict[str, Any]]:
        old_per = self.old_eval['per_expression']; new_per = self.new_eval['per_expression']
        rows = []
        for video_id in sorted(self.meta):
            exprs = self.meta[video_id].get('expressions', {})
            for exp_id in sorted(exprs, key=lambda x: int(x) if str(x).isdigit() else str(x)):
                k = metric_key(video_id, str(exp_id)); old = old_per.get(k, {}); new = new_per.get(k, {})
                old_jf = float(old.get('J&F', 0.0)) * 100.0 if old.get('J&F', 0.0) <= 1 else float(old.get('J&F', 0.0))
                new_jf = float(new.get('J&F', 0.0)) * 100.0 if new.get('J&F', 0.0) <= 1 else float(new.get('J&F', 0.0))
                rows.append({'id': f'{video_id}::{exp_id}', 'video_id': video_id, 'exp_id': str(exp_id), 'expression': exprs[exp_id].get('exp',''), 'old_j': float(old.get('J',0))*100, 'old_f': float(old.get('F',0))*100, 'old_jf': old_jf, 'new_j': float(new.get('J',0))*100, 'new_f': float(new.get('F',0))*100, 'new_jf': new_jf, 'delta_jf': new_jf-old_jf})
        return rows

    def summary(self) -> Dict[str, Any]:
        old = self.old_eval['overall_expression_level']; new = self.new_eval['overall_expression_level']
        return {'old': {'j': old['J'], 'f': old['F'], 'jf': old['J&F']}, 'new': {'j': new['J'], 'f': new['F'], 'jf': new['J&F']}, 'delta_jf': new['J&F']-old['J&F'], 'num_expressions': len(self.records)}

    def list(self, qs: Dict[str, List[str]]) -> Dict[str, Any]:
        rows = self.records
        q = qs.get('q',[''])[0].lower().strip(); bucket = qs.get('bucket',['all'])[0]; sort = qs.get('sort',['delta'])[0]
        if q: rows = [r for r in rows if q in f"{r['video_id']} {r['exp_id']} {r['expression']}".lower()]
        if bucket == 'improved': rows = [r for r in rows if r['delta_jf'] > 0.01]
        elif bucket == 'worse': rows = [r for r in rows if r['delta_jf'] < -0.01]
        elif bucket == 'old_wrong_new_right': rows = [r for r in rows if r['old_jf'] < 50 <= r['new_jf']]
        elif bucket == 'new_wrong': rows = [r for r in rows if r['new_jf'] < 50]
        if sort == 'new_low': rows = sorted(rows, key=lambda r: (r['new_jf'], r['video_id'], int(r['exp_id']) if r['exp_id'].isdigit() else r['exp_id']))
        elif sort == 'old_low': rows = sorted(rows, key=lambda r: (r['old_jf'], r['video_id'], int(r['exp_id']) if r['exp_id'].isdigit() else r['exp_id']))
        elif sort == 'video': rows = sorted(rows, key=lambda r: (r['video_id'], int(r['exp_id']) if r['exp_id'].isdigit() else r['exp_id']))
        else: rows = sorted(rows, key=lambda r: (-abs(r['delta_jf']), r['video_id'], int(r['exp_id']) if r['exp_id'].isdigit() else r['exp_id']))
        page = max(1, int(qs.get('page',['1'])[0] or 1)); size = max(1, min(200, int(qs.get('page_size',['50'])[0] or 50)))
        return {'total': len(rows), 'page': page, 'page_size': size, 'records': rows[(page-1)*size:page*size]}

    def sample(self, video_id: str, exp_id: str) -> Dict[str, Any]:
        r = self.map[(video_id, exp_id)]; frames = [str(x) for x in self.meta[video_id]['frames']]
        return {**r, 'frames': frames, 'old': {'J': r['old_j'], 'F': r['old_f'], 'JF': r['old_jf']}, 'new': {'J': r['new_j'], 'F': r['new_f'], 'JF': r['new_jf']}, 'old_score': self.old_scores.get((video_id, exp_id), {}), 'new_score': self.new_scores.get((video_id, exp_id), {}), 'old_dir': str(self.old_root), 'new_dir': str(self.new_root), 'gt_dir': str(self.data_root / self.split / 'Annotations' / video_id / exp_id)}

    def paths(self, kind: str, video_id: str, exp_id: str, frame: str) -> tuple[Path, Optional[Path]]:
        rgb = self.data_root / self.split / 'JPEGImages' / video_id / f'{frame}.jpg'
        if kind == 'old': mask = self.old_root / 'Annotations' / video_id / exp_id / f'{frame}.png'
        elif kind == 'new': mask = self.new_root / 'Annotations' / video_id / exp_id / f'{frame}.png'
        else: mask = self.data_root / self.split / 'Annotations' / video_id / exp_id / f'{frame}.png'
        return rgb, mask

    def safe(self, p: Path) -> bool:
        try: rp = p.resolve()
        except Exception: return False
        return any(str(rp).startswith(str(root)) for root in self.allowed)


class Handler(BaseHTTPRequestHandler):
    store: Store
    def json(self, obj: Any):
        data=json.dumps(obj,ensure_ascii=False).encode('utf-8'); self.send_response(200); self.send_header('Content-Type','application/json; charset=utf-8'); self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        u=urlparse(self.path); qs=parse_qs(u.query)
        if u.path == '/':
            data=HTML.encode('utf-8'); self.send_response(200); self.send_header('Content-Type','text/html; charset=utf-8'); self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data); return
        if u.path == '/api/summary': self.json(self.store.summary()); return
        if u.path == '/api/list': self.json(self.store.list(qs)); return
        if u.path == '/api/sample': self.json(self.store.sample(qs.get('video_id',[''])[0], qs.get('exp_id',[''])[0])); return
        if u.path == '/overlay':
            kind=qs.get('kind',['gt'])[0]; video=qs.get('video_id',[''])[0]; exp=qs.get('exp_id',[''])[0]; frame=qs.get('frame',[''])[0]; show=qs.get('show',['1'])[0]=='1'
            rgb_path, mask_path = self.store.paths(kind, video, exp, frame)
            if not self.store.safe(rgb_path) or not rgb_path.exists(): self.send_error(404); return
            img = Image.open(rgb_path).convert('RGB')
            if show and mask_path and self.store.safe(mask_path) and mask_path.exists():
                mask = Image.open(mask_path).convert('L').resize(img.size, Image.NEAREST)
                arr=np.array(img).astype(np.float32); m=(np.array(mask)>0)
                color = np.array([255,70,40] if kind=='old' else ([40,180,80] if kind=='new' else [60,130,255]), dtype=np.float32)
                arr[m] = arr[m]*0.45 + color*0.55
                img = Image.fromarray(np.clip(arr,0,255).astype(np.uint8))
            buf=io.BytesIO(); img.save(buf,format='PNG'); data=buf.getvalue(); self.send_response(200); self.send_header('Content-Type','image/png'); self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data); return
        self.send_error(404)
    def log_message(self, fmt, *args): print(f'[{self.log_date_time_string()}] {fmt%args}', flush=True)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',required=True); ap.add_argument('--split',default='valid'); ap.add_argument('--meta-file',default='meta_expressions_challenge.json'); ap.add_argument('--old-root',required=True); ap.add_argument('--new-root',required=True); ap.add_argument('--host',default='127.0.0.1'); ap.add_argument('--port',type=int,default=8802); args=ap.parse_args()
    Handler.store=Store(Path(args.data_root),args.split,Path(args.old_root),Path(args.new_root),args.meta_file)
    srv=ThreadingHTTPServer((args.host,args.port),Handler); print(f'[viewer] http://{args.host}:{args.port} records={len(Handler.store.records)}', flush=True); srv.serve_forever()
if __name__=='__main__': main()
