#!/usr/bin/env python3
"""Interactive ReVOS expression, category, frame, and GT-mask inspector."""

from __future__ import annotations

import argparse
import io
import json
import random
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps
from pycocotools import mask as mask_utils


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path("/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/revos")
DEFAULT_RESULTS = ROOT / "results" / "revos_train_qwen37plus_bbox_categories_3views.json"


HTML = r"""
<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>ReVOS Category Inspector</title>
<style>
:root{--bg:#f4f6f8;--panel:#fff;--line:#d7dde5;--text:#17202a;--muted:#667085;--blue:#1769aa;--green:#18794e;--red:#b42318}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
header{position:sticky;top:0;z-index:5;background:rgba(255,255,255,.97);border-bottom:1px solid var(--line);padding:12px 18px}
h1{font-size:18px;margin:0 0 10px}.toolbar{display:grid;grid-template-columns:minmax(320px,1fr) 190px auto auto auto;gap:8px;align-items:center}
input,select,button{height:36px;border:1px solid var(--line);border-radius:6px;background:#fff;padding:0 10px;font:inherit;font-size:13px}
button{cursor:pointer;font-weight:650}.primary{background:var(--blue);border-color:var(--blue);color:#fff}.toggle{display:flex;align-items:center;gap:7px;padding:0 9px;height:36px;border:1px solid var(--line);border-radius:6px;background:#fff;font-size:13px;white-space:nowrap}.toggle input{width:16px;height:16px;padding:0}
main{padding:14px 18px 28px}.empty{padding:32px;color:var(--muted);text-align:center;background:#fff;border:1px solid var(--line);border-radius:8px}
.headline{background:#fff;border:1px solid var(--line);border-radius:8px;padding:14px 16px}.headline-row{display:flex;gap:16px;justify-content:space-between;align-items:flex-start}.headline h2{font-size:18px;margin:0}.expression{font-size:15px;line-height:1.5;margin:8px 0 0;color:#344054}.badge{display:inline-flex;border:1px solid var(--line);border-radius:999px;padding:3px 8px;font-size:12px;white-space:nowrap}.badge.ok{color:var(--green);border-color:#9bd8bd;background:#edfbf4}.badge.error{color:var(--red);border-color:#f1aaa5;background:#fff3f2}
.facts{display:grid;grid-template-columns:repeat(6,minmax(120px,1fr));border-top:1px solid var(--line);margin-top:14px}.fact{padding:11px 12px 2px 0}.label{font-size:11px;color:var(--muted);margin-bottom:4px}.value{font-size:14px;font-weight:680;overflow-wrap:anywhere}.error-text{margin-top:10px;color:var(--red);font-size:12px}
.section-head{display:flex;justify-content:space-between;align-items:center;margin:17px 0 9px}.section-head h3{font-size:15px;margin:0}.section-head span{font-size:12px;color:var(--muted)}
.frames{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:10px}figure{margin:0;background:#fff;border:1px solid var(--line);border-radius:8px;overflow:hidden}figure img{display:block;width:100%;height:260px;object-fit:contain;background:#101828}figcaption{display:flex;justify-content:space-between;gap:8px;padding:8px 10px;border-top:1px solid var(--line);font-size:12px;color:#475467}.visible{color:var(--green);font-weight:650}.not-visible{color:var(--muted)}
@media(max-width:900px){.toolbar{grid-template-columns:1fr 1fr}.facts{grid-template-columns:repeat(2,1fr)}.frames{grid-template-columns:1fr}figure img{height:auto;max-height:70vh}}
</style>
</head>
<body>
<header>
  <h1>ReVOS Category Inspector</h1>
  <div class="toolbar">
    <input id="sample" value="LV-VIS/train/00053/54" placeholder="video_id/exp_id" />
    <select id="mode"><option value="random3">随机 3 个可见帧</option><option value="visible_all">全部可见帧</option><option value="all">全部视频帧</option></select>
    <button id="query" class="primary">查询</button>
    <button id="reroll">重新随机</button>
    <label class="toggle"><input id="mask" type="checkbox" checked />显示 mask</label>
  </div>
</header>
<main id="app"><div class="empty">输入样本 ID 后查询。</div></main>
<script>
const $=id=>document.getElementById(id); let current=null; let seed=Date.now();
function esc(s){return String(s??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));}
function mediaUrl(sample,idx){return `/media?id=${encodeURIComponent(sample)}&frame=${idx}&mask=${$('mask').checked?'1':'0'}&t=${Date.now()}`;}
function refreshImages(){document.querySelectorAll('img[data-frame]').forEach(img=>{img.src=mediaUrl(current.sample_id,img.dataset.frame)});}
async function load(){const id=$('sample').value.trim();if(!id)return;seed=Date.now();const p=new URLSearchParams({id,mode:$('mode').value,seed});const res=await fetch('/api/sample?'+p);const data=await res.json();if(!res.ok||data.error){$('app').innerHTML=`<div class="empty">${esc(data.error||res.statusText)}</div>`;return;}current=data;history.replaceState(null,'','?id='+encodeURIComponent(data.sample_id));render(data);}
function render(d){const rec=d.api_record||{};const status=rec.status||'not generated';const category=rec.pred_category||'-';const frames=d.frames.map(f=>`<figure><img loading="lazy" data-frame="${f.index}" src="${mediaUrl(d.sample_id,f.index)}"/><figcaption><span>#${String(f.index).padStart(5,'0')} · ${esc(f.name)}</span><span class="${f.mask_visible?'visible':'not-visible'}">${f.mask_visible?'GT visible':'GT empty'}</span></figcaption></figure>`).join('');$('app').innerHTML=`<section class="headline"><div class="headline-row"><div><h2>${esc(d.sample_id)}</h2><p class="expression">${esc(d.expression)}</p></div><span class="badge ${status==='ok'?'ok':'error'}">${esc(status)}</span></div><div class="facts"><div class="fact"><div class="label">API category</div><div class="value">${esc(category)}</div></div><div class="fact"><div class="label">Expression type</div><div class="value">${esc(d.type_name)}</div></div><div class="fact"><div class="label">Frames</div><div class="value">${d.total_frames}</div></div><div class="fact"><div class="label">GT-visible frames</div><div class="value">${d.visible_count}</div></div><div class="fact"><div class="label">GT range</div><div class="value">${esc(d.visible_ranges||'-')}</div></div><div class="fact"><div class="label">Annotation IDs</div><div class="value">${esc((d.anno_ids||[]).join(', '))}</div></div></div>${rec.error?`<div class="error-text">${esc(rec.error)}</div>`:''}</section><div class="section-head"><h3>Frames</h3><span>${d.frames.length} shown · ${esc(d.mode_label)}</span></div><section class="frames">${frames||'<div class="empty">没有可显示帧。</div>'}</section>`;}
$('query').onclick=load;$('reroll').onclick=()=>{seed=Date.now();load()};$('mode').onchange=load;$('mask').onchange=refreshImages;$('sample').addEventListener('keydown',e=>{if(e.key==='Enter')load()});
const initial=new URLSearchParams(location.search).get('id');if(initial){$('sample').value=initial;load();}
</script>
</body>
</html>
"""


def read_json_retry(path: Path, retries: int = 12) -> Any:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            last_error = error
            time.sleep(min(0.15 * (attempt + 1), 1.0))
    raise RuntimeError(f"Unable to read JSON: {path}: {last_error}")


def canonical_sample_id(value: str) -> Tuple[str, str, str]:
    raw = urllib.parse.unquote(value.strip())
    if raw.startswith("revos:"):
        raw = raw[len("revos:"):]
        video_id, separator, exp_id = raw.rpartition(":")
    else:
        video_id, separator, exp_id = raw.rpartition("/")
    if not separator or not video_id or not exp_id:
        raise ValueError("Sample ID must look like LV-VIS/train/00053/54")
    return video_id, str(exp_id), f"{video_id}/{exp_id}"


def decode_rle(rle: Any) -> np.ndarray | None:
    if not rle:
        return None
    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = decoded.any(axis=2)
    return decoded.astype(bool)


def merge_mask(mask_dict: Dict[str, Any], anno_ids: Sequence[Any], frame_index: int) -> np.ndarray | None:
    merged: np.ndarray | None = None
    for anno_id in anno_ids:
        sequence = mask_dict.get(str(anno_id), [])
        if frame_index >= len(sequence):
            continue
        mask = decode_rle(sequence[frame_index])
        if mask is not None:
            merged = mask if merged is None else np.logical_or(merged, mask)
    return merged


def compact_ranges(indices: Iterable[int]) -> str:
    values = sorted(set(int(index) for index in indices))
    if not values:
        return ""
    ranges: List[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(f"{start:05d}" if start == previous else f"{start:05d}-{previous:05d}")
        start = previous = value
    ranges.append(f"{start:05d}" if start == previous else f"{start:05d}-{previous:05d}")
    return ", ".join(ranges)


def find_frame(image_root: Path, video_id: str, frame_name: str) -> Path:
    base = image_root / video_id
    candidate = base / frame_name
    if candidate.suffix and candidate.is_file():
        return candidate
    for suffix in (".jpg", ".jpeg", ".png"):
        candidate = base / f"{frame_name}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Frame not found: {video_id}/{frame_name}")


class Store:
    def __init__(self, meta_path: Path, mask_path: Path, results_path: Path, image_root: Path, max_image_edge: int):
        self.videos = read_json_retry(meta_path)["videos"]
        self.masks = read_json_retry(mask_path)
        self.results_path = results_path
        self.image_root = image_root
        self.max_image_edge = max_image_edge
        self._result_mtime_ns = -1
        self._results: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def refresh_results(self) -> None:
        try:
            mtime_ns = self.results_path.stat().st_mtime_ns
        except FileNotFoundError:
            return
        if mtime_ns == self._result_mtime_ns:
            return
        with self._lock:
            if mtime_ns == self._result_mtime_ns:
                return
            payload = read_json_retry(self.results_path)
            self._results = {
                (str(record.get("video_id")), str(record.get("exp_id"))): record
                for record in payload.get("records", [])
                if isinstance(record, dict) and record.get("video_id") is not None
            }
            self._result_mtime_ns = mtime_ns

    def lookup(self, sample_id: str) -> Tuple[str, str, str, Dict[str, Any], Dict[str, Any]]:
        video_id, exp_id, canonical = canonical_sample_id(sample_id)
        video = self.videos.get(video_id)
        if not isinstance(video, dict):
            raise KeyError(f"Unknown video: {video_id}")
        expression = video.get("expressions", {}).get(exp_id)
        if not isinstance(expression, dict):
            raise KeyError(f"Unknown expression: {canonical}")
        self.refresh_results()
        return video_id, exp_id, canonical, video, expression

    def visible_indices(self, video: Dict[str, Any], expression: Dict[str, Any]) -> List[int]:
        anno_ids = expression.get("anno_id", [])
        visible: List[int] = []
        for frame_index in range(len(video.get("frames", []))):
            if any(
                frame_index < len(self.masks.get(str(anno_id), []))
                and bool(self.masks[str(anno_id)][frame_index])
                for anno_id in anno_ids
            ):
                visible.append(frame_index)
        return visible

    def sample_payload(self, sample_id: str, mode: str, seed: int) -> Dict[str, Any]:
        video_id, exp_id, canonical, video, expression = self.lookup(sample_id)
        frame_names = [str(name) for name in video.get("frames", [])]
        visible = self.visible_indices(video, expression)
        visible_set = set(visible)
        if mode == "all":
            selected = list(range(len(frame_names)))
            mode_label = "全部视频帧"
        elif mode == "visible_all":
            selected = visible
            mode_label = "全部 GT 可见帧"
        else:
            selected = sorted(random.Random(seed).sample(visible, min(3, len(visible)))) if visible else []
            mode_label = "随机 3 个 GT 可见帧"
        type_id = expression.get("type_id")
        type_name = {0: "referring", 1: "reasoning", 2: "unknown"}.get(type_id, str(type_id))
        return {
            "sample_id": canonical,
            "video_id": video_id,
            "exp_id": exp_id,
            "expression": expression.get("exp", ""),
            "type_id": type_id,
            "type_name": type_name,
            "anno_ids": expression.get("anno_id", []),
            "total_frames": len(frame_names),
            "visible_count": len(visible),
            "visible_ranges": compact_ranges(visible),
            "mode": mode,
            "mode_label": mode_label,
            "frames": [
                {"index": index, "name": frame_names[index], "mask_visible": index in visible_set}
                for index in selected
            ],
            "api_record": self._results.get((video_id, exp_id)),
        }

    def render_frame(self, sample_id: str, frame_index: int, show_mask: bool) -> bytes:
        video_id, _exp_id, _canonical, video, expression = self.lookup(sample_id)
        frame_names = [str(name) for name in video.get("frames", [])]
        if frame_index < 0 or frame_index >= len(frame_names):
            raise IndexError(f"Frame index out of range: {frame_index}")
        image = ImageOps.exif_transpose(Image.open(find_frame(self.image_root, video_id, frame_names[frame_index]))).convert("RGB")
        if show_mask:
            mask = merge_mask(self.masks, expression.get("anno_id", []), frame_index)
            if mask is not None:
                if mask.shape != (image.height, image.width):
                    mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
                    mask = np.asarray(mask_image.resize(image.size, Image.Resampling.NEAREST)) > 0
                pixels = np.asarray(image).copy().astype(np.float32)
                pixels[mask] = pixels[mask] * 0.48 + np.array([238, 50, 54], dtype=np.float32) * 0.52
                image = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
        if max(image.size) > self.max_image_edge:
            scale = self.max_image_edge / max(image.size)
            image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=88, optimize=True)
        return output.getvalue()


class Handler(BaseHTTPRequestHandler):
    store: Store

    def send_json(self, payload: Any, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
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
            elif parsed.path == "/api/sample":
                sample_id = (params.get("id") or [""])[0]
                mode = (params.get("mode") or ["random3"])[0]
                seed = int((params.get("seed") or ["0"])[0])
                self.send_json(self.store.sample_payload(sample_id, mode, seed))
            elif parsed.path == "/media":
                sample_id = (params.get("id") or [""])[0]
                frame_index = int((params.get("frame") or ["0"])[0])
                show_mask = (params.get("mask") or ["1"])[0] != "0"
                data = self.store.render_frame(sample_id, frame_index, show_mask)
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_error(404)
        except (KeyError, ValueError, IndexError, FileNotFoundError) as error:
            self.send_json({"error": str(error).strip("'\"")}, status=404)
        except Exception as error:
            self.send_json({"error": str(error)}, status=500)

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--meta-file", default="meta_expressions_train_.json")
    parser.add_argument("--mask-file", default="mask_dict.json")
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--max-image-edge", type=int, default=1100)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8804)
    args = parser.parse_args()
    data_root = Path(args.data_root)
    Handler.store = Store(
        meta_path=data_root / args.meta_file,
        mask_path=data_root / args.mask_file,
        results_path=Path(args.results),
        image_root=data_root,
        max_image_edge=args.max_image_edge,
    )
    print(json.dumps({"event": "start", "url": f"http://{args.host}:{args.port}", "results": args.results}, ensure_ascii=False), flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
