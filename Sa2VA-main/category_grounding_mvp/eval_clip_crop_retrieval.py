#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from transformers import CLIPModel, CLIPProcessor

REVOS = Path('/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/revos')
META = REVOS / 'meta_expressions_train_.json'
MASK_DICT = REVOS / 'mask_dict.json'
VOCAB = Path('category_grounding_mvp/data/openimages_vocab/processed_20260703/categories.jsonl')
MODEL = Path('clip-vit-base-patch32')
OUT = Path('category_grounding_mvp/outputs/clip_vit_b32_crop_retrieval_debug_20260706')


def read_json(p: Path):
    return json.load(open(p, encoding='utf-8'))

def decode(rle):
    if not rle: return None
    x = mask_utils.decode(rle)
    if x.ndim == 3: x = x.any(axis=2)
    return x.astype(bool)

def merge(mask_dict, anno_ids, idx):
    out = None
    for aid in anno_ids:
        seq = mask_dict.get(str(aid)) or mask_dict.get(aid)
        if seq is None or idx >= len(seq): continue
        m = decode(seq[idx])
        if m is None: continue
        out = m if out is None else np.logical_or(out, m)
    return out

def bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0: return None
    return int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)

def visibility(mask):
    b = bbox(mask)
    if b is None: return 0.0
    h,w = mask.shape[:2]
    x0,y0,x1,y1 = b
    bw,bh = max(1,x1-x0), max(1,y1-y0)
    area = float(mask.sum())
    fill = area / max(1.0, bw*bh)
    area_ratio = area / max(1.0, h*w)
    edge = float(x0 <= 1 or y0 <= 1 or x1 >= w-1 or y1 >= h-1)
    return math.log1p(area) + 2.0 * math.sqrt(max(area_ratio,0.0)) + 0.5 * fill - 0.75 * edge

def parse_sample(s):
    parts=s.split('/')
    return '/'.join(parts[:-1]), parts[-1]

def frame_path(root, vid, frame):
    d = root / vid
    for suf in ['.jpg','.jpeg','.png']:
        p = d / f'{frame}{suf}'
        if p.exists(): return p
    return d / f'{frame}.jpg'

def crop_image(image, mask, pad_ratio):
    b = bbox(mask)
    if b is None: return image.convert('RGB')
    x0,y0,x1,y1 = b
    w,h = image.size
    pad = int(round(max(x1-x0, y1-y0) * pad_ratio))
    return image.convert('RGB').crop((max(0,x0-pad), max(0,y0-pad), min(w,x1+pad), min(h,y1+pad)))

def overlay(image, mask, path):
    arr = np.asarray(image.convert('RGB')).copy().astype(np.float32)
    arr[mask] = arr[mask] * 0.45 + np.array([255,48,48], np.float32) * 0.55
    b = bbox(mask)
    if b:
        x0,y0,x1,y1 = b; h,w = arr.shape[:2]
        arr[max(0,y0-2):min(h,y0+2),x0:x1] = [255,255,0]
        arr[max(0,y1-2):min(h,y1+2),x0:x1] = [255,255,0]
        arr[y0:y1,max(0,x0-2):min(w,x0+2)] = [255,255,0]
        arr[y0:y1,max(0,x1-2):min(w,x1+2)] = [255,255,0]
    Image.fromarray(np.clip(arr,0,255).astype(np.uint8)).save(path)

def load_vocab(path):
    rows=[]
    for line in open(path, encoding='utf-8'):
        if not line.strip(): continue
        x=json.loads(line)
        text=(x.get('display_prompt_text') or x.get('display_name') or '').strip()
        if text:
            rows.append({**x, 'retrieval_text': text})
    return rows

@torch.inference_mode()
def encode_text(model, processor, texts, template, device, batch=256):
    feats=[]
    prompted=[template.format(t) for t in texts]
    for i in range(0,len(prompted),batch):
        inp=processor(text=prompted[i:i+batch], padding=True, truncation=True, return_tensors='pt')
        inp={k:v.to(device) for k,v in inp.items()}
        f=model.get_text_features(**inp).float()
        feats.append(torch.nn.functional.normalize(f, dim=-1).cpu())
    return torch.cat(feats)

@torch.inference_mode()
def encode_image(model, processor, images, device):
    inp=processor(images=images, return_tensors='pt')
    inp={k:v.to(device) for k,v in inp.items()}
    f=model.get_image_features(**inp).float()
    return torch.nn.functional.normalize(f, dim=-1)

def top(scores, vocab, k):
    vals, idxs=torch.topk(scores.detach().cpu(), k=min(k, scores.numel()))
    return [{'category_name': vocab[i]['display_name'], 'retrieval_text': vocab[i]['retrieval_text'], 'sam3_prompt_text': vocab[i].get('sam3_prompt_text'), 'mid': vocab[i].get('mid'), 'score': float(v)} for v,i in zip(vals.tolist(), idxs.tolist())]

def ranks(scores, vocab, names):
    order=torch.argsort(scores.detach().cpu(), descending=True)
    out=[]
    for name in names:
        idx=next((i for i,x in enumerate(vocab) if x.get('display_name')==name), None)
        if idx is None: continue
        pos=int((order==idx).nonzero(as_tuple=False)[0].item())+1
        out.append({'category_name': name, 'rank': pos, 'score': float(scores.detach().cpu()[idx])})
    return out

def evaluate(args, sample, model, processor, vocab, text_features_by_template, meta, mask_dict, device):
    vid,eid=parse_sample(sample)
    video=meta['videos'][vid]
    exp=video['expressions'][str(eid)]
    frames=[str(x) for x in video['frames']]
    anno_ids=[int(x) for x in (exp.get('anno_id') or exp.get('obj_id') or [])]
    visible=[]
    for i in range(len(frames)):
        m=merge(mask_dict, anno_ids, i)
        if m is not None and int(m.sum()) >= args.min_area:
            visible.append((i,m,visibility(m)))
    if not visible:
        raise RuntimeError(f'no visible mask: {sample}')
    visible.sort(key=lambda x:x[2], reverse=True)
    idx,mask,vis=visible[0]
    frame=frames[idx]
    image=Image.open(frame_path(args.image_root, vid, frame)).convert('RGB')
    crop=crop_image(image, mask, args.padding_ratio)
    out_dir=args.output_dir / sample.replace('/','_')
    out_dir.mkdir(parents=True, exist_ok=True)
    crop.save(out_dir/'crop.jpg')
    overlay(image, mask, out_dir/'overlay.jpg')
    feats=encode_image(model, processor, [crop, image], device)
    watch=['Person','Dog','Motorcycle','Bird','Wood-burning stove','Salt and pepper shakers']
    methods={}
    for template, text_features in text_features_by_template.items():
        tf=text_features.to(device)
        crop_scores=(feats[0:1] @ tf.T).squeeze(0)
        full_scores=(feats[1:2] @ tf.T).squeeze(0)
        key='label_only' if template == '{}' else 'photo_template'
        methods[f'{key}_crop_pooled']={'top':top(crop_scores,vocab,args.top_k),'watch_ranks':ranks(crop_scores,vocab,watch)}
        methods[f'{key}_full_image_pooled']={'top':top(full_scores,vocab,args.top_k),'watch_ranks':ranks(full_scores,vocab,watch)}
    b=bbox(mask)
    result={
        'model': str(args.model_path), 'sample': sample, 'video_id': vid, 'exp_id': str(eid), 'expression': exp.get('exp'), 'anno_ids': anno_ids,
        'selected_frame_index': idx, 'selected_frame': frame, 'visibility_score': vis, 'image_size': list(image.size),
        'bbox_xyxy': list(b) if b else None, 'mask_area': int(mask.sum()), 'mask_area_ratio': float(mask.sum()/max(1,mask.size)),
        'methods': methods, 'debug_overlay': str(out_dir/'overlay.jpg'), 'debug_crop': str(out_dir/'crop.jpg')
    }
    (out_dir/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2), encoding='utf-8')
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--samples', nargs='+', required=True)
    ap.add_argument('--model-path', type=Path, default=MODEL)
    ap.add_argument('--meta-expressions', type=Path, default=META)
    ap.add_argument('--mask-dict', type=Path, default=MASK_DICT)
    ap.add_argument('--image-root', type=Path, default=REVOS)
    ap.add_argument('--vocab-jsonl', type=Path, default=VOCAB)
    ap.add_argument('--output-dir', type=Path, default=OUT)
    ap.add_argument('--top-k', type=int, default=10)
    ap.add_argument('--min-area', type=int, default=16)
    ap.add_argument('--padding-ratio', type=float, default=0.15)
    ap.add_argument('--device', default='cuda')
    args=ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device=torch.device(args.device if torch.cuda.is_available() or not args.device.startswith('cuda') else 'cpu')
    vocab=load_vocab(args.vocab_jsonl)
    processor=CLIPProcessor.from_pretrained(args.model_path, local_files_only=True)
    model=CLIPModel.from_pretrained(args.model_path, local_files_only=True).to(device).eval()
    texts=[x['retrieval_text'] for x in vocab]
    templates=['{}','a photo of a {}.']
    text_features={t: encode_text(model, processor, texts, t, device) for t in templates}
    meta=read_json(args.meta_expressions); mask_dict=read_json(args.mask_dict)
    results=[]
    for sample in args.samples:
        r=evaluate(args, sample, model, processor, vocab, text_features, meta, mask_dict, device)
        results.append(r)
        print(json.dumps(r,ensure_ascii=False,indent=2), flush=True)
    (args.output_dir/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2), encoding='utf-8')

if __name__ == '__main__': main()
