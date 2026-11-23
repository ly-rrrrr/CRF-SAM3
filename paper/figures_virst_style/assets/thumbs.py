import json, numpy as np, sys
from PIL import Image, ImageDraw
from pycocotools import mask as mu
base="/mnt/project-files/论文/paper_assets/revos_qualitative_20261008/"
out="/mnt/project-files/论文/framework_virst_style/assets/"
idx=[0,12,24]
W,H=480,270
def rgb(i): return Image.open(f"{base}cases/B/rgb/{i:05d}.jpg").convert("RGB")
cand=json.load(open(base+"recall_replay/B/candidate_masks.json"))
def candmask(cid,i):
    r=cand[str(cid)][i]
    if r is None: return None
    r=dict(r); r["counts"]=r["counts"].encode() if isinstance(r["counts"],str) else r["counts"]
    return mu.decode(r).astype(bool)
def predmask(i): return np.array(Image.open(f"{base}cases/B/prediction/{i:05d}.png"))>0
def overlay(im,m,col,a=0.55):
    arr=np.array(im).astype(float)
    if m is not None:
        c=np.array(col,float); arr[m]=arr[m]*(1-a)+c*a
        # contour
        from PIL import ImageFilter
        mm=Image.fromarray((m*255).astype(np.uint8)); edge=np.array(mm.filter(ImageFilter.FIND_EDGES))>0
        edge=np.array(Image.fromarray(edge.astype(np.uint8)*255).filter(ImageFilter.MaxFilter(5)))>0
        arr[edge]=c
    return Image.fromarray(arr.astype(np.uint8))
def stack(frames,name,off=(34,-22)):
    n=len(frames); bw=3
    cw=W+off[0]*(n-1)+2*bw; ch=H+abs(off[1])*(n-1)+2*bw
    canvas=Image.new("RGBA",(cw,ch),(0,0,0,0))
    for k,f in enumerate(frames):
        f=f.resize((W,H),Image.LANCZOS)
        x=off[0]*(n-1-k); y=abs(off[1])*k
        fr=Image.new("RGBA",(W+2*bw,H+2*bw),(255,255,255,255)); fr.paste(f,(bw,bw))
        d=ImageDraw.Draw(fr); d.rectangle([0,0,W+2*bw-1,H+2*bw-1],outline=(150,150,150,255),width=1)
        canvas.alpha_composite(fr,(x,y))
    canvas.save(out+name)
print(len(cand["0"]), len(cand["1"]))
stack([rgb(i) for i in idx],"input_video.png")
stack([overlay(rgb(i),candmask(0,i),(230,145,56)) for i in idx],"cand_R0.png")
stack([overlay(rgb(i),candmask(1,i),(66,120,200)) for i in idx],"cand_R1.png")
stack([overlay(rgb(i),predmask(i),(66,120,200)) for i in idx],"output_mask.png")
