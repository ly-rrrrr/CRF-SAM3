import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

OUT = sys.argv[1]
TRAIN = GREEN            # single colour for every trainable module
LOSS = ("FFF4F4", RED)   # training-only supervision

def ice(x, y):   # frozen badge at (x,y) = top-right corner of a box
    text(x - 0.3, y - 0.12, 0.4, 0.26, [[("❄", "emoji", 10, "5A7FA8")]], wrap=False)
def fire(x, y):
    text(x - 0.3, y - 0.12, 0.4, 0.26, [[("🔥", "emoji", 10, "D2552B")]], wrap=False)
def badge(x, y, lab):
    box(x, y, 0.24, 0.22, [[(lab, "b", 9, "5C8A48")]], pal=("FFFFFF", "6E9A5B"), shape=MSO_SHAPE.OVAL, lw=0.8)
def stage(x, y, w, h, name, col):
    box(x, y, w, h, None, pal=("FAFAFA", "B5B5B5"), dash=True, lw=0.9, radius=0.12)
    text(x + 0.08, y + 0.03, 1.5, 0.26, [[(name, "b,i", 12.5, col)]], align=PP_ALIGN.LEFT)

# =========================================================== (a) inference
Y0 = 0.22
text(0.15, 0.62, 1.4, 0.24, [[("Text Query ", "", 11), ("q", "i", 11)]])
text(0.12, 0.86, 1.46, 0.6, [[("“Which joystick is being held by hands at the end of the video?”", "i", 8.5, SUB)]])
img(ASSET + "input_video.png", 0.15, 1.5, w=1.38)
text(0.15, 2.3, 1.38, 0.22, [[("Video ", "", 10), ("V", "i", 10)]])

# Ground
stage(1.68, Y0, 3.82, 2.42, "Ground", "5C8A48")
box(1.85, 0.66, 1.12, 0.5, ["Expression", "Conditioner"], pal=TRAIN, size=10.5); fire(2.97, 0.66)
box(1.85, 1.66, 1.12, 0.5, ["Qwen3-VL", "Vision Encoder"], pal=GREY, size=10.5); ice(2.97, 1.66)
arrow(1.55, 0.91, 1.85, 0.91)
arrow(1.55, 1.91, 1.85, 1.91)
box(3.2, 0.66, 1.0, 1.5, ["Target-", "Query", "Grounder"], pal=TRAIN, size=11.5); fire(4.2, 0.66)
badge(3.58, 1.86, "b")
arrow(2.97, 0.91, 3.2, 0.91); text(2.97, 0.66, 0.23, 0.22, [[("T", "i", 9.5)]])
arrow(2.97, 1.91, 3.2, 1.91); text(2.97, 1.66, 0.23, 0.22, [[("V", "i", 9.5)]])
box(4.43, 0.66, 0.92, 1.5, [[("Qwen3-VL", "", 11)], [("Decoder", "", 11)], [("prompt: ", "", 8.5, SUB), ("q", "i", 8.5, SUB)]], pal=GREY); ice(5.35, 0.66)
arrow(4.2, 1.41, 4.43, 1.41); text(4.18, 1.15, 0.27, 0.24, [[("V", "i", 9.5), ("′", "", 9.5)]])
text(1.75, 2.24, 3.7, 0.36, [[("GT masks are used only in training (b); inference uses RGB + ", "i", 8.5, SUB), ("q", "i", 8.5, SUB), (" only.", "i", 8.5, SUB)]])

# Recall
stage(5.62, Y0, 3.0, 2.42, "Recall", "4F74A3")
box(5.78, 0.62, 1.2, 0.36, [[("ĉ", "i", 11), (" = “joystick”", "", 10.5)]], pal=YELLOW, radius=0.18)
poly([(5.35, 0.8), (5.78, 0.8)])
box(5.78, 1.22, 1.2, 0.86, [[("SAM3", "", 13)], [("concept prompt", "", 8.5, SUB)], [("+ video ", "", 8.5, SUB), ("V", "i", 8.5, SUB)]], pal=GREY); ice(6.98, 1.22)
arrow(6.38, 0.98, 6.38, 1.22)
img(ASSET + "cand_R0.png", 7.4, 0.55, w=1.1)
img(ASSET + "cand_R1.png", 7.4, 1.29, w=1.1)
text(7.12, 0.7, 0.3, 0.3, [[("τ", "i", 11), ("1", "sub", 11)]])
text(7.12, 1.44, 0.3, 0.3, [[("τ", "i", 11), ("2", "sub", 11)]])
text(7.12, 1.98, 1.45, 0.26, [[("top-", "", 9, SUB), ("K", "i", 9, SUB), (" tubes ", "", 9, SUB), ("C", "i", 9, SUB), ("(", "", 9, SUB), ("q", "i", 9, SUB), (")", "", 9, SUB)]])
text(5.7, 2.24, 2.85, 0.36, [[("Same-category hypotheses are kept.", "i", 8.5, SUB)]])
poly([(6.98, 1.65), (7.12, 1.65)], head=False)
poly([(7.12, 0.86), (7.12, 1.62)], color="808080", lw=0.9, head=False)

# Verify
stage(8.74, Y0, 2.92, 2.42, "Verify", "7D5C96")
box(9.03, 0.62, 1.32, 0.68, [[("Qwen3-VL", "", 11)], [("Verifier (LoRA)", "", 11)]], pal=TRAIN); fire(10.35, 0.62)
badge(10.13, 1.1, "c")
poly([(8.52, 1.2), (8.62, 1.2), (8.62, 0.96), (9.03, 0.96)])
box(10.52, 0.66, 1.02, 0.6, [[("full ", "", 9.5), ("q", "i", 9.5)], [("(anchor prompt)", "", 8, SUB)]], pal=ORANGE, dash=True)
arrow(10.52, 0.96, 10.35, 0.96)
# scores
text(8.86, 1.47, 0.32, 0.2, [[("s", "i", 9.5), ("1", "sub", 9.5)]], align=PP_ALIGN.RIGHT)
box(9.2, 1.5, 0.05, 0.13, None, pal=("E6913A", "E6913A"), radius=0.01, lw=0.5)
text(8.86, 1.7, 0.32, 0.2, [[("s", "i", 9.5), ("2", "sub", 9.5)]], align=PP_ALIGN.RIGHT)
box(9.2, 1.73, 0.95, 0.13, None, pal=("4278C8", "4278C8"), radius=0.01, lw=0.5)
arrow(9.69, 1.3, 9.69, 1.46)
text(8.86, 1.93, 1.5, 0.24, [[("s", "i", 9), ("i", "i,sub", 9), (" = P(Yes | ", "", 9, SUB), ("q", "i", 9, SUB), (", ", "", 9, SUB), ("τ", "i", 9, SUB), ("i", "i,sub", 9, SUB), (")", "", 9, SUB)]])
img(ASSET + "output_mask.png", 10.46, 1.43, w=1.1)
arrow(10.2, 1.8, 10.44, 1.8)
text(10.4, 2.1, 1.25, 0.24, [[("arg max → ", "", 9, SUB), ("Y", "i", 9.5)]])

text(0.15, 2.7, 11.5, 0.3, [[("(a) Inference: Ground → Recall → Verify", "b", 12.5)]])
poly([(0.15, 3.07), (11.65, 3.07)], color="505050", lw=1.0, dash=True, head=False)
poly([(5.78, 3.2), (5.78, 7.0)], color="505050", lw=1.0, dash=True, head=False)

# =========================================================== (b) Ground training
LX, LW = 0.3, 1.9
CX = LX + LW / 2
# inputs
tokens(0.42, 6.42, 5, TRAIN)
text(0.15, 6.62, 2.2, 0.4, [[("Target Queries ", "", 10), ("Q", "i", 10), ("0", "sup", 10)], [("K = 8, category-agnostic", "", 8.5, SUB)]])
tokens(2.42, 6.42, 3, ORANGE, dots=False)
text(2.3, 6.62, 0.9, 0.4, [[("T", "i", 10)], [("expression", "", 8.5, SUB)]])
tokens(3.32, 6.42, 4, BLUE)
text(3.2, 6.62, 1.3, 0.4, [[("V", "i", 10)], [("visual tokens", "", 8.5, SUB)]])
# query decoder
box(LX, 4.88, LW, 1.4, None, pal=("F2F7EE", "6E9A5B"), radius=0.12); fire(LX + LW, 4.88)
text(LX, 4.88, LW, 0.22, [[("Query Decoder ×", "", 9, SUB), ("L", "i", 9, SUB)]])
for name, yy in [("Self-Attention", 5.95), ("Text Cross-Attn", 5.64), ("Visual Cross-Attn", 5.33)]:
    box(LX + 0.14, yy, LW - 0.28, 0.25, [name], pal=TRAIN, size=9.5, radius=0.06)
box(LX + 0.14, 5.1, LW - 0.28, 0.18, ["FFN"], pal=TRAIN, size=8.5, radius=0.05)
arrow(CX, 6.42, CX, 6.2)
for a, b in [(5.95, 5.89), (5.64, 5.58), (5.33, 5.28)]:
    arrow(CX, a, CX, b)
poly([(2.7, 6.42), (2.7, 5.765), (LX + LW - 0.14, 5.765)])
poly([(3.62, 6.42), (3.62, 5.455), (LX + LW - 0.14, 5.455)])
text(2.2, 5.5, 0.45, 0.22, [[("K,V", "", 8, SUB)]])
text(2.2, 5.2, 0.45, 0.22, [[("K,V", "", 8, SUB)]])
# Q^L
tokens(0.42, 4.6, 5, TRAIN)
text(1.85, 4.56, 0.4, 0.24, [[("Q", "i", 10), ("L", "sup", 10)]], align=PP_ALIGN.LEFT)
arrow(CX, 4.88, CX, 4.77)
# response
box(LX, 4.1, LW, 0.38, [[("Response ", "", 10), ("R", "i", 10), ("k,n", "i,sub", 10)], [("γ · cos(", "", 8, SUB), ("W", "i", 8, SUB), ("q", "i,sub", 8, SUB), ("q", "i", 8, SUB), ("k", "i,sub", 8, SUB), (", ", "", 8, SUB), ("W", "i", 8, SUB), ("v", "i,sub", 8, SUB), ("v", "i", 8, SUB), ("n", "i,sub", 8, SUB), (")", "", 8, SUB)]], pal=TRAIN, size=10); fire(LX + LW, 4.1)
arrow(CX, 4.6, CX, 4.48)
# gate
box(LX, 3.62, LW, 0.34, [[("train: σ(", "", 9), ("R", "i", 9), ("k*", "i,sub", 9), (")  ·  test: max", "", 9), ("k", "i,sub", 9), (" σ(", "", 9), ("R", "i", 9), ("k", "i,sub", 9), (")", "", 9)]], pal=WHITE, size=9)
arrow(CX, 4.1, CX, 3.96)
# V bus
poly([(4.2, 6.42), (4.2, 4.0), (3.5, 4.0), (3.5, 3.82)])
arrow(4.2, 5.145, LX + LW - 0.14, 5.145, head=True) if False else None
arrow(4.2, 4.29, LX + LW, 4.29)
# soft modulation (sits right of gate)
box(2.42, 3.34, 1.5, 0.48, [[("Soft Modulation", "", 10)], [("v", "i", 9), ("′ = ", "", 9), ("v", "i", 9), (" + α ", "", 9), ("m", "i", 9), (" ", "", 9), ("A", "i", 9), ("(", "", 9), ("v", "i", 9), (")", "", 9)]], pal=TRAIN); fire(3.92, 3.34)
arrow(LX + LW, 3.79, 2.42, 3.79)
text(2.15, 3.58, 0.3, 0.2, [[("m", "i", 9)]])
# frozen decoder + concept
box(4.45, 3.34, 1.2, 0.48, [[("Qwen3-VL", "", 10)], [("Decoder", "", 10)]], pal=GREY); ice(5.65, 3.34)
arrow(3.92, 3.58, 4.45, 3.58)
text(3.9, 3.3, 0.6, 0.24, [[("V", "i", 9), ("′", "", 9)]])
text(4.3, 3.82, 1.5, 0.22, [[("+ deepstack, prompt ", "", 8, SUB), ("q", "i", 8, SUB)]])
box(4.45, 4.15, 1.2, 0.32, [[("ĉ", "i", 10), (" “joystick”", "", 9.5)]], pal=YELLOW, radius=0.14)
arrow(5.05, 3.82, 5.05, 4.15)
# losses
box(4.45, 4.72, 1.2, 0.5, [[("ℒ", "", 10, RED), ("cat", "sub", 10, RED)], [("CE to category label", "", 8, RED)]], pal=LOSS, dash=True)
arrow(5.05, 4.47, 5.05, 4.72, color=RED, dash=True)
box(4.4, 5.55, 1.3, 0.62, [[("ℒ", "", 10, RED), ("ground", "sub", 10, RED)], [("BCE+Dice vs. GT mask", "", 8, RED)], [("on best slot ", "", 8, RED), ("k*", "i", 8, RED)]], pal=LOSS, dash=True)
poly([(4.4, 5.86), (4.32, 5.86), (4.32, 4.4), (LX + LW, 4.4)], color=RED, dash=True)
text(3.65, 6.62, 2.1, 0.4, [[("ℒ", "", 9.5), ("G", "sub", 9.5), (" = λ", "", 9.5), ("g", "sub", 9.5), ("ℒ", "", 9.5), ("ground", "sub", 9.5), (" + λ", "", 9.5), ("c", "sub", 9.5), ("ℒ", "", 9.5), ("cat", "sub", 9.5)]], align=PP_ALIGN.RIGHT) if False else None

# =========================================================== (c) Verify training
X0 = 5.95
text(X0, 3.2, 1.6, 0.26, [[("offline SAM3 cache", "i", 9, SUB)]])
img(ASSET + "cand_R0.png", X0 + 0.05, 3.48, w=1.3)
box(X0 + 0.12, 4.28, 1.15, 0.24, [[("neg (IoU ≤ 0.1)", "", 8.5, "B06A2A")]], pal=("FFFFFF", "E6913A"), radius=0.06, lw=0.8)
img(ASSET + "cand_R1.png", X0 + 0.05, 4.65, w=1.3)
box(X0 + 0.12, 5.45, 1.15, 0.24, [[("pos (IoU ≥ 0.5)", "", 8.5, "2F5E9E")]], pal=("FFFFFF", "4278C8"), radius=0.06, lw=0.8)
text(X0 + 0.05, 5.68, 1.3, 0.22, ["⋮"], size=10)
box(X0 + 0.05, 5.98, 1.3, 0.5, [[("full expression ", "", 9), ("q", "i", 9)], [("“the marked object …”", "i", 8, SUB)]], pal=ORANGE, dash=True)
# verifier
VX = 7.72
box(VX, 3.48, 1.34, 3.0, None, pal=TRAIN, radius=0.14); fire(VX + 1.34, 3.48)
text(VX, 3.55, 1.34, 0.9, [[("Qwen3-VL", "", 11.5)], [("Verifier", "", 11.5)], [("LoRA ", "", 8.5, SUB), ("r", "i", 8.5, SUB), ("=16", "", 8.5, SUB)], [("vision encoder ❄", "", 8.5, SUB)]])
text(VX, 4.55, 1.34, 0.9, [[("4 frames / candidate", "", 8.5, SUB)], [("mask overlay", "", 8.5, SUB)], [("anchor-subject prompt", "", 8.5, SUB)]])
arrow(X0 + 1.4, 3.88, VX, 3.88)
arrow(X0 + 1.4, 5.05, VX, 5.05)
arrow(X0 + 1.4, 6.23, VX, 6.23)
# outputs
SX = 9.3
text(SX, 3.4, 2.3, 0.25, [[("s", "i", 10), ("i", "i,sub", 10), (" = P(Yes | ", "", 10), ("q", "i", 10), (", ", "", 10), ("τ", "i", 10), ("i", "i,sub", 10), (")", "", 10)]], align=PP_ALIGN.LEFT)
text(SX, 3.68, 0.3, 0.2, [[("s", "i", 9.5), ("1", "sub", 9.5)]])
box(SX + 0.32, 3.72, 0.12, 0.13, None, pal=("E6913A", "E6913A"), radius=0.01, lw=0.5)
text(SX, 3.9, 0.3, 0.2, [[("s", "i", 9.5), ("2", "sub", 9.5)]])
box(SX + 0.32, 3.94, 1.25, 0.13, None, pal=("4278C8", "4278C8"), radius=0.01, lw=0.5)
arrow(VX + 1.34, 3.88, SX, 3.88)
# losses column
LXc, LWc = 9.55, 2.05
box(LXc, 4.3, LWc, 0.55, [[("ℒ", "", 10.5, RED), ("list", "sub", 10.5, RED), ("  listwise competition", "", 9, RED)], [("softmax over ", "", 8, RED), ("C", "i", 8, RED), ("(", "", 8, RED), ("q", "i", 8, RED), ("); raise mass on positives ", "", 8, RED), ("P", "i", 8, RED)]], pal=LOSS, dash=True)
box(LXc, 5.0, LWc, 0.55, [[("ℒ", "", 10.5, RED), ("hard", "sub", 10.5, RED), ("  hard-negative margin", "", 9, RED)], [("softplus(", "", 8, RED), ("r", "i", 8, RED), ("j", "i,sub", 8, RED), (" − ", "", 8, RED), ("r", "i", 8, RED), ("+", "sup", 8, RED), (" + δ),  ", "", 8, RED), ("r", "i", 8, RED), (" = ℓ", "", 8, RED), ("yes", "sub", 8, RED), (" − ℓ", "", 8, RED), ("no", "sub", 8, RED)]], pal=LOSS, dash=True)
box(LXc, 5.7, LWc, 0.55, [[("ℒ", "", 10.5, RED), ("align", "sub", 10.5, RED), ("  mask-aware alignment", "", 9, RED)], [("1 − Σ", "", 8, RED), ("n", "i,sub", 8, RED), (" p", "i", 8, RED), ("n", "i,sub", 8, RED), ("b", "i", 8, RED), ("n", "i,sub", 8, RED), ("  (text–visual hidden sim. in mask)", "", 8, RED)]], pal=LOSS, dash=True)
poly([(9.42, 4.07), (9.42, 4.57), (LXc, 4.57)], color=RED, dash=True)
poly([(9.42, 4.57), (9.42, 5.27), (LXc, 5.27)], color=RED, dash=True)
poly([(VX + 1.34, 5.97), (LXc, 5.97)], color=RED, dash=True)
text(VX + 1.3, 5.72, 0.6, 0.22, [[("hidden", "i", 7.5, RED)]])
text(X0, 6.55, 5.7, 0.4, [[("Candidates are cached offline and labelled by IoU with GT; no gradient flows through SAM3.", "i", 8.5, SUB)]])

# captions & legend
text(0.15, 7.02, 5.6, 0.3, [[("(b) Ground training: target-query grounding", "b", 12)]])
text(5.95, 7.02, 5.7, 0.3, [[("(c) Verify training: candidate competition", "b", 12)]])
lx = 3.2
box(lx, 7.47, 0.28, 0.18, None, pal=GREY, lw=0.8, radius=0.04); text(lx + 0.3, 7.42, 1.0, 0.28, [[("❄", "emoji", 9.5), (" frozen", "", 9.5, SUB)]], align=PP_ALIGN.LEFT)
lx = 4.7
box(lx, 7.47, 0.28, 0.18, None, pal=TRAIN, lw=0.8, radius=0.04); text(lx + 0.3, 7.42, 1.2, 0.28, [[("🔥", "emoji", 9.5), (" trainable", "", 9.5, SUB)]], align=PP_ALIGN.LEFT)
lx = 6.35
box(lx, 7.47, 0.28, 0.18, None, pal=LOSS, dash=True, lw=0.8, radius=0.04); text(lx + 0.3, 7.42, 2.3, 0.28, [[("training-only supervision", "", 9.5, SUB)]], align=PP_ALIGN.LEFT)

prs.save(OUT)
print("saved", OUT)
