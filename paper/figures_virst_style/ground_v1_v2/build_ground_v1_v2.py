"""Ground v1 (left) vs Ground v2 (right) training architecture, VIRST-style, editable PPTX.

  python build_ground_v1_v2.py ground_v1_vs_v2.pptx
"""
import sys, os
sys.path.insert(0, "/mnt/project-files/论文/framework_virst_style/assets")
from common import *  # noqa  (creates prs / SH with helpers)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1]
prs.slide_width = Inches(13.3)
prs.slide_height = Inches(7.15)
TRAIN = GREEN
LOSS = ("FFF4F4", RED)
FIXG = "4E8A3A"


def ice(x, y):
    text(x - 0.3, y - 0.12, 0.4, 0.26, [[("❄", "emoji", 10, "5A7FA8")]], wrap=False)


def fire(x, y):
    text(x - 0.3, y - 0.12, 0.4, 0.26, [[("🔥", "emoji", 10, "D2552B")]], wrap=False)


def mark(x, y, n, good=False, note=None, nw=1.3, left=False):
    col = FIXG if good else RED
    box(x, y, 0.23, 0.23, [[(n, "b", 9, col)]], pal=("FFFFFF", col), shape=MSO_SHAPE.OVAL, lw=1.0)
    if note:
        if left:
            text(x - nw - 0.03, y - 0.03, nw, 0.29, [[(note, "i", 8.5, col)]], align=PP_ALIGN.RIGHT)
        else:
            text(x + 0.26, y - 0.03, nw, 0.29, [[(note, "i", 8.5, col)]], align=PP_ALIGN.LEFT)


def frame(x, y, w, h):
    box(x, y, w, h, None, pal=("FCFCFC", "B5B5B5"), dash=True, lw=0.9, radius=0.12)


def L(name, size=11, col=RED):
    return [[("ℒ", "", size, col), (name, "sub", size, col)]]


# ===================================================================== (a) original Ground
frame(0.15, 0.25, 6.2, 6.05)
# --- column geometry
LC0, LC1 = 0.35, 1.65          # left column (vision)
MC0, MC1 = 2.0, 4.35           # middle column (queries)
RC0, RC1 = 4.7, 5.95           # right column (expression)
lcx, mcx, rcx = (LC0 + LC1) / 2, (MC0 + MC1) / 2, (RC0 + RC1) / 2

# inputs
img(HERE + "/frames_mm.png", 0.4, 5.15, w=1.2)
text(0.3, 5.92, 1.4, 0.24, [[("Frames ", "", 9.5), ("(target visible)", "i", 8.5, SUB)]])
mark(1.68, 5.4, "③")
qx = tokens(4.97, 5.75, 3, ORANGE, dots=False)
text(RC0, 5.95, RC1 - RC0, 0.24, [[("Expression ", "", 9.5), ("q", "i", 9.5)]])
tokens(2.33, 5.75, 6, TRAIN)
text(MC0, 5.95, MC1 - MC0, 0.24, [[("Queries ", "", 9.5), ("Q", "i", 9.5), ("0", "sup", 9.5), (" ×8", "", 9.5)]])

# vision encoder
box(LC0, 4.34, LC1 - LC0, 0.44, ["Vision Encoder"], pal=GREY, size=10); ice(LC1, 4.34)
arrow(lcx, 5.22, lcx, 4.78)
# expression branch
box(RC0, 5.2, RC1 - RC0, 0.3, ["Word Embedding"], pal=GREY, size=9.5, radius=0.08); ice(RC1, 5.2)
arrow(rcx, 5.75, rcx, 5.5)
box(RC0, 4.64, RC1 - RC0, 0.42, ["Expression Encoder"], pal=TRAIN, size=10); fire(RC1, 4.64)
arrow(rcx, 5.2, rcx, 5.06)

# query decoder
box(MC0, 3.95, MC1 - MC0, 1.5, None, pal=("F2F7EE", "6E9A5B"), radius=0.12); fire(MC1, 3.95)
text(MC0, 3.96, MC1 - MC0, 0.22, [[("Query Decoder ×2", "", 9, SUB)]])
for name, yy, hh in [("Self-Attention", 5.03, 0.22), ("Text Cross-Attn", 4.74, 0.22),
                     ("Visual Cross-Attn", 4.45, 0.22), ("FFN", 4.2, 0.17)]:
    box(MC0 + 0.14, yy, MC1 - MC0 - 0.28, hh, [name], pal=TRAIN, size=9 if name != "FFN" else 8.5, radius=0.05)
arrow(mcx, 5.75, mcx, 5.25)
for a, b in [(5.03, 4.96), (4.74, 4.67), (4.45, 4.37)]:
    arrow(mcx, a, mcx, b)
arrow(LC1, 4.56, MC0 + 0.14, 4.56)
text(LC1 - 0.02, 4.32, 0.3, 0.22, [[("V", "i", 9.5)]])
arrow(RC0, 4.85, MC1 - 0.14, 4.85)
text(MC1 + 0.0, 4.6, 0.3, 0.22, [[("T", "i", 9.5)]])
mark(0.3, 3.82, "①", note="pre-LLM features", nw=1.5)

# responses + selector
box(MC0, 3.35, MC1 - MC0, 0.34, [[("Responses ", "", 10), ("R", "i", 10), ("1", "sub", 10), (" … ", "", 10),
                                   ("R", "i", 10), ("8", "sub", 10)]], pal=TRAIN, size=10); fire(MC1, 3.35)
arrow(mcx, 3.95, mcx, 3.69)
box(MC0, 2.75, MC1 - MC0, 0.36, [[("σ(", "", 10, RED), ("R", "i", 10, RED), ("k*", "i,sub", 10, RED), (")", "", 10, RED),
                                   ("   |   max", "", 10), ("k", "i,sub", 10), (" σ(", "", 10), ("R", "i", 10), ("k", "i,sub", 10), (")", "", 10)]],
    pal=WHITE, size=10)
arrow(mcx, 3.35, mcx, 3.11)
mark(MC0 - 0.3, 2.82, "②", note="1 of 8 supervised", nw=1.5, left=True)
text(MC0, 2.53, 1.2, 0.2, [[("train", "i", 8, RED)]], align=PP_ALIGN.CENTER) if False else None
# ground loss
box(RC0, 2.75, RC1 - RC0, 0.36, L("ground"), pal=LOSS, dash=True)
arrow(MC1, 2.93, RC0, 2.93, color=RED, dash=True)
text(RC0 - 0.05, 3.13, RC1 - RC0 + 0.1, 0.22, [[("BCE + Dice", "", 8.5, SUB)]])
mark(RC0 + 0.05, 3.42, "③", note="uniform negatives", nw=1.25)

# soft modulation
box(MC0 + 0.25, 2.1, MC1 - MC0 - 0.5, 0.42, ["Soft Modulation"], pal=TRAIN, size=10.5); fire(MC1 - 0.25, 2.1)
arrow(mcx, 2.75, mcx, 2.52)
text(mcx + 0.02, 2.5, 0.3, 0.24, [[("p", "i", 9.5)]], align=PP_ALIGN.LEFT)
poly([(lcx, 4.34), (lcx, 2.31), (MC0 + 0.25, 2.31)])
text(LC0 - 0.05, 2.04, 1.5, 0.26, [[("V", "i", 9.5), (" + deepstack", "", 8.5, SUB)]])
mark(MC1 - 0.12, 1.9, "④", note="rescales every V token", nw=1.65)

# frozen LLM
box(1.0, 1.28, 4.3, 0.5, [[("Qwen3-VL LLM (36 layers)", "", 11)]], pal=GREY); ice(5.3, 1.28)
arrow(mcx, 2.1, mcx, 1.78)
text(mcx + 0.02, 1.8, 0.4, 0.26, [[("V", "i", 9.5), ("′", "", 9.5)]], align=PP_ALIGN.LEFT)
poly([(qx, 5.835), (6.15, 5.835), (6.15, 1.53), (5.3, 1.53)])
text(5.85, 1.27, 0.3, 0.24, [[("q", "i", 9.5)]])
mark(0.7, 1.41, "⑤", note=None)
text(0.12, 1.66, 0.85, 0.24, [[("frozen", "i", 8.5, RED)]])
# category word
box(2.45, 0.6, 1.45, 0.34, [[("ĉ", "i", 11), (" = “person”", "", 10)]], pal=YELLOW, radius=0.16)
arrow(3.175, 1.28, 3.175, 0.94)
box(4.35, 0.6, 0.95, 0.34, L("cat"), pal=LOSS, dash=True)
arrow(3.9, 0.77, 4.35, 0.77, color=RED, dash=True)

# ===================================================================== (b) Ground v2
frame(6.6, 0.25, 6.55, 6.05)
BX0, BX1 = 6.8, 10.2           # LLM blocks
TGX = 9.38                     # x of <TGT> column
# inputs
img(HERE + "/frames_mm.png", 7.88, 5.38, w=1.05)
mark(9.0, 5.62, "③", good=True)
text(9.26, 5.55, 1.6, 0.44, [[("Frames ", "", 9.5)], [("(+20% target-absent)", "i", 8.5, SUB)]], align=PP_ALIGN.LEFT)
box(7.8, 4.82, 1.22, 0.36, ["Vision Encoder"], pal=GREY, size=9.5, radius=0.08); ice(9.02, 4.82)
arrow(8.41, 5.4, 8.41, 5.18)
# sequence row
SY = 4.36
tokens(6.95, SY, 3, ORANGE, dots=False)
tokens(7.85, SY, 5, BLUE, dots=True)
tokens(9.17, SY, 2, PURPLE, dots=False)
tokens(9.75, SY, 2, YELLOW, dots=False)
text(6.85, 4.56, 0.75, 0.24, [[("q", "i", 9.5)]])
text(9.0, 4.56, 0.82, 0.24, [[("⟨TGT⟩×2", "", 8.5)]])
text(9.65, 4.56, 0.55, 0.24, [[("ĉ", "i", 9.5), (" (GT)", "", 8, SUB)]], align=PP_ALIGN.LEFT)
arrow(8.41, 4.82, 8.41, SY + 0.17)
text(6.68, 4.83, 1.2, 0.44, [[("“a man giving a hand", "i", 8, SUB)], [("to a monkey …”", "i", 8, SUB)]])

# lower LLM block (layers 1..L)
box(BX0, 3.35, BX1 - BX0, 0.6, None, pal=GREY)
text(BX0 + 0.05, 3.35, 2.3, 0.6, [[("Qwen3-VL LLM", "", 10.5)], [("layers 1 … ", "", 9.5, SUB), ("L", "i", 9.5, SUB)]], align=PP_ALIGN.CENTER)
ice(BX1 - 0.86, 3.35)
box(BX1 - 0.82, 3.47, 0.7, 0.36, ["LoRA"], pal=TRAIN, size=9.5, radius=0.07); fire(BX1 - 0.12, 3.47)
for xx in (7.25, 8.41, TGX, 9.97):
    arrow(xx, SY, xx, 3.95)

# upper LLM block (layers L+1..36)
box(BX0, 1.55, BX1 - BX0, 0.6, None, pal=GREY)
text(BX0 + 0.05, 1.55, 2.3, 0.6, [[("Qwen3-VL LLM", "", 10.5)], [("layers ", "", 9.5, SUB), ("L", "i", 9.5, SUB), ("+1 … 36", "", 9.5, SUB)]], align=PP_ALIGN.CENTER)
ice(BX1 - 0.86, 1.55)
box(BX1 - 0.82, 1.67, 0.7, 0.36, ["LoRA"], pal=TRAIN, size=9.5, radius=0.07); fire(BX1 - 0.12, 1.67)
mark(6.72, 2.18, "⑤", good=True, note="LoRA on LLM", nw=1.1)
arrow(7.9, 3.35, 7.9, 2.15)
text(7.93, 2.6, 0.9, 0.26, [[("h", "i", 9.5), ("L", "i,sup", 9.5)]], align=PP_ALIGN.LEFT)

# summary write-back into <TGT>
OX, OY, OR = TGX, 2.75, 0.14
box(OX - OR, OY - OR, 2 * OR, 2 * OR, [[("+", "b", 11)]], pal=("FFFFFF", "404040"), shape=MSO_SHAPE.OVAL, lw=1.0)
arrow(OX, 3.35, OX, OY + OR)
arrow(OX, OY - OR, OX, 2.15)
box(9.78, 2.54, 0.95, 0.42, [[("Summary", "", 9.5)], [("MLP", "", 9.5)]], pal=TRAIN, size=9.5); fire(10.73, 2.54)
arrow(9.78, OY, OX + OR, OY)
text(9.75, 2.96, 1.05, 0.22, [[("mean · top-", "", 8, SUB), ("k", "i", 8, SUB)]])
mark(9.95, 2.24, "④", good=True, note="summary → ⟨TGT⟩", nw=1.12)

# side branch: single-query decoder
DX0, DX1 = 10.85, 12.15
arrow(BX1, 3.65, DX0, 3.65)
text(BX1 + 0.0, 3.68, 0.7, 0.24, [[("E", "i", 8.5), ("L", "sub", 8.5), (", ", "", 8.5), ("V", "i", 8.5), ("L", "sub", 8.5),
                                    (", ", "", 8.5), ("q", "i", 8.5), ("0", "sub", 8.5)]])
text(BX1 - 0.05, 3.35, 0.75, 0.24, [[("layer-", "", 8, SUB), ("L", "i", 8, SUB), (" hook", "", 8, SUB)]])
box(DX0, 3.35, DX1 - DX0, 0.6, [[("Single-Query", "", 9.5)], [("Decoder", "", 9.5)]], pal=TRAIN); fire(DX1, 3.35)
mark(10.38, 4.02, "①", good=True, note="post-reading features", nw=1.45)
mark(12.2, 3.55, "②", good=True)
text(12.18, 3.76, 0.9, 0.22, [[("1 query", "i", 8.5, FIXG)]], align=PP_ALIGN.LEFT)

# heatmap
HX = 10.98
img(HERE + "/heat_v2_mm.png", HX, 2.48, w=0.95)
arrow(11.455, 3.35, 11.455, 3.02)
text(11.95, 2.3, 0.3, 0.24, [[("p", "i", 10)]], align=PP_ALIGN.LEFT)
arrow(HX, OY, 10.73, OY)
text(10.7, 2.5, 0.3, 0.22, [[("sg", "i", 8, SUB)]])
# to recall
arrow(11.455, 2.48, 11.455, 1.95)
text(10.9, 1.62, 1.15, 0.3, [[("to SAM3 Recall", "", 8.5, SUB)]])
# mask loss
box(12.25, 2.58, 0.8, 0.34, L("mask", 10.5), pal=LOSS, dash=True)
arrow(11.93, 2.75, 12.25, 2.75, color=RED, dash=True)
text(12.05, 2.95, 1.1, 0.4, [[("tgt 1 · other-cat 3", "", 7.5, SUB)], [("same-cat 0", "", 7.5, SUB)]])
mark(12.6, 2.22, "③", good=True)

# category word
box(7.18, 0.6, 1.45, 0.34, [[("ĉ", "i", 11), (" = “person”", "", 10)]], pal=YELLOW, radius=0.16)
arrow(7.9, 1.55, 7.9, 0.94)
box(9.05, 0.6, 0.95, 0.34, L("cat"), pal=LOSS, dash=True)
arrow(8.63, 0.77, 9.05, 0.77, color=RED, dash=True)
text(10.15, 0.55, 2.9, 0.44, [[("one forward pass: ", "i", 8.5, SUB), ("q", "i", 8.5, SUB), (" → frames → ⟨TGT⟩ → ", "", 8.5, SUB), ("ĉ", "i", 8.5, SUB)]],
     align=PP_ALIGN.LEFT)

# ===================================================================== captions & legend
text(0.15, 6.35, 6.2, 0.3, [[("(a) Original Ground (training)", "b", 12)]])
text(6.6, 6.35, 6.55, 0.3, [[("(b) Ground v2 (training)", "b", 12)]])
ly = 6.8
items = [(GREY, False, [("❄", "emoji", 9), (" frozen", "", 9, SUB)], 0.9),
         (TRAIN, False, [("🔥", "emoji", 9), (" trainable", "", 9, SUB)], 1.0),
         (LOSS, True, [("training-only loss", "", 9, SUB)], 1.4)]
lx = 2.6
for pal, dsh, lab, w in items:
    box(lx, ly + 0.05, 0.28, 0.17, None, pal=pal, dash=dsh, lw=0.8, radius=0.04)
    text(lx + 0.31, ly, w, 0.27, [lab], align=PP_ALIGN.LEFT)
    lx += 0.31 + w + 0.25
box(lx, ly + 0.02, 0.22, 0.22, [[("n", "b", 8, RED)]], pal=("FFFFFF", RED), shape=MSO_SHAPE.OVAL, lw=0.9)
text(lx + 0.25, ly, 1.25, 0.27, [[("problem in (a)", "", 9, SUB)]], align=PP_ALIGN.LEFT)
lx += 1.6
box(lx, ly + 0.02, 0.22, 0.22, [[("n", "b", 8, FIXG)]], pal=("FFFFFF", FIXG), shape=MSO_SHAPE.OVAL, lw=0.9)
text(lx + 0.25, ly, 1.6, 0.27, [[("its fix in (b)", "", 9, SUB)]], align=PP_ALIGN.LEFT)

# no shadows anywhere (LibreOffice otherwise draws the theme effect under text and boxes)
for sp in SH:
    for er in sp._element.iter(qn("a:effectRef")):
        er.set("idx", "0")
prs.save(OUT)
print("saved", OUT)
