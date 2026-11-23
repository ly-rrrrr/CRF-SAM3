from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.oxml.ns import qn
from lxml import etree
import sys

OUT = sys.argv[1]
ASSET = "/mnt/project-files/论文/framework_virst_style/assets/"
FONT = "Times New Roman"

def C(h): return RGBColor.from_string(h)
# palette (fill, line)
GREY = ("EFEFEF", "8C8C8C")
GREEN = ("DDEBD3", "6E9A5B")
PURPLE = ("ECE1F3", "9474AB")
ORANGE = ("F9DFCC", "C98656")
BLUE = ("D9E4F3", "6C8DB8")
YELLOW = ("FFF2C6", "BE9A2A")
WHITE = ("FFFFFF", "8C8C8C")
RED = "B24A4A"
TXT = "1F1F1F"
SUB = "555555"

prs = Presentation()
prs.slide_width = Inches(13.33)
prs.slide_height = Inches(5.8)
s = prs.slides.add_slide(prs.slide_layouts[6])
SH = s.shapes

def I(v): return Inches(v)

def runs(tf, lines, size=11, align=PP_ALIGN.CENTER, color=TXT):
    """lines: list of lines; each line a str or list of (text, flags) where flags in {'i','b','sub','sup'} and optional size override."""
    tf.clear()
    for li, line in enumerate(lines):
        p = tf.paragraphs[0] if li == 0 else tf.add_paragraph()
        p.alignment = align
        segs = [(line, "")] if isinstance(line, str) else line
        for seg in segs:
            text, flags = seg[0], set(seg[1].split(","))
            sz = seg[2] if len(seg) > 2 else size
            col = seg[3] if len(seg) > 3 else color
            r = p.add_run()
            r.text = text
            f = r.font
            f.name = FONT
            f.size = Pt(sz)
            f.italic = "i" in flags
            f.bold = "b" in flags
            f.color.rgb = C(col)
            rPr = r._r.get_or_add_rPr()
            for tag in ("a:latin", "a:ea", "a:cs"):
                el = rPr.find(qn(tag))
                if el is None:
                    el = etree.SubElement(rPr, qn(tag))
                el.set("typeface", FONT)
            if "sub" in flags: rPr.set("baseline", "-25000")
            if "sup" in flags: rPr.set("baseline", "30000")

def box(x, y, w, h, lines, pal=WHITE, size=12, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
        dash=False, lw=1.0, radius=0.18, color=TXT, align=PP_ALIGN.CENTER, nofill=False, noline=False, anchor=MSO_ANCHOR.MIDDLE):
    sp = SH.add_shape(shape, I(x), I(y), I(w), I(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        sp.adjustments[0] = min(0.5, radius / min(w, h))
    sp.shadow.inherit = False
    if nofill: sp.fill.background()
    else:
        sp.fill.solid(); sp.fill.fore_color.rgb = C(pal[0])
    if noline: sp.line.fill.background()
    else:
        sp.line.color.rgb = C(pal[1]); sp.line.width = Pt(lw)
        if dash: sp.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    tf = sp.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = I(0.04)
    tf.margin_top = tf.margin_bottom = I(0.02)
    tf.vertical_anchor = anchor
    if lines: runs(tf, lines, size, align, color)
    return sp

def text(x, y, w, h, lines, size=11, align=PP_ALIGN.CENTER, color=TXT, anchor=MSO_ANCHOR.MIDDLE):
    return box(x, y, w, h, lines, size=size, nofill=True, noline=True, align=align, color=color, anchor=anchor)

def _arrowhead(line_el, head=True, tail=False):
    ln = line_el
    if tail:
        e = etree.SubElement(ln, qn("a:headEnd")); e.set("type", "triangle"); e.set("w", "med"); e.set("len", "med")
    if head:
        e = etree.SubElement(ln, qn("a:tailEnd")); e.set("type", "triangle"); e.set("w", "med"); e.set("len", "med")

def arrow(x1, y1, x2, y2, color="404040", lw=1.1, dash=False, head=True):
    c = SH.add_connector(MSO_CONNECTOR.STRAIGHT, I(x1), I(y1), I(x2), I(y2))
    c.line.color.rgb = C(color); c.line.width = Pt(lw)
    if dash: c.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    ln = c.line._get_or_add_ln()
    _arrowhead(ln, head=head)
    return c

def poly(pts, color="404040", lw=1.1, dash=False, head=True):
    fb = SH.build_freeform(I(pts[0][0]), I(pts[0][1]), scale=1.0)
    fb.add_line_segments([(I(x), I(y)) for x, y in pts[1:]], close=False)
    sp = fb.convert_to_shape()
    sp.fill.background()
    sp.shadow.inherit = False
    sp.line.color.rgb = C(color); sp.line.width = Pt(lw)
    if dash: sp.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    if head: _arrowhead(sp.line._get_or_add_ln())
    return sp

def tokens(x, y, n, pal, w=0.17, gap=0.06, h=None, dots=True):
    h = h or w
    xs = x
    for i in range(n):
        if dots and i == n - 1:
            text(xs - 0.02, y - 0.05, 0.25, h + 0.05, ["…"], size=11)
            xs += 0.23
        box(xs, y, w, h, None, pal=pal, lw=0.8, radius=0.04)
        xs += w + gap
    return xs - gap  # right edge

def img(path, x, y, w=None, h=None):
    return SH.add_picture(path, I(x), I(y), I(w) if w else None, I(h) if h else None)

# ------------------------------------------------------------------ panel (a)
# stage backgrounds
STAGES = [(0.25, 3.6, "Ground", "6E9A5B"), (3.95, 2.07, "Recall", "6C8DB8"), (6.12, 2.28, "Verify", "9474AB")]
for x, w, name, col in STAGES:
    box(x, 0.28, w, 4.72, None, pal=("FBFBFB", "B5B5B5"), dash=True, lw=0.9, radius=0.12)
    text(x + 0.08, 0.32, 1.4, 0.3, [[(name, "b,i", 13, col)]], align=PP_ALIGN.LEFT)

# --- Ground (bottom -> top)
# inputs
text(0.35, 4.13, 1.75, 0.25, [[("Text Query ", "", 11), ("q", "i", 11)]])
text(0.3, 4.36, 1.85, 0.55, [[("“Which joystick is being held by hands at the end of the video?”", "i", 8.5, SUB)]])
img(ASSET + "input_video.png", 2.16, 4.13, w=1.38)
text(2.16, 4.73, 1.38, 0.25, None)
# encoders
box(0.42, 3.42, 1.55, 0.5, ["Expression", "Conditioner"], pal=GREEN, size=11)
tz = box(2.1, 3.42, 1.5, 0.5, ["Qwen3-VL", "Vision Encoder"], pal=GREY, size=11, shape=MSO_SHAPE.TRAPEZOID)
tz.adjustments[0] = 0.14
arrow(1.2, 4.15, 1.2, 3.92)
arrow(2.85, 4.18, 2.85, 3.92)
# token rows
re = tokens(0.62, 3.06, 5, ORANGE)
text(re + 0.02, 3.02, 0.3, 0.25, [[("T", "i", 11)]], align=PP_ALIGN.LEFT)
re = tokens(2.22, 3.06, 5, BLUE)
text(re + 0.02, 3.02, 0.3, 0.25, [[("V", "i", 11)]], align=PP_ALIGN.LEFT)
arrow(1.2, 3.42, 1.2, 3.25)
arrow(2.85, 3.42, 2.85, 3.25)
# grounder
box(0.42, 2.3, 3.18, 0.56, [[("Target-Query Grounder", "", 13)], [("mask-supervised (training only) · soft token modulation", "", 8.5, SUB)]], pal=GREEN)
box(3.33, 2.33, 0.24, 0.2, [[("b", "b", 8, "6E9A5B")]], pal=("FFFFFF", "6E9A5B"), shape=MSO_SHAPE.OVAL, lw=0.8)
arrow(1.2, 3.06, 1.2, 2.86)
arrow(2.85, 3.06, 2.85, 2.86)
# modulated tokens
re = tokens(1.35, 1.94, 6, ("CFE3D9", "6C8DB8"))
text(re + 0.02, 1.9, 0.35, 0.25, [[("V", "i", 11), ("′", "", 11)]], align=PP_ALIGN.LEFT)
arrow(2.0, 2.3, 2.0, 2.11)
# decoder
box(0.42, 1.38, 3.18, 0.44, ["Qwen3-VL Language Decoder"], pal=GREY, size=12)
arrow(2.0, 1.94, 2.0, 1.82)
# concept
box(1.0, 0.82, 2.02, 0.4, [[("Recall concept ", "", 11), ("ĉ", "i", 11), (" = “joystick”", "", 11)]], pal=YELLOW, radius=0.2)
arrow(2.0, 1.38, 2.0, 1.22)

# --- Recall (top -> bottom)
box(4.17, 0.72, 1.55, 0.6, [[("SAM3", "", 15)], [("recall 24 → keep 7", "", 8.5, SUB)]], pal=GREY)
arrow(3.02, 1.02, 4.17, 1.02)
text(3.1, 0.74, 1.05, 0.25, [[("ĉ", "i", 10), (" + video", "", 10)]])
arrow(4.945, 1.32, 4.945, 1.52)
text(3.95, 1.52, 1.95, 0.25, [[("Candidate Tubes ", "", 10.5), ("C", "i", 10.5), ("(", "", 10.5), ("q", "i", 10.5), (")", "", 10.5)]])
img(ASSET + "cand_R0.png", 4.38, 1.82, w=1.48)
img(ASSET + "cand_R1.png", 4.38, 2.83, w=1.48)
text(3.96, 2.05, 0.42, 0.35, [[("τ", "i", 13), ("1", "sub", 13)]])
text(3.96, 3.06, 0.42, 0.35, [[("τ", "i", 13), ("2", "sub", 13)]])
text(4.38, 3.73, 1.48, 0.25, ["⋮"], size=12)
text(3.97, 4.0, 1.9, 0.85, [[("Same-category hypotheses are kept; ambiguity is resolved later.", "i", 8.5, SUB)]])
# brace to verify
poly([(5.9, 1.95), (5.98, 1.95), (5.98, 3.62), (5.9, 3.62)], color="808080", lw=0.9, head=False)
poly([(5.98, 2.78), (6.22, 2.78), (6.22, 4.17), (6.38, 4.17)])

# --- Verify (bottom -> top)
box(6.38, 3.93, 1.06, 0.5, ["Candidate", "Marking"], pal=WHITE, size=10.5)
box(7.52, 3.93, 0.74, 0.5, [[("Full", "", 10)], [("expr. ", "", 10), ("q", "i", 10)]], pal=ORANGE, size=10, dash=True)
tokens(6.42, 3.53, 4, BLUE)
tokens(7.6, 3.53, 3, ORANGE, dots=False)
arrow(6.9, 3.93, 6.9, 3.71)
arrow(7.89, 3.93, 7.89, 3.71)
box(6.38, 2.78, 1.88, 0.56, [[("Qwen3-VL Verifier", "", 12)], [("LoRA · listwise + hard-neg + align", "", 8, SUB)]], pal=PURPLE)
arrow(7.32, 3.53, 7.32, 3.34)
# scores
text(6.32, 2.25, 0.6, 0.22, [[("s", "i", 10), ("1", "sub", 10)]], align=PP_ALIGN.RIGHT)
box(6.95, 2.29, 0.06, 0.14, None, pal=("E6913A", "E6913A"), radius=0.01, lw=0.5)
text(7.03, 2.25, 0.5, 0.22, ["0.05"], size=9, align=PP_ALIGN.LEFT, color=SUB)
text(6.32, 2.48, 0.6, 0.22, [[("s", "i", 10), ("2", "sub", 10)]], align=PP_ALIGN.RIGHT)
box(6.95, 2.52, 1.0, 0.14, None, pal=("4278C8", "4278C8"), radius=0.01, lw=0.5)
text(7.97, 2.48, 0.4, 0.22, ["0.95"], size=9, align=PP_ALIGN.LEFT, color=SUB)
arrow(7.32, 2.78, 7.32, 2.7, head=False)
arrow(7.32, 2.25, 7.32, 2.02)
text(7.36, 2.02, 0.9, 0.22, [[("arg max", "", 9, SUB)]], align=PP_ALIGN.LEFT)
img(ASSET + "output_mask.png", 6.5, 1.15, w=1.5)
text(6.15, 0.75, 2.2, 0.3, [[("Segmentation Mask ", "", 11), ("Y", "i", 11)]])
arrow(7.32, 2.02, 7.32, 1.98, head=False)

# ------------------------------------------------------------------ divider
poly([(8.6, 0.25), (8.6, 5.0)], color="404040", lw=1.2, dash=True, head=False)

# ------------------------------------------------------------------ panel (b)
LX = 8.95; LW = 1.85; CX = LX + LW / 2
# inputs row
q_right = tokens(9.15, 4.32, 5, GREEN)
text(8.85, 4.55, 2.1, 0.45, [[("Target Queries ", "", 10), ("Q", "i", 10), ("0", "sup", 10)], [("(K = 8, category-agnostic)", "", 8.5, SUB)]])
tokens(11.15, 4.32, 3, ORANGE, dots=False)
text(10.95, 4.55, 1.1, 0.45, [[("Text Feat. ", "", 10), ("T", "i", 10)]])
tokens(12.08, 4.32, 4, BLUE)
text(11.95, 4.55, 1.25, 0.45, [[("Visual Tokens ", "", 10), ("V", "i", 10)]])
# decoder
box(LX, 2.33, LW, 1.8, None, pal=("F2F7EE", "6E9A5B"), radius=0.14)
text(LX, 2.33, LW, 0.24, [[("Query Decoder ×", "", 9.5, SUB), ("L", "i", 9.5, SUB)]])
ops = [("Self-Attention", 3.72), ("Text Cross-Attn", 3.32), ("Visual Cross-Attn", 2.92), ("FFN", 2.6)]
for name, yy in ops[:3]:
    box(LX + 0.15, yy, LW - 0.3, 0.3, [name], pal=GREEN, size=10.5, radius=0.08)
box(LX + 0.15, 2.6, LW - 0.3, 0.24, ["FFN"], pal=GREEN, size=10, radius=0.08)
arrow(CX, 4.32, CX, 4.02)
arrow(CX, 3.72, CX, 3.62)
arrow(CX, 3.32, CX, 3.22)
arrow(CX, 2.92, CX, 2.84)
# K,V from T and V
poly([(11.5, 4.32), (11.5, 3.47), (LX + LW - 0.15, 3.47)])
text(11.0, 3.2, 0.5, 0.25, [[("K,V", "", 9, SUB)]])
poly([(12.45, 4.32), (12.45, 3.07), (LX + LW - 0.15, 3.07)])
text(11.0, 2.8, 0.5, 0.25, [[("K,V", "", 9, SUB)]])
# Q^L
tokens(9.15, 2.0, 5, GREEN)
text(8.72, 1.96, 0.42, 0.25, [[("Q", "i", 10), ("L", "sup", 10)]], align=PP_ALIGN.RIGHT)
arrow(CX, 2.33, CX, 2.17)
# response head
box(LX, 1.38, LW, 0.42, [[("Response ", "", 10.5), ("R", "i", 10.5), ("k,n", "i,sub", 10.5)], [("γ · cos(", "", 8.5, SUB), ("W", "i", 8.5, SUB), ("q", "i,sub", 8.5, SUB), ("q", "i", 8.5, SUB), ("k", "i,sub", 8.5, SUB), (", ", "", 8.5, SUB), ("W", "i", 8.5, SUB), ("v", "i,sub", 8.5, SUB), ("v̄", "i", 8.5, SUB), ("n", "i,sub", 8.5, SUB), (")", "", 8.5, SUB)]], pal=WHITE, size=10)
arrow(CX, 2.0, CX, 1.8)
# V bus to response and modulation
poly([(12.95, 4.32), (12.95, 0.55), (LX + LW, 0.55)])
arrow(12.95, 1.59, LX + LW, 1.59)
# GT supervision (training only)
box(11.12, 1.93, 1.55, 0.52, [[("GT mask → token labels ", "", 9, RED), ("y", "i", 9, RED)], [("BCE + Dice on best slot ", "", 8, RED), ("k", "i", 8, RED), ("*", "sup", 8, RED)]], pal=("FFF5F5", RED), dash=True, size=9)
poly([(11.12, 2.19), (10.62, 2.19), (10.62, 1.8)], color=RED, dash=True)
text(11.12, 2.45, 1.55, 0.22, [[("training only", "i", 8, RED)]])
# gate
box(LX, 0.92, LW, 0.32, [[("m", "i", 10), ("n", "i,sub", 10), (" = max", "", 10), ("k", "i,sub", 10), (" σ(", "", 10), ("R", "i", 10), ("k,n", "i,sub", 10), (")", "", 10)]], pal=WHITE, size=10)
text(LX + LW + 0.03, 0.95, 1.6, 0.27, [[("inference: max over slots", "i", 8, SUB)]], align=PP_ALIGN.LEFT)
arrow(CX, 1.38, CX, 1.24)
# soft modulation
box(LX, 0.38, LW, 0.4, [[("Soft Modulation", "", 10.5)], [("v", "i", 9), ("′", "", 9), (" = ", "", 9), ("v", "i", 9), (" + α ", "", 9), ("m", "i", 9), (" ", "", 9), ("A", "i", 9), ("(", "", 9), ("v", "i", 9), (")", "", 9)]], pal=GREEN)
arrow(CX, 0.92, CX, 0.78)
arrow(CX, 0.38, CX, 0.16)
text(CX + 0.05, 0.08, 2.6, 0.25, [[("to frozen Qwen3-VL decoder (no crop / no pruning)", "i", 8.5, SUB)]], align=PP_ALIGN.LEFT)

# ------------------------------------------------------------------ captions & legend
text(2.3, 5.06, 4.0, 0.3, [[("(a) Overall Architecture", "b", 13)]])
text(9.0, 5.06, 4.0, 0.3, [[("(b) Target-Query Grounding", "b", 13)]])
for lx, pal, lab, dash in [(3.0, GREY, "frozen", False), (4.35, GREEN, "trainable (new modules)", False), (6.6, PURPLE, "trainable (LoRA)", False), (8.45, ("FFF5F5", RED), "training-only supervision", True)]:
    box(lx, 5.45, 0.28, 0.18, None, pal=pal, dash=dash, lw=0.8, radius=0.04)
    text(lx + 0.33, 5.4, 2.2, 0.28, [lab], size=9.5, align=PP_ALIGN.LEFT, color=SUB)

prs.save(OUT)
print("saved", OUT)
