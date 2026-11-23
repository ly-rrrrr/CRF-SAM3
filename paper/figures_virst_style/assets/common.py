from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.oxml.ns import qn
from lxml import etree
import sys

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
prs.slide_width = Inches(11.8)
prs.slide_height = Inches(7.75)
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
            f.name = "Segoe UI Emoji" if "emoji" in flags else FONT
            f.size = Pt(sz)
            f.italic = "i" in flags
            f.bold = "b" in flags
            f.color.rgb = C(col)
            rPr = r._r.get_or_add_rPr()
            for tag in ("a:latin", "a:ea", "a:cs"):
                el = rPr.find(qn(tag))
                if el is None:
                    el = etree.SubElement(rPr, qn(tag))
                el.set("typeface", "Segoe UI Emoji" if "emoji" in flags else FONT)
            if "sub" in flags: rPr.set("baseline", "-25000")
            if "sup" in flags: rPr.set("baseline", "30000")

def box(x, y, w, h, lines, pal=WHITE, size=12, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
        dash=False, lw=1.0, radius=0.18, color=TXT, align=PP_ALIGN.CENTER, nofill=False, noline=False, anchor=MSO_ANCHOR.MIDDLE, wrap=True):
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
    tf.word_wrap = wrap
    tf.margin_left = tf.margin_right = I(0.04)
    tf.margin_top = tf.margin_bottom = I(0.02)
    tf.vertical_anchor = anchor
    if lines: runs(tf, lines, size, align, color)
    return sp

def text(x, y, w, h, lines, size=11, align=PP_ALIGN.CENTER, color=TXT, anchor=MSO_ANCHOR.MIDDLE, wrap=True):
    return box(x, y, w, h, lines, size=size, nofill=True, noline=True, align=align, color=color, anchor=anchor, wrap=wrap)

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

