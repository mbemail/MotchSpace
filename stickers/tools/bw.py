"""Make black-and-white versions of the finished sticker SVGs.

Rule per colour layer:
  near-neutral dark (ink, graphite)      -> black
  very light colour (yellow etc.)        -> white
  other colours: thin lines -> black, filled areas -> white
The white sticker border stays white.
"""
import re, os, subprocess, sys, io
import numpy as np, cv2
from PIL import Image

SRC = "out"
DST = "out_bw"
os.makedirs(f"{DST}/png", exist_ok=True)
BLACK, WHITE = "#16141a", "#ffffff"


def lab_of(hexc):
    rgb = [int(hexc[i:i + 2], 16) for i in (1, 3, 5)]
    L, a, b = cv2.cvtColor(np.uint8([[rgb[::-1]]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float)
    return L * 100 / 255, np.hypot(a - 128, b - 128)


def render_layer(svg, keep):
    """Render only the layers whose fill is `keep`; return alpha mask at ~800px."""
    s = re.sub(r'fill="(#[0-9a-fA-F]{6})"', lambda m: m.group(0) if m.group(1).lower() == keep else 'fill="none"', svg)
    s = s.replace('stroke="#ffffff"', 'stroke="none"')
    png = subprocess.run(["rsvg-convert", "-w", "800"], input=s.encode(), capture_output=True).stdout
    a = np.array(Image.open(io.BytesIO(png)).convert("RGBA"))[..., 3] > 127
    return a


def thickness(mask):
    """Typical stroke width: 2 * area / perimeter."""
    if mask.sum() == 0:
        return 0
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    per = sum(cv2.arcLength(c, True) for c in cs)
    return 2 * mask.sum() / max(per, 1)


def convert(name):
    svg = open(f"{SRC}/{name}.svg").read()
    tags = re.findall(r'<(?:g|path) id="([^"]+)" fill="(#[0-9a-fA-F]{6})"', svg)
    layers = [(i, c.lower()) for i, c in tags if i != "sticker-border"]
    single = len({c for _, c in layers}) == 1
    decide = {}
    for lid, c in layers:
        L, ch = lab_of(c)
        if single:
            v = BLACK                      # one ink colour: it is the drawing
        elif lid.startswith("fill-"):
            v = WHITE                      # pencil fills from the region-fill step
        elif ch < 14 and L < 60:
            v = BLACK                      # ink, graphite, grey line work
        elif L > 78:
            v = WHITE                      # yellow etc.
        else:
            t = thickness(render_layer(svg, c))
            v = BLACK if t < 9 else WHITE  # thin colour = line work, thick = fill
        decide[lid] = v
    def sub(m):
        lid = m.group(1)
        return m.group(0).replace(m.group(2), decide.get(lid, m.group(2)))
    out = re.sub(r'<(?:g|path) id="([^"]+)" fill="(#[0-9a-fA-F]{6})"', sub, svg)
    out = out.replace("<title>", "<title>black and white: ", 1)
    p = f"{DST}/{name}.svg"
    open(p, "w").write(out)
    subprocess.run(["rsvg-convert", "-w", "900", "-o", f"{DST}/png/{name}.png", p])
    return decide


def damage_bw():
    """Crayon piece: threshold the cleaned photo so the blue strokes and dense crayon read as black."""
    sys.argv = ["x"]
    from vec import potrace_paths, border_from_art, tight
    img = cv2.imread("work/damage_clean.png", cv2.IMREAD_UNCHANGED)
    alpha = img[..., 3] > 0
    g = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2LAB)
    L = cv2.GaussianBlur(g[..., 0].astype(np.float32), (0, 0), 1.2)
    b = g[..., 2].astype(np.float32)
    # blue crayon is low in Lab b; red field is high in a. Black = blue strokes + darkest tones.
    a_ = g[..., 1].astype(np.float32)
    bs = cv2.GaussianBlur(b, (0, 0), 1.5)
    # blue crayon: Lab b well below the kraft paper's
    ink = (bs < np.percentile(bs[alpha], 9)) & (L < 170) & alpha
    ink = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)).astype(bool)
    pad = 110
    P = lambda m: np.pad(m, pad)
    cut = border_from_art(P(alpha), 42)
    y0, y1, x0, x1 = tight(cut, 4)
    dc, tc = potrace_paths(cut[y0:y1, x0:x1])
    dp, tp = potrace_paths(P(alpha)[y0:y1, x0:x1])
    di, ti = potrace_paths(P(ink)[y0:y1, x0:x1], turd=4)
    w, h = x1 - x0, y1 - y0
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">\n'
           f'  <title>black and white: damage</title>\n'
           f'  <g id="sticker-border" fill="#ffffff" transform="{tc}"><path d="{dc}"/></g>\n'
           f'  <g id="ink-16141a" fill="#16141a" transform="{ti}"><path d="{di}"/></g>\n</svg>\n')
    p = f"{DST}/damage.svg"
    open(p, "w").write(svg)
    subprocess.run(["rsvg-convert", "-w", "900", "-o", f"{DST}/png/damage.png", p])


if __name__ == "__main__":
    names = sys.argv[1:] or [os.path.basename(f)[:-4] for f in sorted(os.listdir(SRC)) if f.endswith(".svg")]
    for n in names:
        if n == "damage":
            damage_bw(); print(n, "threshold trace"); continue
        print(n, convert(n))
