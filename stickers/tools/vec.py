"""Photo -> clean layered SVG sticker.

Pipeline per piece:
 1. find the art region (sticker paper on the mat, or the whole sheet for drawings)
 2. white-balance against the paper, cluster colors inside the region
 3. merge near-duplicate clusters, snap darks to ink black, lights to paper
 4. smooth each color layer at 2x, drop specks
 5. trace each layer with potrace, stack light -> dark
 6. add a uniform white die-cut border around the art
"""
import cv2, numpy as np, subprocess, os, re, sys, json
from scipy import ndimage as ndi
from crops import CROPS

W = "work"
OUT = os.environ.get("OUT", "out")
os.makedirs(OUT, exist_ok=True)
os.makedirs(f"{OUT}/png", exist_ok=True)

INK_BLACK = (22, 20, 26)

CFG = {
  # sticker = art on a hand-cut white sticker sitting on wax paper
  "stop-sign-guy":   dict(kind="sticker", k=2, group=0.05),
  "moon-eyes":       dict(kind="sticker", k=5),
  "bald-guy":        dict(kind="sticker", k=2),
  "hooded-head":     dict(kind="sticker", k=5),
  "dress-figure":    dict(kind="sticker", k=4),
  "crow":            dict(kind="sticker", k=5),
  "round-face":      dict(kind="sticker", k=5),
  "ha-creature":     dict(kind="sticker", k=5),
  "my-feewins":      dict(kind="sticker", k=6, group=0.08),
  "light-of-my-life":dict(kind="sticker", k=7, merge=14, grabcut=True),
  "fish":            dict(kind="cutout", k=3),
  # drawing = ink on a flat sheet
  "damage":          dict(kind="drawing", k=6, flat=False, merge=10),
  "happy-friday":    dict(kind="drawing", k=2, flat=True, pencil=True),
  "woodland-bird":   dict(kind="drawing", k=8, flat=True, merge=12,
                          erase=[(0.45, 0.80, 1.0, 1.0)]),
  "easter-bunny":    dict(kind="drawing", k=8, flat=True, merge=12),
  "red-thunderbird": dict(kind="drawing", k=2, flat=True),
  "f-off-cat":       dict(kind="drawing", k=2, flat=True, erase=[(0.96,0,1,1),(0,0.45,0.05,1)]),
  "cat-face":        dict(kind="drawing", k=2, flat=True),
}


def lab(img):  # BGR uint8 -> float Lab (OpenCV 8-bit scale)
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)


def largest(mask):
    n, lab_, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return mask.astype(bool)
    i = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
    return lab_ == i


def drop_small(mask, min_area):
    n, lab_, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[lab_]


def disk(r):
    r = max(1, int(round(r)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def flatfield(img):
    """Divide out uneven lighting so paper reads as flat white."""
    s = max(img.shape[:2])
    k = int(s / 12) | 1
    bg = cv2.morphologyEx(img, cv2.MORPH_CLOSE, disk(k // 2))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 2)
    out = np.clip(img.astype(np.float32) / np.maximum(bg, 1) * 245, 0, 255)
    return out.astype(np.uint8)


def region_for_sticker(img):
    """Sticker paper = brightest of three lightness levels and smooth (wax paper is mottled).
    Take the largest such blob and fill its holes."""
    from skimage.filters import threshold_multiotsu
    sm = cv2.medianBlur(img, 7)
    l = lab(sm)[..., 0]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mu = cv2.blur(g, (15, 15)); sd = np.sqrt(np.maximum(cv2.blur(g * g, (15, 15)) - mu * mu, 0))
    t = threshold_multiotsu(l.astype(np.uint8), 3)[1]
    cand = (l > t) & (sd < np.percentile(sd, 70))
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, disk(6))
    paper = largest(cand)
    R = max(8, int(max(img.shape[:2]) * 0.03))
    pad = R * 2
    p2 = cv2.copyMakeBorder(paper.astype(np.uint8), pad, pad, pad, pad, cv2.BORDER_CONSTANT, 0)
    p2 = cv2.morphologyEx(p2, cv2.MORPH_CLOSE, disk(R))[pad:-pad, pad:-pad]
    reg = ndi.binary_fill_holes(p2)
    reg = cv2.erode(reg.astype(np.uint8), disk(R * 0.8)).astype(bool)  # stay off the cut edge
    return reg, paper


def cluster(img, region, k, merge_de, pencil=False, paper_mask=None):
    """k-means in Lab inside region; return per-pixel label image and output colors."""
    f = cv2.bilateralFilter(img, 9, 30, 5)
    f = cv2.medianBlur(f, 5)
    L = lab(f)
    pts = L[region]
    # white balance: brightest low-chroma pixels are the paper
    rgbf = f[region].astype(np.float32)
    lum = pts[:, 0]
    if paper_mask is not None and paper_mask.sum() > 200:
        paper_px = f[paper_mask].astype(np.float32)
    else:
        paper_px = rgbf[lum >= np.percentile(lum, 85)]
    gain = 248.0 / np.maximum(np.median(paper_px, 0), 1)
    sub = pts[:: max(1, len(pts) // 120000)]
    _, _, cen = cv2.kmeans(sub, k, None,
                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.3),
                           6, cv2.KMEANS_PP_CENTERS)
    cen = cen.astype(np.float32)
    # merge near-duplicate centers (marker shading / paper texture splits)
    groups = list(range(k))
    def find(a):
        while groups[a] != a:
            a = groups[a]
        return a
    for a in range(k):
        for b in range(a + 1, k):
            d = np.linalg.norm((cen[a] - cen[b]) * np.array([0.6, 1.0, 1.0]))
            if d < merge_de:
                groups[find(b)] = find(a)
    # assign every region pixel to nearest center
    D = np.stack([np.linalg.norm(L - c, axis=2) for c in cen], -1)
    lbl = np.argmin(D, -1)
    lut = np.array([find(i) for i in range(k)])
    lbl = lut[lbl]
    lbl[~region] = -1
    # output colour per merged group, white-balanced
    out = {}
    for g in sorted(set(find(i) for i in range(k))):
        m = lbl == g
        if m.sum() == 0:
            continue
        c = np.median(f[m].astype(np.float32), 0) * gain  # BGR
        c = np.clip(c, 0, 255)
        out[g] = c[::-1]  # -> RGB
    return lbl, out, gain


def classify_colors(cols, paper_lum=200, pencil=False):
    """Map raw group colours to roles: paper / ink-black / colour (with a bit of saturation boost)."""
    roles = {}
    for g, rgb in cols.items():
        px = np.uint8([[rgb[::-1]]])
        L, a, b = cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0, 0].astype(float)
        chroma = np.hypot(a - 128, b - 128)
        if L > paper_lum and chroma < 22:
            roles[g] = ("paper", (255, 255, 255))
        elif L < 95 and chroma < 20:
            roles[g] = ("ink", INK_BLACK)
        elif pencil and chroma < 14:
            # graphite: keep a soft dark grey
            roles[g] = ("ink", (70, 70, 74))
        else:
            hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0].astype(float)
            hsv[1] = min(255, hsv[1] * 1.25)
            hsv[2] = min(255, hsv[2] * 1.08)
            c = cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2BGR)[0, 0][::-1]
            roles[g] = ("color", tuple(int(x) for x in c))
    return roles


def smooth_layers(lbl, groups, scale, sigma):
    """2x upsample + gaussian vote = clean, curvy class boundaries."""
    H, W_ = lbl.shape
    votes = []
    for g in groups:
        m = (lbl == g).astype(np.float32)
        m = cv2.resize(m, (W_ * scale, H * scale), interpolation=cv2.INTER_LINEAR)
        votes.append(cv2.GaussianBlur(m, (0, 0), sigma))
    bgm = (lbl == -1).astype(np.float32)
    bgm = cv2.resize(bgm, (W_ * scale, H * scale), interpolation=cv2.INTER_NEAREST)
    votes.append(bgm * 10)  # outside region always wins
    v = np.stack(votes, -1)
    win = np.argmax(v, -1)
    return {g: win == i for i, g in enumerate(groups)}


def potrace_paths(mask, turd=20, alpha=1.0, opt=0.3):
    h, w = mask.shape
    pbm = "/tmp/claude-0/-home-user-MotchSpace/92140ed2-0a47-5b77-9654-1640c8dc45d3/scratchpad/_l.pbm"
    with open(pbm, "wb") as fh:
        fh.write(f"P4\n{w} {h}\n".encode())
        fh.write(np.packbits(mask.astype(np.uint8), axis=1).tobytes())
    svg = subprocess.run(["potrace", pbm, "-s", "--flat", "-t", str(turd), "-a", str(alpha),
                          "-O", str(opt), "-u", "4", "-o", "-"], capture_output=True, text=True).stdout
    ds = re.findall(r'<path d="([^"]+)"', svg, re.S)
    tr = re.search(r'<g transform="([^"]+)"', svg)
    return " ".join(d.replace("\n", " ") for d in ds), tr.group(1) if tr else ""


def hexc(c):
    return "#%02x%02x%02x" % tuple(int(x) for x in c)


def write_svg(name, layers, size, title):
    """layers: list of (hex, mask) bottom -> top."""
    h, w = size
    parts = []
    for col, m, lid in layers:
        if m.sum() == 0:
            continue
        d, tr = potrace_paths(m)
        if not d:
            continue
        parts.append(f'  <g id="{lid}" fill="{col}" transform="{tr}"><path d="{d}"/></g>')
    # potrace output units: transform maps 10 units/px with y flip, with -u 4 -> scale(0.25)
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">\n'
           f'  <title>{title}</title>\n' + "\n".join(parts) + "\n</svg>\n")
    p = f"{OUT}/{name}.svg"
    open(p, "w").write(svg)
    subprocess.run(["rsvg-convert", "-w", "900", "-o", f"{OUT}/png/{name}.png", p])
    return p


def tight(mask, pad):
    ys, xs = np.where(mask)
    return max(0, ys.min() - pad), ys.max() + pad + 1, max(0, xs.min() - pad), xs.max() + pad + 1


def border_from_art(art, margin):
    cut = cv2.dilate(art.astype(np.uint8), disk(margin))
    cut = cv2.morphologyEx(cut, cv2.MORPH_CLOSE, disk(margin * 1.5))
    cut = ndi.binary_fill_holes(cut).astype(np.float32)
    cut = cv2.GaussianBlur(cut, (0, 0), margin / 3) > 0.5
    return largest(cut) | cut


def process(name):
    cfg = CFG[name]
    img = cv2.imread(f"{W}/{name}_crop.png")
    # work at ~1400px on the long side
    s = 1400 / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    H, Wd = img.shape[:2]

    paper_mask = None
    if cfg["kind"] == "cutout":
        region, _ = region_for_sticker(img)
    elif cfg["kind"] == "sticker":
        from ink import ink_region
        region, _ = ink_region(img, group=cfg.get("group", 0.035), dark_t=cfg.get("dark_t"))
        if cfg.get("grabcut"):
            from gcut import gc_region
            region |= cv2.erode(gc_region(img).astype(np.uint8), disk(10)).astype(bool)
        region = cv2.dilate(region.astype(np.uint8), disk(3)).astype(bool)
        # paper sample: a ring just outside the art, bright half only
        ring = cv2.dilate(region.astype(np.uint8), disk(30)).astype(bool) & ~cv2.dilate(region.astype(np.uint8), disk(10)).astype(bool)
        lv = lab(img)[..., 0]
        if ring.sum():
            paper_mask = ring & (lv > np.percentile(lv[ring], 70))
    else:
        region = np.ones((H, Wd), bool)
        if cfg.get("flat"):
            img = flatfield(img)
        for (a, b, c, d) in cfg.get("erase", []):
            region[int(b * H):int(d * H), int(a * Wd):int(c * Wd)] = False

    lbl, cols, _ = cluster(img, region, cfg["k"], cfg.get("merge", 16), cfg.get("pencil"), paper_mask)
    roles = classify_colors(cols, pencil=cfg.get("pencil"))

    # fish: the punched eye hole reads as wood -> make it an eye dot
    groups = list(cols)
    scale, sigma = 2, cfg.get("sigma", 1.6)
    masks = smooth_layers(lbl, groups, scale, sigma)
    if cfg["kind"] == "cutout":
        reg2 = cv2.resize(region.astype(np.uint8), (Wd * scale, H * scale), interpolation=cv2.INTER_NEAREST).astype(bool)
    minA = int(cfg.get("min_area", 60) * scale * scale)

    # collapse groups into output layers by colour
    layers = {}
    for g, m in masks.items():
        role, c = roles[g]
        if role == "paper":
            continue
        key = hexc(c)
        layers.setdefault(key, [role, np.zeros_like(m)])
        layers[key][1] |= m
    for key in layers:
        m = layers[key][1].astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, disk(1))
        layers[key][1] = drop_small(m, minA)

    art = np.zeros((H * scale, Wd * scale), bool)
    for _, (_, m) in layers.items():
        art |= m
    if not art.any():
        raise SystemExit(f"{name}: no art found")

    # sticker base
    span = max(np.ptp(np.where(art)[0]), np.ptp(np.where(art)[1]))
    margin = max(8, int(span * cfg.get("margin", 0.045)))
    if cfg["kind"] == "cutout":
        cut = border_from_art(reg2, max(6, margin // 2))
    else:
        cut = border_from_art(art, margin)
    y0, y1, x0, x1 = tight(cut, 6)

    # order: lighter colours first, ink last; let each layer bleed under later ones to kill seams
    def lum(k):
        c = [int(k[i:i + 2], 16) for i in (1, 3, 5)]
        return -(0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2])
    order = sorted(layers, key=lum)
    stack = [("#ffffff", cut, "sticker-border")]
    for i, key in enumerate(order):
        m = layers[key][1]
        later = np.zeros_like(m)
        for k2 in order[i + 1:]:
            later |= layers[k2][1]
        m2 = m | (cv2.dilate(m.astype(np.uint8), disk(2)).astype(bool) & later)
        stack.append((key, m2, f"{layers[key][0]}-{key[1:]}"))
    stack = [(c, m[y0:y1, x0:x1], i) for c, m, i in stack]
    p = write_svg(name, stack, (y1 - y0, x1 - x0), name.replace("-", " "))
    return {"name": name, "colors": [k for k in order], "svg": p}


if __name__ == "__main__":
    names = sys.argv[1:] or list(CFG)
    for n in names:
        try:
            r = process(n)
            print(n, r["colors"], os.path.getsize(r["svg"]) // 1024, "KB")
        except Exception as e:
            import traceback; traceback.print_exc()
            print("FAIL", n, e)
