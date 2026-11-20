"""Photo -> clean layered SVG sticker (v2).

Per piece:
 1. find the art region (ink-anchored for stickers, the sheet for drawings)
 2. white-balance against nearby paper, k-means colours inside the region
 3. snap neutrals to ink-black / paper, keep real colours
 4. 2x upsample + gaussian vote, then absorb specks into their neighbours
 5. optional: flat-fill closed shapes (colored pencil pieces)
 6. potrace each colour, stack light -> dark, add a uniform white die-cut border
"""
import cv2, numpy as np, os, sys
from scipy import ndimage as ndi
from vec import (lab, largest, drop_small, disk, flatfield, cluster, potrace_paths,
                 hexc, write_svg, tight, border_from_art, INK_BLACK, OUT, W)

CFG = {
  "stop-sign-guy":   dict(kind="sticker", k=2, group=0.05, thresh=90, speck=75),
  "moon-eyes":       dict(kind="sticker", k=5, merge=8, speck=250, sigma=2.2),
  "bald-guy":        dict(kind="sticker", k=2),
  "hooded-head":     dict(kind="sticker", k=5),
  "dress-figure":    dict(kind="sticker", k=5, merge=8, speck=150, sigma=2.0),
  "crow":            dict(kind="sticker", k=5),
  "round-face":      dict(kind="sticker", k=5, merge=8, speck=320, sigma=2.4),
  "ha-creature":     dict(kind="sticker", k=5, merge=16, speck=320, sigma=2.6),
  "my-feewins":      dict(kind="sticker", k=7, merge=10, region="grabcut", speck=40),
  "light-of-my-life":dict(kind="sticker", k=7, merge=9, grabcut=True, speck=90),
  "fish":            dict(kind="fish"),
  "damage":          dict(kind="drawing", k=7, merge=6, region="nonwhite", sigma=3.0, speck=400, keep_colors=True),
  "happy-friday":    dict(kind="drawing", k=2, flat=True, pencil=True, ink=(78, 78, 84), erase=[(0, 0.83, 1, 1), (0.90, 0, 1, 1)]),
  "woodland-bird":   dict(kind="drawing", k=6, flat=True, merge=12, fill=True, dark_is_black=True,
                          erase=[(0.45, 0.93, 1.0, 1.0), (0, 0, 1, 0.05), (0.935, 0, 1, 1)]),
  "easter-bunny":    dict(kind="drawing", k=7, flat=True, merge=12, fill=True, dark_is_black=True, erase=[(0, 0.95, 1, 1)]),
  "red-thunderbird": dict(kind="drawing", k=2, flat=True, ink=(214, 44, 56)),
  "f-off-cat":       dict(kind="drawing", k=2, flat=True, ink=INK_BLACK,
                          erase=[(0.975, 0, 1, 1), (0, 0.5, 0.06, 1), (0.92, 0, 1, 0.12)]),
  "cat-face":        dict(kind="drawing", k=2, flat=True, ink=(18, 98, 64)),
}


def chroma_L(rgb):
    px = np.uint8([[list(rgb)[::-1]]])
    L, a, b = cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0, 0].astype(float)
    return L, np.hypot(a - 128, b - 128)


def roles_for(cols, cfg):
    roles = {}
    darkest = min(cols, key=lambda g: chroma_L(cols[g])[0])
    for g, rgb in cols.items():
        L, ch = chroma_L(rgb)
        if cfg.get("fill") and L > 165 and ch < 40:
            roles[g] = ("paper", None)          # pale pencil: handled by region fill
        elif L > 185 and ch < 18:
            roles[g] = ("paper", None)
        elif cfg.get("ink") is not None:
            roles[g] = ("ink", cfg["ink"])
        elif (L < 95 and ch < 13) or (cfg.get("dark_is_black") and g == darkest and ch < 20):
            roles[g] = ("ink", INK_BLACK)
        elif cfg.get("keep_colors"):
            roles[g] = ("color", tuple(int(x) for x in rgb))
        else:
            px = np.uint8([[list(rgb)[::-1]]])
            hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0].astype(float)
            hsv[1] = min(255, hsv[1] * 1.2)
            hsv[2] = min(255, hsv[2] * 1.06)
            c = cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2BGR)[0, 0][::-1]
            roles[g] = ("color", tuple(int(x) for x in c))
    return roles


def ink_region(img, group=0.035, close=0.012):
    h, w = img.shape[:2]; S = max(h, w)
    L = lab(cv2.medianBlur(img, 5))
    l = L[..., 0]; ch = np.hypot(L[..., 1] - 128, L[..., 2] - 128)
    t = min(70, np.percentile(l, 3) + 35)
    dark = cv2.morphologyEx((l < t).astype(np.uint8), cv2.MORPH_OPEN, disk(1))
    n, lb, st, _ = cv2.connectedComponentsWithStats(dark, 8)
    keep = st[:, cv2.CC_STAT_AREA] >= S * 0.02
    keep[0] = False
    # drop saturated dark blobs (exposed cutting mat) and blobs mostly on the crop edge
    mch = ndi.mean(ch, lb, np.arange(n))
    keep &= np.array(mch) < 16
    edge = np.zeros(n);
    for e in (lb[0], lb[-1], lb[:, 0], lb[:, -1]):
        edge += np.bincount(e, minlength=n)
    keep &= edge < 0.25 * np.sqrt(st[:, cv2.CC_STAT_AREA]) + 1
    dark = keep[lb]
    g = largest(cv2.dilate(dark.astype(np.uint8), disk(S * group)))
    ink = dark & g
    c = max(2, int(S * close)); pad = c * 2
    r = cv2.copyMakeBorder(ink.astype(np.uint8), pad, pad, pad, pad, cv2.BORDER_CONSTANT, 0)
    r = cv2.morphologyEx(r, cv2.MORPH_CLOSE, disk(c))[pad:-pad, pad:-pad]
    return ndi.binary_fill_holes(r)


def absorb_specks(lbl, min_area):
    """Every connected blob smaller than min_area (any class) takes its nearest neighbour's label."""
    unknown = np.zeros(lbl.shape, bool)
    for v in np.unique(lbl):
        if v == -1:
            continue
        m = (lbl == v).astype(np.uint8)
        n, cc, st, _ = cv2.connectedComponentsWithStats(m, 4)
        small = st[:, cv2.CC_STAT_AREA] < min_area
        small[0] = False
        unknown |= small[cc]
    if not unknown.any():
        return lbl
    _, (iy, ix) = ndi.distance_transform_edt(unknown, return_indices=True)
    out = lbl[iy, ix]
    return out


def region_fill(lbl2, ink_groups, img2, art_hull, min_area, n_colors=5):
    """Colored-pencil fills: each closed non-ink shape gets one flat colour."""
    lines = np.isin(lbl2, ink_groups)
    space = art_hull & ~lines
    n, cc = cv2.connectedComponents(space.astype(np.uint8), connectivity=4)
    L = lab(img2).reshape(-1, 3)
    ccf = cc.ravel()
    cnt = np.bincount(ccf, minlength=n).astype(float)
    mean = np.stack([np.bincount(ccf, L[:, i], minlength=n) for i in range(3)], 1) / np.maximum(cnt, 1)[:, None]
    chroma = np.hypot(mean[:, 1] - 128, mean[:, 2] - 128)
    # regions touching the outer hull edge are background paper
    border = cv2.dilate((~art_hull).astype(np.uint8), disk(2)).astype(bool)
    touches = np.bincount(cc[border], minlength=n) > 0
    pick = (cnt >= min_area) & (chroma > 6) & ~touches
    pick[0] = False
    idx = np.where(pick)[0]
    if len(idx) < 2:
        return {}
    feats = np.stack([mean[idx, 1] - 128, mean[idx, 2] - 128, (mean[idx, 0] - 200) * 0.3], 1).astype(np.float32).reshape(-1, 3)
    k = min(n_colors, len(idx))
    _, lab_i, cen = cv2.kmeans(feats, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2),
                               8, cv2.KMEANS_PP_CENTERS)
    layers = {}
    for j in range(k):
        members = idx[lab_i.ravel() == j]
        w = cnt[members]
        a, b = (np.average(mean[members, 1], weights=w) - 128, np.average(mean[members, 2], weights=w) - 128)
        Lm = np.average(mean[members, 0], weights=w)
        c = np.hypot(a, b)
        if c < 5:
            continue
        f = min(4.0, max(1.0, 34.0 / c))       # pencil is pale; push it to a printable tint
        lab_px = np.uint8([[[np.clip(min(Lm, 225), 0, 255), np.clip(128 + a * f, 0, 255), np.clip(128 + b * f, 0, 255)]]])
        rgb = cv2.cvtColor(lab_px, cv2.COLOR_LAB2BGR)[0, 0][::-1]
        key = hexc(rgb)
        m = np.isin(cc, members)
        layers[key] = layers.get(key, np.zeros_like(m)) | m
    return layers


def fish(img):
    """White paper fish with blue pen edges, eye punched through."""
    L = lab(cv2.medianBlur(img, 5))
    l, a, b = L[..., 0], L[..., 1], L[..., 2]
    paper = (l > 160) & (np.hypot(a - 128, b - 128) < 25)
    paper = largest(cv2.morphologyEx(paper.astype(np.uint8), cv2.MORPH_OPEN, disk(3)))
    blue = (b < 112) & (l < 200)
    shape = paper | (blue & cv2.dilate(paper.astype(np.uint8), disk(8)).astype(bool))
    shape = cv2.morphologyEx(shape.astype(np.uint8), cv2.MORPH_CLOSE, disk(4)).astype(bool)
    filled = ndi.binary_fill_holes(shape)
    eye = filled & ~shape
    eye = drop_small(eye, 50)
    lbl = np.full(l.shape, -1, int)
    lbl[filled] = 0
    lbl[blue & filled] = 1
    lbl[eye] = 1
    return lbl, {0: ("paper", None), 1: ("ink", (40, 52, 170))}


def compose(name, lbl2, roles, extra_layers=None, margin_frac=0.045):
    layers = {}
    for g, (role, c) in roles.items():
        if role == "paper":
            continue
        key = hexc(c)
        layers.setdefault(key, [role, np.zeros(lbl2.shape, bool)])
        layers[key][1] |= (lbl2 == g)
    for key, m in (extra_layers or {}).items():
        layers.setdefault(key, ["fill", np.zeros(lbl2.shape, bool)])
        layers[key][1] |= m
    art = np.zeros(lbl2.shape, bool)
    for _, m in layers.values():
        art |= m
    paper_inside = np.isin(lbl2, [g for g, (r, _) in roles.items() if r == "paper"])
    ys, xs = np.where(art)
    span = max(np.ptp(ys), np.ptp(xs))
    margin = max(10, int(span * margin_frac))
    pad = margin * 3
    P = lambda m: np.pad(m, pad)
    art_p = P(art)
    cut = border_from_art(art_p, margin)
    if name == "fish":
        cut = border_from_art(P(art | paper_inside), margin // 2)
    y0, y1, x0, x1 = tight(cut, 6)

    def lum(k):
        c = [int(k[i:i + 2], 16) for i in (1, 3, 5)]
        return -(0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2])
    order = sorted(layers, key=lum)
    stack = [("#ffffff", cut, "sticker-border")]
    for i, key in enumerate(order):
        m = P(layers[key][1])
        later = np.zeros_like(m)
        for k2 in order[i + 1:]:
            later |= P(layers[k2][1])
        m2 = m | (cv2.dilate(m.astype(np.uint8), disk(2)).astype(bool) & later)
        stack.append((key, m2, f"{layers[key][0]}-{key[1:]}"))
    stack = [(c, m[y0:y1, x0:x1], i) for c, m, i in stack]
    return write_svg(name, stack, (y1 - y0, x1 - x0), name.replace("-", " ")), order


def process(name):
    cfg = CFG[name]
    img = cv2.imread(f"{W}/{name}_crop.png")
    s = 1400 / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    H, Wd = img.shape[:2]
    scale = 2
    up = lambda m, interp=cv2.INTER_NEAREST: cv2.resize(m, (Wd * scale, H * scale), interpolation=interp)

    if cfg["kind"] == "fish":
        lbl, roles = fish(img)
        lbl2 = up(lbl.astype(np.int32))
        lbl2 = absorb_specks(lbl2, 200)
        return compose(name, lbl2, roles)

    paper_mask = None
    if cfg["kind"] == "sticker":
        if cfg.get("region") == "grabcut":
            from gcut import gc_region
            region = cv2.erode(gc_region(img).astype(np.uint8), disk(8)).astype(bool)
        else:
            region = ink_region(img, group=cfg.get("group", 0.035))
            if cfg.get("grabcut"):
                from gcut import gc_region
                region |= cv2.erode(gc_region(img).astype(np.uint8), disk(10)).astype(bool)
        region = cv2.dilate(region.astype(np.uint8), disk(3)).astype(bool)
        ring = cv2.dilate(region.astype(np.uint8), disk(30)).astype(bool) & \
            ~cv2.dilate(region.astype(np.uint8), disk(10)).astype(bool)
        lv = lab(img)[..., 0]
        if ring.sum():
            paper_mask = ring & (lv > np.percentile(lv[ring], 75))
    else:
        if cfg.get("flat"):
            img = flatfield(img)
        region = np.ones((H, Wd), bool)
        if cfg.get("region") == "nonwhite":
            L = lab(cv2.medianBlur(img, 9))
            white = (L[..., 0] > 200) & (np.hypot(L[..., 1] - 128, L[..., 2] - 128) < 14)
            region = ndi.binary_fill_holes(largest(~white))
            region = cv2.erode(region.astype(np.uint8), disk(4)).astype(bool)
        for (a, b, c, d) in cfg.get("erase", []):
            region[int(b * H):int(d * H), int(a * Wd):int(c * Wd)] = False

    if cfg.get("thresh"):
        lv = lab(cv2.bilateralFilter(img, 7, 30, 5))[..., 0]
        lbl = np.where(lv < cfg["thresh"], 1, 0)
        lbl[~region] = -1
        cols = {0: (255, 255, 255), 1: INK_BLACK}
    else:
        lbl, cols, _ = cluster(img, region, cfg["k"], cfg.get("merge", 16), cfg.get("pencil"), paper_mask)
    roles = roles_for(cols, cfg)
    groups = list(cols)

    # 2x gaussian vote
    sigma = cfg.get("sigma", 1.6)
    votes = [cv2.GaussianBlur(up((lbl == g).astype(np.float32), cv2.INTER_LINEAR), (0, 0), sigma) for g in groups]
    votes.append(up((lbl == -1).astype(np.float32)) * 10)
    win = np.argmax(np.stack(votes, -1), -1)
    lut = np.array(groups + [-1])
    lbl2 = lut[win]
    lbl2 = absorb_specks(lbl2, int(cfg.get("speck", 60) * scale * scale))

    extra = None
    if cfg.get("fill"):
        ink_groups = [g for g, (r, _) in roles.items() if r != "paper"]
        lines = np.isin(lbl2, ink_groups)
        hull = ndi.binary_fill_holes(cv2.morphologyEx(lines.astype(np.uint8), cv2.MORPH_CLOSE, disk(6)))
        img2 = up(cv2.medianBlur(img, 5), cv2.INTER_LINEAR)
        extra = region_fill(lbl2, ink_groups, img2, hull, min_area=int(40 * scale * scale))
    return compose(name, lbl2, roles, extra, cfg.get("margin", 0.045))


def goose():
    """Pixel art: rebuild the exact grid as squares."""
    img = cv2.imread(f"{W}/pixel-goose_crop.png", cv2.IMREAD_GRAYSCALE)
    dark = img < 180
    ys, xs = np.where(dark)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    # cell size: smallest run length of dark pixels along rows/cols, refined by fitting
    best = None
    for cell in np.arange(8.0, 20.0, 0.05):
        nx = round((x1 - x0) / cell); ny = round((y1 - y0) / cell)
        if nx < 3 or ny < 3:
            continue
        cx = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
        cy = y0 + (np.arange(ny) + 0.5) * (y1 - y0) / ny
        g = img[np.ix_(cy.astype(int), cx.astype(int))] < 180
        # crispness: fraction of pixels agreeing with their cell
        rec = np.kron(g, np.ones((1, 1)))
        Yi = np.clip(((np.arange(y0, y1) - y0) * ny / (y1 - y0)).astype(int), 0, ny - 1)
        Xi = np.clip(((np.arange(x0, x1) - x0) * nx / (x1 - x0)).astype(int), 0, nx - 1)
        agree = (g[np.ix_(Yi, Xi)] == dark[y0:y1, x0:x1]).mean()
        if best is None or agree > best[0] + 1e-4:
            best = (agree, nx, ny, g)
    agree, nx, ny, g = best
    print("goose grid", nx, ny, f"{agree:.3f}")
    U = 10
    m = 2
    Wv, Hv = (nx + 2 * m) * U, (ny + 2 * m) * U
    # border: grid cells within 1 of a pixel, as a blocky die-cut outline, rounded via stroke
    big = cv2.dilate(g.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
    big = ndi.binary_fill_holes(big)
    rects = lambda grid: "".join(f"M{(x + m) * U} {(y + m) * U}h{U}v{U}h-{U}z" for y, x in zip(*np.where(grid)))
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {Wv} {Hv}" width="{Wv * 4}" height="{Hv * 4}" shape-rendering="crispEdges">\n'
           f'  <title>pixel goose</title>\n'
           f'  <path id="sticker-border" fill="#ffffff" stroke="#ffffff" stroke-width="{U}" stroke-linejoin="round" d="{rects(big)}"/>\n'
           f'  <path id="ink-5a5a5e" fill="#5a5a5e" d="{rects(g)}"/>\n</svg>\n')
    p = f"{OUT}/pixel-goose.svg"
    open(p, "w").write(svg)
    import subprocess
    subprocess.run(["rsvg-convert", "-w", "900", "-o", f"{OUT}/png/pixel-goose.png", p])
    return p, ["#5a5a5e"]


if __name__ == "__main__":
    names = sys.argv[1:] or list(CFG) + ["pixel-goose"]
    for n in names:
        try:
            p, cols = goose() if n == "pixel-goose" else process(n)
            print(n, cols, os.path.getsize(p) // 1024, "KB")
        except Exception as e:
            import traceback; traceback.print_exc()
            print("FAIL", n, e)
