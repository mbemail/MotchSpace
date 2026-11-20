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
  "stop-sign-guy":   dict(kind="sticker", k=2, group=0.05, thresh=100, dots=300, speck=30, alpha=1.25, opt=0.5, nest=2.6, nest_thicken=1),
  "moon-eyes":       dict(kind="sticker", k=5, merge=8, speck=400, sigma=2.4, flat_colors=2, only_yellow=True, smooth=4.5, alpha=1.33, opt=1.0, nest=14),
  "bald-guy":        dict(kind="sticker", k=2, speck=200, smooth=5.0, alpha=1.33, opt=0.8, nest=14),
  "hooded-head":     dict(kind="sticker", k=5, flat_colors=2, speck=200, smooth=4.5, alpha=1.33, opt=0.8, nest=14),
  "dress-figure":    dict(kind="sticker", k=5, merge=8, speck=60, sigma=1.6, flat_colors=1, inside=False, chroma_split=14, nest=3, close_bottom=0.025),
  "crow":            dict(kind="sticker", k=5),
  "round-face":      dict(kind="sticker", k=5, merge=8, speck=400, sigma=2.4, chroma_split=9, flat_colors=2, smooth=4.5, alpha=1.33, opt=1.0, nest=17, ring=0.045, islands=300, ring_islands=2500),
  "ha-creature":     dict(kind="sticker", k=5, merge=8, speck=400, sigma=2.4, inside=False, chroma_split=9, flat_colors=2, smooth=4.5, alpha=1.33, opt=1.0, nest=16, fill_outline=14, eye_ring=(58, 36), lift=True, local_thin=True, text_ink=118),
  "my-feewins":      dict(kind="sticker", k=7, merge=10, region="grabcut", speck=40, inside=False, rescue=22, flat_colors=3, smooth=4.0, thicken=4, alpha=1.33, opt=0.8, nest=11, nest_thicken=5, thin_to_black=14, text_ink=150, islands=600),
  "light-of-my-life":dict(kind="sticker", k=7, merge=9, grabcut=True, speck=60, inside=False, dark_to_black=120, paper_L=160, flat_colors=2, alpha=1.2, nest=2.2),
  "fish":            dict(kind="fish", nest=3.5, nest_thicken=1),
  "damage":          dict(kind="drawing", k=7, merge=6, region="nonwhite", sigma=3.0, speck=400, keep_colors=True),
  "happy-friday":    dict(kind="drawing", k=2, flat=True, pencil=True, ink=(78, 78, 84), erase=[(0, 0.83, 1, 1), (0.90, 0, 1, 1)]),
  "woodland-bird":   dict(kind="drawing", k=6, flat=True, merge=12, fill=True, dark_is_black=True,
                          erase=[(0.45, 0.93, 1.0, 1.0), (0.72, 0.86, 1.0, 1.0), (0, 0, 1, 0.05), (0.935, 0, 1, 1)], rotate=-90, fill_chroma=3.5, fill_colors=8),
  "easter-bunny":    dict(kind="drawing", k=7, flat=False, merge=12, fill=True, dark_is_black=True),
  "red-thunderbird": dict(kind="drawing", k=2, flat=True, ink=(214, 44, 56)),
  "f-off-cat":       dict(kind="drawing", k=2, flat=True, ink=INK_BLACK,
                          erase=[(0.975, 0, 1, 1), (0, 0.5, 0.06, 1), (0.92, 0, 1, 0.12)]),
  "cat-face":        dict(kind="drawing", k=2, flat=True, ink=(18, 98, 64)),
}

# black-and-white-only variants of pieces whose colour version already passed
CFG.update({
  "crow-smooth":        dict(CFG["crow"], src="crow", paper_L=150, speck=200, smooth=4.0, nest=14, alpha=1.33, opt=1.0, fill_specks=900),
  "dress-figure-sq":    dict(CFG["dress-figure"], src="dress-figure", close_bottom=0.05, square=35),
  "f-off-cat-sq":       dict(CFG["f-off-cat"], src="f-off-cat", bottom_line=0.06,
                             erase=CFG["f-off-cat"]["erase"] + [(0.93, 0, 1, 0.62)]),
  "happy-friday-crow":  dict(CFG["happy-friday"], src="happy-friday", erase=[(0, 0.97, 1, 1), (0.90, 0, 1, 1), (0, 0, 1, 0.2), (0, 0, 0.3, 1)],
                             keep_at=(0.70, 0.45), keep_join=5),
  "light-of-my-life-txt": dict(CFG["light-of-my-life"], src="light-of-my-life", text_bands=[(0.0, 0.2), (0.7, 1.0)], text_L=135, nest=1.6),
  "red-thunderbird-solid": dict(CFG["red-thunderbird"], src="red-thunderbird", solid=7, nest=4, alpha=1.33, opt=1.0),
})


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
        elif L > cfg.get("paper_L", 185) and ch < 18:
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


def region_fill(lbl2, ink_groups, img2, art_hull, min_area, n_colors=5, min_chroma=6):
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
    pick = (cnt >= min_area) & (chroma > min_chroma) & ~touches
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
    blue = (b < 116) & (l < 210)
    paper = (l > 140) & (np.hypot(a - 128, b - 128) < 25)
    shape = cv2.morphologyEx((paper | blue).astype(np.uint8), cv2.MORPH_OPEN, disk(2))
    shape = largest(cv2.morphologyEx(shape, cv2.MORPH_CLOSE, disk(3)))
    filled = ndi.binary_fill_holes(shape)
    eye = filled & ~shape
    eye = drop_small(eye, 50)
    lbl = np.full(l.shape, -1, int)
    lbl[filled] = 0
    lbl[blue & filled] = 1
    lbl[eye] = 1
    return lbl, {0: ("paper", None), 1: ("ink", (40, 52, 170))}


def nest_smooth(masks, sigma, thicken=0, local=False):
    """masks: bottom -> top. Smooth each 'this layer and everything above it' blob, so
    edges get very smooth without seams. Thin parts (text, fine lines) skip the blur."""
    n = len(masks)
    thin = []
    for m in masks:
        dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
        k, cc = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
        mx = np.asarray(ndi.maximum(dt, cc, np.arange(k))) if k > 1 else np.zeros(1)
        mx[0] = 1e9
        t = (mx < sigma * 1.2)[cc] & m
        if local:
            # thin by local width, so a thin ring joined to a thick outline still counts as thin
            lm = cv2.dilate(dt, disk(sigma * 0.9))
            t = t | (m & (lm < sigma * 0.9))
        if thicken > 0 and t.any():
            t = cv2.dilate(t.astype(np.uint8), disk(thicken)).astype(bool)
        thin.append(t)
    out = []
    for i in range(n):
        U = np.zeros_like(masks[0]); T = np.zeros_like(masks[0])
        for j in range(i, n):
            U |= masks[j]; T |= thin[j]
        body = U & ~T
        sm = cv2.GaussianBlur(body.astype(np.float32), (0, 0), sigma) > 0.5
        out.append(sm | T)
    return out


def compose(name, lbl2, roles, extra_layers=None, margin_frac=0.045, nest=0, nest_thicken=0, lift=False, local=False):
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
    if lift:
        # draw the main fill first, then the other colours, then ink (hides fill creeping into small shapes)
        cols_ = [k for k in order if layers[k][0] == "color"]
        if cols_:
            mainc = max(cols_, key=lambda k: layers[k][1].sum())
            order = [mainc] + [k for k in order if k != mainc]
    stack = [("#ffffff", cut, "sticker-border")]
    if nest:
        sm = nest_smooth([P(layers[k][1]) for k in order], nest, nest_thicken, local)
        for key, m in zip(order, sm):
            stack.append((key, m, f"{layers[key][0]}-{key[1:]}"))
        stack = [(c, m[y0:y1, x0:x1], i) for c, m, i in stack]
        return write_svg(name, stack, (y1 - y0, x1 - x0), name.replace("-", " ")), order
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
    img = cv2.imread(f"{W}/{cfg.get('src', name)}_crop.png")
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
        # smooth the paper outline itself before tracing
        shape = cv2.GaussianBlur((lbl2 != -1).astype(np.float32), (0, 0), 5) > 0.5
        lbl2[~shape] = -1
        lbl2[shape & (lbl2 == -1)] = 0
        return compose(name, lbl2, roles, nest=cfg.get("nest", 0), nest_thicken=cfg.get("nest_thicken", 0))

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
        ink = lv < cfg["thresh"]
        if cfg.get("adaptive"):
            bgl = cv2.GaussianBlur(cv2.morphologyEx(lv, cv2.MORPH_CLOSE, disk(9)), (0, 0), 9)
            ink &= lv < bgl * cfg["adaptive"]
        if cfg.get("dots"):
            ink = remove_dots(ink, cfg["dots"])
        lbl = np.where(ink, 1, 0)
        lbl[~region] = -1
        cols = {0: (255, 255, 255), 1: INK_BLACK}
    else:
        lbl, cols, _ = cluster(img, region, cfg["k"], cfg.get("merge", 16), cfg.get("pencil"), paper_mask)
    roles = roles_for(cols, cfg)
    groups = list(cols)
    if cfg.get("rescue"):
        Lr = lab(cv2.medianBlur(img, 5))
        chr_ = np.hypot(Lr[..., 1] - 128, Lr[..., 2] - 128)
        pg = [g for g, (r, _) in roles.items() if r == "paper"]
        m = drop_small(np.isin(lbl, pg) & (chr_ > cfg["rescue"]) & region, 25)
        if m.any():
            gid = max(groups) + 1
            lbl[m] = gid
            c = np.median(img[m].astype(float), 0)[::-1]
            cols[gid] = c
            roles[gid] = roles_for({gid: c}, {})[gid]
            groups.append(gid)
    if cfg.get("text_ink"):
        # pen lettering on the white paper, outside the main outlined shape
        blk0 = [g for g, (r, c) in roles.items() if r == "ink" and c == INK_BLACK]
        if blk0:
            lt = lab(cv2.bilateralFilter(img, 7, 25, 5))[..., 0]
            bm = largest(np.isin(lbl, blk0))
            hull = ndi.binary_fill_holes(cv2.morphologyEx(bm.astype(np.uint8), cv2.MORPH_CLOSE, disk(5)))
            outside = ~cv2.dilate(hull.astype(np.uint8), disk(6)).astype(bool)
            txt = region & outside & (lt < cfg["text_ink"])
            txt = drop_small(txt, 15)
            lbl[txt] = blk0[0]
    if cfg.get("dark_to_black"):
        for g, (r, c) in list(roles.items()):
            if r == "color" and chroma_L(cols[g])[0] < cfg["dark_to_black"]:
                roles[g] = ("ink", INK_BLACK)
    if cfg.get("flat_colors"):
        roles = flatten_colors(roles, lbl, cfg["flat_colors"])
    if cfg.get("only_yellow"):
        def _b(c):
            return cv2.cvtColor(np.uint8([[list(c)[::-1]]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float)[2]
        for g, (r, c) in list(roles.items()):
            if r == "color" and _b(c) < 150:
                roles[g] = ("ink", INK_BLACK)
    if cfg.get("text_bands"):
        blk0 = [g for g, (r, c) in roles.items() if r == "ink" and c == INK_BLACK]
        if blk0:
            lt = lab(cv2.bilateralFilter(img, 7, 25, 5))[..., 0]
            for (y0f, y1f) in cfg["text_bands"]:
                band = np.zeros_like(region); band[int(y0f * H):int(y1f * H)] = True
                txt = drop_small(band & region & (lt < cfg.get("text_L", 140)), 12)
                pg0 = [g for g, (r, _) in roles.items() if r == "paper"]
                if pg0:
                    lbl[band & region & ~txt & np.isin(lbl, blk0)] = pg0[0]
                lbl[txt] = blk0[0]
    if cfg.get("chroma_split"):
        # black ink is near-neutral; purple fill is not. Split them by colour strength.
        blk = [g for g, (r, c) in roles.items() if r == "ink" and c == INK_BLACK]
        def lab_b(c):
            return cv2.cvtColor(np.uint8([[list(c)[::-1]]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float)[2]
        pur = [g for g, (r, c) in roles.items() if r == "color" and lab_b(c) < 128]
        if blk and pur:
            Lc = lab(cv2.medianBlur(cv2.bilateralFilter(img, 9, 30, 5), 7))
            chs = np.hypot(Lc[..., 1] - 128, Lc[..., 2] - 128)
            main = max(pur, key=lambda g: (lbl == g).sum())
            m = np.isin(lbl, blk + pur)
            lbl[m & (chs < cfg["chroma_split"])] = blk[0]
            lbl[m & (chs >= cfg["chroma_split"])] = main

    # 2x gaussian vote
    sigma = cfg.get("sigma", 1.6)
    votes = [cv2.GaussianBlur(up((lbl == g).astype(np.float32), cv2.INTER_LINEAR), (0, 0), sigma) for g in groups]
    votes.append(up((lbl == -1).astype(np.float32)) * 10)
    win = np.argmax(np.stack(votes, -1), -1)
    lut = np.array(groups + [-1])
    lbl2 = lut[win]
    lbl2 = absorb_specks(lbl2, int(cfg.get("speck", 60) * scale * scale))

    if cfg["kind"] == "sticker" and cfg.get("inside", True):
        # colours live inside the black outline; anything outside is paper-edge fringe
        blk = [g for g, (r, c) in roles.items() if r == "ink" and c == INK_BLACK]
        if blk:
            bm = np.isin(lbl2, blk).astype(np.uint8)
            hull = ndi.binary_fill_holes(cv2.morphologyEx(bm, cv2.MORPH_CLOSE, disk(4)))
            hull = cv2.dilate(hull.astype(np.uint8), disk(2)).astype(bool)
            for g, (r, c) in roles.items():
                if r == "color":
                    lbl2[(lbl2 == g) & ~hull] = -1

    if cfg.get("smooth"):
        lbl2 = smooth_pass(lbl2, roles, cfg["smooth"] * scale / 2, cfg.get("thicken", 0) * scale / 2,
                           int(cfg.get("speck", 60) * scale * scale))

    blk_ids = [g for g, (r, c) in roles.items() if r == "ink" and c == INK_BLACK]
    ink_all = [g for g, (r, c) in roles.items() if r == "ink"]
    pg_all = [g for g, (r, _) in roles.items() if r == "paper"]
    def reassign_nearest(lbl_, mask_, exclude):
        """Give pixels in mask_ the label of the nearest pixel not in `exclude` labels."""
        keep = ~np.isin(lbl_, exclude) & ~mask_
        _, (iy, ix) = ndi.distance_transform_edt(~keep, return_indices=True)
        lbl_[mask_] = lbl_[iy, ix][mask_]
    if cfg.get("islands") and blk_ids:
        # small black specks floating inside colour areas
        bm = np.isin(lbl2, blk_ids).astype(np.uint8)
        k, cc, st, _ = cv2.connectedComponentsWithStats(bm, 8)
        small = st[:, cv2.CC_STAT_AREA] < cfg["islands"] * scale * scale
        small[0] = False
        # main fills only: the small blue "rescue" colour also edges the pen lettering
        tot = max(1, (lbl2 != -1).sum())
        col_ids_ = [g for g, (r, _) in roles.items() if r == "color" and (lbl2 == g).sum() > 0.05 * tot]
        rm = np.zeros(k, bool)
        for i in np.where(small)[0]:
            x, y, w, h = st[i, :4]
            y0_, y1_, x0_, x1_ = max(0, y - 6), y + h + 6, max(0, x - 6), x + w + 6
            comp = cc[y0_:y1_, x0_:x1_] == i
            ring_ = cv2.dilate(comp.astype(np.uint8), disk(4)).astype(bool) & ~comp
            sub = lbl2[y0_:y1_, x0_:x1_][ring_]
            # only specks sitting inside a colour area; lettering on paper stays
            rm[i] = sub.size > 0 and np.isin(sub, col_ids_).mean() > 0.7
        reassign_nearest(lbl2, rm[cc], blk_ids)
    if cfg.get("restroke") and blk_ids:
        # rebuild the main outline as an even-width line along its centre
        from skimage.morphology import skeletonize
        bm = np.isin(lbl2, blk_ids)
        dt = cv2.distanceTransform(bm.astype(np.uint8), cv2.DIST_L2, 5)
        k, cc = cv2.connectedComponents(bm.astype(np.uint8), connectivity=8)
        mx = np.asarray(ndi.maximum(dt, cc, np.arange(k))); mx[0] = 0
        main = (mx >= cfg["restroke"])[cc] & bm            # thick parts only, text stays as is
        pre = cv2.GaussianBlur(main.astype(np.float32), (0, 0), cfg["restroke"]) > 0.5
        sk = skeletonize(pre)
        r = float(np.median(dt[sk & main])) if (sk & main).any() else cfg["restroke"]
        new = cv2.dilate(sk.astype(np.uint8), disk(r)).astype(bool)
        new = cv2.GaussianBlur(new.astype(np.float32), (0, 0), r / 2) > 0.5
        reassign_nearest(lbl2, main & ~new, blk_ids)
        lbl2[new] = blk_ids[0]
    if cfg.get("outline") and blk_ids:
        # outer outline as one even-width smooth band; inner details untouched
        art = (lbl2 != -1) & ~np.isin(lbl2, pg_all)
        main = largest(art)
        sil = cv2.GaussianBlur(main.astype(np.float32), (0, 0), cfg["outline"]) > 0.5
        din = cv2.distanceTransform(sil.astype(np.uint8), cv2.DIST_L2, 5)
        bm = np.isin(lbl2, blk_ids)
        S_ = max(sil.shape)
        t = 2 * float(np.median(din[bm & sil & (din < 0.06 * S_)]))
        band = sil & (din <= t)
        extra_blk = bm & main & ~band & (din < 1.25 * t)
        lbl2[main & ~sil] = -1
        reassign_nearest(lbl2, extra_blk, blk_ids + [-1])
        lbl2[band] = blk_ids[0]
    if cfg.get("fill_outline") and blk_ids:
        col_ids_ = [g for g, (r, _) in roles.items() if r == "color"]
        art = (lbl2 != -1) & ~np.isin(lbl2, pg_all)
        main = largest(art)
        inner = np.isin(lbl2, col_ids_) & main
        R_ = cfg["fill_outline"]
        inner = cv2.morphologyEx(inner.astype(np.uint8), cv2.MORPH_CLOSE, disk(R_))
        inner = ndi.binary_fill_holes(inner)
        inner = cv2.GaussianBlur(inner.astype(np.float32), (0, 0), R_) > 0.5
        bm = np.isin(lbl2, blk_ids)
        if cfg.get("eye_ring"):
            # eye = brightest colour; redraw its ring and keep a strip of fill above it
            eye_id = max(col_ids_, key=lambda g: chroma_L(roles[g][1])[0])
            eye = largest(lbl2 == eye_id)
            eye = cv2.GaussianBlur(eye.astype(np.float32), (0, 0), 4) > 0.5
            de = cv2.distanceTransform((~eye).astype(np.uint8), cv2.DIST_L2, 5)
            te, gap = cfg["eye_ring"]
            inner = inner | (de <= te + gap)
            # the black pocket around the eye opens onto the outline; close it near the eye only
            near_eye = de <= te + gap + 130
            closed = ndi.binary_fill_holes(cv2.morphologyEx(inner.astype(np.uint8), cv2.MORPH_CLOSE, disk(70)))
            inner = inner | (closed & near_eye)
            inner = cv2.GaussianBlur(inner.astype(np.float32), (0, 0), 6) > 0.5
        dout = cv2.distanceTransform((~inner).astype(np.uint8), cv2.DIST_L2, 5)
        t = float(np.median(dout[bm & main & ~inner & (dout < 0.05 * max(lbl2.shape))])) * 2
        outer = cv2.GaussianBlur((dout <= t).astype(np.float32), (0, 0), R_ / 2) > 0.5
        lbl2[main & ~outer] = -1
        lbl2[outer & ~inner] = blk_ids[0]
        # gaps inside the interior take the fill colour; black inside stays (eye ring etc.)
        hole = inner & (lbl2 == -1)
        reassign_nearest(lbl2, hole, [-1])
        if cfg.get("eye_ring"):
            main_col = max([g for g in col_ids_ if g != eye_id], key=lambda g: (lbl2 == g).sum())
            ins = inner & ~eye
            lbl2[ins & np.isin(lbl2, blk_ids) & (de > te)] = main_col
            lbl2[ins & (de <= te)] = blk_ids[0]
            lbl2[eye] = eye_id
            # stray black bits left inside the fill
            bmi = (np.isin(lbl2, blk_ids) & inner & (de > te)).astype(np.uint8)
            k3, cc3, st3, _ = cv2.connectedComponentsWithStats(bmi, 8)
            band_ = np.isin(lbl2, blk_ids) & ~inner
            touch = np.unique(cc3[cv2.dilate(band_.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)])
            small3 = st3[:, cv2.CC_STAT_AREA] < 4000
            small3[0] = False; small3[touch] = False
            lbl2[small3[cc3]] = main_col
            # colour outside the main shape (bits near the lettering) becomes paper
            outside = (lbl2 != -1) & ~main & np.isin(lbl2, col_ids_)
            lbl2[outside] = pg_all[0] if pg_all else -1
    if cfg.get("ring") and blk_ids:
        # outer outline as one clean ellipse ring
        art = lbl2 != -1
        cs, _ = cv2.findContours(art.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cs, key=cv2.contourArea)
        (cx, cy), (aw, ah), ang = cv2.fitEllipse(c)
        bm = np.isin(lbl2, blk_ids)
        dt = cv2.distanceTransform(bm.astype(np.uint8), cv2.DIST_L2, 5)
        outer = np.zeros(art.shape, np.uint8)
        cv2.ellipse(outer, ((cx, cy), (aw * 0.985, ah * 0.985), ang), 1, -1)
        edge = outer.astype(bool) & ~cv2.erode(outer, disk(8)).astype(bool)
        t = cfg["ring"] * max(aw, ah)
        inner = np.zeros_like(outer)
        cv2.ellipse(inner, ((cx, cy), (aw * 0.985 - 2 * t, ah * 0.985 - 2 * t), ang), 1, -1)
        ring = outer.astype(bool) & ~inner.astype(bool)
        lbl2[~outer.astype(bool)] = -1
        # black that hugged the old ring's inner edge goes back to the fill
        hug = bm & inner.astype(bool) & ~cv2.erode(inner, disk(int(t * 0.6))).astype(bool)
        hug &= ~ndi.binary_dilation(~bm | ~inner.astype(bool), iterations=0) if False else hug
        reassign_nearest(lbl2, hug & ~ring, blk_ids + [-1])
        lbl2[ring] = blk_ids[0]
        if cfg.get("ring_islands"):
            insd = inner.astype(bool) & ~ring
            bm2 = (np.isin(lbl2, blk_ids) & insd).astype(np.uint8)
            k2, cc2, st2, _ = cv2.connectedComponentsWithStats(bm2, 8)
            sm2 = st2[:, cv2.CC_STAT_AREA] < cfg["ring_islands"] * scale * scale
            sm2[0] = False
            reassign_nearest(lbl2, sm2[cc2], blk_ids + [-1])
            # black hugging the ring's inner side: open it away
            near = insd & ~cv2.erode(inner, disk(int(t * 1.2))).astype(bool)
            bm3 = np.isin(lbl2, blk_ids) & near
            opened = cv2.morphologyEx(np.isin(lbl2, blk_ids).astype(np.uint8), cv2.MORPH_OPEN, disk(int(t * 0.6))).astype(bool)
            reassign_nearest(lbl2, bm3 & ~opened, blk_ids + [-1])
    if cfg.get("solid") and ink_all:
        # merge sketchy parallel strokes into bold solid lines
        im_ = np.isin(lbl2, ink_all)
        r = cfg["solid"] * scale
        sol = cv2.morphologyEx(im_.astype(np.uint8), cv2.MORPH_CLOSE, disk(r))
        sol = cv2.GaussianBlur(sol.astype(np.float32), (0, 0), r / 2) > 0.5
        sol = drop_small(sol, int(400 * scale * scale))
        lbl2[im_ & ~sol] = pg_all[0] if pg_all else -1
        lbl2[sol] = ink_all[0]
    if cfg.get("keep_at"):
        # keep only the figure under one point (e.g. the right-hand crow)
        im_ = np.isin(lbl2, ink_all)
        grp = cv2.dilate(im_.astype(np.uint8), disk(cfg.get("keep_join", 4) * scale))
        k, cc = cv2.connectedComponents(grp, connectivity=8)
        fx, fy = cfg["keep_at"]
        hy, hx = lbl2.shape
        pick = cc[int(fy * hy), int(fx * hx)]
        if pick == 0:
            ys, xs = np.where(cc > 0)
            d = (ys - fy * hy) ** 2 + (xs - fx * hx) ** 2
            pick = cc[ys[d.argmin()], xs[d.argmin()]]
        drop = im_ & (cc != pick)
        lbl2[drop] = pg_all[0] if pg_all else -1
    if cfg.get("thin_to_black") and blk_ids:
        # thin strokes that ended up in the main fill colour (e.g. navy pen lettering) go to black
        col_ids = [g for g, (r, _) in roles.items() if r == "color"]
        if col_ids:
            main = max(col_ids, key=lambda g: (lbl2 == g).sum())
            same = [g for g in col_ids if roles[g][1] == roles[main][1]]
            m = np.isin(lbl2, same).astype(np.uint8)
            dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
            k, cc = cv2.connectedComponents(m, connectivity=8)
            mx = np.asarray(ndi.maximum(dt, cc, np.arange(k))); mx[0] = 1e9
            lbl2[(mx < cfg["thin_to_black"])[cc] & m.astype(bool)] = blk_ids[0]
    if cfg.get("close_bottom") and blk_ids:
        # the sticker was hand-cut through the art; finish the bottom with an outline band
        pg = [g for g, (r, _) in roles.items() if r == "paper"]
        art = (lbl2 != -1)
        cols_ = np.where(art.any(0))[0]
        bottom = np.full(art.shape[1], -1)
        bottom[cols_] = art.shape[0] - 1 - np.argmax(art[::-1, cols_], 0)
        t = int(cfg["close_bottom"] * art.shape[0])
        yy = np.arange(art.shape[0])[:, None]
        # only where the bottom edge is low (the cut), not the raised arm etc.
        low = bottom >= np.percentile(bottom[cols_], 40)
        if cfg.get("square"):
            ybase = int(np.percentile(bottom[cols_][low[cols_]], cfg["square"]))
            lbl2[yy[:, 0] > ybase, :] = -1
            art = lbl2 != -1
            band = art & (yy > ybase - t)
        else:
            band = art & (yy > (bottom - t)[None, :]) & low[None, :]
        band = cv2.morphologyEx(band.astype(np.uint8), cv2.MORPH_CLOSE, disk(6)).astype(bool) & art
        lbl2[band] = blk_ids[0]

    if cfg.get("fill_specks") and blk_ids:
        nb = (lbl2 != -1) & ~np.isin(lbl2, blk_ids)
        k_, cc_, st_, _ = cv2.connectedComponentsWithStats(nb.astype(np.uint8), 4)
        for i in range(1, k_):
            if st_[i, cv2.CC_STAT_AREA] >= cfg["fill_specks"] * scale * scale:
                continue
            x, y, w, h = st_[i, :4]
            sub = cc_[max(0, y - 3):y + h + 3, max(0, x - 3):x + w + 3] == i
            ring_ = cv2.dilate(sub.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~sub
            ln = lbl2[max(0, y - 3):y + h + 3, max(0, x - 3):x + w + 3][ring_]
            if ln.size and np.isin(ln, blk_ids).mean() > 0.95:
                lbl2[cc_ == i] = blk_ids[0]
    if cfg.get("black_below") and blk_ids:
        col_ids_ = [g for g, (r, _) in roles.items() if r == "color"]
        yy_ = np.arange(lbl2.shape[0])[:, None]
        lbl2[np.isin(lbl2, col_ids_) & (yy_ > cfg["black_below"] * lbl2.shape[0])] = blk_ids[0]
    if cfg.get("bottom_line") and ink_all:
        # close an open drawing with a straight line along its bottom
        im_ = np.isin(lbl2, ink_all)
        ys, xs = np.where(im_)
        ymax = ys.max()
        low_ = ys > ymax - cfg["bottom_line"] * lbl2.shape[0]
        x0_, x1_ = xs[low_].min(), xs[low_].max()
        dt = cv2.distanceTransform(im_.astype(np.uint8), cv2.DIST_L2, 5)
        w_ = max(4, int(2 * np.percentile(dt[im_], 90)))
        lbl2[ymax - w_ + 1:ymax + 1, x0_:x1_ + 1] = ink_all[0]

    extra = None
    if cfg.get("fill"):
        ink_groups = [g for g, (r, _) in roles.items() if r != "paper"]
        lines = np.isin(lbl2, ink_groups)
        hull = ndi.binary_fill_holes(cv2.morphologyEx(lines.astype(np.uint8), cv2.MORPH_CLOSE, disk(6)))
        img2 = up(cv2.medianBlur(img, 5), cv2.INTER_LINEAR)
        extra = region_fill(lbl2, ink_groups, img2, hull, min_area=int(40 * scale * scale), min_chroma=cfg.get("fill_chroma", 6), n_colors=cfg.get("fill_colors", 5))
    import vec as _v
    from functools import partial
    _orig = getattr(_v, "_potrace_orig", _v.potrace_paths)
    _v._potrace_orig = _orig
    _v.potrace_paths = partial(_orig, alpha=cfg.get("alpha", 1.0), opt=cfg.get("opt", 0.3))
    try:
        out = compose(name, lbl2, roles, extra, cfg.get("margin", 0.045),
                      cfg.get("nest", 0), cfg.get("nest_thicken", 0), cfg.get("lift", False), cfg.get("local_thin", False))
    finally:
        _v.potrace_paths = _orig
    if cfg.get("rotate"):
        rotate_svg(out[0], cfg["rotate"])
    return out


def remove_dots(ink, max_area):
    """Drop small roundish blobs (paper texture) but keep small dashes (elongated)."""
    n, cc, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    keep = np.ones(n, bool); keep[0] = False
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]
        if a >= max_area:
            continue
        w, h = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        elong = max(w, h) / max(1, min(w, h))
        if a < max_area * 0.25 or elong < 2.2:
            keep[i] = False
    return keep[cc]


def flatten_colors(roles, lbl, n):
    """Merge colour groups (never ink) down to n flat colours, weighted by area."""
    cg = [g for g, (r, _) in roles.items() if r == "color"]
    if len(cg) <= n:
        return roles
    def to_lab(c):
        return cv2.cvtColor(np.uint8([[list(c)[::-1]]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float)
    clusters = [([g], to_lab(roles[g][1]), float((lbl == g).sum()) + 1) for g in cg]
    while len(clusters) > n:
        best = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = np.linalg.norm((clusters[i][1] - clusters[j][1]) * np.array([0.5, 1, 1]))
                if best is None or d < best[0]:
                    best = (d, i, j)
        _, i, j = best
        gi, li, wi = clusters[i]; gj, lj, wj = clusters[j]
        merged = (gi + gj, (li * wi + lj * wj) / (wi + wj), wi + wj)
        clusters = [c for k, c in enumerate(clusters) if k not in (i, j)] + [merged]
    out = dict(roles)
    for gs, lc, _ in clusters:
        # the dominant member's colour reads truer than an average
        dom = max(gs, key=lambda g: chroma_L(roles[g][1])[1])  # most vivid shade
        for g in gs:
            out[g] = ("color", roles[dom][1])
    return out


def smooth_pass(lbl2, roles, sigma, thicken, speck):
    """Second, stronger vote on already-clean labels. Thin ink (text, fine lines) is protected."""
    ink_ids = [g for g, (r, c) in roles.items() if r == "ink"]
    inkm = np.isin(lbl2, ink_ids)
    dt = cv2.distanceTransform(inkm.astype(np.uint8), cv2.DIST_L2, 5)
    n, cc = cv2.connectedComponents(inkm.astype(np.uint8), connectivity=8)
    mx = ndi.maximum(dt, cc, np.arange(n)) if n > 1 else np.zeros(1)
    mx = np.asarray(mx); mx[0] = 1e9
    thin = (mx < sigma * 1.6)[cc] & inkm
    thin_lbl = lbl2.copy()
    # merge identical colours into one id so the vote doesn't split them
    key = {}
    for g, (r, c) in roles.items():
        key[g] = (r, tuple(c) if c is not None else None)
    ids = sorted(set(np.unique(lbl2)))
    canon = {}
    for g in ids:
        if g == -1:
            canon[g] = -1; continue
        k = key.get(g)
        canon[g] = next((h for h in ids if h != -1 and key.get(h) == k), g)
    lut_ids = np.array(ids)
    lbl_c = np.vectorize(canon.get)(lbl2) if len(ids) < 50 else lbl2
    uniq = sorted(set(np.unique(lbl_c)))
    votes = [cv2.GaussianBlur((lbl_c == g).astype(np.float32), (0, 0), sigma) for g in uniq]
    out = np.array(uniq)[np.argmax(np.stack(votes, -1), -1)]
    # outside stays outside, thin ink restored (optionally thickened onto paper)
    out[lbl2 == -1] = -1
    if thin.any():
        t = thin
        if thicken > 0:
            pg = [g for g, (r, _) in roles.items() if r == "paper"]
            grown = cv2.dilate(thin.astype(np.uint8), disk(thicken)).astype(bool)
            t = thin | (grown & (np.isin(out, pg) | (out == -1)))
            t &= cv2.dilate((lbl2 != -1).astype(np.uint8), disk(thicken + 2)).astype(bool)
        src = np.where(thin, lbl2, -9)
        # nearest thin label for grown pixels
        _, (iy, ix) = ndi.distance_transform_edt(~thin, return_indices=True)
        out[t] = lbl2[iy, ix][t]
    return absorb_specks(out, speck)


def rotate_svg(path, deg):
    import re, subprocess
    s = open(path).read()
    m = re.search(r'viewBox="0 0 (\d+) (\d+)"', s)
    w, h = int(m.group(1)), int(m.group(2))
    head_end = s.index("</title>") + len("</title>")
    body = s[head_end:s.rindex("</svg>")]
    if deg == -90:
        tr = f"translate(0,{w}) rotate(-90)"
    else:
        tr = f"translate({h},0) rotate(90)"
    title = s[s.index("<title>"):head_end]
    s2 = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {h} {w}" width="{h}" height="{w}">\n  {title}\n'
          f'  <g transform="{tr}">{body}</g>\n</svg>\n')
    open(path, "w").write(s2)
    name = os.path.basename(path)[:-4]
    subprocess.run(["rsvg-convert", "-w", "900", "-o", f"{OUT}/png/{name}.png", path])


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
