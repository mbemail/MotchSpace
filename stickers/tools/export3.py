import json, os, shutil, re
from PIL import Image
R = "/home/user/MotchSpace/stickers"
for d in ("svg", "svg-bw", "photo"):
    os.makedirs(f"{R}/{d}", exist_ok=True)
exec(open("export.py").read().split("items=[]")[0])  # reuse ORDER, TITLES, NOTES
R2 = {"bald-guy": "FIX: Smooth, not so ribbed look", "dress-figure": "FIX: Just a bit cut off at bottom",
      "fish": "FIX: Make smooth vector", "ha-creature": "FIX: Smooth lines",
      "hooded-head": "FIX: Smoother, very bumpy outline on left side", "light-of-my-life": "FIX: Lost text detail",
      "moon-eyes": "FIX: Retry it's bad", "my-feewins": "FIX: Text thicker, smooth lines",
      "round-face": "FIX: Better, smooth out remaining lines", "stop-sign-guy": "FIX: Retry",
      "woodland-bird": "FIX: Bottom circle cut out"}
N3 = {"dress-figure": "The sticker was hand-cut through the art, so I closed the bottom with a black outline.",
      "woodland-bird": "Circle is back and filled. The pink body and yellow sun now fill in too.",
      "light-of-my-life": "Less smoothing on the lettering to keep its detail.",
      "stop-sign-guy": "Third try. If it still fails, it may need a sharper photo.",
      "moon-eyes": "Purple hourglass and bottom band are back."}
items = []
for n in ORDER:
    shutil.copy(f"out/{n}.svg", f"{R}/svg/{n}.svg")
    shutil.copy(f"out_bw/{n}.svg", f"{R}/svg-bw/{n}.svg")
    im = Image.open(f"work/{n}_crop.png").convert("RGB"); im.thumbnail((900, 900)); im.save(f"{R}/photo/{n}.jpg", quality=82)
def ncol(p):
    return len({c for c in re.findall(r'fill="(#[0-9a-f]{6})"', open(p).read()) if c != "#ffffff"})
for n in ORDER:
    if n in R2:
        items.append(dict(id=n, title=TITLES.get(n, n.replace("-", " ")), svg=f"svg/{n}.svg", photo=f"photo/{n}.jpg",
                          colors="many" if n == "damage" else ncol(f"out/{n}.svg"), note=N3.get(n, ""), r1=R2[n]))
for n in ORDER:
    items.append(dict(id=f"{n}-bw", title=TITLES.get(n, n.replace("-", " ")) + " (black & white)", svg=f"svg-bw/{n}.svg",
                      photo=f"photo/{n}.jpg", colors=1, note="New: black-and-white version.", r1=""))
json.dump(items, open(f"{R}/manifest.json", "w"), indent=1, ensure_ascii=False)
print(len(items), "cards")
