import json, shutil, os
R = "/home/user/MotchSpace/stickers"
exec(open("export.py").read().split("items=[]")[0])
COLOR = {"ha-creature": ("FIX: Leaking everywhere", "Outline rebuilt as one even band around the purple. Nothing outside it."),
         "round-face": ("FIX: Remove stray black in purple", "Specks and the black hugging the inside of the ring are gone.")}
BW = {"ha-creature": ("ha-creature", "FIX: Trim loose black bits", "Rebuilt from the new clean colour version."),
      "dress-figure": ("dress-figure-sq", "FIX: Fill in bottom", "Bottom filled solid black."),
      "happy-friday": ("happy-friday-crow", "FIX: Cut off feet", "Feet and toes no longer clipped."),
      "crow": ("crow-smooth", "PASS: Right side of head, make solid line", "Head line made solid.")}
items = []
for n, (r, note) in COLOR.items():
    shutil.copy(f"out/{n}.svg", f"{R}/svg/{n}.svg")
    items.append(dict(id=n, title=TITLES.get(n, n.replace("-", " ")), svg=f"svg/{n}.svg", photo=f"photo/{n}.jpg", colors="", note=note, r1=r))
for n, (src, r, note) in BW.items():
    shutil.copy(f"out_bw/{src}.svg", f"{R}/svg-bw/{n}.svg")
    items.append(dict(id=f"{n}-bw", title=TITLES.get(n, n.replace("-", " ")) + " (black & white)", svg=f"svg-bw/{n}.svg",
                      photo=f"photo/{n}.jpg", colors=1, note=note, r1=r))
json.dump(items, open(f"{R}/manifest.json", "w"), indent=1, ensure_ascii=False)
print(len(items), "cards")
