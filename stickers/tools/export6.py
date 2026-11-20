import json, shutil, os
R = "/home/user/MotchSpace/stickers"
exec(open("export.py").read().split("items=[]")[0])
COLOR = {"ha-creature": ("FIX: Lost head line above eye", "Eye ring redrawn on its own, with the purple strip back between it and the head outline.")}
BW = {"ha-creature": ("ha-creature", "FIX: Same issue as colour", "Rebuilt from the fixed colour version."),
      "dress-figure": ("dress-figure-sq", "FIX: Needs body texture", "Purple flame texture back in the body; bottom closed with a black band."),
      "crow": ("crow-smooth", "FIX: Make top line solid, there are white splotches", "White specks inside the black lines filled.")}
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
