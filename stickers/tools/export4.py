import json, shutil
R = "/home/user/MotchSpace/stickers"
exec(open("export.py").read().split("items=[]")[0])  # ORDER, TITLES
COLOR = {"ha-creature": ("FIX: Make just smooth lines, trace over with single line vectors", "Outer outline redrawn as one even, smooth line."),
         "round-face": ("FIX: Make just a smooth line", "Outer ring is now one clean circle. Specks inside the purple removed."),
         "moon-eyes": ("FIX: Eyes yellow only", "Purple removed. Yellow eyes on black only."),
         "my-feewins": ("FIX: Some black leaking into the purple", "Black specks inside the purple cleaned up.")}
BW = {"crow": ("crow-smooth", "FIX: Fix top line", "Smoother outline, top of the head cleaned up."),
      "dress-figure": ("dress-figure-sq", "FIX: Needs squared off bottom", "Bottom cut straight across."),
      "f-off-cat": ("f-off-cat-sq", "FIX: Square off bottom of body", "Straight line closes the bottom of the body."),
      "happy-friday": ("happy-friday-crow", "FIX: Right crow only no happy friday", "Right crow only."),
      "light-of-my-life": ("light-of-my-life-txt", "FIX: Fix text", "Lettering re-traced on its own, less smoothing."),
      "red-thunderbird": ("red-thunderbird-solid", "FIX: Make less sketchy, trace in solid vectors", "Sketchy double lines merged into solid strokes."),
      "ha-creature": ("ha-creature", "PASS: Circle inside a", "Rebuilt from the new smoother colour version."),
      "my-feewins": ("my-feewins", "FIX: Leaking black", "Rebuilt from the cleaned colour version.")}
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
