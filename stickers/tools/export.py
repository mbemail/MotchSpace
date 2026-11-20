import json, re, os, shutil
from PIL import Image
R="/home/user/MotchSpace/stickers"
os.makedirs(f"{R}/svg",exist_ok=True); os.makedirs(f"{R}/photo",exist_ok=True)
ORDER=["crow","hooded-head","moon-eyes","round-face","ha-creature","my-feewins","dress-figure","bald-guy",
 "stop-sign-guy","light-of-my-life","f-off-cat","cat-face","red-thunderbird","happy-friday","fish",
 "woodland-bird","easter-bunny","damage","pixel-goose"]
TITLES={"my-feewins":"My Feewins","light-of-my-life":"You're the light of my life","f-off-cat":"F*** off cat",
 "ha-creature":"\"ha\" creature","damage":"DAMAGE","stop-sign-guy":"Stop sign guy (とまれ)"}
NOTES={
 "easter-bunny":"The box reads \"Art original - Jim Oskineegish\". If this design is his, get his OK before selling it.",
 "damage":"Crayon texture kept on purpose. Big file (about 5 MB).",
 "fish":"The punched eye hole is now a solid blue eye.",
 "woodland-bird":"Pale pencil fills were flattened into solid tints. Pencil scribbles under the bird were removed.",
 "light-of-my-life":"Hardest one to trace. Check the orange and the thin lines.",
 "pixel-goose":"Rebuilt pixel by pixel on a 15x16 grid.",
}
items=[]
for n in ORDER:
    src=f"out/{n}.svg"
    if not os.path.exists(src): print("missing",n); continue
    shutil.copy(src,f"{R}/svg/{n}.svg")
    im=Image.open(f"work/{n}_crop.png").convert("RGB"); im.thumbnail((900,900)); im.save(f"{R}/photo/{n}.jpg",quality=82)
    svg=open(src).read()
    cols={c for c in re.findall(r'fill="(#[0-9a-f]{6})"',svg) if c!="#ffffff"}
    ncol = "many" if n=="damage" else len(cols)
    items.append(dict(id=n,title=TITLES.get(n,n.replace("-"," ")),svg=f"svg/{n}.svg",photo=f"photo/{n}.jpg",colors=ncol,note=NOTES.get(n,"")))
json.dump(items,open(f"{R}/manifest.json","w"),indent=1,ensure_ascii=False)
print(len(items),"items")
