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
 "my-feewins":"Known miss: the blue tear came out white.",
}
R1={"crow":"PASS","hooded-head":"FIX: Smooth lines, remove coir","moon-eyes":"FAIL","round-face":"FIX: Remove texture and colour smooth lines",
 "ha-creature":"FIX: Lost top line remove texture and colour smooth lines","my-feewins":"FIX: Remove texture and colour smooth lines make text thicker",
 "dress-figure":"FIX: Lost bottom keep texture, remove colour","bald-guy":"FIX: Smooth out","stop-sign-guy":"FAIL","light-of-my-life":"FAIL",
 "fish":"PASS: Lost fins and tail fucked","woodland-bird":"FIX: Rotate left 90","easter-bunny":"FIX: Bottom cut off fill in if needed"}
ROUND2=["hooded-head","round-face","ha-creature","my-feewins","dress-figure","bald-guy","fish","woodland-bird","easter-bunny",
 "moon-eyes","stop-sign-guy","light-of-my-life"]
NOTES.update({"moon-eyes":"Retry with the cleaner method. Fail it again and I'll drop it.",
 "stop-sign-guy":"Retry with the cleaner method. Fail it again and I'll drop it.",
 "light-of-my-life":"Retry with the cleaner method. Fail it again and I'll drop it.",
 "dress-figure":"Read \"remove colour\" as one flat purple. Tell me if you meant black only.",
 "hooded-head":"Read \"coir\" as colour: one flat purple now.",
 "easter-bunny":"Whole bunny now. The egg came out mint green; say if you want it pale yellow. Also: the box credits Jim Oskineegish as the original artist.",
 "fish":"Fins and tail redrawn from the cut paper shape. The punched eye hole is a solid blue eye."})
items=[]
for n in ORDER:
    src=f"out/{n}.svg"
    if not os.path.exists(src): print("missing",n); continue
    shutil.copy(src,f"{R}/svg/{n}.svg")
    im=Image.open(f"work/{n}_crop.png").convert("RGB"); im.thumbnail((900,900)); im.save(f"{R}/photo/{n}.jpg",quality=82)
    svg=open(src).read()
    cols={c for c in re.findall(r'fill="(#[0-9a-f]{6})"',svg) if c!="#ffffff"}
    ncol = "many" if n=="damage" else len(cols)
    if n in ROUND2:
        items.append(dict(id=n,title=TITLES.get(n,n.replace("-"," ")),svg=f"svg/{n}.svg?v=2",photo=f"photo/{n}.jpg?v=2",colors=ncol,note=NOTES.get(n,""),r1=R1.get(n,"")))
items.sort(key=lambda i: ROUND2.index(i["id"]))
json.dump(items,open(f"{R}/manifest.json","w"),indent=1,ensure_ascii=False)
print(len(items),"items")
