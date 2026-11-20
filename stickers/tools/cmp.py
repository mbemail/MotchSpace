import sys, os
from PIL import Image, ImageDraw
out=os.environ.get("OUT","out"); names=sys.argv[1:]
tiles=[]
for n in names:
    a=Image.open(f"work/{n}_crop.png").convert("RGB"); a.thumbnail((420,420))
    b=Image.open(f"{out}/png/{n}.png").convert("RGBA"); b.thumbnail((420,420))
    bg=Image.new("RGBA",b.size,(60,60,70,255)); bg.alpha_composite(b)
    t=Image.new("RGB",(860,440),"grey"); t.paste(a,(0,20)); t.paste(bg.convert("RGB"),(440,20)); ImageDraw.Draw(t).text((4,4),n,fill="white"); tiles.append(t)
S=Image.new("RGB",(860*2,440*((len(tiles)+1)//2)),"black")
for i,t in enumerate(tiles): S.paste(t,((i%2)*860,(i//2)*440))
S.save(sys.stdout.buffer if False else "prev/cmp.png")
