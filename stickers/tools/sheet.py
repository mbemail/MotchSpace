from PIL import Image, ImageDraw
from crops import CROPS
import math
ims=[]
for k,(n,a,b,c,d,r) in CROPS.items():
    im=Image.open(f"src/{n:02d}.png"); W,H=im.size
    cr=im.crop((int(a*W),int(b*H),int(c*W),int(d*H)))
    if r: cr=cr.rotate(r,expand=True)
    cr.save(f"work/{k}_crop.png")
    cr.thumbnail((300,300)); t=Image.new("RGB",(300,320),"white"); t.paste(cr,(0,20)); ImageDraw.Draw(t).text((4,4),k,fill="black"); ims.append(t)
cols=5; rows=math.ceil(len(ims)/cols); S=Image.new("RGB",(cols*300,rows*320),"grey")
for i,t in enumerate(ims): S.paste(t,((i%cols)*300,(i//cols)*320))
S.save("prev/sheet.png")
