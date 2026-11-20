import cv2, numpy as np, vtracer, os, re, subprocess
from scipy import ndimage as ndi
img=cv2.imread("work/damage_crop.png")
s=1200/max(img.shape[:2]); img=cv2.resize(img,None,fx=s,fy=s,interpolation=cv2.INTER_AREA)
f=cv2.bilateralFilter(img,7,25,5); f=cv2.medianBlur(f,3)
L=cv2.cvtColor(f,cv2.COLOR_BGR2LAB).astype(float)
white=((L[...,0]>205)&(np.hypot(L[...,1]-128,L[...,2]-128)<10))|((L[...,0]>135)&(np.hypot(L[...,1]-128,L[...,2]-128)<9))
n,lb,st,_=cv2.connectedComponentsWithStats((~white).astype(np.uint8),4)
piece=ndi.binary_fill_holes(lb==1+np.argmax(st[1:,4]))
piece=cv2.morphologyEx(piece.astype(np.uint8),cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))).astype(bool)
piece=cv2.erode(piece.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
gain=245/np.median(f[white].astype(float),0)
g=np.clip(f*gain,0,255).astype(np.uint8)
hsv=cv2.cvtColor(g,cv2.COLOR_BGR2HSV).astype(float); hsv[...,1]=np.clip(hsv[...,1]*1.6,0,255); g=cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR)
# base fill = kraft colour so no transparent gaps inside the piece
kraft=np.median(g[piece].astype(float),0)
g[~piece]=255
rgba=cv2.cvtColor(g,cv2.COLOR_BGR2RGBA); rgba[~piece,3]=0
cv2.imwrite("work/damage_clean.png",cv2.cvtColor(rgba,cv2.COLOR_RGBA2BGRA))
vtracer.convert_image_to_svg_py("work/damage_clean.png","work/damage_vt.svg",colormode="color",hierarchical="stacked",
    mode="spline",filter_speckle=5,color_precision=6,layer_difference=14,corner_threshold=60,length_threshold=4.0,splice_threshold=45,path_precision=1)
vt=open("work/damage_vt.svg").read()
H,W=piece.shape
body=re.sub(r'^.*?<svg[^>]*>','',vt,flags=re.S).replace('</svg>','')
# white die-cut border + kraft base, traced with potrace from the piece mask
import sys; sys.argv=['x']
from vec import potrace_paths, border_from_art, tight
pad=110
P=np.pad(piece,pad)
cut=border_from_art(P,42)
dc,tc=potrace_paths(cut); dp,tp=potrace_paths(P)
y0,y1,x0,x1=tight(cut,4)
svg=(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0} {y0} {x1-x0} {y1-y0}" width="{x1-x0}" height="{y1-y0}">\n<title>damage</title>\n'
     f'<g id="sticker-border" fill="#ffffff" transform="{tc}"><path d="{dc}"/></g>\n'
     f'<g id="kraft-base" fill="#%02x%02x%02x" transform="{tp}"><path d="{dp}"/></g>\n'%tuple(int(x) for x in kraft[::-1])+
     f'<g id="crayon" transform="translate({pad},{pad})">{body}</g>\n</svg>\n')
open("out/damage.svg","w").write(svg)
subprocess.run(["rsvg-convert","-w","900","-o","out/png/damage.png","out/damage.svg"])
print(os.path.getsize("out/damage.svg")//1024,"KB")
