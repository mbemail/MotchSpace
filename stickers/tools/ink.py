import cv2,numpy as np
from scipy import ndimage as ndi
from vec import lab, largest, disk
def ink_region(img, dark_t=None, group=0.035, close=0.012, extra=None):
    """Art region grown from the black ink: dark pixels clustered around the biggest dark blob,
    then holes filled (so colour fills inside outlines are included)."""
    h,w=img.shape[:2]; S=max(h,w)
    l=lab(cv2.medianBlur(img,5))[...,0]
    if dark_t is None:
        dark_t=min(70, np.percentile(l,3)+35)
    dark=(l<dark_t).astype(np.uint8)
    dark=cv2.morphologyEx(dark,cv2.MORPH_OPEN,disk(1))
    # ignore dark stuff touching the crop edge (mat, shadows)
    n,lb,st,_=cv2.connectedComponentsWithStats(dark,8)
    edge=set(np.unique(np.concatenate([lb[0],lb[-1],lb[:,0],lb[:,-1]])))
    keep=np.ones(n,bool); keep[0]=False
    for i in edge: keep[i]=False
    keep[st[:,cv2.CC_STAT_AREA]<S*0.02]=False
    dark=keep[lb]
    g=cv2.dilate(dark.astype(np.uint8),disk(S*group))
    g=largest(g)
    ink=dark&g
    c=max(2,int(S*close)); pad=c*2
    r=cv2.copyMakeBorder(ink.astype(np.uint8),pad,pad,pad,pad,cv2.BORDER_CONSTANT,0)
    r=cv2.morphologyEx(r,cv2.MORPH_CLOSE,disk(c))[pad:-pad,pad:-pad]
    r=ndi.binary_fill_holes(r)
    if extra is not None: r|=extra
    return r.astype(bool), dark_t
if __name__=="__main__":
    tiles=[]
    for n in ["stop-sign-guy","moon-eyes","bald-guy","hooded-head","dress-figure","crow","round-face","ha-creature","my-feewins","light-of-my-life"]:
        img=cv2.imread(f"work/{n}_crop.png"); s=1000/max(img.shape[:2]); img=cv2.resize(img,None,fx=s,fy=s,interpolation=cv2.INTER_AREA)
        r,t=ink_region(img); v=img.copy(); v[~r]=(v[~r]*0.25).astype(np.uint8); print(n,t)
        tt=np.full((1000,1000,3),80,np.uint8); tt[:v.shape[0],:v.shape[1]]=v; tiles.append(cv2.resize(tt,(350,350)))
    while len(tiles)%4: tiles.append(np.zeros((350,350,3),np.uint8))
    cv2.imwrite("prev/regions.png",np.vstack([np.hstack(tiles[i:i+4]) for i in range(0,len(tiles),4)]))
