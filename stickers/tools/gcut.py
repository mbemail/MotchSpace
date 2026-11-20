import cv2,numpy as np
from scipy import ndimage as ndi
from vec import region_for_sticker, largest, disk
def gc_region(img):
    h,w=img.shape[:2]; s=600/max(h,w); sm=cv2.resize(img,None,fx=s,fy=s,interpolation=cv2.INTER_AREA)
    sm=cv2.bilateralFilter(sm,9,40,7)
    m=np.full(sm.shape[:2],cv2.GC_PR_BGD,np.uint8)
    # seed: threshold-based region as probable FG, its core as sure FG, border as sure BG
    r,_=region_for_sticker(sm)
    m[r]=cv2.GC_PR_FGD
    core=cv2.erode(r.astype(np.uint8),disk(10)).astype(bool); m[core]=cv2.GC_FGD
    b=max(3,int(0.01*max(sm.shape))); m[:b]=m[-b:]=cv2.GC_BGD; m[:,:b]=m[:,-b:]=cv2.GC_BGD
    bg=np.zeros((1,65)); fg=np.zeros((1,65))
    cv2.grabCut(sm,m,None,bg,fg,6,cv2.GC_INIT_WITH_MASK)
    fgm=(m==cv2.GC_FGD)|(m==cv2.GC_PR_FGD)
    fgm=largest(fgm); fgm=ndi.binary_fill_holes(fgm)
    fgm=cv2.morphologyEx(fgm.astype(np.uint8),cv2.MORPH_OPEN,disk(4))
    return cv2.resize(fgm,(w,h),interpolation=cv2.INTER_NEAREST).astype(bool)
if __name__=="__main__":
    tiles=[]
    for n in ["stop-sign-guy","moon-eyes","bald-guy","hooded-head","dress-figure","crow","round-face","ha-creature","my-feewins","light-of-my-life","fish"]:
        img=cv2.imread(f"work/{n}_crop.png"); s=1000/max(img.shape[:2]); img=cv2.resize(img,None,fx=s,fy=s,interpolation=cv2.INTER_AREA)
        r=gc_region(img); v=img.copy(); v[~r]=(v[~r]*0.25).astype(np.uint8)
        t=np.full((1000,1000,3),80,np.uint8); t[:v.shape[0],:v.shape[1]]=v; tiles.append(cv2.resize(t,(350,350)))
    while len(tiles)%4: tiles.append(np.zeros((350,350,3),np.uint8))
    cv2.imwrite("prev/regions.png",np.vstack([np.hstack(tiles[i:i+4]) for i in range(0,len(tiles),4)]))
