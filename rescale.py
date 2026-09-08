#!/usr/bin/env python3
"""Step 6b: rescale synthetic object geometry to the real medians measured on the
217-frame public train split. Points inside each box are scaled about the box
BOTTOM (objects stay on the ground); labels updated to match.
lumpi_v2 -> lumpi_v3.  Anchors then match both training and inference geometry."""
import glob, json, os, sys
import numpy as np
from multiprocessing import Pool
SRC="/home/qernels/urbantwin_lumpi/data/lumpi_v2"
DST="/home/qernels/urbantwin_lumpi/data/lumpi_v3"
REAL=json.load(open("/home/qernels/urbantwin_lumpi/real_anchors.json"))["anchors"]

def synth_medians():
    acc={}
    for f in sorted(glob.glob(SRC+"/labels/*.txt"))[::20]:   # 500-frame sample is plenty
        for ln in open(f):
            p=ln.split()
            if len(p)<8: continue
            acc.setdefault(p[7],[]).append([float(p[3]),float(p[4]),float(p[5])])
    return {c:np.median(np.array(v),axis=0) for c,v in acc.items()}

def worker(args):
    fid,RATIO=args
    lp=f"{SRC}/labels/{fid}.txt"; pp=f"{SRC}/points/{fid}.npy"
    P=np.load(pp); out_lines=[]; moved=np.zeros(len(P),dtype=bool)
    rows=[]
    for ln in open(lp):
        q=ln.split()
        if len(q)<8: continue
        rows.append(([float(x) for x in q[:7]], q[7]))
    for box,cls in rows:
        r=RATIO.get(cls)
        if r is None:
            out_lines.append(f"{box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f} {box[5]:.6f} {box[6]:.6f} {cls}")
            continue
        cx,cy,cz,dx,dy,dz,hd=box
        bot=cz-dz/2.0
        c,s=np.cos(-hd),np.sin(-hd)
        rel=P[:,:3]-np.array([cx,cy,0.0])
        xr=rel[:,0]*c-rel[:,1]*s; yr=rel[:,0]*s+rel[:,1]*c; zr=P[:,2]-bot
        inside=(np.abs(xr)<=dx/2)&(np.abs(yr)<=dy/2)&(zr>=-0.05)&(zr<=dz+0.05)&(~moved)
        if inside.any():
            xs=xr[inside]*r[0]; ys=yr[inside]*r[1]; zs=zr[inside]*r[2]
            cb,sb=np.cos(hd),np.sin(hd)
            P[inside,0]=cx+xs*cb-ys*sb
            P[inside,1]=cy+xs*sb+ys*cb
            P[inside,2]=bot+zs
            moved|=inside
        ndx,ndy,ndz=dx*r[0],dy*r[1],dz*r[2]
        out_lines.append(f"{cx:.6f} {cy:.6f} {bot+ndz/2.0:.6f} {ndx:.6f} {ndy:.6f} {ndz:.6f} {hd:.6f} {cls}")
    np.save(f"{DST}/points/{fid}.npy",P.astype(np.float32))
    open(f"{DST}/labels/{fid}.txt","w").write("\n".join(out_lines)+"\n")
    return int(moved.sum())

if __name__=="__main__":
    os.makedirs(DST+"/points",exist_ok=True); os.makedirs(DST+"/labels",exist_ok=True)
    os.makedirs(DST+"/ImageSets",exist_ok=True)
    sm=synth_medians()
    RATIO={}
    print(f"{'class':<11}{'synth l/w/h':>22}{'real l/w/h':>22}{'ratio':>24}")
    for c,rv in REAL.items():
        if c not in sm: continue
        s=sm[c]; r=np.array(rv["anchor"])
        RATIO[c]=(float(r[0]/s[0]),float(r[1]/s[1]),float(r[2]/s[2]))
        print(f"{c:<11}{f'{s[0]:.2f}/{s[1]:.2f}/{s[2]:.2f}':>22}"
              f"{f'{r[0]:.2f}/{r[1]:.2f}/{r[2]:.2f}':>22}"
              f"{f'{RATIO[c][0]:.3f}/{RATIO[c][1]:.3f}/{RATIO[c][2]:.3f}':>24}")
    json.dump(RATIO,open("/home/qernels/urbantwin_lumpi/rescale_ratios.json","w"),indent=1)
    ids=[os.path.basename(f)[:-4] for f in sorted(glob.glob(SRC+"/labels/*.txt"))]
    print(f"\nrescaling {len(ids)} frames with 10 workers...",flush=True)
    with Pool(10) as pool:
        tot=0
        for i,n in enumerate(pool.imap_unordered(worker,[(f,RATIO) for f in ids],chunksize=25)):
            tot+=n
            if (i+1)%1000==0: print(f"  {i+1}/{len(ids)}  points moved so far {tot:,}",flush=True)
    for s in ("train","val"):
        os.system(f"cp {SRC}/ImageSets/{s}.txt {DST}/ImageSets/{s}.txt")
    print(f"done. total points moved {tot:,}")
