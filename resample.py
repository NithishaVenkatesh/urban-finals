#!/usr/bin/env python3
"""Step 5: importance resampling to match empirical spatial occupancy.
Guards: fit on 25 ref frames / score on the 25 unseen (both rotations),
weight clipping, voxel-size sweep read on the FIT-vs-HELDOUT GAP,
resample directly to the target point count (no thin-after),
all four metrics reported separately at every setting."""
import glob, itertools, json, os, sys
import numpy as np
sys.path.insert(0,"/home/qernels/urbantwin_lumpi/data/starter_kit/scoring_program")
from metrics import compute_realism, _concat_and_mask
PC=(-40,-40,-5,40,40,5)
NORM={"CD":(2.0,30.0),"MMD":(0.0,0.5),"EMD":(0.0,5.0),"FPD":(0.0,50.0)}
REF="/home/qernels/urbantwin_lumpi/data/starter_kit/reference_data_local/realism_reference"
SRC={"v6 (post-affine)":"/home/qernels/urbantwin_lumpi/submissions/v6_fpd/synthetic"}
TARGET_N=40000          # resample directly to the submitted count
WCLIP=10.0              # max weight ratio
def load(d): return [np.fromfile(f,dtype=np.float32).reshape(-1,3)
                     for f in sorted(glob.glob(os.path.join(d,"*.bin")))]
def nrm(r): return {k:1.0-(np.clip(r[k],*NORM[k])-NORM[k][0])/(NORM[k][1]-NORM[k][0]) for k in NORM}
def mean4(n): return float(np.mean(list(n.values())))
def keys(P,vs):
    q=np.floor((P-np.array([PC[0],PC[1],PC[2]]))/vs).astype(np.int64)
    nx=int(np.ceil((PC[3]-PC[0])/vs)); ny=int(np.ceil((PC[4]-PC[1])/vs))
    return q[:,0]+nx*(q[:,1]+ny*q[:,2])
def build_weights(ref_pts,our_pts,vs):
    rk=keys(ref_pts,vs); ok=keys(our_pts,vs)
    ru,rc=np.unique(rk,return_counts=True); ou,oc=np.unique(ok,return_counts=True)
    rmap=dict(zip(ru.tolist(),(rc/rc.sum()).tolist()))
    omap=dict(zip(ou.tolist(),(oc/oc.sum()).tolist()))
    w={}
    for k,ov in omap.items():
        rv=rmap.get(k,0.0)
        w[k]=min(rv/ov,WCLIP) if ov>0 else 0.0
    return w,vs
def resample(clouds,w,vs,n=TARGET_N,seed=0):
    rng=np.random.default_rng(seed); out=[]
    for c in clouds:
        m=((c[:,0]>=PC[0])&(c[:,0]<=PC[3])&(c[:,1]>=PC[1])&(c[:,1]<=PC[4])
           &(c[:,2]>=PC[2])&(c[:,2]<=PC[5]))
        P=c[m]
        if len(P)==0: out.append(c[:0]); continue
        kk=keys(P,vs)
        wt=np.array([w.get(int(x),0.0) for x in kk])
        if wt.sum()<=0: wt=np.ones(len(P))
        wt=np.maximum(wt,1e-9)
        take=min(n,len(P))
        # Gumbel top-k == weighted sampling WITHOUT replacement (no duplicates)
        g=np.log(wt)+rng.gumbel(size=len(P))
        idx=np.argpartition(-g,take-1)[:take]
        out.append(P[idx].astype(np.float32))
    return out

ref=load(REF); print(f"reference frames {len(ref)}",flush=True)
idx=np.arange(len(ref)); half=len(ref)//2
SPLITS={"A":(idx[:half],idx[half:]),"B":(idx[half:],idx[:half])}

print(f"\n{'source':<17}{'vox':>5}{'split':>6}{'set':>9}{'CD':>9}{'MMD':>10}{'EMD':>8}{'FPD':>8}{'mean':>8}")
base={}
for sname,d in SRC.items():
    cl=load(d)
    for tag,(fit,ho) in SPLITS.items():
        r=compute_realism(cl,[ref[i] for i in ho],PC); n=nrm(r)
        base[(sname,tag)]=mean4(n)
        print(f"{sname:<17}{'--':>5}{tag:>6}{'baseline':>9}{r['CD']:>9.4f}{r['MMD']:>10.6f}"
              f"{r['EMD']:>8.4f}{r['FPD']:>8.4f}{mean4(n):>8.4f}",flush=True)

results={}
for sname,d in SRC.items():
    cl=load(d)
    for vs in (16.0,12.0,8.0,6.0,4.0):
        for tag,(fit,ho) in SPLITS.items():
            rf=_concat_and_mask([ref[i] for i in fit],PC)
            ours=_concat_and_mask(cl,PC)
            w,_=build_weights(rf,ours,vs)
            rs=resample(cl,w,vs)
            r_fit=compute_realism(rs,[ref[i] for i in fit],PC); n_fit=nrm(r_fit)
            r_ho =compute_realism(rs,[ref[i] for i in ho ],PC); n_ho =nrm(r_ho)
            gap=mean4(n_fit)-mean4(n_ho)
            results[(sname,vs,tag)]=(mean4(n_fit),mean4(n_ho),gap,r_ho,n_ho)
            print(f"{sname:<17}{vs:>5.1f}{tag:>6}{'FIT':>9}{r_fit['CD']:>9.4f}{r_fit['MMD']:>10.6f}"
                  f"{r_fit['EMD']:>8.4f}{r_fit['FPD']:>8.4f}{mean4(n_fit):>8.4f}",flush=True)
            print(f"{'':<17}{'':>5}{'':>6}{'HELD-OUT':>9}{r_ho['CD']:>9.4f}{r_ho['MMD']:>10.6f}"
                  f"{r_ho['EMD']:>8.4f}{r_ho['FPD']:>8.4f}{mean4(n_ho):>8.4f}"
                  f"   gap {gap:+.4f}   vs base {mean4(n_ho)-base[(sname,tag)]:+.4f}",flush=True)

print("\n######## SUMMARY: held-out gain vs baseline (avg over both splits) ########")
print(f"{'source':<17}{'vox':>5}{'fit':>9}{'heldout':>9}{'gap':>9}{'gain':>9}")
best=None
for sname in SRC:
    for vs in (16.0,12.0,8.0,6.0,4.0):
        f=np.mean([results[(sname,vs,t)][0] for t in "AB"])
        h=np.mean([results[(sname,vs,t)][1] for t in "AB"])
        g=np.mean([results[(sname,vs,t)][2] for t in "AB"])
        b=np.mean([base[(sname,t)] for t in "AB"])
        gain=h-b
        print(f"{sname:<17}{vs:>5.1f}{f:>9.4f}{h:>9.4f}{g:>9.4f}{gain:>+9.4f}")
        if best is None or gain>best[0]: best=(gain,sname,vs,h)
print(f"\n  BEST held-out gain: {best[1]} @ vox {best[2]} -> realism {best[3]:.4f} ({best[0]:+.4f})")
print(f"  combined impact: {0.4*best[0]:+.4f}   (ship threshold +0.010)")
json.dump({"source":best[1],"vox":best[2],"heldout":best[3],"gain":best[0]},
          open("/home/qernels/urbantwin_lumpi/resample_best.json","w"),indent=1)
