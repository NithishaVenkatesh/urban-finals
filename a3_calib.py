#!/usr/bin/env python3
"""A3: per-class box calibration on top of A1 multi-class emission.
Fits (scale_l, scale_w, scale_h, z_off) per class by MAXIMISING mean matched-pair
3D IoU on the 217 real frames. Scaling is about the BOX BOTTOM, not the centre,
so height changes do not drag the box underground."""
import glob, os, sys, json
import numpy as np
import torch
from scipy.optimize import minimize

sys.path.insert(0,"/home/qernels/urbantwin_lumpi/repos/OpenPCDet")
sys.path.insert(0,"/home/qernels/urbantwin_lumpi/data/starter_kit/scoring_program")
os.chdir("/home/qernels/urbantwin_lumpi/repos/OpenPCDet/tools")

import open3d as o3d
from detection_eval import evaluate_3d_detection
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.utils import common_utils

BASE="/home/qernels/urbantwin_lumpi/data/lumpi_real/test_data"
CKPT="/home/qernels/urbantwin_lumpi/repos/OpenPCDet/output/pointpillar_lumpi/default/ckpt/checkpoint_epoch_30.pth"
CMAP={0:"Person",1:"Car",2:"Bicycle",3:"Motorcycle",4:"Bus",5:"Truck",6:"Van",7:"Unknown"}
NAMES=["Person","Car","Bicycle","Motorcycle","Bus","Truck","Van","Unknown"]
R=[-40,-40,-5,40,40,5]
NMS_T=0.01; SCORE_T=0.01; PRE=4096; POST=500

def crop(p):
    m=((p[:,0]>=R[0])&(p[:,0]<=R[3])&(p[:,1]>=R[1])&(p[:,1]<=R[4])&(p[:,2]>=R[2])&(p[:,2]<=R[5]))
    return p[m]
class BD(DatasetTemplate):
    def __init__(s,dc,cn,lg): super().__init__(dataset_cfg=dc,class_names=cn,training=False,root_path=None,logger=lg)
    def feed(s,pts,fid): return s.prepare_data(data_dict={"points":pts,"frame_id":fid})
    def __len__(s): return 1

def load_all():
    gt={}; clouds={}
    for mdir in sorted(glob.glob(f"{BASE}/Measurement*")):
        mn=os.path.basename(mdir); by={}
        with open(os.path.join(mdir,"Label.csv")) as f:
            next(f)
            for line in f:
                p=line.split(",")
                if len(p)<16: continue
                try:
                    t=float(p[0]); c=int(float(p[7])); v=[float(p[i]) for i in (9,10,11,12,13,14,15)]
                except ValueError: continue
                by.setdefault(int(round(t*10)),[]).append((v,c))
        for ply in sorted(glob.glob(os.path.join(mdir,"lidar","*.ply"))):
            fi=int(os.path.basename(ply)[:-4]); fid=f"{mn}_{fi:03d}"
            gt[fid]=[{"box":g[0],"class":CMAP.get(g[1],"Unknown")}
                     for g in by.get(fi,[]) if abs(g[0][0])<=40 and abs(g[0][1])<=40]
            clouds[fid]=crop(np.asarray(o3d.io.read_point_cloud(ply).points,dtype=np.float32))
    return gt,clouds

def infer(model, ds, clouds):
    pp=model.model_cfg.POST_PROCESSING
    pp.NMS_CONFIG.MULTI_CLASSES_NMS=False; pp.NMS_CONFIG.NMS_THRESH=NMS_T; pp.SCORE_THRESH=SCORE_T
    out={}
    with torch.no_grad():
        for fid,pts in clouds.items():
            bd=ds.collate_batch([ds.feed(pts,fid)]); load_data_to_gpu(bd)
            for m in model.module_list: bd=m(bd)
            cls=bd["batch_cls_preds"][0]; box=bd["batch_box_preds"][0]
            if not bd.get("cls_preds_normalized",False): cls=torch.sigmoid(cls)
            B,S,L=[],[],[]
            for k in range(cls.shape[1]):
                sc=cls[:,k]; m_=sc>=SCORE_T
                if int(m_.sum())==0: continue
                bk=box[m_]; sk=sc[m_]
                keep,_=iou3d_nms_utils.nms_gpu(bk[:,:7].contiguous(),sk,NMS_T,pre_maxsize=PRE)
                keep=keep[:POST]
                B.append(bk[keep]); S.append(sk[keep])
                L.append(torch.full((len(keep),),k+1,dtype=torch.long,device=cls.device))
            if B:
                out[fid]=(torch.cat(B).cpu().numpy(),torch.cat(S).cpu().numpy(),torch.cat(L).cpu().numpy())
            else:
                out[fid]=(np.zeros((0,7),np.float32),np.zeros((0,),np.float32),np.zeros((0,),int))
    return out

def apply_calib(boxes, p):
    """p = [sl, sw, sh, z_off]; scale about the box BOTTOM."""
    b=boxes.copy()
    bottom=b[:,2]-b[:,5]/2.0
    b[:,3]*=p[0]; b[:,4]*=p[1]; b[:,5]*=p[2]
    b[:,2]=bottom+b[:,5]/2.0+p[3]
    return b

def main():
    lg=common_utils.create_logger()
    cfg_from_yaml_file("cfgs/pointpillar_lumpi.yaml",cfg)
    ds=BD(cfg.DATA_CONFIG,cfg.CLASS_NAMES,lg)
    model=build_network(model_cfg=cfg.MODEL,num_class=len(cfg.CLASS_NAMES),dataset=ds)
    model.load_params_from_file(CKPT,logger=lg,to_cpu=False); model.cuda(); model.eval()
    gt,clouds=load_all()
    print(f"frames {len(gt)}  gt {sum(len(v) for v in gt.values())}",flush=True)
    raw=infer(model,ds,clouds)
    print("A1 inference done",flush=True)

    # ---- collect best-IoU matched pairs per class ----
    pairs={c:{"p":[],"g":[]} for c in NAMES}
    for fid,gts in gt.items():
        if not gts: continue
        pb,ps,pl=raw[fid]
        if len(pb)==0: continue
        for cname in set(g["class"] for g in gts):
            gi=[g["box"] for g in gts if g["class"]==cname]
            k=NAMES.index(cname)+1
            sel=np.where(pl==k)[0]
            if not len(sel): continue
            G=torch.tensor(gi,dtype=torch.float32).cuda()
            P=torch.tensor(pb[sel],dtype=torch.float32).cuda()
            iou=iou3d_nms_utils.boxes_iou3d_gpu(G,P).cpu().numpy()
            for r in range(len(gi)):
                j=int(iou[r].argmax())
                if iou[r,j]>0.05:
                    pairs[cname]["p"].append(pb[sel][j]); pairs[cname]["g"].append(gi[r])

    calib={}
    print(f"\n{'class':<11}{'pairs':>7}{'IoU before':>12}{'IoU after':>11}"
          f"{'sl':>7}{'sw':>7}{'sh':>7}{'z_off':>8}")
    for c in NAMES:
        P=np.array(pairs[c]["p"],dtype=np.float32); G=np.array(pairs[c]["g"],dtype=np.float32)
        if len(P)<25:
            print(f"{c:<11}{len(P):>7}   (too few pairs - no calibration)"); continue
        Gt=torch.tensor(G).cuda()
        def negiou(p):
            B=torch.tensor(apply_calib(P,p),dtype=torch.float32).cuda()
            d=iou3d_nms_utils.boxes_iou3d_gpu(Gt,B).diagonal()
            return -float(d.mean())
        before=-negiou([1,1,1,0])
        res=minimize(negiou,[1,1,1,0],method="Nelder-Mead",
                     options={"maxiter":400,"xatol":1e-3,"fatol":1e-4})
        p=res.x; after=-res.fun
        if after<=before: p=np.array([1,1,1,0.0]); after=before
        calib[c]=[float(x) for x in p]
        print(f"{c:<11}{len(P):>7}{before:>12.3f}{after:>11.3f}"
              f"{p[0]:>7.3f}{p[1]:>7.3f}{p[2]:>7.3f}{p[3]:>8.3f}")

    # ---- evaluate: A1 alone vs A1+A3 ----
    for tag,use in (("A1 alone",False),("A1 + A3 calibration",True)):
        preds={}
        for fid,(pb,ps,pl) in raw.items():
            if len(pb)==0: preds[fid]=[]; continue
            b=pb.copy()
            if use:
                for c,p in calib.items():
                    k=NAMES.index(c)+1; m=(pl==k)
                    if m.any(): b[m]=apply_calib(b[m],p)
            preds[fid]=[{"box":[float(v) for v in bb[:7]],"score":float(s),"class":NAMES[int(l)-1]}
                        for bb,s,l in zip(b,ps,pl)]
        det=evaluate_3d_detection(preds,gt,NAMES,iou_thresholds=(0.5,),iou_type="3d")
        print(f"\n===== {tag} =====")
        print(f"  mAP@0.5 R40 (7-class) = {det['mAP_0.5_R40']:.4f}")
        for c in NAMES:
            ap=det["per_class"][c].get("AP_0.5_R40")
            if ap is None or (isinstance(ap,float) and np.isnan(ap)): continue
            print(f"    {c:<11}{ap:.4f}")
        sys.stdout.flush()
    json.dump(calib,open("/home/qernels/urbantwin_lumpi/calib_a3.json","w"),indent=1)
    print("\nwrote calib_a3.json")

    # ================= HELD-OUT VALIDATION =================
    # 2-fold by Measurement id: fit on one fold, score the fold never seen.
    print("\n######## HELD-OUT VALIDATION (2-fold by Measurement) ########",flush=True)
    fids=sorted(gt.keys())
    foldA=[f for f in fids if int(f.split("Measurement")[1][0])%2==0]
    foldB=[f for f in fids if int(f.split("Measurement")[1][0])%2==1]
    def fit_on(fold):
        pr={c:{"p":[],"g":[]} for c in NAMES}
        for fid in fold:
            gts=gt[fid]
            if not gts: continue
            pb,ps,pl=raw[fid]
            if len(pb)==0: continue
            for cname in set(g["class"] for g in gts):
                gi=[g["box"] for g in gts if g["class"]==cname]
                k=NAMES.index(cname)+1; sel=np.where(pl==k)[0]
                if not len(sel): continue
                G=torch.tensor(gi,dtype=torch.float32).cuda()
                P=torch.tensor(pb[sel],dtype=torch.float32).cuda()
                iou=iou3d_nms_utils.boxes_iou3d_gpu(G,P).cpu().numpy()
                for r in range(len(gi)):
                    j=int(iou[r].argmax())
                    if iou[r,j]>0.05:
                        pr[cname]["p"].append(pb[sel][j]); pr[cname]["g"].append(gi[r])
        cal={}
        for c in NAMES:
            P=np.array(pr[c]["p"],dtype=np.float32); G=np.array(pr[c]["g"],dtype=np.float32)
            if len(P)<25: continue
            Gt=torch.tensor(G).cuda()
            def ni(q):
                B=torch.tensor(apply_calib(P,q),dtype=torch.float32).cuda()
                return -float(iou3d_nms_utils.boxes_iou3d_gpu(Gt,B).diagonal().mean())
            r0=-ni([1,1,1,0])
            rr=minimize(ni,[1,1,1,0],method="Nelder-Mead",
                        options={"maxiter":400,"xatol":1e-3,"fatol":1e-4})
            cal[c]=[float(x) for x in (rr.x if -rr.fun>r0 else np.array([1,1,1,0.0]))]
        return cal
    def score_on(fold,cal):
        pd_={}
        for fid in fold:
            pb,ps,pl=raw[fid]
            if len(pb)==0: pd_[fid]=[]; continue
            b=pb.copy()
            for c,q in cal.items():
                k=NAMES.index(c)+1; m=(pl==k)
                if m.any(): b[m]=apply_calib(b[m],q)
            pd_[fid]=[{"box":[float(v) for v in bb[:7]],"score":float(s),"class":NAMES[int(l)-1]}
                      for bb,s,l in zip(b,ps,pl)]
        g_={f:gt[f] for f in fold}
        return evaluate_3d_detection(pd_,g_,NAMES,iou_thresholds=(0.5,),iou_type="3d")["mAP_0.5_R40"]
    for nm,ftr,tst in (("fit A -> test B",foldA,foldB),("fit B -> test A",foldB,foldA)):
        cal=fit_on(ftr)
        fit_score=score_on(ftr,cal); ho=score_on(tst,cal)
        base=score_on(tst,{})
        print(f"  {nm}: fit-fold mAP {fit_score:.4f} | held-out mAP {ho:.4f} "
              f"| held-out uncalibrated {base:.4f} | gain {ho-base:+.4f} "
              f"| optimism {fit_score-ho:+.4f}",flush=True)

if __name__=="__main__": main()
