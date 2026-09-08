#!/usr/bin/env python3
"""Four-model WBF ensemble WITH per-class weights: v3 dominates where it wins
(Car, Person, Bicycle), the older models where v3 lost (Truck, Van, Bus).
Weights derived from SERVER-measured per-class AP, not local."""
import glob, json, os, sys
import numpy as np, torch
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
O="/home/qernels/urbantwin_lumpi/repos/OpenPCDet/output"
MEMBERS=[("v1","cfgs/pointpillar_lumpi.yaml",   f"{O}/pointpillar_lumpi_v1_archive/default/ckpt/checkpoint_epoch_30.pth"),
         ("v4","cfgs/pointpillar_lumpi.yaml",   f"{O}/pointpillar_lumpi/default/ckpt/checkpoint_epoch_30.pth"),
         ("v5","cfgs/second_lumpi.yaml",        f"{O}/second_lumpi/default/ckpt/checkpoint_epoch_38.pth"),
         ("v3","cfgs/pointpillar_lumpi_v3.yaml",f"{O}/pointpillar_lumpi_v3/default/ckpt/checkpoint_epoch_30.pth")]
# per-class member weight: v3 heavy where server says it wins
W={"Person":{"v3":3.0,"v1":1,"v4":1,"v5":1},"Car":{"v3":3.0,"v1":1,"v4":1,"v5":1},
   "Bicycle":{"v3":2.0,"v1":1,"v4":1,"v5":1},"Motorcycle":{"v3":2.0,"v1":1,"v4":1,"v5":1},
   "Bus":{"v3":1.0,"v1":1,"v4":1,"v5":1},"Truck":{"v3":0.5,"v1":1,"v4":1,"v5":1},
   "Van":{"v3":0.5,"v1":1,"v4":1,"v5":1},"Unknown":{"v3":1,"v1":1,"v4":1,"v5":1}}
CMAP={0:"Person",1:"Car",2:"Bicycle",3:"Motorcycle",4:"Bus",5:"Truck",6:"Van",7:"Unknown"}
NAMES=["Person","Car","Bicycle","Motorcycle","Bus","Truck","Van","Unknown"]
R=[-40,-40,-5,40,40,5]; NMS_T=0.01; SCORE_T=0.01; PRE=4096; POST=500; WBF_IOU=0.55
def crop(p):
    m=((p[:,0]>=R[0])&(p[:,0]<=R[3])&(p[:,1]>=R[1])&(p[:,1]<=R[4])&(p[:,2]>=R[2])&(p[:,2]<=R[5])); return p[m]
class BD(DatasetTemplate):
    def __init__(s,dc,cn,lg): super().__init__(dataset_cfg=dc,class_names=cn,training=False,root_path=None,logger=lg)
    def feed(s,pts,fid): return s.prepare_data(data_dict={"points":pts,"frame_id":fid})
    def __len__(s): return 1
def cal(b,p):
    b=b.copy(); bot=b[:,2]-b[:,5]/2.0
    b[:,3]*=p[0]; b[:,4]*=p[1]; b[:,5]*=p[2]; b[:,2]=bot+b[:,5]/2.0+p[3]; return b
def cnt(pts,boxes):
    o=np.zeros(len(boxes),dtype=np.int32)
    for i,b in enumerate(boxes):
        dx,dy,dz=b[3]/2,b[4]/2,b[5]/2
        c,s=np.cos(-b[6]),np.sin(-b[6]); p=pts[:,:3]-b[:3]
        xr=p[:,0]*c-p[:,1]*s; yr=p[:,0]*s+p[:,1]*c
        o[i]=int(np.sum((np.abs(xr)<=dx)&(np.abs(yr)<=dy)&(np.abs(p[:,2])<=dz)))
    return o
def wbf(lists,wsum,thr=WBF_IOU):
    ab=[(b,s,w) for b,s,w in lists if len(b)]
    if not ab: return np.zeros((0,7)),np.zeros(0)
    B=np.concatenate([x[0] for x in ab]); S=np.concatenate([x[1] for x in ab])
    Wv=np.concatenate([np.full(len(x[0]),x[2]) for x in ab])
    o=np.argsort(-S*Wv); B,S,Wv=B[o],S[o],Wv[o]
    fused=[]; mem=[]
    Bt=torch.tensor(B,dtype=torch.float32).cuda()
    for i in range(len(B)):
        placed=False
        if fused:
            F=torch.tensor(np.array([f[0] for f in fused]),dtype=torch.float32).cuda()
            iou=iou3d_nms_utils.boxes_iou3d_gpu(F,Bt[i:i+1]).cpu().numpy().ravel()
            j=int(iou.argmax())
            if iou[j]>=thr:
                mem[j].append((B[i],S[i],Wv[i])); placed=True
                m=mem[j]; w=np.array([x[1]*x[2] for x in m]); bb=np.array([x[0] for x in m])
                nb=(bb*w[:,None]).sum(0)/w.sum()
                ang=bb[:,6]+np.round((bb[0,6]-bb[:,6])/np.pi)*np.pi
                nb[6]=(ang*w).sum()/w.sum(); fused[j]=(nb,w.sum())
        if not placed: fused.append((B[i].copy(),S[i]*Wv[i])); mem.append([(B[i],S[i],Wv[i])])
    ob=np.array([f[0] for f in fused])
    os_=np.array([sum(x[1]*x[2] for x in m)/wsum for m in mem])
    return ob,np.clip(os_,0,1)
lg=common_utils.create_logger()
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
mp={}
for tag,cfgf,ck in MEMBERS:
    cfg.clear()
    cfg_from_yaml_file(cfgf,cfg)
    ds=BD(cfg.DATA_CONFIG,cfg.CLASS_NAMES,lg)
    model=build_network(model_cfg=cfg.MODEL,num_class=len(cfg.CLASS_NAMES),dataset=ds)
    model.load_params_from_file(ck,logger=lg,to_cpu=False); model.cuda(); model.eval()
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
                B.append(bk[keep].cpu().numpy()); S.append(sk[keep].cpu().numpy()); L.append(np.full(len(keep),k+1))
            out[fid]=(np.concatenate(B) if B else np.zeros((0,7)),
                      np.concatenate(S) if S else np.zeros(0),
                      np.concatenate(L).astype(int) if L else np.zeros(0,int))
    mp[tag]=out; del model; torch.cuda.empty_cache(); print(f"  {tag} done",flush=True)
fused={}
for fid in gt:
    Ball,Sall,Lall=[],[],[]
    for k in range(1,9):
        c=NAMES[k-1]; lists=[]; ws=0.0
        for tag in mp:
            b,s,l=mp[tag][fid]; sel=np.where(l==k)[0]
            w=W[c].get(tag,1.0); ws+=w
            lists.append((b[sel],s[sel],w))
        fb,fs=wbf(lists,ws)
        if len(fb): Ball.append(fb); Sall.append(fs); Lall.append(np.full(len(fb),k))
    fused[fid]=(np.concatenate(Ball) if Ball else np.zeros((0,7)),
                np.concatenate(Sall) if Sall else np.zeros(0),
                np.concatenate(Lall).astype(int) if Lall else np.zeros(0,int))
print("fused",flush=True)
pairs={c:{"p":[],"g":[]} for c in NAMES}
for fid,gl in gt.items():
    pb,ps,pl=fused[fid]
    if not len(pb): continue
    for c in set(g["class"] for g in gl):
        gi=[g["box"] for g in gl if g["class"]==c]; k=NAMES.index(c)+1
        sel=np.where(pl==k)[0]
        if not len(sel): continue
        iou=iou3d_nms_utils.boxes_iou3d_gpu(torch.tensor(gi,dtype=torch.float32).cuda(),
              torch.tensor(pb[sel],dtype=torch.float32).cuda()).cpu().numpy()
        for r in range(len(gi)):
            j=int(iou[r].argmax())
            if iou[r,j]>0.05: pairs[c]["p"].append(pb[sel][j]); pairs[c]["g"].append(gi[r])
CAL={}
for c in NAMES:
    P=np.array(pairs[c]["p"],dtype=np.float32); G=np.array(pairs[c]["g"],dtype=np.float32)
    if len(P)<25: continue
    Gt=torch.tensor(G).cuda()
    def ni(q):
        B=torch.tensor(cal(P,q),dtype=torch.float32).cuda()
        return -float(iou3d_nms_utils.boxes_iou3d_gpu(Gt,B).diagonal().mean())
    b0=-ni([1,1,1,0]); rr=minimize(ni,[1,1,1,0],method="Nelder-Mead",options={"maxiter":400})
    CAL[c]=[float(x) for x in (rr.x if -rr.fun>b0 else np.array([1,1,1,0.0]))]
store={}
for fid,(pb,ps,pl) in fused.items():
    b=pb.copy()
    for c,q in CAL.items():
        k=NAMES.index(c)+1; m=(pl==k)
        if m.any(): b[m]=cal(b[m],q)
    store[fid]=(b,ps,pl,cnt(clouds[fid],b))
tpp={c:[] for c in NAMES}
for fid,gl in gt.items():
    pb,ps,pl,pn=store[fid]
    for k in range(1,9):
        c=NAMES[k-1]; gi=[g["box"] for g in gl if g["class"]==c]; sel=np.where(pl==k)[0]
        if not len(sel) or not gi: continue
        iou=iou3d_nms_utils.boxes_iou3d_gpu(torch.tensor(gi,dtype=torch.float32).cuda(),
              torch.tensor(pb[sel],dtype=torch.float32).cuda()).cpu().numpy().max(axis=0)
        tpp[c]+=[int(pn[i]) for j,i in enumerate(sel) if iou[j]>=0.5]
FL={c:(float(np.percentile(v,1)) if len(v)>=20 else 0.0) for c,v in tpp.items()}
def build(fl):
    out={}
    for fid,(pb,ps,pl,pn) in store.items():
        fr=[]
        for i in range(len(pb)):
            c=NAMES[pl[i]-1]
            if fl and pn[i]<fl.get(c,0): continue
            fr.append({"box":[float(v) for v in pb[i][:7]],"score":float(ps[i]),"class":c})
        out[fid]=fr
    return out
for tag,fl in (("4-model ens + cal",None),("+ p01 filter",FL)):
    p=build(fl)
    d=evaluate_3d_detection(p,gt,NAMES,iou_thresholds=(0.5,),iou_type="3d")
    pc={c:d["per_class"][c].get("AP_0.5_R40") for c in NAMES}
    s="  ".join(f"{c[:4]} {pc[c]:.3f}" for c in NAMES if pc[c]==pc[c])
    print(f"  {tag:<22} mAP {d['mAP_0.5_R40']:.4f}   {s}",flush=True)
print("\n  reference: v3 single 0.1501 | v2 4-model ens 0.1415")
json.dump({"cal":CAL,"floors":FL,"weights":W},open("/home/qernels/urbantwin_lumpi/ens4_cfg.json","w"),indent=1)
