#!/usr/bin/env python3
"""Submission G = per-class-weighted 4-model ensemble (v1+v4+v5+v3) + calibration
+ p01 filter, with Truck and Van cherry-picked from D (server-measured better).
Realism frames unchanged."""
import glob, hashlib, json, os, shutil, subprocess, sys
import numpy as np, torch
sys.path.insert(0,"/home/qernels/urbantwin_lumpi/repos/OpenPCDet")
os.chdir("/home/qernels/urbantwin_lumpi/repos/OpenPCDet/tools")
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.utils import common_utils
SK="/home/qernels/urbantwin_lumpi/data/starter_kit"
O="/home/qernels/urbantwin_lumpi/repos/OpenPCDet/output"
MEMBERS=[("v1","cfgs/pointpillar_lumpi.yaml",   f"{O}/pointpillar_lumpi_v1_archive/default/ckpt/checkpoint_epoch_30.pth"),
         ("v4","cfgs/pointpillar_lumpi.yaml",   f"{O}/pointpillar_lumpi/default/ckpt/checkpoint_epoch_30.pth"),
         ("v5","cfgs/second_lumpi.yaml",        f"{O}/second_lumpi/default/ckpt/checkpoint_epoch_38.pth"),
         ("v3","cfgs/pointpillar_lumpi_v3.yaml",f"{O}/pointpillar_lumpi_v3/default/ckpt/checkpoint_epoch_30.pth")]
NAMES=["Person","Car","Bicycle","Motorcycle","Bus","Truck","Van","Unknown"]
NMS_T=0.01; SCORE_T=0.01; PRE=4096; POST=500; WBF_IOU=0.55
C=json.load(open("/home/qernels/urbantwin_lumpi/ens4_cfg.json")); CAL=C["cal"]; FL=C["floors"]; W=C["weights"]
SRC="/home/qernels/urbantwin_lumpi/submissions/subE"
DSRC="/home/qernels/urbantwin_lumpi/submissions/subD"
OUT="/home/qernels/urbantwin_lumpi/submissions/subG"
SWAP={"Truck","Van"}
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
files=sorted(glob.glob(f"{SK}/detection_test_frames/*.bin"))
clouds={os.path.basename(f)[:-4]:np.fromfile(f,dtype=np.float32).reshape(-1,3) for f in files}
mp={}
for tag,cfgf,ck in MEMBERS:
    cfg.clear(); cfg_from_yaml_file(cfgf,cfg)
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
pd_=json.load(open(f"{DSRC}/predictions.json"))
res={}; per={c:0 for c in NAMES}; tot=0
for fid,pts in clouds.items():
    frame=[]
    for k in range(1,9):
        c=NAMES[k-1]
        if c in SWAP: continue                      # take these from D below
        lists=[]; ws=0.0
        for tag in mp:
            b,s,l=mp[tag][fid]; sel=np.where(l==k)[0]
            w=W[c].get(tag,1.0); ws+=w
            lists.append((b[sel],s[sel],w))
        fb,fs=wbf(lists,ws)
        if not len(fb): continue
        if c in CAL: fb=cal(fb,CAL[c])
        ok=cnt(pts,fb)>=FL.get(c,0); fb=fb[ok]; fs=fs[ok]
        for b,s in zip(fb,fs):
            frame.append({"box":[float(v) for v in b[:7]],"score":float(min(max(s,0.0),1.0)),"class":c})
        per[c]+=len(fb)
    for x in pd_[fid]:
        if x["class"] in SWAP: frame.append(x); per[x["class"]]+=1
    res[fid]=frame; tot+=len(frame)
print(f"predictions {tot} over {len(res)} frames")
for c in NAMES: print(f"   {c:<11}{per[c]:>7}{'   <- from D' if c in SWAP else ''}")
os.makedirs(OUT,exist_ok=True)
shutil.rmtree(os.path.join(OUT,"synthetic"),ignore_errors=True)
shutil.copytree(os.path.join(SRC,"synthetic"),os.path.join(OUT,"synthetic"))
shutil.copy(os.path.join(SRC,"declaration.txt"),OUT)
json.dump(res,open(f"{OUT}/predictions.json","w"))
zp=f"{OUT}/submission.zip"
if os.path.exists(zp): os.remove(zp)
subprocess.run(["zip","-q","-r","submission.zip","synthetic","predictions.json","declaration.txt"],cwd=OUT,check=True)
print(f"zip {os.path.getsize(zp)/1e6:.1f} MB  md5 {hashlib.md5(open(zp,'rb').read()).hexdigest()}")
