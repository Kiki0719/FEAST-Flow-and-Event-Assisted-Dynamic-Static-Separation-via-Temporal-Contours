#!/usr/bin/env python3
"""Run OFFICIAL SEA-RAFT repository pretrained model on RGB video -> (T-1,H,W,2) npy.
Requires clone of https://github.com/princeton-vl/SEA-RAFT and checkpoint compatible with --cfg.
"""
import argparse,json,sys
from pathlib import Path
import cv2,numpy as np,torch,torch.nn.functional as F

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True);p.add_argument('--cfg',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--video',required=True)
    p.add_argument('--frames',type=int,default=100);p.add_argument('--width',type=int,default=640)
    p.add_argument('--height',type=int,default=640);p.add_argument('--iters',type=int,default=12)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    for path in (args.repo/'core/raft.py',args.cfg,args.checkpoint):
        if not path.exists():raise FileNotFoundError(path)
    sys.path.insert(0,str((args.repo/'core').resolve()))
    from raft import RAFT
    from utils.utils import InputPadder,load_ckpt
    from argparse import Namespace
    cfg=Namespace(**json.loads(args.cfg.read_text()))
    model=RAFT(cfg).cuda().eval()
    load_ckpt(model,str(args.checkpoint))
    cap=cv2.VideoCapture(args.video)
    if not cap.isOpened():raise RuntimeError('Cannot open '+args.video)
    seq=[]
    while len(seq)<args.frames:
        ok,f=cap.read()
        if not ok:break
        f=cv2.resize(f,(args.width,args.height))
        seq.append(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))
    cap.release()
    if len(seq)<2:raise RuntimeError('Need >=2 RGB video frames')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    output=np.lib.format.open_memmap(args.output,mode='w+',dtype=np.float32,shape=(len(seq)-1,args.height,args.width,2))
    with torch.inference_mode():
        for i in range(len(seq)-1):
            a=torch.from_numpy(seq[i].copy()).permute(2,0,1)[None].float().cuda()
            b=torch.from_numpy(seq[i+1].copy()).permute(2,0,1)[None].float().cuda()
            padder=InputPadder(a.shape);a,b=padder.pad(a,b)
            result=model(a,b,iters=args.iters,test_mode=True)
            flow=padder.unpad(result['flow'][-1])[0].permute(1,2,0).float().cpu().numpy()
            if flow.shape!=(args.height,args.width,2):raise ValueError('SEA-RAFT output resolution mismatch')
            output[i]=flow
            if i%10==0:print('SEA-RAFT flow:',i+1,'/',len(seq)-1,flush=True)
    del output
    print('Saved:',args.output)
if __name__=='__main__':main()
