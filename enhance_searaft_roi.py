#!/usr/bin/env python3
"""Optional local SEA-RAFT refinement for small/rotating parts.

Modes:
  --select-roi: interactively select one or more boxes on RGB first frame.
  --select-only: select/save boxes without loading SEA-RAFT (works on local PC).
  --roi-file: use saved JSON boxes, suitable for headless SSH/AutoDL.
  --roi X1 Y1 X2 Y2: pass box coordinates directly (repeatable).

Normal/simple objects should skip this optional stage.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--repo', type=Path, help='Official SEA-RAFT repository (required unless --select-only)')
    p.add_argument('--cfg', type=Path, help='SEA-RAFT config JSON (required unless --select-only)')
    p.add_argument('--checkpoint', type=Path, help='SEA-RAFT checkpoint (required unless --select-only)')
    p.add_argument('--video', required=True, type=Path, help='RGB video used for the original global optical flow')
    p.add_argument('--base-flow', type=Path, help='Global flow .npy (required unless --select-only)')
    p.add_argument('--output', type=Path, help='New fused flow .npy (required unless --select-only)')
    group = p.add_mutually_exclusive_group()
    group.add_argument('--roi', nargs=4, type=int, action='append', metavar=('X1','Y1','X2','Y2'), help='Manually specified boxes; repeatable')
    group.add_argument('--roi-file', type=Path, help='ROI JSON saved previously')
    group.add_argument('--select-roi', action='store_true', help='Pop up the first frame and select boxes with mouse')
    p.add_argument('--select-only', action='store_true', help='Only select/save ROIs; no CUDA/SEA-RAFT needed')
    p.add_argument('--save-roi', type=Path, help='Save ROI coordinates to JSON')
    p.add_argument('--context', type=float, default=0.25, help='ROI context padding ratio')
    p.add_argument('--infer-size', type=int, default=640, help='Square local inference size')
    p.add_argument('--iters', type=int, default=12)
    p.add_argument('--feather', type=int, default=12, help='ROI boundary feather in image pixels')
    p.add_argument('--frames', type=int, default=None, help='Expected number of RGB frames')
    p.add_argument('--preview', type=Path, help='Save marked first-frame preview PNG')
    return p.parse_args()


def load_video_first(video):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open RGB video: {video}')
    ok, first = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f'Cannot read RGB video first frame: {video}')
    return cap, first


def select_rois(frame):
    try:
        boxes = cv2.selectROIs('Select gear/part ROIs | ENTER finish | ESC cancel', frame, showCrosshair=True, fromCenter=False)
        cv2.destroyAllWindows()
    except cv2.error as e:
        raise RuntimeError('OpenCV graphical ROI selection failed. On headless AutoDL/SSH, use --select-only on your Windows PC, then --roi-file on AutoDL.') from e
    rois = []
    for x, y, w, h in np.asarray(boxes, dtype=np.int32).reshape(-1, 4):
        if w > 0 and h > 0:
            rois.append((int(x), int(y), int(x+w), int(y+h)))
    if not rois:
        raise ValueError('No ROI selected. Draw rectangles and press Enter to finish.')
    return rois


def load_rois(path):
    data = json.loads(path.read_text(encoding='utf-8'))
    raw = data['rois'] if isinstance(data, dict) else data
    rois = []
    for item in raw:
        if isinstance(item, dict):
            item = [item[k] for k in ('x1', 'y1', 'x2', 'y2')]
        if len(item) != 4:
            raise ValueError(f'Invalid ROI JSON item: {item}')
        rois.append(tuple(int(x) for x in item))
    return rois


def save_rois(path, rois, width, height):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {'width': width, 'height': height, 'rois': [list(box) for box in rois]}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('ROI JSON saved:', path, flush=True)


def make_boxes(rois, width, height, context):
    result = []
    for index, (x1, y1, x2, y2) in enumerate(rois, 1):
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f'ROI #{index} invalid: {(x1,y1,x2,y2)}, image {(width,height)}')
        pad = int(round(max(x2-x1, y2-y1) * context))
        crop = (max(0,x1-pad), max(0,y1-pad), min(width,x2+pad), min(height,y2+pad))
        if crop[2] - crop[0] < 8 or crop[3] - crop[1] < 8:
            raise ValueError(f'ROI #{index} too small')
        result.append(((x1,y1,x2,y2),crop))
    return result


def roi_weight(target, crop, feather):
    x1,y1,x2,y2 = target
    cx1,cy1,cx2,cy2 = crop
    mask = np.zeros((cy2-cy1,cx2-cx1), dtype=np.float32)
    mask[y1-cy1:y2-cy1,x1-cx1:x2-cx1] = 1.0
    if feather:
        k = 2*feather+1
        mask = cv2.GaussianBlur(mask,(k,k),sigmaX=max(1,feather/2))
    return np.clip(mask,0,1)[...,None]


def main():
    args = parse_args()
    if args.context < 0 or args.feather < 0 or args.infer_size < 64 or args.iters < 1:
        raise ValueError('Invalid context / feather / infer-size / iters')
    if not args.video.is_file():
        raise FileNotFoundError(args.video)
    cap, first = load_video_first(args.video)
    try:
        if args.select_roi:
            rois = select_rois(first)
        elif args.roi_file:
            rois = load_rois(args.roi_file)
            if isinstance(json.loads(args.roi_file.read_text(encoding='utf-8')),dict):
                metadata = json.loads(args.roi_file.read_text(encoding='utf-8'))
                if ('width' in metadata and 'height' in metadata and
                    (metadata['width'],metadata['height']) != (first.shape[1],first.shape[0])):
                    raise ValueError('ROI file image size does not match video')
        elif args.roi:
            rois = [tuple(b) for b in args.roi]
        else:
            raise ValueError('Provide --select-roi, --roi-file or --roi')
        if not rois:
            raise ValueError('ROI list is empty')
        native_h,native_w = first.shape[:2]
        # ROI coordinates are defined in first-frame/native-image pixels.
        if args.select_only:
            boxes = make_boxes(rois,native_w,native_h,args.context)
            if args.save_roi:
                save_rois(args.save_roi,rois,native_w,native_h)
            elif args.select_roi:
                save_rois(Path('selected_rois.json'),rois,native_w,native_h)
            if args.preview:
                draw_preview(first,boxes,args.preview)
            print('Selected ROIs:',rois)
            return
        required = {'--repo':args.repo, '--cfg':args.cfg, '--checkpoint':args.checkpoint,
                    '--base-flow':args.base_flow, '--output':args.output}
        missing = [name for name,path in required.items() if path is None]
        if missing:
            raise ValueError('Missing arguments: '+', '.join(missing))
        if args.output.resolve() == args.base_flow.resolve():
            raise ValueError('Output must differ from --base-flow')
        if args.output.exists():
            raise FileExistsError(f'Output exists (refusing overwrite): {args.output}')
        for p in (args.repo/'core/raft.py',args.cfg,args.checkpoint,args.base_flow):
            if not p.is_file():
                raise FileNotFoundError(p)
        original = np.load(args.base_flow,mmap_mode='r')
        if original.ndim != 4 or original.shape[-1] != 2 or original.dtype != np.float32:
            raise ValueError(f'Expected float32 flow (T-1,H,W,2), got {original.shape} {original.dtype}')
        n,h,w,_ = original.shape
        if (native_w,native_h) != (w,h):
            raise ValueError(f'Video first frame {(native_w,native_h)} differs from flow {(w,h)}; select ROIs on matching video')
        if args.frames is not None and args.frames != n+1:
            raise ValueError(f'--frames must be {n+1}')
        boxes = make_boxes(rois,w,h,args.context)
        if args.save_roi:
            save_rois(args.save_roi,rois,w,h)
        if args.preview:
            draw_preview(first,boxes,args.preview)
        import torch
        from argparse import Namespace
        sys.path.insert(0,str((args.repo/'core').resolve()))
        from raft import RAFT
        from utils.utils import InputPadder,load_ckpt
        cfg=Namespace(**json.loads(args.cfg.read_text()))
        model=RAFT(cfg).cuda().eval()
        load_ckpt(model,str(args.checkpoint))
        args.output.parent.mkdir(parents=True,exist_ok=True)
        out=np.lib.format.open_memmap(args.output,mode='w+',dtype=np.float32,shape=original.shape)
        try:
            with torch.inference_mode():
                for t in range(n):
                    ok,nxt=cap.read()
                    if not ok:
                        raise RuntimeError(f'Video ended at pair {t}; expected {n}')
                    nxt=cv2.resize(nxt,(w,h),interpolation=cv2.INTER_LINEAR)
                    fused=np.asarray(original[t],dtype=np.float32).copy()
                    if not np.isfinite(fused).all():
                        raise ValueError(f'Non-finite base flow at frame {t}')
                    for index,(target,crop) in enumerate(boxes,1):
                        cx1,cy1,cx2,cy2=crop
                        cw,ch=cx2-cx1,cy2-cy1
                        patch0=cv2.resize(first[cy1:cy2,cx1:cx2],(args.infer_size,args.infer_size))
                        patch1=cv2.resize(nxt[cy1:cy2,cx1:cx2],(args.infer_size,args.infer_size))
                        def tensor(bgr):
                            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                            return torch.from_numpy(rgb.copy()).permute(2,0,1)[None].float().cuda()
                        x,y=tensor(patch0),tensor(patch1)
                        padder=InputPadder(x.shape)
                        x,y=padder.pad(x,y)
                        pred=model(x,y,iters=args.iters,test_mode=True)
                        local=padder.unpad(pred['flow'][-1])[0].permute(1,2,0).float().cpu().numpy()
                        local=cv2.resize(local,(cw,ch),interpolation=cv2.INTER_LINEAR)
                        local[...,0]*=cw/float(args.infer_size)
                        local[...,1]*=ch/float(args.infer_size)
                        if not np.isfinite(local).all():
                            raise ValueError(f'Non-finite local flow at frame {t}, ROI {index}')
                        weight=roi_weight(target,crop,args.feather)
                        base=fused[cy1:cy2,cx1:cx2]
                        fused[cy1:cy2,cx1:cx2]=base*(1-weight)+local*weight
                        del x,y,pred
                    out[t]=fused
                    if t%10==0 or t==n-1:
                        print(f'ROI flow {t+1}/{n} pairs; {len(boxes)} ROI(s)',flush=True)
                    first=nxt
            out.flush()
        except BaseException:
            del out
            args.output.unlink(missing_ok=True)
            raise
        print('Saved:',args.output,'shape:',original.shape)
        print('Use with FEAST Stage A/B --flow-npy',args.output)
    finally:
        cap.release()


def draw_preview(first,boxes,path):
    pic=first.copy()
    for index,(target,crop) in enumerate(boxes,1):
        cv2.rectangle(pic,target[:2],target[2:],(0,255,255),2)
        cv2.rectangle(pic,crop[:2],crop[2:],(0,150,255),1)
        cv2.putText(pic,str(index),target[:2],cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
    path.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(path),pic):
        raise RuntimeError('Failed to write preview: '+str(path))
    print('ROI preview:',path)


if __name__ == '__main__':
    main()
