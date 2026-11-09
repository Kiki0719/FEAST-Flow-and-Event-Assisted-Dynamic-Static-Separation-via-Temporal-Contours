#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Five-object FEAST evaluation against Blender reference motion.

No re-inference; read existing SEA-RAFT, Stage B, Blender surface animation,
and Gaussian-Part binding. Blender motion is reference ONLY for EVALUATION.
It is never used to train or alter FEAST predictions.

Four metrics:
 - Flow GT EPE (px), projected Blender surface motion vs SEA-RAFT
 - Dynamic Part Precision, Recall and F1 (Blender part motion vs Stage B)

GT optical-flow EPE is computed only when a defensible temporal mapping is
available: source_frames must increment by 1 AND the Blender sample count must
match the video frames inferred from SEA-RAFT. If your Blender exports were
subsampled (e.g. source step 10), create a time-mapping JSON and supply
--frame-map-dir. Never silently assume such sampled frames align with video.

An optional mapping JSON per model: FRAME_MAP_DIR/MODEL.json
{"video_frame_indices": [0, 10, 20, ...]} -- one index for each Blender pose.
The sampled Blender poses must correspond exactly to those video frames.
Between successive reference frames, flow is composed through every intervening
video frame; there is no direct long-baseline flow comparison to one flow step.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np

MODELS = ("Fan", "Lego", "lamp", "micphone", "metronome")
PREF = {
    "Fan": ("confirmation_eventvis_tracks", "confirmation_balanced"),
    "Lego": ("confirmation_gear_rebalanced", "confirmation_gear_dense", "confirmation_gear_ultradense", "confirmation_balanced"),
    "lamp": ("confirmation_balanced",),
    "micphone": ("confirmation_balanced",),
    "metronome": ("confirmation_eventvis_tracks", "confirmation_balanced"),
}


def find_stage_b(run, model):
    for name in PREF.get(model, ()):
        p = run / name
        if (p / "motion_confirmation.json").is_file():
            return p
    others = [p for p in run.glob("confirmation*") if (p / "motion_confirmation.json").is_file()]
    return max(others, key=lambda p: p.stat().st_mtime) if others else None


def json_file(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_inputs(outputs, model):
    run = outputs / model / "rgb_part_motion_se3_flow"
    confirm = find_stage_b(run, model)
    if confirm is None:
        raise FileNotFoundError(f"No motion_confirmation.json in {run}")
    confirmation = json_file(confirm / "motion_confirmation.json")
    files = sorted((outputs / f"{model}-init").glob("blender-surface-parts.npz"))
    if not files:
        raise FileNotFoundError(f"Missing Blender surface parts for {model}")
    # Use the exact binding cited by the Stage B run if registered, otherwise prefer
    # a matching binding and do not pick between multiple plausible bindings.
    bindings = sorted((outputs / f"{model}-init").glob("initial-binding*.npz"))
    if not bindings:
        raise FileNotFoundError(f"Missing Gaussian-Part binding for {model}")
    if model == "metronome":
        preferred = [p for p in bindings if "scale244-recovered" in p.name or "depth075" in p.name]
        if len(preferred) == 1:
            bindings = preferred
    if len(bindings) != 1:
        raise RuntimeError(f"Ambiguous binding files for {model}: {[p.name for p in bindings]}. Pass --binding-MODEL")
    return run, confirm, confirmation, files[0], bindings[0]


def read_part_movement(parts_path, binding_path, tol):
    with np.load(parts_path, allow_pickle=False) as d:
        mat = np.asarray(d['object_matrices'], dtype=np.float64)
        meta = json.loads(str(d['metadata_json'].item()))
        source_frames = np.asarray(d['source_frames']) if 'source_frames' in d else None
    if mat.ndim != 4 or mat.shape[2:] != (4, 4):
        raise ValueError('Bad Blender object_matrices shape')
    K, P = mat.shape[:2]
    base_inv = np.linalg.inv(mat[0])
    delta = mat @ base_inv
    residual = np.max(np.abs(delta - np.eye(4)), axis=(0, 2, 3))
    moving = residual > tol
    for k in meta.get('deforming_objects', []):
        if isinstance(k, (int, np.integer)) and 0 <= int(k) < P:
            moving[int(k)] = True
    with np.load(binding_path, allow_pickle=False) as d:
        labels = np.asarray(d['labels'], dtype=np.int32)
        registration = np.asarray(d['registration'], dtype=np.float64)
    label_ids = np.unique(labels[(labels >= 0) & (labels < P)])
    return mat, delta, moving, label_ids, registration, source_frames


def classification_metrics(moving, label_ids, confirmation):
    # Restrict evaluation to parts with Gaussians assigned in this binding.
    # Never use Stage B labels as ground truth.
    ref = set(map(int, label_ids))
    gt = set(np.flatnonzero(moving)) & ref
    predicted = {int(i) for i in (confirmation.get('moving_parts') or [])} & ref
    tp = len(predicted & gt)
    fp = len(predicted - gt)
    fn = len(gt - predicted)
    prec = tp / (tp + fp) if tp + fp else (1.0 if not gt else 0.0)
    recall = tp / (tp + fn) if tp + fn else (1.0 if not predicted else 0.0)
    f1 = 2 * prec * recall / (prec + recall) if prec + recall else 0.0
    return dict(gt_dynamic_parts=len(gt), evaluated_parts=len(ref), confirmed_parts=len(predicted),
                TP=tp, FP=fp, FN=fn, precision=prec, recall=recall, f1=f1)


def project(x, camera, h, w):
    # inspection.json camera.pose uses OpenGL camera-to-world convention.
    pose = np.asarray(camera['pose'], dtype=np.float64)
    w2c = np.linalg.inv(pose @ np.diag([1., -1., -1., 1.]))
    xcam = x @ w2c[:3, :3].T + w2c[:3, 3]
    z = xcam[:, 2]
    fy = h / (2 * np.tan(np.deg2rad(float(camera['fovy_degrees'])) / 2))
    fx = fy * w / h
    with np.errstate(divide='ignore', invalid='ignore'):
        uv = np.stack([fx * xcam[:, 0] / z + w / 2, fy * xcam[:, 1] / z + h / 2], axis=1)
    valid = (z > 1e-6) & np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] <= w-1) & (uv[:, 1] >= 0) & (uv[:, 1] <= h-1)
    return uv, z, valid


def sample_flow(frame, coords):
    h, w = frame.shape[:2]
    xy = np.asarray(coords, np.float64)
    good = np.isfinite(xy).all(axis=1) & (xy[:,0]>=0) & (xy[:,0]<=w-1) & (xy[:,1]>=0) & (xy[:,1]<=h-1)
    out = np.full((len(xy),2), np.nan, np.float64)
    if not good.any():
        return out
    ids = np.flatnonzero(good)
    x,y = xy[ids,0],xy[ids,1]
    x0=np.floor(x).astype(int);y0=np.floor(y).astype(int)
    x1=np.minimum(x0+1,w-1);y1=np.minimum(y0+1,h-1)
    u=(x-x0)[:,None]; v=(y-y0)[:,None]
    f=np.asarray(frame)
    out[ids]=(1-v)*((1-u)*f[y0,x0]+u*f[y0,x1])+v*((1-u)*f[y1,x0]+u*f[y1,x1])
    return out


def resolve_frame_map(model, source_frames, K, nvideo, frame_map_dir):
    manual = frame_map_dir / f'{model}.json' if frame_map_dir else None
    if manual and manual.is_file():
        vals = json_file(manual)['video_frame_indices']
        ix = np.asarray(vals, dtype=int)
        if ix.shape != (K,) or ix[0] < 0 or ix[-1] >= nvideo or not np.all(np.diff(ix)>0):
            raise ValueError(f'Invalid mapping {manual}')
        return ix, f'manual:{manual}'
    if K == nvideo and source_frames is not None and len(source_frames)==K and np.all(np.diff(source_frames)==1):
        return np.arange(K), 'verified_unit_step'
    return None, 'NOT_ALIGNED: needs explicit GT-to-video frame mapping'


def optical_gt_epe(flow_path, parts_path, registration, delta, source_frames, camera, model,
                   frame_map_dir, max_points, max_eval_pairs, seed):
    if not flow_path.exists():
        return None, 'NO_FLOW', 0
    flow = np.load(flow_path, mmap_mode='r')
    if flow.ndim!=4 or flow.shape[-1]!=2:
        raise ValueError('Invalid flow array')
    K,P = delta.shape[:2]
    nvideo=flow.shape[0]+1
    mapped,status = resolve_frame_map(model,source_frames,K,nvideo,frame_map_dir)
    if mapped is None:
        return None,status,0
    with np.load(parts_path, allow_pickle=False) as d:
        rest=d['rest_points']
        point_parts=d['point_parts']
        rng=np.random.default_rng(seed)
        ids = np.arange(len(rest)) if len(rest) <= max_points else np.sort(rng.choice(len(rest), max_points, replace=False))
        X=np.asarray(rest[ids], np.float64)
        owners=np.asarray(point_parts[ids],np.int32)
    # If Blender motion includes nonrigid deformation, object-matrix rigid prediction
    # is not a reliable GT surface motion for those particular parts. Exclude them.
    with np.load(parts_path, allow_pickle=False) as d:
        meta=json.loads(str(d['metadata_json'].item()))
    nonrigid={int(i) for i in meta.get('deforming_objects',[]) if isinstance(i,(int,np.integer))}
    keep=np.array([0<=i<P and i not in nonrigid for i in owners])
    X=X[keep]; owners=owners[keep]
    if len(X)==0:
        return None,'NO_RIGID_SURFACE_POINTS',0
    h,w=flow.shape[1:3]
    GT0 = X @ registration[:3,:3].T + registration[:3,3]
    times=np.arange(K-1)
    if len(times)>max_eval_pairs:
        times=np.unique(np.linspace(0,K-2,max_eval_pairs,dtype=int))
    err_sum, err_n=0.,0
    for t in times:
        # Blender rest_points are in world space of the first sampled Blender frame.
        # Relative world-object motion = M_t @ inverse(M_0).
        D0=delta[t,owners]
        D1=delta[t+1,owners]
        X0=np.einsum('nij,nj->ni', D0[:,:3,:3],X)+D0[:,:3,3]
        X1=np.einsum('nij,nj->ni', D1[:,:3,:3],X)+D1[:,:3,3]
        W0=X0 @ registration[:3,:3].T + registration[:3,3]
        W1=X1 @ registration[:3,:3].T + registration[:3,3]
        u0,z0,g0=project(W0,camera,h,w)
        u1,z1,g1=project(W1,camera,h,w)
        good=g0 & g1
        # Do not mix stationary background samples into motion GT EPE.
        # Exclude near-zero projected motion as these can hide failures on moving parts.
        good &= np.linalg.norm(u1-u0,axis=1) >= 0.05
        idx=np.flatnonzero(good)
        if not len(idx):
            continue
        # Conservative z-buffer for sparse projected reference samples.
        px=np.clip(np.rint(u0[idx,0]).astype(int),0,w-1)
        py=np.clip(np.rint(u0[idx,1]).astype(int),0,h-1)
        pix=py*w+px
        front=np.full(h*w,np.inf,np.float64)
        np.minimum.at(front,pix,z0[idx])
        idx=idx[z0[idx] <= front[pix] + 0.015]
        if not len(idx):
            continue
        uv=u0[idx].copy()
        valid=np.ones(len(idx),bool)
        # Compose SEA-RAFT forward flow when Blender samples correspond to nonconsecutive
        # video frames. This is a prediction of end displacement, not GT EPE per frame.
        for j in range(int(mapped[t]),int(mapped[t+1])):
            ff=sample_flow(flow[j],uv)
            valid &= np.isfinite(ff).all(axis=1)
            uv+=np.nan_to_num(ff, nan=0.0)
            valid &= (uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
        e=np.linalg.norm(uv-u1[idx],axis=1)
        use=valid & np.isfinite(e)
        if use.any():
            err_sum+=float(e[use].sum(dtype=np.float64))
            err_n+=int(use.sum())
    if err_n==0:
        return None,'NO_VALID_GT_FLOW_SAMPLES',0
    return err_sum/err_n, status, err_n


def draw_table(rows, dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    available={x.name for x in font_manager.fontManager.ttflist}
    for font in ('Times New Roman','Liberation Serif','Nimbus Roman','DejaVu Serif'):
        if font in available:
            plt.rcParams['font.family']=font
            break
    col_labels=['Model','Flow GT EPE ↓\n(px)','Precision ↑\n(%)','Recall ↑\n(%)','F1 ↑\n(%)']
    def fl(x, percent=False):
        return '—' if x is None else f'{float(x)*(100 if percent else 1):.2f}'
    cells=[[r['model'], fl(r.get('flow_gt_epe_px')), fl(r.get('precision'),True),
            fl(r.get('recall'),True),fl(r.get('f1'),True)] for r in rows]
    fig,ax=plt.subplots(figsize=(11.5,4.45),dpi=220)
    ax.axis('off')
    fig.patch.set_facecolor('white')
    ax.text(.5,.94,'FEAST — Motion Estimation Evaluation',ha='center',va='top',fontsize=18,fontweight='bold',transform=ax.transAxes)
    ax.text(.5,.85,'Five-object comparison with Blender reference motion',ha='center',va='top',fontsize=11,transform=ax.transAxes)
    tab=ax.table(cellText=cells,colLabels=col_labels,cellLoc='center',colLoc='center',
                 colWidths=[.19,.22,.20,.20,.19],bbox=[.025,.31,.95,.45])
    tab.auto_set_font_size(False)
    for (r,c),cell in tab.get_celld().items():
        cell.set_edgecolor('#333333');cell.set_linewidth(.6)
        cell.set_facecolor('#E9E9E9' if r==0 else 'white')
        cell.set_text_props(fontsize=12,fontweight='bold' if r==0 or c==0 else 'normal')
    ax.text(.035,.22,'GT motion: Blender object matrices; evaluation only, never used for FEAST motion initialization.',
            ha='left',fontsize=9,transform=ax.transAxes)
    ax.text(.035,.16,'Flow EPE: endpoint error of projected rigid surface motion at temporally aligned sampled frames.',
            ha='left',fontsize=9,transform=ax.transAxes)
    ax.text(.035,.10,'— = missing data, unverified frame mapping, or no valid GT samples; not a zero score.',
            ha='left',fontsize=9,transform=ax.transAxes)
    fig.savefig(dest,dpi=220,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--outputs',type=Path,default=Path('outputs'))
    ap.add_argument('--save-dir',type=Path,default=None)
    ap.add_argument('--frame-map-dir',type=Path,default=None,
                    help='Directory containing MODEL.json with exact sampled Blender pose -> video frame indices')
    ap.add_argument('--max-gt-points',type=int,default=12000)
    ap.add_argument('--max-eval-pairs',type=int,default=30)
    ap.add_argument('--motion-tol',type=float,default=1e-4)
    ap.add_argument('--seed',type=int,default=7)
    for m in MODELS:
        ap.add_argument('--binding-'+m,dest='binding_'+m,type=Path,default=None)
    args=ap.parse_args()
    outputs=args.outputs.resolve()
    save=(args.save_dir or outputs/'FEAST-metrics').resolve()
    save.mkdir(parents=True,exist_ok=True)
    rows=[]
    for model in MODELS:
        row={'model':model, 'flow_gt_epe_px':None, 'precision':None, 'recall':None, 'f1':None,
             'gt_epe_status':'', 'status':'OK'}
        try:
            run=outputs/model/'rgb_part_motion_se3_flow'
            confirm=find_stage_b(run,model)
            if confirm is None:
                raise FileNotFoundError(f'No Stage B motion_confirmation.json under {run}')
            confirmation=json_file(confirm/'motion_confirmation.json')
            parts=outputs/f'{model}-init'/'blender-surface-parts.npz'
            if not parts.exists():
                raise FileNotFoundError(str(parts))
            bind=getattr(args,'binding_'+model) or outputs/f'{model}-init'/'initial-binding.npz'
            if not bind.exists():
                candidates=list((outputs/f'{model}-init').glob('initial-binding*.npz'))
                if len(candidates)==1:
                    bind=candidates[0]
                else:
                    raise FileNotFoundError(f'No unique binding; specify --binding-{model}')
            mats, delta, moving, labelled, reg, frames=read_part_movement(parts,bind,args.motion_tol)
            row.update(classification_metrics(moving,labelled,confirmation))
            row['stage_b_dir']=str(confirm)
            row['binding_file']=str(bind)
            camera_file=outputs/f'{model}-inspection'/'inspection.json'
            flow_file=run/'searaft_flow.npy'
            if camera_file.exists():
                gt_epe,status,n=optical_gt_epe(flow_file,parts,reg,delta,frames,json_file(camera_file)['camera'],
                                               model,args.frame_map_dir,args.max_gt_points,args.max_eval_pairs,args.seed)
                row['flow_gt_epe_px']=gt_epe
                row['gt_epe_status']=status
                row['gt_epe_samples']=n
            else:
                row['gt_epe_status']='NO_CAMERA_INSPECTION'
        except Exception as exc:
            row['status']=f'{type(exc).__name__}: {exc}'
        print(f"{model:11s} F1={row.get('f1')} EPE={row.get('flow_gt_epe_px')} [{row['gt_epe_status']}] status={row['status']}",flush=True)
        rows.append(row)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    csv_path=save/'rime_gt_motion_metrics.csv'
    json_path=save/'rime_gt_motion_metrics.json'
    image_path=save/'rime_gt_motion_table.png'
    with csv_path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)
    json_path.write_text(json.dumps({'note':'Blender synthetic reference, not independent real-world GT; GT flow requires verified temporal alignment.',
                                    'rows':rows},ensure_ascii=False,indent=2),encoding='utf-8')
    draw_table(rows,image_path)
    print('Saved:',csv_path,json_path,image_path,sep='\n  ')

if __name__=='__main__':
    main()
