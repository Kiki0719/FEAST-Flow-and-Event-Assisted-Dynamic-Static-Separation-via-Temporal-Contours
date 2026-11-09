"""Offline diagnostics for 2D tracking, PnP, temporal completion and occlusion.
Diagnostic consistency metrics are NOT ground-truth tracking accuracy.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def _csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        out = csv.DictWriter(f, fieldnames=list(rows[0]))
        out.writeheader();out.writerows(rows)


def _scalar(x):
    return float(x) if np.isfinite(x) else None


def _color(part):
    return tuple(int(x) for x in cv2.applyColorMap(np.array([[int(part*37)%256]], np.uint8), cv2.COLORMAP_TURBO)[0,0])


def generate_diagnostics(output_dir, video, *, frames, start_frame=0, save_video=False,
                         initial_ids=None, initial_depth=None, classification=None,
                         occlusion_report=None, extent=1.0):
    out = Path(output_dir); diag = out/'diagnostics';diag.mkdir(parents=True,exist_ok=True)
    with np.load(out/'video_part_poses_raw.npz') as z:
        raw=z['delta_g0']; measured=z['valid'];inliers=z['inliers']; errors=z['reprojection_median_px']
    with np.load(out/'video_part_poses.npz') as z:
        final=z['delta_g0'];unknown=z['unknown'];interpolated=z['interpolated'];
        inferred=z['occlusion_inferred'];static=z['static_assumed'];reliable=z['reliable']
    T,P=measured.shape
    N=T*P
    arrays={key: np.full((T,P),np.nan,np.float32) for key in (
        'query_count','visible_count','visibility_fraction','median_displacement_px',
        'median_step_px','p95_step_px','outside_fraction','mask_membership_fraction')}
    points_by_frame=[[] for _ in range(T)] if save_video else None
    raw_points_by_frame=[[] for _ in range(T)] if save_video else None
    validation_counts=np.zeros((T,P,6),np.int64)
    validation_names=['invisible','candidate','depth_occluded','cross_part_drift','pnp_outlier','pnp_inlier']
    files=sorted((diag/'cotracker_windows').glob('window_*.npz'))
    for path in files:
        with np.load(path) as z:
            start=int(z['start_frame']); ids=z['part_ids'];queries=z['query_xy'];tracks=z['tracks_xy'];vis=z['visibility']
            validation=z['validation'] if 'validation' in z.files else None
        for j in range(len(tracks)):
            t=start+j
            if not 0<=t<T:continue
            # Skip repeated anchor (the last point in the previous window).
            if j==0 and start>0:continue
            for p in np.unique(ids):
                index=np.flatnonzero(ids==p)
                if not len(index):continue
                xy=tracks[j,index]; visible=vis[j,index]>0.5
                good=visible & np.isfinite(xy).all(axis=1)
                disp=np.linalg.norm(xy-queries[index],axis=1)
                h,w=initial_ids.shape if initial_ids is not None else (10**9,10**9)
                outside=(xy[:,0]<0)|(xy[:,0]>=w)|(xy[:,1]<0)|(xy[:,1]>=h)
                arrays['query_count'][t,p]=len(index)
                arrays['visible_count'][t,p]=int(good.sum())
                arrays['visibility_fraction'][t,p]=float(good.mean())
                arrays['outside_fraction'][t,p]=float(outside.mean())
                if good.any():
                    arrays['median_displacement_px'][t,p]=float(np.median(disp[good]))
                    if j>0:
                        step=np.linalg.norm(tracks[j,index]-tracks[j-1,index],axis=1)
                        valid_step=good & (vis[j-1,index]>0.5) & np.isfinite(step)
                        if valid_step.any():
                            arrays['median_step_px'][t,p]=float(np.median(step[valid_step]))
                            arrays['p95_step_px'][t,p]=float(np.percentile(step[valid_step],95))
                    if j==0 and initial_ids is not None:
                        xi=np.rint(xy[good,0]).astype(int).clip(0,w-1)
                        yi=np.rint(xy[good,1]).astype(int).clip(0,h-1)
                        arrays['mask_membership_fraction'][t,p]=float(np.mean(initial_ids[yi,xi]==p))
                if validation is not None:
                    counts=np.bincount(validation[j,index].astype(int),minlength=6)[:6]
                    validation_counts[t,p] += counts
                if points_by_frame is not None:
                    for k in np.flatnonzero(good):
                        raw_points_by_frame[t].append((int(p),float(xy[k,0]),float(xy[k,1])))
                        if validation is None or validation[j,index[k]] in (1,5):
                            points_by_frame[t].append((int(p),float(xy[k,0]),float(xy[k,1])))
    # Rotation relative to initial and between two consecutively measured poses.
    rot=np.full((T,P),np.nan);jump=np.full((T,P),np.nan);trans=np.full((T,P),np.nan);tjump=np.full((T,P),np.nan)
    for p in range(P):
        ids=np.flatnonzero(measured[:,p]);prev=None
        for t in ids:
            M=raw[t,p]
            if not np.isfinite(M).all():continue
            try:
                rot[t,p]=float(np.rad2deg(Rotation.from_matrix(M[:3,:3]).magnitude()))
                trans[t,p]=float(np.linalg.norm(M[:3,3]))
                if prev is not None:
                    A=raw[prev,p]
                    r=Rotation.from_matrix(A[:3,:3]).inv()*Rotation.from_matrix(M[:3,:3])
                    jump[t,p]=float(np.rad2deg(r.magnitude())/max(t-prev,1))
                    tjump[t,p]=float(np.linalg.norm(M[:3,3]-A[:3,3])/max(t-prev,1))
                prev=t
            except ValueError: pass
    rows=[]
    for t in range(T):
        for p in range(P):
            d={'frame':t,'part':p,'pnp_observed':int(measured[t,p]),
               'pnp_inliers':int(inliers[t,p]),
               'pnp_median_reprojection_px':_scalar(errors[t,p]),
               'pose_rotation_from_t0_deg':_scalar(rot[t,p]),
               'pose_translation_from_t0':_scalar(trans[t,p]),
               'rotation_jump_deg_per_frame':_scalar(jump[t,p]),
               'translation_jump_per_frame':_scalar(tjump[t,p]),
               'interpolated':int(interpolated[t,p]),'inferred':int(inferred[t,p]),
               'static_assumed':int(static[t,p]),'unknown':int(unknown[t,p]),
               'reliable':int(reliable[t,p])}
            d.update({k:_scalar(v[t,p]) for k,v in arrays.items()})
            d.update({f'point_{name}':int(validation_counts[t,p,i])
                      for i,name in enumerate(validation_names)})
            rows.append(d)
    _csv(diag/'per_frame_part_metrics.csv',rows)
    def nanstat(z, f, default=None):
        v=np.asarray(z);v=v[np.isfinite(v)];return float(f(v)) if len(v) else default
    statuses={int(x['part']):x for x in classification or []}
    summary=[]
    for p in range(P):
        selection=errors[:,p][measured[:,p]]
        selection=selection[np.isfinite(selection)]
        summary.append({'part':p,'classification':statuses.get(p,{}).get('status','not_recorded'),
            'observed_frames':int(measured[:,p].sum()),'unknown_frames':int(unknown[:,p].sum()),
            'interpolated_frames':int(interpolated[:,p].sum()),'inferred_frames':int(inferred[:,p].sum()),
            'static_assumed_frames':int(static[:,p].sum()),
            'pnp_reprojection_median_px':nanstat(selection,np.median),
            'pnp_inliers_median':nanstat(inliers[:,p][measured[:,p]],np.median),
            'rotation_jump_p95_deg_per_frame':nanstat(jump[:,p],lambda v:np.percentile(v,95)),
            'translation_jump_p95_per_frame':nanstat(tjump[:,p],lambda v:np.percentile(v,95)),
            'tracker_visibility_mean':nanstat(arrays['visibility_fraction'][:,p],np.mean),
            'tracker_step_p95_px':nanstat(arrays['p95_step_px'][:,p],lambda v:np.percentile(v,95)),
            'tracker_query_count_max':nanstat(arrays['query_count'][:,p],np.max),
            **{f'point_{name}_total':int(validation_counts[:,p,i].sum())
               for i,name in enumerate(validation_names)}})
    _csv(diag/'per_part_summary.csv',summary)
    warnings=[]
    if not files: warnings.append('No saved 2D CoTracker windows: LK backend or --verify not used during tracking.')
    if np.any(unknown):warnings.append('Unknown poses remain: finite placeholder matrices are not valid motion.')
    warnings.append('These are self-consistency / observability metrics, NOT ground-truth 2D tracking accuracy.')
    warnings.append('PnP reprojection error alone cannot prove a 3D rigid pose is correct.')
    report={'frames':T,'parts':P,'extent':float(extent),'tracker_windows_saved':len(files),
            'warnings':warnings,'per_part':summary,'occlusion':occlusion_report}
    (diag/'verification_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    if initial_ids is not None:
        ids=np.asarray(initial_ids);base=np.zeros((*ids.shape,3),np.uint8)
        for p in range(P):base[ids==p]=_color(p)
        cv2.imwrite(str(diag/'initial_part_visibility.png'),base)
        if initial_depth is not None:
            z=np.asarray(initial_depth);finite=np.isfinite(z)&(z>0)
            dep=np.zeros_like(z,dtype=np.uint8)
            if finite.any():
                lo,hi=np.percentile(z[finite],[2,98]);dep[finite]=np.uint8(255*np.clip((z[finite]-lo)/max(hi-lo,1e-6),0,1))
            cv2.imwrite(str(diag/'initial_depth.png'),cv2.applyColorMap(dep,cv2.COLORMAP_VIRIDIS))
    # Universal numeric heatmap that is readable without matplotlib.
    cell_w,cell_h=10,26
    heat=np.zeros((P*cell_h,T*cell_w,3),np.uint8)
    for p in range(P):
        for t in range(T):
            if unknown[t,p]: color=(32,35,210)
            elif inferred[t,p]:color=(230,180,30)
            elif interpolated[t,p]:color=(200,145,35)
            elif static[t,p]:color=(110,110,110)
            elif measured[t,p]:color=(40,160,60)
            else:color=(50,50,50)
            cv2.rectangle(heat,(t*cell_w,p*cell_h),((t+1)*cell_w-1,(p+1)*cell_h-2),color,-1)
        cv2.putText(heat,f'{p:02d}',(2,p*cell_h+17),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
    cv2.imwrite(str(diag/'pose_status_timeline.png'),heat)
    if save_video:
        cap=cv2.VideoCapture(str(video));cap.set(cv2.CAP_PROP_POS_FRAMES,start_frame)
        fps=cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps<=0:fps=30.
        writer=None; raw_writer=None
        try:
            for t in range(T):
                ok,frame=cap.read()
                if not ok:break
                h,w=frame.shape[:2]
                if writer is None:
                    writer=cv2.VideoWriter(str(diag/'tracking_overlay.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
                    raw_writer=cv2.VideoWriter(str(diag/'tracking_raw_overlay.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
                    if not writer.isOpened() or not raw_writer.isOpened():
                        raise RuntimeError('OpenCV could not write tracking overlay videos')
                if initial_ids is not None and t==0:
                    import cv2 as _cv2
                    for p in range(P):
                        mask=np.uint8(initial_ids==p)
                        cnt,_=_cv2.findContours(mask,_cv2.RETR_EXTERNAL,_cv2.CHAIN_APPROX_SIMPLE)
                        _cv2.drawContours(frame,cnt,-1,_color(p),1)
                raw_frame=frame.copy()
                for p,x,y in raw_points_by_frame[t]:
                    if 0<=x<w and 0<=y<h:cv2.circle(raw_frame,(int(x),int(y)),2,_color(p),-1)
                for p,x,y in points_by_frame[t]:
                    if 0<=x<w and 0<=y<h:cv2.circle(frame,(int(x),int(y)),2,_color(p),-1)
                statuses_txt=f'Frame {t}/{T-1}  obs {int(measured[t].sum())}/{P}  unknown {int(unknown[t].sum())}'
                cv2.rectangle(frame,(0,0),(min(w-1,590),29),(10,10,10),-1)
                cv2.putText(frame,statuses_txt,(8,21),cv2.FONT_HERSHEY_SIMPLEX,.53,(255,255,255),1,cv2.LINE_AA)
                writer.write(frame)
                raw_writer.write(raw_frame)
        finally:
            cap.release()
            if writer is not None:writer.release()
            if raw_writer is not None:raw_writer.release()
    print('[Verification] Reports and visualizations:',diag,flush=True)
    return report
