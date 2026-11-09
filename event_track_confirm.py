#!/usr/bin/env python3
"""Flow-first Gaussian candidates -> CoTracker3 / Event / depth verification.
No event density classifier; no classical optical flow fallback; no Blender motion GT.
"""
import argparse, json
from pathlib import Path
import cv2, numpy as np, torch
from scipy.ndimage import binary_erosion, binary_dilation, distance_transform_edt
from scipy.spatial import cKDTree
from FEAST.core.gaussian_part_tracking import GaussianPartRenderer
from FEAST.refine_motion_rgb import load_video


def pose_camera(renderer, w, h):
    cam=renderer.camera
    E=np.asarray(cam.world_view_transform.detach().cpu(),np.float64).T
    K=np.array([[w/(2*np.tan(float(cam.FoVx)/2)),0,w/2],
                [0,h/(2*np.tan(float(cam.FoVy)/2)),h/2],[0,0,1]],dtype=np.float64)
    return E,K

def projection(xyz,E,K):
    x=xyz@E[:3,:3].T+E[:3,3]
    uv=x[:,:2]/np.maximum(x[:,2:3],1e-8)
    return uv@K[:2,:2].T+K[:2,2],x[:,2]

def point_part_map(uv,z,labels,w,h):
    """Depth-ordered Gaussian centers; conservative approximation, not splat-alpha visibility."""
    pix=np.rint(uv).astype(np.int32)
    ok=(z>1e-6)&np.isfinite(uv).all(1)&(pix[:,0]>=0)&(pix[:,0]<w)&(pix[:,1]>=0)&(pix[:,1]<h)
    ids=np.flatnonzero(ok)
    ids=ids[np.argsort(z[ids])[::-1]]
    out=np.full((h,w),-1,np.int32)
    out[pix[ids,1],pix[ids,0]]=labels[ids]
    return out

def geometric_depth_visibility(xyz, labels, init_pose, tracks, vis, point_parts, E, K,
                               radius=3.0, depth_tolerance=0.03,
                               k_neighbors=48):
    """Conservative Part-aware surface visibility using *transformed G0* per frame.

    A point is accepted only if a projected Gaussian of its own Part is
    at the locally nearest visible surface. This is a center-splat
    approximation, NOT accurate alpha-composited Gaussian visibility.
    The camera and initial pose must use the same world coordinates.
    """
    T, N, _ = tracks.shape
    P = init_pose.shape[1]
    keep = np.zeros((T, N), dtype=bool)
    counts = []
    groups = [np.flatnonzero(labels == p) for p in range(P)]
    for t in range(T):
        current = np.flatnonzero(vis[t])
        if current.size == 0:
            counts.append((0, 0))
            continue
        transformed = np.empty_like(xyz, dtype=np.float64)
        for p, idx in enumerate(groups):
            if not idx.size:
                continue
            M = init_pose[t, p]
            transformed[idx] = xyz[idx] @ M[:3, :3].T + M[:3, 3]
        uv, z = projection(transformed, E, K)
        valid = (z > 1e-6) & np.isfinite(z) & np.isfinite(uv).all(1)
        source = np.flatnonzero(valid)
        tree = cKDTree(uv[source])
        query = tracks[t, current].astype(np.float64)
        finite = np.isfinite(query).all(1)
        frame_kept = np.zeros(current.size, dtype=bool)
        if np.any(finite):
            nearest_dist, near_i = tree.query(
                query[finite], k=min(k_neighbors, len(source)),
                distance_upper_bound=radius,
                workers=-1,
            )
            if nearest_dist.ndim == 1:
                nearest_dist = nearest_dist[:, None]
                near_i = near_i[:, None]
            good = np.isfinite(nearest_dist) & (near_i < len(source))
            safe_index = np.minimum(near_i, len(source)-1)
            gauss_index = source[safe_index]
            depths = np.where(good, z[gauss_index], np.inf)
            front_depth = np.min(depths, axis=1)
            # Candidate has to be near *frontmost* projected surface.
            own = labels[gauss_index] == point_parts[current[finite]][:, None]
            front_own = own & good & (depths <= front_depth[:, None] + depth_tolerance)
            frame_kept[finite] = np.any(front_own, axis=1)
        keep[t, current] = frame_kept
        counts.append((int(current.size), int(frame_kept.sum())))
        if t % 10 == 0 or t == T-1:
            print(f'Depth check frame {t:03d}: visible={current.size} depth-consistent={frame_kept.sum()}', flush=True)
    return keep, counts

def event_contour(event_path, t0, ms, h, w, minimum=2):
    ev=np.load(event_path,mmap_mode='r')
    assert ev.ndim==2 and ev.shape[1]>=4
    times=np.asarray(ev[:,2]); lo=np.searchsorted(times,t0,'left');hi=np.searchsorted(times,t0+ms/1000.,'left')
    if np.any(times[1:2000]<times[:1999]):
        e=ev[(times>=t0)&(times<t0+ms/1000.)]
    else:e=ev[lo:hi]
    pos=np.zeros((h,w),np.int32);neg=np.zeros_like(pos)
    xy=e[:,:2].astype(np.int32);good=(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
    xy=xy[good];pol=e[good,3]>0
    np.add.at(pos,(xy[pol,1],xy[pol,0]),1)
    np.add.at(neg,(xy[~pol,1],xy[~pol,0]),1)
    raw=((pos>=minimum)|(neg>=minimum)).astype(np.uint8)
    # Close tiny contour gaps, remove isolated noise. This is a binary contour geometry operation,
    # NOT thresholding event density by Part.
    raw=cv2.morphologyEx(raw,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    n,cc,stats,_=cv2.connectedComponentsWithStats(raw,8)
    chosen=np.zeros_like(raw)
    for i in range(1,n):
        if stats[i,cv2.CC_STAT_AREA]>=4:chosen[cc==i]=255
    return chosen,pos,neg

def select_points(partmap, uv, labels, candidate, first_rgb, max_points, margin, min_distance=3,
                  grid_stride=7, texture_quality=0.008, safety_limit=20000):
    """Single-frame Part-owned grid + Shi-Tomasi sampling. No reseeding."""
    h, w = partmap.shape
    gray = cv2.cvtColor(first_rgb, cv2.COLOR_BGR2GRAY)
    result=[]; owners=[]; counts=[]
    for pid in candidate:
        mask = (partmap == pid).astype(np.uint8)
        # Projected-center maps can be sparse: avoid erosion that destroys all pixels.
        # Never dilate across pixels owned by a different Part.
        valid = mask > 0
        if not valid.any():
            continue
        yy,xx = np.where(valid)
        # One representative point per grid cell (near the cell centre).
        cell_x = xx // grid_stride; cell_y = yy // grid_stride
        keys = cell_y.astype(np.int64)*(1+w//grid_stride)+cell_x
        center_err = (xx % grid_stride - grid_stride/2)**2 + (yy % grid_stride-grid_stride/2)**2
        order=np.lexsort((center_err, keys))
        sorted_keys=keys[order]
        first=np.r_[True,sorted_keys[1:]!=sorted_keys[:-1]]
        chosen=order[first]
        grid=np.stack([xx[chosen],yy[chosen]],axis=1).astype(np.float32)
        # Shi-Tomasi corners, strictly clipped to the same owner mask.
        corners=cv2.goodFeaturesToTrack(gray,maxCorners=max(100,int(h*w)),qualityLevel=texture_quality,
                                          minDistance=min_distance,mask=mask)
        if corners is not None:
            corners=corners[:,0,:].astype(np.float32)
            ix=np.rint(corners[:,0]).astype(int).clip(0,w-1)
            iy=np.rint(corners[:,1]).astype(int).clip(0,h-1)
            corners=corners[partmap[iy,ix]==pid]
        else:
            corners=np.empty((0,2),np.float32)
        # Deduplicate grid / texture observations spatially.
        combined=np.concatenate([grid,corners],axis=0)
        kept=[]
        occupancy=set()
        for pt in combined:
            px,py=float(pt[0]),float(pt[1])
            gx=int(round(px/min_distance));gy=int(round(py/min_distance))
            if (gx,gy) in occupancy:continue
            occupancy.add((gx,gy));kept.append(pt)
        if not kept:continue
        pts=np.stack(kept).astype(np.float32)
        # max_points<=0 means no per-Part fixed cap.
        if max_points>0:pts=pts[:max_points]
        if sum(len(a) for a in result)+len(pts)>safety_limit:
            pts=pts[:max(0,safety_limit-sum(len(a) for a in result))]
            print('Safety cap reached; increase --safety-max-points for more tracks',flush=True)
        if len(pts):result.append(pts);owners.extend([pid]*len(pts));counts.append((pid,len(pts)))
        if sum(len(x) for x in result)>=safety_limit:break
    if not result:return np.empty((0,2),np.float32),np.empty(0,np.int32),counts
    return np.concatenate(result),np.asarray(owners,np.int32),counts


def save_visualizations(out, frames, tracks, raw_vis, depth_vis, pids, flow,
                        stride=12, max_track_dots=1500):
    """Render CoTracker visibility/occlusion and external SEA-RAFT flow videos."""
    h,w=frames[0].shape[:2];T=len(frames)
    fourcc=cv2.VideoWriter_fourcc(*'mp4v')
    tw=cv2.VideoWriter(str(out/'cotracker_tracks_depth.mp4'),fourcc,15,(w,h))
    fw=cv2.VideoWriter(str(out/'searaft_flow.mp4'),fourcc,15,(w,h))
    colors={int(pid):tuple(int(x) for x in np.random.default_rng(int(pid)+42).integers(65,256,3)) for pid in np.unique(pids)}
    # Visualization subsampling only; does not affect tracking / motion statistics.
    selected=np.arange(len(pids))
    if len(selected)>max_track_dots:
        selected=np.unique(np.linspace(0,len(pids)-1,max_track_dots,dtype=np.int64))
    legend=(35,35,35)
    for t in range(T):
        img=frames[t].copy()
        tracked=raw_vis[t,selected].astype(bool)
        accepted=tracked & depth_vis[t,selected].astype(bool)
        for local_idx,idx in enumerate(selected):
            if not tracked[local_idx]:continue
            xy=tracks[t,idx]
            if not np.isfinite(xy).all():continue
            x,y=int(round(float(xy[0]))),int(round(float(xy[1])))
            if not(0<=x<w and 0<=y<h):continue
            # Visualization: draw all CoTracker-visible points.
            # Depth consistency remains a separate diagnostic.
            cv2.circle(img,(x,y),1,colors[int(pids[idx])],-1,cv2.LINE_AA)
        cv2.putText(img,f'CoTracker3 | F{t:03d} raw={raw_vis[t].sum()} depth-valid={(raw_vis[t]&depth_vis[t]).sum()}',
                    (8,20),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1,cv2.LINE_AA)
        tw.write(img)
        flow_idx=min(t,T-2)
        uv=np.asarray(flow[flow_idx],dtype=np.float32)
        mag=np.linalg.norm(uv,axis=2)
        ang=np.arctan2(uv[:,:,1],uv[:,:,0])
        hsv=np.zeros((h,w,3),np.uint8)
        hsv[:,:,0]=(((ang+np.pi)/(2*np.pi))*179).astype(np.uint8)
        hsv[:,:,1]=255
        scale=max(1.,float(np.percentile(mag[np.isfinite(mag)],95))) if np.isfinite(mag).any() else 1.
        hsv[:,:,2]=np.clip(mag/scale*255,0,255).astype(np.uint8)
        color=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        blended=cv2.addWeighted(frames[t],.4,color,.6,0)
        for y in range(stride//2,h,stride):
            for x in range(stride//2,w,stride):
                vx,vy=uv[y,x]
                if not(np.isfinite(vx) and np.isfinite(vy)):continue
                endpoint=(int(round(x+np.clip(vx,-stride,stride))),int(round(y+np.clip(vy,-stride,stride))))
                cv2.arrowedLine(blended,(x,y),endpoint,(230,230,230),1,tipLength=.25)
        cv2.putText(blended,f'SEA-RAFT | F{flow_idx:03d}->{flow_idx+1:03d}',(8,20),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1,cv2.LINE_AA)
        fw.write(blended)
    tw.release();fw.release()
    print('Visualizations:',out/'cotracker_tracks_depth.mp4',out/'searaft_flow.mp4',flush=True)

def cotracker(frames,xy,checkpoint,chunk_size):
    try:from cotracker.predictor import CoTrackerPredictor
    except ImportError as e:raise RuntimeError('Official CoTracker3 not installed. Install facebookresearch/co-tracker.') from e
    if not Path(checkpoint).is_file():raise FileNotFoundError('CoTracker3 checkpoint not found: '+checkpoint)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=CoTrackerPredictor(checkpoint=checkpoint).to(device).eval()
    video=torch.from_numpy(np.stack([cv2.cvtColor(f,cv2.COLOR_BGR2RGB) for f in frames])).permute(0,3,1,2).float()[None].to(device)
    xyq=np.concatenate([np.zeros((len(xy),1),np.float32),xy],axis=1)
    tracks=[];vis=[]
    with torch.no_grad():
        for start in range(0,len(xy),chunk_size):
            query=torch.from_numpy(xyq[start:start+chunk_size][None]).to(device)
            pred,v=model(video,queries=query,backward_tracking=False)
            tracks.append(pred[0].detach().cpu().numpy());vis.append(v[0].detach().cpu().numpy().astype(bool))
    return np.concatenate(tracks,axis=1),np.concatenate(vis,axis=1)


def save_cotracker_raw_tracks(out, frames, tracks, raw_vis, pids, trail=8):
    """Render original CoTracker3 tracks without depth filtering."""
    import cv2
    import numpy as np
    from pathlib import Path

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    T, N, _ = tracks.shape
    h, w = frames[0].shape[:2]

    video_path = out / "cotracker_raw_tracks.mp4"

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30.0,
        (w, h)
    )

    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video: {video_path}")

    colors = {
        1: (0, 0, 255),  # Pendulum: red
        2: (0, 255, 0),  # Weight: green
    }

    try:
        for t in range(T):
            img = frames[t].copy()

            for i in range(N):
                color = colors.get(int(pids[i]), (255, 255, 255))

                start = max(0, t - trail)

                for k in range(start + 1, t + 1):
                    if not (raw_vis[k - 1, i] and raw_vis[k, i]):
                        continue

                    a = tracks[k - 1, i]
                    b = tracks[k, i]

                    if not (np.isfinite(a).all() and np.isfinite(b).all()):
                        continue

                    pa = tuple(np.rint(a).astype(int))
                    pb = tuple(np.rint(b).astype(int))

                    cv2.line(img, pa, pb, color, 1, cv2.LINE_AA)

                if not raw_vis[t, i]:
                    continue

                xy = tracks[t, i]

                if not np.isfinite(xy).all():
                    continue

                x, y = np.rint(xy).astype(int)

                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(
                        img, (int(x), int(y)),
                        1, color, -1, cv2.LINE_AA
                    )

            cv2.putText(
                img,
                f"RAW CoTracker3 | Frame {t:03d}",
                (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            writer.write(img)

    finally:
        writer.release()

    print("Raw CoTracker3 trajectory video:", video_path, flush=True)



# ===== Part-wise fixed colors =====
def get_part_color(part_id):
    """Return a stable BGR color for each Gaussian Part."""
    import cv2
    import numpy as np

    hue = (int(part_id) * 47) % 180

    hsv = np.uint8([[[hue, 230, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]

    return tuple(int(x) for x in bgr)
# ==================================


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ('g0','binding','camera','video','events','flow-npy','cotracker-checkpoint','output'):
        ap.add_argument('--'+k,required=True)
    ap.add_argument('--frames',type=int,default=100);ap.add_argument('--width',type=int,default=640);ap.add_argument('--height',type=int,default=640)
    ap.add_argument('--event-t0',type=float,default=0.);ap.add_argument('--event-ms',type=float,default=20.)
    ap.add_argument('--event-pixel-min',type=int,default=2)
    ap.add_argument('--contour-radius',type=int,default=5);ap.add_argument('--min-contour-overlap',type=float,default=.025)
    ap.add_argument('--max-points',type=int,default=0);ap.add_argument('--point-margin',type=int,default=1);ap.add_argument('--point-min-distance',type=int,default=3)
    ap.add_argument('--grid-stride',type=int,default=7)
    ap.add_argument('--texture-quality',type=float,default=.008)
    ap.add_argument('--safety-max-points',type=int,default=20000)
    ap.add_argument('--vis-flow-stride',type=int,default=16)
    ap.add_argument('--cotracker-batch',type=int,default=250);ap.add_argument('--min-tracks',type=int,default=8)
    ap.add_argument('--flow-candidates',required=True,help='flow_first_gaussians.py output/flow_candidate_parts.json')
    ap.add_argument('--pose-init',required=True,help='Existing SE(3) delta_g0 sequence, needed to move G0 before framewise depth verification')
    ap.add_argument('--depth-radius',type=float,default=3.0,help='2D projected neighbor search radius, pixels')
    ap.add_argument('--depth-tolerance',type=float,default=0.03,help='Depth tolerance in same units as G0')
    ap.add_argument('--min-moving-fraction',type=float,default=.25);ap.add_argument('--min-displacement',type=float,default=2.)
    ap.add_argument('--max-flow-error',type=float,default=4.);ap.add_argument('--max-flow-samples',type=int,default=10)
    ap.add_argument('--max-moving-parts',type=int,default=0,help='0: no presumed number of moving parts')
    args=ap.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    frames,fps=load_video(args.video,args.frames,args.width,args.height);T=len(frames);h,w=frames[0].shape[:2]
    with np.load(args.binding,allow_pickle=False) as z:labels=z['labels'].astype(np.int32)
    renderer=GaussianPartRenderer(args.g0,labels,args.camera,w,h)
    xyz=renderer.xyz0.detach().cpu().numpy();P=renderer.parts
    E,K=pose_camera(renderer,w,h);uv,z=projection(xyz,E,K)
    pm=point_part_map(uv,z,labels,w,h)
    contour,pos,neg=event_contour(args.events,args.event_t0,args.event_ms,h,w,args.event_pixel_min)
    # A contour of each Part, not Part-specific event mass or density.
    contour_fatten=cv2.dilate(contour,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*args.contour_radius+1,)*2))>0
    rows=[];candidate=[]
    for p in range(P):
        mask=np.uint8(pm==p)
        edge=cv2.morphologyEx(mask,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))>0
        n=int(edge.sum());hits=int((edge&contour_fatten).sum());ratio=hits/max(n,1)
        row={'part':p,'visible_edge_px':n,'matched_edge_px':hits,'contour_coverage':ratio,'candidate':n>=10 and ratio>=args.min_contour_overlap}
        rows.append(row)
        if row['candidate']:candidate.append(p)
    rgba=cv2.cvtColor(frames[0],cv2.COLOR_BGR2RGB)
    rgba[pos>0]=[255,70,70];rgba[neg>0]=[45,135,255]
    cv2.imwrite(str(out/'event_20ms_overlay.png'),cv2.cvtColor(rgba,cv2.COLOR_RGB2BGR))
    (out/'event_contour_scores.json').write_text(json.dumps(rows,indent=2))
    from FEAST.event_part_visualization import save_event_part_visualization
    save_event_part_visualization(
        out, frames[0], pos, neg, pm, uv, labels,
        list(candidate), rows
    )
    event_candidate=list(candidate)
    flow_result=json.loads(Path(args.flow_candidates).read_text())
    candidate=[int(p) for p in flow_result['candidate_parts'] if 0<=int(p)<P]
    if not candidate:raise RuntimeError('Flow-first found no moving Part candidates. Inspect flow_candidate_gaussian_centers.png; not falling back to event candidates.')
    print('Flow-first candidate Parts:',candidate,'event corroboration:',event_candidate,flush=True)
    flow=np.load(args.flow_npy,mmap_mode='r')
    if flow.shape != (T-1,h,w,2):raise ValueError(f'SEA-RAFT flow expected {(T-1,h,w,2)}, got {flow.shape}')
    if not np.isfinite(flow[0]).all():raise ValueError('SEA-RAFT flow has nonfinite entries')
    pts,pids,counts=select_points(pm,uv,labels,candidate,frames[0],args.max_points,args.point_margin,args.point_min_distance,args.grid_stride,args.texture_quality,args.safety_max_points)
    if not len(pts):raise RuntimeError('No CoTracker RGB feature points inside event contour candidate Parts')
    print('Initial per-Part points:',counts,'total:',len(pts),flush=True)
    tracks,vis=cotracker(frames,pts,args.cotracker_checkpoint,args.cotracker_batch)
    with np.load(args.pose_init, allow_pickle=False) as zpose:
        if 'delta_g0' not in zpose:
            raise KeyError('Pose NPZ requires delta_g0 (T,P,4,4)')
        delta=np.asarray(zpose['delta_g0'],dtype=np.float64)
    if delta.ndim!=4 or delta.shape[0]<T or delta.shape[1:]!=(P,4,4):
        raise ValueError(f'Invalid pose sequence: expected (>= {T},{P},4,4), got {delta.shape}')
    # Stage B: use CoTracker3 visibility directly.
    # Static G0 depth verification is not applicable to moving Parts
    # before their SE(3) trajectories have been estimated.
    raw_vis = vis.copy()
    depth_vis = raw_vis.copy()  # Compatibility with visualization/output
    vis = raw_vis.copy()

    save_visualizations(
        out, frames, tracks, raw_vis, depth_vis, pids, flow,
        stride=args.vis_flow_stride
    )

    # Automatically export raw CoTracker3 trajectories.
    save_cotracker_raw_tracks(
        out, frames, tracks, raw_vis, pids, trail=8
    )

    # Motion evidence: persistent visible 2D displacement above noise + agreement with independent SEA-RAFT flow.
    # Crucially, the model does not assume that every event candidate moves.
    summary=[];moving=[];uncertain=[]
    timestamps=list(range(0,T-1,max(1,(T-1)//args.max_flow_samples)))[:args.max_flow_samples]
    for p in range(P):
        ind=np.flatnonzero(pids==p)
        if p not in candidate or len(ind)<args.min_tracks:
            summary.append({'part':p,'status':'unknown','reason':'no flow candidate or insufficient usable visible CoTracker points','tracks':len(ind)});uncertain.append(p);continue
        good=np.zeros((len(ind),),dtype=bool);motion=np.zeros_like(good)
        deviations=[];mags=[]
        for j,ix in enumerate(ind):
            vel=np.diff(tracks[:,ix,:],axis=0)
            pair=vis[:-1,ix]&vis[1:,ix]
            q=np.flatnonzero(pair)
            if len(q)<max(5,int(.15*(T-1))):continue
            track_magnitude=float(np.median(np.linalg.norm(vel[q],axis=1)))
            # Long-term displacement or repeated measured inter-frame motion above static noise.
            moving_trace=(track_magnitude>=args.min_displacement)
            errs=[]
            for t in timestamps:
                if not pair[t]:continue
                x,y=np.rint(tracks[t,ix]).astype(int)
                if 0<=x<w and 0<=y<h:
                    errs.append(np.linalg.norm(vel[t]-flow[t,y,x]))
            if len(errs)>=2:
                deviations.append(float(np.median(errs)))
                good[j]=np.median(errs)<=args.max_flow_error
                if good[j]:mags.append(track_magnitude)
                motion[j]=moving_trace and good[j]
        moving_fraction=float(np.sum(motion)/max(1,np.sum(good)))
        agreement=float(np.mean(good)) if len(good) else 0.
        status='moving' if (np.sum(good)>=args.min_tracks and moving_fraction>=args.min_moving_fraction and mags and np.median(mags)>=args.min_displacement) else 'unknown'
        row={'part':p,'status':status,'points':len(ind),'agreement_fraction':agreement,'consistent_points':int(np.sum(good)), 'moving_fraction':moving_fraction,'median_flow_error':float(np.median(deviations)) if deviations else None,'median_motion_px':float(np.median(mags)) if mags else None}
        summary.append(row)
        if status=='moving':moving.append(p)
        if status=='unknown':uncertain.append(p)
    if args.max_moving_parts>0 and len(moving)>args.max_moving_parts:
        rank={a['part']:a.get('moving_fraction',0) for a in summary}
        moving=sorted(moving,key=lambda p:rank[p],reverse=True)[:args.max_moving_parts]
    if not moving:print('WARNING: No Parts confirmed moving; writing diagnostics without forcing a moving Part.',flush=True)
    (out/'motion_confirmation.json').write_text(json.dumps({'flow_candidates':candidate,'event_candidates':event_candidate,'moving_parts':moving,'unknown_parts':uncertain,'part_scores':summary,'tracker':'official_cotracker3','flow':'pretrained_SEA_RAFT_external'},indent=2))
    np.savez_compressed(out/'tracking_evidence.npz',tracks=tracks,visibility=vis,raw_visibility=raw_vis,depth_visibility=depth_vis,initial_points=pts,point_parts=pids,query_frame_indices=np.zeros(len(pts),dtype=np.int32),frame_indices=np.arange(T,dtype=np.int32))
    print('Flow candidates:',candidate,'CoTracker3 + SEA-RAFT confirmed:',moving,flush=True)
if __name__=='__main__':main()
