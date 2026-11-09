#!/usr/bin/env python3
"""SEA-RAFT FLOW FIRST: project canonical G0 into fixed camera; test moving Gaussian centers.
Results are candidates (not optical-flow-derived 3D motions). Does NOT overwrite Binding.
"""
import argparse,json
from pathlib import Path
import numpy as np
import cv2
from plyfile import PlyData

def project(xyz,meta,w,h):
    pose=np.asarray(meta['camera']['pose'],dtype=np.float64)
    if pose.shape!=(4,4):raise ValueError('camera pose must be 4x4 OpenGL C2W')
    E=np.linalg.inv(pose@np.diag([1.,-1.,-1.,1.]))
    xyzcv=xyz@E[:3,:3].T+E[:3,3]
    fy=h/(2*np.tan(np.deg2rad(float(meta['camera']['fovy_degrees']))/2))
    fx=fy*w/h
    z=xyzcv[:,2];uv=np.full((len(xyz),2),np.nan)
    ok=z>1e-6
    uv[ok,0]=xyzcv[ok,0]/z[ok]*fx+w/2
    uv[ok,1]=xyzcv[ok,1]/z[ok]*fy+h/2
    ok&=np.isfinite(uv).all(axis=1)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
    return uv,z,ok


def compute(xyz, labels, camera, flow, threshold=1.0,
            min_fraction=.25, max_samples=12,
            depth_tol=.025, part_fraction=.04,
            min_part_points=20):

    T, h, w, c = flow.shape
    if c != 2:
        raise ValueError("Flow last dim must be 2")

    # 1. Project canonical G0 Gaussian centers at frame 0.
    uv, z, good = project(xyz, camera, w, h)

    pix = np.zeros((len(xyz), 2), dtype=np.int32)
    pix[good] = np.rint(uv[good]).astype(np.int32)

    good &= (
        (pix[:, 0] >= 0) & (pix[:, 0] < w) &
        (pix[:, 1] >= 0) & (pix[:, 1] < h)
    )

    # 2. Initial visible Gaussian centers (same z-buffer logic).
    ids = np.flatnonzero(good)
    flat = pix[ids, 1] * w + pix[ids, 0]

    depth = np.full(h * w, np.inf, dtype=np.float64)
    np.minimum.at(depth, flat, z[ids])

    front = good.copy()
    front[ids] = z[ids] <= depth[flat] + depth_tol

    ids = np.flatnonzero(front)
    n = len(xyz)

    positive = np.zeros(n, dtype=np.uint32)
    valid = np.zeros(n, dtype=np.uint32)
    mags = np.zeros(n, dtype=np.float32)

    # 3. Initialize tracked coordinates at the first RGB frame.
    initial = uv[ids].astype(np.float32)
    current = initial.copy()

    active = np.ones(len(ids), dtype=bool)

    # Track maximum and accumulated displacement for diagnostics.
    max_displacement = np.zeros(len(ids), dtype=np.float32)

    # 4. Propagate each Gaussian's 2D sampling position
    #    through every consecutive SEA-RAFT flow field.
    for t in range(T):

        active_ids = np.flatnonzero(active)
        if len(active_ids) == 0:
            break

        xy = current[active_ids]

        inside = (
            np.isfinite(xy).all(axis=1) &
            (xy[:, 0] >= 0) & (xy[:, 0] <= w - 1) &
            (xy[:, 1] >= 0) & (xy[:, 1] <= h - 1)
        )

        active[active_ids[~inside]] = False
        active_ids = active_ids[inside]

        if len(active_ids) == 0:
            continue

        xy = current[active_ids]

        # Bilinear sampling at the propagated subpixel locations.
        flow_frame = np.asarray(flow[t], dtype=np.float32)

        # OpenCV remap requires each output dimension < 32767.
        # Sample Gaussian trajectories in bounded batches.
        sampled = np.empty((len(xy), 2), dtype=np.float32)

        batch_size = 16384

        for start in range(0, len(xy), batch_size):
            end = min(start + batch_size, len(xy))

            sampled[start:end] = cv2.remap(
                flow_frame,
                xy[start:end, 0].reshape(-1, 1),
                xy[start:end, 1].reshape(-1, 1),
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            ).reshape(-1, 2)

        finite = np.isfinite(sampled).all(axis=1)

        active[active_ids[~finite]] = False
        active_ids = active_ids[finite]
        sampled = sampled[finite]

        if len(active_ids) == 0:
            continue

        gaussian_ids = ids[active_ids]

        # Update tracked locations.
        next_xy = current[active_ids] + sampled

        next_inside = (
            np.isfinite(next_xy).all(axis=1) &
            (next_xy[:, 0] >= 0) & (next_xy[:, 0] <= w - 1) &
            (next_xy[:, 1] >= 0) & (next_xy[:, 1] <= h - 1)
        )

        active[active_ids[~next_inside]] = False

        active_ids = active_ids[next_inside]
        gaussian_ids = gaussian_ids[next_inside]
        sampled = sampled[next_inside]
        next_xy = next_xy[next_inside]

        if len(active_ids) == 0:
            continue

        current[active_ids] = next_xy

        flow_magnitude = np.linalg.norm(sampled, axis=1)

        # Displacement relative to the original frame,
        # not flow sampled repeatedly at the original pixel.
        displacement = np.linalg.norm(
            next_xy - initial[active_ids],
            axis=1
        )

        valid[gaussian_ids] += 1
        positive[gaussian_ids] += (
            flow_magnitude >= threshold
        ).astype(np.uint32)

        mags[gaussian_ids] += flow_magnitude.astype(np.float32)

        max_displacement[active_ids] = np.maximum(
            max_displacement[active_ids],
            displacement
        )

        if t % 10 == 0 or t == T - 1:
            print(
                f"Trajectory flow {t + 1}/{T}, "
                f"active={int(active.sum())}, "
                f"moving_observations={int(positive.sum())}",
                flush=True
            )

    # 5. Time-consistent motion evidence.
    score = positive / np.maximum(valid, 1)

    moving = (
        (valid >= max(2, T // 3)) &
        (score >= min_fraction) &
        front
    )

    means = mags / np.maximum(valid, 1)

    # 6. Aggregate motion evidence by original Blender Part IDs.
    stats = []
    parts = []

    for p in np.unique(labels):

        owned = (labels == p) & front

        nfront = int(owned.sum())
        nc = int((owned & moving).sum())

        fraction = nc / max(1, nfront)

        chosen = (
            nfront >= min_part_points and
            nc >= min_part_points and
            fraction >= part_fraction
        )

        stats.append({
            "part": int(p),
            "front_gaussians": nfront,
            "flow_moving_gaussians": nc,
            "moving_fraction": float(fraction),
            "flow_candidate": bool(chosen)
        })

        if chosen:
            parts.append(int(p))

    print(
        "Trajectory-following flow completed:",
        "moving Gaussians =", int(moving.sum()),
        "candidate Parts =", parts,
        flush=True
    )

    return moving, score, means, front, parts, stats, uv

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ('g0','binding','camera','flow-npy','output'):ap.add_argument('--'+k,required=True)
    ap.add_argument('--flow-threshold',type=float,default=1.0)
    ap.add_argument('--min-moving-fraction',type=float,default=.25)
    ap.add_argument('--min-part-fraction',type=float,default=.40)
    ap.add_argument('--min-part-points',type=int,default=20)
    ap.add_argument('--max-samples',type=int,default=12)
    ap.add_argument('--depth-tolerance',type=float,default=.025)
    args=ap.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    v=PlyData.read(args.g0)['vertex'].data
    xyz=np.stack([v['x'],v['y'],v['z']],axis=-1).astype(np.float64)
    with np.load(args.binding,allow_pickle=False) as z:labels=z['labels'].astype(np.int32)
    if len(labels)!=len(xyz):raise ValueError('G0 and Binding Gaussian count mismatch')
    camera=json.loads(Path(args.camera).read_text())
    flow=np.load(args.flow_npy,mmap_mode='r')
    if flow.ndim!=4 or flow.shape[-1]!=2:raise ValueError('Expected flow (T-1,H,W,2)')
    moving,score,mag,front,parts,stats,uv=compute(xyz,labels,camera,flow,args.flow_threshold,args.min_moving_fraction,args.max_samples,args.depth_tolerance,args.min_part_fraction,args.min_part_points)
    idx=np.flatnonzero(moving).astype(np.int32)
    np.savez_compressed(out/'flow_candidates.npz',candidate_dynamic_indices=idx,flow_score=score.astype(np.float32),flow_magnitude=mag,front_visible=front,labels=labels)
    (out/'flow_candidate_parts.json').write_text(json.dumps({'candidate_parts':parts,'part_scores':stats,'method':'visible Gaussian center + SEA-RAFT displacement magnitude','limitations':['Flow is 2D evidence, not 3D motion','center-zbuffer approximate visibility','camera and G0 must be in same world coordinates']},indent=2))
    H,W=flow.shape[1:3]; canvas=np.zeros((H,W,3),np.uint8)
    pi=np.rint(uv[idx]).astype(np.int32); inb=(pi[:,0]>=0)&(pi[:,0]<W)&(pi[:,1]>=0)&(pi[:,1]<H)
    canvas[pi[inb,1],pi[inb,0]]=(0,0,255)
    cv2.imwrite(str(out/'flow_candidate_gaussian_centers.png'),canvas)
    print('Flow candidate Gaussians:',len(idx),'candidate Parts:',parts)
if __name__=='__main__': main()
