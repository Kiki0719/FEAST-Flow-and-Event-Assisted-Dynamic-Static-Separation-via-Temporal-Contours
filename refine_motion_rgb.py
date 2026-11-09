#!/usr/bin/env python3
"""Differentiable RGB-frame supervision of rigid Part transforms.

Keeps the original G0 colors, opacities, scales, and camera frozen.  CoTracker/PnP
is only an optional pose initialization.  No Blender animation ground truth.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from FEAST.core.gaussian_part_tracking import GaussianPartRenderer
from FEAST.core.rigid_pose_optimization import joint_se3_regularizer


def axis_angle_to_rotation(w):
    # Differentiable Rodrigues with stable small-angle coefficients.
    theta2 = (w * w).sum()
    theta = torch.sqrt(theta2 + 1e-12)
    A = torch.where(theta2 < 1e-6, 1 - theta2 / 6 + theta2 * theta2 / 120,
                    torch.sin(theta) / theta)
    B = torch.where(theta2 < 1e-6, 0.5 - theta2 / 24 + theta2 * theta2 / 720,
                    (1 - torch.cos(theta)) / (theta2 + 1e-12))
    wx, wy, wz = w.unbind()
    z = torch.zeros_like(wx)
    K = torch.stack((z, -wz, wy, wz, z, -wx, -wy, wx, z)).reshape(3, 3)
    I = torch.eye(3, dtype=w.dtype, device=w.device)
    return I + A * K + B * (K @ K)


def axis_angle_quaternion(w):
    theta2 = (w * w).sum()
    theta = torch.sqrt(theta2 + 1e-12)
    coef = torch.where(theta2 < 1e-6,
                       0.5 - theta2 / 48 + theta2 * theta2 / 3840,
                       torch.sin(theta / 2) / theta)
    return torch.cat((torch.cos(theta / 2).reshape(1), w * coef))


def qmultiply(q, r):
    # q (4) and r (N,4), scalar first.
    w, x, y, z = q.unbind()
    a, b, c, d = r.unbind(-1)
    return torch.stack((w*a-x*b-y*c-z*d,
                        w*b+x*a+y*d+z*c,
                        w*c-x*d+y*a+z*b,
                        w*d+x*c-y*b+z*a), dim=1)


def torch_to_matrix(rotvec, shift, center):
    R = axis_angle_to_rotation(rotvec)
    t = center + shift - R @ center
    return R, t


def render_with_grad(renderer, part_ids, rotvecs, shifts, centers):
    """Build a differentiable full Gaussian tensor, preserving all static parts."""
    xyz = renderer.xyz0
    quat = renderer.rot0
    for k, pid in enumerate(part_ids):
        inds = renderer.ids[pid]
        if len(inds) == 0:
            continue
        R, t = torch_to_matrix(rotvecs[k], shifts[k], centers[k])
        points = renderer.xyz0.index_select(0, inds) @ R.T + t
        q = axis_angle_quaternion(rotvecs[k])
        new_quats = qmultiply(q, renderer.rot0.index_select(0, inds))
        xyz = xyz.index_copy(0, inds, points)
        quat = quat.index_copy(0, inds, new_quats)
    old_xyz, old_quat = renderer.model._xyz, renderer.model._rotation
    try:
        renderer.model._xyz = xyz
        renderer.model._rotation = quat
        result = renderer_render(renderer)
    finally:
        renderer.model._xyz = old_xyz
        renderer.model._rotation = old_quat
    return result


def renderer_render(renderer):
    from gaussian_renderer import render
    return render(renderer.camera, renderer.model, renderer.pipe, renderer.bg, 0., 0., 0.)


def rgb_from_frame(frame, size, device):
    frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(frame.copy()).to(device=device, dtype=torch.float32).permute(2, 0, 1) / 255.


def load_video(path, n_frames, width, height):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError('Video cannot be opened: ' + str(path))
    frames = []
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    try:
        for i in range(n_frames):
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA))
    finally:
        capture.release()
    if len(frames) < 2:
        raise RuntimeError('Need at least two RGB frames')
    return frames, fps


def initialize_poses(path, n_frames, n_parts, chosen, mode):
    T = np.repeat(np.eye(4, dtype=np.float64)[None, None], n_frames, axis=0)
    T = np.repeat(T, n_parts, axis=1)
    if mode == 'tracking':
        if not path or not Path(path).exists():
            raise FileNotFoundError('Please supply --tracking containing delta_g0')
        with np.load(path, allow_pickle=False) as f:
            data = f['delta_g0']
            if data.shape[1:] != (n_parts, 4, 4) or data.shape[0] < n_frames:
                raise ValueError('tracking delta_g0 shape incompatible with video')
            T[:, chosen] = data[:n_frames, chosen]
    # At the first video frame, G0 must be at identity.
    T[0] = np.eye(4)
    return T



def project_se3_points(xyz0, rv, shift, center, extrinsic, intrinsic):
    """World canonical G0 points -> image pixels through part-center rigid SE3."""
    R, trans = torch_to_matrix(rv, shift, center)
    world = xyz0 @ R.T + trans
    cam = world @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    z = cam[:, 2]
    uv = cam[:, :2] / z.clamp_min(1e-6)[:, None]
    uv = uv @ intrinsic[:2, :2].T + intrinsic[:2, 2]
    return uv, z


def se3_flow_pair_loss(gaussian_samples, angles, shifts, centers, moving,
                       ordered, pair_idx, flow, E, K, width, height,
                       max_gap):
    """Compare SE3-induced 2D displacement with composed SEA-RAFT pixel flow.

    Only motion between consecutive optimized keyframes is used; rejected
    out-of-view or invalid predictions do not contribute.
    """
    later = ordered[pair_idx]
    earlier = 0 if pair_idx == 0 else ordered[pair_idx - 1]
    if later <= earlier or later - earlier > max_gap:
        return angles.sum() * 0.0, 0
    loss_terms = []
    total = 0
    for j, pid in enumerate(moving):
        pts = gaussian_samples[j]
        if pts.shape[0] == 0:
            continue
        if pair_idx == 0:
            rv_a = torch.zeros_like(angles[pair_idx, j])
            sh_a = torch.zeros_like(shifts[pair_idx, j])
        else:
            rv_a, sh_a = angles[pair_idx-1,j], shifts[pair_idx-1,j]
        rv_b, sh_b = angles[pair_idx,j], shifts[pair_idx,j]
        uv_a, za = project_se3_points(pts, rv_a, sh_a, centers[j], E, K)
        uv_b, zb = project_se3_points(pts, rv_b, sh_b, centers[j], E, K)
        valid = (za > 1e-5) & (zb > 1e-5)
        # Flow path is evaluated at the source position; no gradient through
        # the optical-flow sampling, but SE3 projections remain differentiable.
        xy = uv_a.detach().cpu().numpy().astype(np.float32)
        valid_np = valid.detach().cpu().numpy().copy()
        valid_np &= np.isfinite(xy).all(axis=1)
        source_xy = xy.copy()
        for t in range(earlier, later):
            inside = (xy[:,0] >= 1) & (xy[:,0] < width-2) & (xy[:,1] >= 1) & (xy[:,1] < height-2)
            valid_np &= inside
            # SEA-RAFT flow field: (H,W,2), pixel displacement t -> t+1.
            field = np.asarray(flow[t], dtype=np.float32)
            displacement = cv2.remap(field, xy[:,0], xy[:,1],
                                     interpolation=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0).reshape(-1,2)
            valid_np &= np.isfinite(displacement).all(axis=1)
            xy += np.nan_to_num(displacement, nan=0.0)
        valid_np &= (xy[:,0] >= 1) & (xy[:,0] < width-2) & (xy[:,1] >= 1) & (xy[:,1] < height-2)
        valid_np &= (source_xy[:,0] >= 1) & (source_xy[:,0] < width-2)
        valid_np &= (source_xy[:,1] >= 1) & (source_xy[:,1] < height-2)
        if valid_np.sum() < 8:
            continue
        mask = torch.from_numpy(valid_np).to(device=uv_a.device)
        observed = torch.from_numpy((xy - source_xy)[valid_np].copy()).to(device=uv_a.device)
        predicted = (uv_b - uv_a)[mask]
        # Robust loss in PIXELS. flow-weight is dimensioned accordingly.
        term = F.smooth_l1_loss(predicted, observed, beta=2.0)
        loss_terms.append(term)
        total += int(valid_np.sum())
    if not loss_terms:
        return angles.sum() * 0.0, total
    return torch.stack(loss_terms).mean(), total


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--g0', required=True)
    ap.add_argument('--binding', required=True)
    ap.add_argument('--camera', required=True)
    ap.add_argument('--video', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--moving-parts', required=True, help='comma separated, e.g. 2')
    ap.add_argument('--tracking', default='')
    ap.add_argument('--init', choices=['tracking', 'identity'], default='tracking')
    ap.add_argument('--frames', type=int, default=100)
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=640)
    ap.add_argument('--frame-step', type=int, default=5,
                    help='Optimize keyframes then interpolate SE3; 1=all frames')
    ap.add_argument('--steps', type=int, default=35)
    ap.add_argument('--lr-rotation', type=float, default=0.008)
    ap.add_argument('--lr-translation', type=float, default=0.003)
    ap.add_argument('--temporal', type=float, default=0.05)
    ap.add_argument('--anchor', type=float, default=0.005)
    ap.add_argument('--silhouette', type=float, default=0.15)
    ap.add_argument('--image-scale', type=float, default=0.5)
    ap.add_argument('--preview-stride', type=int, default=10)
    ap.add_argument('--joint-steps', type=int, default=180, help='Stage B joint steps; 0 disables')
    ap.add_argument('--joint-batch', type=int, default=3)
    ap.add_argument('--joint-lr', type=float, default=0.001)
    ap.add_argument('--joint-temporal', type=float, default=0.04)
    ap.add_argument('--joint-anchor', type=float, default=0.002)
    ap.add_argument('--flow-npy', default='', help='Precomputed SEA-RAFT (T-1,H,W,2)')
    ap.add_argument('--flow-weight', type=float, default=0.002, help='Pixel flow loss coefficient in Stage B')
    ap.add_argument('--flow-samples-per-part', type=int, default=160)
    ap.add_argument('--flow-max-gap', type=int, default=5,
                    help='Maximum composed flow interval between keyframes')
    args = ap.parse_args()
    if not (0 < args.image_scale <= 1) or args.frame_step < 1 or args.steps < 1:
        ap.error('invalid --image-scale, --frame-step or --steps')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with np.load(args.binding, allow_pickle=False) as d:
        labels = d['labels'].astype(np.int32)
    moving = sorted(set(int(x) for x in args.moving_parts.split(',') if x.strip()))
    if not moving:
        ap.error('moving-parts cannot be empty')
    if any(x < 0 or x > labels.max() for x in moving):
        ap.error('moving-part out of range')
    frames, fps = load_video(args.video, args.frames, args.width, args.height)
    n = len(frames)
    renderer = GaussianPartRenderer(args.g0, labels, args.camera, args.width, args.height)
    device = renderer.xyz0.device
    renderer.model.active_sh_degree = 0
    # The upstream GaussianModel is not necessarily an nn.Module.
    # Freeze its stored leaf tensors explicitly; transformed positions/rotations
    # are new graph tensors and remain differentiable with respect to SE(3).
    for name in ("_xyz", "_rotation", "_scaling", "_opacity", "_features_dc", "_features_rest"):
        v = getattr(renderer.model, name, None)
        if isinstance(v, torch.Tensor):
            v.requires_grad_(False)
    P = renderer.parts
    initial = initialize_poses(args.tracking, n, P, moving, args.init)
    # Load dense flow only when explicitly requested; keep precomputation unchanged.
    flow = None
    E_t = K_t = None
    samples_3d = None
    if args.flow_weight > 0 and args.flow_npy:
        flow = np.load(args.flow_npy, mmap_mode='r')
        if flow.shape != (n-1, args.height, args.width, 2):
            raise ValueError(f'Expected flow {(n-1,args.height,args.width,2)}, got {flow.shape}')
        cam = renderer.camera
        E_t = cam.world_view_transform.detach().to(device=device, dtype=torch.float32).T
        fovx, fovy = float(cam.FoVx), float(cam.FoVy)
        K_t = torch.tensor([[args.width/(2*math.tan(fovx/2)),0,args.width/2],
                            [0,args.height/(2*math.tan(fovy/2)),args.height/2],
                            [0,0,1]], dtype=torch.float32, device=device)
        samples_3d = []
        for pid in moving:
            ids = renderer.ids[pid]
            count = min(len(ids), args.flow_samples_per_part)
            chosen_ids = ids[torch.linspace(0,len(ids)-1,count,device=ids.device).long()]
            samples_3d.append(renderer.xyz0.index_select(0,chosen_ids).detach())
        print(f'SEA-RAFT geometry flow supervision enabled: weight={args.flow_weight}, '
              f'samples/Part={args.flow_samples_per_part}, max_gap={args.flow_max_gap}', flush=True)

    coords0 = renderer.xyz0.detach()
    centers = torch.stack([coords0.index_select(0, renderer.ids[p]).median(dim=0).values
                           for p in moving])
    keys = sorted(set([0, n-1] + list(range(0, n, args.frame_step))))
    solved = {0: np.repeat(np.eye(4)[None], len(moving), axis=0)}
    report = []
    prev_k = 0
    previews = out / 'previews'
    previews.mkdir(exist_ok=True)
    print('RGB supervised rigid optimization:', n, 'frames; moving Parts:', moving,
          'keyframes:', len(keys), 'fps:', fps, flush=True)
    try:
        for index in keys[1:]:
            target = rgb_from_frame(frames[index], (args.width, args.height), device)
            # Initialize from nearest previous optimized pose, or CoTracker estimate.
            mats = initial[index, moving] if args.init == 'tracking' else solved[prev_k]
            # `tracking` is a warm start, never a constraint that must be obeyed.
            r0 = []
            t0 = []
            c_np = centers.detach().cpu().numpy()
            for k, m in enumerate(mats):
                rot = Rotation.from_matrix(m[:3,:3]).as_rotvec()
                r0.append(rot)
                t0.append(m[:3,3] - c_np[k] + m[:3,:3] @ c_np[k])
            rv = torch.nn.Parameter(torch.as_tensor(np.array(r0), dtype=torch.float32, device=device))
            shift = torch.nn.Parameter(torch.as_tensor(np.array(t0), dtype=torch.float32, device=device))
            previous = solved[prev_k]
            prev_centers = []
            for k, pm in enumerate(previous):
                prev_centers.append(pm[:3,:3] @ c_np[k] + pm[:3,3])
            prev_centers = torch.tensor(np.asarray(prev_centers), device=device, dtype=torch.float32)
            optimizer = torch.optim.Adam([
                {'params': [rv], 'lr': args.lr_rotation},
                {'params': [shift], 'lr': args.lr_translation},
            ])
            last_stats = {}
            for step in range(args.steps):
                optimizer.zero_grad(set_to_none=True)
                output = render_with_grad(renderer, moving, rv, shift, centers)
                prediction = output['render'].clamp(0, 1)
                gt = target
                if args.image_scale < 1:
                    hh = max(64, round(args.height * args.image_scale))
                    ww = max(64, round(args.width * args.image_scale))
                    prediction = F.interpolate(prediction[None], (hh,ww), mode='area')[0]
                    gt = F.interpolate(gt[None], (hh,ww), mode='area')[0]
                # Foreground-weighted robust reconstruction; don't let empty background dominate.
                foreground = (gt.detach().amax(0, keepdim=True) > 0.045).float()
                mask = (foreground + 0.10).detach()
                rgb = ((prediction - gt).square() + 1e-4).sqrt()
                rgb_loss = (rgb * mask).sum() / (3. * mask.sum() + 1e-6)
                silhouette_loss = prediction.amax(0).sub(foreground[0]).abs().mean()
                current_centers = centers + shift
                # A Part may rotate while its physical center stays fixed.
                temporal = (current_centers - prev_centers).square().mean()
                regularization = shift.square().mean()
                loss = (rgb_loss + args.silhouette * silhouette_loss
                        + args.temporal * temporal + args.anchor * regularization)
                if not loss.requires_grad:
                    raise RuntimeError('Renderer did not propagate gradients to SE3 parameters')
                loss.backward()
                if rv.grad is None or shift.grad is None:
                    raise RuntimeError('Missing pose gradients: verify differentiable gaussian_renderer')
                torch.nn.utils.clip_grad_norm_([rv, shift], 5.0)
                optimizer.step()
                last_stats = {'frame': index, 'step': step+1, 'loss': float(loss.detach()),
                              'rgb': float(rgb_loss.detach()), 'silhouette': float(silhouette_loss.detach())}
            result = []
            with torch.no_grad():
                for j in range(len(moving)):
                    R, t = torch_to_matrix(rv[j], shift[j], centers[j])
                    m = np.eye(4)
                    m[:3,:3] = R.cpu().numpy()
                    m[:3,3] = t.cpu().numpy()
                    result.append(m)
                if index % args.preview_stride == 0 or index == n-1:
                    output = render_with_grad(renderer, moving, rv, shift, centers)
                    rgb = (output['render'].detach().clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
                    cv2.imwrite(str(previews / f'frame_{index:04d}.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            solved[index] = np.stack(result)
            report.append(last_stats)
            prev_k = index
            print(f"RGB frame {index:03d}/{n-1}: loss={last_stats['loss']:.5f}, rgb={last_stats['rgb']:.5f}", flush=True)
            np.savez_compressed(out / 'keyframes_checkpoint.npz',
                                frame_indices=np.array(sorted(solved)),
                                delta=np.stack([solved[i] for i in sorted(solved)]))
    except KeyboardInterrupt:
        print('Interrupted; previous keyframes saved.', flush=True)
        raise
    finally:
        # make sure unoptimized G0 tensors remain unchanged
        pass
    # Stage A is immutable once Stage B begins. Save its full 100-frame output.
    from scipy.spatial.transform import Slerp
    def interpolate_solves(solution):
        result = np.repeat(np.eye(4)[None, None], n, axis=0)
        result = np.repeat(result, P, axis=1)
        indices = np.asarray(sorted(solution), dtype=float)
        for j, pid in enumerate(moving):
            mats = [solution[int(i)][j] for i in indices]
            rotation = Rotation.from_matrix(np.stack([m[:3, :3] for m in mats]))
            result[:,pid,:3,:3] = Slerp(indices, rotation)(np.arange(n)).as_matrix()
            for axis in range(3):
                result[:,pid,axis,3] = np.interp(np.arange(n), indices, [m[axis,3] for m in mats])
        return result
    def save_stage(directory, transforms):
        directory.mkdir(parents=True, exist_ok=True)
        optimized = np.zeros((n,P), dtype=bool)
        optimized[:,moving] = True
        static = ~optimized
        np.savez_compressed(directory / 'video_part_poses.npz',
            delta_g0=transforms, unknown=np.zeros((n,P),dtype=bool),
            valid=np.ones((n,P),dtype=bool), reliable=optimized,
            static_assumed=static, rgb_optimized=optimized,
            frame_indices=np.arange(n))
    save_stage(out / 'stage_a', interpolate_solves(solved))
    if args.joint_steps > 0:
        if args.joint_batch < 1:
            raise ValueError('--joint-batch must be >= 1')
        ordered = sorted(i for i in solved if i > 0)
        ca = centers.detach().cpu().numpy()
        initial_r, initial_s = [], []
        for i in ordered:
            rs, ss = [], []
            for j, mat in enumerate(solved[i]):
                rs.append(Rotation.from_matrix(mat[:3,:3]).as_rotvec())
                ss.append(mat[:3,3] - ca[j] + mat[:3,:3] @ ca[j])
            initial_r.append(rs)
            initial_s.append(ss)
        initial_r = torch.tensor(np.asarray(initial_r), dtype=torch.float32,device=device)
        initial_s = torch.tensor(np.asarray(initial_s), dtype=torch.float32,device=device)
        angles = torch.nn.Parameter(initial_r.clone())
        shifts_joint = torch.nn.Parameter(initial_s.clone())
        opt = torch.optim.Adam([{'params':[angles], 'lr':args.joint_lr},
                                {'params':[shifts_joint], 'lr':args.joint_lr}])
        weights = (0.1 + (1.0 - initial_r.norm(dim=-1).clamp(max=3.14)/3.14))
        print('Stage B: joint RGB fitting, keyframes:',len(ordered),
              'steps:',args.joint_steps,'batch:',args.joint_batch, flush=True)
        for step in range(args.joint_steps):
            opt.zero_grad(set_to_none=True)
            # Cover entire clip with deterministic staggered frame minibatches.
            indexes = [(step * args.joint_batch + j) % len(ordered)
                       for j in range(min(args.joint_batch, len(ordered)))]
            rgb_value = 0.0
            for ki in indexes:
                index = ordered[ki]
                target = rgb_from_frame(frames[index], (args.width,args.height),device)
                pred = render_with_grad(renderer,moving,angles[ki],shifts_joint[ki],centers)['render'].clamp(0,1)
                if args.image_scale < 1:
                    hh=max(64,round(args.height*args.image_scale))
                    ww=max(64,round(args.width*args.image_scale))
                    pred=F.interpolate(pred[None],(hh,ww),mode='area')[0]
                    target=F.interpolate(target[None],(hh,ww),mode='area')[0]
                foreground=(target.detach().amax(0,keepdim=True)>0.045).float()
                mask=foreground+0.10
                rgb=((pred-target).square()+1e-4).sqrt()
                rgb_loss=(rgb*mask).sum()/(3.0*mask.sum()+1e-6)
                sil=(pred.amax(0)-foreground[0]).abs().mean()
                one_loss=(rgb_loss+args.silhouette*sil)/len(indexes)
                one_loss.backward()
                rgb_value+=float(one_loss.detach())
            # Temporal terms couple all keyframe variables, including earlier ones.
            fixed_rotation = torch.zeros((1,len(moving),3), device=device)
            fixed_shift = torch.zeros_like(fixed_rotation)
            all_angles=torch.cat((fixed_rotation,angles),dim=0)
            all_shifts=torch.cat((fixed_shift,shifts_joint),dim=0)
            frame_steps=torch.as_tensor(np.diff([0]+ordered), device=device,dtype=torch.float32)
            temporal=joint_se3_regularizer(all_angles,all_shifts,frame_steps)
            anchor=((angles-initial_r).square().mean()+
                    (shifts_joint-initial_s).square().mean())
            joint_loss=args.joint_temporal*temporal+args.joint_anchor*anchor
            flow_value = 0.0
            flow_count = 0
            if flow is not None:
                # Cycling adjacent keyframe intervals. Uses same final SE3
                # parameters as the RGB rasterization, no Deform network.
                interval = step % len(ordered)
                flow_loss, flow_count = se3_flow_pair_loss(
                    samples_3d, angles, shifts_joint, centers, moving,
                    ordered, interval, flow, E_t, K_t,
                    args.width, args.height, args.flow_max_gap)
                joint_loss = joint_loss + args.flow_weight * flow_loss
                flow_value = float(flow_loss.detach())
            joint_loss.backward()
            torch.nn.utils.clip_grad_norm_([angles,shifts_joint],5.0)
            opt.step()
            if step==0 or (step+1)%20==0 or step+1==args.joint_steps:
                print(f'Joint {step+1:04d}/{args.joint_steps}: rgb={rgb_value:.5f} flow={flow_value:.5f} flow_points={flow_count} temporal={float(temporal.detach()):.6f}',flush=True)
        with torch.no_grad():
            for i, index in enumerate(ordered):
                mats=[]
                for j in range(len(moving)):
                    rot, translation=torch_to_matrix(angles[i,j],shifts_joint[i,j],centers[j])
                    mat=np.eye(4)
                    mat[:3,:3]=rot.cpu().numpy()
                    mat[:3,3]=translation.cpu().numpy()
                    mats.append(mat)
                solved[index]=np.stack(mats)
        save_stage(out / 'stage_b', interpolate_solves(solved))
        np.savez_compressed(out / 'joint_keyframes_checkpoint.npz',
            frame_indices=np.asarray(sorted(solved)),
            delta=np.stack([solved[i] for i in sorted(solved)]))
    # Interpolate independent rigid components between fitted RGB keyframes.
    transforms = np.repeat(np.eye(4)[None,None], n, axis=0)
    transforms = np.repeat(transforms, P, axis=1)
    from scipy.spatial.transform import Slerp
    for k, pid in enumerate(moving):
        kframes = np.asarray(sorted(solved), dtype=float)
        km = [solved[int(i)][k] for i in kframes]
        rotations = Rotation.from_matrix(np.stack([m[:3,:3] for m in km]))
        slerp = Slerp(kframes, rotations)
        transforms[:,pid,:3,:3] = slerp(np.arange(n)).as_matrix()
        for axis in range(3):
            transforms[:,pid,axis,3] = np.interp(np.arange(n), kframes,
                                                 [m[axis,3] for m in km])
    mask = np.zeros((n,P), dtype=bool)
    mask[:,moving] = True
    assumptions = np.zeros((n,P), dtype=bool)
    assumptions[:,[p for p in range(P) if p not in moving]] = True
    # Deliberate preview semantics: only selected Parts optimized, others static by assumption.
    np.savez_compressed(out / 'video_part_poses.npz',
                        delta_g0=transforms, unknown=np.zeros((n,P),dtype=bool),
                        valid=np.ones((n,P),dtype=bool),
                        reliable=mask, static_assumed=assumptions,
                        rgb_optimized=mask,
                        frame_indices=np.arange(n))
    info = {'method':'differentiable_rgb_rigid_se3', 'moving_parts':moving,
            'other_parts':'static_assumed_not_observed', 'frames':n,'fps':fps,
            'keyframes':keys, 'steps_per_keyframe':args.steps, 'joint_steps':args.joint_steps, 'joint_batch':args.joint_batch,
            'note':'PnP used only to initialize; scores on a single fixed monocular viewpoint do not prove true 3D pose',
            'per_keyframe':report}
    (out / 'rgb_refinement_report.json').write_text(json.dumps(info,indent=2))
    print('Saved:', out / 'video_part_poses.npz')
    print('Next: export_video_part_motion.py using this --tracking file')


if __name__ == '__main__':
    main()
