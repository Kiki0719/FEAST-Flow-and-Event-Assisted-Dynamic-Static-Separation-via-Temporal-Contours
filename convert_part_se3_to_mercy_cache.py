#!/usr/bin/env python3
"""Convert G0 + Gaussian Part labels + relative Part SE(3) to MERCY base motion cache.

Produces:
  poses.npy: (frames, dynamic_gaussians, 7) float32, [world XYZ, quaternion WXYZ]
  dynamic_indices.npy: (dynamic_gaussians,) int32, zero-based rows of the input G0 PLY
  metadata.json: provenance, coordinate conventions, validation notes

The part transformations must be canonical-world -> frame-world matrices, with
frame zero identity. Base Gaussian quaternion attributes rot_0..3 must be WXYZ
(the standard 3DGS convention). No camera transform is applied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(block_size), b''):
            h.update(block)
    return h.hexdigest()


def normalize_quat(q: np.ndarray, what: str) -> np.ndarray:
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if not np.isfinite(norm).all() or np.any(norm < 1e-8):
        raise ValueError(f'{what}: nonfinite or zero quaternion')
    return q / norm


def multiply_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # Broadcasting: a [4], b [N,4]. Both are [w,x,y,z].
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack((aw*bw-ax*bx-ay*by-az*bz,
                     aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx,
                     aw*bz+ax*by-ay*bx+az*bw), axis=-1)


def read_gaussians(ply_path: Path, indices: np.ndarray, expected_count: int):
    try:
        from plyfile import PlyData
    except ImportError as exc:
        raise RuntimeError('Install plyfile in the current environment: pip install plyfile') from exc
    ply = PlyData.read(str(ply_path), mmap=True)
    if 'vertex' not in ply:
        raise ValueError('PLY does not contain a vertex element')
    vertex = ply['vertex'].data
    if len(vertex) != expected_count:
        raise ValueError(f'PLY Gaussian count {len(vertex)} != binding labels {expected_count}; refusing mismatched row order/count')
    names = set(vertex.dtype.names)
    required = ('x', 'y', 'z', 'rot_0', 'rot_1', 'rot_2', 'rot_3')
    missing = [k for k in required if k not in names]
    if missing:
        raise ValueError(f'PLY is missing Gaussian XYZ/rotation fields: {missing}')
    xyz = np.column_stack([np.asarray(vertex[k][indices], dtype=np.float64)
                           for k in ('x','y','z')])
    quat = np.column_stack([np.asarray(vertex[k][indices], dtype=np.float64)
                            for k in ('rot_0','rot_1','rot_2','rot_3')])
    if not np.isfinite(xyz).all():
        raise ValueError('Selected G0 Gaussian coordinates contain nonfinite values')
    quat = normalize_quat(quat, 'G0 rotation')
    return xyz, quat


def validate_transforms(mats: np.ndarray):
    if mats.ndim != 4 or mats.shape[-2:] != (4,4):
        raise ValueError(f'delta_g0 must have shape [T,P,4,4], got {mats.shape}')
    if not np.isfinite(mats).all():
        raise ValueError('delta_g0 contains nonfinite entries')
    eye_row = np.array([0,0,0,1], dtype=np.float64)
    if not np.allclose(mats[:,: ,3,:], eye_row, atol=1e-4):
        raise ValueError('Part transforms have invalid homogeneous last row')
    R = mats[:,:,:3,:3]
    ortho = np.matmul(np.swapaxes(R,-1,-2), R)
    if np.max(np.abs(ortho-np.eye(3))) > 2e-3:
        raise ValueError('Part transforms contain nonrigid rotation matrices')
    if np.max(np.abs(np.linalg.det(R)-1)) > 2e-3:
        raise ValueError('Part transforms contain reflections/nonrigid rotations')
    if not np.allclose(mats[0], np.eye(4), atol=1e-4):
        raise ValueError('First-frame transforms are not identity. Expected canonical-world relative SE(3).')


def create_full_rigid_binding(source: Path, confirmation: Path, target: Path, overwrite: bool) -> Path:
    """Expand verified moving Parts into complete rigid Gaussian groups.

    Preserve every original binding field, retain Stage C's sparse
    dynamic_indices as candidate_dynamic_indices, and never alter G0.
    """
    if source.resolve() == target.resolve():
        raise ValueError('Full-rigid binding output must differ from original binding')
    if target.exists() and not overwrite:
        raise FileExistsError(f'Full-rigid binding exists: {target}. Pass --overwrite to replace.')
    with np.load(source, allow_pickle=False) as z:
        original = {key: z[key].copy() for key in z.files}
    if 'labels' not in original or 'dynamic_indices' not in original:
        raise KeyError('Source binding must contain labels and dynamic_indices')
    labels = np.asarray(original['labels'], dtype=np.int32)
    previous_dynamic = np.asarray(original['dynamic_indices'], dtype=np.int64)
    if previous_dynamic.ndim != 1 or np.any(previous_dynamic < 0) or np.any(previous_dynamic >= len(labels)):
        raise ValueError('Source dynamic_indices are invalid')
    report = json.loads(confirmation.read_text(encoding='utf-8'))
    confirmed = sorted(set(int(p) for p in report['moving_parts']))
    if not confirmed:
        raise ValueError('Stage B has no confirmed moving Parts')
    if any(p < 0 or p > int(labels.max()) for p in confirmed):
        raise ValueError(f'Confirmed Part ID outside source binding: {confirmed}')
    if not set(np.unique(labels[previous_dynamic]).tolist()).issubset(set(confirmed)):
        raise ValueError('Existing dynamic indices include Gaussian Parts not confirmed by Stage B')
    full_dynamic = np.flatnonzero(np.isin(labels, confirmed)).astype(np.int32)
    if not len(full_dynamic):
        raise ValueError('No Gaussians belong to confirmed moving Parts')
    original['candidate_dynamic_indices'] = previous_dynamic.astype(np.int32)
    original['dynamic_indices'] = full_dynamic
    original['motion_source'] = np.asarray('confirmed_part_full_rigid_se3')
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **original)
    print('========== FULL RIGID BINDING ==========', flush=True)
    print(f'Confirmed Parts: {confirmed}', flush=True)
    print(f'Original dynamic Gaussians: {len(previous_dynamic):,}', flush=True)
    print(f'Full rigid dynamic Gaussians: {len(full_dynamic):,}', flush=True)
    print(f'Additional Gaussians: {len(full_dynamic) - len(previous_dynamic):,}', flush=True)
    print(f'Original labels preserved: {np.array_equal(labels, original["labels"])}', flush=True)
    print(f'Full rigid binding saved: {target}', flush=True)
    return target


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--g0', type=Path, required=True, help='Canonical corrected_g0.ply; G0 row order must match binding')
    ap.add_argument('--binding', type=Path, required=True, help='Source Stage C binding or previously generated full-rigid binding')
    ap.add_argument('--confirmation', type=Path, default=None,
                    help='Stage B motion_confirmation.json; when supplied, auto-create a full-rigid binding')
    ap.add_argument('--full-rigid-binding-output', type=Path, default=None,
                    help='Optional path for newly created full-rigid binding, default beside the input binding')
    ap.add_argument('--part-poses', type=Path, required=True, help='Stage D video_part_poses.npz containing delta_g0')
    ap.add_argument('--output', type=Path, required=True, help='New output directory')
    ap.add_argument('--chunk-size', type=int, default=50000, help='Gaussians per batch, default 50000')
    ap.add_argument('--overwrite', action='store_true', help='Allow replacing existing output files')
    args = ap.parse_args()
    if args.chunk_size <= 0:
        ap.error('--chunk-size must be positive')
    for p in (args.g0, args.binding, args.part_poses):
        if not p.is_file():
            raise FileNotFoundError(p)
    if args.full_rigid_binding_output is not None and args.confirmation is None:
        ap.error('--full-rigid-binding-output requires --confirmation')
    # Validate all output targets before mutating anything.
    out = args.output
    poses_path = out / 'poses.npy'
    indices_path = out / 'dynamic_indices.npy'
    meta_path = out / 'metadata.json'
    if not args.overwrite and any(p.exists() for p in (poses_path, indices_path, meta_path)):
        raise FileExistsError(f'One or more cache files exist under {out}; choose a new directory or pass --overwrite')
    if args.confirmation is not None:
        if not args.confirmation.is_file():
            raise FileNotFoundError(args.confirmation)
        full_path = (args.full_rigid_binding_output or
                     args.binding.with_name('final_binding_full_rigid.npz'))
        args.binding = create_full_rigid_binding(
            args.binding, args.confirmation, full_path, args.overwrite
        )
    with np.load(args.binding, allow_pickle=False) as binding:
        if 'labels' not in binding or 'dynamic_indices' not in binding:
            raise KeyError('Binding must contain labels and dynamic_indices')
        labels = binding['labels'].astype(np.int32)
        dynamic = np.asarray(binding['dynamic_indices'], dtype=np.int64)
    if dynamic.ndim != 1 or not len(dynamic):
        raise ValueError('dynamic_indices must be a nonempty 1D array')
    if np.any(dynamic < 0) or np.any(dynamic >= len(labels)):
        raise ValueError('dynamic_indices out of bounds for binding')
    if len(np.unique(dynamic)) != len(dynamic):
        raise ValueError('Duplicate dynamic Gaussian indices')
    # Sort to guarantee a deterministic G0-row ordered output.
    dynamic = np.sort(dynamic)
    if np.any(labels[dynamic] < 0):
        raise ValueError('Unassigned Part in selected dynamic Gaussian indices')
    with np.load(args.part_poses, allow_pickle=False) as part_file:
        if 'delta_g0' not in part_file:
            raise KeyError('Part pose NPZ must contain delta_g0')
        part_T = np.asarray(part_file['delta_g0'], dtype=np.float64)
    validate_transforms(part_T)
    frames, parts = part_T.shape[:2]
    if labels.max() >= parts:
        raise ValueError('Binding references a Part not present in pose array')
    part_ids = labels[dynamic]
    unique_parts = np.unique(part_ids)
    print(f'Input Gaussians: {len(labels):,}, dynamic: {len(dynamic):,}, frames: {frames}, Parts: {unique_parts.tolist()}', flush=True)
    g0_xyz, g0_quat = read_gaussians(args.g0, dynamic, len(labels))
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    poses_path = out / 'poses.npy'
    indices_path = out / 'dynamic_indices.npy'
    meta_path = out / 'metadata.json'
    if not args.overwrite and any(p.exists() for p in (poses_path, indices_path, meta_path)):
        raise FileExistsError(f'One or more output files exist under {out}; choose a new directory or pass --overwrite')
    if len(dynamic) > np.iinfo(np.int32).max:
        raise ValueError('Indices exceed int32 range')
    np.save(indices_path, dynamic.astype(np.int32))
    # Disk-backed array so the full T*N*7 cache never needs to fit in RAM.
    cache = np.lib.format.open_memmap(poses_path, mode='w+', dtype=np.float32,
                                      shape=(frames, len(dynamic), 7))
    for pid in unique_parts:
        ids = np.flatnonzero(part_ids == pid)
        print(f'Part {pid}: {len(ids):,} dynamic Gaussians', flush=True)
        for start in range(0, len(ids), args.chunk_size):
            selected = ids[start:start+args.chunk_size]
            X0 = g0_xyz[selected]
            q0 = g0_quat[selected]
            for t in range(frames):
                mat = part_T[t, pid]
                R = mat[:3, :3]
                trans = mat[:3, 3]
                pos = X0 @ R.T + trans
                quat_xyzw = Rotation.from_matrix(R).as_quat()
                rot_q = quat_xyzw[[3,0,1,2]]
                q = multiply_wxyz(rot_q, q0)
                # Renormalize; preserve quaternion continuity with canonical G0.
                q = normalize_quat(q, 'transformed Gaussian rotation')
                flipped = np.sum(q*q0, axis=-1) < 0
                q[flipped] *= -1
                cache[t, selected, :3] = pos.astype(np.float32)
                cache[t, selected, 3:] = q.astype(np.float32)
    cache.flush()
    # First frame invariance and normalisation checks.
    first_xyz_error = float(np.max(np.abs(cache[0,:,:3].astype(np.float64)-g0_xyz)))
    first_quat_error = float(np.max(np.abs(cache[0,:,3:].astype(np.float64)-g0_quat)))
    if first_xyz_error > 1e-4 or first_quat_error > 1e-4:
        raise RuntimeError(f'First-frame canonical-state validation failed: xyz={first_xyz_error}, quat={first_quat_error}')
    if not np.isfinite(cache).all():
        raise RuntimeError('Output cache contains nonfinite values')
    meta = {
        'format': 'MERCY_base_motion_cache_part_se3_v1',
        'poses_shape': [int(frames), int(len(dynamic)), 7],
        'poses_dtype': 'float32',
        'poses_axes': 'frames,dynamic_gaussians,components',
        'pose_components': 'world_xyz_and_quaternion_wxyz',
        'dynamic_indices': 'zero_based_original_G0_PLY_rows_sorted_ascending',
        'SE3_convention': 'delta_g0[t,p] transforms canonical G0 world coordinates into frame t world coordinates; x_t=R*x_0+t',
        'gaussian_rotation_convention': 'q_t=q_part_wxyz*q_g0_wxyz, standard 3DGS rot_0..3',
        'not_included': ['rendered RGB cache', 'camera metadata', 'event supervision', 'joint report'],
        'frames': int(frames), 'g0_gaussians': int(len(labels)),
        'dynamic_gaussians': int(len(dynamic)), 'moving_part_ids': unique_parts.astype(int).tolist(),
        'first_frame_max_xyz_error': first_xyz_error,
        'first_frame_max_quaternion_component_error': first_quat_error,
        'sources': {str(name): {'path': str(path.resolve()), 'sha256': sha256(path)} for name, path in
                    [('g0', args.g0), ('binding', args.binding), ('part_poses', args.part_poses)]},
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote: {poses_path} [{frames}, {len(dynamic)}, 7]', flush=True)
    print(f'Wrote: {indices_path} [{len(dynamic)}]', flush=True)
    print(f'Wrote: {meta_path}', flush=True)
    print('Note: This creates array-format-compatible motion data, NOT the full baseline renderer/metadata bundle.', flush=True)


if __name__ == '__main__':
    main()
