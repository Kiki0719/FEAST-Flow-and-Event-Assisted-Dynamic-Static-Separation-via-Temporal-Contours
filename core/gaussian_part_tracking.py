
"""Existing 3DGS + fixed-camera RGB: per-Part tracking and 2D/3D PnP.

All transformations returned here are G0-world relative to the first video frame.
Needs the SC-GS CUDA renderer from MonoEvent-RT3D-DMR; no mesh reconstruction.
"""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from gaussian_renderer import render
from FEAST.core.mercy_camera_adapter import make_camera
from scene.gaussian_model import GaussianModel
from FEAST.utils.ply_utils import gaussian_degree
from FEAST.core.group_query_sampling import sample_group_grid, sample_visible_regular_grid


SH_C0 = 0.28209479177387814



def _colors(parts):
    """Generate distinct RGB colors for arbitrary Part counts."""
    import numpy as np

    n = int(parts)
    if n < 1:
        raise ValueError("Part count must be positive")

    # 16 levels per channel -> 4096 unique RGB codes
    levels = np.linspace(0.08, 0.92, 16, dtype=np.float32)

    ids = np.arange(n, dtype=np.int64)

    if n > 4096:
        raise ValueError("More than 4096 Parts")

    # Spread nearby Part IDs across the palette
    codes = (ids * 2053) % 4096

    r = codes // 256
    g = (codes // 16) % 16
    b = codes % 16

    return np.stack(
        [levels[r], levels[g], levels[b]],
        axis=1
    ).astype(np.float32)


def _quat_product(q, r):
    # Both are wxyz quaternions. q: (4,), r: (N,4).
    w, x, y, z = q
    a, b, c, d = r.unbind(-1)
    return torch.stack((
        w*a-x*b-y*c-z*d,
        w*b+x*a+y*d-z*c,
        w*c-x*d+y*a+z*b,
        w*d+x*c-y*b+z*a
    ), dim=1)


class GaussianPartRenderer:
    def __init__(self, ply, labels, camera_file,
                 width=640, height=640, min_alpha=0.80):
        if not torch.cuda.is_available():
            raise RuntimeError('Gaussian rendering requires CUDA')

        self.width, self.height = int(width), int(height)
        self.min_alpha = float(min_alpha)
        if not (0 < self.min_alpha < 1):
            raise ValueError('min_alpha must lie between 0 and 1')

        meta = json.loads(Path(camera_file).read_text())['camera']
        self.pose_gl = np.asarray(meta['pose'], dtype=np.float64)

        if self.pose_gl.shape != (4, 4):
            raise ValueError(
                'camera.pose must be a 4x4 OpenGL camera-to-world matrix'
            )

        fovy = math.radians(float(meta['fovy_degrees']))
        if not (0 < fovy < np.pi):
            raise ValueError('Invalid camera.fovy_degrees')

        fy = self.height / (2.0 * math.tan(fovy / 2.0))
        self.K = np.asarray([
            [fy, 0, self.width / 2.0],
            [0, fy, self.height / 2.0],
            [0, 0, 1]
        ], dtype=np.float64)

        self.dist = np.zeros((5, 1), dtype=np.float64)
        self.c2w_cv = self.pose_gl @ np.diag([1., -1., -1., 1.])
        self.w2c_cv = np.linalg.inv(self.c2w_cv)

        self.camera = make_camera(
            self.pose_gl, self.width, self.height, fovy
        )
        self.pipe = SimpleNamespace(
            convert_SHs_python=False,
            compute_cov3D_python=False,
            debug=False
        )
        self.bg = torch.zeros(3, device='cuda')

        self.model = GaussianModel(
            gaussian_degree(Path(ply)),
            with_motion_mask=False
        )
        self.model.load_ply(str(ply))

        self.xyz0 = self.model._xyz.detach().clone()
        self.rot0 = self.model._rotation.detach().clone()
        self.dc0 = self.model._features_dc.detach().clone()

        labels = np.asarray(labels, dtype=np.int32)
        if len(labels) != len(self.xyz0) or np.min(labels) < 0:
            raise ValueError(
                'Binding labels do not match G0 Gaussian positions'
            )

        self.parts = int(labels.max()) + 1
        self.labels = labels
        self.palette = _colors(self.parts)

        self.ids = [
            torch.as_tensor(
                np.flatnonzero(labels == p),
                dtype=torch.long, device='cuda'
            )
            for p in range(self.parts)
        ]

        # Empty Parts retain their original IDs but have no Gaussians.
        self.empty_parts = [
            p for p, index in enumerate(self.ids)
            if len(index) == 0
        ]
        self.active_parts = [
            p for p, index in enumerate(self.ids)
            if len(index) > 0
        ]
        print(
            f"Gaussian Parts: {len(self.active_parts)} active, "
            f"{len(self.empty_parts)} empty"
        )

        per_gaussian_rgb = self.palette[labels]
        self.dc_label = torch.from_numpy(
            ((per_gaussian_rgb - 0.5) / SH_C0)[:, None, :]
        ).to(device='cuda', dtype=self.dc0.dtype)

        self.model.active_sh_degree = 0

    def unproject(self, uv, z):
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        z = np.asarray(z, dtype=np.float64).reshape(-1)

        cam = np.column_stack((
            (uv[:, 0]-self.K[0, 2])/self.K[0, 0]*z,
            (uv[:, 1]-self.K[1, 2])/self.K[1, 1]*z,
            z
        ))
        return (
            cam @ self.c2w_cv[:3, :3].T
            + self.c2w_cv[:3, 3]
        )

    @torch.no_grad()
    def render_rgb(self, delta):
        """Full-scene BGR render for occlusion motion-hypothesis testing.

        Uses the exact regular 3DGS renderer and restores original G0 tensors;
        unlike snapshot(), this does not render a second color-coded ID pass.
        """
        delta = np.asarray(delta, dtype=np.float64)
        if delta.shape != (self.parts, 4, 4):
            raise ValueError('delta shape must be (P,4,4)')
        self.model._xyz.data.copy_(self.xyz0)
        self.model._rotation.data.copy_(self.rot0)
        try:
            for part, indices in enumerate(self.ids):
                if len(indices) == 0 or np.allclose(delta[part], np.eye(4), atol=1e-9):
                    continue
                matrix = delta[part]
                r = torch.as_tensor(matrix[:3, :3], device='cuda', dtype=torch.float32)
                t = torch.as_tensor(matrix[:3, 3], device='cuda', dtype=torch.float32)
                self.model._xyz.data[indices] = self.xyz0[indices] @ r.T + t
                q = torch.as_tensor(
                    np.roll(Rotation.from_matrix(matrix[:3, :3]).as_quat(), 1),
                    device='cuda', dtype=torch.float32
                )
                self.model._rotation.data[indices] = _quat_product(q, self.rot0[indices])
            out = render(self.camera, self.model, self.pipe, self.bg, 0., 0., 0.)
            rgb = out['render'].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            return cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        finally:
            self.model._xyz.data.copy_(self.xyz0)
            self.model._rotation.data.copy_(self.rot0)

    @torch.no_grad()
    def snapshot(self, delta):
        """Return BGR uint8, z-depth, Part-ID int32, alpha as numpy arrays."""
        delta = np.asarray(delta, dtype=np.float64)
        if delta.shape != (self.parts, 4, 4):
            raise ValueError('delta shape must be (P,4,4)')

        self.model._xyz.data.copy_(self.xyz0)
        self.model._rotation.data.copy_(self.rot0)

        for p, index in enumerate(self.ids):
            if not len(index):
                continue

            matrix = delta[p]
            if not np.allclose(matrix, np.eye(4), atol=1e-9):
                R = torch.as_tensor(
                    matrix[:3, :3],
                    dtype=torch.float32, device='cuda'
                )
                t = torch.as_tensor(
                    matrix[:3, 3],
                    dtype=torch.float32, device='cuda'
                )

                self.model._xyz.data[index] = (
                    self.xyz0[index] @ R.T + t
                )

                q = torch.as_tensor(
                    np.roll(
                        Rotation.from_matrix(
                            matrix[:3, :3]
                        ).as_quat(), 1
                    ),
                    dtype=torch.float32, device='cuda'
                )

                self.model._rotation.data[index] = (
                    _quat_product(q, self.rot0[index])
                )

        try:
            result = render(
                self.camera, self.model, self.pipe,
                self.bg, 0., 0., 0.
            )

            rgb = (
                result['render']
                .clamp(0, 1)
                .permute(1, 2, 0)
                .cpu().numpy()
            )
            depth = (
                result['depth']
                .squeeze().detach().cpu().numpy()
            )
            alpha = (
                result['alpha']
                .squeeze().detach().cpu().numpy()
            )

            # The repository ignores override_color;
            # use SH DC to encode Part IDs.
            self.model._features_dc.data.copy_(self.dc_label)

            result_id = render(
                self.camera, self.model, self.pipe,
                self.bg, 0., 0., 0.
            )
            coded = (
                result_id['render']
                .permute(1, 2, 0)
                .cpu().numpy()
            )

        finally:
            self.model._features_dc.data.copy_(self.dc0)
            self.model._xyz.data.copy_(self.xyz0)
            self.model._rotation.data.copy_(self.rot0)

        valid = (
            (alpha > self.min_alpha)
            & np.isfinite(depth)
            & (depth > 0)
        )

        unblended = coded / np.maximum(alpha[..., None], 1e-4)
        distances = np.sum(
            (
                unblended[:, :, None, :]
                - self.palette[None, None, :, :]
            ) ** 2,
            axis=3
        )

        ids = np.argmin(distances, axis=2).astype(np.int32)
        residual = np.min(distances, axis=2)
        ids[~valid | (residual > 0.035)] = -1

        bgr = cv2.cvtColor(
            (rgb * 255).astype(np.uint8),
            cv2.COLOR_RGB2BGR
        )

        return (
            bgr,
            depth.astype(np.float32),
            ids,
            alpha.astype(np.float32)
        )


class PartMotionTracker:
    def __init__(self, renderer, min_tracks=10,
                 max_corners=240, reprojection_px=4.0,
                 reanchor_every=12, debug_dir=None,
                 cotracker_sampling="corners",
                 adaptive_queries=False, occlusion_depth_margin=0.015, grid_stride=8):
        self.r = renderer
        self.min_tracks = int(min_tracks)
        self.max_corners = int(max_corners)
        self.grid_stride = int(grid_stride)
        if self.grid_stride < 1: raise ValueError("grid_stride must be positive")
        self.adaptive_queries = bool(adaptive_queries)
        self.occlusion_depth_margin = float(occlusion_depth_margin)
        self.query_budget = np.array([min(self.max_corners, max(self.min_tracks, round(48.0 * np.sqrt(len(index)/1000.0)))) if len(index) else 0 for index in getattr(self.r, "ids", [list(range(self.max_corners)) for _ in range(self.r.parts)])], dtype=int)
        self.occluded_counts = np.zeros(self.r.parts, dtype=np.int64)
        self.visible_candidate_counts = np.zeros(self.r.parts, dtype=np.int64)
        self.cross_part_rejected = np.zeros(self.r.parts, dtype=np.int64)
        self.pnp_outlier_counts = np.zeros(self.r.parts, dtype=np.int64)
        self.motion_model_support = [[] for _ in range(self.r.parts)]
        self.point_status = [None] * self.r.parts  # status for the most recent frame
        self.reprojection_px = float(reprojection_px)
        self.reanchor_every = int(reanchor_every)
        self.debug_dir = Path(debug_dir) if debug_dir else None
        if cotracker_sampling not in ("group-grid", "corners"):
            raise ValueError("Unknown CoTracker query sampling mode")
        self.cotracker_sampling = cotracker_sampling

        self.initial_seed_xyz = [None] * self.r.parts
        self.initial_seed_uv = [None] * self.r.parts
        self.prev_gray = None
        self.points = [None] * self.r.parts
        self.xyz = [None] * self.r.parts

        self.delta = np.tile(
            np.eye(4), (self.r.parts, 1, 1)
        ).astype(np.float64)

        self.frame_index = 0
        self.last_good = np.zeros(self.r.parts, dtype=np.int32)
        self.first_frame_rgb_mae = None

    def _seed(self, frame_bgr, gray, depth, part_ids, delta, p):
        mask = np.uint8(part_ids == p) * 255
        # Preserve features on small rigid Parts.
        mask = cv2.erode(
            mask, np.ones((3, 3), np.uint8)
        )

        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_corners,
            qualityLevel=0.012,
            minDistance=5,
            mask=mask,
            blockSize=5
        )

        if corners is None:
            return

        uv = corners.reshape(-1, 2)
        xy = np.round(uv).astype(np.int32)

        xy[:, 0] = np.clip(
            xy[:, 0], 0, self.r.width - 1
        )
        xy[:, 1] = np.clip(
            xy[:, 1], 0, self.r.height - 1
        )

        z = depth[xy[:, 1], xy[:, 0]]
        keep = np.isfinite(z) & (z > 0)
        uv, z = uv[keep], z[keep]

        if len(uv) < self.min_tracks:
            return

        world_now = self.r.unproject(uv, z)
        inverse = np.linalg.inv(delta[p])

        world0 = (
            world_now @ inverse[:3, :3].T
            + inverse[:3, 3]
        )

        self.points[p] = (
            uv.astype(np.float32).reshape(-1, 1, 2)
        )
        self.xyz[p] = np.asarray(
            world0, dtype=np.float32
        )

    def start(self, bgr):
        gray = cv2.cvtColor(
            bgr, cv2.COLOR_BGR2GRAY
        )

        rendered, depth, ids, _ = (
            self.r.snapshot(self.delta)
        )

        self.first_frame_rgb_mae = float(
            np.mean(
                np.abs(
                    rendered.astype(np.float32)
                    - bgr.astype(np.float32)
                )
            ) / 255
        )

        for p in range(self.r.parts):
            if self.cotracker_sampling == 'group-grid':
                self._seed_cotracker_masked(
                    bgr, gray, depth, ids, self.delta, p,
                    np.ones_like(ids, dtype=bool)
                )
            else:
                self._seed(bgr, gray, depth, ids, self.delta, p)

        # This is the immutable pool of initially visible query identities.
        # A subsequently visible previously hidden Gaussian is NEVER seeded.
        for p in range(self.r.parts):
            if self.xyz[p] is not None:
                self.initial_seed_xyz[p] = self.xyz[p].copy()
                self.initial_seed_uv[p] = self.points[p].copy()

        if self.debug_dir:
            self.debug_dir.mkdir(
                parents=True, exist_ok=True
            )

            cv2.imwrite(
                str(self.debug_dir / 'g0_render.png'),
                rendered
            )
            cv2.imwrite(
                str(self.debug_dir / 'first_video.png'),
                bgr
            )
            cv2.imwrite(
                str(
                    self.debug_dir
                    / 'first_frame_overlay.png'
                ),
                cv2.addWeighted(
                    rendered, 0.5, bgr, 0.5, 0
                )
            )

            display = np.zeros_like(bgr)
            for p, color in enumerate(self.r.palette):
                display[ids == p] = (
                    color[::-1] * 255
                ).astype(np.uint8)

            cv2.imwrite(
                str(self.debug_dir / 'part_ids.png'),
                display
            )

        self.prev_gray = gray
        self.initial_query_counts = [
            0 if x is None else len(x) for x in self.points
        ]
        if self.debug_dir and self.cotracker_sampling == 'group-grid':
            # Diagnostic proves each query was selected from a Group mask.
            groups, coords = [], []
            for group_id, points in enumerate(self.points):
                if points is not None:
                    coords.extend(points.reshape(-1, 2).tolist())
                    groups.extend([group_id] * len(points))
            np.savez_compressed(
                self.debug_dir / 'initial_group_queries.npz',
                group_id=np.asarray(groups, dtype=np.int32),
                xy=np.asarray(coords, dtype=np.float32).reshape(-1, 2),
                count_per_group=np.asarray(self.initial_query_counts, dtype=np.int32)
            )

        valid = np.array(
            [x is not None for x in self.points],
            bool
        )
        counts = np.array(
            [
                0 if x is None else len(x)
                for x in self.points
            ],
            np.int32
        )

        # Frame 0 identity is the pose reference.
        return (
            self.delta.copy(),
            valid,
            counts,
            np.zeros(self.r.parts, np.float32)
        )

    def reanchor_cotracker(self, bgr, max_stale_frames=3,
                           appearance_tolerance=0.26):
        """Re-anchor ONLY original frame-zero visible 3D points.

        Never add initially occluded back-layer points even if they later
        appear in a predicted render. Occlusion at current anchor also
        suppresses queries, but never implies motion.
        """
        _, depth, front_ids, _ = self.r.snapshot(self.delta)
        w2c = getattr(self.r, 'w2c_cv', np.linalg.inv(self.r.c2w_cv))
        for p in range(self.r.parts):
            self.points[p] = None
            self.xyz[p] = None
            pts = self.initial_seed_xyz[p]
            if pts is None or self.frame_index-self.last_good[p]>max_stale_frames:
                continue
            now=pts @ self.delta[p,:3,:3].T+self.delta[p,:3,3]
            cam=now @ w2c[:3,:3].T+w2c[:3,3]
            zz=cam[:,2]
            ok=np.isfinite(cam).all(axis=1)&(zz>1e-6)
            uv=np.zeros((len(pts),2),np.float64)
            uv[ok,0]=self.r.K[0,0]*cam[ok,0]/zz[ok]+self.r.K[0,2]
            uv[ok,1]=self.r.K[1,1]*cam[ok,1]/zz[ok]+self.r.K[1,2]
            u=np.rint(uv[:,0]).astype(int);v=np.rint(uv[:,1]).astype(int)
            ok &= (u>=1)&(u<self.r.width-1)&(v>=1)&(v<self.r.height-1)
            inds=np.flatnonzero(ok)
            if len(inds):
                uu=u[inds];vv=v[inds]
                fg=depth[vv,uu]; fid=front_ids[vv,uu]
                margin=self.occlusion_depth_margin*np.maximum(1.,zz[inds])
                blocked=(fid>=0)&(fid!=p)&np.isfinite(fg)&(fg>0)&(fg+margin<zz[inds])
                ok[inds[blocked]]=False
            if ok.sum() < self.min_tracks: continue
            self.points[p]=uv[ok].astype(np.float32).reshape(-1,1,2)
            self.xyz[p]=np.asarray(pts[ok],np.float32)

    def _seed_cotracker_masked(self, bgr, gray, depth, ids, delta, p, mask):
        # Rigid Group labels come from the G0/Blender binding and are
        # attached to each query BEFORE CoTracker runs. CoTracker itself
        # predicts (x,y)/visibility, not a Group-constrained rigid motion.
        if self.cotracker_sampling == 'group-grid':
            uv = sample_visible_regular_grid(
                ids, depth, p, stride=self.grid_stride,
                min_points=self.min_tracks, appearance_mask=mask
            )
            if len(uv) < self.min_tracks:
                return
        else:
            region = cv2.erode(
                np.uint8((ids == p) & mask) * 255,
                np.ones((3, 3), np.uint8)
            )
            corners = cv2.goodFeaturesToTrack(
                gray, maxCorners=int(self.query_budget[p]) if self.adaptive_queries else self.max_corners, qualityLevel=0.012,
                minDistance=5, mask=region, blockSize=5
            )
            if corners is None:
                return
            uv = corners.reshape(-1, 2)
        ix = np.rint(uv[:, 0]).astype(int).clip(0, self.r.width - 1)
        iy = np.rint(uv[:, 1]).astype(int).clip(0, self.r.height - 1)
        z = depth[iy, ix]
        good = np.isfinite(z) & (z > 0)
        uv, z = uv[good], z[good]
        if len(uv) < self.min_tracks:
            return
        world_now = self.r.unproject(uv, z)
        inv = np.linalg.inv(delta[p])
        world0 = world_now @ inv[:3, :3].T + inv[:3, 3]
        self.points[p] = uv.astype(np.float32).reshape(-1, 1, 2)
        self.xyz[p] = np.asarray(world0, dtype=np.float32)

    def step_cotracker(self, xy_by_part, visibility_by_part, frame_index,
                       visibility_threshold=0.5):
        """Measure per-Part 6DoF from CoTracker's (x,y,visibility).

        The 3D query identity is never pruned across a CoTracker window:
        masking the inliers for one frame must not reorder later queries.
        """
        self.frame_index = int(frame_index)
        P = self.r.parts
        measured = np.zeros(P, dtype=bool)
        counts = np.zeros(P, dtype=np.int32)
        errors = np.full(P, np.inf, dtype=np.float32)
        # Depth-aware occlusion check uses LAST accepted poses to render the
        # currently predicted frontmost surface. This is conservative: only
        # definite foreground occlusions suppress measurements. Never infer
        # motion from a missing observation.
        visible_ids = visible_depth = None
        try:
            if hasattr(self.r, "snapshot"):
                _, visible_depth, visible_ids, _ = self.r.snapshot(self.delta)
        except Exception:
            # Do not silently declare points visible if renderer is broken.
            raise RuntimeError('Dynamic depth/Part occlusion rendering failed')
        self.point_status = [None] * P
        for part in range(P):
            xyz = self.xyz[part]
            if xyz is None:
                continue
            xy = np.asarray(xy_by_part.get(part, []), dtype=np.float64)
            visibility = np.asarray(
                visibility_by_part.get(part, []), dtype=np.float64
            ).reshape(-1)
            if xy.shape != (len(xyz), 2) or len(visibility) != len(xyz):
                raise ValueError(f'CoTracker points for Part {part} misaligned')
            keep = (
                np.isfinite(xy).all(axis=1)
                & np.isfinite(visibility)
                & (visibility >= visibility_threshold)
                & (xy[:, 0] >= 2)
                & (xy[:, 0] < self.r.width - 2)
                & (xy[:, 1] >= 2)
                & (xy[:, 1] < self.r.height - 2)
            )
            status = np.zeros(len(xyz), dtype=np.uint8)
            status[keep] = 1  # visible according to CoTracker; not yet validated
            self.point_status[part] = status
            self.visible_candidate_counts[part] += int(keep.sum())
            if keep.any() and visible_depth is not None and visible_ids is not None:
                # Project this Part's known 3D query identity using the last
                # reliable Part transform. This location is independent from
                # CoTracker's potentially drifted screen coordinate.
                pred_world = xyz @ self.delta[part, :3, :3].T + self.delta[part, :3, 3]
                w2c = getattr(self.r, "w2c_cv", None)
                if w2c is None:
                    w2c = np.linalg.inv(self.r.c2w_cv)
                pred_cam = pred_world @ w2c[:3, :3].T + w2c[:3, 3]
                zp = pred_cam[:, 2]
                valid_z = np.isfinite(zp) & (zp > 1e-6)
                pu = np.zeros(len(zp), dtype=np.int64)
                pv = np.zeros(len(zp), dtype=np.int64)
                pu[valid_z] = np.rint(self.r.K[0,0]*pred_cam[valid_z,0]/zp[valid_z]+self.r.K[0,2]).astype(np.int64)
                pv[valid_z] = np.rint(self.r.K[1,1]*pred_cam[valid_z,1]/zp[valid_z]+self.r.K[1,2]).astype(np.int64)
                in_image = valid_z & (pu>=1) & (pu<self.r.width-1) & (pv>=1) & (pv<self.r.height-1)
                use = np.flatnonzero(keep & in_image)
                occluded = np.zeros(len(zp), dtype=bool)
                if len(use):
                    # A different Part in FRONT of the projected point proves
                    # an occlusion. A mere ID difference does not.
                    px, py = pu[use], pv[use]
                    z_front = visible_depth[py, px]
                    front_id = visible_ids[py, px]
                    margin = self.occlusion_depth_margin * max(1., float(np.median(zp[use])))
                    occluded[use] = ((front_id >= 0) & (front_id != part) &
                         np.isfinite(z_front) & (z_front > 0) &
                         (z_front + margin < zp[use]))
                self.occluded_counts[part] += int(np.count_nonzero(keep & occluded))
                status[keep & occluded] = 2  # predicted-coordinate occlusion
                keep &= ~occluded

                # Independently test the actual CoTracker coordinate against
                # the z-buffer of the currently predicted rigid scene.
                # This catches a stationary background query that has drifted
                # onto a nearer, moving foreground Part.
                track_idx = np.flatnonzero(keep & valid_z)
                if len(track_idx):
                    tx = np.rint(xy[track_idx, 0]).astype(np.int64).clip(0, self.r.width-1)
                    ty = np.rint(xy[track_idx, 1]).astype(np.int64).clip(0, self.r.height-1)
                    track_front = visible_ids[ty, tx]
                    track_depth = visible_depth[ty, tx]
                    # Only reject unambiguous nearer *other-Part* occupancy.
                    # Unknown depth or current-Part occupancy is not rejection.
                    margin = self.occlusion_depth_margin * np.maximum(1., zp[track_idx])
                    drift = ((track_front >= 0) & (track_front != part) &
                             np.isfinite(track_depth) & (track_depth > 0) &
                             (track_depth + margin < zp[track_idx]))
                    rejected = track_idx[drift]
                    keep[rejected] = False
                    status[rejected] = 3
                    self.cross_part_rejected[part] += len(rejected)
            if np.count_nonzero(keep) < self.min_tracks:
                continue
            object_pts = np.asarray(xyz[keep], dtype=np.float64)
            image_pts = xy[keep]
            # OpenCV uses 3D G0 world coordinates, converting camera-to-world
            # after PnP just as the original LK implementation does.
            # Two competing PnP hypotheses: global EPNP-RANSAC and a
            # temporally anchored estimate. RANSAC alone can lock onto a
            # repeated blade texture and return a mathematically valid but
            # physically incorrect rigid transform.
            candidates = []
            try:
                ok, rv, tv, ii = cv2.solvePnPRansac(
                    object_pts, image_pts, self.r.K, self.r.dist,
                    iterationsCount=300, reprojectionError=self.reprojection_px,
                    confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
                if ok and ii is not None and len(ii) >= self.min_tracks:
                    candidates.append(('global',rv,tv,ii.reshape(-1)))
            except cv2.error:
                pass

            if self.frame_index > 0 and self.frame_index-self.last_good[part] <= 3:
                # The preceding accepted G0->world pose predicts a camera
                # transform. Use only close predicted correspondences to seed
                # an alternative hypothesis; DO NOT force this hypothesis if
                # the object genuinely rotates rapidly.
                prior_cam = getattr(self.r, "w2c_cv", np.linalg.inv(self.r.c2w_cv)) @ self.delta[part]
                rv_prior, _ = cv2.Rodrigues(prior_cam[:3,:3])
                tv_prior = prior_cam[:3,3].reshape(3,1)
                pred, _ = cv2.projectPoints(
                    object_pts,rv_prior,tv_prior,self.r.K,self.r.dist)
                distance = np.linalg.norm(pred.reshape(-1,2)-image_pts,axis=1)
                near = np.flatnonzero(np.isfinite(distance) &
                    (distance <= max(12., 4.*self.reprojection_px)))
                if len(near) >= self.min_tracks:
                    try:
                        ok, rv, tv = cv2.solvePnP(
                            object_pts[near],image_pts[near],
                            self.r.K,self.r.dist,rv_prior.copy(),tv_prior.copy(),
                            True,flags=cv2.SOLVEPNP_ITERATIVE)
                        if ok:
                            candidates.append(('temporal',rv,tv,near))
                    except cv2.error:
                        pass

            evaluated = []
            for source,rv,tv,_ in candidates:
                # Evaluate all hypotheses against the same original point set,
                # never on just the subset that was used to fit each one.
                pp,_ = cv2.projectPoints(object_pts,rv,tv,self.r.K,self.r.dist)
                residual = np.linalg.norm(pp.reshape(-1,2)-image_pts,axis=1)
                support = np.flatnonzero(np.isfinite(residual) &
                    (residual <= self.reprojection_px))
                if len(support) < self.min_tracks: continue
                evaluated.append((len(support),-float(np.median(residual[support])),
                                  source,rv,tv,support))
            if not evaluated:
                continue
            evaluated.sort(key=lambda x:(x[0],x[1]),reverse=True)
            _,_,pnp_source,rvec,tvec,indices = evaluated[0]
            # Refine ONLY on consensus members, re-evaluate after refinement.
            try:
                cv2.solvePnPRefineLM(
                    object_pts[indices], image_pts[indices],
                    self.r.K,self.r.dist,rvec,tvec)
            except cv2.error:
                continue
            all_proj,_ = cv2.projectPoints(object_pts,rvec,tvec,self.r.K,self.r.dist)
            all_resid=np.linalg.norm(all_proj.reshape(-1,2)-image_pts,axis=1)
            indices=np.flatnonzero(np.isfinite(all_resid) &
                (all_resid <= self.reprojection_px))
            if len(indices) < self.min_tracks: continue
            # Record where valid measurements came from for diagnostics.
            if not hasattr(self,'pnp_hypothesis_counts'):
                self.pnp_hypothesis_counts = [{'global':0,'temporal':0}
                    for _ in range(P)]
            self.pnp_hypothesis_counts[part][pnp_source] += 1
            kept_idx = np.flatnonzero(keep)
            outliers = np.ones(len(kept_idx), dtype=bool)
            outliers[indices] = False
            status[kept_idx[outliers]] = 4
            self.pnp_outlier_counts[part] += int(outliers.sum())
            proj, _ = cv2.projectPoints(
                object_pts[indices], rvec, tvec, self.r.K, self.r.dist
            )
            errors_px = np.linalg.norm(
                proj.reshape(-1, 2) - image_pts[indices], axis=1
            )
            med = float(np.median(errors_px))
            if not np.isfinite(med) or med > self.reprojection_px:
                continue
            R, _ = cv2.Rodrigues(rvec)
            obj_to_cv = np.eye(4, dtype=np.float64)
            obj_to_cv[:3, :3] = R
            obj_to_cv[:3, 3] = tvec.reshape(3)
            delta = self.r.c2w_cv @ obj_to_cv
            if not np.isfinite(delta).all():
                continue
            # Compare moving PnP and fixed-camera STATIC hypotheses on the
            # EXACT SAME PnP inliers. A coherent flow is not motion evidence.
            w2c_static = getattr(self.r, "w2c_cv", np.linalg.inv(self.r.c2w_cv))
            static_cam = object_pts[indices] @ w2c_static[:3,:3].T + w2c_static[:3,3]
            positive = static_cam[:,2] > 1e-6
            static_uv = np.full((len(indices),2), np.inf, dtype=np.float64)
            if positive.any():
                static_uv[positive,0] = self.r.K[0,0]*static_cam[positive,0]/static_cam[positive,2]+self.r.K[0,2]
                static_uv[positive,1] = self.r.K[1,1]*static_cam[positive,1]/static_cam[positive,2]+self.r.K[1,2]
            static_resid=np.linalg.norm(static_uv-image_pts[indices],axis=1)
            static_med=float(np.median(static_resid[np.isfinite(static_resid)])) if np.isfinite(static_resid).any() else np.inf
            self.motion_model_support[part].append({
                'frame':int(self.frame_index), 'static_median_px':static_med,
                'moving_median_px':med, 'inliers':int(len(indices)),
                'gain_px':float(static_med-med) if np.isfinite(static_med) else 0.0
            })
            # Geometric consensus is kept separate from network visibility.
            status[kept_idx[indices]] = 5
            self.delta[part] = delta
            self.last_good[part] = self.frame_index
            measured[part] = True
            counts[part] = len(indices)
            errors[part] = med
        return self.delta.copy(), measured, counts, errors

    def step(self, bgr):
        if self.prev_gray is None:
            raise RuntimeError('Call start() first')

        self.frame_index += 1

        gray = cv2.cvtColor(
            bgr, cv2.COLOR_BGR2GRAY
        )

        measured = np.zeros(
            self.r.parts, dtype=bool
        )
        counts = np.zeros(
            self.r.parts, np.int32
        )
        errors = np.full(
            self.r.parts, np.inf, np.float32
        )

        for p in range(self.r.parts):
            if (
                self.points[p] is None
                or len(self.points[p]) < self.min_tracks
            ):
                continue

            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray,
                gray,
                self.points[p],
                None,
                winSize=(25, 25),
                maxLevel=3,
                criteria=(
                    cv2.TERM_CRITERIA_EPS
                    | cv2.TERM_CRITERIA_COUNT,
                    35, .01
                )
            )

            if next_pts is None or status is None:
                self.points[p] = None
                continue

            keep = status.reshape(-1) != 0
            xy = next_pts.reshape(-1, 2)

            keep &= np.isfinite(xy).all(axis=1)
            keep &= (
                (xy[:, 0] >= 2)
                & (xy[:, 0] < self.r.width - 2)
                & (xy[:, 1] >= 2)
                & (xy[:, 1] < self.r.height - 2)
            )

            xy, xyz = (
                xy[keep],
                self.xyz[p][keep]
            )

            self.points[p] = (
                xy.reshape(-1, 1, 2).astype(np.float32)
            )
            self.xyz[p] = xyz

            if len(xyz) < max(6, self.min_tracks):
                continue

            success, rvec, tvec, inlier = (
                cv2.solvePnPRansac(
                    xyz.astype(np.float64),
                    xy.astype(np.float64),
                    self.r.K,
                    self.r.dist,
                    iterationsCount=150,
                    reprojectionError=self.reprojection_px,
                    confidence=0.995,
                    flags=cv2.SOLVEPNP_EPNP
                )
            )

            if (
                not success
                or inlier is None
                or len(inlier) < self.min_tracks
            ):
                continue

            inlier = inlier.ravel()

            cv2.solvePnPRefineLM(
                xyz[inlier].astype(np.float64),
                xy[inlier].astype(np.float64),
                self.r.K,
                self.r.dist,
                rvec,
                tvec
            )

            projected, _ = cv2.projectPoints(
                xyz[inlier].astype(np.float64),
                rvec, tvec,
                self.r.K, self.r.dist
            )

            error = np.linalg.norm(
                projected.reshape(-1, 2)
                - xy[inlier],
                axis=1
            )

            if np.median(error) > self.reprojection_px:
                continue

            rmat, _ = cv2.Rodrigues(rvec)
            object_to_cv = np.eye(4)

            object_to_cv[:3, :3] = rmat
            object_to_cv[:3, 3] = tvec.ravel()

            estimated = (
                self.r.c2w_cv @ object_to_cv
            )

            if not np.isfinite(estimated).all():
                continue

            self.delta[p] = estimated
            self.last_good[p] = self.frame_index

            measured[p] = True
            counts[p] = len(inlier)
            errors[p] = float(np.median(error))

            self.points[p] = (
                xy[inlier]
                .reshape(-1, 1, 2)
                .astype(np.float32)
            )
            self.xyz[p] = xyz[inlier]

        self.prev_gray = gray

        if (
            self.reanchor_every > 0
            and self.frame_index % self.reanchor_every == 0
        ):
            _, depth, ids, _ = (
                self.r.snapshot(self.delta)
            )

            for p in range(self.r.parts):
                # Short lost intervals can be reacquired
                # using the last measured pose.
                if (
                    measured[p]
                    or self.frame_index
                    - self.last_good[p] <= 3
                ):
                    self._seed(
                        bgr, gray, depth,
                        ids, self.delta, p
                    )

        return (
            self.delta.copy(),
            measured,
            counts,
            errors
        )
