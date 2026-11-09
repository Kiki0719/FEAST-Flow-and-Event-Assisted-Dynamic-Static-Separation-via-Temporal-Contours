
"""Validate and fill per-Part SE(3) trajectories. Never silently invent long gaps."""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def _validate(p):
    if not np.isfinite(p).all():
        return False

    r = p[:3, :3]
    return (
        np.max(np.abs(r.T @ r - np.eye(3))) < 1e-2
        and abs(np.linalg.det(r) - 1.0) < 1e-2
        and np.max(
            np.abs(p[3] - [0, 0, 0, 1])
        ) < 1e-5
    )


def _interpolate(a, b, fraction):
    result = np.eye(4, dtype=np.float64)

    result[:3, 3] = (
        (1.0 - fraction) * a[:3, 3]
        + fraction * b[:3, 3]
    )

    rotations = Rotation.from_matrix(
        np.stack([a[:3, :3], b[:3, :3]])
    )

    result[:3, :3] = Slerp(
        [0, 1], rotations
    )([fraction]).as_matrix()[0]

    return result


def optimize_tracks(
    raw,
    observed,
    max_gap=3,
    allow_hold=False,
    smoothing_sigma=0.0,
    static_parts=(),
    return_masks=False,
    reprojection_errors=None,
    max_reprojection_px=3.0,
    max_step_degrees=30.0,
    max_step_relative=0.20,
    scene_extent=1.0
):
    """
    Returns:
        result: (T,P,4,4), float32
        filled: (T,P), bool
        report: tracking diagnostics

    observed indicates independently estimated poses.
    Long gaps remain unknown with finite diagnostic placeholders; never certified.
    """
    raw = np.asarray(raw, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)

    if raw.ndim != 4 or raw.shape[2:] != (4, 4):
        raise ValueError(
            'raw must have shape (T,P,4,4)'
        )

    t_count, p_count = raw.shape[:2]

    if (
        observed.shape != (t_count, p_count)
        or t_count < 2
    ):
        raise ValueError(
            'Invalid observation mask or frame count'
        )

    if max_gap < 0 or smoothing_sigma < 0:
        raise ValueError(
            'max_gap/smoothing_sigma must be nonnegative'
        )

    static_parts = set(
        int(p) for p in static_parts
    )

    if any(
        p < 0 or p >= p_count
        for p in static_parts
    ):
        raise ValueError(
            'Invalid static Part ID'
        )

    result = raw.copy()
    filled = np.zeros_like(observed)
    unknown = np.zeros_like(observed)
    accepted = observed.copy()
    static_mask = np.zeros_like(observed)
    interpolated_mask = np.zeros_like(observed)
    report = []
    identity = np.eye(4)

    for p in range(p_count):
        if p in static_parts:
            result[:, p] = identity
            static_mask[:, p] = True
            accepted[1:, p] = False

            report.append({
                'part': p,
                'static_assumption': True,
                'observed': 1,
                'interpolated': 0,
                'held': t_count - 1
            })
            continue

        valid = observed[:, p].copy()
        valid[0] = True
        result[0, p] = identity

        for i in range(1, t_count):
            if valid[i] and not _validate(
                result[i, p]
            ):
                valid[i] = False

        if reprojection_errors is not None:
            errs = np.asarray(reprojection_errors)[:, p]
            valid[1:] &= np.isfinite(errs[1:]) & (errs[1:] <= max_reprojection_px)
        # Reject isolated implausible SE(3) jumps only when temporal neighbors agree.
        # Keep ambiguous frames unknown rather than replacing them with a false PnP pose.
        for i in range(2, t_count - 1):
            if not (valid[i-1] and valid[i] and valid[i+1]):
                continue
            left = result[i-1, p]; mid = result[i+1, p]; probe = result[i, p]
            a = Rotation.from_matrix(left[:3,:3]); b = Rotation.from_matrix(mid[:3,:3]); c = Rotation.from_matrix(probe[:3,:3])
            deg = np.degrees((a.inv()*c).magnitude())
            deg_after = np.degrees((c.inv()*b).magnitude())
            direct = np.degrees((a.inv()*b).magnitude())
            trans_left = np.linalg.norm(probe[:3,3]-left[:3,3])
            trans_right = np.linalg.norm(probe[:3,3]-mid[:3,3])
            trans_direct = np.linalg.norm(left[:3,3]-mid[:3,3])
            rotation_spike = (deg > max_step_degrees
                              and deg_after > max_step_degrees
                              and direct < max_step_degrees)
            translation_spike = (trans_left > max_step_relative * scene_extent
                                 and trans_right > max_step_relative * scene_extent
                                 and trans_direct < max_step_relative * scene_extent)
            # A one-frame spike must disagree with BOTH neighbors while those
            # neighbors agree with one another. Real fast rigid rotation is
            # otherwise retained, even when its per-frame angle is large.
            if rotation_spike or translation_spike:
                valid[i] = False
        accepted[:, p] = valid
        measured = np.flatnonzero(valid)

        if len(measured) < 2:
            result[:,p] = identity
            unknown[1:,p] = True
            report.append({'part':p,'static_assumption':False,'observed':int(valid.sum()),
                           'interpolated':0,'held':0,'unknown':t_count-1})
            continue

        interp_count = 0
        held_count = 0

        for i in range(1, t_count):
            if valid[i]:
                continue

            left = measured[measured < i]
            right = measured[measured > i]

            lo = (
                int(left[-1]) if len(left)
                else None
            )
            hi = (
                int(right[0]) if len(right)
                else None
            )

            gap = (
                hi - lo - 1
                if lo is not None and hi is not None
                else t_count
            )

            if (
                lo is not None
                and hi is not None
                and gap <= max_gap
            ):
                result[i, p] = _interpolate(
                    result[lo, p],
                    result[hi, p],
                    (i - lo) / (hi - lo)
                )
                interp_count += 1
                interpolated_mask[i, p] = True

            elif allow_hold and lo is not None:
                result[i, p] = result[lo, p]
                held_count += 1

            else:
                # Unknown: placeholder only. Never flag as restored.
                result[i, p] = result[lo, p] if lo is not None else identity
                unknown[i, p] = True

            filled[i, p] = interpolated_mask[i, p] or (allow_hold and not unknown[i, p] and not valid[i])

        if smoothing_sigma > 0 and not unknown[:, p].any():
            # SO(3) local averaging, rather than
            # smoothing rotation-matrix elements.
            original = result[:, p].copy()

            radius = max(
                1,
                int(np.ceil(3 * smoothing_sigma))
            )

            for i in range(1, t_count):
                neighbors = np.arange(
                    max(0, i-radius),
                    min(t_count, i+radius+1)
                )

                weights = np.exp(
                    -0.5 * (
                        (neighbors-i) / smoothing_sigma
                    ) ** 2
                )
                weights /= weights.sum()

                result[i, p, :3, 3] = np.sum(
                    original[neighbors, :3, 3]
                    * weights[:, None],
                    axis=0
                )

                reference = Rotation.from_matrix(
                    original[i, :3, :3]
                )

                nearby = Rotation.from_matrix(
                    original[neighbors, :3, :3]
                )

                offsets = (
                    reference.inv() * nearby
                ).as_rotvec()

                result[i, p, :3, :3] = (
                    reference
                    * Rotation.from_rotvec(
                        np.sum(
                            offsets * weights[:, None],
                            axis=0
                        )
                    )
                ).as_matrix()

        if not all(
            _validate(pose)
            for pose in result[:, p]
        ):
            raise RuntimeError(
                f'Part {p}: invalid optimized SE(3)'
            )

        report.append({
            'part': p,
            'static_assumption': False,
            'observed': int(valid.sum()),
            'interpolated': interp_count,
            'held': held_count,
            'unknown': int(unknown[:,p].sum())
        })

    if return_masks:
        return result.astype(np.float32), filled, report, {
            'observed': accepted, 'interpolated': interpolated_mask,
            'unknown': unknown, 'static_assumed': static_mask,
        }
    return result.astype(np.float32), filled, report


# Automatic Part motion classification from RGB optical flow and PnP.
import cv2
from scipy.spatial.transform import Rotation


class AutoPartMotionDetector:
    def __init__(
        self, part_ids, num_parts,
        scene_extent=1.0,
        min_visible_px=36,
        optical_px=1.4
    ):
        self.parts = int(num_parts)
        self.scene_extent = max(float(scene_extent), 1e-8)
        self.min_visible_px = min_visible_px
        self.optical_px = optical_px

        self.masks = []
        for p in range(self.parts):
            mask = (part_ids == p).astype(np.uint8)
            eroded = cv2.erode(
                mask, np.ones((3, 3), np.uint8)
            )
            if eroded.sum() >= min_visible_px:
                mask = eroded
            self.masks.append(mask.astype(bool))

        bg = (part_ids < 0).astype(np.uint8)
        self.background = cv2.erode(
            bg, np.ones((5, 5), np.uint8)
        ).astype(bool)

        self.prev_gray = None
        self.motion_hits = np.zeros(self.parts, np.int32)
        self.covered = np.zeros(self.parts, np.int32)

    def observe(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            0.5, 3, 21, 3, 7, 1.5, 0
        )

        if self.background.sum() >= 300:
            global_flow = np.median(
                flow[self.background][::8], axis=0
            )
        else:
            global_flow = np.zeros(2)

        magnitude = np.linalg.norm(
            flow - global_flow, axis=2
        )

        for p, mask in enumerate(self.masks):
            if mask.sum() < self.min_visible_px:
                continue

            local = magnitude[mask]
            fraction = np.mean(local > self.optical_px)
            self.covered[p] += 1

            if (
                fraction >= 0.35
                and np.percentile(local, 75) > self.optical_px
            ):
                self.motion_hits[p] += 1

        self.prev_gray = gray

    def classify(self, poses, valid):
        poses = np.asarray(poses, dtype=np.float64)
        valid = np.asarray(valid, dtype=bool)

        T, P = valid.shape
        if poses.shape != (T, P, 4, 4) or P != self.parts:
            raise ValueError("Part trajectory shape mismatch")

        auto_static = []
        unverified = []
        report = []

        for p in range(P):
            measured = np.flatnonzero(valid[1:, p]) + 1
            count = len(measured)

            pose_reliable = count >= max(8, int(0.50 * (T - 1)))
            flow_reliable = self.covered[p] >= max(8, int(0.50 * (T - 1)))
            flow_ratio = self.motion_hits[p] / max(self.covered[p], 1)
            flow_moving = flow_reliable and flow_ratio >= 0.25
            angle = 0.0
            translation = 0.0
            jump_fraction = 1.0
            if count:
                sample = poses[measured, p]
                good = np.array([_validate(x) for x in sample], bool)
                sample = sample[good]
                ids = measured[good]
                if len(sample):
                    angle = float(np.percentile(np.degrees(
                        Rotation.from_matrix(sample[:, :3, :3]).magnitude()), 80))
                    translation = float(np.percentile(
                        np.linalg.norm(sample[:, :3, 3], axis=1), 80))
                steps = []
                for j in range(1, len(ids)):
                    if ids[j] != ids[j-1]+1: continue
                    A, B = poses[ids[j-1],p], poses[ids[j],p]
                    da = np.degrees((Rotation.from_matrix(A[:3,:3]).inv()
                                     * Rotation.from_matrix(B[:3,:3])).magnitude())
                    dt = np.linalg.norm(B[:3,3]-A[:3,3])
                    steps.append(da > 30 or dt > 0.2 * self.scene_extent)
                jump_fraction = float(np.mean(steps)) if len(steps) >= 4 else 1.0
            pose_moving = pose_reliable and jump_fraction < 0.10 and (
                angle >= 2.0 or translation >= 0.025 * self.scene_extent)
            # Initial-mask Farneback may be contaminated by other moving Parts;
            # optical flow alone cannot certify a moving rigid Part.
            if pose_moving and flow_moving:
                status = 'moving_candidate'  # needs whole-Part RGB render validation
            elif (not flow_moving and flow_reliable and count >= .8*(T-1)
                  and angle < 1.6 and translation < .015*self.scene_extent
                  and jump_fraction < .05):
                status = 'static'
                auto_static.append(p)
            else:
                status = 'unverified_motion_conflict'
                unverified.append(p)
            report.append({
                "part": p,
                "status": status,
                "tracked_frames": count,
                "visible_pixels": int(self.masks[p].sum()),
                "flow_motion_frames": int(
                    self.motion_hits[p]
                ),
                "flow_observed_frames": int(
                    self.covered[p]
                ),
                "rotation_p80_deg": angle,
                "translation_p80": translation,
                "pose_jump_fraction": jump_fraction,
                "flow_motion_ratio": flow_ratio
            })

        return auto_static, unverified, report


def joint_se3_regularizer(rotvecs, center_shifts, dt):
    """Differentiable temporal acceleration loss, invariant to rotation-vector wrapping.

    rotvecs: (K,P,3), center_shifts: (K,P,3), dt: (K-1,).
    For uniform angular motion, successive SO(3) relative rotations should match.
    """
    import torch
    if rotvecs.shape[0] < 3:
        return center_shifts.square().mean()*0
    x,y,z=rotvecs.unbind(-1)
    zero=torch.zeros_like(x)
    skew=torch.stack((zero,-z,y,z,zero,-x,-y,x,zero),dim=-1).reshape(*rotvecs.shape[:-1],3,3)
    rotations=torch.linalg.matrix_exp(skew)
    rel=rotations[:-1].transpose(-1,-2) @ rotations[1:]
    rot_acc=(rel[1:]-rel[:-1]).square().mean()
    velocities=(center_shifts[1:]-center_shifts[:-1]) / dt[:,None,None].clamp_min(1.)
    shift_acc=(velocities[1:]-velocities[:-1]).square().mean()
    return rot_acc + shift_acc
