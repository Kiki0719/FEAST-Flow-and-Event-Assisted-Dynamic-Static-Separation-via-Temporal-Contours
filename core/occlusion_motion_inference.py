"""Occlusion-aware candidate motion inference for PRE-BOUND 3D Gaussian Parts.

No dynamic 3D information is taken from Blender. Proposed hidden-Part motion
is either (a) RGB render-supported or (b) an explicit user-supplied rigid link.
Depth overlap alone NEVER proves that a hidden Part must move.

The renderer must provide:
  xyz0, ids, w2c_cv, K, parts, snapshot(delta), render_rgb(delta)
All motion matrices are G0-world-relative SE(3) transforms.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def _identity(n):
    return np.broadcast_to(np.eye(4, dtype=np.float64), (n, 4, 4)).copy()


def _pose_motion(T):
    return (float(np.rad2deg(Rotation.from_matrix(T[:3, :3]).magnitude()))
            + float(np.linalg.norm(T[:3, 3])) * 100.)


def _relative_disagreement(left, right):
    """Compare G0-relative poses, not absolute object origins."""
    angle = np.rad2deg((Rotation.from_matrix(left[:3, :3]).inv()
                       * Rotation.from_matrix(right[:3, :3])).magnitude())
    translation = np.linalg.norm(left[:3, 3] - right[:3, 3])
    return float(angle), float(translation)


def _json_links(path, total):
    if path is None:
        return {}
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data, dict) or not isinstance(data.get('rigid_links'), list):
        raise ValueError('Rigid link file must be {"rigid_links":[{"source":0,"target":1}]}')
    links = {}
    for item in data['rigid_links']:
        source, target = int(item['source']), int(item['target'])
        if not (0 <= source < total and 0 <= target < total and source != target):
            raise ValueError(f'Invalid rigid link: {source}->{target}')
        if target in links and links[target] != source:
            raise ValueError(f'Conflicting rigid sources for target {target}')
        links[target] = source
    return links


class OcclusionMotionInferencer:
    """Assess visibility, compare source→target SE(3) against static renders.

    Candidate auto-generation uses depth ordering of G0 Gaussian centers, not
    merely the topmost Part ID. Occluded Part candidate pairs are screen-space
    proposals, subject to full-scene RGB render verification.
    """

    def __init__(self, renderer, *, scene_extent, min_depth_overlap=0.03,
                 max_depth_gap_relative=0.30, min_changed_pixels=80,
                 min_render_improvement=0.006, min_win_fraction=0.65,
                 min_evidence_frames=3, max_evaluation_frames=12,
                 max_candidates=24, seed=42):
        self.r = renderer
        self.extent = max(float(scene_extent), 1e-6)
        self.min_depth_overlap = float(min_depth_overlap)
        self.max_depth_gap = float(max_depth_gap_relative) * self.extent
        self.min_changed_pixels = int(min_changed_pixels)
        self.min_render_improvement = float(min_render_improvement)
        self.min_win_fraction = float(min_win_fraction)
        self.min_evidence_frames = int(min_evidence_frames)
        self.max_evaluation_frames = int(max_evaluation_frames)
        self.max_candidates = int(max_candidates)
        self.rng = np.random.default_rng(seed)
        if self.max_evaluation_frames < 3 or self.max_candidates < 1:
            raise ValueError('Too few render frames or candidates')

    def _sample_world(self, part, n=5000):
        idx = self.r.ids[part]
        if len(idx) == 0:
            return np.empty((0, 3), dtype=np.float64)
        # renderer.xyz0 is CUDA Tensor in production and numpy in the fake.
        if hasattr(idx, 'detach'):
            idx = idx.detach().cpu().numpy()
        idx = np.asarray(idx, dtype=np.int64)
        if len(idx) > n:
            idx = self.rng.choice(idx, n, replace=False)
        if hasattr(self.r.xyz0, 'detach'):
            import torch
            index_tensor = torch.as_tensor(idx, dtype=torch.long,
                                           device=self.r.xyz0.device)
            cloud = self.r.xyz0.index_select(0, index_tensor).detach().cpu().numpy()
        else:
            cloud = self.r.xyz0[idx]
        return np.asarray(cloud, dtype=np.float64)

    def depth_occlusion(self, source, target, initial_ids, depth):
        """Fraction of target G0 sample centers *behind visible source pixels*.

        Not a multilayer splat z-buffer: Gaussian centers approximate the
        second depth layer; RGB rendering does the final hypothesis check.
        """
        cloud = self._sample_world(target)
        if not len(cloud):
            return {'overlap_fraction': 0., 'behind_points': 0,
                    'projected_points': 0, 'median_depth_gap': None}
        cam = cloud @ self.r.w2c_cv[:3, :3].T + self.r.w2c_cv[:3, 3]
        valid = cam[:, 2] > 1e-5
        uv = np.full((len(cam), 2), np.nan)
        uv[valid, 0] = self.r.K[0, 0] * cam[valid, 0] / cam[valid, 2] + self.r.K[0, 2]
        uv[valid, 1] = self.r.K[1, 1] * cam[valid, 1] / cam[valid, 2] + self.r.K[1, 2]
        h, w = initial_ids.shape
        valid &= np.isfinite(uv).all(1)
        valid &= (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        projected = int(valid.sum())
        if not projected:
            return {'overlap_fraction': 0., 'behind_points': 0,
                    'projected_points': 0, 'median_depth_gap': None}
        uv_i = np.rint(uv[valid]).astype(np.int64)
        uv_i[:, 0] = uv_i[:, 0].clip(0, w-1)
        uv_i[:, 1] = uv_i[:, 1].clip(0, h-1)
        yy, xx = uv_i[:, 1], uv_i[:, 0]
        gap = cam[valid, 2] - depth[yy, xx]
        behind = ((initial_ids[yy, xx] == source) & np.isfinite(gap)
                  & (gap > 1e-3) & (gap <= self.max_depth_gap))
        count = int(behind.sum())
        return {'overlap_fraction': count / projected,
                'behind_points': count, 'projected_points': projected,
                'median_depth_gap': float(np.median(gap[behind])) if count else None}

    def _sample_frames(self, delta, source):
        magnitudes = np.array([_pose_motion(x) for x in delta[1:, source]], dtype=np.float64)
        moving = np.flatnonzero(magnitudes > 0.2) + 1
        if not len(moving):
            return []
        if len(moving) <= self.max_evaluation_frames:
            return moving.tolist()
        ordered = np.unique(np.concatenate([
            moving[np.linspace(0, len(moving)-1, self.max_evaluation_frames, dtype=int)],
            moving[np.argsort(magnitudes[moving-1])[-min(3, len(moving)):]]
        ]))
        if len(ordered) > self.max_evaluation_frames:
            ordered = ordered[np.linspace(0, len(ordered)-1, self.max_evaluation_frames, dtype=int)]
        return sorted(int(v) for v in ordered)

    def _score_candidate(self, delta, video_frames, source, target,
                         baseline_cache):
        usable = []
        for t in self._sample_frames(delta, source):
            if t not in baseline_cache:
                base = self.r.render_rgb(delta[t]).astype(np.float32) / 255.
                baseline_cache[t] = base
            else:
                base = baseline_cache[t]
            proposed = delta[t].copy()
            proposed[target] = delta[t, source]
            candidate = self.r.render_rgb(proposed).astype(np.float32) / 255.
            gt = video_frames[t].astype(np.float32) / 255.
            if candidate.shape != gt.shape or base.shape != gt.shape:
                raise ValueError('Video and renderer resolution mismatch')
            difference = np.max(np.abs(candidate - base), axis=2)
            mask = (difference >= 3. / 255.).astype(np.uint8)
            # Small edge halo accommodates antialiasing and alignment noise.
            mask = cv2.dilate(mask, np.ones((3, 3), np.uint8)) > 0
            changed = int(mask.sum())
            if changed < self.min_changed_pixels:
                continue
            baseline_mae = float(np.abs(gt - base)[mask].mean())
            proposal_mae = float(np.abs(gt - candidate)[mask].mean())
            improvement = baseline_mae - proposal_mae
            usable.append({'frame': int(t), 'changed_pixels': changed,
                           'baseline_mae': baseline_mae,
                           'linked_mae': proposal_mae,
                           'improvement': improvement})
        if not usable:
            return {'supported': False, 'reason': 'render_indistinguishable',
                    'evaluated_frames': 0, 'win_fraction': 0., 'mean_improvement': 0.,
                    'details': []}
        improvements = np.array([x['improvement'] for x in usable])
        win_fraction = float(np.mean(improvements > self.min_render_improvement / 2))
        avg = float(np.mean(improvements))
        return {'supported': bool(len(usable) >= self.min_evidence_frames
                                  and win_fraction >= self.min_win_fraction
                                  and avg >= self.min_render_improvement),
                'reason': 'render_supported' if (
                    len(usable) >= self.min_evidence_frames
                    and win_fraction >= self.min_win_fraction
                    and avg >= self.min_render_improvement
                ) else 'insufficient_render_support',
                'evaluated_frames': len(usable), 'win_fraction': win_fraction,
                'mean_improvement': avg, 'details': usable}

    def run(self, delta, observed, video_frames, classifications,
            *, rigid_links=None, initial_ids=None, initial_depth=None,
            unknown_mask=None, source_valid=None):
        delta = np.asarray(delta, dtype=np.float64).copy()
        observed = np.asarray(observed, dtype=bool)
        T, P = observed.shape
        if unknown_mask is None:
            unknown_mask = ~observed
        unknown_mask = np.asarray(unknown_mask, bool)
        if unknown_mask.shape != (T,P):
            raise ValueError('Unknown mask dimension mismatch')
        if source_valid is None:
            source_valid = observed
        source_valid = np.asarray(source_valid, bool)
        if source_valid.shape != (T,P):
            raise ValueError('Source-valid mask dimension mismatch')
        if delta.shape != (T, P, 4, 4) or len(video_frames) != T:
            raise ValueError('Pose/video dimensions mismatch')
        if initial_ids is None or initial_depth is None:
            _, d, ids, _ = self.r.snapshot(_identity(P))
            initial_ids = ids if initial_ids is None else initial_ids
            initial_depth = d if initial_depth is None else initial_depth
        if initial_depth is None or initial_depth.shape != initial_ids.shape:
            raise ValueError('Initial depth map required')
        prior = _json_links(rigid_links, P)
        measured_moving = sorted({int(x['part']) for x in classifications
                                   if x['status'] == 'moving'})
        unverified = {int(x['part']) for x in classifications
                      if x['status'].startswith('unverified')
                      or x['status'] == 'motion_detected_pose_unreliable'}
        # Only independently supported sources may propagate motion.
        proposed_sources = sorted(set(measured_moving) | set(prior.values()))
        # Never use another inferred Part as an automatic source.
        baseline_cache = {}
        inferred = np.zeros((T, P), dtype=bool)
        prior_only = np.zeros((T, P), dtype=bool)
        origin = np.full(P, -1, dtype=np.int32)
        decisions, candidates = [], []
        for source in proposed_sources:
            if source not in measured_moving:
                decisions.append({'source': source, 'reason': 'source_not_measured_moving'})
                continue
            for target in range(P):
                if source == target:
                    continue
                if not unknown_mask[1:, target].any() and target not in prior:
                    continue
                if target in prior and prior[target] != source:
                    continue
                geo = self.depth_occlusion(source, target, initial_ids, initial_depth)
                if target not in prior and (geo['overlap_fraction'] < self.min_depth_overlap
                                             or geo['behind_points'] < 10):
                    continue
                candidates.append((source, target, geo))
        candidates.sort(key=lambda x: (x[1] not in prior, -x[2]['overlap_fraction']))
        for source, target, geometry in candidates[:self.max_candidates]:
            if origin[target] != -1:
                continue
            # No override of independently measured contradictory target poses.
            idx = np.flatnonzero(observed[1:, target] & source_valid[1:,source]) + 1
            disagreement = [_relative_disagreement(delta[t, target], delta[t, source])
                            for t in idx]
            if disagreement and (max(x[0] for x in disagreement) > 6.
                                 or max(x[1] for x in disagreement) > 0.025 * self.extent):
                decisions.append({'source': source, 'target': target,
                                  'status': 'rejected_measured_pose_conflict',
                                  'geometry': geometry})
                continue
            evaluation = self._score_candidate(delta, video_frames, source, target,
                                               baseline_cache)
            linked = target in prior and prior[target] == source
            accept = evaluation['supported'] or linked
            decision = {'source': source, 'target': target,
                        'status': ('rgb_render_supported' if evaluation['supported']
                                   else 'explicit_rigid_link_prior' if linked
                                   else 'rejected_insufficient_evidence'),
                        'geometry': geometry, 'rgb_test': evaluation}
            if accept:
                to_set = unknown_mask[:, target] & source_valid[:,source]
                to_set[0] = False
                delta[to_set, target] = delta[to_set, source]
                inferred[to_set, target] = True
                prior_only[to_set, target] = not evaluation['supported']
                if to_set.any():
                    origin[target] = source
                baseline_cache.clear()  # accepted changes alter baseline renders
                decision['filled_frames'] = int(to_set.sum())
            decisions.append(decision)
        unresolved_mask = unknown_mask & (~inferred)
        unresolved_mask[0, :] = False
        unresolved = [int(x) for x in np.flatnonzero(unresolved_mask.any(axis=0))]
        summary = {'method': 'G0 second-layer projected depth candidate + '
                             'full-scene RGB render hypothesis comparison',
                   'sources_measured_moving': measured_moving,
                   'candidates_tested': len(decisions),
                   'accepted_targets': [int(x) for x in np.flatnonzero(origin >= 0)],
                   'unresolved_parts': unresolved,
                   'frame_count': T, 'decisions': decisions,
                   'unresolved_frame_counts': unresolved_mask.sum(axis=0).astype(int).tolist(),
                   'warning': 'Fully invisible motion is unobservable from RGB; '
                              'explicit rigid links are prior-inferred, not RGB-measured.'}
        return delta.astype(np.float32), inferred, prior_only, origin, summary
