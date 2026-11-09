"""Deterministic, structure-aware 2D queries for each pre-bound rigid Group.

CoTracker3 predicts trajectories of supplied queries. It does *not* accept
semantic/rigid Group labels as constraints. We therefore sample points within
the rendered G0 Group-ID regions and carry the Group identity externally.
"""
from __future__ import annotations

import cv2
import numpy as np


def sample_group_grid(group_ids, depth, group_id, max_points=1000,
                      min_points=10, candidate_limit=6000,
                      appearance_mask=None, erode_px=3):
    """Return <=max_points [x,y] queries from one visible rigid Group.

    The source ID map must come from depth-aware rendered G0, not just a 2D
    image segmentation. A deterministic farthest-point pass spreads samples
    across narrow, disconnected and differently sized visible regions.
    Samples retain valid G0 depth for unprojection to 3D later on.
    """
    ids = np.asarray(group_ids)
    depth = np.asarray(depth)
    if ids.ndim != 2 or ids.shape != depth.shape:
        raise ValueError('Group-ID and depth maps must have matching HxW shape')
    if max_points < 1 or min_points < 1 or candidate_limit < min_points:
        raise ValueError('Invalid sampling point limits')
    allowed = ((ids == int(group_id)) & np.isfinite(depth) & (depth > 0))
    if appearance_mask is not None:
        m = np.asarray(appearance_mask, dtype=bool)
        if m.shape != allowed.shape:
            raise ValueError('Appearance mask shape mismatch')
        allowed &= m
    if erode_px >= 3:
        core = cv2.erode(allowed.astype(np.uint8),
                         np.ones((erode_px, erode_px), np.uint8)) > 0
        # Keep small/thin groups accessible, but never expand their masks.
        if np.count_nonzero(core) >= min_points:
            allowed = core
    ys, xs = np.nonzero(allowed)
    if len(xs) < min_points:
        return np.empty((0, 2), dtype=np.float32)

    # Cap candidate processing cost, but always retain a spatially uniform
    # subsequence over row-major group pixels (also covers narrow groups).
    if len(xs) > candidate_limit:
        selected = np.linspace(0, len(xs) - 1, candidate_limit).astype(np.int64)
        xs, ys = xs[selected], ys[selected]
    candidates = np.column_stack((xs, ys)).astype(np.float32)
    want = min(int(max_points), len(candidates))
    if want == len(candidates):
        return candidates

    # Farthest-point sampling in the 2D mask (NOT 3D rigid segmentation).
    # Restrict to candidate mask pixels; no background/depth-hole queries.
    center = candidates.mean(axis=0)
    first = int(np.argmin(np.sum((candidates - center) ** 2, axis=1)))
    selected_idx = np.empty(want, dtype=np.int64)
    selected_idx[0] = first
    min_distance_sq = np.sum((candidates - candidates[first]) ** 2, axis=1)
    min_distance_sq[first] = -1.0
    for i in range(1, want):
        next_idx = int(np.argmax(min_distance_sq))
        selected_idx[i] = next_idx
        new_dist = np.sum((candidates - candidates[next_idx]) ** 2, axis=1)
        np.minimum(min_distance_sq, new_dist, out=min_distance_sq)
        min_distance_sq[selected_idx[:i+1]] = -1.0
    return candidates[selected_idx]


def sample_visible_regular_grid(group_ids, depth, group_id, *, stride=8,
                                min_points=10, appearance_mask=None, erode_px=3):
    """Regular CoTracker query grid over initially frontmost depth-valid Part pixels.

    Every returned pixel belongs to the renderer's frontmost Part ID. No
    Gaussian-count quota or 1000-point cap; stride is the spatial density.
    """
    ids=np.asarray(group_ids); z=np.asarray(depth)
    if ids.shape != z.shape or ids.ndim != 2 or stride < 1:
        raise ValueError('Invalid ID/depth or grid stride')
    m=(ids==int(group_id)) & np.isfinite(z) & (z>0)
    if appearance_mask is not None:
        a=np.asarray(appearance_mask,bool)
        if a.shape != m.shape: raise ValueError('Appearance mask mismatch')
        m &= a
    if erode_px >= 3:
        core=cv2.erode(m.astype(np.uint8),np.ones((erode_px,erode_px),np.uint8))>0
        if core.sum() >= min_points: m=core
    h,w=m.shape
    # Sample a regular 2D lattice, not a quota-based farthest-point subset.
    yy,xx=np.mgrid[stride//2:h:stride, stride//2:w:stride]
    xy=np.column_stack((xx.ravel(),yy.ravel()))
    valid=m[xy[:,1],xy[:,0]]
    result=xy[valid].astype(np.float32)
    # Preserve narrow or small but genuinely visible parts with a finer lattice.
    if len(result)<min_points and stride>1:
        yy,xx=np.mgrid[0:h:max(1,stride//2),0:w:max(1,stride//2)]
        xy=np.column_stack((xx.ravel(),yy.ravel()))
        result=xy[m[xy[:,1],xy[:,0]]].astype(np.float32)
    return result if len(result)>=min_points else np.empty((0,2),np.float32)
