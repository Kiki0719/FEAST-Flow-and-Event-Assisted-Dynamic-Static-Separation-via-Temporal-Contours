"""CoTracker3 Offline adapter for static-G0 multi-Part pose initialization.

Meta CoTracker3 is used only for (x,y) trajectories and visibility. All 3D
coordinates, Part IDs, camera transforms and Blender structures are original.
"""
from __future__ import annotations

import numpy as np


def load_cotracker3(device='cuda', checkpoint=None):
    import torch
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CoTracker3 requested CUDA, but CUDA is unavailable')
    if checkpoint:
        # pip install -e /path/to/co-tracker
        from cotracker.predictor import CoTrackerPredictor
        model = CoTrackerPredictor(
            checkpoint=str(checkpoint), offline=True, v2=False,
            window_len=60
        )
    else:
        # Official checkpoint; downloaded/cached automatically on first call.
        model = torch.hub.load(
            'facebookresearch/co-tracker', 'cotracker3_offline',
            pretrained=True, trust_repo=True
        )
    return model.to(device).eval()


def _model_outputs(model, video, xy, batch_points, device):
    """Predict all 2D query positions for one window (points at t=0).

    Batches avoid processing all parts' points simultaneously on 24GB GPUs.
    Returned point order equals input query order.
    """
    import torch
    if len(xy) == 0:
        return (np.zeros((len(video), 0, 2), dtype=np.float32),
                np.zeros((len(video), 0), dtype=np.float32))
    rgb = np.stack([x[:, :, ::-1] for x in video]).copy()
    tensor = torch.from_numpy(rgb).permute(0, 3, 1, 2)[None].float().to(device)
    all_tracks, all_visibility = [], []
    try:
        for start in range(0, len(xy), batch_points):
            batch = xy[start:start + batch_points]
            prompt = np.concatenate(
                [np.zeros((len(batch), 1), dtype=np.float32), batch],
                axis=1
            )
            queries = torch.from_numpy(prompt)[None].to(device)
            with torch.inference_mode():
                pred, visibility = model(tensor, queries=queries)
            track_np = pred[0].detach().cpu().numpy()
            vis_np = visibility[0].detach().cpu().numpy()
            if vis_np.ndim == 3 and vis_np.shape[-1] == 1:
                vis_np = vis_np[..., 0]
            if track_np.shape != (len(video), len(batch), 2):
                raise RuntimeError(f'Unexpected CoTracker tracks: {track_np.shape}')
            if vis_np.shape != (len(video), len(batch)):
                raise RuntimeError(f'Unexpected CoTracker visibility: {vis_np.shape}')
            all_tracks.append(track_np)
            all_visibility.append(vis_np)
    finally:
        del tensor
    return (np.concatenate(all_tracks, axis=1),
            np.concatenate(all_visibility, axis=1))


def predict_part_poses(frames, tracker, model, *, window_frames=48,
                       batch_points=384, visibility_threshold=0.5,
                       max_stale_frames=3, device='cuda', diagnostic_sink=None):
    """Yield (frame, SE3, measured, inliers, errors) for all frames.

    At each window boundary, the last processed video frame is reused as
    the new query anchor. Recently measured poses render new depth/Part masks.
    Windows [0..47], [47..94], etc. share exactly one frame, without a time
    jump; the first frame is never solved by PnP.
    """
    if len(frames) < 2 or window_frames < 2 or batch_points < 1:
        raise ValueError('Invalid CoTracker video/window/batch dimensions')
    first = tracker.start(frames[0])
    yield 0, *first
    window_start = 0
    while window_start < len(frames) - 1:
        if window_start > 0:
            tracker.reanchor_cotracker(
                frames[window_start], max_stale_frames=max_stale_frames
            )
        end = min(window_start + window_frames, len(frames))
        track_ids, query_xy = [], []
        for p, pts in enumerate(tracker.points):
            if pts is not None and tracker.xyz[p] is not None:
                xy = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
                if len(xy) >= tracker.min_tracks:
                    track_ids.extend([p] * len(xy))
                    query_xy.extend(xy.tolist())
        query_xy = np.asarray(query_xy, dtype=np.float32).reshape(-1, 2)
        track_ids = np.asarray(track_ids, dtype=np.int32)
        if len(query_xy):
            track, visibility = _model_outputs(
                model, frames[window_start:end], query_xy,
                batch_points, device
            )
        else:
            track = np.zeros((end-window_start, 0, 2), dtype=np.float32)
            visibility = np.zeros((end-window_start, 0), dtype=np.float32)
        slices = {
            p: np.flatnonzero(track_ids == p) for p in np.unique(track_ids)
        }
        validation = np.zeros((end-window_start, len(track_ids)), dtype=np.uint8)
        validation[0] = np.where(visibility[0] >= visibility_threshold, 1, 0).astype(np.uint8)
        for t in range(window_start + 1, end):
            j = t - window_start
            xy_by = {int(p): track[j, ii] for p, ii in slices.items()}
            vis_by = {int(p): visibility[j, ii] for p, ii in slices.items()}
            result = tracker.step_cotracker(
                xy_by, vis_by, frame_index=t,
                visibility_threshold=visibility_threshold
            )
            # Stable query order: never remove entries from a CoTracker window.
            for p, ii in slices.items():
                state = tracker.point_status[int(p)]
                if state is not None and len(state) == len(ii):
                    validation[j, ii] = state
            yield t, *result
        if diagnostic_sink is not None:
            diagnostic_sink(window_start, track_ids, query_xy, track, visibility,
                            validation)
        window_start = end - 1
