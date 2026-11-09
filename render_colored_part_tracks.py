import argparse
import colorsys
import cv2
import numpy as np
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--tracking", required=True)
p.add_argument("--video", required=True)
p.add_argument("--output", required=True)
args = p.parse_args()

with np.load(args.tracking, allow_pickle=False) as z:
    print("Tracking fields:")
    for k in z.files:
        print(" ", k, z[k].shape)
    arrays = {k: z[k] for k in z.files}

def pick(keys):
    for k in keys:
        if k in arrays:
            return arrays[k]
    return None

tracks = pick([
    "tracks", "pred_tracks", "track_xy",
    "track_points", "trajectories", "points2d"
])

parts = pick([
    "part_ids", "track_part_ids", "track_parts",
    "point_part_ids", "point_parts", "query_part_ids", "labels"
])

visible = pick([
    "visibility", "visibilities", "visible",
    "pred_visibility", "track_visibility"
])

if tracks is None or parts is None:
    raise RuntimeError(
        "Cannot identify tracks or per-track Part IDs. "
        "See printed NPZ fields."
    )

tracks = np.asarray(tracks)
parts = np.asarray(parts).reshape(-1).astype(int)

if tracks.ndim == 4 and tracks.shape[0] == 1:
    tracks = tracks[0]

if tracks.ndim != 3 or tracks.shape[-1] != 2:
    raise ValueError(f"Unexpected tracks shape: {tracks.shape}")

# Convert [N,T,2] to [T,N,2] where applicable
if tracks.shape[0] == len(parts) and tracks.shape[1] != len(parts):
    tracks = tracks.transpose(1, 0, 2)

T, N, _ = tracks.shape
if N != len(parts):
    raise ValueError(
        f"Track/Part mismatch: {tracks.shape}, {parts.shape}"
    )

if visible is None:
    visible = np.ones((T, N), dtype=bool)
else:
    visible = np.asarray(visible)
    if visible.ndim == 3 and visible.shape[0] == 1:
        visible = visible[0]
    if visible.shape == (N, T) and N != T:
        visible = visible.T
    if visible.shape != (T, N):
        raise ValueError(f"Unexpected visibility: {visible.shape}")
    visible = visible > 0.5

ids = sorted(set(parts.tolist()))
colors = {}

for i, pid in enumerate(ids):
    hue = i / max(1, len(ids))
    rgb = colorsys.hsv_to_rgb(hue, 0.9, 1.0)
    colors[pid] = tuple(int(c * 255) for c in rgb[::-1])

print("Part IDs:", ids)
print("Tracks:", N)
print("Frames:", T)

cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    raise RuntimeError("Cannot open RGB video")

fps = cap.get(cv2.CAP_PROP_FPS) or 30
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)

writer = cv2.VideoWriter(
    str(output),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

if not writer.isOpened():
    raise RuntimeError("Cannot open output video")

written = 0

for t in range(T):
    ok, frame = cap.read()
    if not ok:
        break

    for j in range(N):
        if not visible[t, j]:
            continue

        xy = tracks[t, j]
        if not np.isfinite(xy).all():
            continue

        x, y = np.rint(xy).astype(int)

        if not (0 <= x < w and 0 <= y < h):
            continue

        color = colors[int(parts[j])]

        # Draw recent trajectory
        for k in range(max(1, t - 8), t + 1):
            if not (visible[k - 1, j] and visible[k, j]):
                continue

            a = tracks[k - 1, j]
            b = tracks[k, j]

            if not (np.isfinite(a).all() and np.isfinite(b).all()):
                continue

            cv2.line(
                frame,
                tuple(np.rint(a).astype(int)),
                tuple(np.rint(b).astype(int)),
                color, 1, cv2.LINE_AA
            )

        cv2.circle(
            frame, (int(x), int(y)),
            2, color, -1, cv2.LINE_AA
        )

    # Legend
    for i, pid in enumerate(ids):
        lx = 12 + (i % 5) * 120
        ly = 24 + (i // 5) * 26

        cv2.circle(frame, (lx, ly), 6, colors[pid], -1)

        cv2.putText(
            frame, f"Part {pid}",
            (lx + 12, ly + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (255, 255, 255),
            1, cv2.LINE_AA
        )

    writer.write(frame)
    written += 1

cap.release()
writer.release()

print("Rendered frames:", written)
print("Saved:", output)
