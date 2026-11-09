#!/usr/bin/env python3
"""Original MERCY SEA-RAFT HSV flow + white-arrow visualization."""

import argparse
from pathlib import Path

import cv2
import numpy as np


def visualize_flow(frame, flow, frame_index, stride=12):
    h, w = frame.shape[:2]

    uv = np.asarray(flow, dtype=np.float32)

    if uv.shape != (h, w, 2):
        raise ValueError(
            f"Flow shape mismatch: {uv.shape}, expected {(h, w, 2)}"
        )

    # Original MERCY HSV optical-flow visualization.
    mag = np.linalg.norm(uv, axis=2)
    ang = np.arctan2(uv[:, :, 1], uv[:, :, 0])

    hsv = np.zeros((h, w, 3), np.uint8)

    hsv[:, :, 0] = (
        ((ang + np.pi) / (2 * np.pi)) * 179
    ).astype(np.uint8)

    hsv[:, :, 1] = 255

    finite = np.isfinite(mag)

    scale = (
        max(1.0, float(np.percentile(mag[finite], 95)))
        if finite.any() else 1.0
    )

    hsv[:, :, 2] = np.clip(
        np.nan_to_num(mag, nan=0.0, posinf=0.0, neginf=0.0)
        / scale * 255,
        0,
        255
    ).astype(np.uint8)

    color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Original: 40% RGB + 60% HSV flow.
    blended = cv2.addWeighted(
        frame, 0.4,
        color, 0.6,
        0
    )

    # Original fine white motion-vector arrows.
    for y in range(stride // 2, h, stride):
        for x in range(stride // 2, w, stride):
            vx, vy = uv[y, x]

            if not (np.isfinite(vx) and np.isfinite(vy)):
                continue

            endpoint = (
                int(round(x + np.clip(vx, -stride, stride))),
                int(round(y + np.clip(vy, -stride, stride)))
            )

            cv2.arrowedLine(
                blended,
                (x, y),
                endpoint,
                (230, 230, 230),
                1,
                tipLength=0.25
            )

    cv2.putText(
        blended,
        f"SEA-RAFT | F{frame_index:03d}->{frame_index + 1:03d}",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    return blended


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", required=True)
    parser.add_argument("--flow", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--fps", type=float, default=15.0)

    args = parser.parse_args()

    if args.stride <= 0 or args.fps <= 0:
        parser.error("--stride and --fps must be positive")

    flow = np.load(args.flow, mmap_mode="r")

    if flow.ndim != 4 or flow.shape[-1] != 2:
        raise ValueError("Expected flow shape (T-1,H,W,2)")

    T, H, W, _ = flow.shape

    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        raise RuntimeError("Cannot open video: " + args.video)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (W, H)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Cannot create video: " + str(output))

    try:
        # Match the original MERCY convention:
        # Last RGB frame reuses the last forward flow.
        for t in range(T + 1):
            ok, frame = cap.read()

            if not ok:
                break

            frame = cv2.resize(frame, (W, H))

            flow_idx = min(t, T - 1)

            result = visualize_flow(
                frame,
                flow[flow_idx],
                flow_idx,
                stride=args.stride
            )

            writer.write(result)

            if t % 10 == 0:
                print(
                    f"Rendered {t + 1}/{T + 1}",
                    flush=True
                )

    finally:
        cap.release()
        writer.release()

    print("Saved:", output.resolve())


if __name__ == "__main__":
    main()
