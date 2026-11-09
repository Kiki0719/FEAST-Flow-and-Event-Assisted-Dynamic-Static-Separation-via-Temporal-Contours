import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--tracking", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--trail", type=int, default=8)
    args = ap.parse_args()

    with np.load(args.tracking, allow_pickle=False) as d:
        tracks = d["tracks"].astype(np.float32)
        vis = d["raw_visibility"].astype(bool)
        pids = d["point_parts"].astype(np.int32)

    T, N, _ = tracks.shape

    colors = {
        int(pid): tuple(
            int(v) for v in cv2.cvtColor(
                np.uint8([[[int(pid) * 137 % 180, 220, 255]]]),
                cv2.COLOR_HSV2BGR
            )[0, 0]
        )
        for pid in np.unique(pids)
    }

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    W = 640
    H = 640

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Cannot create output video")

    try:
        for t in range(T):
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.resize(frame, (W, H))

            for i in range(N):
                color = colors.get(int(pids[i]), (255, 255, 255))

                # Trajectory segments: only connect visible observations.
                start = max(0, t - args.trail)

                for k in range(start + 1, t + 1):
                    if not (vis[k - 1, i] and vis[k, i]):
                        continue

                    a = tracks[k - 1, i]
                    b = tracks[k, i]

                    if not (
                        np.isfinite(a).all()
                        and np.isfinite(b).all()
                    ):
                        continue

                    pa = tuple(np.rint(a).astype(int))
                    pb = tuple(np.rint(b).astype(int))

                    cv2.line(
                        frame, pa, pb,
                        color, 1, cv2.LINE_AA
                    )

                if not vis[t, i]:
                    continue

                xy = tracks[t, i]
                if not np.isfinite(xy).all():
                    continue

                x, y = np.rint(xy).astype(int)

                if 0 <= x < W and 0 <= y < H:
                    # Subpixel-style point: approximately 0.5 px radius
                    overlay = frame.copy()
                    cv2.circle(
                        overlay, (int(x), int(y)),
                        1, color, -1, cv2.LINE_AA
                    )
                    cv2.addWeighted(
                        overlay, 0.5, frame, 0.5, 0, dst=frame
                    )

            cv2.putText(
                frame,
                f"RAW CoTracker3 | Frame {t:03d}",
                (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            writer.write(frame)

            if t % 10 == 0:
                print(f"Rendered {t + 1}/{T}", flush=True)

    finally:
        cap.release()
        writer.release()

    print("Saved:", out.resolve())


if __name__ == "__main__":
    main()
