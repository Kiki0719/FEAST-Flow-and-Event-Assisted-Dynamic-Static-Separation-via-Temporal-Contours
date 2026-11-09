"""Initialize part-wise world-space SE(3) from CoTracker tracks and first-frame G0 depth."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from FEAST.core.gaussian_part_tracking import (
    GaussianPartRenderer,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--g0", required=True)
    parser.add_argument("--binding", required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--tracking", required=True)
    parser.add_argument("--moving-parts", default="1,2")
    parser.add_argument("--output", required=True)
    parser.add_argument("--reprojection-px", type=float, default=4.0)
    parser.add_argument("--min-inliers", type=int, default=8)
    args = parser.parse_args()

    with np.load(args.binding, allow_pickle=False) as d:
        labels = d["labels"].astype(np.int32)

    with np.load(args.tracking, allow_pickle=False) as d:
        tracks = d["tracks"].astype(np.float64)
        visibility = d["raw_visibility"].astype(bool)
        point_parts = d["point_parts"].astype(np.int32)

    n_frames, n_points, _ = tracks.shape
    n_parts = int(labels.max()) + 1
    moving = [
        int(p) for p in args.moving_parts.split(",") if p.strip()
    ]

    assert visibility.shape == (n_frames, n_points)
    assert point_parts.shape == (n_points,)

    renderer = GaussianPartRenderer(
        args.g0, labels, args.camera,
        width=640, height=640
    )

    # First-frame canonical geometry only.
    identity = np.tile(
        np.eye(4, dtype=np.float64), (n_parts, 1, 1)
    )

    _, depth, part_map, _ = renderer.snapshot(identity)

    depth = np.asarray(depth)
    part_map = np.asarray(part_map)

    if depth.shape != (640, 640):
        raise ValueError(f"Unexpected depth shape: {depth.shape}")

    if part_map.shape != (640, 640):
        raise ValueError(f"Unexpected Part map shape: {part_map.shape}")

    delta = np.tile(
        np.eye(4, dtype=np.float64),
        (n_frames, n_parts, 1, 1)
    )

    reliable = np.zeros((n_frames, n_parts), dtype=bool)
    inlier_counts = np.zeros((n_frames, n_parts), dtype=np.int32)
    median_errors = np.full(
        (n_frames, n_parts), np.nan, dtype=np.float64
    )

    all_rows = []

    for pid in moving:
        ids = np.flatnonzero(point_parts == pid)

        if len(ids) < args.min_inliers:
            raise RuntimeError(
                f"Part {pid}: insufficient CoTracker points ({len(ids)})"
            )

        uv0 = tracks[0, ids]
        xy = np.rint(uv0).astype(np.int32)

        inside = (
            visibility[0, ids]
            & np.isfinite(uv0).all(axis=1)
            & (xy[:, 0] >= 0)
            & (xy[:, 0] < 640)
            & (xy[:, 1] >= 0)
            & (xy[:, 1] < 640)
        )

        local_ids = ids[inside]
        uv0 = uv0[inside]
        xy = xy[inside]

        z = depth[xy[:, 1], xy[:, 0]]
        owner = part_map[xy[:, 1], xy[:, 0]]

        valid = (
            np.isfinite(z)
            & (z > 0)
            & (owner == pid)
        )

        local_ids = local_ids[valid]
        uv0 = uv0[valid]
        z = z[valid]

        if len(local_ids) < args.min_inliers:
            raise RuntimeError(
                f"Part {pid}: only {len(local_ids)} first-frame "
                "depth/Part-consistent points"
            )

        xyz0 = renderer.unproject(uv0, z)

        print(
            f"\nPart {pid}: {len(local_ids)} valid 3D-2D seeds",
            flush=True
        )

        reliable[0, pid] = True
        inlier_counts[0, pid] = len(local_ids)
        median_errors[0, pid] = 0.0

        previous = np.eye(4, dtype=np.float64)

        for t in range(1, n_frames):
            uv = tracks[t, local_ids]

            valid = (
                visibility[t, local_ids]
                & np.isfinite(uv).all(axis=1)
                & (uv[:, 0] >= 0)
                & (uv[:, 0] < 640)
                & (uv[:, 1] >= 0)
                & (uv[:, 1] < 640)
            )

            obj = np.ascontiguousarray(
                xyz0[valid], dtype=np.float64
            )
            img = np.ascontiguousarray(
                uv[valid], dtype=np.float64
            )

            accepted = False
            count = 0
            median_error = None

            if len(obj) >= args.min_inliers:
                try:
                    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                        obj,
                        img,
                        renderer.K,
                        renderer.dist,
                        iterationsCount=300,
                        reprojectionError=args.reprojection_px,
                        confidence=0.999,
                        flags=cv2.SOLVEPNP_EPNP,
                    )

                    if (
                        ok
                        and inliers is not None
                        and len(inliers) >= args.min_inliers
                    ):
                        idx = inliers.reshape(-1)

                        rvec, tvec = cv2.solvePnPRefineLM(
                            obj[idx],
                            img[idx],
                            renderer.K,
                            renderer.dist,
                            rvec,
                            tvec,
                        )

                        projected, _ = cv2.projectPoints(
                            obj, rvec, tvec,
                            renderer.K, renderer.dist
                        )

                        residuals = np.linalg.norm(
                            projected.reshape(-1, 2) - img,
                            axis=1
                        )

                        support = (
                            np.isfinite(residuals)
                            & (residuals <= args.reprojection_px)
                        )

                        count = int(support.sum())

                        if count >= args.min_inliers:
                            median_error = float(
                                np.median(residuals[support])
                            )

                            rotation, _ = cv2.Rodrigues(rvec)

                            obj_to_cv = np.eye(4, dtype=np.float64)
                            obj_to_cv[:3, :3] = rotation
                            obj_to_cv[:3, 3] = tvec.reshape(3)

                            # OpenCV camera coordinates -> G0 world coordinates.
                            pose = renderer.c2w_cv @ obj_to_cv

                            if np.isfinite(pose).all():
                                previous = pose
                                accepted = True

                except cv2.error:
                    pass

            # Hold the previous estimate when PnP is unreliable.
            # Such frames remain marked unreliable.
            delta[t, pid] = previous
            reliable[t, pid] = accepted
            inlier_counts[t, pid] = count

            if median_error is not None:
                median_errors[t, pid] = median_error

            all_rows.append({
                "frame": t,
                "part": pid,
                "reliable": bool(accepted),
                "inliers": count,
                "median_reprojection_px": median_error,
            })

            if t % 10 == 0 or t == n_frames - 1:
                print(
                    f"  Frame {t:03d}: "
                    f"inliers={count} "
                    f"reliable={accepted} "
                    f"reprojection={median_error}",
                    flush=True
                )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output,
        delta_g0=delta.astype(np.float32),
        reliable=reliable,
        inlier_counts=inlier_counts,
        median_reprojection_px=median_errors,
        moving_parts=np.asarray(moving, dtype=np.int32),
    )

    diagnostic = output.with_suffix(".diagnostics.json")
    diagnostic.write_text(
        json.dumps(all_rows, indent=2)
    )

    print("\n========== INITIALIZATION COMPLETE ==========")
    for pid in moving:
        n_ok = int(reliable[:, pid].sum())
        print(f"Part {pid}: reliable={n_ok}/{n_frames}")

    print("SE3:", output)
    print("Diagnostics:", diagnostic)


if __name__ == "__main__":
    main()
