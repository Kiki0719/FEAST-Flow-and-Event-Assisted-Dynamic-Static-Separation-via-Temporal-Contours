
import json
from pathlib import Path

import cv2
import numpy as np



def part_color(pid):
    """Deterministic colors, unique for Part IDs 0..1079."""
    pid = int(pid)
    hue = (pid * 73) % 180
    band = (pid // 180) % 6
    saturation = 155 + (band % 3) * 40
    value = 210 + (band // 3) * 40
    hsv = np.uint8([[[hue, saturation, value]]])
    return tuple(
        int(x) for x in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    )


def _extract_part_names(metadata):
    """Read explicit Blender Part ID/name mappings when available."""
    result = {}

    if not isinstance(metadata, dict):
        return result

    # Blender exporter stores object names in Part ID order.
    objects = metadata.get("objects")
    if isinstance(objects, list) and all(
        isinstance(name, str) for name in objects
    ):
        for pid, name in enumerate(objects):
            if name:
                result[pid] = name

    for key in ("part_names", "object_names"):
        value = metadata.get(key)

        if isinstance(value, list):
            for pid, name in enumerate(value):
                if isinstance(name, str) and name:
                    result[pid] = name

        elif isinstance(value, dict):
            for pid, name in value.items():
                if str(pid).lstrip("-").isdigit() and isinstance(name, str):
                    result[int(pid)] = name

    for key in ("parts", "part_info", "part_metadata", "objects"):
        records = metadata.get(key)

        if isinstance(records, dict):
            records = list(records.values())

        if not isinstance(records, list):
            continue

        for record in records:
            if not isinstance(record, dict):
                continue

            pid = record.get(
                "part_id",
                record.get("id", record.get("part_index"))
            )
            name = record.get(
                "name",
                record.get("object_name", record.get("part_name"))
            )

            if pid is not None and str(pid).lstrip("-").isdigit():
                if isinstance(name, str) and name:
                    result[int(pid)] = name

    return result


def _load_blender_names(output_dir):
    """Locate Blender metadata without changing the Stage B interface."""
    output_dir = Path(output_dir).resolve()

    for parent in [output_dir, *output_dir.parents]:
        if parent.name != "outputs":
            continue

        # outputs/Lego/rgb_part_motion_se3_flow/confirmation_xxx
        relative = output_dir.relative_to(parent)
        if not relative.parts:
            continue

        model = relative.parts[0]
        source = parent / f"{model}-init" / "blender-surface-parts.npz"

        if not source.exists():
            continue

        try:
            with np.load(source, allow_pickle=False) as data:
                if "metadata_json" not in data:
                    continue
                metadata = json.loads(str(data["metadata_json"].item()))

            names = _extract_part_names(metadata)
            print(
                f"Blender metadata: {source} | "
                f"Resolved names: {len(names)}",
                flush=True
            )
            return names

        except Exception as exc:
            print(f"Warning: cannot parse Blender metadata: {exc}")

    return {}


def _event_overlay(rgb, positive, negative):
    """Draw red/blue event pixels and contours over the original RGB."""
    output = rgb.copy()

    red = np.array([0, 0, 255], dtype=np.float32)
    blue = np.array([255, 0, 0], dtype=np.float32)

    # Blend event pixels with the underlying RGB.
    for mask, color in ((positive, red), (negative, blue)):
        if np.any(mask):
            output[mask] = (
                output[mask].astype(np.float32) * 0.25
                + color * 0.75
            ).astype(np.uint8)

    # Explicit red/blue outlines.
    for mask, color in (
        (positive, (0, 0, 255)),
        (negative, (255, 0, 0))
    ):
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(
            output, contours, -1, color, 1, cv2.LINE_AA
        )

    return output


def _safe_filename(name):
    import re
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")[:80]

def _title(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 38),
                  (20, 20, 20), -1)
    cv2.putText(
        img, text, (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.63, (255, 255, 255), 1, cv2.LINE_AA
    )
    return img


def _make_maps(pm, uv, labels, height, width):
    """
    Prefer Stage B's actual pixel-to-Part map.
    Fall back to projected Gaussian centers if needed.
    """
    arr = np.asarray(pm)
    unique_parts = np.unique(labels.astype(np.int32))
    unique_parts = unique_parts[unique_parts >= 0]

    if arr.ndim == 2 and arr.shape == (height, width):
        return {
            int(p): arr == p
            for p in unique_parts
        }, "pixel_part_map"

    if arr.ndim == 3:
        if arr.shape[1:] == (height, width):
            return {
                int(p): arr[int(p)] > 0
                for p in unique_parts
                if int(p) < arr.shape[0]
            }, "part_mask_stack"

        if arr.shape[:2] == (height, width):
            return {
                int(p): arr[:, :, int(p)] > 0
                for p in unique_parts
                if int(p) < arr.shape[2]
            }, "part_mask_stack"

    # Fallback: Gaussian projected-center occupancy.
    # This is not a visibility-resolved Part mask.
    uv = np.asarray(uv)
    labels = np.asarray(labels)

    if uv.ndim != 2 or uv.shape[1] < 2:
        raise ValueError(
            f"Cannot visualize Part assignment: pm={arr.shape}, "
            f"uv={uv.shape}"
        )

    result = {}
    for p in unique_parts:
        mask = np.zeros((height, width), dtype=np.uint8)
        xy = uv[labels == p, :2]
        xy = xy[np.isfinite(xy).all(axis=1)]
        xy = np.rint(xy).astype(np.int32)
        inside = (
            (xy[:, 0] >= 0) & (xy[:, 0] < width) &
            (xy[:, 1] >= 0) & (xy[:, 1] < height)
        )
        xy = xy[inside]
        mask[xy[:, 1], xy[:, 0]] = 1
        result[int(p)] = mask.astype(bool)

    return result, "projected_centers_fallback"


def save_event_part_visualization(
    output_dir, frame, pos, neg, pm, uv, labels,
    event_candidates, rows
):
    out = Path(output_dir) / "event_part_debug"
    out.mkdir(parents=True, exist_ok=True)

    rgb = np.asarray(frame).copy()
    h, w = rgb.shape[:2]

    pos = np.asarray(pos)
    neg = np.asarray(neg)

    if pos.shape != (h, w) or neg.shape != (h, w):
        raise ValueError(
            f"Event map shape mismatch: "
            f"pos={pos.shape}, neg={neg.shape}, image={(h,w)}"
        )

    positive = pos > 0
    negative = neg > 0
    active = positive | negative

    masks, source = _make_maps(pm, uv, labels, h, w)
    blender_part_names = _load_blender_names(output_dir)

    # Events over the original RGB: red positive, blue negative.
    event_overlay = _event_overlay(rgb, positive, negative)

    # Part-assignment color image.
    part_map = np.zeros_like(rgb)
    for pid, mask in masks.items():
        part_map[mask] = part_color(pid)

    # Overlay Part boundaries on original RGB.
    boundaries = rgb.copy()
    for pid, mask in masks.items():
        if not np.any(mask):
            continue
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(
            boundaries, contours, -1,
            part_color(pid),
            1
        )

    # Side-by-side summary.
    a = _title(rgb.copy(), "Original RGB")
    b = _title(event_overlay, "Events: red + / blue -")
    c = _title(part_map, "Gaussian Part assignment")
    d = _title(boundaries, "Projected Part boundaries")

    summary = np.vstack([
        np.hstack([a, b]),
        np.hstack([c, d])
    ])
    cv2.imwrite(str(out / "event_part_overview.png"), summary)

    stats = []

    for pid, mask in sorted(masks.items()):
        color = part_color(pid)
        name = blender_part_names.get(int(pid), f"Part {pid}")

        view = rgb.copy()

        # Darken pixels outside this Part.
        view[~mask] = (
            view[~mask].astype(np.float32) * 0.25
        ).astype(np.uint8)

        # All events inside the Part.
        p_hit = mask & positive
        n_hit = mask & negative
        view = _event_overlay(view, p_hit, n_hit)

        # Draw Part boundary.
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(
            view, contours, -1, color, 1
        )

        count_pixels = int(mask.sum())
        event_pixels = int((mask & active).sum())
        positive_pixels = int(p_hit.sum())
        negative_pixels = int(n_hit.sum())
        density = event_pixels / max(count_pixels, 1)

        status = (
            "EVENT CANDIDATE"
            if pid in event_candidates
            else "NOT CANDIDATE"
        )

        cv2.rectangle(
            view, (0, 0), (w, 90), (15, 15, 15), -1
        )
        cv2.putText(
            view, f"Part {pid} / {name}: {status}",
            (12, 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.63, (255, 255, 255), 1, cv2.LINE_AA
        )
        cv2.putText(
            view,
            f"Active pixels: {event_pixels} / {count_pixels}",
            (12, 51), cv2.FONT_HERSHEY_SIMPLEX,
            0.53, (255, 255, 255), 1, cv2.LINE_AA
        )
        cv2.putText(
            view, f"Event pixel density: {density:.2%}",
            (12, 76), cv2.FONT_HERSHEY_SIMPLEX,
            0.53, (255, 255, 255), 1, cv2.LINE_AA
        )

        cv2.imwrite(
            str(out / f"part_{pid}_{_safe_filename(name)}.png"),
            view
        )

        stats.append({
            "part": int(pid),
            "name": name,
            "is_event_candidate": pid in event_candidates,
            "mask_pixels": count_pixels,
            "active_event_pixels": event_pixels,
            "positive_event_pixels": positive_pixels,
            "negative_event_pixels": negative_pixels,
            "event_pixel_density": density,
        })

    (out / "event_debug_statistics.json").write_text(
        json.dumps({
            "mask_source": source,
            "part_name_source": "Blender metadata",
            "resolved_blender_names": len(blender_part_names),
            "note": (
                "These are active-event pixel overlaps, "
                "not independent proof of Part motion. "
                "Scores may use different rules."
            ),
            "original_candidate_scores": rows,
            "part_overlap_statistics": stats,
        }, indent=2, ensure_ascii=False, default=str)
    )

    print(
        "Event-Part visualization saved:",
        out.resolve(),
        flush=True
    )


