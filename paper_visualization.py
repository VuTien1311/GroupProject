"""Original | overlay | relative-depth presentation; no model inference.

The raw Depth Anything output is used once, after inference. Its higher raw
values indicate nearer structure in this pipeline. Colors are normalized per
image and must never be interpreted as meters or compared across images.
"""
from pathlib import Path

import cv2
import numpy as np


def colorize_depth(raw_depth):
    values = np.asarray(raw_depth, dtype=np.float32)
    if values.ndim != 2 or 0 in values.shape:
        raise ValueError("Depth must be a non-empty HxW map.")
    valid = np.isfinite(values) & (values > 0)
    normalized = np.zeros(values.shape, dtype=np.float32)
    if valid.any():
        low, high = np.percentile(values[valid], [2, 98])
        if float(high - low) > 1e-6:
            normalized[valid] = np.clip((values[valid] - low) / (high - low), 0, 1)
        else:
            normalized[valid] = 0.5
    else:
        low = high = None
    view = cv2.applyColorMap(np.rint(normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    view[~valid] = (24, 24, 24)
    info = {
        "normalization": "per-image positive finite raw depth, percentiles 2/98",
        "percentile_2": float(low) if low is not None else None,
        "percentile_98": float(high) if high is not None else None,
        "valid_fraction": float(valid.mean()),
        "meaning": "Warm colors: relatively nearer; cool colors: relatively farther. Not meters.",
        "invalid_color_bgr": [24, 24, 24],
    }
    return view, info


def make_comparison(original, overlay, depth_view, panel_width=640):
    frames = [np.asarray(image) for image in (original, overlay, depth_view)]
    for image in frames:
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("Each panel must be a uint8 HxWx3 BGR image.")
        if 0 in image.shape[:2]:
            raise ValueError("Panel images cannot be empty.")
    if any(image.shape != frames[0].shape for image in frames[1:]):
        raise ValueError("Original, overlay and depth must have matching source geometry.")
    if panel_width < 320:
        raise ValueError("panel_width must be at least 320.")

    source_h, source_w = frames[0].shape[:2]
    panel_height = max(1, round(source_h * panel_width / source_w))
    header, footer, gap = 52, 84, 12
    canvas = np.full((header + panel_height + footer, 3 * panel_width + 4 * gap, 3), 20, np.uint8)
    headings = ("ORIGINAL", "OVERLAY", "RELATIVE DEPTH")
    captions = ("Unmodified input image", "Objects + road/lane + advisory", "Per-image depth colors; NOT meters")
    for index, image in enumerate(frames):
        left = gap + index * (panel_width + gap)
        resized = cv2.resize(image, (panel_width, panel_height), interpolation=cv2.INTER_AREA if panel_width < source_w else cv2.INTER_LINEAR)
        canvas[header:header + panel_height, left:left + panel_width] = resized
        cv2.putText(canvas, headings[index], (left + 10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (235, 235, 235), 2, cv2.LINE_AA)
        cv2.putText(canvas, captions[index], (left + 8, header + panel_height + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (195, 195, 195), 1, cv2.LINE_AA)

    depth_left = gap + 2 * (panel_width + gap)
    bar_width = panel_width - 16
    gradient = np.tile(np.linspace(0, 255, bar_width).astype(np.uint8), (10, 1))
    legend = cv2.applyColorMap(gradient, cv2.COLORMAP_TURBO)
    bar_y = header + panel_height + 34
    canvas[bar_y:bar_y + 10, depth_left + 8:depth_left + 8 + bar_width] = legend
    text_y = header + panel_height + 67
    cv2.putText(canvas, "FAR", (depth_left + 8, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (215, 215, 215), 1, cv2.LINE_AA)
    cv2.putText(canvas, "NEAR", (depth_left + panel_width - 52, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (215, 215, 215), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Static-image research output; no vehicle control.", (gap + 8, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (165, 165, 165), 1, cv2.LINE_AA)
    return canvas


def write_png(path, image):
    path = Path(path)
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise OSError(f"Failed to encode image: {path}")
    # tofile handles Unicode paths on Windows, unlike some cv2.imwrite builds.
    encoded.tofile(str(path))
    if not path.is_file() or path.stat().st_size == 0:
        raise OSError(f"Failed to write image: {path}")


def save_visualizations(original, overlay, raw_depth, output_dir, stem):
    depth_view, normalization = colorize_depth(raw_depth)
    if depth_view.shape != original.shape:
        raise ValueError("Raw depth must match the input image resolution.")
    comparison = make_comparison(original, overlay, depth_view)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    files = {
        "original": output / f"{stem}_original.png",
        "overlay": output / f"{stem}_overlay.png",
        "relative_depth": output / f"{stem}_relative_depth.png",
        "comparison": output / f"{stem}_comparison.png",
        "raw_depth": output / f"{stem}_raw_depth.npy",
    }
    for key, frame in (("original", original), ("overlay", overlay), ("relative_depth", depth_view), ("comparison", comparison)):
        write_png(files[key], frame)
    np.save(str(files["raw_depth"]), np.asarray(raw_depth), allow_pickle=False)
    return {"files": {key: str(path.resolve()) for key, path in files.items()}, "depth_display": normalization, "layout": "original | overlay | relative depth"}
