import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

from depth_anything_v2.dpt import DepthAnythingV2
from paper_visualization import save_visualizations


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HYBRID_DIR = os.path.join(BASE_DIR, "hybrid2")
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, HYBRID_DIR)

from hybrid2.backbone import HybridNetsBackbone  # noqa: E402
from hybrid2.utils.utils import letterbox  # noqa: E402


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
YOLO_DEVICE = 0 if DEVICE == "cuda" else "cpu"
HYBRID_INPUT_SIZE = 640
TARGET_DETECTION_CLASSES = {
    "person",
    "rider",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "train",
}
DEFAULT_DETECTION_CONFIDENCE = 0.18
DEFAULT_DETECTION_IMAGE_SIZE = 960
DEFAULT_CAMERA_HEIGHT_M = 1.50
DEFAULT_HORIZONTAL_FOV_DEG = 70.0
DEFAULT_HORIZON_RATIO = 0.46
MAX_ESTIMATED_DISTANCE_M = 150.0
DISTANCE_MODES = ("auto", "metric", "relative")
ENHANCE_INPUT_MODES = ("off", "auto", "always")
YOLO_TTA_MODES = ("off", "dual", "full")
TILE_INFERENCE_MODES = ("off", "auto", "on")
PRECISION_MODES = ("recall", "balanced", "strict")


def first_existing_path(*paths):
    for path in paths:
        resolved = path if os.path.isabs(path) else os.path.join(BASE_DIR, path)
        if os.path.exists(resolved):
            return path
    return None


BEST_BDD_YOLO_WEIGHT = first_existing_path(
    os.path.join(
        "runs",
        "static_yolo",
        "bdd100k_yolo11s_hard_ft",
        "weights",
        "best.pt",
    ),
    os.path.join(
        "runs",
        "static_yolo",
        "bdd100k_yolo11s_fresh_strong",
        "weights",
        "best.pt",
    ),
    os.path.join(
        "runs",
        "static_yolo",
        "bdd100k_yolo11s_detect",
        "weights",
        "best.pt",
    ),
)
MAX_INFERENCE_CONFIG = {
    "yolo_weight": "yolo11m-seg.pt",
    "aux_yolo_weight": BEST_BDD_YOLO_WEIGHT,
    "confidence": DEFAULT_DETECTION_CONFIDENCE,
    "image_size": DEFAULT_DETECTION_IMAGE_SIZE,
    "yolo_imgszs": [960],
    "depth_size": 518,
    "enhance_input": "auto",
    "yolo_tta": "full",
    "tile_inference": "off",
    "precision_mode": "strict",
}

# Approximate real-world heights support a weak monocular geometry cross-check.
# The road-calibrated Depth Anything estimate remains the primary signal.
CLASS_HEIGHT_M = {
    "person": 1.70,
    "rider": 1.70,
    "bicycle": 1.40,
    "car": 1.50,
    "motorcycle": 1.40,
    "bus": 3.20,
    "truck": 3.20,
    "train": 3.50,
}
VULNERABLE_CLASSES = {"person", "bicycle", "motorcycle"}
VEHICLE_CLASSES = {"car", "bus", "truck", "train"}
GROUND_OBJECT_CLASSES = VEHICLE_CLASSES | {"person", "rider", "bicycle", "motorcycle"}
CLASS_ASPECT_LIMITS = {
    "person": (0.18, 1.45),
    "rider": (0.22, 1.90),
    "bicycle": (0.55, 4.80),
    "motorcycle": (0.50, 4.80),
    "car": (0.55, 6.50),
    "bus": (0.70, 8.50),
    "truck": (0.65, 8.50),
    "train": (0.70, 10.0),
}
ACTION_SEVERITY = {
    "CLEAR": 0,
    "MONITOR": 1,
    "SLOW": 2,
    "BRAKE": 3,
    "STOP": 4,
}
ACTION_COLORS = {
    "CLEAR": (0, 210, 0),
    "MONITOR": (0, 220, 220),
    "SLOW": (0, 165, 255),
    "BRAKE": (0, 80, 255),
    "STOP": (0, 0, 255),
}
SCENE_STATUS_COLORS = {
    "GOOD": (0, 210, 0),
    "LIMITED": (0, 210, 255),
    "DEGRADED": (0, 80, 255),
}

DEFAULT_IMAGES = [
    "b1ceb32e-3f481b43.jpg",
    "b1ceb32e-51852abe.jpg",
    "b1d0a191-2ed2269e.jpg",
    "b1d0a191-06deb55d.jpg",
    "b1d0a191-28f0e779.jpg",
    "b1d7b3ac-2a92e19f.jpg",
    "b1ee702d-525fcebf.jpg",
    "b1e1a7b8-a7426a97.jpg",
    "b1d22449-117aa773.jpg",
    "b1d3907b-2278601b.jpg",
    "b1d8735d-eee9f184.jpg",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def expand_image_inputs(inputs, limit=None):
    image_paths = []
    for item in inputs:
        path = item if os.path.isabs(item) else os.path.join(BASE_DIR, item)
        if os.path.isdir(path):
            for root, directories, files in os.walk(path):
                directories.sort()
                for filename in sorted(files):
                    if os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS:
                        image_paths.append(os.path.join(root, filename))
        elif os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS:
            image_paths.append(path)
        else:
            print(f"[INPUT] Skipped unsupported or missing path: {path}")

        if limit is not None and len(image_paths) >= limit:
            return image_paths[:limit]
    return image_paths[:limit] if limit is not None else image_paths


def load_image_list(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip() and not line.lstrip().startswith("#")]


def load_state_dict(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def blend_mask(image, mask, color, alpha):
    blended = image.copy()
    blended[mask] = (
        image[mask].astype(np.float32) * (1.0 - alpha)
        + np.asarray(color, dtype=np.float32) * alpha
    ).astype(np.uint8)
    return blended


def assess_scene_quality(image):
    """Estimate whether a static frame provides reliable visual evidence."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    p10, median, p90 = np.percentile(gray, [10, 50, 90])
    dynamic_range = float(p90 - p10)
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    glare_fraction = float(np.mean(gray >= 245))

    illumination_score = float(np.clip((median - 18.0) / 72.0, 0.0, 1.0))
    contrast_score = float(np.clip((dynamic_range - 20.0) / 80.0, 0.0, 1.0))
    sharpness_score = float(
        np.clip(
            (np.log1p(laplacian_variance) - np.log1p(12.0))
            / (np.log1p(500.0) - np.log1p(12.0)),
            0.0,
            1.0,
        )
    )
    score = float(
        0.40 * illumination_score
        + 0.30 * contrast_score
        + 0.30 * sharpness_score
    )

    flags = []
    if median < 45.0:
        flags.append("LOW_LIGHT")
    if dynamic_range < 55.0:
        flags.append("LOW_CONTRAST")
    if laplacian_variance < 60.0:
        flags.append("BLUR")
    if glare_fraction > 0.025 and median < 90.0:
        flags.append("GLARE")

    if score < 0.35 or ("LOW_LIGHT" in flags and "BLUR" in flags):
        status = "DEGRADED"
    elif score < 0.58 or flags:
        status = "LIMITED"
    else:
        status = "GOOD"
    return {
        "status": status,
        "score": score,
        "flags": flags,
        "median_brightness": float(median),
        "dynamic_range_p10_p90": dynamic_range,
        "laplacian_variance": laplacian_variance,
        "glare_fraction": glare_fraction,
    }


def apply_gamma(image, gamma):
    table = np.asarray(
        [np.clip(((value / 255.0) ** gamma) * 255.0, 0, 255) for value in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(image, table)


def apply_clahe_luma(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    luma, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_luma = clahe.apply(luma)
    return cv2.cvtColor(cv2.merge((enhanced_luma, channel_a, channel_b)), cv2.COLOR_LAB2BGR)


def apply_unsharp_mask(image, amount=0.35, sigma=1.2):
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)


def enhance_yolo_input(image, scene_quality, mode):
    """Light image enhancement for YOLO only; depth and road models keep raw input."""
    info = {
        "mode": mode,
        "applied": False,
        "operations": [],
        "reason_flags": list(scene_quality["flags"]),
    }
    if mode == "off":
        return image, info

    flags = set(scene_quality["flags"])
    should_enhance = mode == "always" or bool(flags & {"LOW_LIGHT", "LOW_CONTRAST", "BLUR"})
    if not should_enhance:
        return image, info

    enhanced = image.copy()
    median = scene_quality["median_brightness"]
    dynamic_range = scene_quality["dynamic_range_p10_p90"]

    if mode == "always" or "LOW_LIGHT" in flags:
        gamma = 0.70 if median < 35.0 else 0.82
        enhanced = apply_gamma(enhanced, gamma)
        info["operations"].append(f"gamma_{gamma:.2f}")

    if mode == "always" or "LOW_CONTRAST" in flags or dynamic_range < 70.0:
        clip_limit = 2.4 if "LOW_CONTRAST" in flags else 1.8
        enhanced = apply_clahe_luma(enhanced, clip_limit=clip_limit)
        info["operations"].append(f"clahe_luma_{clip_limit:.1f}")

    if "BLUR" in flags and "LOW_LIGHT" not in flags:
        enhanced = apply_unsharp_mask(enhanced)
        info["operations"].append("mild_unsharp")

    info["applied"] = bool(info["operations"])
    return enhanced if info["applied"] else image, info


def parse_imgszs(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        parts = value
    else:
        parts = str(value).replace(";", ",").split(",")

    sizes = []
    for part in parts:
        text = str(part).strip()
        if not text:
            continue
        size = int(text)
        if size <= 0:
            raise ValueError("YOLO image sizes must be positive.")
        sizes.append(size)
    if not sizes:
        raise ValueError("At least one YOLO image size is required.")
    return sorted(set(sizes))


def build_tiles(image_shape, tile_size=640, overlap=0.20):
    """Return overlapping crop windows for detecting small/far objects."""
    height, width = image_shape[:2]
    tile_size = int(max(256, tile_size))
    overlap = float(np.clip(overlap, 0.0, 0.65))
    if height <= tile_size and width <= tile_size:
        return []

    stride = max(64, int(tile_size * (1.0 - overlap)))
    max_x = max(width - tile_size, 0)
    max_y = max(height - tile_size, 0)
    xs = list(range(0, max_x + 1, stride))
    ys = list(range(0, max_y + 1, stride))
    if not xs or xs[-1] != max_x:
        xs.append(max_x)
    if not ys or ys[-1] != max_y:
        ys.append(max_y)

    tiles = []
    seen = set()
    for y1 in ys:
        for x1 in xs:
            x2 = min(width, x1 + tile_size)
            y2 = min(height, y1 + tile_size)
            x1 = max(0, x2 - tile_size)
            y1 = max(0, y2 - tile_size)
            window = (int(x1), int(y1), int(x2), int(y2))
            if window not in seen:
                seen.add(window)
                tiles.append(window)
    return tiles


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_intersection_area(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    return max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0,
        min(ay2, by2) - max(ay1, by1),
    )


def suppress_nested_duplicates(detections):
    """Remove likely duplicate same-class boxes that survive weighted fusion."""
    accepted = []
    suppressed = []
    for detection in sorted(detections, key=lambda item: item["confidence"], reverse=True):
        duplicate_of = None
        candidate_box = detection["box"]
        candidate_area = max(box_area(candidate_box), 1.0)
        cx = (candidate_box[0] + candidate_box[2]) * 0.5
        cy = (candidate_box[1] + candidate_box[3]) * 0.5
        for kept in accepted:
            if kept["class_name"] != detection["class_name"]:
                continue
            kept_box = kept["box"]
            kept_area = max(box_area(kept_box), 1.0)
            intersection = box_intersection_area(candidate_box, kept_box)
            smaller_containment = intersection / min(candidate_area, kept_area)
            iou = box_iou(candidate_box, kept_box)
            kept_cx = (kept_box[0] + kept_box[2]) * 0.5
            kept_cy = (kept_box[1] + kept_box[3]) * 0.5
            center_dx = abs(cx - kept_cx) / max(abs(kept_box[2] - kept_box[0]), 1.0)
            center_dy = abs(cy - kept_cy) / max(abs(kept_box[3] - kept_box[1]), 1.0)
            nested_same_object = (
                smaller_containment >= 0.86
                and center_dx <= 0.38
                and center_dy <= 0.42
            )
            if iou >= 0.46 or nested_same_object:
                duplicate_of = kept
                break
        if duplicate_of is None:
            accepted.append(detection)
        else:
            suppressed.append(
                {
                    "class": detection["class_name"],
                    "confidence": round(float(detection["confidence"]), 4),
                    "merged_count": detection.get("merged_count", 1),
                    "detector_sources": detection.get("detector_sources", []),
                    "duplicate_of_confidence": round(float(duplicate_of["confidence"]), 4),
                    "reasons": ["DUPLICATE_NESTED_BOX"],
                }
            )
    return accepted, suppressed


def weighted_merge_detections(detections, iou_threshold=0.55):
    """Class-aware weighted-box fusion for detections from multiple YOLO passes."""
    if not detections:
        return []

    pending = sorted(detections, key=lambda item: item["confidence"], reverse=True)
    merged = []
    while pending:
        seed = pending.pop(0)
        cluster = [seed]
        remaining = []
        for candidate in pending:
            if (
                candidate["class_name"] == seed["class_name"]
                and box_iou(candidate["box"], seed["box"]) >= iou_threshold
            ):
                cluster.append(candidate)
            else:
                remaining.append(candidate)
        pending = remaining

        weights = np.asarray([item["confidence"] for item in cluster], dtype=np.float64)
        boxes = np.asarray([item["box"] for item in cluster], dtype=np.float64)
        fused_box = np.average(boxes, axis=0, weights=np.maximum(weights, 1e-3))
        best = max(cluster, key=lambda item: (item["mask"] is not None, item["confidence"]))
        fused = dict(best)
        fused["box"] = fused_box
        support_bonus = min(0.045, 0.012 * (len(cluster) - 1))
        fused["confidence"] = float(
            min(
                0.99,
                max(weights) * 0.88 + float(np.mean(weights)) * 0.12 + support_bonus,
            )
        )
        fused["raw_confidence_max"] = float(max(weights))
        fused["raw_confidence_mean"] = float(np.mean(weights))
        fused["detector_sources"] = sorted(
            {
                f"{item['detector']}:{item['variant']}"
                for item in cluster
            }
        )
        fused["detector_roles"] = sorted({item["detector"] for item in cluster})
        fused["merged_count"] = len(cluster)
        merged.append(fused)
    return sorted(merged, key=lambda item: item["confidence"], reverse=True)


def adaptive_detection_confidence(base_confidence, scene_quality):
    flags = set(scene_quality["flags"])
    if scene_quality["status"] == "DEGRADED":
        return max(0.14, base_confidence * 0.82)
    if flags & {"LOW_LIGHT", "LOW_CONTRAST", "BLUR"}:
        return max(0.15, base_confidence * 0.90)
    if scene_quality["status"] == "LIMITED":
        return max(0.16, base_confidence * 0.94)
    return base_confidence


def filter_weak_yolo_detections(
    detections,
    detection_confidence,
    support_passes,
    precision_mode="balanced",
):
    """Drop weak one-off detections created by aggressive ensembles/TTA."""
    if support_passes <= 1:
        return detections, []

    if precision_mode == "strict":
        min_single_pass_conf = max(0.28, detection_confidence + 0.08)
        min_any_conf = max(0.18, detection_confidence * 0.96)
    elif precision_mode == "recall":
        min_single_pass_conf = max(0.14, detection_confidence + 0.02)
        min_any_conf = max(0.11, detection_confidence * 0.75)
    else:
        min_single_pass_conf = max(0.20, detection_confidence + 0.05)
        min_any_conf = max(0.15, detection_confidence * 0.88)
    filtered = []
    suppressed = []
    for detection in detections:
        detector_roles = set(detection.get("detector_roles", [])) or {
            source.split(":", 1)[0] for source in detection.get("detector_sources", [])
        }
        cross_detector = len(detector_roles) >= 2
        single_pass = detection.get("merged_count", 1) <= 1
        confidence = float(detection["confidence"])

        reasons = []
        if confidence < min_any_conf and not cross_detector:
            reasons.append("BELOW_MIN_SUPPORTED_CONFIDENCE")
        if single_pass and confidence < min_single_pass_conf and not cross_detector:
            reasons.append("LOW_CONFIDENCE_SINGLE_PASS")
        if (
            precision_mode == "strict"
            and single_pass
            and confidence < 0.34
            and not cross_detector
        ):
            reasons.append("STRICT_SINGLE_PASS_REJECT")

        if reasons:
            suppressed.append(
                {
                    "class": detection["class_name"],
                    "confidence": round(confidence, 4),
                    "raw_confidence_max": round(float(detection.get("raw_confidence_max", confidence)), 4),
                    "merged_count": detection.get("merged_count", 1),
                    "detector_sources": detection.get("detector_sources", []),
                    "reasons": reasons,
                }
            )
        else:
            filtered.append(detection)
    return filtered, suppressed


def ground_support_score(support_mask, box, image_shape):
    height, width = image_shape[:2]
    x1, y1, x2, y2 = box
    box_w = max(x2 - x1, 1)
    box_h = max(y2 - y1, 1)
    pad_x = max(2, int(box_w * 0.18))
    pad_y = max(3, int(box_h * 0.16))
    sx1 = int(np.clip(x1 - pad_x, 0, width - 1))
    sx2 = int(np.clip(x2 + pad_x, sx1 + 1, width))
    sy1 = int(np.clip(y2 - pad_y, 0, height - 1))
    sy2 = int(np.clip(y2 + pad_y * 2, sy1 + 1, height))
    region = support_mask[sy1:sy2, sx1:sx2]
    if region.size == 0:
        return 0.0
    return float(region.mean())


def class_geometry_reasons(
    class_name,
    box,
    image_shape,
    horizon_y,
    confidence,
    mask_source,
    mask_coverage,
    ground_score,
    relation,
    detector_roles,
    precision_mode,
):
    height, width = image_shape[:2]
    x1, y1, x2, y2 = box
    box_w = max(x2 - x1, 1)
    box_h = max(y2 - y1, 1)
    box_area_ratio = (box_w * box_h) / max(height * width, 1)
    aspect = box_w / box_h
    cross_detector = len(detector_roles) >= 2
    strict = precision_mode == "strict"
    reasons = []

    low_aspect, high_aspect = CLASS_ASPECT_LIMITS.get(class_name, (0.15, 12.0))
    if (aspect < low_aspect or aspect > high_aspect) and confidence < (0.62 if strict else 0.52):
        reasons.append("IMPLAUSIBLE_CLASS_ASPECT_RATIO")

    if (
        class_name in GROUND_OBJECT_CLASSES
        and y2 < horizon_y - height * 0.035
        and confidence < (0.52 if strict else 0.42)
        and not cross_detector
    ):
        reasons.append("ABOVE_HORIZON_LOW_CONFIDENCE")

    if (
        class_name in VEHICLE_CLASSES
        and ground_score < (0.010 if strict else 0.004)
        and relation == "SIDE"
        and confidence < (0.42 if strict else 0.32)
        and not cross_detector
    ):
        reasons.append("NO_ROAD_OR_LANE_SUPPORT")

    if (
        mask_source == "segmentation"
        and mask_coverage < (0.035 if strict else 0.020)
        and confidence < (0.45 if strict else 0.34)
    ):
        reasons.append("SPARSE_SEGMENTATION_MASK")

    if (
        mask_source == "segmentation"
        and mask_coverage > 0.94
        and box_area_ratio > 0.035
        and confidence < (0.36 if strict else 0.28)
    ):
        reasons.append("BOX_SHAPED_LOW_CONFIDENCE_MASK")

    return reasons


def rectangle_overlap(rect_a, rect_b):
    ax1, ay1, ax2, ay2 = rect_a
    bx1, by1, bx2, by2 = rect_b
    return max(0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0,
        min(ay2, by2) - max(ay1, by1),
    )


def draw_all_detection_labels(image, annotations):
    """Draw a compact, non-hidden label for every accepted detection."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.40 if image.shape[1] < 1500 else 0.46
    thickness = 1
    height, width = image.shape[:2]
    occupied = [(8, 8, min(width - 8, 700), 162)]

    # Place important and nearby objects first, then fill in every remaining one.
    annotations = sorted(
        annotations,
        key=lambda item: (
            -ACTION_SEVERITY[item["action"]],
            item["display_sort_distance"],
            -item["box_area"],
        ),
    )
    for item in annotations:
        x1, y1, x2, y2 = item["box"]
        text = item["label"]
        color = item["color"]
        (text_w, text_h), baseline = cv2.getTextSize(
            text,
            font,
            font_scale,
            thickness,
        )
        label_w = text_w + 6
        label_h = text_h + baseline + 6
        candidates = [
            (x1, y1 - label_h - 3),
            (x1, y1 + 3),
            (x2 - label_w, y1 - label_h - 3),
            (x2 - label_w, y2 + 3),
            (x1, y2 + 3),
        ]

        best = None
        best_overlap = None
        for candidate_x, candidate_y in candidates:
            draw_x = int(np.clip(candidate_x, 0, max(0, width - label_w)))
            draw_y = int(np.clip(candidate_y, 0, max(0, height - label_h)))
            rect = (draw_x, draw_y, draw_x + label_w, draw_y + label_h)
            overlap = sum(rectangle_overlap(rect, used) for used in occupied)
            if best_overlap is None or overlap < best_overlap:
                best = rect
                best_overlap = overlap
            if overlap == 0:
                break

        draw_x1, draw_y1, draw_x2, draw_y2 = best
        cv2.rectangle(image, (draw_x1, draw_y1), (draw_x2, draw_y2), (18, 18, 18), -1)
        cv2.rectangle(image, (draw_x1, draw_y1), (draw_x2, draw_y2), color, 1)
        cv2.putText(
            image,
            text,
            (draw_x1 + 3, draw_y2 - baseline - 3),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        anchor_x = int(np.clip((x1 + x2) / 2, 0, width - 1))
        anchor_y = int(np.clip(y1, 0, height - 1))
        label_anchor_x = int(np.clip(anchor_x, draw_x1, draw_x2))
        label_anchor_y = draw_y2 if draw_y2 <= anchor_y else draw_y1
        cv2.line(
            image,
            (label_anchor_x, label_anchor_y),
            (anchor_x, anchor_y),
            color,
            1,
            cv2.LINE_AA,
        )
        occupied.append(best)


def append_detection_summary(image, annotations):
    """Append a readable detail table so no accepted detection is undocumented."""
    if not annotations:
        return image

    width = image.shape[1]
    columns = 2 if width < 1500 else 3
    rows = int(np.ceil(len(annotations) / columns))
    header_height = 30
    row_height = 24
    panel_height = header_height + rows * row_height + 8
    panel = np.full((panel_height, width, 3), 22, dtype=np.uint8)
    cv2.putText(
        panel,
        f"ALL DETECTED OBJECTS: {len(annotations)} | original u: smaller = nearer",
        (10, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    column_width = width // columns
    for index, item in enumerate(sorted(annotations, key=lambda entry: entry["object_id"])):
        column = index // rows
        row = index % rows
        x = column * column_width + 10
        y = header_height + row * row_height + 17
        detail = (
            f"#{item['object_id']:02d} {item['class_name']} | "
            f"{item['distance_label']} {item['distance_band']} {item['distance_status']} | "
            f"{item['relation']} | {item['action']} | conf {item['confidence']:.2f}"
        )
        cv2.putText(
            panel,
            detail,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            item["color"],
            1,
            cv2.LINE_AA,
        )
    return cv2.vconcat([image, panel])


def draw_decision_panel(
    image,
    action,
    critical_label,
    object_count,
    advisory_confidence,
    scene_quality,
    depth_scan,
):
    color = ACTION_COLORS[action]
    panel_width = min(image.shape[1] - 20, 700)
    cv2.rectangle(image, (10, 10), (panel_width, 158), (20, 20, 20), -1)
    cv2.rectangle(image, (10, 10), (panel_width, 158), color, 3)
    cv2.putText(
        image,
        f"ADVISORY: {action}",
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"Critical: {critical_label} | Objects: {object_count}",
        (24, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        (
            f"Evidence: {advisory_confidence:.2f} | Scene: "
            f"{scene_quality['status']} {scene_quality['score']:.2f} "
            f"{','.join(scene_quality['flags']) or 'OK'}"
        ),
        (24, 94),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        SCENE_STATUS_COLORS[scene_quality["status"]],
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        (
            f"Depth path: {depth_scan['status']} {depth_scan['confidence']:.2f} | "
            f"u F/M/N: {depth_scan['profile_label']}"
        ),
        (24, 119),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (220, 200, 120),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "Steering: NOT VERIFIED from a single front image",
        (24, 144),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )


class ADASPipeline:
    def __init__(
        self,
        yolo_weight,
        aux_yolo_weight=None,
        camera_height_m=None,
        horizontal_fov_deg=None,
        horizon_ratio=None,
        distance_mode="auto",
    ):
        print(f"[SYSTEM] Loading AI pipeline on {DEVICE.upper()}...")
        self.camera_height_m = (
            camera_height_m
            if camera_height_m is not None
            else DEFAULT_CAMERA_HEIGHT_M
        )
        self.camera_height_source = "manual" if camera_height_m is not None else "default"
        self.horizontal_fov_deg = horizontal_fov_deg
        self.horizon_ratio = horizon_ratio
        self.distance_mode = distance_mode

        hybrid_weight = os.path.join(HYBRID_DIR, "weights", "hybridnets.pth")
        depth_weight = os.path.join(BASE_DIR, "checkpoints", "depth_anything_v2_vitl.pth")
        yolo_weight = self._resolve_path(yolo_weight)
        aux_yolo_weight = self._resolve_path(aux_yolo_weight) if aux_yolo_weight else None

        print("-> Loading HybridNets road/lane segmentation...")
        self.hybrid_model = HybridNetsBackbone(
            num_classes=1,
            seg_classes=2,
            compound_coef=3,
        )
        self.hybrid_model.load_state_dict(load_state_dict(hybrid_weight))
        self.hybrid_model.to(DEVICE).eval()

        print("-> Loading Depth Anything V2 (vitl)...")
        depth_config = {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        }
        self.depth_model = DepthAnythingV2(**depth_config)
        self.depth_model.load_state_dict(load_state_dict(depth_weight))
        self.depth_model.to(DEVICE).eval()

        self.detectors = []
        self.last_yolo_filter_info = {}
        self.yolo_model = self._load_yolo_detector("primary", yolo_weight)
        if aux_yolo_weight:
            self._load_yolo_detector("aux-bdd", aux_yolo_weight)
        if not self.detectors:
            raise RuntimeError("No YOLO detector could be loaded.")

        self.hybrid_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    @staticmethod
    def _resolve_path(path):
        return path if os.path.isabs(path) else os.path.join(BASE_DIR, path)

    def _load_yolo_detector(self, role, weight_path):
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Missing YOLO weight for {role}: {weight_path}")
        print(f"-> Loading YOLO detector [{role}]: {os.path.basename(weight_path)}...")
        model = YOLO(weight_path)
        classes = [
            class_id
            for class_id, class_name in model.names.items()
            if class_name.lower() in TARGET_DETECTION_CLASSES
        ]
        selected_names = [model.names[class_id] for class_id in classes]
        if not selected_names:
            raise ValueError(f"No target traffic classes found in {weight_path}")
        self.detectors.append(
            {
                "role": role,
                "model": model,
                "classes": classes,
                "names": model.names,
                "weight": weight_path,
            }
        )
        print(f"-> {role} obstacle classes: {', '.join(selected_names)}")
        return model

    def get_hybridnets_masks(self, image):
        height, width = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ratio = HYBRID_INPUT_SIZE / max(height, width)
        resized = cv2.resize(
            rgb,
            (int(width * ratio), int(height * ratio)),
            interpolation=cv2.INTER_AREA,
        )
        (network_image, _), _, (pad_w, pad_h) = letterbox(
            (resized, None),
            HYBRID_INPUT_SIZE,
            auto=True,
            scaleup=False,
        )
        input_tensor = self.hybrid_transform(network_image).unsqueeze(0).to(DEVICE)

        with torch.inference_mode():
            segmentation = self.hybrid_model(input_tensor)[-1]

        mask = torch.argmax(segmentation, dim=1)[0].cpu().numpy().astype(np.uint8)

        top = int(round(pad_h - 0.1))
        bottom = int(round(pad_h + 0.1))
        left = int(round(pad_w - 0.1))
        right = int(round(pad_w + 0.1))
        y_end = mask.shape[0] - bottom if bottom else mask.shape[0]
        x_end = mask.shape[1] - right if right else mask.shape[1]
        mask = mask[top:y_end, left:x_end]
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

        road_mask = (mask == 1).astype(np.uint8)
        lane_mask = (mask == 2).astype(np.uint8)

        road_mask = cv2.morphologyEx(
            road_mask,
            cv2.MORPH_CLOSE,
            np.ones((11, 11), np.uint8),
        )
        road_mask = cv2.morphologyEx(
            road_mask,
            cv2.MORPH_OPEN,
            np.ones((5, 5), np.uint8),
        )
        return road_mask, lane_mask

    @staticmethod
    def get_relative_distance_map(raw_depth):
        valid = raw_depth[np.isfinite(raw_depth)]
        if valid.size == 0:
            return np.full(raw_depth.shape, 50.0, dtype=np.float32)

        far_depth, near_depth = np.percentile(valid, [5, 95])
        scale = max(float(near_depth - far_depth), 1e-6)
        proximity = np.clip((raw_depth - far_depth) / scale, 0.0, 1.0)
        return ((1.0 - proximity) * 100.0).astype(np.float32)

    @staticmethod
    def get_legacy_distance_map(raw_depth):
        """Restore the original inverse-depth unit: lower means nearer."""
        distance_map = np.full(raw_depth.shape, np.nan, dtype=np.float32)
        valid = np.isfinite(raw_depth) & (raw_depth > 1e-4)
        distance_map[valid] = np.clip(
            1000.0 / raw_depth[valid],
            0.1,
            999.0,
        ).astype(np.float32)
        return distance_map

    @staticmethod
    def get_object_map_value(distance_map, object_mask, box):
        x1, y1, x2, y2 = box
        box_height = max(y2 - y1, 1)

        # Prefer the lower body of an object and remove uncertain mask edges.
        lower_body = np.zeros_like(object_mask, dtype=np.uint8)
        lower_start = max(y1, y2 - int(box_height * 0.65))
        lower_body[lower_start:y2, x1:x2] = 1
        sample_mask = (object_mask.astype(np.uint8) & lower_body).astype(np.uint8)
        sample_mask = cv2.erode(sample_mask, np.ones((3, 3), np.uint8), iterations=1)

        values = distance_map[sample_mask == 1]
        values = values[np.isfinite(values)]
        if values.size < 20:
            values = distance_map[object_mask]
            values = values[np.isfinite(values)]
        if values.size == 0:
            return None

        # The lower quartile estimates the nearest solid surface without using one noisy pixel.
        return float(np.percentile(values, 25))

    @staticmethod
    def get_object_legacy_unit(
        legacy_distance_map,
        object_mask,
        box,
        mask_source,
        mask_quality,
    ):
        """Estimate original inverse-depth units from independent object regions."""
        x1, y1, x2, y2 = box
        box_width = max(x2 - x1, 1)
        box_height = max(y2 - y1, 1)
        erode_size = max(3, int(min(box_width, box_height) * 0.045))
        if erode_size % 2 == 0:
            erode_size += 1
        core_mask = cv2.erode(
            object_mask.astype(np.uint8),
            np.ones((erode_size, erode_size), np.uint8),
            iterations=1,
        )
        if core_mask.sum() < 20:
            core_mask = object_mask.astype(np.uint8)

        regions = [
            (0.18, 0.82, 0.18, 0.82, "center"),
            (0.12, 0.88, 0.48, 0.88, "lower"),
            (0.28, 0.72, 0.25, 0.72, "inner"),
        ]
        region_estimates = []
        all_values = []
        for x_start, x_end, y_start, y_end, name in regions:
            region_mask = np.zeros_like(core_mask)
            rx1 = x1 + int(box_width * x_start)
            rx2 = x1 + int(box_width * x_end)
            ry1 = y1 + int(box_height * y_start)
            ry2 = y1 + int(box_height * y_end)
            region_mask[ry1:ry2, rx1:rx2] = core_mask[ry1:ry2, rx1:rx2]
            values = legacy_distance_map[region_mask == 1]
            values = values[np.isfinite(values) & (values > 0.1) & (values < 999.0)]
            if values.size < 10:
                continue
            low, high = np.percentile(values, [12, 88])
            values = values[(values >= low) & (values <= high)]
            if values.size == 0:
                continue
            estimate = float(np.median(values))
            region_estimates.append((name, estimate, int(values.size)))
            all_values.append(values)

        if not region_estimates:
            values = legacy_distance_map[object_mask]
            values = values[np.isfinite(values) & (values > 0.1) & (values < 999.0)]
            if values.size == 0:
                return {
                    "value": None,
                    "min": None,
                    "max": None,
                    "confidence": 0.0,
                    "status": "UNAVAILABLE",
                    "regions": [],
                }
            region_estimates = [("fallback", float(np.median(values)), int(values.size))]
            all_values = [values]

        estimates = np.asarray([item[1] for item in region_estimates], dtype=np.float64)
        weights = np.asarray([min(item[2], 300) for item in region_estimates], dtype=np.float64)
        center = float(np.average(np.log(estimates), weights=weights))
        value = float(np.exp(center))
        agreement_ratio = float(estimates.max() / max(estimates.min(), 1e-6))
        values = np.concatenate(all_values)
        q20, q80 = np.percentile(values, [20, 80])
        relative_spread = float((q80 - q20) / max(value, 1e-6))

        confidence = float(
            np.clip(1.35 / agreement_ratio, 0.0, 1.0)
            * np.exp(-relative_spread * 0.85)
            * np.clip(values.size / 120.0, 0.25, 1.0)
        )
        if mask_source != "segmentation":
            confidence *= 0.58
        confidence *= float(np.clip(0.45 + mask_quality * 0.75, 0.35, 1.0))
        if x1 <= 1 or y1 <= 1 or x2 >= legacy_distance_map.shape[1] - 1:
            confidence *= 0.72
        if y2 >= legacy_distance_map.shape[0] - 1:
            confidence *= 0.82

        if confidence >= 0.68 and agreement_ratio <= 1.35:
            status = "STABLE"
        elif confidence >= 0.35 and agreement_ratio <= 1.80:
            status = "CHECK"
        else:
            status = "UNCERTAIN"

        uncertainty = float(
            np.clip(
                0.10 + relative_spread * 0.48 + (1.0 - confidence) * 0.35,
                0.16,
                0.80,
            )
        )
        return {
            "value": value,
            "min": max(0.1, min(value * (1.0 - uncertainty), float(q20))),
            "max": min(999.0, max(value * (1.0 + uncertainty), float(q80))),
            "confidence": confidence,
            "status": status,
            "agreement_ratio": agreement_ratio,
            "regions": [
                {"name": name, "unit": round(estimate, 2), "pixels": pixels}
                for name, estimate, pixels in region_estimates
            ],
        }

    @staticmethod
    def distance_band(legacy_unit):
        if legacy_unit is None:
            return "UNKNOWN"
        if legacy_unit < 4.0:
            return "VERY-NEAR"
        if legacy_unit < 12.0:
            return "NEAR"
        if legacy_unit < 30.0:
            return "MID"
        if legacy_unit < 80.0:
            return "FAR"
        return "VERY-FAR"

    @staticmethod
    def estimate_horizon_y(road_mask, default_ratio):
        """Estimate the road vanishing row from stable central-road coverage."""
        height, width = road_mask.shape
        default_y = float(height * default_ratio)
        central = road_mask[:, int(width * 0.25) : int(width * 0.75)]
        minimum_coverage = max(6, int(central.shape[1] * 0.025))
        valid_rows = central.sum(axis=1) >= minimum_coverage
        run_length = max(8, int(height * 0.018))

        detected_y = None
        start = int(height * 0.25)
        end = int(height * 0.65)
        for y in range(start, max(start, end - run_length)):
            if valid_rows[y : y + run_length].mean() >= 0.80:
                detected_y = float(y)
                break

        if detected_y is None:
            return default_y, "default", 0.25

        detected_y = float(np.clip(detected_y, height * 0.32, height * 0.58))
        horizon_y = detected_y * 0.75 + default_y * 0.25
        return horizon_y, "road-mask", 0.65

    @staticmethod
    def get_focal_px(image_width, horizontal_fov_deg):
        half_fov = np.deg2rad(horizontal_fov_deg) / 2.0
        return float(image_width / (2.0 * np.tan(half_fov)))

    @staticmethod
    def get_horizontal_fov(image_path, manual_fov_deg):
        if manual_fov_deg is not None:
            return float(manual_fov_deg), "manual"
        try:
            with Image.open(image_path) as image:
                focal_35mm = image.getexif().get(41989)
            if focal_35mm:
                focal_35mm = float(focal_35mm)
                horizontal_fov = np.rad2deg(2.0 * np.arctan(36.0 / (2.0 * focal_35mm)))
                if 20.0 <= horizontal_fov <= 150.0:
                    return float(horizontal_fov), "exif-35mm"
        except (OSError, TypeError, ValueError, ZeroDivisionError):
            pass
        return DEFAULT_HORIZONTAL_FOV_DEG, "default"

    @staticmethod
    def calibrate_metric_depth(
        raw_depth,
        road_mask,
        ego_corridor,
        camera_height_m,
        focal_px,
        horizon_y,
    ):
        """Fit Depth Anything inverse depth to flat-road metric geometry."""
        height, width = raw_depth.shape
        calibration_mask = (
            (road_mask > 0)
            & (ego_corridor > 0)
            & np.isfinite(raw_depth)
        )
        start_y = max(int(horizon_y + height * 0.045), 0)
        end_y = min(int(height * 0.92), height)

        raw_points = []
        inverse_meter_points = []
        minimum_pixels = max(8, int(width * 0.012))
        row_step = max(2, height // 180)
        for y in range(start_y, end_y, row_step):
            row_values = raw_depth[y][calibration_mask[y]]
            if row_values.size < minimum_pixels:
                continue
            raw_points.append(float(np.median(row_values)))
            inverse_meter_points.append(
                float((y - horizon_y) / max(camera_height_m * focal_px, 1e-6))
            )

        info = {
            "method": "flat-road-depth-anything",
            "status": "unavailable",
            "confidence": 0.0,
            "sample_rows": len(raw_points),
            "horizon_y": round(float(horizon_y), 2),
            "focal_px": round(float(focal_px), 2),
            "camera_height_m": round(float(camera_height_m), 3),
        }
        if len(raw_points) < 12:
            return None, info

        x = np.asarray(raw_points, dtype=np.float64)
        y = np.asarray(inverse_meter_points, dtype=np.float64)
        keep = np.ones(x.shape, dtype=bool)
        slope = offset = 0.0
        for _ in range(4):
            if keep.sum() < 10:
                break
            slope, offset = np.polyfit(x[keep], y[keep], 1)
            residual = y - (slope * x + offset)
            center = np.median(residual[keep])
            mad = np.median(np.abs(residual[keep] - center))
            threshold = max(0.012, mad * 3.5)
            keep = np.abs(residual - center) <= threshold

        if keep.sum() < 10 or slope <= 0:
            return None, info

        correlation = float(np.corrcoef(x[keep], y[keep])[0, 1])
        predicted = slope * x[keep] + offset
        residual_rmse = float(np.sqrt(np.mean((predicted - y[keep]) ** 2)))
        confidence = float(
            np.clip((correlation - 0.35) / 0.55, 0.0, 1.0)
            * np.clip(keep.sum() / 45.0, 0.0, 1.0)
            * np.exp(-residual_rmse * 14.0)
        )
        if not np.isfinite(correlation) or correlation < 0.35 or confidence < 0.08:
            return None, info

        inverse_metric = slope * raw_depth.astype(np.float64) + offset
        metric_depth = np.full(raw_depth.shape, np.nan, dtype=np.float32)
        valid = np.isfinite(inverse_metric) & (inverse_metric > (1.0 / 250.0))
        metric_depth[valid] = np.clip(
            1.0 / inverse_metric[valid],
            0.5,
            MAX_ESTIMATED_DISTANCE_M,
        ).astype(np.float32)

        info.update(
            {
                "status": "estimated",
                "confidence": round(confidence, 3),
                "sample_rows": int(keep.sum()),
                "correlation": round(correlation, 4),
                "inverse_depth_scale": round(float(slope), 8),
                "inverse_depth_offset": round(float(offset), 8),
            }
        )
        return metric_depth, info

    @staticmethod
    def fuse_object_distance(
        metric_depth,
        relative_distance,
        legacy_distance_map,
        object_mask,
        mask_source,
        mask_quality,
        box,
        class_name,
        confidence,
        image_shape,
        calibration_confidence,
        geometry_confidence,
        camera_height_m,
        focal_px,
        horizon_y,
        metric_geometry_trusted,
    ):
        """Fuse depth, ground contact, and class-height geometry."""
        height, _ = image_shape[:2]
        x1, y1, x2, y2 = box
        box_height = max(y2 - y1, 1)
        estimates = []

        relative_unit = ADASPipeline.get_object_map_value(
            relative_distance,
            object_mask,
            box,
        )
        legacy_estimate = ADASPipeline.get_object_legacy_unit(
            legacy_distance_map,
            object_mask,
            box,
            mask_source,
            mask_quality,
        )
        legacy_unit = legacy_estimate["value"]
        if metric_depth is not None:
            depth_m = ADASPipeline.get_object_map_value(metric_depth, object_mask, box)
            if depth_m is not None:
                estimates.append(
                    ("road_calibrated_depth", depth_m, 0.15 + 0.75 * calibration_confidence)
                )

        ground_denominator = y2 - horizon_y
        if ground_denominator > height * 0.025:
            ground_m = camera_height_m * focal_px / ground_denominator
            ground_weight = 0.18 + 0.38 * geometry_confidence
            if y2 >= height * 0.985:
                ground_weight *= 0.35
            estimates.append(("ground_contact", ground_m, ground_weight))

        nominal_height = CLASS_HEIGHT_M.get(class_name)
        if nominal_height and y1 > 2 and y2 < height - 2:
            size_m = nominal_height * focal_px / box_height
            estimates.append(
                (
                    "class_height",
                    size_m,
                    0.18 * float(confidence) * (0.55 + 0.45 * geometry_confidence),
                )
            )

        estimates = [
            (name, float(np.clip(value, 0.5, MAX_ESTIMATED_DISTANCE_M)), weight)
            for name, value, weight in estimates
            if np.isfinite(value) and value > 0 and weight > 0
        ]
        if not estimates:
            return {
                "distance_m": None,
                "distance_min_m": None,
                "distance_max_m": None,
                "metric_trusted": False,
                "relative_unit": relative_unit,
                "legacy_unit": legacy_unit,
                "legacy_estimate": legacy_estimate,
                "confidence": 0.0,
                "confidence_label": "UNAVAILABLE",
                "source_agreement_ratio": None,
                "sources": [],
            }

        values = np.asarray([item[1] for item in estimates], dtype=np.float64)
        weights = np.asarray([item[2] for item in estimates], dtype=np.float64)
        source_agreement_ratio = float(values.max() / max(values.min(), 1e-6))

        # With three independent estimates, reject one clear outlier before
        # fusing. This prevents a truncated box or depth leak from dominating.
        if len(estimates) >= 3:
            log_median = float(np.median(np.log(values)))
            deviations = np.abs(np.log(values) - log_median)
            keep = deviations <= np.log(1.85)
            if keep.sum() >= 2:
                estimates = [item for item, use in zip(estimates, keep) if use]
                values = values[keep]
                weights = weights[keep]
                source_agreement_ratio = float(values.max() / max(values.min(), 1e-6))

        log_values = np.log(values)
        center = float(np.average(log_values, weights=weights))
        spread = float(np.sqrt(np.average((log_values - center) ** 2, weights=weights)))
        distance_m = float(np.exp(center))
        source_strength = min(1.0, float(weights.sum()) / 1.25)
        estimate_confidence = float(np.clip(source_strength * np.exp(-spread * 1.8), 0.0, 1.0))
        estimate_confidence *= float(np.clip(1.45 / source_agreement_ratio, 0.30, 1.0))
        metric_trusted = bool(
            metric_geometry_trusted
            and len(estimates) >= 2
            and source_agreement_ratio <= 1.55
            and estimate_confidence >= 0.42
        )
        if estimate_confidence >= 0.68:
            confidence_label = "HIGH"
        elif estimate_confidence >= 0.38:
            confidence_label = "MEDIUM"
        else:
            confidence_label = "LOW"
        uncertainty_fraction = float(
            np.clip(
                0.16 + spread * 0.65 + (1.0 - estimate_confidence) * 0.32,
                0.18,
                0.75,
            )
        )

        distance_min_m = max(
            0.5,
            min(distance_m * (1.0 - uncertainty_fraction), float(values.min()) * 0.88),
        )
        distance_max_m = min(
            MAX_ESTIMATED_DISTANCE_M,
            max(distance_m * (1.0 + uncertainty_fraction), float(values.max()) * 1.12),
        )
        return {
            "distance_m": distance_m,
            "distance_min_m": distance_min_m,
            "distance_max_m": distance_max_m,
            "metric_trusted": metric_trusted,
            "relative_unit": relative_unit,
            "legacy_unit": legacy_unit,
            "legacy_estimate": legacy_estimate,
            "confidence": estimate_confidence,
            "confidence_label": confidence_label,
            "source_agreement_ratio": source_agreement_ratio,
            "sources": [
                {"name": name, "distance_m": round(value, 2), "weight": round(weight, 3)}
                for name, value, weight in estimates
            ],
        }

    @staticmethod
    def format_distance_label(distance_estimate, distance_mode):
        legacy_unit = distance_estimate["legacy_unit"]
        legacy_estimate = distance_estimate["legacy_estimate"]
        show_metric = distance_mode == "metric" or (
            distance_mode == "auto" and distance_estimate["metric_trusted"]
        )
        if show_metric and distance_estimate["distance_m"] is not None:
            low = distance_estimate["distance_min_m"]
            high = distance_estimate["distance_max_m"]
            if high < 20.0:
                return f"~{low:.1f}-{high:.1f}m"
            return f"~{low:.0f}-{high:.0f}m"
        if legacy_unit is not None:
            if legacy_estimate["status"] == "STABLE":
                return f"{legacy_unit:.1f}u"
            low = legacy_estimate["min"]
            high = legacy_estimate["max"]
            return f"{low:.1f}-{high:.1f}u?"
        return "distance?"

    @staticmethod
    def metric_distance_for_risk(distance_estimate, distance_mode):
        if distance_mode == "metric":
            return distance_estimate["distance_m"]
        if distance_mode == "auto" and distance_estimate["metric_trusted"]:
            # Use the near side of the trusted range for conservative risk.
            return distance_estimate["distance_min_m"]
        return None

    @staticmethod
    def get_ego_corridor(road_mask, lane_mask):
        """Estimate the currently visible forward-driving corridor.

        Lane pixels guide the corridor boundaries. A perspective trapezoid is
        used where lane markings are missing or hidden by another vehicle.
        """
        height, width = road_mask.shape
        horizon = int(height * 0.35)
        corridor = np.zeros((height, width), dtype=np.uint8)

        lane_guidance = cv2.dilate(
            lane_mask.astype(np.uint8),
            np.ones((7, 5), np.uint8),
            iterations=1,
        )
        previous_center = width / 2
        for y in range(horizon, height):
            progress = (y - horizon) / max(height - horizon - 1, 1)
            expected_half_width = width * (0.04 + 0.19 * progress)
            band_y1 = max(horizon, y - 2)
            band_y2 = min(height, y + 3)
            lane_x = np.where(lane_guidance[band_y1:band_y2].any(axis=0))[0]

            left = lane_x[lane_x < previous_center - 4]
            right = lane_x[lane_x > previous_center + 4]
            left_x = int(left.max()) if left.size else None
            right_x = int(right.min()) if right.size else None

            center = previous_center
            half_width = expected_half_width
            if left_x is not None and right_x is not None:
                measured_width = right_x - left_x
                candidate_center = (left_x + right_x) / 2
                if (
                    expected_half_width * 0.8
                    <= measured_width
                    <= expected_half_width * 3.2
                    and abs(candidate_center - previous_center)
                    <= expected_half_width * 0.35
                ):
                    center = candidate_center
                    half_width = measured_width * 0.46

            max_shift = width * (0.003 + 0.006 * progress)
            center = np.clip(
                center,
                previous_center - max_shift,
                previous_center + max_shift,
            )
            half_width = np.clip(
                half_width,
                expected_half_width * 0.65,
                expected_half_width * 1.35,
            )

            x1 = max(0, int(center - half_width))
            x2 = min(width, int(center + half_width))
            corridor[y, x1:x2] = 1
            previous_center = center

        corridor = cv2.morphologyEx(
            corridor,
            cv2.MORPH_CLOSE,
            np.ones((21, 11), np.uint8),
        )
        return corridor

    @staticmethod
    def assess_depth_corridor(
        legacy_distance_map,
        road_mask,
        ego_corridor,
        horizon_y,
        scene_quality,
    ):
        """Summarize Depth Anything on the path without requiring YOLO boxes."""
        height, width = legacy_distance_map.shape
        top = int(np.clip(max(horizon_y + height * 0.025, height * 0.38), 0, height - 1))
        bottom = int(np.clip(height * 0.88, top + 1, height))
        rows = np.zeros((height, width), dtype=bool)
        rows[top:bottom] = True
        corridor_roi = (ego_corridor > 0) & rows
        valid = (
            corridor_roi
            & np.isfinite(legacy_distance_map)
            & (legacy_distance_map > 0.1)
            & (legacy_distance_map < 999.0)
        )
        roi_pixels = int(corridor_roi.sum())
        valid_pixels = int(valid.sum())
        valid_fraction = valid_pixels / max(roi_pixels, 1)
        road_valid = valid & (road_mask > 0)
        road_coverage = int(road_valid.sum()) / max(valid_pixels, 1)
        sample_mask = road_valid if road_valid.sum() >= 300 else valid

        depth_profile = {}
        span = max(bottom - top, 1)
        bands = (
            ("far", top, top + int(span * 0.34)),
            ("mid", top + int(span * 0.34), top + int(span * 0.67)),
            ("near", top + int(span * 0.67), bottom),
        )
        for name, y1, y2 in bands:
            band_mask = np.zeros((height, width), dtype=bool)
            band_mask[y1:y2] = True
            band_values = legacy_distance_map[sample_mask & band_mask]
            band_values = band_values[np.isfinite(band_values)]
            depth_profile[name] = (
                float(np.median(band_values)) if band_values.size >= 30 else None
            )

        ordered = [depth_profile[name] for name in ("far", "mid", "near")]
        valid_pairs = 0
        correct_pairs = 0
        for first, second in zip(ordered, ordered[1:]):
            if first is None or second is None:
                continue
            valid_pairs += 1
            if first > second:
                correct_pairs += 1
        monotonic_consistency = correct_pairs / max(valid_pairs, 1)
        signal_confidence = float(
            np.clip(valid_fraction, 0.0, 1.0)
            * (0.45 + 0.55 * np.clip(road_coverage / 0.60, 0.0, 1.0))
            * (0.55 + 0.45 * monotonic_consistency)
        )
        confidence = float(
            signal_confidence * (0.35 + 0.65 * float(scene_quality["score"]))
        )

        if valid_fraction < 0.55 or valid_pairs < 2:
            status = "UNAVAILABLE"
        elif scene_quality["status"] == "DEGRADED" or confidence < 0.38:
            status = "LIMITED"
        elif monotonic_consistency < 0.50:
            status = "CHECK"
        else:
            status = "CONSISTENT"
        if status == "UNAVAILABLE":
            confidence = min(confidence, 0.25)

        def format_value(value):
            return f"{value:.1f}" if value is not None else "?"

        return {
            "used": True,
            "status": status,
            "confidence": confidence,
            "profile_original_u": depth_profile,
            "profile_label": "/".join(format_value(value) for value in ordered),
            "valid_fraction": valid_fraction,
            "road_coverage": road_coverage,
            "monotonic_consistency": monotonic_consistency,
            "available_profile_bands": sum(value is not None for value in ordered),
            "analysis_rows": [top, bottom],
        }

    @staticmethod
    def get_path_relation(object_mask, box, ego_corridor):
        height, width = ego_corridor.shape
        x1, y1, x2, y2 = box
        box_height = max(y2 - y1, 1)

        lower_object = np.zeros_like(ego_corridor, dtype=np.uint8)
        lower_y = max(y1, y2 - max(8, int(box_height * 0.30)))
        lower_object[lower_y:y2, x1:x2] = object_mask[lower_y:y2, x1:x2]
        lower_count = int(lower_object.sum())
        path_overlap = (
            float(ego_corridor[lower_object == 1].mean()) if lower_count else 0.0
        )

        foot_x = int(np.clip((x1 + x2) / 2, 0, width - 1))
        foot_y = int(np.clip(y2 - 1, 0, height - 1))
        foot_in_path = bool(ego_corridor[foot_y, foot_x])

        near_kernel = max(11, int(min(height, width) * 0.035))
        if near_kernel % 2 == 0:
            near_kernel += 1
        near_corridor = cv2.dilate(
            ego_corridor,
            np.ones((near_kernel, near_kernel), np.uint8),
            iterations=1,
        )
        near_overlap = (
            float(near_corridor[lower_object == 1].mean()) if lower_count else 0.0
        )

        if foot_in_path or path_overlap >= 0.10:
            return "PATH", path_overlap
        if near_overlap >= 0.10:
            return "EDGE", near_overlap
        return "SIDE", near_overlap

    @staticmethod
    def assess_risk(
        distance_m,
        relative_distance,
        legacy_distance,
        legacy_confidence,
        legacy_status,
        distance_confidence,
        detection_confidence,
        scene_quality_score,
        relation,
        class_name,
        box,
        image_shape,
    ):
        height, width = image_shape[:2]
        x1, y1, x2, y2 = box
        area_ratio = ((x2 - x1) * (y2 - y1)) / max(height * width, 1)
        bottom_ratio = y2 / max(height, 1)

        relative_score = (
            (100.0 - relative_distance) * 0.45
            if relative_distance is not None
            else 0.0
        )
        if legacy_distance is None:
            legacy_score = 0.0
        elif legacy_distance <= 4.0:
            legacy_score = 54.0
        elif legacy_distance <= 12.0:
            legacy_score = float(np.interp(legacy_distance, [4.0, 12.0], [54.0, 36.0]))
        elif legacy_distance <= 30.0:
            legacy_score = float(np.interp(legacy_distance, [12.0, 30.0], [36.0, 16.0]))
        elif legacy_distance <= 80.0:
            legacy_score = float(np.interp(legacy_distance, [30.0, 80.0], [16.0, 3.0]))
        else:
            legacy_score = 0.0
        legacy_score *= 0.35 + 0.65 * legacy_confidence

        if distance_m is not None:
            if distance_m <= 3.0:
                metric_score = 56.0
            elif distance_m <= 7.0:
                metric_score = float(np.interp(distance_m, [3.0, 7.0], [56.0, 46.0]))
            elif distance_m <= 15.0:
                metric_score = float(np.interp(distance_m, [7.0, 15.0], [46.0, 30.0]))
            elif distance_m <= 30.0:
                metric_score = float(np.interp(distance_m, [15.0, 30.0], [30.0, 14.0]))
            elif distance_m <= 60.0:
                metric_score = float(np.interp(distance_m, [30.0, 60.0], [14.0, 4.0]))
            else:
                metric_score = max(0.0, 4.0 - (distance_m - 60.0) * 0.04)
            score = metric_score * (0.70 + 0.30 * distance_confidence)
            score = max(score, relative_score * 0.45, legacy_score * 0.70)
        else:
            score = max(relative_score, legacy_score)

        score += min(14.0, area_ratio * 170.0)
        score += max(0.0, bottom_ratio - 0.45) * 14.0
        if relation == "PATH":
            score += 32.0
        elif relation == "EDGE":
            score += 15.0
        if class_name in VULNERABLE_CLASSES:
            score += 9.0
        if relation in {"PATH", "EDGE"} and distance_confidence < 0.35:
            score += 4.0

        distance_evidence = max(float(legacy_confidence), float(distance_confidence))
        evidence_confidence = float(
            np.sqrt(max(float(detection_confidence), 0.0) * distance_evidence)
            * (0.65 + 0.35 * float(scene_quality_score))
        )

        # A single front image cannot reliably predict whether a side object
        # will enter the path. Side objects therefore cannot trigger BRAKE/STOP.
        if relation == "SIDE":
            if (
                class_name in VULNERABLE_CLASSES
                and (
                    (distance_m is not None and distance_m <= 12.0)
                    or (relative_distance is not None and relative_distance <= 45)
                    or (
                        legacy_distance is not None
                        and legacy_confidence >= 0.35
                        and legacy_distance <= 12.0
                    )
                )
                and bottom_ratio >= 0.70
            ):
                action = "SLOW"
                if (
                    detection_confidence < 0.25
                    and legacy_status in {"UNCERTAIN", "UNAVAILABLE"}
                ):
                    action = "MONITOR"
                return min(float(score), 45.0), action, evidence_confidence
            side_score = min(float(score), 39.0)
            return (
                side_score,
                "MONITOR" if side_score >= 24 else "CLEAR",
                evidence_confidence,
            )

        # EDGE objects deserve caution but do not trigger a full stop without
        # confirmed overlap with the forward path.
        if relation == "EDGE":
            score = min(score, 69.0)

        if score >= 78:
            action = "STOP"
        elif score >= 60:
            action = "BRAKE"
        elif score >= 40:
            action = "SLOW"
        elif score >= 24:
            action = "MONITOR"
        else:
            action = "CLEAR"

        # A static monocular frame must not emit STOP merely because a small or
        # distant detection overlaps the estimated path.
        immediate_metric = (
            distance_m is not None
            and distance_confidence >= 0.55
            and distance_m <= 4.0
        )
        immediate_legacy = (
            legacy_distance is not None
            and legacy_confidence >= 0.60
            and legacy_distance <= 4.0
            and area_ratio >= 0.025
            and bottom_ratio >= 0.68
        )
        if action == "STOP" and not (
            relation == "PATH" and (immediate_metric or immediate_legacy)
        ):
            action = "BRAKE"

        credible_brake = (
            relation in {"PATH", "EDGE"}
            and (
                area_ratio >= 0.005
                or bottom_ratio >= 0.68
                or (
                    legacy_distance is not None
                    and legacy_confidence >= 0.45
                    and legacy_distance <= 14.0
                )
            )
        )
        if action == "BRAKE" and not credible_brake:
            action = "SLOW"

        if relation == "EDGE" and action == "STOP":
            action = "BRAKE"

        # Keep low-confidence detections visible, but do not let weak evidence
        # issue an aggressive static-image advisory.
        if action == "STOP" and (
            detection_confidence < 0.55
            or evidence_confidence < 0.48
            or legacy_status != "STABLE"
        ):
            action = "BRAKE"
        if action == "BRAKE" and (
            detection_confidence < 0.30 or evidence_confidence < 0.28
        ):
            action = "SLOW"
        if (
            legacy_status in {"UNCERTAIN", "UNAVAILABLE"}
            and detection_confidence < 0.35
            and ACTION_SEVERITY[action] > ACTION_SEVERITY["MONITOR"]
        ):
            action = "MONITOR"
        if scene_quality_score < 0.35:
            if action == "BRAKE" and (
                detection_confidence < 0.45 or legacy_status != "STABLE"
            ):
                action = "SLOW"
            if (
                action == "SLOW"
                and detection_confidence < 0.25
                and legacy_status != "STABLE"
            ):
                action = "MONITOR"
            if (
                action == "SLOW"
                and detection_confidence < 0.30
                and legacy_distance is not None
                and legacy_distance > 30.0
            ):
                action = "MONITOR"
        return float(np.clip(score, 0.0, 100.0)), action, evidence_confidence

    def _extract_yolo_detections(
        self,
        result,
        detector,
        variant,
        image_width,
        flip_back=False,
        offset=(0, 0),
        canvas_shape=None,
        tile_shape=None,
    ):
        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy().astype(np.float32)
        if result.masks is not None:
            masks = list(result.masks.data.cpu().numpy())
            masks.extend([None] * (len(boxes) - len(masks)))
        else:
            masks = [None] * len(boxes)

        detections = []
        offset_x, offset_y = offset
        for box, class_id, confidence, mask in zip(boxes, classes, confidences, masks):
            if flip_back:
                x1, y1, x2, y2 = box
                box = np.asarray([image_width - x2, y1, image_width - x1, y2], dtype=np.float32)
                if mask is not None:
                    mask = np.fliplr(mask).copy()
            if offset_x or offset_y:
                box = box + np.asarray([offset_x, offset_y, offset_x, offset_y], dtype=np.float32)
            if mask is not None and canvas_shape is not None and tile_shape is not None:
                tile_h, tile_w = tile_shape[:2]
                if mask.shape[:2] != (tile_h, tile_w):
                    mask = cv2.resize(mask, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST)
                full_mask = np.zeros(canvas_shape[:2], dtype=mask.dtype)
                y_end = min(offset_y + tile_h, full_mask.shape[0])
                x_end = min(offset_x + tile_w, full_mask.shape[1])
                paste_h = max(0, y_end - offset_y)
                paste_w = max(0, x_end - offset_x)
                if paste_h > 0 and paste_w > 0:
                    full_mask[offset_y:y_end, offset_x:x_end] = mask[:paste_h, :paste_w]
                mask = full_mask
            class_name = detector["names"][int(class_id)]
            detections.append(
                {
                    "box": box,
                    "class_id": int(class_id),
                    "class_name": class_name,
                    "confidence": float(confidence),
                    "mask": mask,
                    "detector": detector["role"],
                    "variant": variant,
                    "detector_sources": [f"{detector['role']}:{variant}"],
                    "merged_count": 1,
                }
            )
        return detections

    def _run_yolo_inference(
        self,
        detector,
        image,
        imgsz,
        confidence,
        max_det,
    ):
        kwargs = {
            "verbose": False,
            "conf": confidence,
            "iou": 0.50,
            "classes": detector["classes"],
            "retina_masks": True,
            "agnostic_nms": True,
            "max_det": max_det,
            "device": YOLO_DEVICE,
        }
        try:
            return detector["model"](image, imgsz=imgsz, **kwargs)[0]
        except RuntimeError as exc:
            message = str(exc).lower()
            is_oom = "out of memory" in message or ("cuda" in message and "memory" in message)
            if not is_oom:
                raise
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            fallback_imgsz = min(imgsz, 640)
            if fallback_imgsz >= imgsz:
                print(
                    f"  YOLO pass skipped after CUDA memory error: "
                    f"{detector['role']} imgsz={imgsz}"
                )
                return None
            print(
                f"  YOLO CUDA memory retry: {detector['role']} "
                f"imgsz={imgsz} -> {fallback_imgsz}"
            )
            try:
                return detector["model"](image, imgsz=fallback_imgsz, **kwargs)[0]
            except RuntimeError as retry_exc:
                retry_message = str(retry_exc).lower()
                retry_oom = "out of memory" in retry_message or (
                    "cuda" in retry_message and "memory" in retry_message
                )
                if retry_oom:
                    if DEVICE == "cuda":
                        torch.cuda.empty_cache()
                    print(
                        f"  YOLO pass skipped after retry OOM: "
                        f"{detector['role']} imgsz={fallback_imgsz}"
                    )
                    return None
                raise

    def detect_yolo_objects(
        self,
        raw_image,
        yolo_image,
        enhancement_info,
        detection_confidence,
        detection_image_size,
        yolo_tta,
        yolo_imgszs=None,
        tile_inference="off",
        tile_size=640,
        tile_overlap=0.20,
        precision_mode="balanced",
    ):
        height, width = raw_image.shape[:2]
        yolo_imgszs = parse_imgszs(yolo_imgszs or [detection_image_size])
        variants = []
        if yolo_tta in {"dual", "full"} and enhancement_info["applied"]:
            variants.append(("raw", raw_image, False))
        variants.append(("enhanced" if enhancement_info["applied"] else "raw", yolo_image, False))
        if yolo_tta == "full":
            for name, image, _ in list(variants):
                variants.append((f"{name}_hflip", cv2.flip(image, 1), True))

        seen = set()
        unique_variants = []
        for name, image, flip_back in variants:
            key = (name, flip_back)
            if key not in seen:
                seen.add(key)
                unique_variants.append((name, image, flip_back))

        detections = []
        for detector in self.detectors:
            for imgsz in yolo_imgszs:
                for variant_name, image, flip_back in unique_variants:
                    result = self._run_yolo_inference(
                        detector,
                        image,
                        imgsz,
                        detection_confidence,
                        max_det=180,
                    )
                    if result is None:
                        continue
                    detections.extend(
                        self._extract_yolo_detections(
                            result,
                            detector,
                            f"{variant_name}@{imgsz}",
                            width,
                            flip_back=flip_back,
                        )
                    )

        run_tiles = tile_inference == "on" or (
            tile_inference == "auto"
            and (width > tile_size * 1.25 or height > tile_size * 1.05)
        )
        tiles = build_tiles(raw_image.shape, tile_size, tile_overlap) if run_tiles else []
        tile_imgsz = max(min(max(yolo_imgszs), tile_size), min(yolo_imgszs))
        if tiles:
            for detector in self.detectors:
                for tile_index, (x1, y1, x2, y2) in enumerate(tiles, start=1):
                    crop = yolo_image[y1:y2, x1:x2]
                    result = self._run_yolo_inference(
                        detector,
                        crop,
                        tile_imgsz,
                        detection_confidence,
                        max_det=90,
                    )
                    if result is None:
                        continue
                    detections.extend(
                        self._extract_yolo_detections(
                            result,
                            detector,
                            f"tile{tile_index}@{tile_imgsz}",
                            crop.shape[1],
                            offset=(x1, y1),
                            canvas_shape=raw_image.shape,
                            tile_shape=crop.shape,
                        )
                    )

        merged = weighted_merge_detections(detections, iou_threshold=0.55)
        support_passes = (
            len(self.detectors) * len(unique_variants) * len(yolo_imgszs)
            + len(self.detectors) * len(tiles)
        )
        deduped, duplicate_suppressed = suppress_nested_duplicates(merged)
        filtered, weak_suppressed = filter_weak_yolo_detections(
            deduped,
            detection_confidence,
            support_passes,
            precision_mode,
        )
        self.last_yolo_filter_info = {
            "raw_detections": len(detections),
            "merged_before_filter": len(merged),
            "duplicate_suppressed": len(duplicate_suppressed),
            "accepted_after_filter": len(filtered),
            "weak_suppressed": len(weak_suppressed),
            "support_passes": support_passes,
            "precision_mode": precision_mode,
            "duplicate_suppressed_examples": duplicate_suppressed[:20],
            "weak_suppressed_examples": weak_suppressed[:20],
        }
        print(
            "  YOLO detector ensemble: "
            f"raw={len(detections)} merged={len(merged)} accepted={len(filtered)} "
            f"duplicate_suppressed={len(duplicate_suppressed)} "
            f"weak_suppressed={len(weak_suppressed)} "
            f"detectors={len(self.detectors)} variants={len(unique_variants)} "
            f"scales={','.join(str(size) for size in yolo_imgszs)} "
            f"tiles={len(tiles)}"
        )
        return filtered

    def process_image(
        self,
        image_path,
        output_dir,
        detection_confidence=DEFAULT_DETECTION_CONFIDENCE,
        detection_image_size=DEFAULT_DETECTION_IMAGE_SIZE,
        depth_input_size=518,
        enhance_input="auto",
        yolo_tta="dual",
        adaptive_conf=True,
        yolo_imgszs=None,
        tile_inference="off",
        tile_size=640,
        tile_overlap=0.20,
        precision_mode="balanced",
    ):
        image_path = self._resolve_path(image_path)
        print(f"\n[ANALYZE] {image_path}")
        raw_image = cv2.imread(image_path)
        if raw_image is None:
            print("  Skipped: unable to read image.")
            return None

        height, width = raw_image.shape[:2]
        scene_quality = assess_scene_quality(raw_image)
        print(
            "  Scene quality: "
            f"{scene_quality['status']} ({scene_quality['score']:.2f}) "
            f"{','.join(scene_quality['flags']) or 'OK'}"
        )
        annotated = raw_image.copy()
        yolo_image, enhancement_info = enhance_yolo_input(
            raw_image,
            scene_quality,
            enhance_input,
        )
        if enhancement_info["applied"]:
            print(
                "  YOLO input enhancement: "
                + ", ".join(enhancement_info["operations"])
            )
        else:
            print(f"  YOLO input enhancement: {enhance_input} (raw image)")
        effective_detection_confidence = (
            adaptive_detection_confidence(detection_confidence, scene_quality)
            if adaptive_conf
            else detection_confidence
        )
        if effective_detection_confidence != detection_confidence:
            print(
                "  Adaptive YOLO confidence: "
                f"{detection_confidence:.3f} -> {effective_detection_confidence:.3f}"
            )

        road_mask, lane_mask = self.get_hybridnets_masks(raw_image)
        annotated = blend_mask(annotated, road_mask == 1, (0, 180, 0), 0.28)
        annotated = blend_mask(annotated, lane_mask == 1, (0, 220, 255), 0.45)
        # Keep the estimated ego corridor internal for PATH/EDGE/SIDE risk
        # evaluation. Its inferred outline is unstable and should not be drawn.
        ego_corridor = self.get_ego_corridor(road_mask, lane_mask)
        ground_support_mask = (
            (road_mask > 0) | (lane_mask > 0) | (ego_corridor > 0)
        ).astype(np.uint8)

        with torch.inference_mode():
            raw_depth = self.depth_model.infer_image(raw_image, depth_input_size)
        relative_distance = self.get_relative_distance_map(raw_depth)
        legacy_distance = self.get_legacy_distance_map(raw_depth)
        default_horizon_ratio = (
            self.horizon_ratio
            if self.horizon_ratio is not None
            else DEFAULT_HORIZON_RATIO
        )
        if self.horizon_ratio is None:
            horizon_y, horizon_source, horizon_confidence = self.estimate_horizon_y(
                road_mask,
                default_horizon_ratio,
            )
        else:
            horizon_y = float(height * self.horizon_ratio)
            horizon_source = "manual"
            horizon_confidence = 1.0
        depth_scan = self.assess_depth_corridor(
            legacy_distance,
            road_mask,
            ego_corridor,
            horizon_y,
            scene_quality,
        )
        print(
            "  Depth-only path scan: "
            f"{depth_scan['status']} ({depth_scan['confidence']:.2f}) "
            f"u F/M/N={depth_scan['profile_label']}"
        )
        horizontal_fov_deg, fov_source = self.get_horizontal_fov(
            image_path,
            self.horizontal_fov_deg,
        )
        focal_px = self.get_focal_px(width, horizontal_fov_deg)
        metric_depth, calibration_info = self.calibrate_metric_depth(
            raw_depth,
            road_mask,
            ego_corridor,
            self.camera_height_m,
            focal_px,
            horizon_y,
        )
        calibration_info.update(
            {
                "horizon_source": horizon_source,
                "horizon_confidence": round(float(horizon_confidence), 3),
                "horizontal_fov_deg": round(float(horizontal_fov_deg), 3),
                "horizontal_fov_source": fov_source,
                "camera_height_source": self.camera_height_source,
            }
        )
        fov_confidence = 0.90 if fov_source in {"manual", "exif-35mm"} else 0.65
        geometry_confidence = float(
            np.clip(horizon_confidence * fov_confidence, 0.0, 1.0)
        )
        effective_metric_confidence = float(
            calibration_info["confidence"] * geometry_confidence
        )
        calibration_info["geometry_confidence"] = round(geometry_confidence, 3)
        calibration_info["effective_metric_confidence"] = round(
            effective_metric_confidence,
            3,
        )
        metric_geometry_trusted = bool(
            self.camera_height_source == "manual"
            and fov_source in {"manual", "exif-35mm"}
            and horizon_source in {"manual", "road-mask"}
        )
        calibration_info["metric_geometry_trusted"] = metric_geometry_trusted
        calibration_info["distance_mode"] = self.distance_mode
        print(
            "  Distance calibration: "
            f"{calibration_info['status']} "
            f"(effective confidence={effective_metric_confidence:.2f}, "
            f"horizon={horizon_y:.0f}px/{horizon_source})"
        )

        yolo_detections = self.detect_yolo_objects(
            raw_image,
            yolo_image,
            enhancement_info,
            effective_detection_confidence,
            detection_image_size,
            yolo_tta,
            yolo_imgszs,
            tile_inference,
            tile_size,
            tile_overlap,
            precision_mode,
        )

        detected_objects = 0
        global_action = "CLEAR"
        global_advisory_confidence = 0.0
        critical_label = "none"
        critical_rank = (-1, -1.0)
        risk_records = []
        annotation_records = []
        suppressed_records = []
        if yolo_detections:
            for source_detection_id, detection in enumerate(yolo_detections, start=1):
                mask = detection["mask"]
                box_values = detection["box"]
                class_name = detection["class_name"]
                confidence = detection["confidence"]
                x1, y1, x2, y2 = map(int, box_values)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                box_area = (x2 - x1) * (y2 - y1)
                box_width = x2 - x1
                box_height = y2 - y1
                center_x = (x1 + x2) / 2
                box_area_ratio = box_area / max(height * width, 1)
                if box_area < 16:
                    continue

                # Suppress the ego vehicle hood while keeping real close vehicles.
                is_ego_hood = (
                    y1 >= height * 0.68
                    and y2 >= height * 0.85
                    and box_width >= width * 0.25
                    and width * 0.20 <= center_x <= width * 0.80
                    and box_height <= height * 0.35
                )
                if is_ego_hood:
                    continue

                if mask is None:
                    object_mask = np.zeros((height, width), dtype=bool)
                    object_mask[y1:y2, x1:x2] = True
                    mask_source = "box"
                elif mask.shape != (height, width):
                    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                    object_mask = mask > 0.5
                    mask_source = "segmentation"
                else:
                    object_mask = mask > 0.5
                    mask_source = "segmentation"
                box_mask = np.zeros((height, width), dtype=bool)
                box_mask[y1:y2, x1:x2] = True
                object_mask &= box_mask
                if not object_mask.any():
                    continue
                mask_coverage = float(
                    object_mask[y1:y2, x1:x2].mean()
                )
                if mask_source == "segmentation":
                    # Very sparse or almost-box-shaped masks are less reliable
                    # for sampling the object's true depth.
                    coverage_quality = 1.0 - min(
                        abs(mask_coverage - 0.48) / 0.48,
                        1.0,
                    )
                    mask_quality = float(np.clip(0.35 + coverage_quality * 0.65, 0.35, 1.0))
                else:
                    mask_quality = 0.45

                relation, _ = self.get_path_relation(
                    object_mask,
                    (x1, y1, x2, y2),
                    ego_corridor,
                )
                detector_roles = set(detection.get("detector_roles", [])) or {
                    source.split(":", 1)[0]
                    for source in detection.get("detector_sources", [])
                }
                ground_score = ground_support_score(
                    ground_support_mask,
                    (x1, y1, x2, y2),
                    raw_image.shape,
                )
                geometry_issues = []
                if (
                    class_name in {"car", "person", "bicycle", "motorcycle", "rider"}
                    and box_area_ratio > 0.28
                    and y2 < height * 0.85
                ):
                    geometry_issues.append("FLOATING_OVERSIZE_BOX")
                if (
                    confidence < 0.18
                    and box_area_ratio > 0.06
                    and mask_source == "segmentation"
                    and mask_coverage > 0.92
                ):
                    geometry_issues.append("LOW_CONFIDENCE_LARGE_SOLID_MASK")
                geometry_issues.extend(
                    class_geometry_reasons(
                        class_name,
                        (x1, y1, x2, y2),
                        raw_image.shape,
                        horizon_y,
                        confidence,
                        mask_source,
                        mask_coverage,
                        ground_score,
                        relation,
                        detector_roles,
                        precision_mode,
                    )
                )
                if geometry_issues:
                    suppressed_records.append(
                        {
                            "source_detection_id": source_detection_id,
                            "class": class_name,
                            "confidence": round(float(confidence), 4),
                            "detector_sources": detection["detector_sources"],
                            "merged_count": detection["merged_count"],
                            "box_xyxy": [x1, y1, x2, y2],
                            "box_area_ratio": round(float(box_area_ratio), 4),
                            "mask_coverage": round(mask_coverage, 4),
                            "ground_support_score": round(float(ground_score), 4),
                            "corridor_relation": relation,
                            "reasons": geometry_issues,
                        }
                    )
                    continue

                distance_estimate = self.fuse_object_distance(
                    metric_depth,
                    relative_distance,
                    legacy_distance,
                    object_mask,
                    mask_source,
                    mask_quality,
                    (x1, y1, x2, y2),
                    class_name,
                    confidence,
                    raw_image.shape,
                    effective_metric_confidence,
                    geometry_confidence,
                    self.camera_height_m,
                    focal_px,
                    horizon_y,
                    metric_geometry_trusted,
                )
                relative_unit = distance_estimate["relative_unit"]
                distance_m = distance_estimate["distance_m"]
                legacy_unit = distance_estimate["legacy_unit"]
                legacy_estimate = distance_estimate["legacy_estimate"]
                if relative_unit is None and distance_m is None and legacy_unit is None:
                    continue

                risk_score, action, evidence_confidence = self.assess_risk(
                    self.metric_distance_for_risk(
                        distance_estimate,
                        self.distance_mode,
                    ),
                    relative_unit,
                    legacy_unit,
                    legacy_estimate["confidence"],
                    legacy_estimate["status"],
                    distance_estimate["confidence"],
                    confidence,
                    scene_quality["score"],
                    relation,
                    class_name,
                    (x1, y1, x2, y2),
                    raw_image.shape,
                )
                weak_context_issues = []
                if (
                    confidence < 0.20
                    and detection.get("merged_count", 1) <= 1
                    and len(detector_roles) < 2
                    and relation == "SIDE"
                    and legacy_estimate["status"] != "STABLE"
                ):
                    weak_context_issues.append("LOW_CONFIDENCE_UNSTABLE_SIDE_OBJECT")
                if (
                    confidence < 0.18
                    and distance_estimate["confidence"] < 0.32
                    and legacy_estimate["status"] in {"UNCERTAIN", "UNAVAILABLE"}
                    and ACTION_SEVERITY[action] <= ACTION_SEVERITY["MONITOR"]
                ):
                    weak_context_issues.append("WEAK_VISUAL_AND_DEPTH_EVIDENCE")
                if (
                    confidence < 0.26
                    and distance_estimate["confidence"] < 0.40
                    and legacy_estimate["status"] in {"UNCERTAIN", "UNAVAILABLE"}
                    and ACTION_SEVERITY[action] <= ACTION_SEVERITY["MONITOR"]
                ):
                    weak_context_issues.append("LOW_CONFIDENCE_MONITOR_ONLY_BOX")
                if (
                    precision_mode == "strict"
                    and scene_quality["status"] == "DEGRADED"
                    and class_name in VEHICLE_CLASSES
                    and confidence < 0.42
                    and len(detector_roles) < 2
                    and distance_estimate["confidence"] < 0.34
                    and legacy_estimate["status"] in {"UNCERTAIN", "UNAVAILABLE"}
                    and ACTION_SEVERITY[action] <= ACTION_SEVERITY["MONITOR"]
                ):
                    weak_context_issues.append(
                        "DEGRADED_MONITOR_ONLY_UNCONFIRMED_VEHICLE"
                    )
                if (
                    precision_mode == "strict"
                    and class_name in VEHICLE_CLASSES
                    and confidence < 0.38
                    and ground_score < 0.010
                    and relation == "SIDE"
                    and distance_estimate["confidence"] < 0.55
                    and len(detector_roles) < 2
                ):
                    weak_context_issues.append("STRICT_NO_GROUND_SUPPORT_SIDE_BOX")
                if weak_context_issues:
                    suppressed_records.append(
                        {
                            "source_detection_id": source_detection_id,
                            "class": class_name,
                            "confidence": round(float(confidence), 4),
                            "detector_sources": detection["detector_sources"],
                            "merged_count": detection["merged_count"],
                            "box_xyxy": [x1, y1, x2, y2],
                            "box_area_ratio": round(float(box_area_ratio), 4),
                            "mask_coverage": round(mask_coverage, 4),
                            "ground_support_score": round(float(ground_score), 4),
                            "distance_confidence": round(
                                float(distance_estimate["confidence"]),
                                3,
                            ),
                            "original_unit_status": legacy_estimate["status"],
                            "corridor_relation": relation,
                            "reasons": weak_context_issues,
                        }
                    )
                    continue
                color = ACTION_COLORS[action]
                distance_label = self.format_distance_label(
                    distance_estimate,
                    self.distance_mode,
                )
                distance_band = self.distance_band(legacy_unit)
                object_id = detected_objects + 1
                compact_label = (
                    f"#{object_id:02d} {class_name} {distance_label} "
                    f"{distance_band} {legacy_estimate['status']}"
                )
                if action in {"SLOW", "BRAKE", "STOP"}:
                    annotated = blend_mask(annotated, object_mask, color, 0.30)

                # HybridNets only visualizes road/lane. Every YOLO detection is
                # evaluated using depth and its relation to the visible ego corridor.
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                annotation_records.append(
                    {
                        "object_id": object_id,
                        "class_name": class_name,
                        "confidence": float(confidence),
                        "box": (x1, y1, x2, y2),
                        "box_area": box_area,
                        "color": color,
                        "label": compact_label,
                        "distance_label": distance_label,
                        "distance_band": distance_band,
                        "distance_status": legacy_estimate["status"],
                        "display_sort_distance": (
                            legacy_unit if legacy_unit is not None else 999.0
                        ),
                        "relation": relation,
                        "action": action,
                    }
                )

                detected_objects += 1
                object_rank = (ACTION_SEVERITY[action], risk_score)
                if object_rank > critical_rank:
                    critical_rank = object_rank
                    critical_label = (
                        f"{class_name} {relation} {distance_label} risk{risk_score:.0f}"
                    )
                if ACTION_SEVERITY[action] > ACTION_SEVERITY[global_action]:
                    global_action = action
                    global_advisory_confidence = evidence_confidence
                elif action == global_action:
                    global_advisory_confidence = max(
                        global_advisory_confidence,
                        evidence_confidence,
                    )
                risk_records.append(
                    {
                        "class": class_name,
                        "object_id": object_id,
                        "confidence": round(float(confidence), 4),
                        "detector_sources": detection["detector_sources"],
                        "merged_count": detection["merged_count"],
                        "box_xyxy": [x1, y1, x2, y2],
                        "estimated_distance_m": (
                            round(float(distance_m), 2)
                            if distance_m is not None
                            else None
                        ),
                        "estimated_distance_range_m": (
                            [
                                round(float(distance_estimate["distance_min_m"]), 2),
                                round(float(distance_estimate["distance_max_m"]), 2),
                            ]
                            if distance_m is not None
                            else None
                        ),
                        "distance_confidence": round(
                            float(distance_estimate["confidence"]),
                            3,
                        ),
                        "distance_confidence_label": distance_estimate[
                            "confidence_label"
                        ],
                        "original_unit_confidence": round(
                            float(legacy_estimate["confidence"]),
                            3,
                        ),
                        "original_unit_status": legacy_estimate["status"],
                        "original_unit_range": (
                            [
                                round(float(legacy_estimate["min"]), 2),
                                round(float(legacy_estimate["max"]), 2),
                            ]
                            if legacy_unit is not None
                            else None
                        ),
                        "original_unit_regions": legacy_estimate["regions"],
                        "mask_source": mask_source,
                        "mask_coverage": round(mask_coverage, 4),
                        "mask_quality": round(mask_quality, 3),
                        "ground_support_score": round(float(ground_score), 4),
                        "metric_trusted": distance_estimate["metric_trusted"],
                        "source_agreement_ratio": (
                            round(float(distance_estimate["source_agreement_ratio"]), 3)
                            if distance_estimate["source_agreement_ratio"] is not None
                            else None
                        ),
                        "displayed_distance": distance_label,
                        "distance_band": distance_band,
                        "distance_sources": distance_estimate["sources"],
                        "original_inverse_depth_unit": (
                            round(float(legacy_unit), 2)
                            if legacy_unit is not None
                            else None
                        ),
                        "relative_unit": (
                            round(float(relative_unit), 2)
                            if relative_unit is not None
                            else None
                        ),
                        "corridor_relation": relation,
                        "risk_score": round(float(risk_score), 2),
                        "risk_evidence_confidence": round(
                            float(evidence_confidence),
                            3,
                        ),
                        "advisory": action,
                    }
                )

        if detected_objects == 0:
            critical_label = (
                f"depth-only path {depth_scan['status']} "
                f"u {depth_scan['profile_label']}"
            )
            if depth_scan["status"] in {"LIMITED", "UNAVAILABLE", "CHECK"}:
                global_action = "MONITOR"
                global_advisory_confidence = depth_scan["confidence"]

        draw_all_detection_labels(annotated, annotation_records)
        draw_decision_panel(
            annotated,
            global_action,
            critical_label,
            detected_objects,
            global_advisory_confidence,
            scene_quality,
            depth_scan,
        )
        overlay_frame = annotated.copy()
        annotated = append_detection_summary(annotated, annotation_records)

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"upgraded_{os.path.basename(image_path)}")
        cv2.imwrite(output_path, annotated)
        visualization = save_visualizations(
            raw_image,
            overlay_frame,
            raw_depth,
            output_dir,
            os.path.splitext(os.path.basename(output_path))[0],
        )
        report_path = os.path.splitext(output_path)[0] + ".json"
        with open(report_path, "w", encoding="utf-8") as report_file:
            json.dump(
                {
                    "image": image_path,
                    "visualization": visualization,
                    "global_advisory": global_action,
                    "global_advisory_confidence": round(
                        float(global_advisory_confidence),
                        3,
                    ),
                    "critical_object": critical_label,
                    "steering_advisory": "NOT_VERIFIED_FROM_SINGLE_FRONT_IMAGE",
                    "scene_quality": {
                        "status": scene_quality["status"],
                        "score": round(float(scene_quality["score"]), 3),
                        "flags": scene_quality["flags"],
                        "median_brightness": round(
                            float(scene_quality["median_brightness"]),
                            2,
                        ),
                        "dynamic_range_p10_p90": round(
                            float(scene_quality["dynamic_range_p10_p90"]),
                            2,
                        ),
                        "laplacian_variance": round(
                            float(scene_quality["laplacian_variance"]),
                            2,
                        ),
                        "glare_fraction": round(
                            float(scene_quality["glare_fraction"]),
                            5,
                        ),
                    },
                    "depth_corridor_scan": {
                        "used_without_yolo_boxes": True,
                        "status": depth_scan["status"],
                        "confidence": round(float(depth_scan["confidence"]), 3),
                        "profile_original_u": {
                            key: round(float(value), 2) if value is not None else None
                            for key, value in depth_scan["profile_original_u"].items()
                        },
                        "profile_meaning": (
                            "Depth Anything path profile from far/mid/near corridor bands; "
                            "smaller original u means nearer."
                        ),
                        "valid_fraction": round(float(depth_scan["valid_fraction"]), 3),
                        "road_coverage": round(float(depth_scan["road_coverage"]), 3),
                        "monotonic_consistency": round(
                            float(depth_scan["monotonic_consistency"]),
                            3,
                        ),
                        "available_profile_bands": depth_scan[
                            "available_profile_bands"
                        ],
                        "analysis_rows": depth_scan["analysis_rows"],
                    },
                    "distance_calibration": calibration_info,
                    "yolo_input_enhancement": enhancement_info,
                    "yolo_detection_settings": {
                        "base_confidence": round(float(detection_confidence), 4),
                        "effective_confidence": round(float(effective_detection_confidence), 4),
                        "adaptive_confidence": bool(adaptive_conf),
                        "imgszs": parse_imgszs(yolo_imgszs or [detection_image_size]),
                        "tta_mode": yolo_tta,
                        "tile_inference": tile_inference,
                        "tile_size": int(tile_size),
                        "tile_overlap": round(float(tile_overlap), 3),
                        "precision_mode": precision_mode,
                        "precision_filter": self.last_yolo_filter_info,
                        "detectors": [
                            {
                                "role": detector["role"],
                                "weight": os.path.basename(detector["weight"]),
                            }
                            for detector in self.detectors
                        ],
                    },
                    "distance_unit": (
                        "estimated_meter_range_when_trusted_else_original_inverse_depth_unit"
                        if self.distance_mode == "auto"
                        else self.distance_mode
                    ),
                    "original_unit_meaning": (
                        "Depth Anything inverse-depth unit restored from the original pipeline; "
                        "smaller values mean nearer objects and values are not meters."
                    ),
                    "distance_confidence_meaning": (
                        "Agreement between monocular depth, flat-road geometry, and class-size estimates; "
                        "not guaranteed absolute accuracy."
                    ),
                    "object_count": detected_objects,
                    "suppressed_detection_count": len(suppressed_records),
                    "suppressed_detections": suppressed_records,
                    "detections": risk_records,
                    "limitations": [
                        "Distances marked ~m are monocular estimates, not calibrated sensor measurements.",
                        "Set camera height, horizontal FOV, and horizon override for the actual camera to improve accuracy.",
                        "A single image cannot estimate closing speed or time-to-collision.",
                        "A depth-only path scan cannot identify object class and is capped at MONITOR without a YOLO detection.",
                        "A front camera cannot verify blind spots or safe steering.",
                        "Advisories must not directly control vehicle actuators.",
                    ],
                },
                report_file,
                indent=2,
            )
        print(
            f"  Saved: {output_path} | objects={detected_objects} "
            f"| suppressed={len(suppressed_records)} "
            f"| advisory={global_action} | critical={critical_label}"
        )
        return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Static-image ADAS analysis in max-quality mode with YOLO, "
            "HybridNets, and Depth Anything."
        )
    )
    parser.add_argument("images", nargs="*", default=None)
    parser.add_argument(
        "--profile",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--yolo-weight",
        default=None,
        help="Override the primary YOLO detection or segmentation weight.",
    )
    parser.add_argument(
        "--aux-yolo-weight",
        default=None,
        help="Optional second YOLO detector weight for ensemble fusion, e.g. the BDD-trained checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(BASE_DIR, "upgraded_outputs"),
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="YOLO confidence threshold. Lower values find more distant objects.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="YOLO inference image size. Larger values improve small-object detection.",
    )
    parser.add_argument(
        "--yolo-imgszs",
        default=None,
        help="Comma-separated YOLO image sizes for multi-scale inference, e.g. 768,960,1152.",
    )
    parser.add_argument(
        "--depth-size",
        type=int,
        default=None,
        help="Depth Anything input size. Lower values are faster but less detailed.",
    )
    parser.add_argument(
        "--enhance-input",
        choices=ENHANCE_INPUT_MODES,
        default=None,
        help=(
            "Light preprocessing for YOLO only: auto enhances low-light/low-contrast "
            "frames, off keeps raw input, always applies mild enhancement."
        ),
    )
    parser.add_argument(
        "--yolo-tta",
        choices=YOLO_TTA_MODES,
        default=None,
        help="YOLO test-time augmentation/ensemble mode: off, dual(raw+enhanced), or full(+horizontal flip).",
    )
    parser.add_argument(
        "--tile-inference",
        choices=TILE_INFERENCE_MODES,
        default=None,
        help="Run overlapping crop inference to improve distant/small-object recall.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=640,
        help="Tile size in pixels when --tile-inference is auto/on.",
    )
    parser.add_argument(
        "--tile-overlap",
        type=float,
        default=0.20,
        help="Tile overlap ratio when --tile-inference is auto/on.",
    )
    parser.add_argument(
        "--precision-mode",
        choices=PRECISION_MODES,
        default=None,
        help="False-positive control: strict is safest, balanced is moderate, recall keeps more weak boxes.",
    )
    parser.add_argument(
        "--no-adaptive-conf",
        action="store_true",
        help="Disable automatic YOLO confidence lowering on degraded scenes.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many images when an input is a directory.",
    )
    parser.add_argument(
        "--image-list",
        default=None,
        help="Text file containing one image path per line.",
    )
    parser.add_argument(
        "--camera-height-m",
        type=float,
        default=None,
        help="Measured camera height above the road in meters. Required for trusted metric output.",
    )
    parser.add_argument(
        "--horizontal-fov-deg",
        type=float,
        default=None,
        help="Camera horizontal field of view in degrees; otherwise EXIF or a 70-degree fallback is used.",
    )
    parser.add_argument(
        "--horizon-ratio",
        type=float,
        default=None,
        help="Optional manual horizon row divided by image height; otherwise estimated from road mask.",
    )
    parser.add_argument(
        "--distance-mode",
        choices=DISTANCE_MODES,
        default="auto",
        help="auto shows meters only when calibrated; metric forces experimental meters; relative shows stable scene units.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = MAX_INFERENCE_CONFIG
    yolo_weight = args.yolo_weight or config["yolo_weight"]
    aux_yolo_weight = args.aux_yolo_weight if args.aux_yolo_weight is not None else config["aux_yolo_weight"]
    confidence = args.conf if args.conf is not None else config["confidence"]
    image_size = args.imgsz if args.imgsz is not None else config["image_size"]
    depth_size = args.depth_size if args.depth_size is not None else config["depth_size"]
    enhance_input = args.enhance_input or config["enhance_input"]
    yolo_tta = args.yolo_tta or config["yolo_tta"]
    yolo_imgszs = (
        parse_imgszs(args.yolo_imgszs)
        if args.yolo_imgszs
        else parse_imgszs(config["yolo_imgszs"])
    )
    tile_inference = args.tile_inference or config["tile_inference"]
    precision_mode = args.precision_mode or config["precision_mode"]
    print(
        f"[MODE] max: yolo={yolo_weight}, "
        f"aux_yolo={aux_yolo_weight or 'none'}, "
        f"conf={confidence}, imgsz={image_size}, depth_size={depth_size}, "
        f"enhance_input={enhance_input}, yolo_tta={yolo_tta}, "
        f"yolo_imgszs={','.join(str(size) for size in yolo_imgszs)}, "
        f"tile_inference={tile_inference}, precision_mode={precision_mode}, "
        f"adaptive_conf={not args.no_adaptive_conf}"
    )
    image_inputs = list(args.images or [])
    if args.image_list:
        image_inputs.extend(load_image_list(args.image_list))
    if not image_inputs:
        image_inputs = list(DEFAULT_IMAGES)
    images = expand_image_inputs(image_inputs, args.limit)
    if not images:
        raise FileNotFoundError("No readable image inputs were found.")
    print(f"[INPUT] Static images selected: {len(images)}")

    if args.camera_height_m is not None and args.camera_height_m <= 0:
        raise ValueError("--camera-height-m must be positive.")
    if args.horizontal_fov_deg is not None and not 20.0 <= args.horizontal_fov_deg <= 150.0:
        raise ValueError("--horizontal-fov-deg must be between 20 and 150.")
    if args.horizon_ratio is not None and not 0.20 <= args.horizon_ratio <= 0.75:
        raise ValueError("--horizon-ratio must be between 0.20 and 0.75.")
    if args.tile_size < 256:
        raise ValueError("--tile-size must be at least 256.")
    if not 0.0 <= args.tile_overlap <= 0.65:
        raise ValueError("--tile-overlap must be between 0.0 and 0.65.")

    pipeline = ADASPipeline(
        yolo_weight,
        aux_yolo_weight=aux_yolo_weight,
        camera_height_m=args.camera_height_m,
        horizontal_fov_deg=args.horizontal_fov_deg,
        horizon_ratio=args.horizon_ratio,
        distance_mode=args.distance_mode,
    )
    for image in images:
        pipeline.process_image(
            image,
            args.output_dir,
            confidence,
            image_size,
            depth_size,
            enhance_input,
            yolo_tta,
            not args.no_adaptive_conf,
            yolo_imgszs,
            tile_inference,
            args.tile_size,
            args.tile_overlap,
            precision_mode,
        )


if __name__ == "__main__":
    main()
