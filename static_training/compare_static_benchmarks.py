"""Compare detections and advisories from two static benchmark runs."""

import argparse
import json
from pathlib import Path

import numpy as np


SEVERITY = {"CLEAR": 0, "MONITOR": 1, "SLOW": 2, "BRAKE": 3, "STOP": 4}


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0,
        min(ay2, by2) - max(ay1, by1),
    )
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return intersection / max(area_a + area_b - intersection, 1)


def load_reports(directory):
    reports = {}
    for path in directory.glob("*.json"):
        if path.name == "benchmark_analysis.json":
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        reports[Path(report["image"]).name] = report
    return reports


def match_detections(first, second, threshold=0.35):
    candidates = []
    for first_index, item_a in enumerate(first):
        for second_index, item_b in enumerate(second):
            if item_a["class"] != item_b["class"]:
                continue
            overlap = iou(item_a["box_xyxy"], item_b["box_xyxy"])
            if overlap >= threshold:
                candidates.append((overlap, first_index, second_index))
    candidates.sort(reverse=True)
    used_first = set()
    used_second = set()
    matches = []
    for overlap, first_index, second_index in candidates:
        if first_index in used_first or second_index in used_second:
            continue
        used_first.add(first_index)
        used_second.add(second_index)
        matches.append((first_index, second_index, overlap))
    return matches, used_first, used_second


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    first_reports = load_reports(args.first)
    second_reports = load_reports(args.second)
    common = sorted(set(first_reports) & set(second_reports))
    rows = []
    distance_differences = []
    first_only_total = second_only_total = matches_total = 0
    advisory_changes = 0

    for name in common:
        first = first_reports[name]
        second = second_reports[name]
        first_detections = first.get("detections", [])
        second_detections = second.get("detections", [])
        matches, used_first, used_second = match_detections(first_detections, second_detections)
        first_only = len(first_detections) - len(used_first)
        second_only = len(second_detections) - len(used_second)
        first_only_total += first_only
        second_only_total += second_only
        matches_total += len(matches)

        local_differences = []
        for first_index, second_index, _ in matches:
            first_unit = first_detections[first_index].get("original_inverse_depth_unit")
            second_unit = second_detections[second_index].get("original_inverse_depth_unit")
            if first_unit and second_unit:
                difference = abs(first_unit - second_unit) / max(first_unit, second_unit)
                local_differences.append(difference)
                distance_differences.append(difference)

        first_advisory = first.get("global_advisory")
        second_advisory = second.get("global_advisory")
        if first_advisory != second_advisory:
            advisory_changes += 1
        rows.append(
            {
                "image": second.get("image"),
                "matched": len(matches),
                "first_only": first_only,
                "second_only": second_only,
                "first_advisory": first_advisory,
                "second_advisory": second_advisory,
                "severity_delta": SEVERITY.get(second_advisory, 0) - SEVERITY.get(first_advisory, 0),
                "mean_distance_difference": (
                    float(np.mean(local_differences)) if local_differences else None
                ),
            }
        )

    summary = {
        "common_images": len(common),
        "matched_detections": matches_total,
        "first_only_detections": first_only_total,
        "second_only_detections": second_only_total,
        "advisory_changes": advisory_changes,
        "median_matched_distance_difference": (
            float(np.median(distance_differences)) if distance_differences else None
        ),
        "p90_matched_distance_difference": (
            float(np.percentile(distance_differences, 90)) if distance_differences else None
        ),
    }
    args.output.write_text(
        json.dumps({"summary": summary, "images": rows}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"Comparison: {args.output.resolve()}")


if __name__ == "__main__":
    main()
