"""Analyze static-image ADAS reports for consistency and likely failure modes."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


SEVERITY = {"CLEAR": 0, "MONITOR": 1, "SLOW": 2, "BRAKE": 3, "STOP": 4}


def rank_correlation(xs, ys):
    if len(xs) < 3 or np.std(xs) < 1e-6 or np.std(ys) < 1e-6:
        return None
    x_rank = np.argsort(np.argsort(np.asarray(xs)))
    y_rank = np.argsort(np.argsort(np.asarray(ys)))
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def analyze_report(path):
    report = json.loads(path.read_text(encoding="utf-8"))
    detections = report.get("detections", [])
    suppressed_detections = report.get("suppressed_detections", [])
    issues = []
    statuses = Counter(item.get("original_unit_status", "MISSING") for item in detections)
    low_confidence = sum(item.get("original_unit_confidence", 0.0) < 0.35 for item in detections)
    descriptions = sum(
        bool(item.get("displayed_distance") and item.get("distance_band"))
        for item in detections
    )

    bottom_rows = []
    units = []
    for item in detections:
        unit = item.get("original_inverse_depth_unit")
        box = item.get("box_xyxy")
        if unit is not None and box and item.get("original_unit_status") == "STABLE":
            bottom_rows.append(box[3])
            units.append(unit)
    perspective_correlation = rank_correlation(bottom_rows, [-value for value in units])

    if not detections:
        issues.append("NO_DETECTIONS")
    if descriptions != len(detections):
        issues.append("MISSING_DESCRIPTIONS")
    if detections and statuses["UNCERTAIN"] / len(detections) >= 0.25:
        issues.append("HIGH_UNCERTAIN_RATIO")
    if detections and low_confidence / len(detections) >= 0.25:
        issues.append("LOW_DISTANCE_CONFIDENCE")
    if perspective_correlation is not None and perspective_correlation < -0.20:
        issues.append("DEPTH_ORDER_CONFLICT")

    close_side_stop = any(
        item.get("corridor_relation") == "SIDE"
        and SEVERITY.get(item.get("advisory"), 0) >= SEVERITY["BRAKE"]
        for item in detections
    )
    if close_side_stop:
        issues.append("SIDE_OBJECT_BRAKE")

    calibration = report.get("distance_calibration", {})
    if calibration.get("metric_geometry_trusted") is False:
        trusted_metric_displayed = any(
            str(item.get("displayed_distance", "")).endswith("m")
            for item in detections
        )
        if trusted_metric_displayed:
            issues.append("UNTRUSTED_METRIC_DISPLAY")

    scene_quality = report.get("scene_quality", {})
    scene_status = scene_quality.get("status")
    depth_scan = report.get("depth_corridor_scan", {})
    advisory_confidence = report.get("global_advisory_confidence")
    if not depth_scan.get("used_without_yolo_boxes"):
        issues.append("DEPTH_SCAN_MISSING")
    if scene_status == "DEGRADED":
        issues.append("DEGRADED_SCENE")
    if (
        advisory_confidence is not None
        and SEVERITY.get(report.get("global_advisory"), 0) >= SEVERITY["SLOW"]
        and advisory_confidence < 0.30
    ):
        issues.append("LOW_CONFIDENCE_ADVISORY")

    return {
        "report": str(path),
        "image": report.get("image"),
        "objects": len(detections),
        "suppressed": len(suppressed_detections),
        "advisory": report.get("global_advisory"),
        "stable": statuses["STABLE"],
        "check": statuses["CHECK"],
        "uncertain": statuses["UNCERTAIN"],
        "low_confidence": low_confidence,
        "described": descriptions,
        "perspective_correlation": perspective_correlation,
        "scene_status": scene_status,
        "scene_score": scene_quality.get("score"),
        "depth_scan_status": depth_scan.get("status"),
        "depth_scan_confidence": depth_scan.get("confidence"),
        "advisory_confidence": advisory_confidence,
        "issues": issues,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = [analyze_report(path) for path in sorted(args.reports.glob("*.json"))]
    if not rows:
        raise FileNotFoundError(f"No JSON reports found in {args.reports}")

    issue_counts = Counter(issue for row in rows for issue in row["issues"])
    advisories = Counter(row["advisory"] for row in rows)
    total_objects = sum(row["objects"] for row in rows)
    total_suppressed = sum(row["suppressed"] for row in rows)
    fully_described = sum(row["described"] for row in rows)
    stable = sum(row["stable"] for row in rows)
    check = sum(row["check"] for row in rows)
    uncertain = sum(row["uncertain"] for row in rows)
    zero_detection = sum(row["objects"] == 0 for row in rows)
    scene_statuses = Counter(row["scene_status"] or "NOT_REPORTED" for row in rows)
    depth_scan_statuses = Counter(
        row["depth_scan_status"] or "NOT_REPORTED"
        for row in rows
    )

    summary = {
        "images": len(rows),
        "objects": total_objects,
        "suppressed_detections": total_suppressed,
        "fully_described": fully_described,
        "description_coverage": fully_described / max(total_objects, 1),
        "stable": stable,
        "check": check,
        "uncertain": uncertain,
        "stable_ratio": stable / max(total_objects, 1),
        "zero_detection_images": zero_detection,
        "scene_statuses": dict(scene_statuses),
        "depth_scan_statuses": dict(depth_scan_statuses),
        "advisories": dict(advisories),
        "issue_counts": dict(issue_counts),
    }
    output = args.output or args.reports / "benchmark_analysis.json"
    output.write_text(
        json.dumps({"summary": summary, "images": rows}, indent=2),
        encoding="utf-8",
    )

    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "image",
            "objects",
            "suppressed",
            "advisory",
            "stable",
            "check",
            "uncertain",
            "low_confidence",
            "described",
            "perspective_correlation",
            "scene_status",
            "scene_score",
            "depth_scan_status",
            "depth_scan_confidence",
            "advisory_confidence",
            "issues",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            record = dict(row)
            record["issues"] = ",".join(record["issues"])
            writer.writerow({field: record.get(field) for field in fields})

    print(json.dumps(summary, indent=2))
    print(f"Analysis: {output.resolve()}")


if __name__ == "__main__":
    main()
