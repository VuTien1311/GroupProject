"""Select a reproducible difficult static-image benchmark from BDD100K."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


CATEGORIES = ("dark", "low_contrast", "blur", "glare", "complex")


def image_metrics(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (320, 180), interpolation=cv2.INTER_AREA)
    brightness = float(small.mean())
    contrast = float(small.std())
    blur = float(cv2.Laplacian(small, cv2.CV_64F).var())
    glare = float((small >= 245).mean())
    darkness = float((small <= 35).mean())
    edges = cv2.Canny(small, 60, 160)
    complexity = float((edges > 0).mean())
    return {
        "path": str(path),
        "brightness": brightness,
        "contrast": contrast,
        "blur": blur,
        "glare": glare,
        "darkness": darkness,
        "complexity": complexity,
    }


def normalize(values):
    values = np.asarray(values, dtype=np.float64)
    low, high = np.percentile(values, [2, 98])
    return np.clip((values - low) / max(high - low, 1e-9), 0.0, 1.0)


def assign_scores(records):
    brightness = normalize([item["brightness"] for item in records])
    contrast = normalize([item["contrast"] for item in records])
    blur = normalize([item["blur"] for item in records])
    glare = normalize([item["glare"] for item in records])
    darkness = normalize([item["darkness"] for item in records])
    complexity = normalize([item["complexity"] for item in records])
    for index, item in enumerate(records):
        item["scores"] = {
            "dark": float(0.65 * darkness[index] + 0.35 * (1.0 - brightness[index])),
            "low_contrast": float(1.0 - contrast[index]),
            "blur": float(1.0 - blur[index]),
            "glare": float(0.75 * glare[index] + 0.25 * brightness[index]),
            "complex": float(complexity[index]),
        }


def select_balanced(records, count):
    selected = []
    used = set()
    per_category = count // len(CATEGORIES)
    quotas = {name: per_category for name in CATEGORIES}
    for name in CATEGORIES[: count % len(CATEGORIES)]:
        quotas[name] += 1

    for category in CATEGORIES:
        ranked = sorted(
            records,
            key=lambda item: item["scores"][category],
            reverse=True,
        )
        added = 0
        for item in ranked:
            if item["path"] in used:
                continue
            chosen = dict(item)
            chosen["category"] = category
            chosen["difficulty_score"] = chosen["scores"][category]
            selected.append(chosen)
            used.add(item["path"])
            added += 1
            if added >= quotas[category]:
                break
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "images" / "100k" / "val",
    )
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("static_training") / "hard_bdd100k",
    )
    args = parser.parse_args()

    paths = sorted(args.images.rglob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No JPG images found under {args.images}")

    records = []
    for index, path in enumerate(paths, start=1):
        metrics = image_metrics(path)
        if metrics:
            records.append(metrics)
        if index % 1000 == 0:
            print(f"Scanned {index}/{len(paths)} images")

    assign_scores(records)
    selected = select_balanced(records, args.count)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    list_path = args.output_dir / "images.txt"
    list_path.write_text("\n".join(item["path"] for item in selected) + "\n", encoding="utf-8")

    csv_path = args.output_dir / "manifest.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "path",
            "category",
            "difficulty_score",
            "brightness",
            "contrast",
            "blur",
            "glare",
            "darkness",
            "complexity",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in selected:
            writer.writerow({field: item[field] for field in fields})

    print(f"Selected {len(selected)} difficult images")
    for category in CATEGORIES:
        print(f"  {category}: {sum(item['category'] == category for item in selected)}")
    print(f"Image list: {list_path.resolve()}")


if __name__ == "__main__":
    main()
