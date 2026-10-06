"""Create contact sheets for benchmark reports matching an issue or advisory."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--issue", default=None)
    parser.add_argument("--advisory", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=4)
    args = parser.parse_args()

    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    rows = analysis["images"]
    if args.issue:
        rows = [row for row in rows if args.issue in row["issues"]]
    if args.advisory:
        rows = [row for row in rows if row["advisory"] == args.advisory]
    if not rows:
        raise ValueError("No benchmark rows matched the requested filters.")

    tile_width, tile_height = 480, 300
    label_height = 48
    tiles = []
    for row in rows:
        image = cv2.imread(row["image"], cv2.IMREAD_COLOR)
        if image is None:
            continue
        scale = min(tile_width / image.shape[1], tile_height / image.shape[0])
        resized = cv2.resize(
            image,
            (int(image.shape[1] * scale), int(image.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
        tile = np.full((tile_height + label_height, tile_width, 3), 20, dtype=np.uint8)
        x = (tile_width - resized.shape[1]) // 2
        y = (tile_height - resized.shape[0]) // 2
        tile[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
        title = f"{Path(row['image']).name} | objects={row['objects']} | {row['advisory']}"
        subtitle = ",".join(row["issues"]) or "no analyzer issue"
        cv2.putText(tile, title, (6, tile_height + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        cv2.putText(tile, subtitle, (6, tile_height + 39), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 210, 255), 1)
        tiles.append(tile)

    columns = min(args.columns, len(tiles))
    rows_count = int(np.ceil(len(tiles) / columns))
    blank = np.full_like(tiles[0], 20)
    while len(tiles) < rows_count * columns:
        tiles.append(blank.copy())
    sheet_rows = [
        cv2.hconcat(tiles[index : index + columns])
        for index in range(0, len(tiles), columns)
    ]
    sheet = cv2.vconcat(sheet_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), sheet)
    print(f"Saved {len(rows)} images to {args.output.resolve()}")


if __name__ == "__main__":
    main()
