"""Run the recovered three-model pipeline with the PDF model profile.

Does not alter the recovered source or launch training. Exact paper result
reproduction is not established; see README.md for the missing experiment data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def load_config():
    return json.loads((ROOT / "paper_config.json").read_text(encoding="utf-8"))


def pipeline_arguments(args):
    cfg = load_config()
    command = [
        str(ROOT / "test3.py"),
        *[str(Path(p).resolve()) for p in args.images],
        "--yolo-weight", str(ROOT / cfg["yolo_weight"]),
        "--aux-yolo-weight", "",
        "--conf", str(cfg["confidence"]),
        "--imgsz", str(cfg["image_size"]),
        "--yolo-imgszs", ",".join(map(str, cfg["yolo_imgszs"])),
        "--depth-size", str(cfg["depth_size"]),
        "--distance-mode", cfg["distance_mode"],
        "--enhance-input", cfg["enhance_input"],
        "--yolo-tta", cfg["yolo_tta"],
        "--tile-inference", cfg["tile_inference"],
        "--precision-mode", cfg["precision_mode"],
        "--no-adaptive-conf",
        "--output-dir", str(Path(args.output_dir).resolve()),
    ]
    if args.image_list:
        command.extend(["--image-list", str(Path(args.image_list).resolve())])
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    return command


def check_weights(verify_hash=False):
    manifest = json.loads((ROOT / "weights_manifest.json").read_text(encoding="utf-8"))
    missing = []
    for entry in manifest["models"]:
        path = ROOT / entry["path"]
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(str(path))
            continue
        if entry.get("bytes") and path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Checkpoint size mismatch: {path}")
        if verify_hash and entry.get("sha256"):
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest().lower() != entry["sha256"].lower():
                raise ValueError(f"Checkpoint SHA-256 mismatch: {path}")
        print(f"WEIGHT_OK: {entry['path']}")
    if missing:
        raise FileNotFoundError("Missing pretrained checkpoints:\n" + "\n".join(missing))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="*")
    parser.add_argument("--image-list")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "paper"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dry-run", action="store_true", help="Print settings without loading models.")
    parser.add_argument("--check", action="store_true", help="Check imports and weight hashes without inference.")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if not (args.images or args.image_list or args.check or args.dry_run):
        parser.error("Supply image paths, a directory, or --image-list.")
    return args


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config()
    command = pipeline_arguments(args)
    print(json.dumps({"config": cfg, "device": args.device, "pipeline_argv": command}, indent=2))
    if args.dry_run:
        return 0
    check_weights(verify_hash=args.check)
    if args.image_list and not Path(args.image_list).is_file():
        raise FileNotFoundError(args.image_list)
    for image in args.images:
        if not Path(image).exists():
            raise FileNotFoundError(image)

    sys.path.insert(0, str(ROOT))
    import test3
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Use --device cpu or configure CUDA.")
    if args.device != "auto":
        test3.DEVICE = args.device
        test3.YOLO_DEVICE = 0 if args.device == "cuda" else "cpu"
    if args.check:
        print(f"IMPORT_OK | torch={torch.__version__} | device={test3.DEVICE}")
        return 0

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Keep configuration separate from per-image reports for the audit tools.
    run_info = output.parent / f"{output.name}_run_config.json"
    run_info.write_text(json.dumps({"config": cfg, "device": test3.DEVICE, "pipeline_argv": command}, indent=2), encoding="utf-8")
    old_argv, old_cwd = sys.argv, Path.cwd()
    try:
        sys.argv = command
        os.chdir(ROOT)
        test3.main()
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
