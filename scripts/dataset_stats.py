#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Export aggregate composition of a local YOLO dataset, without copying samples."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from _common import class_names, read_yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".dng", ".heic", ".pfm", ".mpo"}


def summarize_split(root, split, nc):
    images_dir, labels_dir = root / split / "images", root / split / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise ValueError(f"Expected {split}/images and {split}/labels under dataset root.")
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    counts, images_by_class = Counter(), Counter()
    missing = empty = invalid = 0
    for image in images:
        label = labels_dir / f"{image.stem}.txt"
        if not label.exists():
            missing += 1
            continue
        lines = [line.split() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            empty += 1
        seen = set()
        for parts in lines:
            try:
                cls = int(parts[0])
                coords = list(map(float, parts[1:]))
                if len(coords) != 4 or not 0 <= cls < nc:
                    raise ValueError
                if not all(math.isfinite(v) and 0 <= v <= 1 for v in coords) or min(coords[2:]) <= 0:
                    raise ValueError
                counts[cls] += 1
                seen.add(cls)
            except (ValueError, IndexError):
                invalid += 1
        images_by_class.update(seen)
    label_stems = {p.stem for p in labels_dir.glob("*.txt")}
    return {"images": len(images), "label_files": len(label_stems),
            "bounding_boxes": sum(counts.values()), "empty_label_images": empty,
            "missing_label_images": missing,
            "orphan_label_files": len(label_stems - {p.stem for p in images}),
            "invalid_annotation_rows": invalid,
            "instances_by_class": {i: counts[i] for i in range(nc)},
            "images_by_class": {i: images_by_class[i] for i in range(nc)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Local dataset root")
    parser.add_argument("--classes", type=Path, required=True, help="Dataset YAML defining class names")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dataset_stats"))
    args = parser.parse_args()
    try:
        names = class_names(read_yaml(args.classes))
        splits = {key: summarize_split(args.root.expanduser().resolve(), folder, len(names))
                  for key, folder in (("train", "train"), ("val", "valid"), ("test", "test"))}
    except ValueError as error:
        parser.error(str(error))
    total_images = sum(s["images"] for s in splits.values())
    for split in splits.values():
        split["image_fraction"] = split["images"] / total_images if total_images else 0
    summary = {"dataset_identifier": args.root.name,
               "snapshot_time_utc": datetime.now(timezone.utc).isoformat(),
               "method": "Count immediate image files and matching YOLO box labels in train/valid/test; no filenames or samples exported.",
               "scope": "Local filesystem snapshot; historical training dataset identity is not independently verified.",
               "num_classes": len(names), "total_images": total_images,
               "total_bounding_boxes": sum(s["bounding_boxes"] for s in splits.values()),
               "splits": splits}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "dataset_statistics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "class_distribution.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_id", "class_name", "train_images", "train_boxes", "val_images", "val_boxes",
                         "test_images", "test_boxes", "total_image_class_memberships", "total_boxes"])
        for cls, name in sorted(names.items()):
            values = [v for s in splits.values() for v in (s["images_by_class"][cls], s["instances_by_class"][cls])]
            writer.writerow([cls, name, *values, sum(values[0::2]), sum(values[1::2])])
    print(f"Exported aggregate statistics: {total_images:,} images, {summary['total_bounding_boxes']:,} boxes")


if __name__ == "__main__":
    main()
