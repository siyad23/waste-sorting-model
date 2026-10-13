#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Evaluate on your authorized validation/test split and export correct class IDs."""

import argparse
import csv
import json
from pathlib import Path

from _common import DEFAULT_WEIGHTS, dataset_config, existing_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluate"))
    args = parser.parse_args()
    if args.batch <= 0 or args.imgsz <= 0 or args.workers < 0:
        parser.error("batch and imgsz must be positive; workers must be nonnegative.")
    try:
        data, names = dataset_config(args.data, (args.split,))
        weights = existing_file(args.weights)
    except (ValueError, TypeError) as error:
        parser.error(str(error))

    from ultralytics import YOLO

    model = YOLO(str(weights), task="detect")
    if model.names != names:
        parser.error("Dataset class IDs/names must match the weights for meaningful evaluation.")
    output = args.output_dir.expanduser().resolve()
    # Disable plots: validation previews contain images from the supplied dataset.
    results = model.val(data=str(data), split=args.split, device=args.device,
                        batch=args.batch, imgsz=args.imgsz, workers=args.workers,
                        project=str(output.parent), name=output.name, exist_ok=True,
                        plots=False, save_json=False, save_txt=False)
    output.mkdir(parents=True, exist_ok=True)
    summary = {"split": args.split, "map50": float(results.box.map50),
               "map50_95": float(results.box.map), "precision": float(results.box.mp),
               "recall": float(results.box.mr),
               "evaluated_class_ids": [int(i) for i in results.box.ap_class_index]}
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    measured = {int(cls): row for row, cls in enumerate(results.box.ap_class_index)}
    with (output / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_id", "class_name", "status", "ap50", "ap50_95", "precision", "recall"])
        for cls, name in sorted(names.items()):
            if cls not in measured:
                writer.writerow([cls, name, "no_ground_truth", "", "", "", ""])
            else:
                row = measured[cls]
                writer.writerow([cls, name, "measured", float(results.box.ap50[row]),
                                 float(results.box.ap[row]), float(results.box.p[row]), float(results.box.r[row])])
    print(json.dumps(summary, indent=2))
    print(f"Metrics written to {output}")


if __name__ == "__main__":
    main()
