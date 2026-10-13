#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Run detection on images, folders, videos, URLs, or a webcam."""

import argparse
import json
from pathlib import Path

from _common import DEFAULT_WEIGHTS, existing_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Image/folder/video/URL, or webcam index (0)")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, type=Path)
    parser.add_argument("--device", default="cpu", help="cpu, 0, or another PyTorch device")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/predict"))
    parser.add_argument("--save", action="store_true", help="Also save annotated images/video")
    args = parser.parse_args()
    if not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1 or args.imgsz <= 0:
        parser.error("conf and iou must be in [0, 1]; imgsz must be positive.")
    try:
        weights = existing_file(args.weights)
    except ValueError as error:
        parser.error(str(error))

    from ultralytics import YOLO

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights), task="detect")
    source = int(args.source) if args.source.isdecimal() else args.source
    results = model.predict(
        source=source, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
        device=args.device, stream=True, save=args.save,
        project=str(output.parent), name=output.name, exist_ok=True,
    )
    count = 0
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for index, result in enumerate(results):
            detections = []
            boxes = result.boxes
            for cls, conf, xyxy, xywhn in zip(
                boxes.cls.cpu().tolist(), boxes.conf.cpu().tolist(),
                boxes.xyxy.cpu().tolist(), boxes.xywhn.cpu().tolist(),
            ):
                detections.append({
                    "class_id": int(cls), "class_name": result.names[int(cls)],
                    "confidence": conf, "xyxy_pixels": xyxy, "xywh_normalized": xywhn,
                })
            record = {"index": index, "source": str(result.path),
                      "height": result.orig_shape[0], "width": result.orig_shape[1],
                      "detections": detections}
            handle.write(json.dumps(record, allow_nan=False) + "\n")
            handle.flush()
            count += 1
    print(f"Wrote {count} image/frame records to {output / 'predictions.jsonl'}")


if __name__ == "__main__":
    main()
