#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Start new training from local weights on your own YOLO detection dataset."""

import argparse
from pathlib import Path

from _common import DEFAULT_WEIGHTS, ROOT, class_names, dataset_config, existing_file, read_yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Your dataset YAML with an absolute path")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "finetune.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--lr0", type=float)
    parser.add_argument("--device", default="cpu", help="cpu, 0, or another PyTorch device")
    parser.add_argument("--project", type=Path, default=Path("outputs/train"))
    parser.add_argument("--name", default="finetune")
    parser.add_argument("--allow-class-change", action="store_true",
                        help="Explicitly allow a new label set/order; see docs/TRAINING.md")
    args = parser.parse_args()
    try:
        data, names = dataset_config(args.data, ("train", "val"))
        weights = existing_file(args.weights)
        settings = read_yaml(args.config)
        forbidden = {"model", "data", "resume", "task", "mode", "device", "project", "name"}
        if forbidden.intersection(settings):
            raise ValueError(f"Use CLI flags for {', '.join(sorted(forbidden))}; remove them from --config.")
        canonical = class_names(read_yaml(ROOT / "configs" / "data.yaml"))
        if names != canonical and not args.allow_class_change:
            raise ValueError("Class IDs/names differ from the released model. Preserve configs/data.yaml "
                             "or explicitly use --allow-class-change for a new label set.")
        for key in ("epochs", "batch", "imgsz", "workers", "lr0"):
            if getattr(args, key) is not None:
                settings[key] = getattr(args, key)
        for key in ("epochs", "imgsz"):
            if settings.get(key, 1) <= 0:
                raise ValueError(f"{key} must be positive.")
        if settings.get("batch", 1) == 0 or settings.get("batch", 1) < -1:
            raise ValueError("batch must be positive, or -1 for GPU auto-batch.")
        if settings.get("workers", 0) < 0 or settings.get("lr0", 0.001) <= 0:
            raise ValueError("workers must be nonnegative and lr0 must be positive.")
    except (ValueError, TypeError) as error:
        parser.error(str(error))

    from ultralytics import YOLO

    model = YOLO(str(weights), task="detect")
    if model.names != names and not args.allow_class_change:
        parser.error("The supplied weights have a different label mapping. Use --allow-class-change "
                     "only when intentionally transferring to a new label set.")
    if model.names != names:
        print("Transferring to a different class mapping. Class-specific outputs need retraining.")
    # The released best.pt has no optimizer state. Always start a new run.
    model.train(data=str(data), resume=False, device=args.device,
                project=str(args.project.expanduser().resolve()), name=args.name,
                exist_ok=False, **settings)
    print(f"Training outputs: {model.trainer.save_dir}")


if __name__ == "__main__":
    main()
