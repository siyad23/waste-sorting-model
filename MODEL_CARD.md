# Model card: YOLO11n Waste Sorting

## Research paper and citation

This model was developed for Siyad et al. (2026),
[AI-driven waste management system using deep learning model for detection and
novel dual-axis actuator-based waste categorization](https://doi.org/10.1016/j.rineng.2026.111286),
*Results in Engineering*, 30, 111286.

If you use this model, its weights, code, or a fine-tuned derivative, cite the
paper in your publications and project documentation or acknowledgments.
The full reference and BibTeX are in [README.md](README.md#cite-this-work),
with importable metadata in [CITATION.bib](CITATION.bib) and
[CITATION.cff](CITATION.cff). Use of the released software and weights is
governed by [AGPL-3.0](LICENSE).

## Identity and intended use

This is a YOLO11 nano object detector trained for 19 waste-related categories.
It outputs axis-aligned boxes, class IDs, and confidence scores. Intended uses
include waste sorting research, inference experiments, and initialization for
fine-tuning on authorized data. A class prediction does not determine material
chemistry, disposal instructions, or local recyclability.

| Property | Recorded value |
| --- | --- |
| Source checkpoint | `runs/detect/train22/weights/best.pt` |
| Released checkpoint | `weights/best.pt` (byte-for-byte copy) |
| Architecture | Ultralytics YOLO11n, three detection scales (strides 8/16/32) |
| Parameters | 2,593,545 before inference fusion |
| Classes | 19; IDs and names in `metadata/classes.json` |
| Input | 3-channel images; training target size 640 |
| Checkpoint framework | Ultralytics 8.3.146 |
| Checkpoint timestamp | 2025-10-18T09:14:52.219681 (timezone not recorded) |
| File size | 5,485,018 bytes |
| License recorded by checkpoint | AGPL-3.0 |

SHA-256:

```text
f64e94c7da5f655a82b60a368add374e6e9769c0b03b97c51bfa8c27942ee09c
```

Ultralytics handles input resizing/letterboxing and inference postprocessing.
The command defaults to confidence 0.25 and NMS IoU 0.7. The nominal image size
is a resize target; rectangular inference can use a different padded shape.

## Training and data

The available history has 100 epochs. Epoch 100 has the best recorded
mAP@0.50:0.95 and matches the scalar training metrics embedded in `best.pt`.
The finalized checkpoint stores `epoch=-1`, `optimizer=None`, and
`best_fitness=None`; these stripped fields cannot independently establish the
selected epoch or support optimizer-state resume.

The original training script selects pretrained `models/yolo11n.pt` by default.
The saved arguments reference `last.pt` and a resume path, documenting a resumed
run. The initial base-weight hash, interruption history, and exact command are
unavailable. No later continuation checkpoint is included.

The dataset export is identified locally as `automatedWasteSortingSystem-13`.
A release-time scan found 43,513 training, 6,258 validation, and 5,363 test images.
These are counts of stored files, not verified unique original captures. The
original partition procedure, capture locations, source-image provenance,
augmentation lineage, and cross-split duplicate checks were not recorded in
the inspected files. The snapshot is not proof that every file is identical to
the historical training dataset.

Full hyperparameters and annotation structure are in
[docs/TRAINING.md](docs/TRAINING.md) and [docs/DATASET.md](docs/DATASET.md).

## Evaluation evidence

The following values describe the saved local benchmark and training records.
For published experimental results, consult the
[paper](https://doi.org/10.1016/j.rineng.2026.111286).

| Metric | Saved benchmark JSON | Training history, epoch 100 |
| --- | ---: | ---: |
| mAP@0.50 | 0.905153 | 0.90519 |
| mAP@0.50:0.95 | 0.722534 | 0.72269 |
| Mean precision | 0.878786 | 0.87887 |
| Mean recall | 0.854334 | 0.85405 |

The benchmark values come from the original project's
`results/benchmark_results/benchmark_metrics.json`. The original benchmark
script calls `model.val(data=...)` without a split override; validation is the
inferred split. There is no saved invocation, dataset fingerprint, or benchmark
checkpoint hash. These are historical reported metrics, not a newly verified
test-set evaluation.

The per-class arrays identify classes using `ap_class_index`. The IDs are
`0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18`.
`metrics/per_class_metrics.csv` uses those IDs, leaving class 6 unscored.
Treating array positions as class IDs would shift every label after `bolt_awss`.

No overall F1 or throughput claim is included. Taking the maximum of a per-class
F1 array is not an overall F1. The inspected records do not support a reproducible
end-to-end FPS measurement. Other narrative reports contain inconsistent labels
and scores and are not used as release evidence.

## Known limitations

- `bottle_awss` has 4,001 training boxes, **zero validation boxes**, and only
  28 test boxes in 9 test images. Validation cannot assess this class.
- Classes are imbalanced: `plasticPacket_awss` has 40,154 boxes overall, while
  `foodContainer_awss` has 435. See `metadata/class_distribution.csv`.
- Historical validation AP@0.50:0.95 is lowest for `leaf_awss` (0.3722),
  `bolt_awss` (0.3784), and `cigarette_awss` (0.4152).
- Label descriptions interpret the names; the original annotation policy is
  unavailable. Material composition and boundaries between similar packaging
  classes are not fully specified.
- Generalization to new cameras, lighting, backgrounds, mixed piles, small or
  occluded objects, and unfamiliar packaging has not been established.

## Distribution and reproducibility

The dataset owner has not granted this publisher distribution rights. Images
and annotations are excluded. This package provides the actual checkpoint,
architecture, training source, hyperparameters, scalar history, and aggregate
data structure; it supports inference and new fine-tuning without promising
bit-identical reconstruction of the original run.

The code and weights retain AGPL-3.0 licensing. Upstream source is available at
[Ultralytics v8.3.146](https://github.com/ultralytics/ultralytics/tree/v8.3.146).
See [NOTICE.md](NOTICE.md) for the distinction between this release and the
excluded dataset.
