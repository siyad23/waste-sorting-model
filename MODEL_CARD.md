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

This is a YOLO11 nano object detector trained for 19 waste-related classes.
The paper groups these classes into four sorting categories: **biodegradable,
plastic, metal, and hazardous**. The full mapping is in
[README.md](README.md#waste-categories).
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
| Main sorting categories | Biodegradable, plastic, metal, hazardous |
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

## Published evaluation

The benchmark for this release is **Table 3B** of the
[paper](https://doi.org/10.1016/j.rineng.2026.111286). All values below are
percentages, transcribed from the publication.

| Metric | Training set | Validation set | Test set |
| --- | ---: | ---: | ---: |
| mAP@0.50 | 90.79% | 89.69% | 90.17% |
| mAP@0.50:0.95 | 73.37% | 71.05% | 70.43% |
| Mean precision | 90.12% | 86.44% | 85.48% |
| Mean recall | 87.99% | 84.50% | 84.99% |
| F1 score | 89.04% | 85.46% | 85.23% |

Section 2.4.3 describes precision and recall as macro-averaged across classes
and F1 as their harmonic mean. The reported test precision, recall, and F1 use
the confidence threshold that maximized F1 during validation. mAP summarizes
the precision–recall curve across confidence thresholds.

[metrics/benchmark_summary.json](metrics/benchmark_summary.json) provides the
published values, source DOI, and table reference. These are paper-reported
results; local software checks are documented in
[docs/VALIDATION.md](docs/VALIDATION.md). Training logs and the superseded local
benchmark are retained as historical records in `metrics/training_history.csv`
and [provenance/legacy_benchmark/](provenance/legacy_benchmark/README.md).

## Known limitations

- In the local dataset snapshot, `bottle_awss` has 4,001 training boxes,
  **zero validation boxes**, and only 28 test boxes in 9 test images.
  Evaluation on that local validation split cannot assess this class.
- Classes are imbalanced: `plasticPacket_awss` has 40,154 boxes overall, while
  `foodContainer_awss` has 435. See `metadata/class_distribution.csv`.
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
