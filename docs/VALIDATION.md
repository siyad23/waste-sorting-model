# Local release validation

Validated on **2026-10-13** using a standalone copy of this directory outside
the research project. The existing Python environment supplied dependencies;
a fresh internet-based dependency installation was not performed.

Environment: Python 3.12.3, Ultralytics 8.3.146, PyTorch 2.7.0+cu128,
torchvision 0.22.0+cu128, NumPy 1.26.4, OpenCV 4.11.0.86, and PyYAML 6.0.2.
All execution used **CPU**; CUDA execution was not tested.

## Checks performed

- The copied checkpoint matches the original byte-for-byte and passes the
  SHA-256 check in `scripts/verify_weights.py`.
- All Python files compile, JSON files parse, and YAML files parse.
- `configs/architecture.yaml` instantiates a 19-class YOLO11n with 2,593,545
  parameters, matching the checkpoint's unfused model.
- Inference on two generated images at the default resize size of 640 produced
  two JSONL records and two annotated images. Image dimensions, detection
  records, and saved output files were checked.
- One epoch of actual fine-tuning from the released weights on four generated
  training images and two generated validation images produced new `best.pt`
  and `last.pt` files. Smoke settings were image size 64, batch 2, workers 0,
  AMP disabled, plots disabled, and mosaic disabled.
- Evaluation ran with both released and fine-tuned weights. Ground truth used
  nonconsecutive class IDs **6 and 18**. CSV export scored those IDs correctly
  and left the other 17 classes unscored.
- Transfer to a new two-class label set completed a one-epoch training run
  using `--allow-class-change`. Its new checkpoint was evaluated successfully.
- A changed class mapping without the explicit flag and an unedited placeholder
  dataset root were rejected with usage errors.
- The composition utility counted the eight generated images and boxes correctly.
- Aggregate release statistics were cross-checked: 55,134 images, 106,044 boxes,
  19 class rows, and zero validation boxes for class 6.
- Git's file inventory includes `weights/best.pt` despite the parent project's
  weight-file ignore rule. The package has no original sample images, labels,
  caches, or dataset directory.

All generated fixtures, trained smoke checkpoints, predictions, and test logs
were kept outside this release directory. The original training dataset and
checkpoint were not modified. The generated geometric fixtures test software
execution and file formats; they do not measure model accuracy or generalization.

## Recheck checkpoint integrity

This uses only Python's standard library:

```bash
python scripts/verify_weights.py
```

For a substantive evaluation, use `scripts/evaluate.py` with authorized,
representative held-out data and a label mapping matching the chosen checkpoint.
