# Training methodology and fine-tuning

## Evidence for the released run

`provenance/train22_args.yaml` preserves the saved arguments exactly, including
historical paths and the resume setting. `provenance/original_training.py` is
the original project's training implementation (`train_backup.py`), which the
interactive `train.py` invoked. `metrics/training_history.csv` contains the
100 recorded epoch rows. The checkpoint embeds matching epoch-100 scalar
metrics and Ultralytics version 8.3.146.

The original script defaults to pretrained `models/yolo11n.pt`. The saved run
arguments instead reference `runs/detect/train22/weights/last.pt`, because the
run was resumed. Initial base-weight hash, interruption chronology, original
invocation, exact data fingerprint, and original package lockfile are not
available. This package documents the evidence rather than claiming exact
reconstruction of an uninterrupted original run.

## Architecture and objective

YOLO11n takes three-channel images and predicts 19 classes with axis-aligned
boxes at strides 8, 16, and 32. The model has 2,593,545 parameters before
inference fusion. `configs/architecture.yaml` is extracted from the checkpoint
and can be loaded with Ultralytics 8.3.146. It records the nano scaling,
convolution/C3k2 backbone, SPPF/C2PSA blocks, upsample/concatenation head, and
three-scale Detect output.

Training uses the framework's detection loss with weights `box=7.5`,
`cls=0.5`, and `dfl=1.5`. `nbs=64` is the nominal batch-size setting; it is
distinct from the stored training batch size of 29.

## Saved settings

| Setting | Value |
| --- | --- |
| Epoch budget / early-stopping patience | 100 / 50 |
| Image size / batch | 640 / 29 |
| Optimizer | AdamW |
| Initial learning rate / final multiplier | 0.01 / 0.01 |
| Momentum (Adam beta1 setting) | 0.937 |
| Weight decay | 0.0005 |
| Warmup | 3 epochs, momentum 0.8, bias LR 0.1 |
| LR scheduling | `cos_lr=False`; framework default linear schedule |
| Seed / deterministic mode | 0 / true |
| Device / workers | CUDA device 0 / 8 |
| Caching / mixed precision | Disk / AMP enabled |
| Fraction / multi-scale / rectangular training | 1.0 / false / false |
| Frozen layers | None |
| Validation | Enabled during training, `val` split |
| Saves | Best/last plus periodic saves every 10 epochs |

The existing project README reports an RTX 4070 Laptop GPU with 8 GB VRAM.
The run arguments only establish device `0`; GPU identity was not embedded
in the inspected checkpoint. The release was locally validated on CPU with
Python 3.12.3, PyTorch 2.7.0+cu128, and Ultralytics 8.3.146; that is a verified
loading environment, not a recovered complete historical environment.

The logged learning rate at epoch 100 is `0.000199`. This is recorded history,
not an assumption that the final multiplier is already reached in that row.

## Augmentation and preprocessing

| Parameter | Value |
| --- | --- |
| HSV hue / saturation / value | 0.015 / 0.7 / 0.4 |
| Rotation | ±10 degrees |
| Translation / scale gain | 0.1 / 0.5 |
| Shear / perspective | 0 / 0 |
| Vertical / horizontal flip probability | 0.5 / 0.5 |
| Mosaic / mixup probability | 1.0 / 0.1 |
| Disable mosaic for final epochs | 10 |
| Copy-paste / cutmix / BGR swapping | 0 / 0 / 0 |

Framework detection preprocessing handles image loading, resizing/letterboxing,
pixel scaling, batching, and configured augmentations. The saved full arguments
also contain settings for other Ultralytics tasks; these are not evidence that
classification or segmentation augmentations were used for this detector.

All locally inspected labels are five-column detection boxes. An original
polygon-to-box utility is archived as
`provenance/original_polygon_to_box.py`; its enclosing-box conversion is
described in [DATASET.md](DATASET.md). Sample-level conversion history is unknown.

## Checkpoint selection and recorded results

Epoch 100 has the largest mAP@0.50:0.95 in the available history and matches
the training metrics in the released checkpoint. Its recorded precision is
0.87887, recall 0.85405, mAP@0.50 0.90519, and mAP@0.50:0.95 0.72269.
Historical benchmark values are documented separately in
[../MODEL_CARD.md](../MODEL_CARD.md); they are not verified independent test
results. Missing validation coverage for class 6 limits checkpoint assessment.

## Fine-tune from the released weights

```bash
python scripts/finetune.py --data local-data.yaml \
  --config configs/finetune.yaml --device 0 --epochs 50 --batch 8
```

The supplied fine-tuning recipe is a **new suggested starting point**: AdamW,
LR 0.001, 50 epochs, patience 15, batch 8, image size 640, and four workers.
It is distinct from the historical recipe. Use validation to choose settings
for your data. Additional Ultralytics hyperparameters can be placed in a copied
config YAML. Command-line epochs, batch, image size, workers, and LR override
that file. Model/data/device/output selection is done through CLI flags.

The finalized `best.pt` has no optimizer or scheduler state. The script loads
its parameters and uses `resume=False` to start a fresh optimizer and output
directory. Original weights are not overwritten. For future interrupted runs,
Ultralytics resume requires a suitable unfinished training checkpoint rather
than this finalized release checkpoint.

For your own data with the same task, preserve all 19 names and ID meanings.
New class sets or intentionally reordered/renamed labels require:

```bash
python scripts/finetune.py --data your-new-classes.yaml --allow-class-change --device 0
```

With a changed class count, Ultralytics rebuilds the detection head and
transfers compatible parameters. With the same class count but changed
semantics/order, output parameters may retain their old class associations;
the flag acknowledges that mismatch, and those outputs need retraining.
The script does not automatically remap head weights by label name. Evaluate
the resulting fine-tuned checkpoint against its own new label mapping.

## Retrain using the historical hyperparameters

`configs/training_recipe.yaml` retains relevant training parameters while
removing the original machine paths, device selection, and resume instruction.
If you independently obtain a compatible initial YOLO11n checkpoint and
authorized data, use:

```bash
python scripts/finetune.py --weights /absolute/path/to/yolo11n.pt \
  --data local-data.yaml --config configs/training_recipe.yaml \
  --allow-class-change --device 0 --name retrain
```

The class-change flag is required because the upstream initial checkpoint has
a different label set. Passing the released `best.pt` instead would perform
additional fine-tuning. Reusing the recipe cannot guarantee identical metrics
without the original data and environment.

## Archived source

Files in `provenance/` are historical reference material. Use `scripts/` for
the portable interface. The original interactive training script has extra
analysis dependencies and assumptions about CUDA and project paths. The
original polygon converter depends on `supervision` and modifies labels in
place; it is not needed for inference or training on box annotations. These
archived tools are not part of the release's supported command interface.
