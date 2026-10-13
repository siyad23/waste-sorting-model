# YOLO11n Waste Sorting Model

Research model accompanying
[**AI-driven waste management system using deep learning model for detection and
novel dual-axis actuator-based waste categorization**](https://doi.org/10.1016/j.rineng.2026.111286),
published in *Results in Engineering* (2026).

This pretrained YOLO11n object detector recognizes **19 waste-related classes**,
grouped into **four main sorting categories: biodegradable, plastic, metal,
and hazardous**. It includes runnable inference, fine-tuning, and evaluation
commands. The released checkpoint is an exact copy of
`runs/detect/train22/weights/best.pt` from the original project.

**Repository:** [siyad23/waste-sorting-model](https://github.com/siyad23/waste-sorting-model)

**If you use this model, its weights, code, or a fine-tuned derivative, cite the
paper below.** Use of the released software and weights is governed by
[AGPL-3.0](LICENSE); include the research citation in publications and in the
documentation or acknowledgments of projects that use this work.

**Dataset contents are not distributed.** This repository includes class definitions,
aggregate dataset composition, the annotation specification, and training records.
Inference needs only the weights and your own input. Fine-tuning and evaluation
require a dataset you are authorized to use.

## Waste categories

The paper's sorting system groups the 19 detection classes into four main
categories, as listed in
[Table 2B](https://doi.org/10.1016/j.rineng.2026.111286):

| Main category | Detection classes | Class count |
| --- | --- | ---: |
| **Biodegradable** | Paper bag, orange, leaf, cigarette, paper cup, banana, apple, paper, juice box, egg | 10 |
| **Plastic** | Plastic packet, bottle, plastic food packet, food container | 4 |
| **Metal** | Bolt, nut, aluminum can, aluminum foil | 4 |
| **Hazardous** | Battery | 1 |

The model and inference scripts output the **19 individual class IDs and names**.
Use the grouping above when mapping detections to the paper's four-bin sorting
system. Stable class IDs and stored names are documented in
[docs/DATASET.md](docs/DATASET.md#class-ids-and-meanings).

## Cite this work

> Siyad, R. K., Islam, M. M., Nasseef, A. O., Ahmed, K. S., Hasan, M. S., &
> Mashfy, M. M. (2026). AI-driven waste management system using deep learning
> model for detection and novel dual-axis actuator-based waste categorization.
> *Results in Engineering, 30*, 111286.
> [doi:10.1016/j.rineng.2026.111286](https://doi.org/10.1016/j.rineng.2026.111286)

```bibtex
@article{siyad2026waste,
  author = {Siyad, Rahat Khan and Islam, Md Mehrabul and Nasseef, Abdullah Omar and Ahmed, Khandaker Siam and Hasan, Muhammad Samin and Mashfy, Mohammad Muhtasim},
  title = {{AI}-driven waste management system using deep learning model for detection and novel dual-axis actuator-based waste categorization},
  journal = {Results in Engineering},
  volume = {30},
  pages = {111286},
  year = {2026},
  doi = {10.1016/j.rineng.2026.111286},
  url = {https://doi.org/10.1016/j.rineng.2026.111286}
}
```

Download [CITATION.bib](CITATION.bib) for your reference manager.
[CITATION.cff](CITATION.cff) provides citation metadata for GitHub's
**Cite this repository** feature, with the paper as the preferred citation.
When reporting results from this release or your own fine-tuning, also identify
the checkpoint and evaluation setup used; see [MODEL_CARD.md](MODEL_CARD.md).

## Included files

```text
waste-sorting-model/
├── README.md
├── MODEL_CARD.md
├── CITATION.cff                     # GitHub citation metadata; preferred paper
├── CITATION.bib                     # Importable BibTeX reference
├── LICENSE                          # AGPL-3.0
├── NOTICE.md                        # Attribution and dataset rights
├── requirements.txt
├── weights/
│   ├── best.pt                      # 5,485,018 bytes; actual trained checkpoint
│   └── SHA256SUMS
├── configs/
│   ├── data.yaml                    # Portable 19-class dataset template
│   ├── architecture.yaml            # Architecture from the checkpoint
│   ├── finetune.yaml                # Suggested settings for new fine-tuning
│   └── training_recipe.yaml         # Historical training hyperparameters
├── metadata/
│   ├── classes.json                 # Stable IDs, names, and descriptions
│   ├── model.json                   # Checkpoint identity and environment
│   ├── dataset_statistics.json      # Aggregate local dataset snapshot
│   └── class_distribution.csv       # Image/box counts by class and split
├── metrics/
│   ├── benchmark_summary.json       # Paper Table 3B: train, validation, test
│   └── training_history.csv         # 100 epochs of scalar metrics
├── scripts/
│   ├── predict.py
│   ├── finetune.py
│   ├── evaluate.py
│   ├── dataset_stats.py
│   ├── verify_weights.py
│   └── _common.py
├── docs/
│   ├── DATASET.md
│   ├── TRAINING.md
│   ├── PUBLISHING.md
│   └── VALIDATION.md
└── provenance/                      # Original source and historical records
    ├── legacy_benchmark/            # Archived local benchmark, superseded by paper
    ├── original_training.py
    ├── original_polygon_to_box.py
    └── train22_args.yaml
```

## Install

Clone the published repository, then run commands from its root.
**Python 3.12** matches the tested environment; the pinned dependencies also
target Python 3.10–3.12.

```bash
git clone https://github.com/siyad23/waste-sorting-model.git
cd waste-sorting-model

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# CPU build; install PyTorch before the other dependencies.
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python scripts/verify_weights.py
```

For a compatible NVIDIA GPU/driver, substitute the CUDA 12.8 build for the CPU
installation command:

```bash
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

See the [official PyTorch installation history](https://pytorch.org/get-started/previous-versions/)
for these builds. PyTorch build selection is separate from the saved model weights.

## Run inference

Supply your own image, image folder, or video; no original dataset is needed.

```bash
python scripts/predict.py --source /path/to/image.jpg --save
python scripts/predict.py --source /path/to/images --save --output-dir outputs/images
python scripts/predict.py --source /path/to/video.mp4 --save --device 0
python scripts/predict.py --source 0 --device 0
```

The default device is **CPU**. Use `--device 0` for the first CUDA GPU. Defaults are
`--imgsz 640 --conf 0.25 --iou 0.7`; tune confidence for your application.

Every command writes `predictions.jsonl` in its output directory, with one record
per image or video frame. Each detection contains `class_id`, `class_name`,
`confidence`, pixel coordinates `[xmin, ymin, xmax, ymax]`, and normalized
`[center_x, center_y, width, height]`. An image with no detections has an empty
`detections` list. `index` is the zero-based record number in that invocation.
`--save` additionally writes annotated images/video. Reusing an output directory
replaces its JSONL file; use distinct directories to retain separate runs.

Python API:

```python
from ultralytics import YOLO

model = YOLO("weights/best.pt")
for result in model.predict("/path/to/image.jpg", device="cpu", imgsz=640):
    for box in result.boxes:
        class_id = int(box.cls.item())
        print(model.names[class_id], float(box.conf.item()), box.xyxy.tolist())
```

## Fine-tune on your dataset

1. Arrange your authorized data as described in [docs/DATASET.md](docs/DATASET.md).
2. Copy `configs/data.yaml` to a local YAML, and set `path` to your dataset's
   **absolute** root directory. Keep the 19 class IDs and names for the same task.
3. Start a new run from the released weights:

```bash
cp configs/data.yaml local-data.yaml
# Edit local-data.yaml: path: /absolute/path/to/your/waste-dataset
python scripts/finetune.py --data local-data.yaml --epochs 50 --batch 8 --device 0
```

For CPU or a smaller GPU, reduce `--batch`; CPU training can be slow. Training
outputs, including new `weights/best.pt` and `weights/last.pt`, are saved below
`outputs/train/finetune` (or an incremented run name).

The released checkpoint has no optimizer state, so this starts **new fine-tuning**
with `resume=False`. A different class set is supported with an explicit
`--allow-class-change`; read [docs/TRAINING.md](docs/TRAINING.md) before changing
class IDs or meanings.

## Evaluate

```bash
python scripts/evaluate.py --data local-data.yaml --split val --device 0
python scripts/evaluate.py --data local-data.yaml --split test --device 0 --output-dir outputs/test
python scripts/evaluate.py --data local-data.yaml --weights outputs/train/finetune/weights/best.pt --device 0
```

These commands write scalar JSON and a per-class CSV; they check that dataset and
checkpoint labels agree. Classes without ground-truth examples are left unscored.

## Performance reported in the paper

The following results are reproduced from **Table 3B** of the
[paper](https://doi.org/10.1016/j.rineng.2026.111286). All values are percentages.

| Metric | Training set | Validation set | Test set |
| --- | ---: | ---: | ---: |
| mAP@0.50 | 90.79% | 89.69% | **90.17%** |
| mAP@0.50:0.95 | 73.37% | 71.05% | **70.43%** |
| Mean precision | 90.12% | 86.44% | **85.48%** |
| Mean recall | 87.99% | 84.50% | **84.99%** |
| F1 score | 89.04% | 85.46% | **85.23%** |

Machine-readable published results are in
[metrics/benchmark_summary.json](metrics/benchmark_summary.json).
See [MODEL_CARD.md](MODEL_CARD.md#published-evaluation) for evaluation context
and [docs/VALIDATION.md](docs/VALIDATION.md) for local software checks.

## Dataset documentation and licensing

The local dataset snapshot contains **55,134 images and 106,044 bounding boxes**.
Only aggregate counts are included. No original images, image filenames,
per-image annotations, caches, collages, or training-image previews are released.
Exact training reproduction requires authorized access to the original data.

The code, configuration, documentation, and model weights in this release use
**AGPL-3.0**, preserving the licensing recorded by the checkpoint and
[Ultralytics](https://www.ultralytics.com/license). The release does not grant
rights to the excluded dataset. See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).
Include the accompanying paper citation when using this work, as described in
[Cite this work](#cite-this-work).

The model is published at
[siyad23/waste-sorting-model](https://github.com/siyad23/waste-sorting-model).
Repository maintenance and release guidance are in
[docs/PUBLISHING.md](docs/PUBLISHING.md).
