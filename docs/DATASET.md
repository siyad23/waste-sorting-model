# Dataset structure, classes, and annotation specification

## Availability and provenance

The original dataset is **not included**, because this publisher does not have
distribution rights. The available local export is named
`automatedWasteSortingSystem-13`. Its source configuration identifies Roboflow
workspace `trashdetection-qbpwh`, project `automatedwastesortingsystem-uilcx`,
version 13. These are provenance identifiers, not an access grant or a promise
that the original export is publicly obtainable.

The source configuration's dataset license field is not reused as a license
for excluded images or labels. See [../NOTICE.md](../NOTICE.md).

## Composition

Counts were recomputed from local files when preparing this release. They are
a filesystem snapshot, not a reconstructed historical dataset manifest.

| Split | Images | Boxes | Images with empty labels |
| --- | ---: | ---: | ---: |
| Training (`train`) | 43,513 | 83,041 | 552 |
| Validation (`valid`, YAML key `val`) | 6,258 | 12,437 | 98 |
| Test (`test`) | 5,363 | 10,566 | 131 |
| Total | **55,134** | **106,044** | **781** |

All image files had matching label files, with no orphan label files. All
106,044 nonempty rows used five-column detection annotations with valid class
IDs, finite normalized coordinates, and positive widths/heights.

Images can contain multiple classes and multiple instances of a class. Thus
per-class image counts are overlapping **image–class memberships**, not the
number of distinct dataset images. `metadata/class_distribution.csv` includes
both image memberships and box counts for every class and split.

**Validation has no `bottle_awss` examples.** The stored partitioning was kept
for documentation; no resplitting or removal of samples was performed.
Original split criteria, source-image duplicate checks, capture conditions,
annotation review policy, and augmentation lineage were not available. Stored
image counts must not be described as verified independent original captures.

## Class IDs and meanings

The paper's **Table 2B** groups the 19 classes into **biodegradable, plastic,
metal, and hazardous** waste. See the complete
[category mapping in the README](../README.md#waste-categories).
The detector outputs the individual class IDs below; applications can map them
to the four main categories for sorting.

Use exactly the following ordering with the released weights. The `_awss`
suffix is part of each stored class name. Human-readable descriptions interpret
the names; they are **not the original annotation manual**. Exact inclusion
rules, object-part labeling, and material distinctions remain undocumented.

| ID | Stored name | Description | Train boxes | Val boxes | Test boxes |
| ---: | --- | --- | ---: | ---: | ---: |
| 0 | `aluminumCan_awss` | Aluminum beverage or food can. | 892 | 213 | 105 |
| 1 | `aluminumFoil_awss` | Sheet or crumpled aluminum foil. | 403 | 44 | 18 |
| 2 | `apple_awss` | Apple fruit. | 1,876 | 384 | 120 |
| 3 | `banana_awss` | Banana fruit or peel. | 1,904 | 249 | 197 |
| 4 | `battery_awss` | Discarded battery; chemistry is not encoded. | 2,170 | 539 | 503 |
| 5 | `bolt_awss` | Threaded bolt fastener. | 2,951 | 407 | 561 |
| 6 | `bottle_awss` | Bottle; material is not encoded by the label. | 4,001 | 0 | 28 |
| 7 | `cigarette_awss` | Cigarette or cigarette butt. | 2,747 | 497 | 479 |
| 8 | `egg_awss` | Egg or eggshell. | 706 | 53 | 200 |
| 9 | `foodContainer_awss` | Food container; material is not encoded by the label. | 311 | 55 | 69 |
| 10 | `juiceBox_awss` | Juice carton or juice box. | 1,032 | 103 | 105 |
| 11 | `leaf_awss` | Plant leaf. | 6,019 | 1,715 | 663 |
| 12 | `nut_awss` | Nut fastener, paired conceptually with bolt; not an edible nut. | 2,877 | 398 | 295 |
| 13 | `orange_awss` | Orange fruit. | 7,746 | 607 | 612 |
| 14 | `paperBag_awss` | Paper shopping or carrying bag. | 10,427 | 1,951 | 1,715 |
| 15 | `paperCup_awss` | Paper drinking cup. | 2,788 | 279 | 372 |
| 16 | `paper_awss` | Loose paper material. | 1,327 | 335 | 291 |
| 17 | `plasticFoodPacket_awss` | Plastic packet or wrapper associated with food. | 1,063 | 221 | 267 |
| 18 | `plasticPacket_awss` | Plastic packet or bag; original boundary against food packets is undocumented. | 31,801 | 4,387 | 3,966 |

In particular, the intended distinction between generic plastic packets and
food packets needs an explicit annotation policy in any new dataset. The paper
groups bottle and food-container classes under **Plastic**, although their
stored label names do not encode a material subtype. The `nut_awss` class is
grouped under **Metal** alongside `bolt_awss` in the paper.

Machine-readable labels are in [../metadata/classes.json](../metadata/classes.json),
and the training YAML is [../configs/data.yaml](../configs/data.yaml).

## Directory layout for your own data

```text
/absolute/path/to/your/waste-dataset/
├── train/
│   ├── images/
│   │   └── example_train.jpg
│   └── labels/
│       └── example_train.txt
├── valid/
│   ├── images/
│   │   └── example_valid.jpg
│   └── labels/
│       └── example_valid.txt
└── test/                            # Optional for fine-tuning
    ├── images/
    │   └── example_test.jpg
    └── labels/
        └── example_test.txt
```

Copy the template YAML, set `path` to the **absolute** dataset root, and keep
split paths relative to it. Ultralytics resolves relative root paths using its
global dataset settings; an absolute root avoids dependence on another
person's settings. `val` is the YAML key even though the folder here is `valid`.
If no test set exists, remove the `test` entry. These example filenames are
invented; no real samples or filename manifests are supplied.

## Annotation format

Use the [Ultralytics detection format](https://docs.ultralytics.com/datasets/detect/):
one text file per image with the same stem, and one row per object:

```text
class_id center_x center_y width height
```

IDs start at zero. Coordinates are normalized by image size, with values in
`[0, 1]` and positive box width/height. For pixel-space corners and an image
of width `W`, height `H`:

```text
center_x = (xmin + xmax) / (2 * W)
center_y = (ymin + ymax) / (2 * H)
width    = (xmax - xmin) / W
height   = (ymax - ymin) / H
```

An invented example for a 1000 × 800 image with an aluminum-can box from
`(100, 160)` to `(300, 480)` is:

```text
0 0.200000 0.400000 0.200000 0.400000
```

For a background image, an empty label file explicitly records no objects.
Ultralytics also permits a missing label for background images; check that
missing files are intentional before training. This model uses detection
boxes, not segmentation polygons. The archived conversion source reduces a
polygon to its axis-aligned enclosing rectangle. The available artifacts do
not identify which images originally used polygons.

## Recompute aggregate composition

This utility expects the `train/valid/test` layout above and exports counts
without exporting images, annotation rows, or filenames:

```bash
python scripts/dataset_stats.py --root /absolute/path/to/your/waste-dataset \
  --classes local-data.yaml --output-dir outputs/dataset_stats
```

It reports missing/empty labels, orphan labels, invalid rows, and counts per
class. It does not verify image decoding, duplicate captures, annotation
quality, or independence between splits. Invalid rows are counted separately
and excluded from the valid-box totals. Supply all three splits when using this
composition utility.
