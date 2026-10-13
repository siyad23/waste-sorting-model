# Licensing, attribution, and dataset rights

## Research attribution

This model accompanies the research paper by Rahat Khan Siyad, Md Mehrabul
Islam, Abdullah Omar Nasseef, Khandaker Siam Ahmed, Muhammad Samin Hasan, and
Mohammad Muhtasim Mashfy:

[AI-driven waste management system using deep learning model for detection and
novel dual-axis actuator-based waste categorization](https://doi.org/10.1016/j.rineng.2026.111286).
*Results in Engineering*, 30 (2026), 111286.

If you use this model, its weights, code, or a fine-tuned derivative, cite this
paper in publications and project documentation or acknowledgments.
See [README.md](README.md#cite-this-work) for the formatted reference and
[CITATION.bib](CITATION.bib) or [CITATION.cff](CITATION.cff) for citation metadata.
Use of the released software and weights is governed by the license below.

## Released software and model

This release's code, configuration, documentation, and trained weights are
distributed under the GNU Affero General Public License version 3
(`AGPL-3.0-only`). The complete license text is in [LICENSE](LICENSE).

The checkpoint itself records
`AGPL-3.0 (https://ultralytics.com/license)`. YOLO11 architecture and framework
are provided by Ultralytics. Upstream source for the recorded version:

- [Ultralytics v8.3.146](https://github.com/ultralytics/ultralytics/tree/v8.3.146)
- [Upstream license](https://github.com/ultralytics/ultralytics/blob/v8.3.146/LICENSE)
- [Ultralytics licensing information](https://www.ultralytics.com/license)

The architecture YAML is extracted from the trained checkpoint and retains
upstream attribution. Original project training and conversion scripts are
preserved under `provenance/`. Dependencies retain their respective licenses.

## Excluded dataset

The publisher does not have distribution rights to the original dataset.
**No images or original annotations are licensed or distributed here.**
Class names, aggregate composition, a dataset configuration template, and format
documentation describe the training setup without providing dataset contents.

The original export configuration contains a dataset-level `license: MIT`
field. It is not treated as permission to redistribute third-party source
images or annotations, and that field has intentionally been omitted from the
portable dataset configuration. No dataset download script is included.
Rights to excluded data remain with their respective owners.

The examples in the annotation documentation are newly invented numeric
examples. Local validation uses generated images, not samples from the
restricted dataset. Consumers must independently obtain authorized data for
fine-tuning or evaluation.
