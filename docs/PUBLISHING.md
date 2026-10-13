# Repository maintenance and releases

The model is published at
[siyad23/waste-sorting-model](https://github.com/siyad23/waste-sorting-model).
The default branch is `main`.

## Get the published model

```bash
git clone https://github.com/siyad23/waste-sorting-model.git
cd waste-sorting-model
python3 scripts/verify_weights.py
```

Follow [../README.md](../README.md#install) for dependency installation and
inference, fine-tuning, and evaluation commands.

## Maintain the repository

Review changes from the repository root before committing and pushing updates:

```bash
python3 scripts/verify_weights.py
git status --short
git diff --check
git diff
```

Keep [../CITATION.cff](../CITATION.cff) and [../CITATION.bib](../CITATION.bib) in
the repository root alongside the README. The CFF file identifies the paper as
the preferred citation for GitHub's **Cite this repository** feature. Preserve
the paper citation in the documentation of model releases and fine-tuned
derivatives.

Use **Table 3B** of the [paper](https://doi.org/10.1016/j.rineng.2026.111286)
as the source for the published model benchmark. Keep the README, model card,
and [../metrics/benchmark_summary.json](../metrics/benchmark_summary.json)
consistent. Label any new evaluations with their own dataset, split, weights,
and settings.

`weights/best.pt` is approximately 5.5 MB (5.23 MiB), so ordinary Git storage is
sufficient. The repository's `.gitignore` permits this checkpoint and excludes
local datasets, images/videos, generated outputs, environments, and additional
weight/export files. Keep that allowlist narrow when adding future releases.

The repository distributes aggregate dataset statistics and training records.
Original images, per-image annotations, caches, and dataset directories are
excluded. Numeric examples in [DATASET.md](DATASET.md) are invented.

## Create a model release

Use the repository's
[GitHub Releases page](https://github.com/siyad23/waste-sorting-model/releases)
to publish a versioned release. Attach `weights/best.pt` and
`weights/SHA256SUMS`, identify the associated commit, and include a link to the
paper in the release notes. Verify the checkpoint checksum before uploading;
when releasing new weights, update their checksum, model metadata, and model
card together.

Follow the AGPL-3.0 terms in [../LICENSE](../LICENSE) and retain
[../NOTICE.md](../NOTICE.md). Rights to the excluded dataset remain with its
owners.
