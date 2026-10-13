# Publish this directory as a separate GitHub repository

This folder is self-contained. Publish **only this folder**, not the parent
research repository: the parent contains restricted data and unrelated
artifacts. No Git repository has been initialized here and nothing has been
pushed by preparing these files.

Copy the directory to the location intended for your new repository, then run
the following commands from the copied directory. Replace the GitHub remote
with your own newly created, empty repository.

```bash
python scripts/verify_weights.py
git init -b main
git add .
git diff --cached --stat
git commit -m "Release YOLO11n waste sorting weights and reproducibility documentation"
git remote add origin https://github.com/YOUR_USERNAME/waste-sorting-model.git
git push -u origin main
```

`weights/best.pt` is approximately 5.5 MB (5.23 MiB), so ordinary Git storage is
sufficient for this checkpoint. The nested `.gitignore` explicitly permits this
one weight file, including when the parent project ignores `*.pt`. It ignores
local datasets, images/videos, generated outputs, environments, and additional
weight/export files. Keep that allowlist narrow when adding future releases.

The package deliberately has no dataset directory, per-image labels, actual
sample images, caches, image manifests, or collages. Numeric examples in
`docs/DATASET.md` are invented. Scalar training histories and aggregate
statistics are included.

Keep [../CITATION.cff](../CITATION.cff) and [../CITATION.bib](../CITATION.bib) in
the new repository's root, alongside the README. The CFF file identifies the
research paper as the preferred citation for GitHub's **Cite this repository**
feature. Retain the README's citation instructions when publishing the model
or a fine-tuned derivative, and cite the
[paper](https://doi.org/10.1016/j.rineng.2026.111286) in the project documentation
or acknowledgments.

An optional GitHub Release can attach `weights/best.pt` and `weights/SHA256SUMS`
under a version tag. Preserve the checksum when copying the checkpoint. Follow
the AGPL-3.0 terms in [../LICENSE](../LICENSE) and retain
[../NOTICE.md](../NOTICE.md); the excluded dataset's rights are not granted by
this repository.
