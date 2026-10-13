# SPDX-License-Identifier: AGPL-3.0-only
"""Shared local-file and dataset checks for the release commands."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "weights" / "best.pt"


def existing_file(value):
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"File does not exist: {path}")
    return path


def read_yaml(path):
    with existing_file(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def class_names(data):
    names = data.get("names")
    if isinstance(names, list):
        names = dict(enumerate(names))
    elif isinstance(names, dict):
        names = {int(key): value for key, value in names.items()}
    else:
        raise ValueError("Dataset YAML must define names as a list or ID-to-name mapping.")
    if not names or sorted(names) != list(range(len(names))):
        raise ValueError("Class IDs must be consecutive integers starting at zero.")
    if any(not isinstance(name, str) or not name for name in names.values()):
        raise ValueError("Each class name must be a nonempty string.")
    if len(set(names.values())) != len(names):
        raise ValueError("Class names must be unique.")
    if data.get("nc", len(names)) != len(names):
        raise ValueError("nc must match the number of class names.")
    return names


def dataset_config(path, splits):
    """Require explicit local paths; never execute a YAML download recipe."""
    path = existing_file(path)
    data = read_yaml(path)
    names = class_names(data)
    if "download" in data:
        raise ValueError("Remove the download entry; these commands use locally supplied data.")
    root = Path(str(data.get("path", ""))).expanduser()
    if not root.is_absolute() or not root.is_dir():
        raise ValueError("Set path in your dataset YAML to an existing ABSOLUTE dataset directory.")
    for split in splits:
        value = data.get(split)
        if not value:
            raise ValueError(f"Dataset YAML is missing the {split} split.")
        for entry in value if isinstance(value, list) else [value]:
            target = root / str(entry)
            if not target.exists():
                raise ValueError(f"Dataset {split} path does not exist: {target}")
    return path, names
