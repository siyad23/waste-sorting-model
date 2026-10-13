#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Verify the release checkpoint without loading PyTorch/pickle or needing data."""

import hashlib
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    for line in (root / "weights" / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, filename = line.split(maxsplit=1)
        path = root / "weights" / filename
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise SystemExit(f"SHA-256 mismatch: {path}")
        print(f"OK: {filename} ({path.stat().st_size:,} bytes), SHA-256 {expected}")


if __name__ == "__main__":
    main()
