"""Tamper evidence for the shipped model weights.

``train.py`` writes ``models/SHA256SUMS`` (standard ``sha256sum`` format) after
saving the weights; the dashboard verifies the files against it at start-up and
shows the result, so an analyst on an air-gapped machine can tell whether the
weights are the ones that produced the published benchmark.

    python -m src.integrity          # verify (same as: sha256sum -c models/SHA256SUMS)
"""

from __future__ import annotations

import hashlib
import os
import sys
from typing import Dict, Iterable


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def write_checksums(paths: Iterable[str], out: str) -> None:
    lines = [f"{sha256_file(p)}  {os.path.basename(p)}" for p in paths]
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


def verify(paths: Iterable[str], sums_path: str) -> Dict[str, object]:
    """Compare each file with its entry in ``sums_path``; ``verified`` is True only if every file matches."""
    expected: Dict[str, str] = {}
    if os.path.isfile(sums_path):
        with open(sums_path, encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) == 2:
                    expected[parts[1].lstrip("*")] = parts[0].lower()
    files = {}
    for p in paths:
        name = os.path.basename(p)
        actual = sha256_file(p) if os.path.isfile(p) else None
        files[name] = {"sha256": actual, "ok": actual is not None and actual == expected.get(name)}
    return {"verified": bool(files) and all(f["ok"] for f in files.values()), "files": files}


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models = os.path.join(root, "models")
    res = verify([os.path.join(models, "world_model.pt"), os.path.join(models, "baseline_lr.json")],
                 os.path.join(models, "SHA256SUMS"))
    for name, f in res["files"].items():
        print(f"{name}: {'OK' if f['ok'] else 'MISMATCH'}  {f['sha256']}")
    sys.exit(0 if res["verified"] else 1)
