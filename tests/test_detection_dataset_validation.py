"""Unit tests for scripts/validate_detection_dataset.py.

Builds small in-memory YOLO-layout datasets and checks that
``validate_dataset`` accepts valid data, canonicalises the ``val``/``valid``
alias, rejects malformed labels, and writes a JSON report.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.validate_detection_dataset import validate_dataset


def _make_image(parent: Path, name: str) -> None:
    parent.mkdir(parents=True, exist_ok=True)
    (parent / name).write_bytes(b"")  # empty placeholder file is fine; only listing matters


def _make_label(parent: Path, name: str, rows) -> None:
    parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(" ".join(str(c) for c in row) for row in rows) + "\n"
    (parent / name).write_text(text, encoding="utf-8")


def _build_dataset(root: Path, split_dirs=("train", "valid", "test"), per_split=1,
                   label_rows=None, labels=True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(
        "path: .\ntrain: train/images\nval: valid/images\ntest: test/images\n"
        "nc: 2\nnames: {0: apple, 1: banana}\n",
        encoding="utf-8",
    )
    for sp in split_dirs:
        img_dir = root / sp / "images"
        lbl_dir = root / sp / "labels"
        for i in range(per_split):
            _make_image(img_dir, f"img_{i}.jpg")
            if labels:
                rows = label_rows if label_rows is not None else [
                    ("0", "0.5", "0.5", "0.2", "0.2"),
                    ("1", "0.2", "0.2", "0.3", "0.3"),
                ]
                _make_label(lbl_dir, f"img_{i}.txt", rows)
    return root


def test_clean_dataset_passes(tmp_path: Path):
    root = _build_dataset(tmp_path)
    report = validate_dataset(root)

    assert report["status"] == "pass"
    assert report["dataset_root"] == str(root)
    assert report["summary"]["total_images"] == 3
    assert report["summary"]["total_labels"] == 3
    assert set(report["summary"]["splits"]) == {"train", "valid", "test"}


def test_val_alias_accepted(tmp_path: Path):
    root = _build_dataset(tmp_path, split_dirs=("train", "val", "test"))
    report = validate_dataset(root)

    assert report["status"] == "pass"
    assert "valid" in report["summary"]["splits"]
    assert report["summary"]["splits"]["valid"]["folder"] == "val"
    assert report["summary"]["splits"]["valid"]["canonical_split"] == "valid"


def test_bad_label_rows_fail(tmp_path: Path):
    # A 2-field row is invalid for YOLO (expects 5 fields).
    root = _build_dataset(tmp_path, label_rows=[("0", "0.5")])
    report = validate_dataset(root)

    assert report["status"] == "fail"
    assert report["errors"]
    assert any(r["bad_label_rows"] for r in report["summary"]["splits"].values())


def test_writes_json_report(tmp_path: Path):
    root = _build_dataset(tmp_path)
    out = tmp_path / "sub" / "report.json"
    report = validate_dataset(root, out)

    assert out.is_file()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["status"] == report["status"]
    assert data["summary"]["data_yaml"]["nc"] == 2  # sanity: data.yaml parsed
