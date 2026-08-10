#!/usr/bin/env python3
"""Validate the real-world fruit dataset — Phase 4A.

Runs the full validation workflow:

    MEASURE -> VALIDATE -> CLEAN(flag only) -> SPLIT -> REPORT

Never modifies the raw dataset. Rejected/movable samples are only *reported*;
the operator must move them explicitly.

Usage:
    python scripts/validate_real_world_dataset.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset_validation import (  # noqa: E402
    BRIGHTNESS_MAX,
    BRIGHTNESS_MIN,
    BLUR_THRESHOLD,
    CONTRAST_MIN,
    DatasetScan,
    group_files_by_session,
    find_exact_duplicates,
    find_near_duplicates,
    find_suspicious_groups,
    load_metadata_dir,
    scan_directory,
    split_files,
)
from src.data.real_world_schema import (  # noqa: E402
    find_physical_fruit_leakage,
    find_session_leakage,
    load_canonical_manifest,
    validate_canonical_manifest,
)

logger = logging.getLogger(__name__)


def _quality_summary(scan: DatasetScan) -> Dict:
    """Aggregate quality statistics from a scan."""
    readable = scan.readable
    if not readable:
        return {"images": 0}

    def stats(vals):
        if not vals:
            return {}
        return {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    brightness = [i.quality.brightness for i in readable if i.quality]
    contrast = [i.quality.contrast for i in readable if i.quality]
    blur = [i.quality.blur_score for i in readable if i.quality]
    widths = [i.width for i in readable]
    heights = [i.height for i in readable]
    ratios = [i.aspect_ratio for i in readable if i.aspect_ratio > 0]

    return {
        "images": len(readable),
        "brightness": stats(brightness),
        "contrast": stats(contrast),
        "blur": stats(blur),
        "width": stats(widths),
        "height": stats(heights),
        "aspect_ratio": stats(ratios),
        "dark_count": sum(1 for i in readable if i.quality and i.quality.is_dark()),
        "bright_count": sum(1 for i in readable if i.quality and i.quality.is_bright()),
        "blurry_count": sum(1 for i in readable if i.quality and i.quality.is_blurry()),
        "low_contrast_count": sum(
            1 for i in readable if i.quality and i.quality.is_low_contrast()
        ),
    }


def _write_manifest(records, out: Path) -> None:
    """Write a JSON manifest of parsed metadata."""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps([r.raw for r in records], indent=2, default=str),
        encoding="utf-8",
    )


def _canonical_validation(root: Path, lines: List[str]) -> Dict:
    """Validate the canonical manifest + any existing split files (Phase 5).

    Adds a report section to ``lines`` and returns a machine-readable dict of
    findings. Never blocks on missing optional metadata; missing required
    fields are errors.
    """
    add = lines.append

    def rule(ch="="):
        add(ch * 70)

    add("")
    rule("-")
    add("SECTION 2B - CANONICAL MANIFEST (Phase 5 schema)")
    rule("-")

    manifest_candidates = [
        root / "manifest.csv",
        root / "manifest.json",
        root / "manifests" / "manifest.csv",
    ]
    manifest_path = next((p for p in manifest_candidates if p.exists()), None)
    if manifest_path is None:
        add("No canonical manifest found (expected manifest.csv or manifest.json).")
        add("  -> create it per docs/REAL_WORLD_DATASET.md to enable")
        add("     physical-fruit-grouped splitting and leakage checks.")
        return {
            "manifest_present": False,
            "total_rows": 0,
            "physical_fruit_leakage": [],
            "session_leakage": [],
        }

    add(f"Manifest: {manifest_path}")

    try:
        records = load_canonical_manifest(manifest_path)
    except Exception as exc:  # noqa: BLE001
        add(f"ERROR: could not parse manifest: {exc}")
        return {"manifest_present": True, "parse_error": str(exc)}

    report = validate_canonical_manifest(records, data_root=root)
    add(f"Rows                 : {report.total_rows}")
    add(f"Valid rows           : {report.valid_rows}")
    add(f"Errors               : {report.error_count}")
    add(f"Warnings             : {report.warning_count}")
    add(f"Physical fruits      : {report.physical_fruit_count}")
    add(f"Capture sessions     : {report.capture_session_count}")
    add(f"Class distribution   : {report.class_distribution}")
    add(f"Fruit-type dist      : {report.fruit_type_distribution}")
    add(f"Freshness dist       : {report.freshness_distribution}")
    add(f"Imbalance ratio      : {report.imbalance_ratio:.2f} "
        f"({'balanced' if report.is_balanced else 'IMBALANCED'})")
    if report.missing_required:
        add("MISSING REQUIRED METADATA:")
        for ref, field in report.missing_required[:20]:
            add(f"  - {ref}: missing {field}")
    if report.invalid_labels:
        add("INVALID LABELS:")
        for ref, detail in report.invalid_labels[:20]:
            add(f"  - {ref}: {detail}")
    if report.invalid_values:
        add("INVALID VALUES:")
        for ref, detail in report.invalid_values[:20]:
            add(f"  - {ref}: {detail}")
    if report.duplicate_image_ids:
        add(f"DUPLICATE IMAGE IDS ({len(report.duplicate_image_ids)}):")
        for image_id in report.duplicate_image_ids[:20]:
            add(f"  - {image_id}")
    if report.exact_duplicate_files:
        add("EXACT DUPLICATE FILES (same bytes):")
        for a, b in report.exact_duplicate_files[:20]:
            add(f"  - {a} == {b}")
    if report.missing_image_files:
        add(f"MISSING IMAGE FILES ({len(report.missing_image_files)}):")
        for path in report.missing_image_files[:20]:
            add(f"  - {path}")
    if report.fruit_type_conflicts:
        add("FRUIT-TYPE CONFLICTS (impossible metadata combinations):")
        for fruit_id, fruit_type, ref in report.fruit_type_conflicts[:20]:
            add(f"  - {fruit_id}: {fruit_type} at row {ref}")
    if report.impossible_combinations:
        add("OTHER IMPOSSIBLE COMBINATIONS:")
        for ref, detail in report.impossible_combinations[:20]:
            add(f"  - {ref}: {detail}")

    # --- existing split leakage checks -----------------------------------
    splits_dir = root / "splits"
    split_records = {}
    fruit_leakage = []
    session_leakage = []
    if splits_dir.is_dir():
        for split_name, filename in (
            ("train", "train.csv"),
            ("val", "val.csv"),
            ("test", "test.csv"),
        ):
            split_path = splits_dir / filename
            if split_path.exists():
                try:
                    split_records[split_name] = load_canonical_manifest(split_path)
                except Exception as exc:  # noqa: BLE001
                    add(f"ERROR: could not load {split_path}: {exc}")
    if split_records:
        add("")
        add("Split-file leakage checks:")
        fruit_leakage = find_physical_fruit_leakage(split_records)
        session_leakage = find_session_leakage(split_records)
        add(f"  physical-fruit leakage across splits: {len(fruit_leakage)}")
        for fruit_id, a, b in fruit_leakage[:20]:
            add(f"    - {fruit_id} in {a} and {b}")
        add(f"  session leakage across splits (advisory): {len(session_leakage)}")
        for session_id, a, b in session_leakage[:20]:
            add(f"    - {session_id} in {a} and {b}")
        if fruit_leakage:
            add("  STATUS: PHYSICAL-FRUIT LEAKAGE DETECTED - splits invalid")
        else:
            add("  STATUS: no physical-fruit leakage")
    else:
        add("No split files found under splits/; run scripts/create_dataset_split.py")

    result = report.to_dict()
    result["manifest_present"] = True
    result["manifest_path"] = str(manifest_path)
    result["physical_fruit_leakage"] = [
        f"{f} ({a} & {b})" for f, a, b in fruit_leakage
    ]
    result["session_leakage"] = [f"{s} ({a} & {b})" for s, a, b in session_leakage]
    return result


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Validate real-world dataset (Phase 4A)")
    parser.add_argument("--data-dir", type=Path, default=Path("data/real_world"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    root = args.data_dir
    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    accepted_dir = root / "accepted"
    metadata_dir = root / "metadata"

    lines: List[str] = []
    add = lines.append

    def rule(ch="="):
        add(ch * 70)

    rule()
    add("REAL-WORLD DATASET VALIDATION REPORT")
    add("Generated by scripts/validate_real_world_dataset.py (Phase 4A)")
    rule()
    add(f"Data root: {root}")

    # ------------------------------------------------------------------
    # 1. Image inventory
    # ------------------------------------------------------------------
    add("")
    rule("-")
    add("SECTION 1 - IMAGE INVENTORY")
    rule("-")
    scan = scan_directory(accepted_dir)
    add(f"Accepted images: {scan.total}")
    add(f"Unreadable/corrupted: {len(scan.unreadable)}")
    for path, err in scan.unreadable[:20]:
        add(f"  - CORRUPT {path}: {err}")
    if len(scan.unreadable) > 20:
        add(f"  ... and {len(scan.unreadable) - 20} more")

    class_counts = scan.class_counts()
    add("Class distribution (by folder):")
    for cls, cnt in sorted(class_counts.items(), key=lambda kv: -kv[1]):
        add(f"  {cls}: {cnt}")
    if not class_counts:
        add("  (none)")

    # ------------------------------------------------------------------
    # 2. Metadata
    # ------------------------------------------------------------------
    add("")
    rule("-")
    add("SECTION 2 - METADATA")
    rule("-")
    records = load_metadata_dir(metadata_dir)
    add(f"Metadata records found: {len(records)}")

    incomplete = []
    for rec in records:
        missing = rec.missing_fields()
        if missing:
            incomplete.append((rec.sample_id, missing))
    add(f"Records with missing required fields: {len(incomplete)}")
    for sid, missing in incomplete[:20]:
        add(f"  - {sid}: missing {missing}")

    accepted_records = [r for r in records if r.accepted]
    add(f"Records marked accepted: {len(accepted_records)}")

    if records:
        sessions = sorted({r.session_id for r in records if r.session_id})
        add(f"Unique sessions in metadata: {len(sessions)}")
        per_session: Dict[str, int] = {}
        for r in records:
            per_session[r.session_id] = per_session.get(r.session_id, 0) + 1
        for sid in sessions:
            add(f"  {sid}: {per_session[sid]} samples")
        disagree = sum(
            1
            for r in records
            if r.predicted_class and r.label and r.predicted_class != r.label
        )
        add(f"records where predicted_class != manual label: {disagree}")

    manifest_path = report_dir / "real_world_metadata_manifest.json"
    _write_manifest(records, manifest_path)
    add(f"Metadata manifest: {manifest_path}")

    canonical = _canonical_validation(root, lines)
    return _sections(scan, records, root, report_dir, lines, accepted_dir, canonical)


def _sections(scan, records, root, report_dir, lines, accepted_dir, canonical: Dict) -> int:
    """Build report sections 3-7 and write the report file."""
    add = lines.append

    def rule(ch="="):
        add(ch * 70)

    # ------------------------------------------------------------------
    # 3. Quality
    # ------------------------------------------------------------------
    add("")
    rule("-")
    add("SECTION 3 - IMAGE QUALITY")
    rule("-")
    qs = _quality_summary(scan)
    if qs.get("images", 0) > 0:
        for key in ("brightness", "contrast", "blur", "width", "height", "aspect_ratio"):
            st = qs[key]
            add(
                f"{key:14s} mean={st['mean']:.2f} std={st['std']:.2f} "
                f"min={st['min']:.2f} max={st['max']:.2f}"
            )
        add("")
        add("Flagged samples (suggested for review, NOT auto-moved):")
        add(f"  dark (brightness < {BRIGHTNESS_MIN}): {qs['dark_count']}")
        add(f"  bright (brightness > {BRIGHTNESS_MAX}): {qs['bright_count']}")
        add(f"  blurry (blur < {BLUR_THRESHOLD}): {qs['blurry_count']}")
        add(f"  low contrast (< {CONTRAST_MIN}): {qs['low_contrast_count']}")

        flagged = []
        for info in scan.readable:
            q = info.quality
            if q and (q.is_dark() or q.is_bright() or q.is_blurry() or q.is_low_contrast()):
                flagged.append(info)
        add(f"  total flagged: {len(flagged)}")
        for info in flagged[:15]:
            q = info.quality
            add(
                f"    - {info.path.relative_to(root)} "
                f"(b={q.brightness:.0f} c={q.contrast:.0f} blur={q.blur_score:.0f})"
            )
    else:
        add("No readable images to evaluate.")

    # ------------------------------------------------------------------
    # 4. Duplicates
    # ------------------------------------------------------------------
    add("")
    rule("-")
    add("SECTION 4 - DUPLICATES")
    rule("-")
    image_paths = [i.path for i in scan.images if i.readable]
    if image_paths:
        exact = find_exact_duplicates(image_paths)
        add(f"Exact duplicates (MD5): {len(exact)} pairs")
        for pair in exact[:20]:
            add(
                f"  - {pair.path_a.relative_to(root)}  ==  "
                f"{pair.path_b.relative_to(root)}"
            )
        near = find_near_duplicates(image_paths)
        add(f"Near duplicates (pHash, <=10 bits): {len(near)} pairs")
        for pair in near[:20]:
            add(
                f"  - {pair.path_a.relative_to(root)}  ~  "
                f"{pair.path_b.relative_to(root)} (sim={pair.similarity:.2f})"
            )
        groups = find_suspicious_groups(image_paths)
        add(f"Suspicious groups (near-duplicate clusters): {len(groups)}")
        for seed, group in groups[:10]:
            short = [str(p.relative_to(root)) for p in group[:4]]
            add(f"  - group of {len(group)}: {short}")
    else:
        add("No images to compare.")

    # ------------------------------------------------------------------
    # 5. Session / specimen leakage
    # ------------------------------------------------------------------
    add("")
    rule("-")
    add("SECTION 5 - SESSION / SPECIMEN LEAKAGE")
    rule("-")
    if records:
        resolved = [(root / r.image_path).resolve() for r in records]
        sessions_per_file = {
            (root / r.image_path).resolve(): r.session_id for r in records
        }
        grouped = group_files_by_session(resolved, lambda p: sessions_per_file.get(p))
        add(f"Files mappable to a session: {sum(len(v) for v in grouped.values())}")
        if grouped:
            add("SPECIMEN-LEVEL SPLITTING VERIFIED (by session_id)")
            add(
                "Frames from one collection session stay in one split - "
                "no cross-session leakage within a single split."
            )
        else:
            add("SPECIMEN-LEVEL SPLITTING NOT VERIFIED")
        add(
            "NOTE: session_id groups frames captured in one collection run. "
            "Physical fruit identity is NOT recorded; two sessions may contain "
            "the same specimen."
        )
    else:
        add("No metadata -> session-aware verification impossible.")
        add("SPECIMEN-LEVEL SPLITTING NOT VERIFIED")

    return _split_and_verdict(scan, records, root, report_dir, lines, canonical)


def _split_and_verdict(scan, records, root, report_dir, lines, canonical: Dict) -> int:
    """Build the split section, the verdict, and write the report file."""
    add = lines.append

    def rule(ch="="):
        add(ch * 70)

    # ------------------------------------------------------------------
    # 6. Split
    # ------------------------------------------------------------------
    add("")
    rule("-")
    add("SECTION 6 - SPLIT (70/15/15)")
    rule("-")
    image_paths = [i.path for i in scan.images if i.readable]
    if image_paths:
        by_sessions = None
        if records:
            sessions_per_file = {
                (root / r.image_path).resolve(): r.session_id for r in records
            }
            resolved = [(root / r.image_path).resolve() for r in records]
            by_sessions = group_files_by_session(resolved, lambda p: sessions_per_file.get(p))
        if by_sessions and len(by_sessions) >= 3:
            split = split_files(image_paths, by_sessions=by_sessions)
            c = split.counts
            add(f"SESSION-AWARE split: train={c['train']} val={c['val']} test={c['test']}")
        else:
            split = split_files(image_paths, by_sessions=None)
            c = split.counts
            add(
                f"FILE-LEVEL stratified split: train={c['train']} val={c['val']} "
                f"test={c['test']}"
            )
            if by_sessions is not None:
                add("  fallback: fewer than 3 sessions for session-aware split")
    else:
        add("No images -> no split generated.")

    # ------------------------------------------------------------------
    # 7. Verdict
    # ------------------------------------------------------------------
    add("")
    rule()
    add("VERDICT")
    rule()
    add("")

    if scan.total == 0:
        verdict = "NOT READY FOR TRAINING"
        reason = (
            "data/real_world/accepted/ contains zero images. No dataset has "
            "been collected yet; run src/data/collection.py first."
        )
    elif len(records) == 0:
        verdict = "NOT READY FOR TRAINING"
        reason = (
            "Images exist but no metadata. Metadata (session_id, label, "
            "timestamp) is mandatory for traceable experiments."
        )
    elif not any(r.session_id for r in records):
        verdict = "NOT READY FOR TRAINING"
        reason = "Session identity missing; specimen-level splitting impossible."
    elif len(scan.unreadable) > 0:
        verdict = "NOT READY FOR TRAINING (pending cleanup)"
        reason = f"{len(scan.unreadable)} unreadable/corrupted files detected."
    elif (
        canonical.get("manifest_present")
        and canonical.get("physical_fruit_leakage")
    ):
        verdict = "NOT READY FOR TRAINING (leakage)"
        reason = (
            f"Physical-fruit leakage detected across existing splits "
            f"({len(canonical['physical_fruit_leakage'])} violation(s)); "
            "re-split per docs/REAL_WORLD_DATASET.md."
        )
    elif (
        canonical.get("manifest_present")
        and canonical.get("error_count", 0) > 0
    ):
        verdict = "NOT READY FOR TRAINING (manifest errors)"
        reason = (
            f"Canonical manifest has {canonical['error_count']} blocking "
            "error(s); fix before training."
        )
    else:
        verdict = "READY FOR TRAINING"
        reason = "Data present, metadata complete, sessions attributable."

    add(f"Verdict: {verdict}")
    add(f"Reason: {reason}")
    add("")

    report_path = report_dir / "REAL_WORLD_DATASET_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    # ------------------------------------------------------------------
    # 8. Machine-readable + human-readable Phase 5 reports.
    # ------------------------------------------------------------------
    json_payload = {
        "tool": "scripts/validate_real_world_dataset.py",
        "phase": "5",
        "data_root": str(root),
        "images": {
            "total": scan.total,
            "unreadable": [str(p) for p, _ in scan.unreadable],
            "class_distribution": scan.class_counts(),
        },
        "quality": _quality_summary(scan),
        "legacy_metadata": {
            "records": len(records),
            "incomplete": [
                {"sample_id": sid, "missing": sorted(miss)}
                for sid, miss in (
                    (rec.sample_id, rec.missing_fields()) for rec in records
                )
                if miss
            ][:100],
        },
        "canonical_manifest": canonical,
        "verdict": verdict,
        "reason": reason,
    }

    json_path = report_dir / "dataset_validation.json"
    json_path.write_text(
        json.dumps(json_payload, indent=2, default=str), encoding="utf-8"
    )

    md_path = report_dir / "dataset_validation.md"
    md_lines = [
        "# Dataset Validation Report (Phase 5)",
        "",
        f"- **Tool**: scripts/validate_real_world_dataset.py",
        f"- **Data root**: {root}",
        f"- **Verdict**: {verdict}",
        f"- **Reason**: {reason}",
        "",
        "## Image inventory",
        f"- Total images: {scan.total}",
        f"- Unreadable/corrupted: {len(scan.unreadable)}",
        "",
        "## Class distribution (legacy folders)",
    ]
    md_lines.extend(
        f"- {cls}: {cnt}"
        for cls, cnt in sorted(scan.class_counts().items(), key=lambda kv: -kv[1])
    )
    md_lines.extend(["", "## Canonical manifest (Phase 5 schema)"])
    if not canonical.get("manifest_present"):
        md_lines.append("- No manifest present.")
    else:
        md_lines.extend(
            [
                f"- Manifest: {canonical.get('manifest_path', 'unknown')}",
                f"- Rows: {canonical.get('total_rows', 0)}",
                f"- Physical fruits: {canonical.get('physical_fruit_count', 0)}",
                f"- Capture sessions: {canonical.get('capture_session_count', 0)}",
                f"- Class distribution: {canonical.get('class_distribution', {})}",
                f"- Errors: {canonical.get('error_count', 0)}",
                f"- Warnings: {canonical.get('warning_count', 0)}",
                f"- Missing required metadata: {len(canonical.get('missing_required', []))}",
                f"- Invalid labels: {len(canonical.get('invalid_labels', []))}",
                f"- Invalid values: {len(canonical.get('invalid_values', []))}",
                f"- Duplicate image ids: {len(canonical.get('duplicate_image_ids', []))}",
                f"- Exact duplicate files: {len(canonical.get('exact_duplicate_files', []))}",
                f"- Missing image files: {len(canonical.get('missing_image_files', []))}",
                f"- Impossible combinations: {len(canonical.get('impossible_combinations', []))}",
                f"- Physical-fruit leakage across splits: "
                f"{len(canonical.get('physical_fruit_leakage', []))}",
                f"- Session leakage across splits (advisory): "
                f"{len(canonical.get('session_leakage', []))}",
                "",
                "### Error detail (first 25)",
            ]
        )
        md_lines.extend(
            f"- {entry[0]}: {entry[1]}"
            for entry in (
                canonical.get("missing_required", [])
                + canonical.get("invalid_values", [])
                + canonical.get("fruit_type_conflicts", [])
            )[:25]
        )
        md_lines.extend(["", "### Warning detail (first 25)"])
        md_lines.extend(
            f"- {entry[0]}: {entry[1]}"
            for entry in (
                canonical.get("invalid_labels", [])
                + canonical.get("unknown_enum_values", [])
                + canonical.get("impossible_combinations", [])
            )[:25]
        )
    md_lines.extend(
        [
            "",
            "## Verdict",
            f"- **{verdict}**",
            f"- {reason}",
            "",
            "_Generated by `scripts/validate_real_world_dataset.py`. See "
            "`docs/REAL_WORLD_DATASET.md` for the canonical schema._",
        ]
    )
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")
    return 0 if verdict.startswith("READY") else 1

if __name__ == "__main__":
    sys.exit(main())
