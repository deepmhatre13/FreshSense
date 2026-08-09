"""FreshSense Repository Health Check.

Run from the repository root:

    python scripts/health_check.py

Checks:
  1. Package structure  — src/ and all subpackages have __init__.py
  2. Core imports       — critical modules can be imported
  3. Canonical pipeline — detection_pipeline.py is importable
  4. Model checkpoint   — models/checkpoints/best_model.pth exists
  5. Dataset            — data/raw or data/processed contains images
  6. PyTest             — pytest is installed and can collect tests
  7. Dependencies       — key package versions are reported
"""

from __future__ import annotations

import importlib
import importlib.util
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _check(label: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    msg = f"  {label:<40} {status}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return ok


def _section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print("─" * 60)


# ---------------------------------------------------------------------------
# 1. Package structure
# ---------------------------------------------------------------------------
def check_structure() -> bool:
    _section("Package Structure")
    expected_packages = [
        "src",
        "src/inference",
        "src/models",
        "src/training",
        "src/preprocessing",
        "src/data",
        "src/detection",
        "src/utils",
    ]
    ok = True
    for pkg in expected_packages:
        pkg_dir = ROOT / pkg
        init = pkg_dir / "__init__.py"
        exists = pkg_dir.is_dir() and init.is_file()
        ok = _check(f"{pkg}/__init__.py", exists) and ok
    return ok


# ---------------------------------------------------------------------------
# 2. Core imports
# ---------------------------------------------------------------------------
def check_imports() -> bool:
    _section("Core Imports")
    # Ensure repository root is on sys.path
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    critical = [
        "src",
        "src.inference",
        "src.models",
        "src.training",
        "src.preprocessing",
        "src.data",
        "src.detection",
        "src.utils",
    ]
    ok = True
    for mod in critical:
        try:
            importlib.import_module(mod)
            ok = _check(f"import {mod}", True) and ok
        except Exception as exc:
            ok = _check(f"import {mod}", False, str(exc)[:80]) and ok
    return ok


# ---------------------------------------------------------------------------
# 3. Canonical pipeline
# ---------------------------------------------------------------------------
def check_pipeline() -> bool:
    _section("Canonical Pipelines")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    pipelines = [
        ("src.inference.detection_pipeline", "DetectionPipeline",
         "Phase 4 canonical pipeline (detect→track→classify)"),
        ("src.inference.pipeline", "Pipeline",
         "Phase 3 pipeline (single-fruit, no detector)"),
    ]
    ok = True
    for mod_name, cls_name, label in pipelines:
        try:
            mod = importlib.import_module(mod_name)
            has_cls = hasattr(mod, cls_name)
            ok = _check(f"{cls_name} ({label[:30]})", has_cls) and ok
        except Exception as exc:
            ok = _check(f"{cls_name}", False, str(exc)[:80]) and ok

    # Verify session analyzer semantics
    try:
        from src.inference.session_analyzer import SessionAnalyzer
        analyzer = SessionAnalyzer()
        # Single fruit tracked across 3 frames must NOT count as 3 fruits
        for i in range(3):
            analyzer.add_frame({
                "timestamp": float(i),
                "predictions": [{"fruit_name": "apple", "freshness_class": "fresh"}],
                "confidences": [0.9],
                "tracking_ids": [42],
            })
        summary = analyzer.analyze("health-check")
        correct = summary.unique_tracks == 1 and summary.total_detections == 3
        ok = _check(
            "SessionAnalyzer unique-track semantics",
            correct,
            f"unique={summary.unique_tracks}, total_det={summary.total_detections}",
        ) and ok
    except Exception as exc:
        ok = _check("SessionAnalyzer unique-track semantics", False, str(exc)[:80]) and ok

    return ok


# ---------------------------------------------------------------------------
# 4. Model checkpoint
# ---------------------------------------------------------------------------
def check_model() -> bool:
    _section("Model Checkpoint")
    ckpt = ROOT / "models" / "checkpoints" / "best_model.pth"
    exists = ckpt.is_file()
    size_mb = ckpt.stat().st_size / 1_048_576 if exists else 0
    _check(
        "models/checkpoints/best_model.pth",
        exists,
        f"{size_mb:.1f} MB" if exists else "NOT AVAILABLE",
    )
    return True  # missing checkpoint is not a structural failure


# ---------------------------------------------------------------------------
# 5. Dataset
# ---------------------------------------------------------------------------
def check_dataset() -> bool:
    _section("Dataset")
    data_dir = ROOT / "data"
    subdirs = ["raw", "processed", "real_world"]
    any_images = False
    for sub in subdirs:
        path = data_dir / sub
        if path.is_dir():
            count = sum(1 for _ in path.rglob("*.jpg")) + sum(
                1 for _ in path.rglob("*.png")
            )
            _check(f"data/{sub}", True, f"{count} images")
            if count > 0:
                any_images = True
        else:
            _check(f"data/{sub}", False, "directory missing")
    if not any_images:
        print("  ⚠  No dataset images found — training not possible")
    return True  # dataset absence is not a structural failure


# ---------------------------------------------------------------------------
# 6. PyTest
# ---------------------------------------------------------------------------
def check_pytest() -> bool:
    _section("PyTest")
    spec = importlib.util.find_spec("pytest")
    pytest_ok = spec is not None
    _check("pytest installed", pytest_ok)

    if pytest_ok:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q",
             "--tb=no", "-p", "no:warnings"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        output = result.stdout + result.stderr
        # Extract collected count
        import re
        m = re.search(r"(\d+) tests? collected", output)
        errors = re.search(r"(\d+) errors?", output)
        collected = m.group(1) if m else "?"
        err_count = errors.group(1) if errors else "0"
        all_ok = result.returncode == 0
        _check(
            "pytest --collect-only",
            all_ok,
            f"{collected} collected, {err_count} errors",
        )
    return True


# ---------------------------------------------------------------------------
# 7. Dependencies
# ---------------------------------------------------------------------------
def check_dependencies() -> bool:
    _section("Key Dependencies")
    deps = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("cv2", "opencv-python"),
        ("PIL", "Pillow"),
        ("albumentations", "albumentations"),
        ("numpy", "numpy"),
        ("sklearn", "scikit-learn"),
    ]
    all_ok = True
    for import_name, pkg_name in deps:
        try:
            mod = importlib.import_module(import_name)
            ver = getattr(mod, "__version__", "?")
            _check(f"{pkg_name}", True, f"v{ver}")
        except ImportError:
            _check(f"{pkg_name}", False, "not installed")
            all_ok = False

    # Check CUDA
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        device = torch.cuda.get_device_name(0) if cuda_ok else "CPU only"
        _check("CUDA available", cuda_ok, device)
    except Exception:
        pass

    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print()
    print("=" * 60)
    print("  FreshSense Repository Health Check")
    print("=" * 60)

    results = {
        "structure": check_structure(),
        "imports": check_imports(),
        "pipeline": check_pipeline(),
        "model": check_model(),
        "dataset": check_dataset(),
        "pytest": check_pytest(),
        "dependencies": check_dependencies(),
    }

    # Summary
    print()
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    label_map = {
        "structure": "Package structure",
        "imports": "Core imports",
        "pipeline": "Canonical pipeline",
        "model": "Model checkpoint",
        "dataset": "Dataset",
        "pytest": "PyTest",
        "dependencies": "Dependencies",
    }
    all_pass = True
    for key, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {label_map[key]:<30} {status}")
        if not ok:
            all_pass = False

    print()
    if all_pass:
        print("  Overall: READY FOR TESTING")
    else:
        print("  Overall: ISSUES FOUND — review FAIL items above")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
