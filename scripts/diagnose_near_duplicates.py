"""Diagnostic: verify near-duplicate algorithm handles CASEs A-E."""
from __future__ import annotations
import sys
from pathlib import Path
import cv2
import numpy as np
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from src.data.freshness_dataset_builder import (
    CandidateImage, compute_image_phash, deduplicate_exact,
    find_near_duplicates, hamming_distance, sha256_file,
)


def make_cand(path, phash, sha=None):
    return CandidateImage(
        path=path, source_dataset="Diag", source_label="x",
        canonical_class="Apple_fresh", fruit="Apple",
        freshness_state="fresh", license="CC0",
        source_url="http://example.com",
        sha256=sha or ("sha_" + str(path)),
        perceptual_hash=phash, width=100, height=100, file_size=1000,
    )


def make_hash(bits_set):
    chars = ["0"] * 64
    for b in bits_set:
        chars[b] = "1"
    return "".join(chars)



def run_case_a(path_a, path_b):
    """CASE A: identical image -> exact duplicate detected."""
    print("\n=== CASE A: identical image -> exact duplicate ===")
    sha_a = sha256_file(path_a)
    sha_b = sha256_file(path_b)
    dup = sha_a == sha_b
    cands = [make_cand(str(path_a), compute_image_phash(path_a)),
             make_cand(str(path_b), compute_image_phash(path_b))]
    cands[1].sha256 = sha_b
    cands[0].sha256 = sha_a
    _, report = deduplicate_exact(cands)
    ok = dup and report.exact_duplicates == 1 and report.unique_images == 1
    print("  SHA256 equal: " + str(dup))
    print("  dedup exact=" + str(report.exact_duplicates) + " unique=" + str(report.unique_images))
    print("  RESULT: " + ("PASS" if ok else "FAIL"))
    return ok


def run_case_b(img):
    """CASE B: same image with resize/recompression -> near duplicate."""
    print("\n=== CASE B: resize/recompress -> near duplicate ===")
    tmp = ROOT_DIR / "scratch" / "diag_case_b"
    tmp.mkdir(parents=True, exist_ok=True)
    p_orig = tmp / "orig.jpg"
    cv2.imwrite(str(p_orig), img)
    p_mod = tmp / "mod.jpg"
    resized = cv2.resize(img, (img.shape[1] - 4, img.shape[0] - 4))
    cv2.imwrite(str(p_mod), resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
    ph_orig = compute_image_phash(p_orig)
    ph_mod = compute_image_phash(p_mod)
    dist = hamming_distance(ph_orig, ph_mod)
    print("  Hamming distance: " + str(dist))
    ok = dist <= 6 and ph_orig != "" and ph_mod != ""
    print("  RESULT: " + ("PASS" if ok else "FAIL"))
    return ok


def run_case_c(img_a, img_b):
    """CASE C: two visually unrelated images -> NOT near duplicate."""
    print("\n=== CASE C: unrelated images -> NOT near duplicate ===")
    tmp = ROOT_DIR / "scratch" / "diag_case_c"
    tmp.mkdir(parents=True, exist_ok=True)
    p_a = tmp / "apple.jpg"
    p_b = tmp / "guava.jpg"
    cv2.imwrite(str(p_a), img_a)
    cv2.imwrite(str(p_b), img_b)
    ph_a = compute_image_phash(p_a)
    ph_b = compute_image_phash(p_b)
    dist = hamming_distance(ph_a, ph_b)
    print("  Hamming distance: " + str(dist))
    cands = [make_cand(str(p_a), ph_a, sha256_file(p_a)),
             make_cand(str(p_b), ph_b, sha256_file(p_b))]
    report = find_near_duplicates(cands)
    ok = dist > 6 and report.near_duplicate_pairs == 0
    print("  near_duplicate_pairs: " + str(report.near_duplicate_pairs))
    print("  RESULT: " + ("PASS" if ok else "FAIL"))
    return ok


def run_case_d():
    """CASE D: same bucket prefix but full distance > 6 -> NOT grouped."""
    print("\n=== CASE D: cross-class false-positive prevention ===")
    H1 = make_hash({8})
    H2 = make_hash({10, 11, 12, 13, 14, 15, 16, 17})
    dist = hamming_distance(H1, H2)
    print("  Hamming distance: " + str(dist) + " (same prefix: " + str(H1[:8] == H2[:8]) + ")")
    cands = [make_cand("apple_d/1.jpg", H1), make_cand("banana_d/1.jpg", H2)]
    cands[0].canonical_class = "Apple_fresh"
    cands[1].canonical_class = "banana_fresh"
    report = find_near_duplicates(cands)
    ok = dist > 6 and report.near_duplicate_pairs == 0
    print("  near_duplicate_pairs: " + str(report.near_duplicate_pairs))
    print("  RESULT: " + ("PASS" if ok else "FAIL"))
    return ok


def run_case_e():
    """CASE E: transitive chain A~B, B~C, A!=C -> separate groups."""
    print("\n=== CASE E: transitive chain A~B, B~C, A!=C ===")
    A = make_hash({10, 11, 12, 13})
    B = make_hash({10, 11, 12, 14})
    C = make_hash({10, 11, 12, 14, 15, 16, 17, 18, 19})
    d_ab = hamming_distance(A, B)
    d_bc = hamming_distance(B, C)
    d_ac = hamming_distance(A, C)
    print("  Ham(A,B)=" + str(d_ab) + ", Ham(B,C)=" + str(d_bc) + ", Ham(A,C)=" + str(d_ac))
    cands = [make_cand("trans/A.jpg", A), make_cand("trans/B.jpg", B), make_cand("trans/C.jpg", C)]
    report = find_near_duplicates(cands)
    print("  groups=" + str(report.near_duplicate_groups) + " pairs=" + str(report.near_duplicate_pairs))
    for g in report.groups:
        members = [m["path"] for m in g["members"]]
        print("  group: " + str(members))
    a_b_grouped = False
    a_c_grouped = False
    for g in report.groups:
        paths = [m["path"] for m in g["members"]]
        if "trans/A.jpg" in paths and "trans/B.jpg" in paths:
            a_b_grouped = True
        if "trans/A.jpg" in paths and "trans/C.jpg" in paths:
            a_c_grouped = True
    ok = d_ab <= 6 and d_bc <= 6 and d_ac > 6 and a_b_grouped and not a_c_grouped
    print("  A-B grouped=" + str(a_b_grouped) + " A-C grouped=" + str(a_c_grouped))
    print("  RESULT: " + ("PASS" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    results = []
    tmp = ROOT_DIR / "scratch" / "diag_case_a"
    tmp.mkdir(parents=True, exist_ok=True)
    img = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
    p_a = tmp / "a.jpg"
    p_b = tmp / "b.jpg"
    cv2.imwrite(str(p_a), img)
    cv2.imwrite(str(p_b), img)
    results.append(("CASE A", run_case_a(p_a, p_b)))
    base = np.zeros((128, 128, 3), dtype=np.uint8)
    for i in range(128):
        for j in range(128):
            base[i, j] = [int(i * 1.5) % 256, int(j * 1.5) % 256, int((i + j) * 0.8) % 256]
    results.append(("CASE B", run_case_b(base)))
    apple_img = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    guava_img = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    results.append(("CASE C", run_case_c(apple_img, guava_img)))
    results.append(("CASE D", run_case_d()))
    results.append(("CASE E", run_case_e()))
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print("  [" + status + "] " + name)
    print("=" * 60)
    print("Overall: " + ("ALL PASS" if all_pass else "FAILURES DETECTED"))
    sys.exit(0 if all_pass else 1)
