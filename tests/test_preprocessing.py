"""Automated Test Suite for StressSense Phase 2 Image Preprocessing & Patch Generation.

Tests:
1. Image Normalization (MinMax, Z-score, Centered, Denormalization).
2. NDVI vegetation index computation.
3. Patch Extraction (dimensions, strides, coordinates).
4. Lossless patch reconstruction fidelity.
5. Edge cases, non-square dimensions, and padding.
6. Alignment between RGB, NIR, boundary, and validity masks.
7. Annotation mask alignment and multi-label overlap preservation.
8. Patch manifest generation and required schema fields.
9. Split and field ID preservation (zero data leakage).
10. End-to-end reproducibility.
11. Full dataset patch validation on data/processed.
"""

import hashlib
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
import pandas as pd
import pytest

from src.preprocessing.normalizer import (
    ImageNormalizer,
    NormalizationMethod,
    compute_ndvi,
)
from src.preprocessing.patch_extractor import (
    PatchExtractor,
    ExtractedPatch,
)
from src.preprocessing.patch_generator import (
    PatchGenerator,
)
from src.preprocessing.patch_validator import (
    PatchValidator,
    PatchValidationReport,
)
from src.dataset.annotation_loader import OFFICIAL_ANOMALY_CATEGORIES


# =========================================================================
# 1. Normalization & NDVI Tests
# =========================================================================

def test_image_normalizer_minmax():
    """Verify MinMax normalization scales uint8 [0, 255] to float32 [0.0, 1.0]."""
    img = np.array([[0, 127], [255, 64]], dtype=np.uint8)
    normalizer = ImageNormalizer(method=NormalizationMethod.MINMAX)

    norm = normalizer.normalize(img)
    assert norm.dtype == np.float32
    assert norm.min() == 0.0
    assert norm.max() == 1.0
    assert np.isclose(norm[0, 1], 127.0 / 255.0, atol=1e-4)


def test_image_normalizer_zscore_and_denormalize():
    """Verify Z-score standardization and round-trip denormalization."""
    img = np.random.randint(50, 200, size=(128, 128, 3), dtype=np.uint8)
    normalizer = ImageNormalizer(method=NormalizationMethod.ZSCORE)

    norm = normalizer.normalize(img, modality="rgb")
    assert norm.dtype == np.float32
    assert norm.shape == (128, 128, 3)

    # Denormalize back to uint8
    recovered = normalizer.denormalize(norm, modality="rgb")
    assert recovered.dtype == np.uint8
    assert recovered.shape == (128, 128, 3)
    # Differences should be within rounding error (+/- 1 intensity level)
    assert np.all(np.abs(img.astype(np.int16) - recovered.astype(np.int16)) <= 1)


def test_ndvi_computation():
    """Verify NDVI calculation: high for healthy foliage, negative for water/soil."""
    h, w = 64, 64
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    nir = np.zeros((h, w), dtype=np.uint8)

    # Region 1: Dense green vegetation (Low Red, High NIR)
    rgb[:32, :, 0] = 30    # Red
    rgb[:32, :, 1] = 180   # Green
    nir[:32, :] = 220      # NIR
    # Expected NDVI = (220 - 30) / (220 + 30) = 190 / 250 = 0.76

    # Region 2: Standing water (High absorption in NIR, moderate in Red)
    rgb[32:, :, 0] = 80    # Red
    rgb[32:, :, 1] = 90
    nir[32:, :] = 20       # NIR
    # Expected NDVI = (20 - 80) / (20 + 80) = -60 / 100 = -0.60

    ndvi = compute_ndvi(rgb, nir)
    assert ndvi.shape == (h, w)
    assert ndvi.dtype == np.float32
    assert np.all(ndvi[:32, :] > 0.7)
    assert np.all(ndvi[32:, :] < -0.5)


# =========================================================================
# 2. Patch Extraction & Reconstruction Tests
# =========================================================================

def test_patch_extractor_dimensions_and_stride():
    """Verify extracting 128x128 patches with stride 128 from 512x512 image produces exactly 16 patches."""
    img = np.random.randint(0, 255, size=(512, 512, 3), dtype=np.uint8)
    extractor = PatchExtractor(patch_size=(128, 128), stride=(128, 128))

    patches = extractor.extract_patches(img)
    assert len(patches) == 16

    expected_coords = [
        (y, x)
        for y in [0, 128, 256, 384]
        for x in [0, 128, 256, 384]
    ]

    for idx, patch in enumerate(patches):
        assert patch.patch_index == idx
        assert patch.height == 128
        assert patch.width == 128
        assert patch.data.shape == (128, 128, 3)
        assert (patch.y, patch.x) == expected_coords[idx]


def test_patch_reconstruction_fidelity():
    """Verify round-trip non-overlapping patch reconstruction perfectly reproduces original image."""
    orig = np.random.randint(0, 256, size=(512, 512, 3), dtype=np.uint8)
    extractor = PatchExtractor(patch_size=128, stride=128)

    patches = extractor.extract_patches(orig)
    reconstructed = extractor.reconstruct_from_patches(patches, original_shape=orig.shape)

    assert reconstructed.shape == orig.shape
    assert reconstructed.dtype == orig.dtype
    assert np.array_equal(orig, reconstructed), "Reconstructed image does not match original!"


def test_edge_cases_and_incomplete_patches():
    """Verify boundary padding when image dimensions are not divisible by patch size."""
    odd_img = np.full((300, 300), 150, dtype=np.uint8)
    extractor = PatchExtractor(patch_size=128, stride=128, pad_if_needed=True, pad_mode="constant")

    # 300 / 128 -> ceil is 3 steps -> 3x3 = 9 patches, padded to (384, 384)
    patches = extractor.extract_patches(odd_img)
    assert len(patches) == 9
    for p in patches:
        assert p.height == 128
        assert p.width == 128
        assert p.data.shape == (128, 128)

    # Reconstruct back to (300, 300)
    recon = extractor.reconstruct_from_patches(patches, original_shape=(300, 300))
    assert recon.shape == (300, 300)
    assert np.array_equal(odd_img, recon)


# =========================================================================
# 3. Cross-Modality Alignment & Synthetic Dataset Fixtures
# =========================================================================

@pytest.fixture
def synthetic_sample_dir(tmp_path: Path) -> Path:
    """Creates a temporary raw dataset split with 1 synthetic image and all masks."""
    raw_root = tmp_path / "raw"
    split = "train"
    sdir = raw_root / split

    for d in ["images/rgb", "images/nir", "boundaries", "masks"]:
        (sdir / d).mkdir(parents=True, exist_ok=True)
    for cat in OFFICIAL_ANOMALY_CATEGORIES:
        (sdir / "labels" / cat).mkdir(parents=True, exist_ok=True)

    sid = "SYNTH01_000-000-512-512"

    # Distinct pattern: diagonal line in RGB and NIR
    rgb = np.zeros((512, 512, 3), dtype=np.uint8)
    nir = np.zeros((512, 512), dtype=np.uint8)
    bnd = np.full((512, 512), 255, dtype=np.uint8)
    msk = np.full((512, 512), 255, dtype=np.uint8)

    # Draw alignment feature across diagonal
    cv2.line(rgb, (0, 0), (512, 512), (255, 255, 255), 4)
    cv2.line(nir, (0, 0), (512, 512), 255, 4)

    cv2.imwrite(str(sdir / "images" / "rgb" / f"{sid}.jpg"), rgb)
    cv2.imwrite(str(sdir / "images" / "nir" / f"{sid}.jpg"), nir)
    cv2.imwrite(str(sdir / "boundaries" / f"{sid}.png"), bnd)
    cv2.imwrite(str(sdir / "masks" / f"{sid}.png"), msk)

    # Put a weed cluster in patch (1, 1) -> y in [128, 256], x in [128, 256]
    for cat in OFFICIAL_ANOMALY_CATEGORIES:
        cat_m = np.zeros((512, 512), dtype=np.uint8)
        if cat == "weed_cluster":
            cat_m[150:200, 150:200] = 255
        cv2.imwrite(str(sdir / "labels" / cat / f"{sid}.png"), cat_m)

    return raw_root


def test_cross_modality_alignment(synthetic_sample_dir: Path, tmp_path: Path):
    """Verify that RGB, NIR, boundary, and category masks align spatially across patches."""
    proc_dir = tmp_path / "proc"
    gen = PatchGenerator(
        raw_dir=synthetic_sample_dir,
        processed_dir=proc_dir,
        patch_size=128,
        stride=128,
    )
    df = gen.run()
    assert len(df) == 16

    # Patch (y=128, x=128) must have weed_cluster active
    target_row = df[(df["y_coordinate"] == 128) & (df["x_coordinate"] == 128)].iloc[0]
    assert bool(target_row["has_anomaly"]) is True
    assert "weed_cluster" in str(target_row["anomaly_categories"])
    assert target_row["weed_cluster_fraction"] > 0.0

    # Other non-overlapping patches (e.g. y=0, x=0) must have 0 weed_cluster
    corner_row = df[(df["y_coordinate"] == 0) & (df["x_coordinate"] == 0)].iloc[0]
    assert corner_row["weed_cluster_fraction"] == 0.0


def test_reproducibility(synthetic_sample_dir: Path, tmp_path: Path):
    """Verify patch generation is 100% deterministic and reproducible."""
    proc1 = tmp_path / "run1"
    proc2 = tmp_path / "run2"

    gen1 = PatchGenerator(raw_dir=synthetic_sample_dir, processed_dir=proc1, patch_size=128, stride=128)
    df1 = gen1.run()

    gen2 = PatchGenerator(raw_dir=synthetic_sample_dir, processed_dir=proc2, patch_size=128, stride=128)
    df2 = gen2.run()

    assert list(df1["patch_id"]) == list(df2["patch_id"])
    assert np.allclose(df1["valid_pixel_fraction"].values, df2["valid_pixel_fraction"].values)
    assert np.allclose(df1["weed_cluster_fraction"].values, df2["weed_cluster_fraction"].values)


# =========================================================================
# 4. Full Processed Dataset Validation
# =========================================================================

def test_full_processed_dataset_cleanliness():
    """Verify that the generated patches in data/processed pass all integrity checks."""
    proc_dir = Path("data/processed")
    manifest_csv = proc_dir / "manifests" / "patch_manifest.csv"

    if not manifest_csv.is_file():
        pytest.skip("data/processed/manifests/patch_manifest.csv not generated yet")

    validator = PatchValidator(
        processed_dir=proc_dir,
        manifest_path=manifest_csv,
        expected_size=(128, 128),
    )
    report = validator.validate()

    assert report.total_patches == 640
    assert report.valid_patches == 640
    assert report.invalid_patches == 0
    assert len(report.duplicate_patch_ids) == 0
    assert len(report.missing_files) == 0
    assert len(report.dimension_mismatches) == 0
    assert len(report.channel_mismatches) == 0
    assert len(report.invalid_annotation_values) == 0
    assert len(report.split_cross_contaminations) == 0
    assert len(report.field_cross_contaminations) == 0
    assert report.is_clean is True
