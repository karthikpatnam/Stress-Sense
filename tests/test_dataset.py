"""Automated unit test suite for Agriculture-Vision dataset loading, validation, and inspection.

Covers:
1. Image loading (RGB 3-channel, uint8 and normalized float32).
2. NIR image loading (1-channel, shape verification, normalization).
3. 4-channel stacked RGB+NIR loading.
4. Annotation mask loading (boundaries, validity masks).
5. Category anomaly mask loading (all 6 official categories, multi-hot array).
6. Preservation of overlapping anomaly classes without merging.
7. Image-mask dimension correspondence.
8. Invalid and corrupted file handling.
9. Missing file detection.
10. Dataset split identification via inspector.
11. Annotation binary value validation ({0, 255} and {0, 1}).
12. Data leakage prevention (field ID separation across splits).
"""

from pathlib import Path
from typing import Dict
import cv2
import numpy as np
import pandas as pd
import pytest

from src.dataset.image_loader import (
    DEFAULT_EXPECTED_SHAPE,
    load_rgb_image,
    load_nir_image,
    load_stacked_rgb_nir,
    validate_image_dimensions,
)
from src.dataset.annotation_loader import (
    OFFICIAL_ANOMALY_CATEGORIES,
    load_binary_mask,
    load_boundary_mask,
    load_validity_mask,
    load_category_mask,
    load_all_category_masks,
    load_multihot_mask,
)
from src.dataset.dataset_validator import (
    validate_sample,
    validate_dataset_split,
)
from src.dataset.dataset_inspector import (
    inspect_dataset,
    inspect_split,
)


@pytest.fixture
def mock_dataset_dir(tmp_path: Path) -> Path:
    """Create a self-contained, minimal mock Agriculture-Vision dataset hierarchy for testing."""
    root = tmp_path / "mock_dataset"
    for split in ["train", "val", "test"]:
        (root / split / "images" / "rgb").mkdir(parents=True, exist_ok=True)
        (root / split / "images" / "nir").mkdir(parents=True, exist_ok=True)
        (root / split / "boundaries").mkdir(parents=True, exist_ok=True)
        (root / split / "masks").mkdir(parents=True, exist_ok=True)
        if split != "test":
            for cat in OFFICIAL_ANOMALY_CATEGORIES:
                (root / split / "labels" / cat).mkdir(parents=True, exist_ok=True)

    # Create one valid sample in train with multi-label overlapping anomalies
    sid = "FIELD01_100-200-612-712"
    h, w = 512, 512

    # RGB
    rgb = np.full((h, w, 3), 100, dtype=np.uint8)
    rgb[:, :, 1] = 180  # green canopy
    cv2.imwrite(str(root / "train" / "images" / "rgb" / f"{sid}.jpg"), rgb)

    # NIR
    nir = np.full((h, w), 210, dtype=np.uint8)
    cv2.imwrite(str(root / "train" / "images" / "nir" / f"{sid}.jpg"), nir)

    # Boundary
    bnd = np.full((h, w), 255, dtype=np.uint8)
    cv2.imwrite(str(root / "train" / "boundaries" / f"{sid}.png"), bnd)

    # Mask
    msk = np.full((h, w), 255, dtype=np.uint8)
    cv2.imwrite(str(root / "train" / "masks" / f"{sid}.png"), msk)

    # Category masks: cloud_shadow AND weed_cluster overlap in the center
    for cat in OFFICIAL_ANOMALY_CATEGORIES:
        cmask = np.zeros((h, w), dtype=np.uint8)
        if cat == "cloud_shadow":
            cmask[100:300, 100:300] = 255
        elif cat == "weed_cluster":
            cmask[200:400, 200:400] = 255  # Overlaps [200:300, 200:300]
        cv2.imwrite(str(root / "train" / "labels" / cat / f"{sid}.png"), cmask)

    return root


# =========================================================================
# 1. Image Loading Tests
# =========================================================================

def test_rgb_image_loading(mock_dataset_dir: Path):
    """Test loading RGB images: shape, channels, and normalization."""
    rgb_path = mock_dataset_dir / "train" / "images" / "rgb" / "FIELD01_100-200-612-712.jpg"
    img = load_rgb_image(rgb_path)

    assert isinstance(img, np.ndarray)
    assert img.shape == (512, 512, 3)
    assert img.dtype == np.uint8

    # Test normalized float32
    img_norm = load_rgb_image(rgb_path, normalize=True)
    assert img_norm.dtype == np.float32
    assert 0.0 <= img_norm.min() <= img_norm.max() <= 1.0


def test_nir_image_loading(mock_dataset_dir: Path):
    """Test loading Near-Infrared single-channel images."""
    nir_path = mock_dataset_dir / "train" / "images" / "nir" / "FIELD01_100-200-612-712.jpg"

    # Default 2D array
    nir = load_nir_image(nir_path, keep_dim=False)
    assert isinstance(nir, np.ndarray)
    assert nir.shape == (512, 512)
    assert nir.dtype == np.uint8

    # keep_dim = True
    nir_3d = load_nir_image(nir_path, keep_dim=True)
    assert nir_3d.shape == (512, 512, 1)

    # Normalized
    nir_norm = load_nir_image(nir_path, normalize=True)
    assert nir_norm.dtype == np.float32
    assert 0.0 <= nir_norm.min() <= nir_norm.max() <= 1.0


def test_stacked_rgb_nir_loading(mock_dataset_dir: Path):
    """Test loading stacked 4-channel [R, G, B, NIR] arrays."""
    rgb_path = mock_dataset_dir / "train" / "images" / "rgb" / "FIELD01_100-200-612-712.jpg"
    nir_path = mock_dataset_dir / "train" / "images" / "nir" / "FIELD01_100-200-612-712.jpg"

    stacked = load_stacked_rgb_nir(rgb_path, nir_path)
    assert isinstance(stacked, np.ndarray)
    assert stacked.shape == (512, 512, 4)
    assert stacked.dtype == np.uint8


# =========================================================================
# 2. Annotation & Mask Loading Tests
# =========================================================================

def test_boundary_and_validity_mask_loading(mock_dataset_dir: Path):
    """Test loading field boundary and validity masks."""
    bnd_path = mock_dataset_dir / "train" / "boundaries" / "FIELD01_100-200-612-712.png"
    msk_path = mock_dataset_dir / "train" / "masks" / "FIELD01_100-200-612-712.png"

    bnd = load_boundary_mask(bnd_path, to_boolean=False)
    assert bnd.shape == (512, 512)
    assert np.all(np.isin(np.unique(bnd), [0, 255]))

    msk_bool = load_validity_mask(msk_path, to_boolean=True)
    assert msk_bool.shape == (512, 512)
    assert np.all(np.isin(np.unique(msk_bool), [0, 1]))


def test_category_masks_loading_and_overlap_preservation(mock_dataset_dir: Path):
    """Verify loading all 6 anomaly categories and that overlapping classes are preserved."""
    labels_dir = mock_dataset_dir / "train" / "labels"
    sid = "FIELD01_100-200-612-712"

    cat_masks = load_all_category_masks(labels_dir, sid, to_boolean=True)
    assert set(cat_masks.keys()) == set(OFFICIAL_ANOMALY_CATEGORIES)
    for cat, m in cat_masks.items():
        assert m.shape == (512, 512)
        assert np.all(np.isin(np.unique(m), [0, 1]))

    # Check multi-label overlap: both cloud_shadow and weed_cluster must be active
    assert np.any(cat_masks["cloud_shadow"] > 0)
    assert np.any(cat_masks["weed_cluster"] > 0)
    overlap_region = (cat_masks["cloud_shadow"] > 0) & (cat_masks["weed_cluster"] > 0)
    assert np.sum(overlap_region) == 100 * 100  # 10,000 overlapping pixels

    # Multi-hot stacked array (H, W, 6)
    multihot = load_multihot_mask(labels_dir, sid)
    assert multihot.shape == (512, 512, 6)
    assert multihot.dtype == np.uint8


# =========================================================================
# 3. Dimension & Integrity Checks
# =========================================================================

def test_dimension_validation_mismatch(tmp_path: Path):
    """Ensure dimension mismatch raises ValueError."""
    bad_img_path = tmp_path / "bad_dim.png"
    bad_array = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.imwrite(str(bad_img_path), bad_array)

    with pytest.raises(ValueError, match="dimensions"):
        load_rgb_image(bad_img_path, expected_hw=(512, 512))

    with pytest.raises(ValueError, match="dimensions"):
        load_nir_image(bad_img_path, expected_hw=(512, 512))


def test_corrupted_and_missing_file_handling(tmp_path: Path):
    """Ensure non-existent and unreadable files trigger appropriate exceptions."""
    non_existent = tmp_path / "does_not_exist.jpg"

    with pytest.raises(FileNotFoundError):
        load_rgb_image(non_existent)

    with pytest.raises(FileNotFoundError):
        load_nir_image(non_existent)

    with pytest.raises(FileNotFoundError):
        load_boundary_mask(non_existent)

    # Corrupted / 0-byte file
    corrupt_path = tmp_path / "corrupt.jpg"
    corrupt_path.write_bytes(b"NOT_A_VALID_JPEG_HEADER")

    with pytest.raises(ValueError):
        load_rgb_image(corrupt_path)


# =========================================================================
# 4. Validator & Inspector Tests
# =========================================================================

def test_sample_and_split_validation(mock_dataset_dir: Path):
    """Test sample and split validator reporting."""
    train_dir = mock_dataset_dir / "train"
    sid = "FIELD01_100-200-612-712"

    sample_res = validate_sample(sid, train_dir, has_labels=True)
    assert sample_res.is_valid is True
    assert sample_res.rgb_valid is True
    assert sample_res.nir_valid is True
    assert sample_res.boundary_valid is True
    assert sample_res.mask_valid is True
    assert sample_res.has_overlap is True
    assert "cloud_shadow" in sample_res.categories_present
    assert "weed_cluster" in sample_res.categories_present

    split_rep = validate_dataset_split(train_dir, split_name="train", has_labels=True)
    assert split_rep.total_samples == 1
    assert split_rep.valid_samples == 1
    assert split_rep.invalid_samples == 0
    assert split_rep.overlap_samples_count == 1
    assert split_rep.is_clean() is True


def test_missing_modality_detection(mock_dataset_dir: Path):
    """Test that validator catches missing NIR or boundary files."""
    train_dir = mock_dataset_dir / "train"
    sid = "FIELD01_100-200-612-712"

    # Remove NIR file
    nir_file = train_dir / "images" / "nir" / f"{sid}.jpg"
    nir_file.unlink()

    res = validate_sample(sid, train_dir, has_labels=True)
    assert res.is_valid is False
    assert res.nir_valid is False
    assert any("NIR image missing" in e for e in res.errors)


def test_dataset_split_identification(mock_dataset_dir: Path):
    """Test inspector discovers splits and modalities accurately."""
    report = inspect_dataset(mock_dataset_dir)
    assert set(report.splits_found) == {"train", "val", "test"}
    assert report.total_samples == 1
    assert report.all_dimensions_uniform is True
    assert report.detected_dimensions == ["512x512"]


# =========================================================================
# 5. Data Leakage Prevention Tests
# =========================================================================

def test_data_leakage_manifest_field_separation():
    """Verify that field IDs in the manifest have 0 intersection between train, val, and test."""
    manifest_path = Path("data/raw/data_manifest.csv")
    if not manifest_path.is_file():
        pytest.skip("data_manifest.csv not present locally")

    df = pd.read_csv(manifest_path)
    train_fields = set(df[df["split"] == "train"]["flightcode"].dropna())
    val_fields = set(df[df["split"] == "val"]["flightcode"].dropna())
    test_fields = set(df[df["split"] == "test"]["flightcode"].dropna())

    # Ensure no leakage
    assert len(train_fields.intersection(val_fields)) == 0, "Data leakage detected between train and val!"
    assert len(train_fields.intersection(test_fields)) == 0, "Data leakage detected between train and test!"
    assert len(val_fields.intersection(test_fields)) == 0, "Data leakage detected between val and test!"


# =========================================================================
# 6. Representative Dataset Integrity Test
# =========================================================================

def test_representative_dataset_validation_clean():
    """Verify that the representative Agriculture-Vision subset in data/raw is 100% clean."""
    dataset_dir = Path("data/raw/Agriculture-Vision-2021")
    if not dataset_dir.is_dir():
        pytest.skip("Agriculture-Vision-2021 dataset subset not generated yet")

    for split in ["train", "val", "test"]:
        sdir = dataset_dir / split
        has_labels = (split != "test")
        report = validate_dataset_split(sdir, split_name=split, has_labels=has_labels)
        assert report.total_samples > 0
        assert report.invalid_samples == 0, f"Found invalid samples in {split}: {report.errors}"
        assert report.is_clean() is True
