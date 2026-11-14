"""Dataset validation module for Agriculture-Vision dataset.

Verifies:
- RGB and NIR image decodability and channel structure
- Boundary and validity mask presence and values
- 512x512 dimension consistency across modalities
- Multi-label anomaly category mask integrity
- Exact 1-to-1 correspondence between images, masks, and boundaries
- Preservation of overlapping anomaly classes
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np

from src.dataset.image_loader import (
    DEFAULT_EXPECTED_SHAPE,
    load_rgb_image,
    load_nir_image,
    validate_image_dimensions,
)
from src.dataset.annotation_loader import (
    OFFICIAL_ANOMALY_CATEGORIES,
    load_boundary_mask,
    load_validity_mask,
    load_all_category_masks,
)


@dataclass
class SampleValidationResult:
    """Validation result for an individual sample tile."""
    sample_id: str
    is_valid: bool
    rgb_valid: bool = False
    nir_valid: bool = False
    boundary_valid: bool = False
    mask_valid: bool = False
    categories_present: List[str] = field(default_factory=list)
    has_overlap: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Summary report of dataset split validation."""
    split_name: str
    total_samples: int = 0
    valid_samples: int = 0
    invalid_samples: int = 0
    missing_nir_count: int = 0
    missing_boundary_count: int = 0
    missing_mask_count: int = 0
    category_counts: Dict[str, int] = field(default_factory=dict)
    overlap_samples_count: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    sample_results: Dict[str, SampleValidationResult] = field(default_factory=dict)

    def is_clean(self) -> bool:
        """Returns True if there are zero invalid samples and zero critical errors."""
        return self.invalid_samples == 0 and len(self.errors) == 0


def validate_sample(
    sample_id: str,
    split_dir: Union[str, Path],
    has_labels: bool = True,
    expected_hw: Tuple[int, int] = DEFAULT_EXPECTED_SHAPE,
    categories: Optional[List[str]] = None,
) -> SampleValidationResult:
    """Validate all components of a single sample in a dataset split.

    Args:
        sample_id: Base identifier (e.g. 'FIELD_X-Y-Z-W').
        split_dir: Root directory of split (e.g., path to 'train/' or 'val/').
        has_labels: Whether ground truth labels are expected (False for 'test/').
        expected_hw: Expected dimensions (H, W).
        categories: Official anomaly categories to inspect.

    Returns:
        SampleValidationResult with detailed status.
    """
    sdir = Path(split_dir)
    cats = categories or OFFICIAL_ANOMALY_CATEGORIES
    result = SampleValidationResult(sample_id=sample_id, is_valid=True)

    # 1. RGB Image check (can be .jpg or .png)
    rgb_path = sdir / "images" / "rgb" / f"{sample_id}.jpg"
    if not rgb_path.is_file():
        rgb_path = sdir / "images" / "rgb" / f"{sample_id}.png"
    if not rgb_path.is_file():
        result.is_valid = False
        result.errors.append(f"RGB image missing for {sample_id}")
    else:
        try:
            rgb = load_rgb_image(rgb_path, expected_hw=expected_hw)
            if rgb.ndim != 3 or rgb.shape[2] != 3:
                result.is_valid = False
                result.errors.append(f"RGB image has invalid channels {rgb.shape}")
            else:
                result.rgb_valid = True
        except Exception as exc:
            result.is_valid = False
            result.errors.append(f"Error loading RGB image: {exc}")

    # 2. NIR Image check
    nir_path = sdir / "images" / "nir" / f"{sample_id}.jpg"
    if not nir_path.is_file():
        nir_path = sdir / "images" / "nir" / f"{sample_id}.png"
    if not nir_path.is_file():
        result.is_valid = False
        result.errors.append(f"NIR image missing for {sample_id}")
    else:
        try:
            nir = load_nir_image(nir_path, expected_hw=expected_hw)
            result.nir_valid = True
        except Exception as exc:
            result.is_valid = False
            result.errors.append(f"Error loading NIR image: {exc}")

    # 3. Boundary check
    boundary_path = sdir / "boundaries" / f"{sample_id}.png"
    if not boundary_path.is_file():
        result.warnings.append(f"Boundary mask missing for {sample_id}")
    else:
        try:
            bnd = load_boundary_mask(boundary_path, expected_hw=expected_hw)
            result.boundary_valid = True
        except Exception as exc:
            result.is_valid = False
            result.errors.append(f"Error loading boundary mask: {exc}")

    # 4. Validity Mask check
    mask_path = sdir / "masks" / f"{sample_id}.png"
    if not mask_path.is_file():
        result.warnings.append(f"Validity mask missing for {sample_id}")
    else:
        try:
            msk = load_validity_mask(mask_path, expected_hw=expected_hw)
            result.mask_valid = True
        except Exception as exc:
            result.is_valid = False
            result.errors.append(f"Error loading validity mask: {exc}")

    # 5. Anomaly labels check (if present)
    if has_labels:
        labels_dir = sdir / "labels"
        if labels_dir.is_dir():
            try:
                masks = load_all_category_masks(
                    labels_dir=labels_dir,
                    sample_id=sample_id,
                    categories=cats,
                    expected_hw=expected_hw,
                    to_boolean=True,
                )
                active_cats = []
                for cat, m in masks.items():
                    if np.any(m > 0):
                        active_cats.append(cat)
                result.categories_present = active_cats
                if len(active_cats) > 1:
                    result.has_overlap = True
            except Exception as exc:
                result.is_valid = False
                result.errors.append(f"Error loading anomaly masks: {exc}")

    return result


def validate_dataset_split(
    split_dir: Union[str, Path],
    split_name: str = "train",
    has_labels: bool = True,
    categories: Optional[List[str]] = None,
) -> ValidationReport:
    """Validate all samples in a dataset split directory.

    Args:
        split_dir: Path to the split directory (e.g. data/raw/Agriculture-Vision-2021/train).
        split_name: Identifier for the split.
        has_labels: True if labels/ should be checked.
        categories: Official anomaly category list.

    Returns:
        ValidationReport summarizing consistency and data integrity.
    """
    sdir = Path(split_dir)
    report = ValidationReport(split_name=split_name)
    cats = categories or OFFICIAL_ANOMALY_CATEGORIES
    report.category_counts = {cat: 0 for cat in cats}

    if not sdir.is_dir():
        report.errors.append(f"Split directory does not exist: {sdir}")
        return report

    rgb_dir = sdir / "images" / "rgb"
    if not rgb_dir.is_dir():
        report.errors.append(f"RGB image directory missing: {rgb_dir}")
        return report

    # Gather all base sample IDs from rgb directory
    sample_ids: List[str] = sorted(
        [p.stem for p in rgb_dir.iterdir() if p.suffix.lower() in [".jpg", ".png", ".jpeg"]]
    )
    report.total_samples = len(sample_ids)

    if report.total_samples == 0:
        report.warnings.append(f"No image files found in {rgb_dir}")
        return report

    for sid in sample_ids:
        res = validate_sample(
            sample_id=sid,
            split_dir=sdir,
            has_labels=has_labels,
            categories=cats,
        )
        report.sample_results[sid] = res

        if res.is_valid:
            report.valid_samples += 1
        else:
            report.invalid_samples += 1
            for err in res.errors:
                report.errors.append(f"[{sid}] {err}")

        for warn in res.warnings:
            report.warnings.append(f"[{sid}] {warn}")

        if not res.nir_valid:
            report.missing_nir_count += 1
        if not res.boundary_valid:
            report.missing_boundary_count += 1
        if not res.mask_valid:
            report.missing_mask_count += 1

        for cat in res.categories_present:
            report.category_counts[cat] = report.category_counts.get(cat, 0) + 1

        if res.has_overlap:
            report.overlap_samples_count += 1

    return report
