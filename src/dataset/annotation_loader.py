"""Annotation loading and processing utilities for Agriculture-Vision dataset.

Handles:
- Field boundary masks
- Validity masks
- Anomaly category masks for the 6 official challenge categories:
  ['cloud_shadow', 'double_plant', 'planter_skip', 'standing_water', 'waterway', 'weed_cluster']
- Preserves overlapping multi-label anomalies without merging classes.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

from src.dataset.image_loader import DEFAULT_EXPECTED_SHAPE, validate_image_dimensions


OFFICIAL_ANOMALY_CATEGORIES: List[str] = [
    "cloud_shadow",
    "double_plant",
    "planter_skip",
    "standing_water",
    "waterway",
    "weed_cluster",
]


def load_binary_mask(
    mask_path: Union[str, Path],
    expected_hw: Optional[Tuple[int, int]] = DEFAULT_EXPECTED_SHAPE,
    to_boolean: bool = False,
) -> np.ndarray:
    """Load a single-channel binary mask from disk.

    Standard Agriculture-Vision masks use 255 for foreground and 0 for background.

    Args:
        mask_path: Path to the .png mask file.
        expected_hw: Expected (height, width).
        to_boolean: If True, values are converted to 0 and 1 uint8. If False,
            preserves original uint8 values (0 or 255).

    Returns:
        np.ndarray of shape (H, W) with uint8 dtype.

    Raises:
        FileNotFoundError: If the mask file does not exist.
        ValueError: If file cannot be decoded or dimensions mismatch.
    """
    path = Path(mask_path)
    if not path.is_file():
        raise FileNotFoundError(f"Mask file not found: {path}")

    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        try:
            with Image.open(path) as pil_img:
                mask = np.array(pil_img.convert("L"))
        except Exception as exc:
            raise ValueError(f"Corrupted or unreadable mask file: {path} ({exc})") from exc

    if expected_hw is not None and not validate_image_dimensions(mask, expected_hw):
        raise ValueError(
            f"Mask dimensions {mask.shape} do not match expected {expected_hw} in {path}"
        )

    # Validate binary values
    unique_vals = np.unique(mask)
    if not np.all(np.isin(unique_vals, [0, 255])):
        # Some masks might have threshold artifacts; binarize at 127
        mask = np.where(mask > 127, np.uint8(255), np.uint8(0))

    if to_boolean:
        return (mask > 127).astype(np.uint8)

    return mask


def load_boundary_mask(
    boundary_path: Union[str, Path],
    expected_hw: Optional[Tuple[int, int]] = DEFAULT_EXPECTED_SHAPE,
    to_boolean: bool = False,
) -> np.ndarray:
    """Load the field boundary mask for an image.

    Args:
        boundary_path: Path to the boundary mask .png file.
        expected_hw: Expected (height, width).
        to_boolean: Whether to return binary values as {0, 1} instead of {0, 255}.

    Returns:
        np.ndarray of shape (H, W).
    """
    return load_binary_mask(boundary_path, expected_hw=expected_hw, to_boolean=to_boolean)


def load_validity_mask(
    validity_path: Union[str, Path],
    expected_hw: Optional[Tuple[int, int]] = DEFAULT_EXPECTED_SHAPE,
    to_boolean: bool = False,
) -> np.ndarray:
    """Load the validity mask for an image (valid non-empty aerial coverage).

    Args:
        validity_path: Path to the mask/validity .png file.
        expected_hw: Expected (height, width).
        to_boolean: Whether to return binary values as {0, 1} instead of {0, 255}.

    Returns:
        np.ndarray of shape (H, W).
    """
    return load_binary_mask(validity_path, expected_hw=expected_hw, to_boolean=to_boolean)


def load_category_mask(
    category_mask_path: Union[str, Path],
    expected_hw: Optional[Tuple[int, int]] = DEFAULT_EXPECTED_SHAPE,
    to_boolean: bool = False,
) -> np.ndarray:
    """Load a specific anomaly category mask (e.g., weed_cluster, standing_water).

    Args:
        category_mask_path: Path to the category .png file.
        expected_hw: Expected (height, width).
        to_boolean: Whether to return {0, 1} instead of {0, 255}.

    Returns:
        np.ndarray of shape (H, W).
    """
    return load_binary_mask(category_mask_path, expected_hw=expected_hw, to_boolean=to_boolean)


def load_all_category_masks(
    labels_dir: Union[str, Path],
    sample_id: str,
    categories: Optional[List[str]] = None,
    expected_hw: Tuple[int, int] = DEFAULT_EXPECTED_SHAPE,
    to_boolean: bool = False,
) -> Dict[str, np.ndarray]:
    """Load all individual anomaly category masks for a given sample.

    Preserves overlapping categories by returning a dictionary mapping each
    category name to its individual binary mask array. If an anomaly mask file
    does not exist for this sample, an all-zero mask is returned for that category.

    Args:
        labels_dir: Directory containing category subdirectories (e.g. labels/).
        sample_id: Base identifier of the sample (e.g., 'FIELD_X-Y-Z-W').
        categories: List of category names to load. Defaults to OFFICIAL_ANOMALY_CATEGORIES.
        expected_hw: Expected (height, width).
        to_boolean: Whether values are {0, 1} or {0, 255}.

    Returns:
        Dict[str, np.ndarray]: Mapping category -> (H, W) uint8 binary mask.
    """
    base_dir = Path(labels_dir)
    cats = categories or OFFICIAL_ANOMALY_CATEGORIES
    zero_val = 0
    result: Dict[str, np.ndarray] = {}

    for cat in cats:
        cat_file = base_dir / cat / f"{sample_id}.png"
        if cat_file.is_file():
            result[cat] = load_category_mask(
                cat_file, expected_hw=expected_hw, to_boolean=to_boolean
            )
        else:
            # Category not present in this tile; all-zero background
            result[cat] = np.zeros(expected_hw, dtype=np.uint8)

    return result


def load_multihot_mask(
    labels_dir: Union[str, Path],
    sample_id: str,
    categories: Optional[List[str]] = None,
    expected_hw: Tuple[int, int] = DEFAULT_EXPECTED_SHAPE,
) -> np.ndarray:
    """Load all category masks as a multi-hot stacked array of shape (H, W, num_classes).

    Values are 1 for presence and 0 for absence.
    Overlapping categories are preserved across separate channel slices.

    Args:
        labels_dir: Directory containing category subdirectories.
        sample_id: Base identifier of the sample.
        categories: Category names in channel order.
        expected_hw: Expected (height, width).

    Returns:
        np.ndarray of shape (H, W, num_classes) with uint8 {0, 1}.
    """
    cats = categories or OFFICIAL_ANOMALY_CATEGORIES
    cat_masks = load_all_category_masks(
        labels_dir=labels_dir,
        sample_id=sample_id,
        categories=cats,
        expected_hw=expected_hw,
        to_boolean=True,
    )
    # Stack along the last axis
    stacked = np.stack([cat_masks[cat] for cat in cats], axis=-1)
    return stacked
