"""StressSense Dataset Package.

Modules for loading, validating, inspecting, and documenting the Agriculture-Vision dataset.
"""

from src.dataset.image_loader import (
    load_rgb_image,
    load_nir_image,
    load_stacked_rgb_nir,
    validate_image_dimensions,
)
from src.dataset.annotation_loader import (
    load_boundary_mask,
    load_validity_mask,
    load_category_mask,
    load_all_category_masks,
    load_multihot_mask,
    OFFICIAL_ANOMALY_CATEGORIES,
)
from src.dataset.dataset_validator import (
    validate_sample,
    validate_dataset_split,
    ValidationReport,
)
from src.dataset.dataset_inspector import (
    inspect_dataset,
    DatasetInspectionReport,
)

__all__ = [
    "load_rgb_image",
    "load_nir_image",
    "load_stacked_rgb_nir",
    "validate_image_dimensions",
    "load_boundary_mask",
    "load_validity_mask",
    "load_category_mask",
    "load_all_category_masks",
    "load_multihot_mask",
    "OFFICIAL_ANOMALY_CATEGORIES",
    "validate_sample",
    "validate_dataset_split",
    "ValidationReport",
    "inspect_dataset",
    "DatasetInspectionReport",
]
