"""StressSense Preprocessing Package.

Modules for image normalization, patch extraction, dataset patch generation,
validation, and visual reporting.
"""

from src.preprocessing.normalizer import (
    ImageNormalizer,
    NormalizationMethod,
    compute_ndvi,
    DEFAULT_CHANNEL_MEANS,
    DEFAULT_CHANNEL_STDS,
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
from src.preprocessing.visualizer import (
    plot_preprocessing_pipeline_showcase,
    plot_patch_grid_decomposition,
    plot_multilabel_patch_overlays,
    plot_patch_statistics_summary,
)

__all__ = [
    "ImageNormalizer",
    "NormalizationMethod",
    "compute_ndvi",
    "DEFAULT_CHANNEL_MEANS",
    "DEFAULT_CHANNEL_STDS",
    "PatchExtractor",
    "ExtractedPatch",
    "PatchGenerator",
    "PatchValidator",
    "PatchValidationReport",
    "plot_preprocessing_pipeline_showcase",
    "plot_patch_grid_decomposition",
    "plot_multilabel_patch_overlays",
    "plot_patch_statistics_summary",
]
