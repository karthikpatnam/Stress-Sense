"""Empirical heuristic spectral-contrast scoring engine for StressSense adaptive sampling.

This module computes empirical multi-spectral contrast scores using an
uncalibrated Relative Spectral Difference Index (RSDI) contrast heuristic and
spatial heterogeneity metrics.

SCIENTIFIC INTEGRITY & TERMINOLOGY DISCLOSURES:
1. Nature of Imagery:
   Input imagery consists of uncalibrated 8-bit digital numbers (DN in [0, 255])
   stored in lossy JPEG format without sensor gain/offset tables, downwelling light
   sensor telemetry, or radiometric calibration targets.
2. Heuristic Spectral-Contrast Index (RSDI):
   The computed index ((NIR - Red) / (NIR + Red + eps)) is an empirical heuristic
   spectral-contrast score. It is NOT calibrated physical NDVI, and does NOT establish
   actual canopy vigor loss or physiological health state.
3. Algorithmic Priority Classes:
   Categorical tiers (HIGH_STRESS, MODERATE_STRESS, HEALTHY) are algorithmic priority
   classes designed solely to steer drone sampling budgets, NOT independently validated
   biological or physiological plant pathology classifications.
4. Ground-Truth Decoupling:
   Spectral stress scoring is strictly unsupervised and decoupled from
   ground-truth anomaly annotations.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.adaptive.grid_generator import GridRegion


class StressCategory(str, Enum):
    """Categorical classification of regional crop stress."""

    HEALTHY = "healthy"
    MODERATE_STRESS = "moderate_stress"
    HIGH_STRESS = "high_stress"
    INVALID = "invalid"


@dataclass
class RegionStressScore:
    """Stores multi-spectral stress scoring metrics for a single grid region.

    Attributes:
        region_id: Identifier matching the corresponding GridRegion.
        row: Row index in the 2D grid.
        col: Column index in the 2D grid.
        mean_ndvi: Mean NDVI across valid pixels in the region.
        std_ndvi: Standard deviation of NDVI across valid pixels.
        deficit_score: Empirical spectral contrast deficit score relative to ndvi_target in [0.0, 1.0].
        heterogeneity_score: Spatial spectral heterogeneity score in [0.0, 1.0].
        combined_stress_score: Combined empirical heuristic stress score in [0.0, 1.0].
        stress_category: Categorical priority classification (HEALTHY, MODERATE_STRESS, HIGH_STRESS, INVALID).
        valid_pixel_count: Number of valid agricultural pixels analyzed.
        total_pixel_count: Total pixel area of the grid region.
        valid_fraction: Ratio of valid pixels to total pixels [0.0, 1.0].
        is_valid: True if valid_fraction >= min_valid_fraction and region has valid data.
        gt_has_anomaly: Ground-truth anomaly presence, populated ONLY when authentic labels exist.
        gt_anomaly_fraction: Ground-truth anomaly pixel fraction, if authentic labels exist.
        gt_categories: Ground-truth anomaly category names, if authentic labels exist.
    """

    region_id: str
    row: int
    col: int
    mean_ndvi: float
    std_ndvi: float
    deficit_score: float
    heterogeneity_score: float
    combined_stress_score: float
    stress_category: StressCategory
    valid_pixel_count: int
    total_pixel_count: int
    valid_fraction: float
    is_valid: bool = True
    gt_has_anomaly: Optional[bool] = None
    gt_anomaly_fraction: Optional[float] = None
    gt_categories: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert stress score metrics to a serializable dictionary."""
        return {
            "region_id": self.region_id,
            "row": self.row,
            "col": self.col,
            "mean_ndvi": round(self.mean_ndvi, 5),
            "std_ndvi": round(self.std_ndvi, 5),
            "deficit_score": round(self.deficit_score, 5),
            "heterogeneity_score": round(self.heterogeneity_score, 5),
            "combined_stress_score": round(self.combined_stress_score, 5),
            "stress_category": self.stress_category.value,
            "valid_pixel_count": self.valid_pixel_count,
            "total_pixel_count": self.total_pixel_count,
            "valid_fraction": round(self.valid_fraction, 5),
            "is_valid": self.is_valid,
            "gt_has_anomaly": self.gt_has_anomaly,
            "gt_anomaly_fraction": round(self.gt_anomaly_fraction, 5)
            if self.gt_anomaly_fraction is not None
            else None,
            "gt_categories": ",".join(self.gt_categories) if self.gt_categories else "none",
        }


class StressScorer:
    """Calculates multi-spectral crop stress scores across spatial grid regions.

    SCIENTIFIC VALIDITY, SENSOR ACQUISITION & CALIBRATION DISCLOSURE:
    1. Nature of Input Data:
       The Agriculture-Vision dataset provides 4-channel imagery (RGB + NIR) as 8-bit
       digital numbers (DN in [0, 255]) saved in lossy JPEG format. No radiometric
       calibration coefficients, solar irradiance telemetry (DLS), calibration tarp
       targets, or sensor gain/offset tables are provided in the dataset release.
    2. Numerical Scaling vs. Reflectance Conversion:
       Rescaling raw uint8 [0, 255] values to [0.0, 1.0] via division by 255.0 is
       purely an arithmetic normalization to float32. It does NOT establish physical
       surface reflectance (Top-of-Canopy HCRF) compatibility between channels.
    3. Index Definition:
       The computed index is an uncalibrated empirical pseudo-NDVI or Relative Spectral
       Difference Index (RSDI). While it provides effective relative spatial contrast
       for ranking healthy canopy versus bare soil/anomalies within an orthomosaic,
       it is NOT a validated physical surface reflectance NDVI.
    4. Heuristic Calibration Parameters:
       Parameters ndvi_target (default 0.70) and ndvi_stressed (default 0.20) are
       empirical spatial-contrast calibration parameters tuned for the Agriculture-Vision
       imagery, NOT validated agronomic or biological physiological stress thresholds.
    5. Decoupling from Ground-Truth Anomaly Labels:
       Spectral stress scores are strictly unsupervised optical proxies; they are
       completely decoupled from Agriculture-Vision ground-truth labels (e.g. planter
       skip, cloud shadow, weed cluster).
    """

    def __init__(
        self,
        ndvi_target: float = 0.70,
        ndvi_stressed: float = 0.20,
        tau_low: float = 0.35,
        tau_high: float = 0.70,
        weight_deficit: float = 0.75,
        weight_heterogeneity: float = 0.25,
        eps: float = 1e-7,
        min_valid_fraction: float = 0.05,
    ) -> None:
        """Initialize the StressScorer.

        Args:
            ndvi_target: Experimental heuristic parameter representing target reference
                         empirical RSDI score (default 0.70; calibrated to 0.32 for dataset dynamic range).
            ndvi_stressed: Experimental heuristic parameter representing severe
                           spectral deficit reference empirical RSDI score (default 0.20).
            tau_low: Upper threshold for HEALTHY stress category (default 0.35).
            tau_high: Lower threshold for HIGH_STRESS category (default 0.70).
            weight_deficit: Weight assigned to canopy deficit in combined score (default 0.75).
            weight_heterogeneity: Weight assigned to spectral variance (default 0.25).
            eps: Numerical stability constant to prevent division-by-zero (default 1e-7).
            min_valid_fraction: Minimum valid pixel fraction for a region to be evaluated.
        """
        if ndvi_target <= ndvi_stressed:
            raise ValueError(
                f"ndvi_target ({ndvi_target}) must be strictly greater than ndvi_stressed ({ndvi_stressed})"
            )
        if not (0.0 <= tau_low < tau_high <= 1.0):
            raise ValueError(
                f"Thresholds must satisfy 0.0 <= tau_low < tau_high <= 1.0, got tau_low={tau_low}, tau_high={tau_high}"
            )
        if abs((weight_deficit + weight_heterogeneity) - 1.0) > 1e-5:
            raise ValueError(
                f"Weights must sum to 1.0, got {weight_deficit} + {weight_heterogeneity} = {weight_deficit + weight_heterogeneity}"
            )

        self.ndvi_target = float(ndvi_target)
        self.ndvi_stressed = float(ndvi_stressed)
        self.tau_low = float(tau_low)
        self.tau_high = float(tau_high)
        self.weight_deficit = float(weight_deficit)
        self.weight_heterogeneity = float(weight_heterogeneity)
        self.eps = float(eps)
        self.min_valid_fraction = float(min_valid_fraction)

    def validate_and_align_inputs(
        self,
        rgb: np.ndarray,
        nir: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Validate array shapes, channel ordering, and check pre-normalization compatibility.

        Args:
            rgb: RGB image array of shape (H, W, 3) with Red at channel index 0.
            nir: NIR image array of shape (H, W) or (H, W, 1).

        Returns:
            Tuple of (red_float32, nir_float32) numerically rescaled to float32 in [0.0, 1.0].
            Note: This is an arithmetic rescaling for numerical stability, NOT a physical
            conversion to calibrated surface reflectance.

        Raises:
            ValueError: If shapes mismatch, channel counts are invalid, or input values
                        exhibit negative values indicative of invalid independent z-score normalization.
        """
        if not isinstance(rgb, np.ndarray) or not isinstance(nir, np.ndarray):
            raise TypeError("Both rgb and nir must be NumPy ndarrays")

        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(
                f"Expected 3-channel RGB image with shape (H, W, 3), got {rgb.shape}"
            )

        nir_sq = nir.squeeze() if nir.ndim == 3 else nir
        if nir_sq.ndim != 2:
            raise ValueError(f"Expected 2D Near-Infrared image (H, W), got {nir.shape}")

        if (rgb.shape[0], rgb.shape[1]) != (nir_sq.shape[0], nir_sq.shape[1]):
            raise ValueError(
                f"Spatial dimension mismatch between RGB {rgb.shape[:2]} and NIR {nir_sq.shape[:2]}"
            )

        # Defensive check against independently z-score normalized or zero-centered inputs.
        # Raw digital number (DN) inputs must be non-negative.
        # Negative values indicate pre-normalized or standardized distributions.
        if np.any(rgb < 0) or np.any(nir_sq < 0):
            raise ValueError(
                "Negative values detected in input channels. NDVI calculation requires "
                "compatible raw digital numbers prior to independent channel standardization."
            )

        red_f = rgb[:, :, 0].astype(np.float32)
        nir_f = nir_sq.astype(np.float32)

        # Check if values are in uint8 [0, 255] or float32 [0.0, 1.0]
        # Bring both Red and NIR onto a standard [0.0, 1.0] scale consistently
        red_max = float(np.max(red_f)) if red_f.size > 0 else 0.0
        nir_max = float(np.max(nir_f)) if nir_f.size > 0 else 0.0

        if red_max > 1.0 or nir_max > 1.0:
            red_f = red_f / 255.0
            nir_f = nir_f / 255.0

        return red_f, nir_f

    def compute_ndvi_map(
        self,
        red: np.ndarray,
        nir: np.ndarray,
    ) -> np.ndarray:
        """Calculate the full-resolution pixel-wise NDVI map.

        Formula:
            NDVI = (NIR - Red) / (NIR + Red + eps)

        Args:
            red: Float32 array of Red channel reflectance in [0.0, 1.0].
            nir: Float32 array of Near-Infrared reflectance in [0.0, 1.0].

        Returns:
            np.ndarray of shape (H, W) with float32 values clipped to [-1.0, 1.0].
        """
        numerator = nir - red
        denominator = nir + red + self.eps
        ndvi = numerator / denominator
        return np.clip(ndvi, -1.0, 1.0).astype(np.float32)

    def compute_canopy_deficit(self, mean_ndvi: float) -> float:
        """Calculate empirical heuristic spectral contrast deficit score from mean regional RSDI.

        Formula:
            Deficit = clip((ndvi_target - mean_ndvi) / (ndvi_target - ndvi_stressed), 0.0, 1.0)

        Args:
            mean_ndvi: Mean regional RSDI empirical spectral contrast value.

        Returns:
            Deficit score in [0.0, 1.0]. 0.0 = minimal spectral deficit; 1.0 = high spectral contrast deficit.
            Note: This is an empirical heuristic score, not confirmed biological pathology.
        """
        val = (self.ndvi_target - mean_ndvi) / (self.ndvi_target - self.ndvi_stressed + self.eps)
        return float(np.clip(val, 0.0, 1.0))

    def compute_heterogeneity(self, std_ndvi: float) -> float:
        """Calculate spatial spectral heterogeneity from NDVI standard deviation.

        Formula:
            Heterogeneity = min(1.0, 3.0 * std_ndvi)

        Args:
            std_ndvi: Standard deviation of NDVI across valid regional pixels.

        Returns:
            Heterogeneity score in [0.0, 1.0].
        """
        val = 3.0 * float(std_ndvi)
        return float(min(1.0, max(0.0, val)))

    def compute_combined_score(self, deficit: float, heterogeneity: float) -> float:
        """Compute weighted combined stress score.

        Formula:
            Score = clip(weight_deficit * Deficit + weight_heterogeneity * Heterogeneity, 0.0, 1.0)

        Args:
            deficit: Canopy deficit score [0.0, 1.0].
            heterogeneity: Spatial heterogeneity score [0.0, 1.0].

        Returns:
            Combined score in [0.0, 1.0].
        """
        score = (self.weight_deficit * deficit) + (self.weight_heterogeneity * heterogeneity)
        return float(np.clip(score, 0.0, 1.0))

    def categorize_stress(self, score: float, is_valid: bool = True) -> StressCategory:
        """Categorize stress score into discrete agronomic tiers.

        Args:
            score: Combined stress score in [0.0, 1.0].
            is_valid: False if region lacks sufficient valid sensor coverage.

        Returns:
            StressCategory enum: INVALID, HEALTHY, MODERATE_STRESS, or HIGH_STRESS.
        """
        if not is_valid:
            return StressCategory.INVALID
        if score >= self.tau_high:
            return StressCategory.HIGH_STRESS
        elif score >= self.tau_low:
            return StressCategory.MODERATE_STRESS
        else:
            return StressCategory.HEALTHY

    def score_single_region(
        self,
        ndvi_map: np.ndarray,
        region: GridRegion,
        validity_mask: Optional[np.ndarray] = None,
        boundary_mask: Optional[np.ndarray] = None,
        ground_truth_mask: Optional[np.ndarray] = None,
        ground_truth_categories: Optional[List[str]] = None,
        category_masks: Optional[Dict[str, np.ndarray]] = None,
        category_overlap_threshold: float = 0.10,
    ) -> RegionStressScore:
        """Compute the multi-spectral stress score for an individual grid region.

        Args:
            ndvi_map: Pre-computed full-tile NDVI array of shape (H, W).
            region: GridRegion specifying spatial bounding coordinates.
            validity_mask: Optional 2D array of sensor validity coverage (>0 = valid).
            boundary_mask: Optional 2D array of farmland boundary (>0 = inside).
            ground_truth_mask: Optional 2D array of authentic anomaly masks (>0 = anomaly).
                               Used STRICTLY for evaluation logging; never leaks into score.
            ground_truth_categories: Optional list of anomaly names if ground-truth present.

        Returns:
            RegionStressScore dataclass instance with full scoring telemetry.
        """
        # Extract sub-regions
        y0, y1 = region.y_min, region.y_max
        x0, x1 = region.x_min, region.x_max
        ndvi_sub = ndvi_map[y0:y1, x0:x1]
        total_pixels = ndvi_sub.size

        # Combine validity and boundary masks
        valid_pixel_mask = np.ones_like(ndvi_sub, dtype=bool)

        if validity_mask is not None:
            v_sub = validity_mask[y0:y1, x0:x1]
            valid_pixel_mask = valid_pixel_mask & (v_sub > 0)

        if boundary_mask is not None:
            b_sub = boundary_mask[y0:y1, x0:x1]
            valid_pixel_mask = valid_pixel_mask & (b_sub > 0)

        valid_count = int(np.count_nonzero(valid_pixel_mask))
        valid_fraction = float(valid_count / total_pixels) if total_pixels > 0 else 0.0
        is_valid = valid_fraction >= self.min_valid_fraction and valid_count > 0

        # Handle invalid regions safely without division-by-zero or false stress alarms
        if not is_valid:
            gt_has_anomaly = None
            gt_frac = None
            if ground_truth_mask is not None:
                gt_sub = ground_truth_mask[y0:y1, x0:x1]
                gt_has_anomaly = bool(np.any(gt_sub > 0))
                gt_frac = float(np.count_nonzero(gt_sub > 0) / total_pixels) if total_pixels > 0 else 0.0

            region_categories = None
            if category_masks is not None:
                matched_cats = []
                for cat, m in category_masks.items():
                    c_pixels = int(np.count_nonzero(m[y0:y1, x0:x1] > 0))
                    if c_pixels > 0:
                        frac = float(c_pixels / total_pixels) if total_pixels > 0 else 0.0
                        if frac >= category_overlap_threshold:
                            matched_cats.append(cat)
                if matched_cats:
                    region_categories = matched_cats
            elif ground_truth_categories is not None and gt_has_anomaly:
                region_categories = ground_truth_categories

            return RegionStressScore(
                region_id=region.region_id,
                row=region.row,
                col=region.col,
                mean_ndvi=0.0,
                std_ndvi=0.0,
                deficit_score=0.0,
                heterogeneity_score=0.0,
                combined_stress_score=0.0,
                stress_category=StressCategory.INVALID,
                valid_pixel_count=valid_count,
                total_pixel_count=total_pixels,
                valid_fraction=valid_fraction,
                is_valid=False,
                gt_has_anomaly=gt_has_anomaly,
                gt_anomaly_fraction=gt_frac,
                gt_categories=region_categories,
            )

        # Extract valid NDVI pixels
        valid_ndvi = ndvi_sub[valid_pixel_mask]
        mean_ndvi = float(np.mean(valid_ndvi))
        std_ndvi = float(np.std(valid_ndvi))

        # Calculate score components
        deficit = self.compute_canopy_deficit(mean_ndvi)
        heterogeneity = self.compute_heterogeneity(std_ndvi)
        combined = self.compute_combined_score(deficit, heterogeneity)
        category = self.categorize_stress(combined, is_valid=True)

        # Record ground-truth metrics for validation benchmark if provided
        gt_has_anomaly = None
        gt_frac = None
        if ground_truth_mask is not None:
            gt_sub = ground_truth_mask[y0:y1, x0:x1]
            gt_pixels = int(np.count_nonzero(gt_sub > 0))
            gt_has_anomaly = gt_pixels > 0
            gt_frac = float(gt_pixels / total_pixels) if total_pixels > 0 else 0.0

        region_categories = None
        if category_masks is not None:
            matched_cats = []
            for cat, m in category_masks.items():
                c_pixels = int(np.count_nonzero(m[y0:y1, x0:x1] > 0))
                if c_pixels > 0:
                    frac = float(c_pixels / total_pixels) if total_pixels > 0 else 0.0
                    if frac >= category_overlap_threshold:
                        matched_cats.append(cat)
            if matched_cats:
                region_categories = matched_cats
        elif ground_truth_categories is not None and gt_has_anomaly:
            region_categories = ground_truth_categories

        return RegionStressScore(
            region_id=region.region_id,
            row=region.row,
            col=region.col,
            mean_ndvi=mean_ndvi,
            std_ndvi=std_ndvi,
            deficit_score=deficit,
            heterogeneity_score=heterogeneity,
            combined_stress_score=combined,
            stress_category=category,
            valid_pixel_count=valid_count,
            total_pixel_count=total_pixels,
            valid_fraction=valid_fraction,
            is_valid=True,
            gt_has_anomaly=gt_has_anomaly,
            gt_anomaly_fraction=gt_frac,
            gt_categories=region_categories,
        )

    def score_all_regions(
        self,
        rgb: np.ndarray,
        nir: np.ndarray,
        regions: List[GridRegion],
        validity_mask: Optional[np.ndarray] = None,
        boundary_mask: Optional[np.ndarray] = None,
        ground_truth_mask: Optional[np.ndarray] = None,
        ground_truth_categories: Optional[List[str]] = None,
        category_masks: Optional[Dict[str, np.ndarray]] = None,
        category_overlap_threshold: float = 0.10,
    ) -> List[RegionStressScore]:
        """Compute stress scores across an entire collection of grid regions.

        Args:
            rgb: Raw RGB image array (H, W, 3).
            nir: Raw NIR image array (H, W) or (H, W, 1).
            regions: List of GridRegion objects produced by GridGenerator.
            validity_mask: Optional sensor flight coverage validity mask (H, W).
            boundary_mask: Optional agricultural field boundary mask (H, W).
            ground_truth_mask: Optional combined ground truth anomaly mask (H, W).
            ground_truth_categories: Optional list of anomaly category names.

        Returns:
            List of RegionStressScore objects corresponding to the input regions.
        """
        red_f, nir_f = self.validate_and_align_inputs(rgb, nir)
        ndvi_map = self.compute_ndvi_map(red_f, nir_f)

        scores: List[RegionStressScore] = []
        for reg in regions:
            score_obj = self.score_single_region(
                ndvi_map=ndvi_map,
                region=reg,
                validity_mask=validity_mask,
                boundary_mask=boundary_mask,
                ground_truth_mask=ground_truth_mask,
                ground_truth_categories=ground_truth_categories,
                category_masks=category_masks,
                category_overlap_threshold=category_overlap_threshold,
            )
            scores.append(score_obj)

        return scores
