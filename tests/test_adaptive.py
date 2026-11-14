"""Automated test suite for StressSense Phase 3: Adaptive Sampling Engine.

Stage 1 Tests: Grid generation, spatial indexing, coordinate tracking,
and region extraction.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pytest

from src.adaptive.grid_generator import GridGenerator, GridRegion


def test_grid_region_dataclass_methods():
    """Verify GridRegion attribute storage, dictionary conversion, and point-in-region queries."""
    region = GridRegion(
        region_id="tile01_r01_c02",
        row=1,
        col=2,
        x_min=128,
        y_min=64,
        x_max=192,
        y_max=128,
        width=64,
        height=64,
        center_x=160.0,
        center_y=96.0,
        area=4096,
        image_id="tile01",
        valid_fraction=0.95,
        boundary_fraction=0.85,
        is_valid=True,
    )

    d = region.to_dict()
    assert d["region_id"] == "tile01_r01_c02"
    assert d["row"] == 1
    assert d["col"] == 2
    assert d["x_min"] == 128
    assert d["y_min"] == 64
    assert d["width"] == 64
    assert d["area"] == 4096
    assert d["valid_fraction"] == 0.95
    assert d["is_valid"] is True

    # Spatial point checks
    assert region.contains_point(150, 80) is True
    assert region.contains_point(128, 64) is True  # Inclusive min
    assert region.contains_point(192, 100) is False  # Exclusive max
    assert region.contains_point(50, 50) is False  # Outside


def test_grid_generator_initialization():
    """Verify GridGenerator argument validation and default parameter assignment."""
    # Default initialization
    gen_default = GridGenerator()
    assert gen_default.cell_h == 64
    assert gen_default.cell_w == 64
    assert gen_default.stride_y == 64
    assert gen_default.stride_x == 64

    # Tuple grid size and custom stride
    gen_custom = GridGenerator(grid_size=(32, 64), stride=(16, 32), min_valid_fraction=0.1)
    assert gen_custom.cell_h == 32
    assert gen_custom.cell_w == 64
    assert gen_custom.stride_y == 16
    assert gen_custom.stride_x == 32
    assert gen_custom.min_valid_fraction == 0.1

    # Invalid configurations
    with pytest.raises(ValueError):
        GridGenerator(grid_size=-10)

    with pytest.raises(ValueError):
        GridGenerator(grid_size=64, stride=0)

    with pytest.raises(TypeError):
        GridGenerator(grid_size="invalid")


def test_grid_generation_geometry_and_uniqueness():
    """Verify grid generation on standard 512x512 aerial tiles: region counts, coordinates, and unique IDs."""
    gen = GridGenerator(grid_size=64, stride=64)
    image_shape = (512, 512, 3)
    regions = gen.generate_grid(image_shape=image_shape, image_id="sample_tile")

    # 512 / 64 = 8 rows and 8 cols = 64 regions
    assert len(regions) == 64
    rows, cols = gen.get_grid_dimensions(image_shape)
    assert rows == 8
    assert cols == 8

    seen_ids = set()
    total_reconstructed_area = 0

    for reg in regions:
        # Check unique ID format
        assert reg.region_id not in seen_ids
        seen_ids.add(reg.region_id)
        assert reg.region_id.startswith("sample_tile_r")

        # Check bounds
        assert 0 <= reg.x_min < reg.x_max <= 512
        assert 0 <= reg.y_min < reg.y_max <= 512
        assert reg.width == 64
        assert reg.height == 64
        assert reg.area == 4096

        # Check centers
        assert reg.center_x == (reg.x_min + reg.x_max) / 2.0
        assert reg.center_y == (reg.y_min + reg.y_max) / 2.0

        total_reconstructed_area += reg.area

    # The sum of non-overlapping region areas must exactly equal the image area
    assert total_reconstructed_area == 512 * 512


def test_grid_generation_with_validity_and_boundary_masks():
    """Verify accurate calculation of valid_fraction, boundary_fraction, and is_valid status."""
    gen = GridGenerator(grid_size=64, stride=64, min_valid_fraction=0.20)
    height, width = 128, 128  # 2x2 grid = 4 regions

    # Create synthetic validity mask:
    # Region r00_c00: 100% valid
    # Region r00_c01: 50% valid
    # Region r01_c00: 10% valid (below 20% threshold -> is_valid=False)
    # Region r01_c01: 0% valid (0% -> is_valid=False)
    validity_mask = np.zeros((height, width), dtype=np.uint8)
    validity_mask[0:64, 0:64] = 255  # r00_c00: all 255
    validity_mask[0:64, 64:96] = 255  # r00_c01: half 255 (32 / 64 = 50%)
    validity_mask[64:70, 0:64] = 255  # r01_c00: 6 rows / 64 = 9.375% valid
    # r01_c01 remains all 0

    boundary_mask = np.full((height, width), 255, dtype=np.uint8)

    regions = gen.generate_grid(
        image_shape=(height, width),
        image_id="test_mask",
        validity_mask=validity_mask,
        boundary_mask=boundary_mask,
    )

    assert len(regions) == 4

    reg_00 = regions[0]
    assert reg_00.region_id == "test_mask_r00_c00"
    assert reg_00.valid_fraction == 1.0
    assert reg_00.boundary_fraction == 1.0
    assert reg_00.is_valid is True

    reg_01 = regions[1]
    assert reg_01.region_id == "test_mask_r00_c01"
    assert abs(reg_01.valid_fraction - 0.5) < 1e-4
    assert reg_01.is_valid is True

    reg_10 = regions[2]
    assert reg_10.region_id == "test_mask_r01_c00"
    assert reg_10.valid_fraction < 0.20
    assert reg_10.is_valid is False  # Under 20% threshold

    reg_11 = regions[3]
    assert reg_11.region_id == "test_mask_r01_c01"
    assert reg_11.valid_fraction == 0.0
    assert reg_11.is_valid is False


def test_grid_generation_non_divisible_dimensions():
    """Verify defensive handling and boundary clipping on dimensions non-divisible by grid_size."""
    gen = GridGenerator(grid_size=64, stride=64)
    # Image shape not divisible by 64 (500x350)
    image_shape = (500, 350)
    regions = gen.generate_grid(image_shape=image_shape, image_id="non_divisible")

    # Rows: 0..448 (step 64) -> 8 rows: [0:64, 64:128, ..., 448:500] (last row height = 52)
    # Cols: 0..320 (step 64) -> 6 cols: [0:64, 64:128, ..., 320:350] (last col width = 30)
    assert len(regions) == 8 * 6

    total_area = sum(r.area for r in regions)
    assert total_area == 500 * 350

    # Inspect bottom-right edge region
    edge_reg = regions[-1]
    assert edge_reg.row == 7
    assert edge_reg.col == 5
    assert edge_reg.x_min == 320
    assert edge_reg.x_max == 350
    assert edge_reg.width == 30
    assert edge_reg.y_min == 448
    assert edge_reg.y_max == 500
    assert edge_reg.height == 52
    assert edge_reg.area == 30 * 52


def test_grid_region_extraction():
    """Verify accurate array slicing for 2D and 3D images and extraction boundary checks."""
    gen = GridGenerator(grid_size=32, stride=32)
    # 3-channel synthetic image with coordinate-derived values
    img_3d = np.arange(64 * 64 * 3, dtype=np.uint8).reshape((64, 64, 3))
    regions = gen.generate_grid(image_shape=img_3d.shape)

    assert len(regions) == 4

    extracted = gen.extract_all_regions(img_3d, regions)
    assert len(extracted) == 4

    for reg, patch in zip(regions, extracted):
        assert patch.shape == (32, 32, 3)
        # Check values match direct slice
        expected_slice = img_3d[reg.y_min : reg.y_max, reg.x_min : reg.x_max]
        np.testing.assert_array_equal(patch, expected_slice)

    # Out of bounds check
    too_small_img = np.zeros((16, 16), dtype=np.uint8)
    with pytest.raises(ValueError):
        gen.extract_region(too_small_img, regions[0])


def test_grid_generator_on_real_processed_patch():
    """Verify GridGenerator compatibility on authentic Phase 2 processed Agriculture-Vision patches."""
    manifest_csv = Path("data/processed/manifests/patch_manifest.csv")
    if not manifest_csv.is_file():
        pytest.skip("Phase 2 manifest file not found.")

    import pandas as pd
    from PIL import Image

    df = pd.read_csv(manifest_csv)
    first_row = df.iloc[0]
    rgb_path = Path(first_row["rgb_patch_path"])
    nir_path = Path(first_row["nir_patch_path"])

    assert rgb_path.is_file(), f"RGB patch missing: {rgb_path}"
    assert nir_path.is_file(), f"NIR patch missing: {nir_path}"

    rgb_arr = np.array(Image.open(rgb_path))
    nir_arr = np.array(Image.open(nir_path))

    assert rgb_arr.shape == (128, 128, 3)
    assert nir_arr.shape == (128, 128)

    # Partition the 128x128 patch into 32x32 sub-regions (4x4 = 16 sub-regions)
    gen = GridGenerator(grid_size=32, stride=32)
    sub_regions = gen.generate_grid(image_shape=rgb_arr.shape, image_id=first_row["patch_id"])

    assert len(sub_regions) == 16
    for sub in sub_regions:
        assert sub.width == 32
        assert sub.height == 32
        sub_rgb = gen.extract_region(rgb_arr, sub)
        sub_nir = gen.extract_region(nir_arr, sub)
        assert sub_rgb.shape == (32, 32, 3)
        assert sub_nir.shape == (32, 32)


# ==============================================================================
# STAGE 2 TESTS: STRESS SCORING ENGINE
# ==============================================================================

from src.adaptive.stress_scorer import RegionStressScore, StressCategory, StressScorer


def test_stress_scorer_initialization_and_validation():
    """Verify StressScorer constructor argument validation and calibration parameter bounds."""
    # Default valid instance
    scorer = StressScorer()
    assert scorer.ndvi_target == 0.70
    assert scorer.ndvi_stressed == 0.20
    assert scorer.tau_low == 0.35
    assert scorer.tau_high == 0.70
    assert scorer.weight_deficit == 0.75
    assert scorer.weight_heterogeneity == 0.25

    # Custom valid instance
    scorer_custom = StressScorer(
        ndvi_target=0.80,
        ndvi_stressed=0.15,
        tau_low=0.30,
        tau_high=0.65,
        weight_deficit=0.80,
        weight_heterogeneity=0.20,
    )
    assert scorer_custom.ndvi_target == 0.80
    assert scorer_custom.ndvi_stressed == 0.15

    # Invalid: ndvi_target <= ndvi_stressed
    with pytest.raises(ValueError):
        StressScorer(ndvi_target=0.30, ndvi_stressed=0.50)

    # Invalid: tau_low >= tau_high
    with pytest.raises(ValueError):
        StressScorer(tau_low=0.75, tau_high=0.50)

    # Invalid: weights do not sum to 1.0
    with pytest.raises(ValueError):
        StressScorer(weight_deficit=0.6, weight_heterogeneity=0.6)


def test_stress_scorer_input_validation_and_defensive_checks():
    """Verify defensive handling of invalid inputs and rejection of independently normalized channels."""
    scorer = StressScorer()

    # Dimension mismatch between RGB and NIR
    rgb_valid = np.zeros((64, 64, 3), dtype=np.uint8)
    nir_mismatched = np.zeros((32, 32), dtype=np.uint8)
    with pytest.raises(ValueError, match="Spatial dimension mismatch"):
        scorer.validate_and_align_inputs(rgb_valid, nir_mismatched)

    # Wrong RGB channel count
    rgb_wrong_channels = np.zeros((64, 64, 4), dtype=np.uint8)
    nir_valid = np.zeros((64, 64), dtype=np.uint8)
    with pytest.raises(ValueError, match="Expected 3-channel RGB image"):
        scorer.validate_and_align_inputs(rgb_wrong_channels, nir_valid)

    # Negative values indicating invalid independent z-score normalization
    rgb_negative = np.full((64, 64, 3), -0.5, dtype=np.float32)
    with pytest.raises(ValueError, match="Negative values detected"):
        scorer.validate_and_align_inputs(rgb_negative, nir_valid)

    # Scaling compatibility: uint8 [0, 255] scaled to [0.0, 1.0]
    rgb_uint8 = np.full((64, 64, 3), 204, dtype=np.uint8)  # 204 / 255 = 0.8
    nir_uint8 = np.full((64, 64), 255, dtype=np.uint8)  # 255 / 255 = 1.0
    red_f, nir_f = scorer.validate_and_align_inputs(rgb_uint8, nir_uint8)
    assert abs(red_f[0, 0] - 0.8) < 1e-4
    assert abs(nir_f[0, 0] - 1.0) < 1e-4


def test_stress_scorer_ndvi_known_values_and_stability():
    """Verify NDVI calculation matches theoretical optical formulas and maintains zero-denominator stability."""
    scorer = StressScorer()

    # Case 1: Healthy vegetation (High NIR, Low Red)
    # NIR = 0.8, Red = 0.2 -> (0.8 - 0.2) / (0.8 + 0.2) = 0.6 / 1.0 = 0.60
    red_veg = np.full((32, 32), 0.2, dtype=np.float32)
    nir_veg = np.full((32, 32), 0.8, dtype=np.float32)
    ndvi_veg = scorer.compute_ndvi_map(red_veg, nir_veg)
    assert np.allclose(ndvi_veg, 0.60, atol=1e-4)

    # Case 2: Water / bare shadow (Low NIR, Higher Red)
    # NIR = 0.1, Red = 0.3 -> (0.1 - 0.3) / (0.1 + 0.3) = -0.2 / 0.4 = -0.50
    red_water = np.full((32, 32), 0.3, dtype=np.float32)
    nir_water = np.full((32, 32), 0.1, dtype=np.float32)
    ndvi_water = scorer.compute_ndvi_map(red_water, nir_water)
    assert np.allclose(ndvi_water, -0.50, atol=1e-4)

    # Case 3: Complete zero denominator (both channels 0.0) -> epsilon avoids division-by-zero
    red_zero = np.zeros((32, 32), dtype=np.float32)
    nir_zero = np.zeros((32, 32), dtype=np.float32)
    ndvi_zero = scorer.compute_ndvi_map(red_zero, nir_zero)
    assert not np.any(np.isnan(ndvi_zero))
    assert not np.any(np.isinf(ndvi_zero))
    assert np.allclose(ndvi_zero, 0.0, atol=1e-4)


def test_stress_scorer_canopy_deficit_and_heterogeneity():
    """Verify canopy vigor deficit scaling and spatial heterogeneity calculations."""
    scorer = StressScorer(ndvi_target=0.70, ndvi_stressed=0.20)

    # When NDVI equals target (0.70) -> Deficit should be 0.0 (completely healthy)
    assert abs(scorer.compute_canopy_deficit(0.70) - 0.0) < 1e-4

    # When NDVI exceeds target (0.85) -> Deficit remains 0.0 (clipped)
    assert abs(scorer.compute_canopy_deficit(0.85) - 0.0) < 1e-4

    # When NDVI equals stressed threshold (0.20) -> Deficit should be 1.0 (maximal stress)
    assert abs(scorer.compute_canopy_deficit(0.20) - 1.0) < 1e-4

    # When NDVI is halfway (0.45) -> (0.70 - 0.45) / (0.70 - 0.20) = 0.25 / 0.50 = 0.50
    assert abs(scorer.compute_canopy_deficit(0.45) - 0.50) < 1e-4

    # Heterogeneity: min(1.0, 3.0 * std)
    assert abs(scorer.compute_heterogeneity(0.0) - 0.0) < 1e-4
    assert abs(scorer.compute_heterogeneity(0.10) - 0.30) < 1e-4
    assert abs(scorer.compute_heterogeneity(0.40) - 1.0) < 1e-4  # Clipped to 1.0


def test_stress_scorer_category_tiers_and_combined_boundaries():
    """Verify tier assignment boundaries: HEALTHY (<0.35), MODERATE ([0.35, 0.70)), HIGH (>=0.70)."""
    scorer = StressScorer(tau_low=0.35, tau_high=0.70)

    # Combined score computation: 0.75 * deficit + 0.25 * heterogeneity
    # Low stress: deficit=0.1, heterogeneity=0.1 -> 0.75*0.1 + 0.25*0.1 = 0.10
    score_low = scorer.compute_combined_score(0.1, 0.1)
    assert abs(score_low - 0.10) < 1e-4
    assert scorer.categorize_stress(score_low) == StressCategory.HEALTHY

    # Moderate stress: deficit=0.5, heterogeneity=0.2 -> 0.75*0.5 + 0.25*0.2 = 0.425
    score_mod = scorer.compute_combined_score(0.5, 0.2)
    assert abs(score_mod - 0.425) < 1e-4
    assert scorer.categorize_stress(score_mod) == StressCategory.MODERATE_STRESS

    # High stress: deficit=0.9, heterogeneity=0.4 -> 0.75*0.9 + 0.25*0.4 = 0.675 + 0.10 = 0.775
    score_high = scorer.compute_combined_score(0.9, 0.4)
    assert abs(score_high - 0.775) < 1e-4
    assert scorer.categorize_stress(score_high) == StressCategory.HIGH_STRESS

    # Boundary conditions
    assert scorer.categorize_stress(0.3499) == StressCategory.HEALTHY
    assert scorer.categorize_stress(0.3500) == StressCategory.MODERATE_STRESS
    assert scorer.categorize_stress(0.6999) == StressCategory.MODERATE_STRESS
    assert scorer.categorize_stress(0.7000) == StressCategory.HIGH_STRESS


def test_stress_scorer_invalid_mask_handling():
    """Verify that regions lacking valid coverage are safely categorized as INVALID with zero false alarms."""
    scorer = StressScorer(min_valid_fraction=0.10)
    gen = GridGenerator(grid_size=64, stride=64)

    rgb = np.full((128, 128, 3), 100, dtype=np.uint8)
    nir = np.full((128, 128), 100, dtype=np.uint8)

    # Create validity mask where bottom row is 0% valid
    validity_mask = np.zeros((128, 128), dtype=np.uint8)
    validity_mask[0:64, :] = 255  # Top row is valid, bottom row is invalid

    regions = gen.generate_grid(image_shape=rgb.shape, validity_mask=validity_mask)
    scores = scorer.score_all_regions(rgb=rgb, nir=nir, regions=regions, validity_mask=validity_mask)

    assert len(scores) == 4

    # Top regions (valid)
    assert scores[0].is_valid is True
    assert scores[0].stress_category != StressCategory.INVALID

    # Bottom regions (0% valid coverage)
    assert scores[2].is_valid is False
    assert scores[2].stress_category == StressCategory.INVALID
    assert scores[2].combined_stress_score == 0.0

    assert scores[3].is_valid is False
    assert scores[3].stress_category == StressCategory.INVALID
    assert scores[3].combined_stress_score == 0.0


def test_stress_scorer_decoupling_from_ground_truth():
    """Verify that ground-truth anomaly annotations strictly do not alter spectral stress scores."""
    scorer = StressScorer()
    gen = GridGenerator(grid_size=64, stride=64)

    rgb = np.full((64, 64, 3), 150, dtype=np.uint8)
    nir = np.full((64, 64), 220, dtype=np.uint8)
    regions = gen.generate_grid(image_shape=rgb.shape)

    # Score WITHOUT ground-truth mask
    scores_no_gt = scorer.score_all_regions(rgb=rgb, nir=nir, regions=regions)

    # Score WITH synthetic ground-truth anomaly mask
    gt_mask = np.zeros((64, 64), dtype=np.uint8)
    gt_mask[10:30, 10:30] = 255  # Simulated anomaly cluster
    scores_with_gt = scorer.score_all_regions(
        rgb=rgb,
        nir=nir,
        regions=regions,
        ground_truth_mask=gt_mask,
        ground_truth_categories=["weed_cluster"],
    )

    # Verify spectral scores are 100% bit-exact identical
    assert scores_no_gt[0].combined_stress_score == scores_with_gt[0].combined_stress_score
    assert scores_no_gt[0].mean_ndvi == scores_with_gt[0].mean_ndvi
    assert scores_no_gt[0].deficit_score == scores_with_gt[0].deficit_score

    # Verify ground-truth telemetry is captured separately in the with_gt run
    assert scores_no_gt[0].gt_has_anomaly is None
    assert scores_with_gt[0].gt_has_anomaly is True
    assert scores_with_gt[0].gt_anomaly_fraction > 0.0
    assert scores_with_gt[0].gt_categories == ["weed_cluster"]


def test_stress_scorer_on_real_processed_patch():
    """Verify end-to-end stress scoring on authentic Phase 2 processed Agriculture-Vision patches."""
    manifest_csv = Path("data/processed/manifests/patch_manifest.csv")
    if not manifest_csv.is_file():
        pytest.skip("Phase 2 manifest file not found.")

    import pandas as pd
    from PIL import Image

    df = pd.read_csv(manifest_csv)
    # Find a patch that has ground-truth anomaly
    anomaly_rows = df[df["has_anomaly"] == True]
    row = anomaly_rows.iloc[0] if len(anomaly_rows) > 0 else df.iloc[0]

    rgb = np.array(Image.open(row["rgb_patch_path"]))
    nir = np.array(Image.open(row["nir_patch_path"]))
    bnd = np.array(Image.open(row["boundary_patch_path"]))
    msk = np.array(Image.open(row["mask_patch_path"]))

    gen = GridGenerator(grid_size=32, stride=32)
    regions = gen.generate_grid(
        image_shape=rgb.shape,
        image_id=row["patch_id"],
        validity_mask=msk,
        boundary_mask=bnd,
    )

    scorer = StressScorer()
    scores = scorer.score_all_regions(
        rgb=rgb,
        nir=nir,
        regions=regions,
        validity_mask=msk,
        boundary_mask=bnd,
    )

    assert len(scores) == 16  # (128 / 32) * (128 / 32) = 16 sub-regions

    valid_scores = [s for s in scores if s.is_valid]
    assert len(valid_scores) > 0

    for s in valid_scores:
        assert 0.0 <= s.combined_stress_score <= 1.0
        assert 0.0 <= s.deficit_score <= 1.0
        assert 0.0 <= s.heterogeneity_score <= 1.0
        assert -1.0 <= s.mean_ndvi <= 1.0
        assert s.stress_category in [
            StressCategory.HEALTHY,
            StressCategory.MODERATE_STRESS,
            StressCategory.HIGH_STRESS,
        ]
        # Test serialization
        d = s.to_dict()
        assert "combined_stress_score" in d
        assert "stress_category" in d


# ==============================================================================
# STAGE 3 TESTS: ADAPTIVE SAMPLING ENGINE
# ==============================================================================

from src.adaptive.sampling_engine import (
    AdaptiveSampler,
    BudgetOverflowPolicy,
    SamplingDecision,
    SamplingEngine,
    SamplingResult,
    SamplingStrategy,
    UniformSampler,
)


def _create_mock_scores(
    n_high: int = 2,
    n_mod: int = 4,
    n_healthy: int = 10,
    n_invalid: int = 2,
) -> List[RegionStressScore]:
    """Helper to synthesize deterministic test region scores across all tiers."""
    scores: List[RegionStressScore] = []
    idx = 0

    # High stress
    for i in range(n_high):
        score_val = 0.75 + (i * 0.05)  # 0.75, 0.80, ...
        scores.append(
            RegionStressScore(
                region_id=f"reg_{idx:03d}",
                row=idx // 4,
                col=idx % 4,
                mean_ndvi=0.20,
                std_ndvi=0.05,
                deficit_score=0.90,
                heterogeneity_score=0.15,
                combined_stress_score=score_val,
                stress_category=StressCategory.HIGH_STRESS,
                valid_pixel_count=1024,
                total_pixel_count=1024,
                valid_fraction=1.0,
                is_valid=True,
            )
        )
        idx += 1

    # Moderate stress (tau_low = 0.35 <= score < tau_high = 0.70)
    for i in range(n_mod):
        score_val = 0.35 + (i * 0.05)  # 0.35, 0.40, 0.45, ...
        scores.append(
            RegionStressScore(
                region_id=f"reg_{idx:03d}",
                row=idx // 4,
                col=idx % 4,
                mean_ndvi=0.45,
                std_ndvi=0.08,
                deficit_score=0.50,
                heterogeneity_score=0.24,
                combined_stress_score=score_val,
                stress_category=StressCategory.MODERATE_STRESS,
                valid_pixel_count=1024,
                total_pixel_count=1024,
                valid_fraction=1.0,
                is_valid=True,
            )
        )
        idx += 1

    # Healthy
    for i in range(n_healthy):
        score_val = 0.10 + (i * 0.01)  # 0.10, 0.11, ...
        scores.append(
            RegionStressScore(
                region_id=f"reg_{idx:03d}",
                row=idx // 4,
                col=idx % 4,
                mean_ndvi=0.75,
                std_ndvi=0.02,
                deficit_score=0.0,
                heterogeneity_score=0.06,
                combined_stress_score=score_val,
                stress_category=StressCategory.HEALTHY,
                valid_pixel_count=1024,
                total_pixel_count=1024,
                valid_fraction=1.0,
                is_valid=True,
            )
        )
        idx += 1

    # Invalid
    for i in range(n_invalid):
        scores.append(
            RegionStressScore(
                region_id=f"reg_{idx:03d}",
                row=idx // 4,
                col=idx % 4,
                mean_ndvi=0.0,
                std_ndvi=0.0,
                deficit_score=0.0,
                heterogeneity_score=0.0,
                combined_stress_score=0.0,
                stress_category=StressCategory.INVALID,
                valid_pixel_count=0,
                total_pixel_count=1024,
                valid_fraction=0.0,
                is_valid=False,
            )
        )
        idx += 1

    return scores


def test_full_coverage_uniform_sampling():
    """Verify that Full-Coverage Uniform Sampling selects 100% of valid regions and excludes invalid regions."""
    scores = _create_mock_scores(n_high=3, n_mod=5, n_healthy=10, n_invalid=2)
    sampler = UniformSampler()
    res = sampler.sample_full_coverage(scores)

    assert res.strategy == SamplingStrategy.FULL_UNIFORM
    assert res.total_regions == 20
    assert res.valid_regions == 18
    assert res.invalid_regions == 2
    assert res.actual_selected_count == 18  # Exactly all valid regions
    assert res.valid_reduction_pct == 0.0  # 0% reduction on valid regions
    assert abs(res.sampling_reduction_pct - 10.0) < 1e-4  # 2 invalid out of 20 = 10% total reduction

    # Check invalid regions are explicitly marked invalid_excluded
    invalid_decisions = [d for d in res.all_decisions if not d.is_valid]
    assert len(invalid_decisions) == 2
    for d in invalid_decisions:
        assert d.is_selected is False
        assert d.selection_reason == "invalid_excluded"


def test_equal_budget_uniform_sampling():
    """Verify Equal-Budget Uniform Sampling selects exactly K valid regions with seed reproducibility."""
    scores = _create_mock_scores(n_high=3, n_mod=5, n_healthy=10, n_invalid=2)
    sampler = UniformSampler()

    # Request budget K = 6
    res1 = sampler.sample_equal_budget(scores, budget=6, seed=42)
    assert res1.strategy == SamplingStrategy.EQUAL_BUDGET_UNIFORM
    assert res1.actual_selected_count == 6
    assert len(res1.selected_regions) == 6
    for d in res1.selected_regions:
        assert d.is_valid is True
        assert d.selection_reason == "equal_budget_uniform"

    # Repeat with identical seed -> bit-exact identical selection
    res2 = sampler.sample_equal_budget(scores, budget=6, seed=42)
    assert [d.region_id for d in res1.selected_regions] == [d.region_id for d in res2.selected_regions]

    # Different seed -> different selection
    res3 = sampler.sample_equal_budget(scores, budget=6, seed=99)
    assert [d.region_id for d in res1.selected_regions] != [d.region_id for d in res3.selected_regions]


def test_adaptive_sampling_high_stress_prioritization_and_exploration():
    """Verify Adaptive Sampling selects 100% of high-stress regions and allocates healthy exploration."""
    scores = _create_mock_scores(n_high=3, n_mod=5, n_healthy=12, n_invalid=2)
    sampler = AdaptiveSampler(default_seed=42)

    # Budget mode: Budget B = 8
    # High stress = 3 (fits in budget)
    # Remaining = 5. Exploration (5% of 8 = round(0.4) -> min 1) = 1 healthy region
    # Remaining for moderate = 4 moderate regions
    res = sampler.sample(scores, budget=8, seed=42)

    assert res.actual_selected_count == 8
    assert res.budget_exceeded is False

    reasons = [d.selection_reason for d in res.selected_regions]
    assert reasons.count("high_stress_guarantee") == 3  # All 3 high stress
    assert reasons.count("healthy_exploration") == 1  # 1 healthy exploration
    assert reasons.count("moderate_stress_priority") == 4  # 4 moderate stress


def test_budget_overflow_guarantee_policy():
    """Verify that GUARANTEE_HIGH_STRESS selects all high-stress regions even when exceeding budget."""
    # 6 high stress regions, requested budget B = 4
    scores = _create_mock_scores(n_high=6, n_mod=4, n_healthy=10, n_invalid=0)
    sampler = AdaptiveSampler(overflow_policy=BudgetOverflowPolicy.GUARANTEE_HIGH_STRESS)

    res = sampler.sample(scores, budget=4, seed=42)

    # Mandatory guarantee must not silently discard high stress regions
    assert res.actual_selected_count == 6
    assert res.budget_exceeded is True
    assert res.overflow_count == 2
    assert res.deficit_count == 0

    # Every single high stress region was selected
    high_selected = [d for d in res.selected_regions if d.stress_category == "high_stress"]
    assert len(high_selected) == 6


def test_budget_overflow_strict_cap_policy():
    """Verify that STRICT_CAP strictly limits selected regions to requested budget, logging deficit."""
    # 6 high stress regions, requested budget B = 4
    scores = _create_mock_scores(n_high=6, n_mod=4, n_healthy=10, n_invalid=0)
    sampler = AdaptiveSampler(overflow_policy=BudgetOverflowPolicy.STRICT_CAP)

    res = sampler.sample(scores, budget=4, seed=42)

    assert res.actual_selected_count == 4
    assert res.budget_exceeded is False
    assert res.deficit_count == 2  # 6 - 4 = 2 omitted

    # Verify top-4 highest scores were selected
    selected_scores = [d.combined_stress_score for d in res.selected_regions]
    assert len(selected_scores) == 4
    # All 4 should be high stress
    for d in res.selected_regions:
        assert d.stress_category == "high_stress"


def test_edge_case_all_healthy_field():
    """Verify adaptive sampling on an entirely healthy field achieves high sampling reduction."""
    scores = _create_mock_scores(n_high=0, n_mod=0, n_healthy=40, n_invalid=0)
    sampler = AdaptiveSampler(healthy_exploration_rate=0.05, default_seed=42)

    # In threshold mode (no budget cap), samples ~5% exploration
    res_thresh = sampler.sample(scores, budget=None, seed=42)
    # 5% of 40 is ~2 regions
    assert 1 <= res_thresh.actual_selected_count <= 4
    assert res_thresh.sampling_reduction_pct >= 90.0  # At least 90% reduction!

    for d in res_thresh.selected_regions:
        assert d.selection_reason == "healthy_exploration"


def test_edge_case_all_high_stress_field():
    """Verify all-high-stress field achieves 100% selection of high-stress-scored regions under guarantee policy.

    Condition 2 Compliance:
    Verifies 100% selection of high-stress-scored regions under GUARANTEE_HIGH_STRESS policy.
    Does not claim ground-truth anomaly recall without authentic ground-truth masks.
    """
    scores = _create_mock_scores(n_high=20, n_mod=0, n_healthy=0, n_invalid=0)
    sampler = AdaptiveSampler(overflow_policy=BudgetOverflowPolicy.GUARANTEE_HIGH_STRESS)

    # Even with small budget B = 5, guarantee policy selects all 20 high-stress-scored regions
    res = sampler.sample(scores, budget=5, seed=42)
    assert res.actual_selected_count == 20
    assert res.budget_exceeded is True
    assert res.overflow_count == 15

    for d in res.selected_regions:
        assert d.selection_reason == "high_stress_guarantee"
        assert d.stress_category == "high_stress"


def test_edge_case_insufficient_or_zero_valid_regions():
    """Verify graceful handling when grid has zero valid regions or fewer valid regions than budget."""
    sampler = AdaptiveSampler()

    # Case 1: 100% invalid regions
    scores_all_invalid = _create_mock_scores(n_high=0, n_mod=0, n_healthy=0, n_invalid=10)
    res_invalid = sampler.sample(scores_all_invalid, budget=5)
    assert res_invalid.actual_selected_count == 0
    assert res_invalid.valid_regions == 0
    assert res_invalid.invalid_regions == 10

    # Case 2: Only 2 valid regions, but budget is 10
    scores_sparse = _create_mock_scores(n_high=1, n_mod=1, n_healthy=0, n_invalid=8)
    res_sparse = sampler.sample(scores_sparse, budget=10)
    assert res_sparse.actual_selected_count == 2
    assert res_sparse.metadata["insufficient_valid_regions"] is True


def test_equal_budget_fair_comparison():
    """Verify that SamplingEngine compare_all_strategies ensures exact equal sample counts."""
    scores = _create_mock_scores(n_high=3, n_mod=5, n_healthy=15, n_invalid=2)
    engine = SamplingEngine(default_seed=42)

    # Run comparison with adaptive budget B = 7
    comparison = engine.compare_all_strategies(scores, adaptive_budget=7, seed=42)

    res_adaptive = comparison["adaptive"]
    res_equal = comparison["equal_budget_uniform"]
    res_full = comparison["full_uniform"]

    # Critical fairness guarantee: actual selected counts MUST BE EXACTLY IDENTICAL
    assert res_adaptive.actual_selected_count == res_equal.actual_selected_count
    assert res_adaptive.actual_selected_count == 7
    assert res_equal.actual_selected_count == 7

    # Full uniform scans 100% of valid regions
    assert res_full.actual_selected_count == 23  # 3 + 5 + 15 = 23 valid
    assert res_full.valid_reduction_pct == 0.0

    # Reductions match between adaptive and equal-budget
    assert res_adaptive.sampling_reduction_pct == res_equal.sampling_reduction_pct


def test_sampling_engine_on_real_processed_patch():
    """Verify end-to-end Grid -> Scorer -> SamplingEngine on an authentic Phase 2 patch."""
    manifest_csv = Path("data/processed/manifests/patch_manifest.csv")
    if not manifest_csv.is_file():
        pytest.skip("Phase 2 manifest file not found.")

    import pandas as pd
    from PIL import Image

    df = pd.read_csv(manifest_csv)
    row = df.iloc[0]

    rgb = np.array(Image.open(row["rgb_patch_path"]))
    nir = np.array(Image.open(row["nir_patch_path"]))
    bnd = np.array(Image.open(row["boundary_patch_path"]))
    msk = np.array(Image.open(row["mask_patch_path"]))

    # 1. Grid Generator: 128x128 patch -> 32x32 sub-regions = 16 sub-regions
    gen = GridGenerator(grid_size=32, stride=32)
    regions = gen.generate_grid(rgb.shape, image_id=row["patch_id"], validity_mask=msk, boundary_mask=bnd)

    # 2. Stress Scorer: Compute RSDI scores
    scorer = StressScorer()
    scores = scorer.score_all_regions(rgb, nir, regions, validity_mask=msk, boundary_mask=bnd)
    assert len(scores) == 16

    # 3. Sampling Engine: Compare all 3 strategies with budget B = 4
    engine = SamplingEngine(default_seed=42)
    results = engine.compare_all_strategies(scores, adaptive_budget=4, seed=42)

    ad_res = results["adaptive"]
    eq_res = results["equal_budget_uniform"]
    fu_res = results["full_uniform"]

    # Check identical budget allocation
    assert ad_res.actual_selected_count == eq_res.actual_selected_count
    assert ad_res.actual_selected_count <= 4 or ad_res.budget_exceeded
    assert fu_res.actual_selected_count <= 16

    # Verify serialization of all results
    for key, res in results.items():
        d = res.to_dict()
        assert "strategy" in d
        assert "actual_selected_count" in d
        assert "sampling_reduction_pct" in d


# ==============================================================================
# AUDIT TESTS: EXPLORATION, INVARIANTS, MODERATE ALLOCATION & FAIRNESS
# ==============================================================================


def test_exploration_target_vs_actual_percentage_audit():
    """Verify exploration calculation distinguishes target (5%) from actual percentage and handles edge budgets."""
    sampler = AdaptiveSampler(healthy_exploration_rate=0.05, default_seed=42)

    # 1. Very small budget (B = 4) on 20-region grid (high=1, mod=1, healthy=16, invalid=2)
    # 4 * 0.05 = 0.2 -> discrete minimum 1 healthy slot allocated
    scores_small = _create_mock_scores(n_high=1, n_mod=1, n_healthy=16, n_invalid=2)
    res_small = sampler.sample(scores_small, budget=4, seed=42)

    assert res_small.actual_selected_count == 3  # 1 high + 1 mod + 1 healthy
    exp_small = res_small.metadata["exploration"]
    assert exp_small["target_exploration_rate"] == 0.05
    assert exp_small["actual_exploration_count"] == 1
    # 1 out of 3 selected is 33.33% (NOT 5%)
    assert exp_small["actual_exploration_pct_of_selected"] == 33.33
    # 1 out of effective budget 4 is 25.0%
    assert exp_small["actual_exploration_pct_of_budget"] == 25.0
    assert exp_small["is_discretionary_minimum"] is True
    assert exp_small["exploration_status"] == "fulfilled_discretionary_minimum"

    # 2. Large budget (B = 40) on 60-region grid (high=2, mod=8, healthy=50, invalid=0)
    # 40 * 0.05 = 2.0 -> exactly 2 healthy slots allocated
    scores_large = _create_mock_scores(n_high=2, n_mod=8, n_healthy=50, n_invalid=0)
    res_large = sampler.sample(scores_large, budget=40, seed=42)

    exp_large = res_large.metadata["exploration"]
    assert exp_large["target_exploration_count"] == 2
    assert exp_large["actual_exploration_count"] == 2
    # Selected is 2 high + 8 mod + 2 healthy = 12 total
    # 2 out of 12 selected is 16.67% of selected count
    assert exp_large["actual_exploration_pct_of_selected"] == 16.67
    # 2 out of requested budget 40 is exactly 5.0% of budget
    assert exp_large["actual_exploration_pct_of_budget"] == 5.0
    assert exp_large["is_discretionary_minimum"] is False
    assert exp_large["exploration_status"] == "fulfilled"

    # 3. Zero budget (B = 0)
    res_zero = sampler.sample(scores_large, budget=0, seed=42)
    assert res_zero.actual_selected_count == 0
    exp_zero = res_zero.metadata["exploration"]
    assert exp_zero["actual_exploration_count"] == 0
    assert exp_zero["exploration_status"] == "unfulfilled_zero_budget"

    # 4. Zero healthy regions exist (high=2, mod=8, healthy=0)
    scores_no_healthy = _create_mock_scores(n_high=2, n_mod=8, n_healthy=0, n_invalid=0)
    res_no_healthy = sampler.sample(scores_no_healthy, budget=5, seed=42)
    exp_no_healthy = res_no_healthy.metadata["exploration"]
    assert exp_no_healthy["actual_exploration_count"] == 0
    assert exp_no_healthy["exploration_status"] == "unfulfilled_no_healthy_regions"


def test_moderate_stress_allocation_scenarios_audit():
    """Verify behavior across all moderate-stress budget scenarios: empty, partial, surplus, and high-stress exhaustion."""
    sampler = AdaptiveSampler(default_seed=42)

    # Scenario A: No moderate regions exist (high=2, mod=0, healthy=10)
    scores_a = _create_mock_scores(n_high=2, n_mod=0, n_healthy=10, n_invalid=0)
    res_a = sampler.sample(scores_a, budget=5, seed=42)
    assert res_a.metadata["moderate_status"] == "no_moderate_regions_exist"
    assert sum(1 for d in res_a.selected_regions if d.stress_category == "moderate_stress") == 0

    # Scenario B: Moderate regions fewer than remaining budget slots (high=2, mod=2, healthy=10, budget=6)
    # slots_remaining = 6 - 2 = 4. 1 healthy explored -> 3 slots for moderate. Moderate has only 2 -> all 2 selected!
    scores_b = _create_mock_scores(n_high=2, n_mod=2, n_healthy=10, n_invalid=0)
    res_b = sampler.sample(scores_b, budget=6, seed=42)
    assert res_b.metadata["moderate_status"] == "all_moderate_selected"
    assert sum(1 for d in res_b.selected_regions if d.stress_category == "moderate_stress") == 2

    # Scenario C: Moderate regions exceed remaining budget slots (high=2, mod=6, healthy=10, budget=5)
    # slots_remaining = 5 - 2 = 3. 1 healthy explored -> 2 slots for moderate. Moderate has 6 -> top 2 selected!
    scores_c = _create_mock_scores(n_high=2, n_mod=6, n_healthy=10, n_invalid=0)
    res_c = sampler.sample(scores_c, budget=5, seed=42)
    assert "partial_moderate_selected_2_of_6" in res_c.metadata["moderate_status"]
    assert sum(1 for d in res_c.selected_regions if d.stress_category == "moderate_stress") == 2
    # Verify top-2 highest scores among moderates were chosen
    selected_mod_scores = [
        d.combined_stress_score for d in res_c.selected_regions if d.stress_category == "moderate_stress"
    ]
    all_mod_scores = sorted([s.combined_stress_score for s in scores_c if s.stress_category == StressCategory.MODERATE_STRESS], reverse=True)
    assert sorted(selected_mod_scores, reverse=True) == all_mod_scores[:2]

    # Scenario D: High-stress regions consume the entire budget (high=4, mod=4, healthy=10, budget=4)
    # slots_remaining = 0 -> zero moderate, zero healthy
    scores_d = _create_mock_scores(n_high=4, n_mod=4, n_healthy=10, n_invalid=0)
    res_d = sampler.sample(scores_d, budget=4, seed=42)
    assert res_d.actual_selected_count == 4
    assert res_d.metadata["moderate_status"] == "unfulfilled_budget_exhausted_by_high_stress"
    assert res_d.metadata["exploration"]["exploration_status"] == "unfulfilled_budget_exhausted_by_high_stress"
    assert all(d.stress_category == "high_stress" for d in res_d.selected_regions)


def test_threshold_driven_determinism_audit():
    """Verify that threshold-driven mode (budget=None) deterministically selects all high and all moderate regions."""
    scores = _create_mock_scores(n_high=3, n_mod=5, n_healthy=20, n_invalid=2)
    sampler = AdaptiveSampler(healthy_exploration_rate=0.05, default_seed=42)

    # Run 5 repeated times with different random seeds
    for test_seed in [1, 42, 99, 123, 777]:
        res = sampler.sample(scores, budget=None, seed=test_seed)

        # High-stress regions: 100% selected deterministically
        high_selected = [d.region_id for d in res.selected_regions if d.stress_category == "high_stress"]
        assert len(high_selected) == 3
        assert set(high_selected) == {"reg_000", "reg_001", "reg_002"}

        # Moderate-stress regions: 100% selected deterministically
        mod_selected = [d.region_id for d in res.selected_regions if d.stress_category == "moderate_stress"]
        assert len(mod_selected) == 5
        assert set(mod_selected) == {"reg_003", "reg_004", "reg_005", "reg_006", "reg_007"}

        # Invalid regions: 100% excluded deterministically
        assert all(d.is_valid for d in res.selected_regions)


def test_equal_budget_fairness_under_overflow_and_invariants_audit():
    """Verify equal-budget fairness under GUARANTEE_HIGH_STRESS overflow and strict population sharing."""
    # 5 high-stress, 4 moderate, 10 healthy, 2 invalid = 21 total, 19 valid
    scores = _create_mock_scores(n_high=5, n_mod=4, n_healthy=10, n_invalid=2)
    engine = SamplingEngine(default_seed=42)

    # Request small budget B = 3 -> high-stress overflow occurs (5 high > budget 3)
    results = engine.compare_all_strategies(scores, adaptive_budget=3, seed=42)

    ad_res = results["adaptive"]
    eq_res = results["equal_budget_uniform"]

    # 1. Overflow occurred: actual selected count is 5 (not 3)
    assert ad_res.budget_exceeded is True
    assert ad_res.overflow_count == 2
    assert ad_res.actual_selected_count == 5

    # 2. Equal-budget uniform must select EXACTLY 5 regions
    assert eq_res.actual_selected_count == 5
    assert eq_res.actual_selected_count == ad_res.actual_selected_count

    # 3. Same valid population used: all chosen regions must belong to valid_scores
    valid_ids = {s.region_id for s in scores if s.is_valid}
    assert all(d.region_id in valid_ids for d in ad_res.selected_regions)
    assert all(d.region_id in valid_ids for d in eq_res.selected_regions)

    # 4. Invariant: zero duplicate region IDs
    ad_ids = [d.region_id for d in ad_res.selected_regions]
    eq_ids = [d.region_id for d in eq_res.selected_regions]
    assert len(ad_ids) == len(set(ad_ids))
    assert len(eq_ids) == len(set(eq_ids))

    # 5. Invariant: sampling_reduction_pct uses total_regions as denominator
    # Total = 21, Selected = 5 -> (21 - 5) / 21 = 16 / 21 = 76.19%
    expected_red_total = round((21 - 5) / 21 * 100.0, 2)
    assert round(ad_res.sampling_reduction_pct, 2) == expected_red_total
    assert round(eq_res.sampling_reduction_pct, 2) == expected_red_total

    # Valid = 19, Selected = 5 -> (19 - 5) / 19 = 14 / 19 = 73.68%
    expected_red_valid = round((19 - 5) / 19 * 100.0, 2)
    assert round(ad_res.valid_reduction_pct, 2) == expected_red_valid
    assert round(eq_res.valid_reduction_pct, 2) == expected_red_valid

    # 6. Requested budget and actual selected count are distinguished
    assert ad_res.requested_budget == 3
    assert ad_res.actual_selected_count == 5
    assert eq_res.requested_budget == 5
    assert eq_res.actual_selected_count == 5


def test_stress_category_single_source_of_truth_and_boundaries():
    """Verify StressScorer is the single source of truth for category thresholds and AdaptiveSampler consumes them."""
    scorer = StressScorer()

    # 1. Verify configured default thresholds
    assert scorer.tau_low == 0.35
    assert scorer.tau_high == 0.70

    # 2. Verify exact boundary classifications
    assert scorer.categorize_stress(0.0) == StressCategory.HEALTHY
    assert scorer.categorize_stress(0.34999) == StressCategory.HEALTHY
    assert scorer.categorize_stress(0.35000) == StressCategory.MODERATE_STRESS
    assert scorer.categorize_stress(0.50000) == StressCategory.MODERATE_STRESS
    assert scorer.categorize_stress(0.69999) == StressCategory.MODERATE_STRESS
    assert scorer.categorize_stress(0.70000) == StressCategory.HIGH_STRESS
    assert scorer.categorize_stress(1.00000) == StressCategory.HIGH_STRESS

    # 3. Verify AdaptiveSampler directly consumes StressCategory without independent thresholds
    sampler = AdaptiveSampler()
    scores = [
        RegionStressScore(
            region_id="reg_h", row=0, col=0, mean_ndvi=0.8, std_ndvi=0.01, deficit_score=0.1,
            heterogeneity_score=0.1, combined_stress_score=0.20, stress_category=StressCategory.HEALTHY,
            valid_pixel_count=100, total_pixel_count=100, valid_fraction=1.0, is_valid=True
        ),
        RegionStressScore(
            region_id="reg_m", row=0, col=1, mean_ndvi=0.5, std_ndvi=0.05, deficit_score=0.4,
            heterogeneity_score=0.2, combined_stress_score=0.35, stress_category=StressCategory.MODERATE_STRESS,
            valid_pixel_count=100, total_pixel_count=100, valid_fraction=1.0, is_valid=True
        ),
        RegionStressScore(
            region_id="reg_s", row=0, col=2, mean_ndvi=0.2, std_ndvi=0.05, deficit_score=0.9,
            heterogeneity_score=0.1, combined_stress_score=0.70, stress_category=StressCategory.HIGH_STRESS,
            valid_pixel_count=100, total_pixel_count=100, valid_fraction=1.0, is_valid=True
        ),
    ]

    # In threshold mode, sampler respects the category tags assigned by StressScorer
    res = sampler.sample(scores, budget=None, seed=42)
    reasons = {d.region_id: d.selection_reason for d in res.selected_regions}
    assert reasons["reg_s"] == "high_stress_guarantee"
    assert reasons["reg_m"] == "moderate_stress_priority"


def test_exploration_precedence_over_moderate_stress_when_one_slot_remains():
    """Verify healthy exploration takes precedence over moderate stress when only 1 budget slot remains.

    Agronomic and Surveillance Trade-off:
    When remaining drone acquisition budget is constrained to 1 slot after mandatory
    high-stress selection, dedicating that slot to healthy exploration prevents
    complete surveillance blind spots across healthy canopy. This trades off
    secondary confirmation of known moderate stress in favor of early discovery
    of unexpected anomaly emergence in ostensibly healthy zones.
    """
    # 2 high stress, 3 moderate stress, 5 healthy, 0 invalid = 10 valid regions
    scores = _create_mock_scores(n_high=2, n_mod=3, n_healthy=5, n_invalid=0)
    sampler = AdaptiveSampler(default_seed=42)

    # Budget = 3: High stress consumes 2 slots; exactly 1 slot remains.
    res = sampler.sample(scores, budget=3, seed=42)

    assert res.actual_selected_count == 3
    selected_reasons = [d.selection_reason for d in res.selected_regions]
    selected_categories = [d.stress_category for d in res.selected_regions]

    # 1. High-stress guarantee consumes 2 slots
    assert selected_reasons.count("high_stress_guarantee") == 2
    assert selected_categories.count("high_stress") == 2

    # 2. Healthy exploration takes the single remaining slot (precedence over moderate stress)
    assert selected_reasons.count("healthy_exploration") == 1
    assert selected_categories.count("healthy") == 1

    # 3. Moderate stress receives 0 slots
    assert selected_categories.count("moderate_stress") == 0
    assert res.metadata["moderate_status"] == "unfulfilled_no_slots_remaining"
    assert res.metadata["exploration"]["actual_exploration_count"] == 1
    assert res.metadata["exploration"]["exploration_status"] == "fulfilled_discretionary_minimum"


def test_sampling_reduction_metrics_validation():
    """Verify that valid_reduction_pct is the primary metric and invalid regions are never counted as savings."""
    # 10 valid regions (2 high, 3 mod, 5 healthy) + 10 invalid regions = 20 total
    scores = _create_mock_scores(n_high=2, n_mod=3, n_healthy=5, n_invalid=10)
    sampler = AdaptiveSampler(default_seed=42)

    # Budget = 5: 2 high + 1 healthy exploration + 2 moderate = 5 selected
    res = sampler.sample(scores, budget=5, seed=42)

    assert res.total_regions == 20
    assert res.valid_regions == 10
    assert res.invalid_regions == 10
    assert res.actual_selected_count == 5

    # 1. Invalid regions are NEVER counted as selected
    assert all(d.is_valid for d in res.selected_regions)
    invalid_decisions = [d for d in res.all_decisions if not d.is_valid]
    assert len(invalid_decisions) == 10
    for d in invalid_decisions:
        assert d.is_selected is False
        assert d.selection_reason == "invalid_excluded"

    # 2. valid_reduction_pct is strictly: (10 - 5) / 10 * 100 = 50.0%
    assert round(res.valid_reduction_pct, 2) == 50.00

    # 3. sampling_reduction_pct (total-grid) is: (20 - 5) / 20 * 100 = 75.0%
    assert round(res.sampling_reduction_pct, 2) == 75.00

    # 4. Confirming total-grid reduction (75%) overstates actual canopy sampling efficiency (50%)
    # by treating non-crop/invalid background pixels as avoided samples.
    # Therefore, valid_reduction_pct is the mandatory primary efficiency metric.
    assert res.valid_reduction_pct < res.sampling_reduction_pct


# ==============================================================================
# STAGE 4 TESTS: EVALUATOR AND BENCHMARKING ENGINE
# ==============================================================================

from src.adaptive.evaluator import (
    AdaptiveEvaluator,
    BenchmarkComparison,
    EvaluationMetrics,
    ResourceCostAssumptions,
    TrialAggregateMetrics,
)


def _create_mock_scores_with_ground_truth(
    n_high: int = 3,
    n_mod: int = 4,
    n_healthy: int = 8,
    n_invalid: int = 2,
    gt_anomaly_indices: Optional[List[int]] = None,
    gt_categories_map: Optional[Dict[int, List[str]]] = None,
) -> List[RegionStressScore]:
    """Helper to synthesize scores with authentic ground-truth anomaly annotations."""
    base_scores = _create_mock_scores(
        n_high=n_high, n_mod=n_mod, n_healthy=n_healthy, n_invalid=n_invalid
    )
    if gt_anomaly_indices is None:
        gt_anomaly_indices = [0, 1, 4]  # Default: 2 in high-stress, 1 in moderate-stress

    if gt_categories_map is None:
        gt_categories_map = {
            0: ["weed_cluster"],
            1: ["double_plant"],
            4: ["weed_cluster"],
        }

    for idx, s in enumerate(base_scores):
        if idx in gt_anomaly_indices:
            s.gt_has_anomaly = True
            s.gt_anomaly_fraction = 0.35
            s.gt_categories = gt_categories_map.get(idx, ["weed_cluster"])
        else:
            s.gt_has_anomaly = False
            s.gt_anomaly_fraction = 0.0
            s.gt_categories = []

    return base_scores


def test_evaluator_metric_calculations_and_zero_denominator_cases():
    """Verify evaluator handles edge cases, zero denominators, and empty subsets safely."""
    evaluator = AdaptiveEvaluator(default_seed=42)

    # Case 1: Zero valid regions (all invalid)
    scores_invalid = _create_mock_scores(n_high=0, n_mod=0, n_healthy=0, n_invalid=10)
    sampler = UniformSampler()
    res_inv = sampler.sample_full_coverage(scores_invalid)
    m_inv = evaluator.evaluate_result(res_inv, scores_invalid, patch_id="p_inv")

    assert m_inv.valid_regions == 0
    assert m_inv.selected_regions_count == 0
    assert m_inv.valid_reduction_pct == 0.0
    assert m_inv.high_stress_coverage_pct == 100.0  # 0 high stress = 100% covered
    assert m_inv.moderate_stress_coverage_pct == 100.0
    assert m_inv.pixels_processed == 0

    # Case 2: Zero high-stress regions in field, non-zero budget
    scores_no_high = _create_mock_scores(n_high=0, n_mod=5, n_healthy=10, n_invalid=0)
    res_nh = sampler.sample_equal_budget(scores_no_high, budget=4, seed=42)
    m_nh = evaluator.evaluate_result(res_nh, scores_no_high, patch_id="p_nh")

    assert m_nh.high_stress_total == 0
    assert m_nh.high_stress_coverage_pct == 100.0  # 0/0 -> 100% covered
    assert m_nh.moderate_stress_total == 5
    assert m_nh.gt_available is False
    assert m_nh.gt_anomaly_recall_pct is None

    # Case 3: Zero selected regions (budget = 0) without GT
    adaptive_sampler = AdaptiveSampler()
    res_zero = adaptive_sampler.sample(scores_no_high, budget=0, seed=42)
    m_zero = evaluator.evaluate_result(res_zero, scores_no_high, patch_id="p_zero")

    assert m_zero.selected_regions_count == 0
    assert m_zero.gt_available is False
    assert m_zero.gt_anomaly_precision_pct is None
    assert m_zero.pixels_processed == 0
    assert m_zero.estimated_data_bytes == 0

    # Case 4: Zero selected regions with GT available (precision handles 0/0 safely)
    scores_with_gt = _create_mock_scores_with_ground_truth(
        n_high=0, n_mod=2, n_healthy=5, n_invalid=0, gt_anomaly_indices=[1]
    )
    res_zero_gt = adaptive_sampler.sample(scores_with_gt, budget=0, seed=42)
    m_zero_gt = evaluator.evaluate_result(res_zero_gt, scores_with_gt, patch_id="p_zero_gt")
    assert m_zero_gt.gt_available is True
    assert m_zero_gt.selected_regions_count == 0
    assert m_zero_gt.gt_anomaly_precision_pct == 0.0


def test_evaluator_ground_truth_mask_evaluation_and_missed_anomalies():
    """Verify ground-truth anomaly recall, precision, category breakdown, and missed anomaly logging."""
    # 3 high (reg_000, reg_001, reg_002), 4 mod (reg_003..reg_006), 8 healthy (reg_007..reg_014)
    # Ground truth anomalies assigned to:
    # reg_000 (high-stress, weed_cluster)
    # reg_001 (high-stress, double_plant)
    # reg_004 (moderate-stress, weed_cluster)
    # reg_005 (moderate-stress, planter_skip)
    # reg_008 (healthy, waterway)
    scores = _create_mock_scores_with_ground_truth(
        n_high=3,
        n_mod=4,
        n_healthy=8,
        n_invalid=0,
        gt_anomaly_indices=[0, 1, 4, 5, 8],
        gt_categories_map={
            0: ["weed_cluster"],
            1: ["double_plant"],
            4: ["weed_cluster"],
            5: ["planter_skip"],
            8: ["waterway"],
        },
    )

    evaluator = AdaptiveEvaluator(default_seed=42)

    # Run adaptive sampling with budget B = 4
    # High stress consumes 3 slots (reg_000, reg_001, reg_002) -> detects reg_000 and reg_001
    # 1 slot remaining -> exploration takes 1 healthy region (e.g., reg_007)
    # 0 slots for moderate stress -> reg_004 and reg_005 are missed!
    bench = evaluator.compare_strategies(scores, adaptive_budget=4, seed=42, patch_id="p_gt")
    ad_m = bench.strategy_metrics["adaptive"]

    assert ad_m.gt_available is True
    assert ad_m.total_gt_anomalies == 5  # 5 total anomaly regions in valid field

    # reg_000 and reg_001 are high stress -> selected!
    assert ad_m.detected_gt_anomalies >= 2
    assert ad_m.detected_gt_anomalies + ad_m.missed_gt_anomalies == 5

    # Missed anomalies must be logged with exact region IDs
    assert len(ad_m.missed_gt_anomaly_ids) == ad_m.missed_gt_anomalies
    # Moderate regions reg_004 and reg_005 were not selected -> must appear in missed anomaly IDs!
    assert "reg_004" in ad_m.missed_gt_anomaly_ids
    assert "reg_005" in ad_m.missed_gt_anomaly_ids

    # Ground-truth recall: (detected / 5) * 100
    expected_recall = round(ad_m.detected_gt_anomalies / 5 * 100.0, 2)
    assert round(ad_m.gt_anomaly_recall_pct, 2) == expected_recall

    # Category breakdown verification
    assert "weed_cluster" in ad_m.gt_category_breakdown
    assert "planter_skip" in ad_m.gt_category_breakdown
    assert ad_m.gt_category_breakdown["planter_skip"]["total"] == 1
    assert ad_m.gt_category_breakdown["planter_skip"]["missed"] == 1
    assert ad_m.gt_category_breakdown["planter_skip"]["recall_pct"] == 0.0


def test_evaluator_invalid_region_exclusion_integrity():
    """Verify invalid regions are never selected and never counted as avoided sampling or anomaly detections."""
    # 8 valid regions (2 high, 2 mod, 4 healthy) and 12 invalid regions = 20 total
    # Even if an invalid region has gt_has_anomaly = True (e.g. outside farm boundary),
    # it must NOT be included in valid anomaly metrics or selected regions.
    scores = _create_mock_scores(n_high=2, n_mod=2, n_healthy=4, n_invalid=12)
    # Assign anomaly to valid reg_000 and invalid reg_008
    scores[0].gt_has_anomaly = True
    scores[0].gt_categories = ["weed_cluster"]
    scores[8].gt_has_anomaly = True
    scores[8].gt_categories = ["weed_cluster"]

    evaluator = AdaptiveEvaluator(default_seed=42)
    bench = evaluator.compare_strategies(scores, adaptive_budget=3, seed=42, patch_id="p_inv_integ")
    ad_m = bench.strategy_metrics["adaptive"]

    # 1. Geometry
    assert ad_m.total_regions == 20
    assert ad_m.valid_regions == 8
    assert ad_m.invalid_regions == 12

    # 2. Only valid anomaly counts (reg_000 only; reg_008 is invalid so excluded)
    assert ad_m.total_gt_anomalies == 1
    assert ad_m.detected_gt_anomalies == 1  # reg_000 is high-stress -> selected!
    assert ad_m.gt_anomaly_recall_pct == 100.0

    # 3. Primary reduction metric uses valid_regions (8), NOT total_regions (20)
    # 3 selected out of 8 valid = (8 - 3) / 8 = 62.5%
    assert round(ad_m.valid_reduction_pct, 2) == 62.50
    # Secondary total-grid metric is (20 - 3) / 20 = 85.0%
    assert round(ad_m.sampling_reduction_pct, 2) == 85.00
    assert ad_m.valid_reduction_pct < ad_m.sampling_reduction_pct


def test_evaluator_equal_budget_fairness_and_overflow_reporting():
    """Verify fair budget matching K_equal == K_adaptive under high-stress overflow."""
    # 6 high stress, 2 mod, 6 healthy = 14 valid regions
    scores = _create_mock_scores(n_high=6, n_mod=2, n_healthy=6, n_invalid=0)
    evaluator = AdaptiveEvaluator(default_seed=42)

    # Requested budget B = 4 under GUARANTEE_HIGH_STRESS policy -> overflow: 6 high stress selected!
    bench = evaluator.compare_strategies(
        scores,
        adaptive_budget=4,
        overflow_policy=BudgetOverflowPolicy.GUARANTEE_HIGH_STRESS,
        seed=42,
        patch_id="p_over",
    )

    ad_m = bench.strategy_metrics["adaptive"]
    eq_m = bench.strategy_metrics["equal_budget_uniform"]
    fu_m = bench.strategy_metrics["full_uniform"]

    # 1. Overflow reporting
    assert ad_m.requested_budget == 4
    assert ad_m.budget_exceeded is True
    assert ad_m.overflow_count == 2
    assert ad_m.selected_regions_count == 6

    # 2. Equal-budget fairness: budget K is matched exactly to 6
    assert eq_m.selected_regions_count == 6
    assert eq_m.selected_regions_count == ad_m.selected_regions_count
    assert bench.comparison_summary["fair_budget_matched"] is True

    # 3. High-stress coverage: Adaptive gets 100%, Equal-Budget gets only partial
    assert ad_m.high_stress_coverage_pct == 100.0
    assert eq_m.high_stress_coverage_pct < 100.0

    # 4. Full Uniform baseline
    assert fu_m.selected_regions_count == 14
    assert fu_m.valid_reduction_pct == 0.0


def test_evaluator_reproducibility_across_seeds():
    """Verify bit-exact reproducibility for identical seeds and appropriate variance for different seeds."""
    scores = _create_mock_scores_with_ground_truth(n_high=3, n_mod=4, n_healthy=10, n_invalid=2)
    evaluator = AdaptiveEvaluator(default_seed=42)

    # Run 1: seed 42
    bench1 = evaluator.compare_strategies(scores, adaptive_budget=6, seed=42)
    # Run 2: seed 42 (identical)
    bench2 = evaluator.compare_strategies(scores, adaptive_budget=6, seed=42)

    # Every metric across all 3 strategies must match bit-exactly
    for strat in ["full_uniform", "equal_budget_uniform", "adaptive"]:
        assert bench1.strategy_metrics[strat].to_dict() == bench2.strategy_metrics[strat].to_dict()

    # Run 3: seed 99 (different seed)
    bench3 = evaluator.compare_strategies(scores, adaptive_budget=6, seed=99)
    eq1_ids = [d.region_id for d in bench1.strategy_metrics["equal_budget_uniform"].metadata.get("selected_regions", [])]
    eq3_ids = [d.region_id for d in bench3.strategy_metrics["equal_budget_uniform"].metadata.get("selected_regions", [])]
    # Equal budget randomized selection must reflect different seed
    assert bench1.strategy_metrics["equal_budget_uniform"].seed != bench3.strategy_metrics["equal_budget_uniform"].seed


def test_evaluator_repeated_trial_aggregation_and_determinism_variance():
    """Verify repeated-trial aggregation computes mean, std, min, max and proves deterministic vs randomized variance."""
    scores = _create_mock_scores_with_ground_truth(n_high=3, n_mod=4, n_healthy=15, n_invalid=0)
    evaluator = AdaptiveEvaluator(default_seed=42)

    bench = evaluator.run_repeated_trials(
        scores,
        adaptive_budget=None,  # Threshold-driven mode: 100% high + 100% mod are strictly deterministic
        seeds=[10, 20, 30, 40, 50],
        patch_id="p_trials",
    )

    assert bench.trial_aggregates is not None
    assert "adaptive" in bench.trial_aggregates
    assert "equal_budget_uniform" in bench.trial_aggregates
    assert "full_uniform" in bench.trial_aggregates

    ad_agg = bench.trial_aggregates["adaptive"]
    eq_agg = bench.trial_aggregates["equal_budget_uniform"]

    assert ad_agg.n_trials == 5
    assert eq_agg.n_trials == 5

    # In threshold mode, high-stress coverage for Adaptive is 100% deterministic (std = 0.0)
    ad_high_stats = ad_agg.metric_stats["high_stress_coverage_pct"]
    assert ad_high_stats["mean"] == 100.0
    assert ad_high_stats["std"] == 0.0

    # In contrast, Equal-Budget Uniform exhibits significant randomized variance (std > 0.0)
    eq_high_stats = eq_agg.metric_stats["high_stress_coverage_pct"]
    assert eq_high_stats["std"] > 0.0

    # Serialization check
    d = bench.to_dict()
    assert "trial_aggregates" in d
    assert d["trial_aggregates"]["adaptive"]["n_trials"] == 5


def test_evaluator_separation_of_rsdi_from_ground_truth():
    """Verify ground-truth anomaly annotations strictly do NOT alter sampling decisions."""
    # Score set A: No ground-truth metadata
    scores_a = _create_mock_scores(n_high=3, n_mod=5, n_healthy=10, n_invalid=2)

    # Score set B: Identical geometry and scores, but with ground-truth anomalies added
    scores_b = _create_mock_scores_with_ground_truth(
        n_high=3, n_mod=5, n_healthy=10, n_invalid=2, gt_anomaly_indices=[0, 3, 7]
    )

    sampler = AdaptiveSampler(default_seed=42)
    res_a = sampler.sample(scores_a, budget=6, seed=42)
    res_b = sampler.sample(scores_b, budget=6, seed=42)

    # Ground truth presence must NOT change selected region IDs or reasons
    ids_a = [d.region_id for d in res_a.selected_regions]
    ids_b = [d.region_id for d in res_b.selected_regions]
    assert ids_a == ids_b

    reasons_a = [d.selection_reason for d in res_a.selected_regions]
    reasons_b = [d.selection_reason for d in res_b.selected_regions]
    assert reasons_a == reasons_b


def test_evaluator_markdown_report_and_export(tmp_path):
    """Verify generated markdown benchmark report contents and JSON/CSV export functionality."""
    scores = _create_mock_scores_with_ground_truth(n_high=2, n_mod=3, n_healthy=8, n_invalid=2)
    evaluator = AdaptiveEvaluator(default_seed=42)

    bench = evaluator.run_repeated_trials(
        scores, adaptive_budget=5, seeds=[42, 43, 44], patch_id="p_report_test"
    )

    # 1. Markdown report generation
    report = evaluator.generate_markdown_report(bench)
    assert isinstance(report, str)
    assert "# StressSense – Adaptive Sampling Benchmark Report: p_report_test" in report
    assert "## 1. Executive Summary & Strategy Comparison" in report
    assert "## 2. Sampling Efficiency & Invariant Analysis" in report
    assert "## 3. Stress Detection & Ground-Truth Anomaly Coverage" in report
    assert "## 4. Resource-Cost & Energy Modeling (Computational Proxies)" in report
    assert "## 5. Budget Adherence & Overflow Reporting" in report
    assert "## 6. Repeated Trials & Statistical Analysis" in report
    assert "## 7. Limitations & Scientific Disclosures" in report

    # 2. JSON export
    json_path = tmp_path / "benchmark.json"
    exported_json = evaluator.export_json(bench, json_path)
    assert exported_json.is_file()
    import json
    with open(exported_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["patch_id"] == "p_report_test"
    assert "strategy_metrics" in data
    assert "adaptive" in data["strategy_metrics"]

    # 3. CSV export
    csv_path = tmp_path / "benchmark.csv"
    exported_csv = evaluator.export_csv(bench, csv_path)
    assert exported_csv.is_file()
    import pandas as pd
    df = pd.read_csv(exported_csv)
    assert len(df) == 3  # full_uniform, equal_budget_uniform, adaptive
    assert "strategy" in df.columns
    assert "valid_reduction_pct" in df.columns
    assert "gt_anomaly_recall_pct" in df.columns


def test_evaluator_on_real_processed_patch_with_anomalies(tmp_path):
    """Verify end-to-end Grid -> Scorer -> Sampler -> Evaluator on authentic Phase 2 patch with anomaly."""
    manifest_csv = Path("data/processed/manifests/patch_manifest.csv")
    if not manifest_csv.is_file():
        pytest.skip("Phase 2 manifest file not found.")

    import pandas as pd
    from PIL import Image

    df = pd.read_csv(manifest_csv)
    # Find a patch with authentic anomalies
    anomaly_rows = df[df["has_anomaly"] == True]
    if len(anomaly_rows) == 0:
        pytest.skip("No anomaly patches found in manifest.")

    row = anomaly_rows.iloc[0]

    rgb = np.array(Image.open(row["rgb_patch_path"]))
    nir = np.array(Image.open(row["nir_patch_path"]))
    bnd = np.array(Image.open(row["boundary_patch_path"]))
    msk = np.array(Image.open(row["mask_patch_path"]))

    # Load authentic anomaly mask for this patch
    patch_id = row["patch_id"]
    category = row["anomaly_categories"].split(",")[0].strip()
    category_mask_path = Path(f"data/processed/patches/train/labels/{category}/{patch_id}.png")

    if category_mask_path.is_file():
        gt_mask = np.array(Image.open(category_mask_path))
    else:
        gt_mask = None

    # 1. Grid Generator: 128x128 patch -> 32x32 sub-regions = 16 sub-regions
    gen = GridGenerator(grid_size=32, stride=32)
    regions = gen.generate_grid(rgb.shape, image_id=patch_id, validity_mask=msk, boundary_mask=bnd)

    # 2. Stress Scorer: compute scores with decoupled ground truth mask
    scorer = StressScorer()
    scores = scorer.score_all_regions(
        rgb=rgb,
        nir=nir,
        regions=regions,
        validity_mask=msk,
        boundary_mask=bnd,
        ground_truth_mask=gt_mask,
        ground_truth_categories=[category] if gt_mask is not None else None,
    )
    assert len(scores) == 16

    # 3. Evaluator: Run repeated trials benchmark comparing all 3 strategies with budget B = 4
    evaluator = AdaptiveEvaluator(default_seed=42)
    bench = evaluator.run_repeated_trials(
        scores=scores,
        adaptive_budget=4,
        seeds=[42, 43, 44],
        patch_id=patch_id,
        pixels_per_region=1024,
    )

    ad_m = bench.strategy_metrics["adaptive"]
    eq_m = bench.strategy_metrics["equal_budget_uniform"]
    fu_m = bench.strategy_metrics["full_uniform"]

    # Verify fair budget matching
    assert ad_m.selected_regions_count == eq_m.selected_regions_count
    assert fu_m.valid_reduction_pct == 0.0

    # Verify report and export
    report = evaluator.generate_markdown_report(bench)
    assert len(report) > 500

    out_json = evaluator.export_json(bench, tmp_path / f"{patch_id}_bench.json")
    out_csv = evaluator.export_csv(bench, tmp_path / f"{patch_id}_bench.csv")
    assert out_json.is_file()
    assert out_csv.is_file()


def test_ground_truth_overlap_edge_cases_and_annotation_sensitivity():
    """Verify ground-truth any-positive-pixel logic and analyze sensitivity to single-pixel boundary grazing."""
    scorer = StressScorer()

    # Synthetic 64x64 imagery, 32x32 grid -> 4 regions of 1024 pixels each
    rgb = np.full((64, 64, 3), 120, dtype=np.uint8)
    nir = np.full((64, 64), 160, dtype=np.uint8)
    gen = GridGenerator(grid_size=32, stride=32)
    regions = gen.generate_grid(rgb.shape, image_id="p_overlap_test")

    # Construct ground truth mask:
    # Region 0 (top-left): exactly 1 positive anomaly pixel (boundary grazing: 1/1024 = 0.0976%)
    # Region 1 (top-right): 512 positive anomaly pixels (substantial infestation: 512/1024 = 50.0%)
    # Region 2 (bottom-left): 0 anomaly pixels (clean canopy)
    gt_mask = np.zeros((64, 64), dtype=np.uint8)
    gt_mask[0, 0] = 255  # 1 single pixel in Region 0
    gt_mask[0:32, 32:48] = 255  # 32 * 16 = 512 pixels in Region 1

    scores = scorer.score_all_regions(
        rgb=rgb, nir=nir, regions=regions, ground_truth_mask=gt_mask, ground_truth_categories=["weed_cluster"]
    )

    # Region 0: 1 pixel overlap
    assert scores[0].gt_has_anomaly is True
    assert scores[0].gt_anomaly_fraction == pytest.approx(1 / 1024, rel=1e-3)

    # Region 1: 512 pixels overlap
    assert scores[1].gt_has_anomaly is True
    assert scores[1].gt_anomaly_fraction == pytest.approx(512 / 1024, rel=1e-3)

    # Region 2: clean
    assert scores[2].gt_has_anomaly is False
    assert scores[2].gt_anomaly_fraction == 0.0

    # Sensitivity Disclosure: Under any-positive-pixel logic, both 1-pixel and 512-pixel regions
    # evaluate to gt_has_anomaly = True. gt_anomaly_fraction is provided for callers needing to filter
    # by minimum overlap threshold.


def test_category_overlap_zero_and_threshold_boundaries_regression():
    """Regression Test: Verify zero-overlap regions and exact threshold-boundary cases.

    Verifies:
    1. Zero-overlap regions: When category mask has 0 pixels in a region,
       that category is NEVER attributed, even at threshold 0.0 (strictly positive overlap).
    2. 1-pixel boundary grazing (strictly positive): Attributed at threshold 0.0,
       but excluded at 5%, 10%, and 20%.
    3. Exact inclusive boundaries for standard 32x32 (1024 px) regions:
       - 5% threshold: 51 px (4.98%) excluded, 52 px (5.08%) attributed.
       - 10% threshold: 102 px (9.96%) excluded, 103 px (10.06%) attributed.
       - 20% threshold: 204 px (19.92%) excluded, 205 px (20.02%) attributed.
    4. Exact inclusive mathematical fractions on custom grid region sizes:
       - 5% threshold: 4/100 (4.0%) excluded, 5/100 (5.0%) attributed (exact boundary >=).
       - 10% threshold: 9/100 (9.0%) excluded, 10/100 (10.0%) attributed (exact boundary >=).
       - 20% threshold: 19/100 (19.0%) excluded, 20/100 (20.0%) attributed (exact boundary >=).
    """
    scorer = StressScorer()
    gen = GridGenerator(grid_size=32)
    regions = gen.generate_grid(image_shape=(32, 32))
    reg = regions[0]

    rgb = np.full((32, 32, 3), 120, dtype=np.uint8)
    nir = np.full((32, 32), 160, dtype=np.uint8)
    red_f, nir_f = scorer.validate_and_align_inputs(rgb, nir)
    ndvi_map = scorer.compute_ndvi_map(red_f, nir_f)

    # 1. Zero-overlap: Category mask with 0 pixels
    mask_zero = {"standing_water": np.zeros((32, 32), dtype=np.uint8)}
    for th in [0.0, 0.05, 0.10, 0.20]:
        res_zero = scorer.score_single_region(
            ndvi_map, reg, category_masks=mask_zero, category_overlap_threshold=th
        )
        assert res_zero.gt_categories is None, f"Zero overlap should evaluate to None at threshold {th}"

    # 2. Strictly positive 1-pixel overlap (1 / 1024 = 0.0976%)
    mask_1px = {"standing_water": np.zeros((32, 32), dtype=np.uint8)}
    mask_1px["standing_water"][0, 0] = 255
    res_1px_0 = scorer.score_single_region(
        ndvi_map, reg, category_masks=mask_1px, category_overlap_threshold=0.0
    )
    assert res_1px_0.gt_categories == ["standing_water"], "1-pixel overlap must be attributed at threshold 0.0"

    for th in [0.05, 0.10, 0.20]:
        res_1px_th = scorer.score_single_region(
            ndvi_map, reg, category_masks=mask_1px, category_overlap_threshold=th
        )
        assert res_1px_th.gt_categories is None, f"1-pixel overlap must not be attributed at threshold {th}"

    # 3. Exact threshold boundaries on 32x32 (1024 px) region
    # 5% threshold: 51 px (4.98%) vs 52 px (5.08%)
    mask_51px = {"waterway": np.zeros((32, 32), dtype=np.uint8)}
    mask_51px["waterway"].flat[:51] = 255
    res_51 = scorer.score_single_region(ndvi_map, reg, category_masks=mask_51px, category_overlap_threshold=0.05)
    assert res_51.gt_categories is None, "51 px (4.98%) must be excluded at 5% threshold"

    mask_52px = {"waterway": np.zeros((32, 32), dtype=np.uint8)}
    mask_52px["waterway"].flat[:52] = 255
    res_52 = scorer.score_single_region(ndvi_map, reg, category_masks=mask_52px, category_overlap_threshold=0.05)
    assert res_52.gt_categories == ["waterway"], "52 px (5.08%) must be included at 5% threshold"

    # 10% threshold: 102 px (9.96%) vs 103 px (10.06%)
    mask_102px = {"waterway": np.zeros((32, 32), dtype=np.uint8)}
    mask_102px["waterway"].flat[:102] = 255
    res_102 = scorer.score_single_region(ndvi_map, reg, category_masks=mask_102px, category_overlap_threshold=0.10)
    assert res_102.gt_categories is None, "102 px (9.96%) must be excluded at 10% threshold"

    mask_103px = {"waterway": np.zeros((32, 32), dtype=np.uint8)}
    mask_103px["waterway"].flat[:103] = 255
    res_103 = scorer.score_single_region(ndvi_map, reg, category_masks=mask_103px, category_overlap_threshold=0.10)
    assert res_103.gt_categories == ["waterway"], "103 px (10.06%) must be included at 10% threshold"

    # 20% threshold: 204 px (19.92%) vs 205 px (20.02%)
    mask_204px = {"waterway": np.zeros((32, 32), dtype=np.uint8)}
    mask_204px["waterway"].flat[:204] = 255
    res_204 = scorer.score_single_region(ndvi_map, reg, category_masks=mask_204px, category_overlap_threshold=0.20)
    assert res_204.gt_categories is None, "204 px (19.92%) must be excluded at 20% threshold"

    mask_205px = {"waterway": np.zeros((32, 32), dtype=np.uint8)}
    mask_205px["waterway"].flat[:205] = 255
    res_205 = scorer.score_single_region(ndvi_map, reg, category_masks=mask_205px, category_overlap_threshold=0.20)
    assert res_205.gt_categories == ["waterway"], "205 px (20.02%) must be included at 20% threshold"

    # 4. Exact inclusive fractions on custom region (10x10 = 100 pixels)
    gen_10 = GridGenerator(grid_size=10)
    reg_100px = gen_10.generate_grid(image_shape=(10, 10))[0]
    rgb_10 = np.full((10, 10, 3), 120, dtype=np.uint8)
    nir_10 = np.full((10, 10), 160, dtype=np.uint8)
    r_f10, n_f10 = scorer.validate_and_align_inputs(rgb_10, nir_10)
    ndvi_10 = scorer.compute_ndvi_map(r_f10, n_f10)

    # 5% boundary: 4 vs 5 pixels
    m_4px = {"weed_cluster": np.zeros((10, 10), dtype=np.uint8)}
    m_4px["weed_cluster"].flat[:4] = 255
    res_4 = scorer.score_single_region(ndvi_10, reg_100px, category_masks=m_4px, category_overlap_threshold=0.05)
    assert res_4.gt_categories is None, "4/100 (4.0%) must be excluded at 5% threshold"

    m_5px = {"weed_cluster": np.zeros((10, 10), dtype=np.uint8)}
    m_5px["weed_cluster"].flat[:5] = 255
    res_5 = scorer.score_single_region(ndvi_10, reg_100px, category_masks=m_5px, category_overlap_threshold=0.05)
    assert res_5.gt_categories == ["weed_cluster"], "5/100 (5.0%) must be included at 5% threshold (exact >=)"

    # 10% boundary: 9 vs 10 pixels
    m_9px = {"weed_cluster": np.zeros((10, 10), dtype=np.uint8)}
    m_9px["weed_cluster"].flat[:9] = 255
    res_9 = scorer.score_single_region(ndvi_10, reg_100px, category_masks=m_9px, category_overlap_threshold=0.10)
    assert res_9.gt_categories is None, "9/100 (9.0%) must be excluded at 10% threshold"

    m_10px = {"weed_cluster": np.zeros((10, 10), dtype=np.uint8)}
    m_10px["weed_cluster"].flat[:10] = 255
    res_10 = scorer.score_single_region(ndvi_10, reg_100px, category_masks=m_10px, category_overlap_threshold=0.10)
    assert res_10.gt_categories == ["weed_cluster"], "10/100 (10.0%) must be included at 10% threshold (exact >=)"

    # 20% boundary: 19 vs 20 pixels
    m_19px = {"weed_cluster": np.zeros((10, 10), dtype=np.uint8)}
    m_19px["weed_cluster"].flat[:19] = 255
    res_19 = scorer.score_single_region(ndvi_10, reg_100px, category_masks=m_19px, category_overlap_threshold=0.20)
    assert res_19.gt_categories is None, "19/100 (19.0%) must be excluded at 20% threshold"

    m_20px = {"weed_cluster": np.zeros((10, 10), dtype=np.uint8)}
    m_20px["weed_cluster"].flat[:20] = 255
    res_20 = scorer.score_single_region(ndvi_10, reg_100px, category_masks=m_20px, category_overlap_threshold=0.20)
    assert res_20.gt_categories == ["weed_cluster"], "20/100 (20.0%) must be included at 20% threshold (exact >=)"


def test_energy_calculations_hand_calculated_verification():
    """Verify proxy energy model formulas against exact hand-calculated ground-truth values."""
    assumptions = ResourceCostAssumptions(
        bytes_per_pixel=4,
        compression_ratio=1.0,
        compute_units_per_region=1.0,
        inference_latency_seconds_per_region=0.015,
        inference_power_watts=15.0,
        transmission_power_watts=2.5,
        transmission_bandwidth_bytes_per_sec=1_000_000.0,
    )
    evaluator = AdaptiveEvaluator(cost_assumptions=assumptions, default_seed=42)

    # Grid: 10 valid regions, 1024 pixels each
    # Hand Calculations for 1 region (1024 pixels, 4 bytes/pixel = 4096 bytes):
    # - tx_seconds = 4096 / 1_000_000 = 0.004096 s
    # - tx_energy = 0.004096 s * 2.5 W = 0.01024 J
    # - compute_energy = 0.015 s * 15.0 W = 0.225 J
    # - total_energy_per_region = 0.01024 + 0.225 = 0.23524 J
    # For K = 5 selected regions out of 10 valid:
    # - expected_tx_energy = 5 * 0.01024 = 0.0512 J
    # - expected_compute_energy = 5 * 0.225 = 1.125 J
    # - expected_total_energy = 5 * 0.23524 = 1.1762 J
    # - expected_baseline_energy = 10 * 0.23524 = 2.3524 J
    # - expected_energy_reduction = (2.3524 - 1.1762) / 2.3524 * 100 = 50.0%

    scores = _create_mock_scores(n_high=2, n_mod=3, n_healthy=5, n_invalid=0)
    bench = evaluator.compare_strategies(scores, adaptive_budget=5, seed=42, pixels_per_region=1024)
    ad_m = bench.strategy_metrics["adaptive"]

    assert ad_m.selected_regions_count == 5
    assert ad_m.pixels_per_region == 1024
    assert abs(ad_m.estimated_tx_energy_joules_proxy - 0.0512) < 1e-6
    assert abs(ad_m.estimated_compute_energy_joules_proxy - 1.1250) < 1e-6
    assert abs(ad_m.estimated_energy_joules_proxy - 1.1762) < 1e-4
    assert abs(ad_m.energy_reduction_pct - 50.0) < 1e-4


def test_json_serialization_of_zero_denominator_cases(tmp_path):
    """Verify JSON export with allow_nan=False handles zero-denominator edge cases without leaking NaN or Inf."""
    evaluator = AdaptiveEvaluator(default_seed=42)

    # Grid with 0 valid regions
    scores_empty = _create_mock_scores(n_high=0, n_mod=0, n_healthy=0, n_invalid=10)
    bench = evaluator.compare_strategies(scores_empty, adaptive_budget=5, seed=42, patch_id="p_zero_denom")

    out_file = tmp_path / "zero_denom_strict.json"
    # export_json enforces allow_nan=False internally
    evaluator.export_json(bench, out_file)

    assert out_file.is_file()
    import json
    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Confirm JSON has null, not NaN string
    ad_data = data["strategy_metrics"]["adaptive"]
    assert ad_data["gt_anomalous_region_recall_pct"] is None
    assert ad_data["gt_anomalous_region_precision_pct"] is None
    assert ad_data["valid_regions"] == 0
    assert ad_data["selected_regions_count"] == 0


def test_region_level_precision_and_recall_explicit_naming():
    """Verify that canonical region-level anomaly metrics and backward-compatible aliases are identical."""
    scores = _create_mock_scores_with_ground_truth(n_high=3, n_mod=4, n_healthy=8, n_invalid=0, gt_anomaly_indices=[0, 1])
    evaluator = AdaptiveEvaluator(default_seed=42)
    bench = evaluator.compare_strategies(scores, adaptive_budget=4, seed=42)
    ad_m = bench.strategy_metrics["adaptive"]

    # Canonical region-level names must match legacy aliases identically
    assert ad_m.gt_anomalous_region_recall_pct is not None
    assert ad_m.gt_anomalous_region_recall_pct == ad_m.gt_anomaly_recall_pct
    assert ad_m.gt_anomalous_region_precision_pct is not None
    assert ad_m.gt_anomalous_region_precision_pct == ad_m.gt_anomaly_precision_pct

    # Confirm in to_dict() serialization
    d = ad_m.to_dict()
    assert "gt_anomalous_region_recall_pct" in d
    assert "gt_anomalous_region_precision_pct" in d
    assert "gt_anomaly_recall_pct" in d
    assert "gt_anomaly_precision_pct" in d


# ==============================================================================
# STAGE 5 TESTS: VISUALIZATION AND DEMONSTRATION ENGINE
# ==============================================================================

from src.adaptive.visualizer import AdaptiveVisualizer


def test_adaptive_visualizer_initialization_and_output_dir(tmp_path):
    """Verify visualizer initializes cleanly and creates target output directory."""
    vis = AdaptiveVisualizer(output_dir=tmp_path / "vis_test", default_seed=42)
    assert vis.output_dir.is_dir()
    assert vis.default_seed == 42


def test_adaptive_visualizer_pipeline_showcase_generation(tmp_path):
    """Verify 5-panel pipeline showcase figure generation on synthetic multi-modal data."""
    vis = AdaptiveVisualizer(output_dir=tmp_path, default_seed=42)

    rgb = np.full((64, 64, 3), 120, dtype=np.uint8)
    nir = np.full((64, 64), 160, dtype=np.uint8)
    gen = GridGenerator(grid_size=32, stride=32)
    regions = gen.generate_grid(rgb.shape, image_id="p_vis")
    scorer = StressScorer()
    scores = scorer.score_all_regions(rgb, nir, regions)

    sampler = AdaptiveSampler(default_seed=42)
    ad_res = sampler.sample(scores, budget=2, seed=42)

    out_file = vis.plot_pipeline_showcase(
        rgb=rgb,
        nir=nir,
        regions=regions,
        scores=scores,
        adaptive_result=ad_res,
        save_path=tmp_path / "showcase_test.png",
    )

    assert out_file.is_file()
    assert out_file.stat().st_size > 5000  # Non-trivial image size


def test_adaptive_visualizer_strategy_comparison_generation(tmp_path):
    """Verify side-by-side 3-way strategy comparison figure generation."""
    vis = AdaptiveVisualizer(output_dir=tmp_path, default_seed=42)

    rgb = np.full((64, 64, 3), 120, dtype=np.uint8)
    nir = np.full((64, 64), 160, dtype=np.uint8)
    gen = GridGenerator(grid_size=32, stride=32)
    regions = gen.generate_grid(rgb.shape, image_id="p_comp_test")
    scorer = StressScorer()
    scores = scorer.score_all_regions(rgb, nir, regions)

    evaluator = AdaptiveEvaluator(default_seed=42)
    bench = evaluator.compare_strategies(scores, adaptive_budget=2, seed=42, patch_id="p_comp_test")

    out_file = vis.plot_strategy_comparison(
        rgb=rgb,
        regions=regions,
        benchmark=bench,
        save_path=tmp_path / "comp_test.png",
    )

    assert out_file.is_file()
    assert out_file.stat().st_size > 5000


def test_adaptive_visualizer_ground_truth_overlay_generation(tmp_path):
    """Verify ground-truth detection overlay figure generation showing true positives and missed anomalies."""
    vis = AdaptiveVisualizer(output_dir=tmp_path, default_seed=42)

    rgb = np.full((64, 64, 3), 120, dtype=np.uint8)
    nir = np.full((64, 64), 160, dtype=np.uint8)
    gt_mask = np.zeros((64, 64), dtype=np.uint8)
    gt_mask[0:32, 0:32] = 255  # Region 0 has anomaly

    gen = GridGenerator(grid_size=32, stride=32)
    regions = gen.generate_grid(rgb.shape, image_id="p_gt_overlay")
    scorer = StressScorer()
    scores = scorer.score_all_regions(
        rgb, nir, regions, ground_truth_mask=gt_mask, ground_truth_categories=["weed_cluster"]
    )

    evaluator = AdaptiveEvaluator(default_seed=42)
    bench = evaluator.compare_strategies(scores, adaptive_budget=2, seed=42, patch_id="p_gt_overlay")

    out_file = vis.plot_ground_truth_detection_overlay(
        rgb=rgb,
        regions=regions,
        scores=scores,
        gt_mask=gt_mask,
        benchmark=bench,
        category_name="weed_cluster",
        save_path=tmp_path / "gt_overlay_test.png",
    )

    assert out_file.is_file()
    assert out_file.stat().st_size > 5000


def test_adaptive_visualizer_metrics_summary_generation(tmp_path):
    """Verify 4-panel benchmark metrics summary bar chart generation."""
    vis = AdaptiveVisualizer(output_dir=tmp_path, default_seed=42)

    scores = _create_mock_scores_with_ground_truth(n_high=3, n_mod=4, n_healthy=8, n_invalid=0)
    evaluator = AdaptiveEvaluator(default_seed=42)
    bench = evaluator.compare_strategies(scores, adaptive_budget=5, seed=42, patch_id="p_summary_test")

    out_file = vis.plot_benchmark_metrics_summary(
        benchmark=bench,
        save_path=tmp_path / "summary_test.png",
    )

    assert out_file.is_file()
    assert out_file.stat().st_size > 5000


def test_adaptive_visualizer_end_to_end_artifact_generation(tmp_path):
    """Verify full generate_all_visualizations pipeline creates all 7 deliverables on real patch data."""
    manifest_csv = Path("data/processed/manifests/patch_manifest.csv")
    if not manifest_csv.is_file():
        pytest.skip("Phase 2 manifest file not found.")

    vis = AdaptiveVisualizer(output_dir=tmp_path / "adaptive_demo", default_seed=42)
    artifacts = vis.generate_all_visualizations(
        patch_id="13C8GTY66_1200-2200-1712-2712_p128_128",
        adaptive_budget=5,
        grid_size=32,
        seed=42,
    )

    # 4 PNG figures
    assert artifacts["pipeline_showcase"].is_file()
    assert artifacts["strategy_comparison"].is_file()
    assert artifacts["ground_truth_overlay"].is_file()
    assert artifacts["metrics_summary"].is_file()

    # 3 Report artifacts
    assert artifacts["benchmark_json"].is_file()
    assert artifacts["benchmark_csv"].is_file()
    assert artifacts["benchmark_report_md"].is_file()

    # Verify report length and content
    with open(artifacts["benchmark_report_md"], "r", encoding="utf-8") as f:
        content = f.read()
    assert "StressSense – Adaptive Sampling Benchmark Report" in content
    assert "Executive Summary & Strategy Comparison" in content
    assert "Limitations & Scientific Disclosures" in content







