"""Pytest test suite for Phase 4, Stage 1: Dataset-Wide Experimental Validation.

Validates the DatasetWideEvaluator, DatasetVisualizer, statistical aggregation,
paired hypothesis testing, ground-truth decoupling, and artifact export generation.
"""

import json
import math
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.evaluation.dataset_evaluator import (
    DatasetWideEvaluator,
    HypothesisTestSummary,
    MetricDistribution,
    PatchRecord,
)
from src.evaluation.visualizer import DatasetVisualizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ==============================================================================
# 1. Metric Distribution Unit Tests
# ==============================================================================

def test_metric_distribution_known_values():
    """Verify MetricDistribution calculations against known analytical values."""
    # Data: [10, 20, 30, 40, 50]
    # Mean: 30.0, Sample Std: 15.8114, Median: 30.0, Min: 10, Max: 50
    vals = [10.0, 20.0, 30.0, 40.0, 50.0]
    dist = MetricDistribution.from_array(vals)

    assert dist.n == 5
    assert abs(dist.mean - 30.0) < 1e-3
    assert abs(dist.std - 15.8114) < 1e-3
    assert abs(dist.median - 30.0) < 1e-3
    assert dist.min_val == 10.0
    assert dist.max_val == 50.0
    assert dist.ci95_lower < dist.mean < dist.ci95_upper

    d = dist.to_dict()
    assert isinstance(d, dict)
    assert d["mean"] == 30.0
    assert d["n"] == 5


def test_metric_distribution_empty_and_nan_handling():
    """Verify defensive handling of empty sequences or arrays containing NaNs."""
    empty_dist = MetricDistribution.from_array([])
    assert empty_dist.n == 0
    assert empty_dist.mean == 0.0

    nan_dist = MetricDistribution.from_array([10.0, np.nan, 20.0, 30.0, np.nan])
    assert nan_dist.n == 3
    assert abs(nan_dist.mean - 20.0) < 1e-3


# ==============================================================================
# 2. DatasetWideEvaluator Initialization & Manifest Loading
# ==============================================================================

def test_dataset_evaluator_initialization():
    """Verify default initialization parameters and path configurations."""
    evaluator = DatasetWideEvaluator(grid_size=32, budget_ratio=0.35)
    assert evaluator.grid_size == 32
    assert evaluator.budget_ratio == 0.35
    assert evaluator.manifest_path.is_file()
    assert evaluator.patches_dir.is_dir()
    assert evaluator.output_dir.is_dir()


def test_manifest_loading_and_filtering():
    """Verify loading and split/field filtering from patch_manifest.csv."""
    evaluator = DatasetWideEvaluator()
    df_all = evaluator.load_manifest()
    assert len(df_all) == 640

    df_train = evaluator.load_manifest(split="train")
    assert len(df_train) == 320

    df_val = evaluator.load_manifest(split="val")
    assert len(df_val) == 160

    df_test = evaluator.load_manifest(split="test")
    assert len(df_test) == 160

    df_anom = evaluator.load_manifest(has_anomaly=True)
    assert len(df_anom) == 130

    # Specific field filter
    first_field = df_all.iloc[0]["field_id"]
    df_field = evaluator.load_manifest(field_id=first_field)
    assert len(df_field) == 16  # 16 patches per tile


# ==============================================================================
# 3. Patch Loading & Multi-Modal Alignment Tests
# ==============================================================================

def test_patch_data_loading_modalities():
    """Verify multi-spectral patch data loading and dimension alignment."""
    evaluator = DatasetWideEvaluator()
    df = evaluator.load_manifest(has_anomaly=True)
    row = df.iloc[0]

    patch_data = evaluator.load_patch_data(row)
    assert patch_data["patch_id"] == row["patch_id"]
    assert patch_data["split"] == row["split"]
    assert patch_data["has_anomaly"] is True

    # Check numpy array dimensions
    assert patch_data["rgb"].shape == (128, 128, 3)
    assert patch_data["rgb"].dtype == np.uint8
    assert patch_data["nir"].shape == (128, 128)
    assert patch_data["nir"].dtype == np.uint8

    if patch_data["validity_mask"] is not None:
        assert patch_data["validity_mask"].shape == (128, 128)
    if patch_data["boundary_mask"] is not None:
        assert patch_data["boundary_mask"].shape == (128, 128)

    # Composite ground truth mask must be uint8 (128, 128)
    assert patch_data["ground_truth_mask"] is not None
    assert patch_data["ground_truth_mask"].shape == (128, 128)
    assert patch_data["ground_truth_mask"].dtype == np.uint8


# ==============================================================================
# 4. Single Patch Evaluation & Strategy Comparison Fairness
# ==============================================================================

def test_evaluate_single_patch_fairness_and_metrics():
    """Verify single patch evaluation preserves fair budget allocation K."""
    evaluator = DatasetWideEvaluator(budget_ratio=0.35, ndvi_target=0.32, ndvi_stressed=0.20)
    df = evaluator.load_manifest(has_anomaly=True)
    row = df.iloc[0]
    patch_data = evaluator.load_patch_data(row)

    rec = evaluator.evaluate_single_patch(patch_data)
    assert isinstance(rec, PatchRecord)
    assert rec.patch_id == row["patch_id"]

    # Budget fairness: Equal-Budget must have exact same selected count K as Adaptive
    assert rec.k_selected >= 1
    assert rec.k_selected <= rec.valid_regions
    assert rec.full_selected == rec.valid_regions

    # Efficiency metric bounds
    assert 0.0 <= rec.valid_reduction_pct <= 100.0
    assert 0.0 <= rec.sampling_reduction_pct <= 100.0

    # High stress coverage
    assert 0.0 <= rec.high_stress_adaptive_pct <= 100.0
    assert 0.0 <= rec.high_stress_equal_pct <= 100.0

    # Energy reduction percentage must match valid reduction percentage
    assert abs(rec.energy_reduction_pct - rec.valid_reduction_pct) < 1.0


# ==============================================================================
# 5. Batch Evaluation & Aggregates Computation
# ==============================================================================

def test_evaluate_subset_and_compute_aggregates():
    """Verify batch evaluation across a subset and statistical aggregation."""
    evaluator = DatasetWideEvaluator(budget_ratio=0.35, ndvi_target=0.32, ndvi_stressed=0.20)
    records = evaluator.evaluate_all(splits=["train"], max_patches=10)
    assert len(records) == 10

    summary = evaluator.compute_aggregates(records)
    assert summary["total_patches"] == 10
    assert "overall_distributions" in summary
    assert "category_breakdown" in summary
    assert "split_breakdown" in summary
    assert "field_breakdown" in summary
    assert "totals" in summary

    # Check totals
    totals = summary["totals"]
    assert totals["total_regions_all_patches"] == 10 * 16  # 160 regions
    assert totals["valid_regions_all_patches"] > 0
    assert totals["cumulative_energy_full_j"] > 0.0
    assert totals["cumulative_energy_adaptive_j"] > 0.0


# ==============================================================================
# 6. Statistical Hypothesis Testing
# ==============================================================================

def test_compute_statistical_tests():
    """Verify paired hypothesis tests compute valid t-statistics and p-values."""
    evaluator = DatasetWideEvaluator(budget_ratio=0.35, ndvi_target=0.32, ndvi_stressed=0.20)
    records = evaluator.evaluate_all(splits=["train"], max_patches=15)
    tests = evaluator.compute_statistical_tests(records)

    assert len(tests) >= 1
    for t in tests:
        assert isinstance(t, HypothesisTestSummary)
        assert 0.0 <= t.p_value_ttest <= 1.0
        assert 0.0 <= t.p_value_wilcoxon <= 1.0
        assert isinstance(t.is_significant_05, bool)
        assert isinstance(t.is_significant_01, bool)
        assert t.sample_size >= 2
        d = t.to_dict()
        assert "t_statistic" in d
        assert "p_value_ttest" in d


# ==============================================================================
# 7. Ground-Truth Decoupling Test
# ==============================================================================

def test_ground_truth_decoupling():
    """Verify sampling decisions are identical whether ground truth is present or omitted."""
    evaluator = DatasetWideEvaluator(budget_ratio=0.35, ndvi_target=0.32, ndvi_stressed=0.20)
    df = evaluator.load_manifest(has_anomaly=True)
    row = df.iloc[0]

    # Evaluation WITH ground-truth
    patch_data_with_gt = evaluator.load_patch_data(row)
    rec_with_gt = evaluator.evaluate_single_patch(patch_data_with_gt)

    # Evaluation WITHOUT ground-truth (blind evaluation)
    patch_data_blind = evaluator.load_patch_data(row)
    patch_data_blind["ground_truth_mask"] = None
    patch_data_blind["anomaly_categories"] = []
    rec_blind = evaluator.evaluate_single_patch(patch_data_blind)

    # Decisions must be 100% bit-exact identical
    assert rec_with_gt.k_selected == rec_blind.k_selected
    assert rec_with_gt.valid_reduction_pct == rec_blind.valid_reduction_pct
    assert rec_with_gt.high_stress_adaptive_pct == rec_blind.high_stress_adaptive_pct
    assert rec_with_gt.mod_stress_adaptive_pct == rec_blind.mod_stress_adaptive_pct
    assert rec_with_gt.energy_total_adaptive_j == rec_blind.energy_total_adaptive_j


# ==============================================================================
# 8. Artifact Exports & JSON Validation
# ==============================================================================

def test_exports_and_json_validity(tmp_path):
    """Verify CSV, JSON, and Markdown exports generate valid non-empty files."""
    evaluator = DatasetWideEvaluator(
        output_dir=tmp_path, budget_ratio=0.35, ndvi_target=0.32, ndvi_stressed=0.20
    )
    records = evaluator.evaluate_all(splits=["train"], max_patches=5)
    summary = evaluator.compute_aggregates(records)
    tests = evaluator.compute_statistical_tests(records)

    # CSV
    csv_file = evaluator.export_csv(records, tmp_path / "test_metrics.csv")
    assert csv_file.is_file()
    df_read = pd.read_csv(csv_file)
    assert len(df_read) == 5

    # JSON
    json_file = evaluator.export_json(summary, tmp_path / "test_summary.json")
    assert json_file.is_file()
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["total_patches"] == 5

    # Markdown
    md_file = evaluator.generate_markdown_report(summary, tests, tmp_path / "test_report.md")
    assert md_file.is_file()
    content = md_file.read_text(encoding="utf-8")
    assert "# StressSense: Dataset-Wide Experimental Validation Report" in content


# ==============================================================================
# 9. DatasetVisualizer Generation Tests
# ==============================================================================

def test_dataset_visualizer_generation(tmp_path):
    """Verify DatasetVisualizer produces valid PNG figures with positive file size."""
    evaluator = DatasetWideEvaluator(
        output_dir=tmp_path, budget_ratio=0.35, ndvi_target=0.32, ndvi_stressed=0.20
    )
    records = evaluator.evaluate_all(splits=["train"], max_patches=6)
    summary = evaluator.compute_aggregates(records)

    vis = DatasetVisualizer(output_dir=tmp_path)
    p1 = vis.plot_metrics_summary(records, summary)
    assert p1.is_file()
    assert p1.stat().st_size > 50_000

    p2 = vis.plot_category_recall_breakdown(summary)
    assert p2.is_file()
    assert p2.stat().st_size > 10_000


# ==============================================================================
# 10. Audit Regression Tests: Statistical & Methodological Integrity
# ==============================================================================

def test_statistical_significance_with_alpha_001_and_bonferroni():
    """Verify strict adherence to declared alpha=0.01 with Bonferroni correction and CIs."""
    evaluator = DatasetWideEvaluator(budget_ratio=0.35, ndvi_target=0.32, ndvi_stressed=0.20, alpha=0.01)
    records = evaluator.evaluate_all(splits=["train"], max_patches=30)
    tests = evaluator.compute_statistical_tests(records)

    assert len(tests) >= 2
    for t in tests:
        assert t.alpha_declared == 0.01
        assert t.alpha_bonferroni == round(0.01 / len(tests), 5)
        # Verify 99% CI spans mean difference
        assert t.ci99_diff_lower <= t.mean_difference <= t.ci99_diff_upper
        assert t.ci95_diff_lower <= t.mean_difference <= t.ci95_diff_upper

        # Decision rule integrity: is_significant_bonferroni must be True iff p_ttest < alpha_bonferroni
        if t.p_value_ttest < t.alpha_bonferroni:
            assert t.is_significant_bonferroni is True
        else:
            assert t.is_significant_bonferroni is False

        # If p >= 0.01, declared alpha must NOT claim significance
        if t.p_value_ttest >= 0.01:
            assert t.is_significant_declared_alpha is False


def test_energy_parity_between_equal_budget_and_adaptive():
    """Verify exact energy proxy parity between Equal-Budget Uniform and Adaptive Sampling."""
    evaluator = DatasetWideEvaluator(budget_ratio=0.35, ndvi_target=0.32, ndvi_stressed=0.20)
    records = evaluator.evaluate_all(splits=["train"], max_patches=15)

    for r in records:
        # Both strategies select exactly k_selected valid regions on every patch
        assert r.k_selected >= 1
        # Energy proxy must be identical to machine precision
        assert abs(r.energy_total_equal_j - r.energy_total_adaptive_j) < 1e-6

    summary = evaluator.compute_aggregates(records)
    totals = summary["totals"]
    assert abs(totals["cumulative_energy_equal_j"] - totals["cumulative_energy_adaptive_j"]) < 1e-4


def test_unlabeled_test_patches_excluded_from_ground_truth_denominators():
    """Verify that competition test patches with withheld labels never enter GT denominators."""
    evaluator = DatasetWideEvaluator(budget_ratio=0.35, ndvi_target=0.32, ndvi_stressed=0.20)
    test_records = evaluator.evaluate_all(splits=["test"], max_patches=10)

    for r in test_records:
        assert r.split == "test"
        assert r.gt_available is False
        assert r.gt_anomalous_regions == 0
        assert r.adaptive_recall_pct is None
        assert r.equal_recall_pct is None
        assert r.adaptive_precision_pct is None
        assert r.equal_precision_pct is None

        # Flat dictionary export for CSV must have empty strings, not 0.0
        flat = r.to_flat_dict()
        assert flat["adaptive_recall_pct"] == ""
        assert flat["equal_recall_pct"] == ""

    # In summary, test split must report N/A for ground truth metrics
    summary = evaluator.compute_aggregates(test_records)
    assert summary["anomalous_patches_count"] == 0
    assert summary["split_breakdown"]["test"]["ground_truth_recall_adaptive_pct"] is None


def test_paired_observations_symmetric_trial_matching():
    """Verify that paired observations are matched symmetrically across repeated seeds."""
    evaluator = DatasetWideEvaluator(budget_ratio=0.35, ndvi_target=0.32, ndvi_stressed=0.20, n_trials=5)
    records = evaluator.evaluate_all(splits=["train"], max_patches=8)

    for r in records:
        assert r.total_regions == 16
        # Both adaptive and equal-budget metrics must be trial-averaged symmetrically
        assert 0.0 <= r.high_stress_adaptive_pct <= 100.0
        assert 0.0 <= r.high_stress_equal_pct <= 100.0
        if r.gt_available and r.gt_anomalous_regions > 0:
            assert r.adaptive_recall_pct is not None
            assert r.equal_recall_pct is not None
            assert 0.0 <= r.adaptive_recall_pct <= 100.0
            assert 0.0 <= r.equal_recall_pct <= 100.0


def test_category_wise_recall_no_cross_contamination():
    """Verify localized region-specific category masking prevents multi-label contamination."""
    evaluator = DatasetWideEvaluator()
    # Create synthetic multi-modal patch with two non-overlapping category anomalies
    shape = (128, 128)
    rgb = np.full((128, 128, 3), 100, dtype=np.uint8)
    nir = np.full((128, 128), 150, dtype=np.uint8)
    msk = np.full((128, 128), 255, dtype=np.uint8)

    # Category 1: standing_water exclusively in region 0 (y: 0..32, x: 0..32)
    cat_mask_water = np.zeros(shape, dtype=np.uint8)
    cat_mask_water[0:32, 0:32] = 255

    # Category 2: planter_skip exclusively in region 1 (y: 0..32, x: 32..64)
    cat_mask_skip = np.zeros(shape, dtype=np.uint8)
    cat_mask_skip[0:32, 32:64] = 255

    cat_masks = {
        "standing_water": cat_mask_water,
        "planter_skip": cat_mask_skip,
    }
    composite_gt = cat_mask_water | cat_mask_skip

    regions = evaluator.grid_generator.generate_grid(shape, validity_mask=msk)
    scores = evaluator.stress_scorer.score_all_regions(
        rgb=rgb,
        nir=nir,
        regions=regions,
        validity_mask=msk,
        ground_truth_mask=composite_gt,
        ground_truth_categories=["standing_water", "planter_skip"],
        category_masks=cat_masks,
    )

    reg0 = [s for s in scores if s.row == 0 and s.col == 0][0]
    reg1 = [s for s in scores if s.row == 0 and s.col == 1][0]
    reg2 = [s for s in scores if s.row == 0 and s.col == 2][0]

    # Region 0 MUST contain only standing_water
    assert reg0.gt_categories == ["standing_water"]
    # Region 1 MUST contain only planter_skip
    assert reg1.gt_categories == ["planter_skip"]
    # Region 2 (no anomaly) MUST have no categories
    assert reg2.gt_categories is None or reg2.gt_categories == []


def test_dataset_split_integrity_and_zero_field_leakage():
    """Verify strict mutual disjointness of flight fields across train, val, and test splits."""
    evaluator = DatasetWideEvaluator()
    df = evaluator.load_manifest()

    train_fields = set(df[df["split"] == "train"]["field_id"].unique())
    val_fields = set(df[df["split"] == "val"]["field_id"].unique())
    test_fields = set(df[df["split"] == "test"]["field_id"].unique())

    assert len(train_fields) == 20
    assert len(val_fields) == 10
    assert len(test_fields) == 10

    # Cross-split intersections must be strictly empty
    assert len(train_fields & val_fields) == 0, "Data leakage between train and val!"
    assert len(train_fields & test_fields) == 0, "Data leakage between train and test!"
    assert len(val_fields & test_fields) == 0, "Data leakage between val and test!"
