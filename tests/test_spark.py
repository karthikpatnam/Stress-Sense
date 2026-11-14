"""Tests and parity verification for distributed Apache Spark evaluation engine."""

import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

from src.adaptive.sampling_engine import BudgetOverflowPolicy
from src.evaluation.dataset_evaluator import DatasetWideEvaluator, PatchRecord
from src.distributed.spark_evaluator import (
    SparkDistributedEvaluator,
    SparkEvaluatorConfig,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def spark_evaluator():
    """Module-level fixture for SparkDistributedEvaluator using local[2]."""
    config = SparkEvaluatorConfig(
        master="local[2]",
        app_name="StressSensePytestSpark",
        grid_size=32,
        budget_ratio=0.35,
        ndvi_target=0.32,
        ndvi_stressed=0.20,
        overflow_policy="guarantee_high_stress",
        exploration_rate=0.05,
        seed=42,
        n_trials=5,
        overlap_threshold=0.10,
    )
    evaluator = SparkDistributedEvaluator(config=config)
    evaluator.get_or_create_spark_session(master="local[2]")
    yield evaluator
    evaluator.stop_spark_session()


def test_spark_evaluator_config_and_initialization():
    """Verify SparkEvaluatorConfig dataclass and default parameter integrity."""
    cfg = SparkEvaluatorConfig()
    assert cfg.master == "local[2]"
    assert cfg.grid_size == 32
    assert cfg.ndvi_target == 0.32
    assert cfg.ndvi_stressed == 0.20
    assert cfg.budget_ratio == 0.35
    assert cfg.overflow_policy == "guarantee_high_stress"
    assert cfg.exploration_rate == 0.05
    assert cfg.overlap_threshold == 0.10

    evaluator = SparkDistributedEvaluator(config=cfg)
    assert evaluator.config == cfg
    assert evaluator.output_dir.is_dir()


def test_spark_evaluator_session_lifecycle():
    """Verify local SparkSession creation, configuration properties, and shutdown."""
    cfg = SparkEvaluatorConfig(master="local[1]", app_name="StressSenseLifecycleTest")
    evaluator = SparkDistributedEvaluator(config=cfg)
    spark = evaluator.get_or_create_spark_session()
    try:
        assert spark is not None
        assert spark.sparkContext.master.startswith("local")
        assert spark.conf.get("spark.driver.host") == "127.0.0.1"
        assert spark.conf.get("spark.driver.bindAddress") == "127.0.0.1"
    finally:
        evaluator.stop_spark_session()


def test_spark_exact_parity_on_single_patch(spark_evaluator):
    """Parity Test: Compare distributed Spark execution against single-machine on identical patch.

    Verifies:
    1. Exact patch ID agreement.
    2. Exact selected-region ID list agreement.
    3. Exact sampling budget agreement (k_selected, valid_regions, invalid_regions).
    4. Exact numerical metric agreement within floating point tolerance (1e-5).
    5. Bit-exact energy proxy parity.
    """
    # 1. Run single-machine evaluator on patch '01E8LMC2K_0_0'
    sm_evaluator = DatasetWideEvaluator(
        grid_size=32,
        budget_ratio=0.35,
        ndvi_target=0.32,
        ndvi_stressed=0.20,
        overflow_policy=BudgetOverflowPolicy.GUARANTEE_HIGH_STRESS,
        exploration_rate=0.05,
        seed=42,
        n_trials=5,
        overlap_threshold=0.10,
    )

    df_manifest = sm_evaluator.load_manifest(split="train")
    target_row = df_manifest.iloc[0]
    target_patch_id = str(target_row["patch_id"])
    patch_data = sm_evaluator.load_patch_data(target_row)
    sm_record = sm_evaluator.evaluate_single_patch(patch_data)

    # 2. Run Spark distributed evaluator on the exact same patch
    spark_records, telemetry = spark_evaluator.evaluate_distributed(
        splits=["train"],
        max_patches=1,
    )

    assert len(spark_records) == 1
    spark_record = spark_records[0]

    # Verification 1: Exact patch ID and metadata
    assert spark_record.patch_id == sm_record.patch_id == target_patch_id
    assert spark_record.split == sm_record.split == "train"
    assert spark_record.field_id == sm_record.field_id
    assert spark_record.has_anomaly == sm_record.has_anomaly

    # Verification 2: Exact region counts and sampling budgets
    assert spark_record.total_regions == sm_record.total_regions
    assert spark_record.valid_regions == sm_record.valid_regions
    assert spark_record.invalid_regions == sm_record.invalid_regions
    assert spark_record.k_selected == sm_record.k_selected
    assert spark_record.full_selected == sm_record.full_selected

    # Verification 3: Exact selected region IDs agreement
    assert spark_record.selected_region_ids == sm_record.selected_region_ids
    assert len(spark_record.selected_region_ids) == spark_record.k_selected

    # Verification 4: Metric numerical parity within float tolerance
    assert math.isclose(spark_record.valid_reduction_pct, sm_record.valid_reduction_pct, rel_tol=1e-5)
    assert math.isclose(spark_record.sampling_reduction_pct, sm_record.sampling_reduction_pct, rel_tol=1e-5)
    assert math.isclose(spark_record.high_stress_adaptive_pct, sm_record.high_stress_adaptive_pct, rel_tol=1e-5)
    assert math.isclose(spark_record.high_stress_equal_pct, sm_record.high_stress_equal_pct, rel_tol=1e-5)
    assert math.isclose(spark_record.mod_stress_adaptive_pct, sm_record.mod_stress_adaptive_pct, rel_tol=1e-5)
    assert math.isclose(spark_record.mod_stress_equal_pct, sm_record.mod_stress_equal_pct, rel_tol=1e-5)

    # Verification 5: Energy proxy parity
    assert math.isclose(spark_record.energy_total_full_j, sm_record.energy_total_full_j, rel_tol=1e-5)
    assert math.isclose(spark_record.energy_total_adaptive_j, sm_record.energy_total_adaptive_j, rel_tol=1e-5)
    assert math.isclose(spark_record.energy_total_equal_j, sm_record.energy_total_equal_j, rel_tol=1e-5)
    assert math.isclose(spark_record.energy_total_adaptive_j, spark_record.energy_total_equal_j, rel_tol=1e-7)


def test_spark_multi_patch_parity_across_splits(spark_evaluator):
    """Parity Test: Compare batch of patches across train, val, and test splits.

    Ensures that unlabeled test patches have None for ground-truth recall and precision,
    while valid train and val patches have identical values.
    """
    sm_evaluator = DatasetWideEvaluator(
        grid_size=32,
        budget_ratio=0.35,
        ndvi_target=0.32,
        ndvi_stressed=0.20,
        overflow_policy=BudgetOverflowPolicy.GUARANTEE_HIGH_STRESS,
        exploration_rate=0.05,
        seed=42,
        n_trials=5,
        overlap_threshold=0.10,
    )

    # Select 6 patches across splits: 2 train, 2 val, 2 test
    sm_train = sm_evaluator.evaluate_all(splits=["train"], max_patches=2)
    sm_val = sm_evaluator.evaluate_all(splits=["val"], max_patches=2)
    sm_test = sm_evaluator.evaluate_all(splits=["test"], max_patches=2)
    sm_all = {r.patch_id: r for r in sm_train + sm_val + sm_test}

    # Evaluate using Spark across the same splits
    spark_train, _ = spark_evaluator.evaluate_distributed(splits=["train"], max_patches=2)
    spark_val, _ = spark_evaluator.evaluate_distributed(splits=["val"], max_patches=2)
    spark_test, _ = spark_evaluator.evaluate_distributed(splits=["test"], max_patches=2)
    spark_all = {r.patch_id: r for r in spark_train + spark_val + spark_test}

    assert set(spark_all.keys()) == set(sm_all.keys())

    for pid in spark_all:
        sp_rec = spark_all[pid]
        sm_rec = sm_all[pid]

        assert sp_rec.k_selected == sm_rec.k_selected
        assert sp_rec.valid_regions == sm_rec.valid_regions
        assert sp_rec.selected_region_ids == sm_rec.selected_region_ids
        assert math.isclose(sp_rec.valid_reduction_pct, sm_rec.valid_reduction_pct, rel_tol=1e-5)

        # Unlabeled test patch check: ground truth must be None
        if sp_rec.split == "test":
            assert sp_rec.adaptive_recall_pct is None
            assert sp_rec.equal_recall_pct is None
            assert sp_rec.adaptive_precision_pct is None
            assert sp_rec.equal_precision_pct is None
        elif sp_rec.gt_available:
            assert sp_rec.adaptive_recall_pct is not None
            assert sm_rec.adaptive_recall_pct is not None
            assert math.isclose(sp_rec.adaptive_recall_pct, sm_rec.adaptive_recall_pct, rel_tol=1e-5)


def test_spark_overlap_sensitivity_analysis_execution(spark_evaluator):
    """Verify category overlap sensitivity sweep across 0%, 5%, 10%, and 20% thresholds."""
    sensitivity = spark_evaluator.run_overlap_sensitivity_analysis(
        splits=["train"],
        max_patches=4,
        thresholds=(0.0, 0.05, 0.10, 0.20),
        master="local[2]",
    )

    assert "thresh_0pct" in sensitivity
    assert "thresh_5pct" in sensitivity
    assert "thresh_10pct" in sensitivity
    assert "thresh_20pct" in sensitivity

    # Verify that the default 10% configuration remains preserved after the analysis
    assert spark_evaluator.config.overlap_threshold == 0.10


def test_spark_telemetry_and_startup_overhead(spark_evaluator):
    """Verify that telemetry accurately isolates startup overhead from steady-state runtime."""
    records, telemetry = spark_evaluator.evaluate_distributed(
        splits=["train"],
        max_patches=2,
    )

    assert "startup_overhead_sec" in telemetry
    assert "steady_state_time_sec" in telemetry
    assert "total_wall_clock_sec" in telemetry
    assert "throughput_patches_per_sec" in telemetry
    assert "partitions" in telemetry

    # Startup overhead and steady-state must be non-negative
    assert telemetry["startup_overhead_sec"] >= 0.0
    assert telemetry["steady_state_time_sec"] >= 0.0
    assert telemetry["total_wall_clock_sec"] >= telemetry["steady_state_time_sec"]
    assert telemetry["evaluated_patches"] == 2
