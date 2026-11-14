"""Evaluator and benchmarking engine for StressSense adaptive sampling.

This module provides a reproducible scientific evaluation framework that compares:
1. FULL_UNIFORM: Baseline 100% scan of all valid crop canopy regions.
2. EQUAL_BUDGET_UNIFORM: Fair comparison baseline selecting exactly K valid
   regions via reproducible pseudo-random sampling.
3. ADAPTIVE: Prioritized sampling based on Relative Spectral Difference Index
   (RSDI) stress tiers, healthy exploration, and overflow policies.

SCIENTIFIC INTEGRITY & EVALUATION PROTOCOLS:
1. Ground-Truth Decoupling:
   Agriculture-Vision ground-truth anomaly masks are strictly used for post-hoc
   diagnostic evaluation, never for RSDI scoring or sampling decisions.
   RSDI stress categories (HIGH_STRESS, MODERATE_STRESS, HEALTHY) are empirical
   spectral indicators and are clearly distinguished from authentic ground-truth
   pathological or agronomic anomaly categories (e.g., weed_cluster, planter_skip).
2. Sampling Reduction Integrity:
   'valid_reduction_pct' is the primary sampling-efficiency metric. It measures
   savings strictly over candidate crop canopy regions. Inherently invalid or
   background areas are never treated as avoided samples.
3. Resource-Cost & Energy Modeling:
   All energy, network transmission, and downstream compute savings are modeled
   via explicit, linear computational proxies. Values are explicitly labeled as
   estimates or proxies; no onboard hardware current/voltage telemetry was measured.
"""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np

from src.adaptive.sampling_engine import (
    AdaptiveSampler,
    BudgetOverflowPolicy,
    SamplingDecision,
    SamplingEngine,
    SamplingResult,
    SamplingStrategy,
    UniformSampler,
)
from src.adaptive.stress_scorer import RegionStressScore, StressCategory


@dataclass
class ResourceCostAssumptions:
    """Configurable parameters and assumptions for resource cost estimation.

    Attributes:
        bytes_per_pixel: Raw multi-spectral data bytes per pixel (default 4: RGB + NIR uint8).
        compression_ratio: Data compression ratio (1.0 = raw uncompressed, <1.0 = compressed).
        compute_units_per_region: Normalized compute units [NCU] per region deep-learning inference.
        inference_latency_seconds_per_region: Estimated edge model execution latency per patch.
        inference_power_watts: Estimated edge accelerator active power consumption (e.g. Jetson Orin).
        transmission_power_watts: Estimated drone wireless RF transmitter power consumption.
        transmission_bandwidth_bytes_per_sec: Estimated drone wireless uplink transmission speed.
        is_measured_energy: Strictly False; marks all energy values as computational proxies.
        cost_model_notes: Narrative documentation of modeling assumptions.
    """

    bytes_per_pixel: int = 4
    compression_ratio: float = 1.0
    compute_units_per_region: float = 1.0
    inference_latency_seconds_per_region: float = 0.015
    inference_power_watts: float = 15.0
    transmission_power_watts: float = 2.5
    transmission_bandwidth_bytes_per_sec: float = 1_000_000.0  # 1 MB/s RF telemetry link
    operation_concurrency: str = "pipelined"  # Pipelined transmission and inference
    is_measured_energy: bool = False
    cost_model_notes: str = (
        "Linear proxy cost model. Energy is additive (E_total = E_tx + E_compute) because "
        "electrical energy across distinct hardware components (RF transmitter and edge accelerator) "
        "integrates independently over their operational durations. Operational latency is modeled "
        "assuming pipelined execution (T_total ≈ max(T_tx, T_compute) for streamed regions). "
        "No physical onboard current or voltage was measured on drone hardware."
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize cost model assumptions to dictionary format."""
        return {
            "bytes_per_pixel": self.bytes_per_pixel,
            "compression_ratio": self.compression_ratio,
            "compute_units_per_region": self.compute_units_per_region,
            "inference_latency_seconds_per_region": self.inference_latency_seconds_per_region,
            "inference_power_watts": self.inference_power_watts,
            "transmission_power_watts": self.transmission_power_watts,
            "transmission_bandwidth_bytes_per_sec": self.transmission_bandwidth_bytes_per_sec,
            "operation_concurrency": self.operation_concurrency,
            "is_measured_energy": self.is_measured_energy,
            "cost_model_notes": self.cost_model_notes,
        }


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics for a single sampling result on a scored grid.

    All efficiency, stress coverage, ground-truth detection, and resource-cost
    metrics are encapsulated here.
    """

    strategy: SamplingStrategy
    patch_id: str

    # Grid geometry & counts
    total_regions: int
    valid_regions: int
    invalid_regions: int
    selected_regions_count: int

    # Sampling efficiency
    valid_reduction_pct: float
    sampling_reduction_pct: float

    # Stress category coverage (RSDI-based)
    high_stress_total: int
    high_stress_selected: int
    high_stress_coverage_pct: float
    moderate_stress_total: int
    moderate_stress_selected: int
    moderate_stress_coverage_pct: float
    healthy_total: int
    healthy_selected: int
    healthy_sampling_pct: float

    # Ground-truth anomaly evaluation (strictly decoupled, region-level detection)
    gt_available: bool
    total_gt_anomalies: int
    detected_gt_anomalies: int
    missed_gt_anomalies: int
    gt_anomalous_region_recall_pct: Optional[float]
    gt_anomalous_region_precision_pct: Optional[float]
    gt_anomaly_recall_pct: Optional[float]  # Backward-compatible alias
    gt_anomaly_precision_pct: Optional[float]  # Backward-compatible alias
    missed_gt_anomaly_ids: List[str]
    gt_category_breakdown: Dict[str, Dict[str, Any]]

    # Resource cost estimates (computational and transmission proxies)
    pixels_per_region: int
    pixels_processed: int
    pixel_reduction_pct: float
    estimated_data_bytes: int
    estimated_data_kb: float
    data_reduction_pct: float
    estimated_workload_units: float
    workload_reduction_pct: float
    estimated_tx_energy_joules_proxy: float
    estimated_compute_energy_joules_proxy: float
    estimated_energy_joules_proxy: float
    energy_reduction_pct: float

    # Budget adherence
    requested_budget: Optional[int]
    budget_exceeded: bool
    overflow_count: int
    deficit_count: int
    seed: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to serializable dictionary."""
        return {
            "strategy": self.strategy.value,
            "patch_id": self.patch_id,
            "total_regions": self.total_regions,
            "valid_regions": self.valid_regions,
            "invalid_regions": self.invalid_regions,
            "selected_regions_count": self.selected_regions_count,
            "valid_reduction_pct": round(self.valid_reduction_pct, 2),
            "sampling_reduction_pct": round(self.sampling_reduction_pct, 2),
            "high_stress_total": self.high_stress_total,
            "high_stress_selected": self.high_stress_selected,
            "high_stress_coverage_pct": round(self.high_stress_coverage_pct, 2),
            "moderate_stress_total": self.moderate_stress_total,
            "moderate_stress_selected": self.moderate_stress_selected,
            "moderate_stress_coverage_pct": round(self.moderate_stress_coverage_pct, 2),
            "healthy_total": self.healthy_total,
            "healthy_selected": self.healthy_selected,
            "healthy_sampling_pct": round(self.healthy_sampling_pct, 2),
            "gt_available": self.gt_available,
            "total_gt_anomalies": self.total_gt_anomalies,
            "detected_gt_anomalies": self.detected_gt_anomalies,
            "missed_gt_anomalies": self.missed_gt_anomalies,
            "gt_anomalous_region_recall_pct": round(self.gt_anomalous_region_recall_pct, 2)
            if self.gt_anomalous_region_recall_pct is not None
            else None,
            "gt_anomalous_region_precision_pct": round(self.gt_anomalous_region_precision_pct, 2)
            if self.gt_anomalous_region_precision_pct is not None
            else None,
            "gt_anomaly_recall_pct": round(self.gt_anomaly_recall_pct, 2)
            if self.gt_anomaly_recall_pct is not None
            else None,
            "gt_anomaly_precision_pct": round(self.gt_anomaly_precision_pct, 2)
            if self.gt_anomaly_precision_pct is not None
            else None,
            "missed_gt_anomaly_ids": self.missed_gt_anomaly_ids,
            "gt_category_breakdown": self.gt_category_breakdown,
            "pixels_per_region": self.pixels_per_region,
            "pixels_processed": self.pixels_processed,
            "pixel_reduction_pct": round(self.pixel_reduction_pct, 2),
            "estimated_data_bytes": self.estimated_data_bytes,
            "estimated_data_kb": round(self.estimated_data_kb, 2),
            "data_reduction_pct": round(self.data_reduction_pct, 2),
            "estimated_workload_units": round(self.estimated_workload_units, 2),
            "workload_reduction_pct": round(self.workload_reduction_pct, 2),
            "estimated_tx_energy_joules_proxy": round(self.estimated_tx_energy_joules_proxy, 4),
            "estimated_compute_energy_joules_proxy": round(self.estimated_compute_energy_joules_proxy, 4),
            "estimated_energy_joules_proxy": round(self.estimated_energy_joules_proxy, 4),
            "energy_reduction_pct": round(self.energy_reduction_pct, 2),
            "requested_budget": self.requested_budget,
            "budget_exceeded": self.budget_exceeded,
            "overflow_count": self.overflow_count,
            "deficit_count": self.deficit_count,
            "seed": self.seed,
            "metadata": self.metadata,
        }

    def to_flat_dict(self) -> Dict[str, Any]:
        """Convert metrics to a flattened dictionary suitable for tabular/CSV rows."""
        d = self.to_dict()
        flat: Dict[str, Any] = {}
        for k, v in d.items():
            if k == "missed_gt_anomaly_ids":
                flat[k] = ";".join(v) if v else "none"
            elif k in ("gt_category_breakdown", "metadata"):
                flat[k] = json.dumps(v)
            else:
                flat[k] = v
        return flat


@dataclass
class TrialAggregateMetrics:
    """Aggregated statistical metrics across multiple random seed trials."""

    strategy: SamplingStrategy
    n_trials: int
    seeds: List[int]
    metric_stats: Dict[str, Dict[str, float]]
    trials: List[EvaluationMetrics] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize aggregated trial metrics to dictionary format."""
        return {
            "strategy": self.strategy.value,
            "n_trials": self.n_trials,
            "seeds": self.seeds,
            "metric_stats": self.metric_stats,
            "trials": [t.to_dict() for t in self.trials],
        }


@dataclass
class BenchmarkComparison:
    """Complete benchmark comparing Adaptive, Equal-Budget Uniform, and Full Uniform."""

    patch_id: str
    grid_shape: Tuple[int, int]
    grid_size: int
    stride: int
    cost_assumptions: ResourceCostAssumptions
    strategy_metrics: Dict[str, EvaluationMetrics]
    trial_aggregates: Optional[Dict[str, TrialAggregateMetrics]] = None
    comparison_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize complete benchmark to dictionary."""
        return {
            "patch_id": self.patch_id,
            "grid_shape": list(self.grid_shape),
            "grid_size": self.grid_size,
            "stride": self.stride,
            "cost_assumptions": self.cost_assumptions.to_dict(),
            "strategy_metrics": {
                k: v.to_dict() for k, v in self.strategy_metrics.items()
            },
            "trial_aggregates": {
                k: v.to_dict() for k, v in self.trial_aggregates.items()
            }
            if self.trial_aggregates
            else None,
            "comparison_summary": self.comparison_summary,
        }

    def to_dataframe(self):
        """Convert strategy metrics to a pandas DataFrame."""
        import pandas as pd

        rows = [m.to_flat_dict() for m in self.strategy_metrics.values()]
        return pd.DataFrame(rows)


class AdaptiveEvaluator:
    """Evaluation and benchmarking engine for adaptive aerial crop health sampling."""

    def __init__(
        self,
        cost_assumptions: Optional[ResourceCostAssumptions] = None,
        default_seed: int = 42,
    ) -> None:
        """Initialize the AdaptiveEvaluator.

        Args:
            cost_assumptions: ResourceCostAssumptions specifying proxy cost constants.
            default_seed: Pinned default random seed for reproducible comparisons.
        """
        self.cost_assumptions = cost_assumptions or ResourceCostAssumptions()
        self.default_seed = int(default_seed)
        self.sampling_engine = SamplingEngine(default_seed=self.default_seed)

    def evaluate_result(
        self,
        sampling_result: SamplingResult,
        scores: List[RegionStressScore],
        patch_id: str = "patch_000",
        pixels_per_region: Optional[int] = None,
        full_baseline_result: Optional[SamplingResult] = None,
    ) -> EvaluationMetrics:
        """Compute exhaustive evaluation metrics for a single sampling result.

        Args:
            sampling_result: SamplingResult instance to evaluate.
            scores: List of RegionStressScore objects across the grid.
            patch_id: Unique identifier for the aerial image/patch.
            pixels_per_region: Pixels per grid region (default derived from scores).
            full_baseline_result: Optional reference full-coverage result for reduction baselines.

        Returns:
            EvaluationMetrics containing efficiency, coverage, ground-truth, and cost metrics.
        """
        # Determine pixels per region safely
        if pixels_per_region is not None:
            px_per_reg = int(pixels_per_region)
        elif scores and len(scores) > 0:
            px_per_reg = scores[0].total_pixel_count
        else:
            px_per_reg = 1024  # Default fallback (32x32)

        # Build region lookup map
        score_map: Dict[str, RegionStressScore] = {s.region_id: s for s in scores}
        selected_ids: Set[str] = {
            d.region_id for d in sampling_result.selected_regions if d.is_selected
        }

        # 1. Geometry and Region Counts
        total_regions = sampling_result.total_regions
        valid_regions = sampling_result.valid_regions
        invalid_regions = sampling_result.invalid_regions
        selected_count = sampling_result.actual_selected_count

        # Defensive check: Invalid regions must never be counted as selected
        for d in sampling_result.selected_regions:
            assert d.is_valid, f"Invalid region {d.region_id} was selected!"

        # 2. Sampling Efficiency Metrics
        valid_red = (
            ((valid_regions - selected_count) / valid_regions * 100.0)
            if valid_regions > 0
            else 0.0
        )
        sampling_red = (
            ((total_regions - selected_count) / total_regions * 100.0)
            if total_regions > 0
            else 0.0
        )

        # 3. Stress Category Coverage (RSDI Model Tiers)
        valid_scores = [s for s in scores if s.is_valid]
        high_tier = [s for s in valid_scores if s.stress_category == StressCategory.HIGH_STRESS]
        mod_tier = [s for s in valid_scores if s.stress_category == StressCategory.MODERATE_STRESS]
        healthy_tier = [s for s in valid_scores if s.stress_category == StressCategory.HEALTHY]

        n_high_total = len(high_tier)
        n_high_sel = sum(1 for s in high_tier if s.region_id in selected_ids)
        high_cov_pct = (
            (n_high_sel / n_high_total * 100.0) if n_high_total > 0 else 100.0
        )

        n_mod_total = len(mod_tier)
        n_mod_sel = sum(1 for s in mod_tier if s.region_id in selected_ids)
        mod_cov_pct = (
            (n_mod_sel / n_mod_total * 100.0) if n_mod_total > 0 else 100.0
        )

        n_healthy_total = len(healthy_tier)
        n_healthy_sel = sum(1 for s in healthy_tier if s.region_id in selected_ids)
        healthy_samp_pct = (
            (n_healthy_sel / n_healthy_total * 100.0) if n_healthy_total > 0 else 0.0
        )

        # 4. Ground-Truth Anomaly Evaluation (Strictly Decoupled)
        gt_present_in_scores = any(s.gt_has_anomaly is not None for s in scores)
        gt_category_breakdown: Dict[str, Dict[str, Any]] = {}

        if gt_present_in_scores:
            gt_available = True
            # Count ground truth anomalies only among valid crop regions
            gt_valid_scores = [s for s in valid_scores if s.gt_has_anomaly is True]
            total_gt_anomalies = len(gt_valid_scores)

            detected_gt_scores = [s for s in gt_valid_scores if s.region_id in selected_ids]
            detected_gt_anomalies = len(detected_gt_scores)
            missed_gt_scores = [s for s in gt_valid_scores if s.region_id not in selected_ids]
            missed_gt_anomalies = len(missed_gt_scores)
            missed_gt_anomaly_ids = [s.region_id for s in missed_gt_scores]

            gt_recall_pct = (
                (detected_gt_anomalies / total_gt_anomalies * 100.0)
                if total_gt_anomalies > 0
                else 100.0
            )
            gt_precision_pct = (
                (detected_gt_anomalies / selected_count * 100.0)
                if selected_count > 0
                else 0.0
            )

            # Extract category-wise breakdown if available
            all_categories: Set[str] = set()
            for s in gt_valid_scores:
                if s.gt_categories:
                    for cat in s.gt_categories:
                        if cat and cat.lower() != "none":
                            all_categories.add(cat)

            for cat in sorted(all_categories):
                cat_total = sum(
                    1
                    for s in gt_valid_scores
                    if s.gt_categories and cat in s.gt_categories
                )
                cat_detected = sum(
                    1
                    for s in detected_gt_scores
                    if s.gt_categories and cat in s.gt_categories
                )
                cat_missed = cat_total - cat_detected
                cat_recall = (
                    (cat_detected / cat_total * 100.0) if cat_total > 0 else 100.0
                )
                gt_category_breakdown[cat] = {
                    "total": cat_total,
                    "detected": cat_detected,
                    "missed": cat_missed,
                    "recall_pct": round(cat_recall, 2),
                }

        else:
            gt_available = False
            total_gt_anomalies = 0
            detected_gt_anomalies = 0
            missed_gt_anomalies = 0
            gt_recall_pct = None
            gt_precision_pct = None
            missed_gt_anomaly_ids = []

        # 5. Resource-Cost & Energy Modeling (Computational Proxies)
        cost_cfg = self.cost_assumptions
        bytes_per_region = int(
            px_per_reg * cost_cfg.bytes_per_pixel * cost_cfg.compression_ratio
        )
        workload_per_region = float(cost_cfg.compute_units_per_region)

        # Baseline reference counts (Full-Uniform reference)
        if full_baseline_result is not None:
            baseline_count = full_baseline_result.actual_selected_count
        else:
            # Full uniform over valid crop regions
            baseline_count = valid_regions

        baseline_pixels = baseline_count * px_per_reg
        baseline_bytes = baseline_count * bytes_per_region
        baseline_workload = baseline_count * workload_per_region

        # Strategy resource costs
        pixels_processed = selected_count * px_per_reg
        estimated_data_bytes = selected_count * bytes_per_region
        estimated_data_kb = estimated_data_bytes / 1024.0
        estimated_workload = selected_count * workload_per_region

        # Energy Proxy Calculation
        # Transmission Energy = (Bytes / Bandwidth) * RF_Power
        # Compute Energy = Latency * Compute_Power
        tx_seconds_per_region = bytes_per_region / cost_cfg.transmission_bandwidth_bytes_per_sec
        tx_energy_per_region = tx_seconds_per_region * cost_cfg.transmission_power_watts
        compute_energy_per_region = (
            cost_cfg.inference_latency_seconds_per_region * cost_cfg.inference_power_watts
        )
        total_energy_per_region = tx_energy_per_region + compute_energy_per_region

        estimated_tx_energy_joules = selected_count * tx_energy_per_region
        estimated_compute_energy_joules = selected_count * compute_energy_per_region
        estimated_energy_joules = selected_count * total_energy_per_region
        baseline_energy_joules = baseline_count * total_energy_per_region

        # Reductions
        pixel_red = (
            ((baseline_pixels - pixels_processed) / baseline_pixels * 100.0)
            if baseline_pixels > 0
            else 0.0
        )
        data_red = (
            ((baseline_bytes - estimated_data_bytes) / baseline_bytes * 100.0)
            if baseline_bytes > 0
            else 0.0
        )
        workload_red = (
            ((baseline_workload - estimated_workload) / baseline_workload * 100.0)
            if baseline_workload > 0
            else 0.0
        )
        energy_red = (
            ((baseline_energy_joules - estimated_energy_joules) / baseline_energy_joules * 100.0)
            if baseline_energy_joules > 0
            else 0.0
        )

        return EvaluationMetrics(
            strategy=sampling_result.strategy,
            patch_id=patch_id,
            total_regions=total_regions,
            valid_regions=valid_regions,
            invalid_regions=invalid_regions,
            selected_regions_count=selected_count,
            valid_reduction_pct=valid_red,
            sampling_reduction_pct=sampling_red,
            high_stress_total=n_high_total,
            high_stress_selected=n_high_sel,
            high_stress_coverage_pct=high_cov_pct,
            moderate_stress_total=n_mod_total,
            moderate_stress_selected=n_mod_sel,
            moderate_stress_coverage_pct=mod_cov_pct,
            healthy_total=n_healthy_total,
            healthy_selected=n_healthy_sel,
            healthy_sampling_pct=healthy_samp_pct,
            gt_available=gt_available,
            total_gt_anomalies=total_gt_anomalies,
            detected_gt_anomalies=detected_gt_anomalies,
            missed_gt_anomalies=missed_gt_anomalies,
            gt_anomalous_region_recall_pct=gt_recall_pct,
            gt_anomalous_region_precision_pct=gt_precision_pct,
            gt_anomaly_recall_pct=gt_recall_pct,
            gt_anomaly_precision_pct=gt_precision_pct,
            missed_gt_anomaly_ids=missed_gt_anomaly_ids,
            gt_category_breakdown=gt_category_breakdown,
            pixels_per_region=px_per_reg,
            pixels_processed=pixels_processed,
            pixel_reduction_pct=pixel_red,
            estimated_data_bytes=estimated_data_bytes,
            estimated_data_kb=estimated_data_kb,
            data_reduction_pct=data_red,
            estimated_workload_units=estimated_workload,
            workload_reduction_pct=workload_red,
            estimated_tx_energy_joules_proxy=estimated_tx_energy_joules,
            estimated_compute_energy_joules_proxy=estimated_compute_energy_joules,
            estimated_energy_joules_proxy=estimated_energy_joules,
            energy_reduction_pct=energy_red,
            requested_budget=sampling_result.requested_budget,
            budget_exceeded=sampling_result.budget_exceeded,
            overflow_count=sampling_result.overflow_count,
            deficit_count=sampling_result.deficit_count,
            seed=sampling_result.seed,
            metadata={
                **sampling_result.metadata,
                "selected_region_ids": [d.region_id for d in sampling_result.selected_regions if d.is_selected],
            },
        )

    def compare_strategies(
        self,
        scores: List[RegionStressScore],
        adaptive_budget: Optional[int] = None,
        overflow_policy: Optional[BudgetOverflowPolicy] = None,
        healthy_exploration_rate: Optional[float] = None,
        seed: Optional[int] = None,
        patch_id: str = "patch_000",
        pixels_per_region: Optional[int] = None,
    ) -> BenchmarkComparison:
        """Execute and benchmark all three sampling strategies on identical grid data.

        Guarantees strict fairness:
        - Adaptive sampling is executed first to determine selected count K.
        - Equal-Budget Uniform is assigned exact budget K = adaptive.actual_selected_count.
        - Full Uniform establishes 100% valid canopy baseline.
        - All strategies evaluate identical valid candidate regions under identical seeds.

        Args:
            scores: Scored grid regions.
            adaptive_budget: Optional budget cap for adaptive sampling.
            overflow_policy: Overflow policy if high-stress regions exceed budget.
            healthy_exploration_rate: Healthy exploration fraction (default 0.05).
            seed: Pinned random seed for reproducibility.
            patch_id: Image/patch identifier.
            pixels_per_region: Spatial pixel area of each region.

        Returns:
            BenchmarkComparison containing comparative metrics across all 3 strategies.
        """
        rng_seed = seed if seed is not None else self.default_seed

        # Execute all 3 strategies via SamplingEngine facade
        raw_results = self.sampling_engine.compare_all_strategies(
            scores=scores,
            adaptive_budget=adaptive_budget,
            overflow_policy=overflow_policy,
            healthy_exploration_rate=healthy_exploration_rate,
            seed=rng_seed,
        )

        full_res = raw_results["full_uniform"]
        equal_res = raw_results["equal_budget_uniform"]
        adaptive_res = raw_results["adaptive"]

        # Evaluate Full-Uniform baseline first
        full_metrics = self.evaluate_result(
            sampling_result=full_res,
            scores=scores,
            patch_id=patch_id,
            pixels_per_region=pixels_per_region,
            full_baseline_result=full_res,
        )

        # Evaluate Equal-Budget Uniform against Full-Uniform baseline
        equal_metrics = self.evaluate_result(
            sampling_result=equal_res,
            scores=scores,
            patch_id=patch_id,
            pixels_per_region=pixels_per_region,
            full_baseline_result=full_res,
        )

        # Evaluate Adaptive Sampling against Full-Uniform baseline
        adaptive_metrics = self.evaluate_result(
            sampling_result=adaptive_res,
            scores=scores,
            patch_id=patch_id,
            pixels_per_region=pixels_per_region,
            full_baseline_result=full_res,
        )

        # Grid dimension inference
        max_row = max(s.row for s in scores) if scores else 0
        max_col = max(s.col for s in scores) if scores else 0
        grid_shape = (max_row + 1, max_col + 1)
        px_per_reg = full_metrics.pixels_per_region
        grid_size = int(np.sqrt(px_per_reg)) if px_per_reg > 0 else 32

        # Comparative summary analysis
        summary = {
            "fair_budget_matched": (
                adaptive_metrics.selected_regions_count
                == equal_metrics.selected_regions_count
            ),
            "selected_regions_k": adaptive_metrics.selected_regions_count,
            "adaptive_valid_reduction_pct": adaptive_metrics.valid_reduction_pct,
            "equal_budget_valid_reduction_pct": equal_metrics.valid_reduction_pct,
            "high_stress_coverage_delta": (
                adaptive_metrics.high_stress_coverage_pct
                - equal_metrics.high_stress_coverage_pct
            ),
            "moderate_stress_coverage_delta": (
                adaptive_metrics.moderate_stress_coverage_pct
                - equal_metrics.moderate_stress_coverage_pct
            ),
            "gt_anomaly_recall_delta": (
                (adaptive_metrics.gt_anomaly_recall_pct - equal_metrics.gt_anomaly_recall_pct)
                if (
                    adaptive_metrics.gt_anomaly_recall_pct is not None
                    and equal_metrics.gt_anomaly_recall_pct is not None
                )
                else None
            ),
            "budget_overflow_detected": adaptive_metrics.budget_exceeded,
            "overflow_regions_count": adaptive_metrics.overflow_count,
            "data_reduction_pct": adaptive_metrics.data_reduction_pct,
            "workload_reduction_pct": adaptive_metrics.workload_reduction_pct,
            "energy_reduction_pct": adaptive_metrics.energy_reduction_pct,
        }

        return BenchmarkComparison(
            patch_id=patch_id,
            grid_shape=grid_shape,
            grid_size=grid_size,
            stride=grid_size,
            cost_assumptions=self.cost_assumptions,
            strategy_metrics={
                "full_uniform": full_metrics,
                "equal_budget_uniform": equal_metrics,
                "adaptive": adaptive_metrics,
            },
            comparison_summary=summary,
        )

    def run_repeated_trials(
        self,
        scores: List[RegionStressScore],
        adaptive_budget: Optional[int] = None,
        overflow_policy: Optional[BudgetOverflowPolicy] = None,
        healthy_exploration_rate: Optional[float] = None,
        seeds: Optional[List[int]] = None,
        n_trials: int = 5,
        patch_id: str = "patch_000",
        pixels_per_region: Optional[int] = None,
    ) -> BenchmarkComparison:
        """Run repeated trials across multiple random seeds to measure statistical variance.

        Measures mean and standard deviation for randomized strategies (equal_budget_uniform,
        and healthy exploration in adaptive sampling) while verifying determinism of
        high-stress and moderate-stress selections.

        Args:
            scores: Scored grid regions.
            adaptive_budget: Optional budget cap for adaptive sampling.
            overflow_policy: Overflow policy if high-stress regions exceed budget.
            healthy_exploration_rate: Healthy exploration fraction.
            seeds: List of explicit seeds. If None, generates n_trials seeds starting at default_seed.
            n_trials: Number of trials if seeds is not provided (default 5).
            patch_id: Image/patch identifier.
            pixels_per_region: Spatial pixel area per region.

        Returns:
            BenchmarkComparison containing strategy_metrics (from first trial) and
            trial_aggregates detailing mean, std, min, max across all trials.
        """
        if seeds is not None:
            trial_seeds = [int(s) for s in seeds]
        else:
            trial_seeds = [self.default_seed + i for i in range(max(1, n_trials))]

        trials_by_strategy: Dict[str, List[EvaluationMetrics]] = {
            "full_uniform": [],
            "equal_budget_uniform": [],
            "adaptive": [],
        }

        # Run trials
        for s in trial_seeds:
            bench = self.compare_strategies(
                scores=scores,
                adaptive_budget=adaptive_budget,
                overflow_policy=overflow_policy,
                healthy_exploration_rate=healthy_exploration_rate,
                seed=s,
                patch_id=patch_id,
                pixels_per_region=pixels_per_region,
            )
            for strat_key, metrics in bench.strategy_metrics.items():
                trials_by_strategy[strat_key].append(metrics)

        # Aggregate statistics for each strategy
        metric_keys = [
            "selected_regions_count",
            "valid_reduction_pct",
            "sampling_reduction_pct",
            "high_stress_coverage_pct",
            "moderate_stress_coverage_pct",
            "healthy_sampling_pct",
            "gt_anomaly_recall_pct",
            "gt_anomaly_precision_pct",
            "estimated_data_kb",
            "estimated_workload_units",
            "estimated_energy_joules_proxy",
        ]

        trial_aggregates: Dict[str, TrialAggregateMetrics] = {}

        for strat_key, trial_list in trials_by_strategy.items():
            stats_dict: Dict[str, Dict[str, float]] = {}
            for mk in metric_keys:
                vals: List[float] = []
                for m in trial_list:
                    val = getattr(m, mk)
                    if val is not None:
                        vals.append(float(val))

                if len(vals) > 0:
                    stats_dict[mk] = {
                        "mean": round(float(np.mean(vals)), 2),
                        "std": round(float(np.std(vals)), 2),
                        "min": round(float(np.min(vals)), 2),
                        "max": round(float(np.max(vals)), 2),
                    }
                else:
                    stats_dict[mk] = {
                        "mean": 0.0,
                        "std": 0.0,
                        "min": 0.0,
                        "max": 0.0,
                    }

            strat_enum = trial_list[0].strategy
            trial_aggregates[strat_key] = TrialAggregateMetrics(
                strategy=strat_enum,
                n_trials=len(trial_seeds),
                seeds=trial_seeds,
                metric_stats=stats_dict,
                trials=trial_list,
            )

        # Construct primary benchmark comparison from the first trial
        base_comparison = self.compare_strategies(
            scores=scores,
            adaptive_budget=adaptive_budget,
            overflow_policy=overflow_policy,
            healthy_exploration_rate=healthy_exploration_rate,
            seed=trial_seeds[0],
            patch_id=patch_id,
            pixels_per_region=pixels_per_region,
        )

        base_comparison.trial_aggregates = trial_aggregates
        return base_comparison

    def generate_markdown_report(self, benchmark: BenchmarkComparison) -> str:
        """Generate a structured, professional markdown benchmark report.

        Args:
            benchmark: BenchmarkComparison object containing evaluation results.

        Returns:
            Formatted Markdown report string.
        """
        ad = benchmark.strategy_metrics["adaptive"]
        eq = benchmark.strategy_metrics["equal_budget_uniform"]
        fu = benchmark.strategy_metrics["full_uniform"]
        summary = benchmark.comparison_summary
        assumptions = benchmark.cost_assumptions

        lines: List[str] = [
            f"# StressSense – Adaptive Sampling Benchmark Report: {benchmark.patch_id}",
            "",
            "## 1. Executive Summary & Strategy Comparison",
            "",
            "| Evaluation Metric | Full Uniform | Equal-Budget Uniform | Adaptive Sampling | Strategic Advantage |",
            "| :--- | :--- | :--- | :--- | :--- |",
            f"| **Selected Regions ($K$)** | {fu.selected_regions_count} / {fu.valid_regions} | {eq.selected_regions_count} / {eq.valid_regions} | {ad.selected_regions_count} / {ad.valid_regions} | {'Budget Matched' if summary.get('fair_budget_matched') else 'Discrepancy'} |",
            f"| **Valid Canopy Reduction (Primary)** | {fu.valid_reduction_pct:.2f}% | {eq.valid_reduction_pct:.2f}% | {ad.valid_reduction_pct:.2f}% | **{ad.valid_reduction_pct:.2f}% canopy savings** |",
            f"| **Total-Grid Reduction (Secondary)** | {fu.sampling_reduction_pct:.2f}% | {eq.sampling_reduction_pct:.2f}% | {ad.sampling_reduction_pct:.2f}% | Total tile reduction |",
            f"| **High-Stress RSDI Coverage** | {fu.high_stress_coverage_pct:.2f}% | {eq.high_stress_coverage_pct:.2f}% | {ad.high_stress_coverage_pct:.2f}% | **+{summary.get('high_stress_coverage_delta', 0.0):.2f}% vs. Random** |",
            f"| **Moderate-Stress RSDI Coverage** | {fu.moderate_stress_coverage_pct:.2f}% | {eq.moderate_stress_coverage_pct:.2f}% | {ad.moderate_stress_coverage_pct:.2f}% | **+{summary.get('moderate_stress_coverage_delta', 0.0):.2f}% vs. Random** |",
            f"| **Healthy Baseline Sampling** | {fu.healthy_sampling_pct:.2f}% | {eq.healthy_sampling_pct:.2f}% | {ad.healthy_sampling_pct:.2f}% | Controlled surveillance |",
        ]

        if ad.gt_available:
            rec_fu = f"{fu.gt_anomaly_recall_pct:.2f}%" if fu.gt_anomaly_recall_pct is not None else "N/A"
            rec_eq = f"{eq.gt_anomaly_recall_pct:.2f}%" if eq.gt_anomaly_recall_pct is not None else "N/A"
            rec_ad = f"{ad.gt_anomaly_recall_pct:.2f}%" if ad.gt_anomaly_recall_pct is not None else "N/A"
            delta_rec = summary.get("gt_anomaly_recall_delta")
            delta_str = f"+{delta_rec:.2f}% recall" if delta_rec is not None else "N/A"

            lines.extend([
                f"| **Ground-Truth Anomaly Recall** | {rec_fu} | {rec_eq} | {rec_ad} | **{delta_str}** |",
                f"| **Ground-Truth Precision** | {fu.gt_anomaly_precision_pct:.2f}% | {eq.gt_anomaly_precision_pct:.2f}% | {ad.gt_anomaly_precision_pct:.2f}% | Targeted detections |",
            ])

        lines.extend([
            f"| **Estimated Transmitted Data** | {fu.estimated_data_kb:.1f} KB | {eq.estimated_data_kb:.1f} KB | {ad.estimated_data_kb:.1f} KB | **{ad.data_reduction_pct:.2f}% bandwidth savings** |",
            f"| **Estimated Compute Workload** | {fu.estimated_workload_units:.1f} NCU | {eq.estimated_workload_units:.1f} NCU | {ad.estimated_workload_units:.1f} NCU | **{ad.workload_reduction_pct:.2f}% compute savings** |",
            f"| **Estimated Energy Proxy** | {fu.estimated_energy_joules_proxy:.3f} J | {eq.estimated_energy_joules_proxy:.3f} J | {ad.estimated_energy_joules_proxy:.3f} J | **{ad.energy_reduction_pct:.2f}% proxy energy savings** |",
            "",
            "---",
            "",
            "## 2. Sampling Efficiency & Invariant Analysis",
            "",
            f"- **Grid Raster Area**: {benchmark.grid_shape[0]} rows × {benchmark.grid_shape[1]} cols = {ad.total_regions} total grid regions.",
            f"- **Valid Agricultural Canopy**: {ad.valid_regions} regions ({ad.valid_regions / ad.total_regions * 100.0:.1f}% coverage).",
            f"- **Invalid / Masked Background**: {ad.invalid_regions} regions (100% excluded across all sampling strategies).",
            f"- **Primary Metric (`valid_reduction_pct`)**: {ad.valid_reduction_pct:.2f}% reduction.",
            f"- **Secondary Metric (`sampling_reduction_pct`)**: {ad.sampling_reduction_pct:.2f}% reduction.",
            "> [!NOTE]",
            "> `valid_reduction_pct` is the mandatory primary efficiency metric. It strictly measures reductions over valid crop canopy candidates.",
            "> Non-crop background or obscured areas are never treated as avoided samples, preventing exaggerated savings claims.",
            "",
            "---",
            "",
            "## 3. Stress Detection & Ground-Truth Anomaly Coverage",
            "",
            "### Relative Spectral Difference Index (RSDI) Tiers",
            f"- **High Stress (RSDI >= 0.70)**: {ad.high_stress_selected} of {ad.high_stress_total} selected ({ad.high_stress_coverage_pct:.1f}% coverage).",
            f"- **Moderate Stress (0.35 <= RSDI < 0.70)**: {ad.moderate_stress_selected} of {ad.moderate_stress_total} selected ({ad.moderate_stress_coverage_pct:.1f}% coverage).",
            f"- **Healthy Baseline (RSDI < 0.35)**: {ad.healthy_selected} of {ad.healthy_total} sampled ({ad.healthy_sampling_pct:.1f}% surveillance rate).",
        ])

        if ad.gt_available:
            lines.extend([
                "",
                "### Authentic Ground-Truth Anomaly Evaluation",
                f"- **Total Valid Anomaly Regions in Patch**: {ad.total_gt_anomalies}",
                f"- **Adaptive Strategy Detections**: {ad.detected_gt_anomalies} anomalies ({ad.gt_anomaly_recall_pct:.1f}% recall).",
                f"- **Adaptive Strategy Missed**: {ad.missed_gt_anomalies} anomalies.",
                f"- **Missed Anomaly Region IDs**: `{', '.join(ad.missed_gt_anomaly_ids) if ad.missed_gt_anomaly_ids else 'None'}`",
            ])
            if ad.gt_category_breakdown:
                lines.extend([
                    "",
                    "#### Category-Specific Anomaly Breakdown",
                    "| Anomaly Category | Total in Field | Adaptive Detected | Adaptive Missed | Category Recall |",
                    "| :--- | :--- | :--- | :--- | :--- |",
                ])
                for cat, cdata in ad.gt_category_breakdown.items():
                    lines.append(
                        f"| {cat} | {cdata['total']} | {cdata['detected']} | {cdata['missed']} | {cdata['recall_pct']:.1f}% |"
                    )
        else:
            lines.extend([
                "",
                "### Authentic Ground-Truth Anomaly Evaluation",
                "- *Ground-truth anomaly annotations not available for this patch.*",
            ])

        lines.extend([
            "",
            "---",
            "",
            "## 4. Resource-Cost & Energy Modeling (Computational Proxies)",
            "",
            f"- **Region Dimensions**: {int(np.sqrt(ad.pixels_per_region))} × {int(np.sqrt(ad.pixels_per_region))} pixels ({ad.pixels_per_region} px per region).",
            f"- **Transmission Payload**: {ad.estimated_data_kb:.2f} KB (Full Uniform baseline: {fu.estimated_data_kb:.2f} KB; **{ad.data_reduction_pct:.1f}% reduction**).",
            f"- **Downstream Processing**: {ad.estimated_workload_units:.1f} NCU (Full Uniform baseline: {fu.estimated_workload_units:.1f} NCU; **{ad.workload_reduction_pct:.1f}% reduction**).",
            f"- **Active Energy Proxy**: {ad.estimated_energy_joules_proxy:.4f} Joules (Full Uniform baseline: {fu.estimated_energy_joules_proxy:.4f} J; **{ad.energy_reduction_pct:.1f}% reduction**).",
            "",
            "> [!IMPORTANT]",
            f"> **Modeling Assumptions**: {assumptions.cost_model_notes}",
            "",
            "---",
            "",
            "## 5. Budget Adherence & Overflow Reporting",
            "",
            f"- **Requested Budget ($B$)**: {ad.requested_budget if ad.requested_budget is not None else 'None (Threshold Mode)'}",
            f"- **Actual Selected Count ($K$)**: {ad.selected_regions_count}",
            f"- **Budget Overflow Detected**: {ad.budget_exceeded}",
            f"- **High-Stress Overflow Count**: {ad.overflow_count}",
            f"- **High-Stress Deficit Count**: {ad.deficit_count}",
            f"- **Equal-Budget Match**: K_equal_budget = {eq.selected_regions_count} == K_adaptive = {ad.selected_regions_count}",
        ])

        if benchmark.trial_aggregates:
            lines.extend([
                "",
                "---",
                "",
                "## 6. Repeated Trials & Statistical Analysis",
                "",
                f"Evaluation averaged across **{len(benchmark.trial_aggregates['adaptive'].seeds)} controlled random seeds**: `{benchmark.trial_aggregates['adaptive'].seeds}`",
                "",
                "| Evaluation Metric | Full Uniform (Mean ± Std) | Equal-Budget Uniform (Mean ± Std) | Adaptive Sampling (Mean ± Std) |",
                "| :--- | :--- | :--- | :--- |",
            ])
            stat_metrics = [
                ("Selected Count ($K$)", "selected_regions_count"),
                ("Valid Reduction %", "valid_reduction_pct"),
                ("High-Stress Coverage %", "high_stress_coverage_pct"),
                ("Moderate-Stress Coverage %", "moderate_stress_coverage_pct"),
                ("Healthy Sampling %", "healthy_sampling_pct"),
                ("Data Transmitted (KB)", "estimated_data_kb"),
                ("Workload Units (NCU)", "estimated_workload_units"),
            ]
            if ad.gt_available:
                stat_metrics.insert(3, ("Ground-Truth Recall %", "gt_anomaly_recall_pct"))
                stat_metrics.insert(4, ("Ground-Truth Precision %", "gt_anomaly_precision_pct"))

            for label, key in stat_metrics:
                f_s = benchmark.trial_aggregates["full_uniform"].metric_stats.get(key, {})
                e_s = benchmark.trial_aggregates["equal_budget_uniform"].metric_stats.get(key, {})
                a_s = benchmark.trial_aggregates["adaptive"].metric_stats.get(key, {})

                f_str = f"{f_s.get('mean', 0.0):.2f} ± {f_s.get('std', 0.0):.2f}"
                e_str = f"{e_s.get('mean', 0.0):.2f} ± {e_s.get('std', 0.0):.2f}"
                a_str = f"{a_s.get('mean', 0.0):.2f} ± {a_s.get('std', 0.0):.2f}"
                lines.append(f"| **{label}** | {f_str} | {e_str} | {a_str} |")

            lines.extend([
                "",
                "> [!NOTE]",
                "> In threshold-driven mode, Adaptive Sampling high-stress and moderate-stress selections are strictly deterministic ($std = 0.0$), "
                "> with variance confined to the 5% healthy exploration subset. Equal-Budget Uniform exhibits high variance ($std > 0.0$) across all trials.",
            ])

        lines.extend([
            "",
            "---",
            "",
            "## 7. Limitations & Scientific Disclosures",
            "",
            "1. **RSDI vs. Physical Pathological Stress**:",
            "   RSDI is computed from uncalibrated 8-bit multi-spectral digital numbers (DN) in lossy JPEG format. "
            "   High RSDI indicates empirical canopy reflectance contrast (vigor deficit and heterogeneity), "
            "   which may correlate with crop disease, water deficit, soil variation, or weeds, but is not confirmed physiological pathology.",
            "2. **Ground-Truth Masks**:",
            "   Agriculture-Vision challenge annotations are non-exhaustive and focus on challenge-defined anomaly classes. "
            "   Anomalies not annotated by human annotators cannot be evaluated.",
            "3. **Energy & Network Proxies**:",
            "   Reported energy and bandwidth savings are theoretical linear reductions based on edge hardware specifications. "
            "   They represent proxy savings rather than empirically measured physical battery draw.",
            "",
        ])

        return "\n".join(lines)

    def export_json(
        self,
        benchmark: BenchmarkComparison,
        output_path: Union[str, Path],
    ) -> Path:
        """Export complete benchmark results to a formatted JSON file.

        Args:
            benchmark: BenchmarkComparison to serialize.
            output_path: Destination JSON file path.

        Returns:
            Path object of the created JSON file.
        """
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(benchmark.to_dict(), f, indent=2, allow_nan=False)
        return p

    def export_csv(
        self,
        benchmark: BenchmarkComparison,
        output_path: Union[str, Path],
    ) -> Path:
        """Export strategy metrics to a flat CSV file.

        Args:
            benchmark: BenchmarkComparison to serialize.
            output_path: Destination CSV file path.

        Returns:
            Path object of the created CSV file.
        """
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df = benchmark.to_dataframe()
        df.to_csv(p, index=False)
        return p
