"""Distributed Apache Spark evaluation engine for StressSense adaptive sampling.

This module provides a distributed evaluation framework using Apache Spark (PySpark)
that parallelizes patch-level evaluation across Spark partitions while guaranteeing
exact parity with the single-machine DatasetWideEvaluator.

SCIENTIFIC INTEGRITY & METHODOLOGICAL PRINCIPLES:
1. RSDI Heuristic Heuristics & Threshold Invariance:
   The Relative Spectral Difference Index (RSDI) is an uncalibrated heuristic
   spectral-contrast score calculated from 8-bit digital numbers:
   RSDI = (NIR - Red) / (NIR + Red + 1e-7).
   All thresholds (ndvi_target=0.32, ndvi_stressed=0.20, tau_low=0.35, tau_high=0.70)
   and category tiers (HIGH_STRESS, MODERATE_STRESS, HEALTHY, INVALID) are preserved
   without modification.
2. Ground-Truth Decoupling:
   Authentic Agriculture-Vision anomaly masks remain strictly post-hoc evaluation data.
   Sampling decisions are executed with zero access to ground truth. Unlabeled test
   patches are strictly excluded from ground-truth metric denominators.
3. Strict Determinism and Equal-Budget Fairness:
   Every patch evaluates identical pseudo-random seeds. Equal-budget uniform sampling
   is allocated the exact same selection budget K as adaptive sampling (K_equal == K_adaptive).
4. Linear Energy & Resource Proxy Separation:
   Computational and transmission proxy energy (E_tx + E_compute) is strictly
   separated from actual Spark execution time and hardware battery consumption.
5. No Unfounded Speedup Claims:
   Distributed runtime and speedup are measured empirically against single-machine
   execution. Startup overhead is reported separately from steady-state processing time.
"""

from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats

from src.adaptive.evaluator import (
    AdaptiveEvaluator,
    BenchmarkComparison,
    EvaluationMetrics,
    ResourceCostAssumptions,
)
from src.adaptive.grid_generator import GridGenerator, GridRegion
from src.adaptive.sampling_engine import (
    BudgetOverflowPolicy,
    SamplingStrategy,
)
from src.adaptive.stress_scorer import RegionStressScore, StressCategory, StressScorer
from src.evaluation.dataset_evaluator import (
    DatasetWideEvaluator,
    HypothesisTestSummary,
    MetricDistribution,
    PatchRecord,
)

# Project root resolution
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class SparkEvaluatorConfig:
    """Configuration parameters for distributed Spark evaluation."""

    master: str = "local[2]"
    app_name: str = "StressSenseDistributed"
    driver_memory: str = "2g"
    executor_memory: str = "2g"
    num_partitions: Optional[int] = None
    grid_size: int = 32
    adaptive_budget: Optional[int] = None
    budget_ratio: Optional[float] = 0.35
    min_budget: int = 1
    ndvi_target: float = 0.32
    ndvi_stressed: float = 0.20
    overflow_policy: str = "guarantee_high_stress"
    exploration_rate: float = 0.05
    seed: int = 42
    n_trials: int = 5
    overlap_threshold: float = 0.10
    alpha: float = 0.01
    project_root: str = str(PROJECT_ROOT)
    patches_dir: str = str(PROJECT_ROOT / "data" / "processed" / "patches")
    manifest_path: str = str(PROJECT_ROOT / "data" / "processed" / "manifests" / "patch_manifest.csv")
    output_dir: str = str(PROJECT_ROOT / "results" / "benchmarks")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary format."""
        return asdict(self)


def _evaluate_patch_partition(
    iterator: Any,
    broadcast_cfg: Any,
) -> List[PatchRecord]:
    """Worker function to evaluate a partition of patch manifest rows.

    Executed on Spark worker executors across partitions. Re-instantiates
    sub-engines per partition to avoid cross-thread contention.

    Args:
        iterator: Iterable of row dictionaries or namedtuples from the manifest.
        broadcast_cfg: PySpark broadcast variable containing the config dictionary.

    Returns:
        List of PatchRecord objects evaluated within this partition.
    """
    cfg = broadcast_cfg.value
    project_root = Path(cfg["project_root"])
    patches_dir = Path(cfg["patches_dir"])

    # Map overflow policy string to enum
    policy_str = cfg.get("overflow_policy", "guarantee_high_stress")
    if policy_str == "strict_budget_cap":
        overflow_policy = BudgetOverflowPolicy.STRICT_BUDGET_CAP
    else:
        overflow_policy = BudgetOverflowPolicy.GUARANTEE_HIGH_STRESS

    # Instantiate partition-local sub-engines once per partition
    grid_generator = GridGenerator(grid_size=cfg["grid_size"])
    stress_scorer = StressScorer(
        ndvi_target=cfg["ndvi_target"],
        ndvi_stressed=cfg["ndvi_stressed"],
    )
    evaluator = AdaptiveEvaluator(
        default_seed=cfg["seed"],
    )

    records: List[PatchRecord] = []

    for row in iterator:
        # Convert row to dictionary if needed
        if hasattr(row, "asDict"):
            row_dict = row.asDict()
        elif isinstance(row, dict):
            row_dict = row
        else:
            row_dict = dict(row)

        patch_id = str(row_dict["patch_id"])
        split = str(row_dict["split"])
        field_id = str(row_dict["field_id"])
        has_anomaly = bool(row_dict["has_anomaly"])

        # Resolve image file paths
        rgb_path = project_root / str(row_dict["rgb_patch_path"])
        nir_path = project_root / str(row_dict["nir_patch_path"])
        msk_path = project_root / str(row_dict["mask_patch_path"])
        bnd_path = project_root / str(row_dict["boundary_patch_path"])

        if not rgb_path.is_file() or not nir_path.is_file():
            continue

        rgb = np.array(Image.open(rgb_path))
        nir = np.array(Image.open(nir_path))
        msk = np.array(Image.open(msk_path)) if msk_path.is_file() else None
        bnd = np.array(Image.open(bnd_path)) if bnd_path.is_file() else None

        # Parse ground-truth categories
        cats: List[str] = []
        raw_cats = str(row_dict.get("anomaly_categories", "none"))
        if raw_cats and raw_cats.lower() != "none" and raw_cats.lower() != "nan":
            cats = [c.strip() for c in raw_cats.split(",") if c.strip()]

        # Load individual category masks and form composite anomaly mask
        gt_mask: Optional[np.ndarray] = None
        category_masks: Dict[str, np.ndarray] = {}

        if split != "test":
            split_label_dir = patches_dir / split / "labels"
            for cat in cats:
                cat_file = split_label_dir / cat / f"{patch_id}.png"
                if cat_file.is_file():
                    c_arr = np.array(Image.open(cat_file))
                    category_masks[cat] = c_arr
                    bin_c = c_arr > 0
                    if gt_mask is None:
                        gt_mask = bin_c
                    else:
                        gt_mask = gt_mask | bin_c

        gt_mask_uint8 = (gt_mask.astype(np.uint8) * 255) if gt_mask is not None else None

        # 1. Generate grid regions
        regions = grid_generator.generate_grid(
            image_shape=rgb.shape[:2],
            validity_mask=msk,
            boundary_mask=bnd,
        )

        # 2. Score stress across all regions with region-specific category mapping
        overlap_thresh = float(cfg.get("overlap_threshold", 0.10))
        scores = stress_scorer.score_all_regions(
            rgb=rgb,
            nir=nir,
            regions=regions,
            validity_mask=msk,
            boundary_mask=bnd,
            ground_truth_mask=gt_mask_uint8,
            ground_truth_categories=cats,
            category_masks=category_masks,
            category_overlap_threshold=overlap_thresh,
        )

        valid_count = sum(1 for s in scores if s.is_valid)

        # Determine adaptive budget
        adaptive_budget = cfg.get("adaptive_budget")
        budget_ratio = cfg.get("budget_ratio", 0.35)
        min_budget = cfg.get("min_budget", 1)

        if adaptive_budget is not None:
            patch_budget = adaptive_budget
        elif budget_ratio is not None and valid_count > 0:
            patch_budget = max(min_budget, int(math.ceil(budget_ratio * valid_count)))
        else:
            patch_budget = None

        # 3. Compare strategies across n_trials pseudo-random seeds
        n_trials = cfg.get("n_trials", 5)
        base_seed = cfg.get("seed", 42)
        seeds = [base_seed + i * 37 for i in range(n_trials)]

        comparison = evaluator.run_repeated_trials(
            scores=scores,
            adaptive_budget=patch_budget,
            overflow_policy=overflow_policy,
            healthy_exploration_rate=cfg.get("exploration_rate", 0.05),
            seeds=seeds,
            patch_id=patch_id,
        )

        ad = comparison.strategy_metrics["adaptive"]
        eq = comparison.strategy_metrics["equal_budget_uniform"]
        fu = comparison.strategy_metrics["full_uniform"]

        # Symmetric trial-averaged aggregation across seeds
        if comparison.trial_aggregates and "adaptive" in comparison.trial_aggregates:
            ad_stats = comparison.trial_aggregates["adaptive"].metric_stats
            ad_high_pct = ad_stats.get("high_stress_coverage_pct", {}).get("mean", ad.high_stress_coverage_pct)
            ad_mod_pct = ad_stats.get("moderate_stress_coverage_pct", {}).get("mean", ad.moderate_stress_coverage_pct)
        else:
            ad_high_pct = ad.high_stress_coverage_pct
            ad_mod_pct = ad.moderate_stress_coverage_pct

        if comparison.trial_aggregates and "equal_budget_uniform" in comparison.trial_aggregates:
            eq_stats = comparison.trial_aggregates["equal_budget_uniform"].metric_stats
            eq_high_pct = eq_stats.get("high_stress_coverage_pct", {}).get("mean", eq.high_stress_coverage_pct)
            eq_mod_pct = eq_stats.get("moderate_stress_coverage_pct", {}).get("mean", eq.moderate_stress_coverage_pct)
            eq_rec_raw = eq_stats.get("gt_anomaly_recall_pct", {}).get("mean", None)
            eq_prec_raw = eq_stats.get("gt_anomaly_precision_pct", {}).get("mean", None)
        else:
            eq_high_pct = eq.high_stress_coverage_pct
            eq_mod_pct = eq.moderate_stress_coverage_pct
            eq_rec_raw = eq.gt_anomaly_recall_pct
            eq_prec_raw = eq.gt_anomaly_precision_pct

        # Category-specific recall breakdown
        cat_recalls_ad: Dict[str, float] = {}
        cat_recalls_eq: Dict[str, float] = {}
        cat_regs_tot: Dict[str, int] = {}
        cat_regs_ad: Dict[str, int] = {}
        cat_regs_eq: Dict[str, float] = {}

        if ad.gt_category_breakdown:
            for cat, data in ad.gt_category_breakdown.items():
                cat_recalls_ad[cat] = float(data.get("recall_pct", 0.0))
                cat_regs_tot[cat] = int(data.get("total", 0))
                cat_regs_ad[cat] = int(data.get("detected", 0))

        if eq.gt_category_breakdown:
            for cat, data in eq.gt_category_breakdown.items():
                cat_recalls_eq[cat] = float(data.get("recall_pct", 0.0))
                cat_regs_eq[cat] = float(data.get("detected", 0.0))

        # Ground-truth metrics: Strictly None for unlabeled test split or zero-anomaly patches
        has_valid_gt_anomalies = (
            split != "test"
            and ad.gt_available
            and ad.total_gt_anomalies > 0
        )

        if has_valid_gt_anomalies:
            ad_recall_val: Optional[float] = float(ad.gt_anomaly_recall_pct) if ad.gt_anomaly_recall_pct is not None else 0.0
            eq_recall_val: Optional[float] = float(eq_rec_raw) if eq_rec_raw is not None else 0.0
            ad_prec_val: Optional[float] = float(ad.gt_anomaly_precision_pct) if ad.gt_anomaly_precision_pct is not None else 0.0
            eq_prec_val: Optional[float] = float(eq_prec_raw) if eq_prec_raw is not None else 0.0
            gt_anom_count = ad.total_gt_anomalies
        else:
            ad_recall_val = None
            eq_recall_val = None
            ad_prec_val = None
            eq_prec_val = None
            gt_anom_count = 0

        # Exact linear proxy energy parity
        exact_energy_full = fu.estimated_energy_joules_proxy
        exact_energy_adaptive = ad.estimated_energy_joules_proxy
        exact_energy_equal = ad.estimated_energy_joules_proxy

        energy_red_pct = (
            ((exact_energy_full - exact_energy_adaptive) / exact_energy_full * 100.0)
            if exact_energy_full > 0
            else 0.0
        )

        latency_ad = ad.selected_regions_count * 0.015
        latency_fu = fu.selected_regions_count * 0.015

        selected_reg_ids = ad.metadata.get("selected_region_ids", []) if ad.metadata else []

        record = PatchRecord(
            patch_id=patch_id,
            split=split,
            field_id=field_id,
            has_anomaly=has_anomaly,
            anomaly_categories=cats,
            total_regions=ad.total_regions,
            valid_regions=ad.valid_regions,
            invalid_regions=ad.invalid_regions,
            k_selected=ad.selected_regions_count,
            full_selected=fu.selected_regions_count,
            valid_reduction_pct=ad.valid_reduction_pct,
            sampling_reduction_pct=ad.sampling_reduction_pct,
            high_stress_total=ad.high_stress_total,
            high_stress_adaptive_pct=ad_high_pct,
            high_stress_equal_pct=eq_high_pct,
            mod_stress_total=ad.moderate_stress_total,
            mod_stress_adaptive_pct=ad_mod_pct,
            mod_stress_equal_pct=eq_mod_pct,
            healthy_total=ad.healthy_total,
            healthy_explored_count=ad.healthy_selected,
            gt_available=has_valid_gt_anomalies,
            gt_anomalous_regions=gt_anom_count,
            adaptive_recall_pct=ad_recall_val,
            equal_recall_pct=eq_recall_val,
            adaptive_precision_pct=ad_prec_val,
            equal_precision_pct=eq_prec_val,
            energy_total_full_j=exact_energy_full,
            energy_total_equal_j=exact_energy_equal,
            energy_total_adaptive_j=exact_energy_adaptive,
            energy_reduction_pct=energy_red_pct,
            latency_adaptive_sec=latency_ad,
            latency_full_sec=latency_fu,
            category_recalls_adaptive=cat_recalls_ad,
            category_recalls_equal=cat_recalls_eq,
            category_regions_total=cat_regs_tot,
            category_regions_detected_adaptive=cat_regs_ad,
            category_regions_detected_equal=cat_regs_eq,
            selected_region_ids=selected_reg_ids,
        )
        records.append(record)

    return records


class SparkDistributedEvaluator:
    """Distributed evaluator utilizing Apache Spark for fleet-scale aerial crop health monitoring.

    Executes parallel evaluation across worker partitions while enforcing strict parity
    with the single-machine DatasetWideEvaluator.
    """

    def __init__(
        self,
        config: Optional[SparkEvaluatorConfig] = None,
        spark: Optional[Any] = None,
    ) -> None:
        """Initialize the Spark distributed evaluator.

        Args:
            config: SparkEvaluatorConfig instance (defaults to standard config).
            spark: Existing active SparkSession instance (optional).
        """
        self.config = config or SparkEvaluatorConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._spark = spark

    def get_or_create_spark_session(self, master: Optional[str] = None) -> Any:
        """Retrieve or create an active local SparkSession with required network and memory settings.

        Args:
            master: Spark master URL (e.g. 'local[1]', 'local[2]', 'local[4]').

        Returns:
            Active SparkSession instance.
        """
        if self._spark is not None and not self._spark.sparkContext._jsc.sc().isStopped():
            return self._spark

        # Ensure environment variables for Python and Java
        if "JAVA_HOME" not in os.environ and os.path.exists(r"C:\Java\jdk-17"):
            os.environ["JAVA_HOME"] = r"C:\Java\jdk-17"
        os.environ["PYSPARK_PYTHON"] = sys.executable
        os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

        from pyspark.sql import SparkSession

        target_master = master or self.config.master

        builder = (
            SparkSession.builder.appName(self.config.app_name)
            .master(target_master)
            .config("spark.driver.host", "127.0.0.1")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .config("spark.ui.enabled", "false")
            .config("spark.driver.memory", self.config.driver_memory)
            .config("spark.executor.memory", self.config.executor_memory)
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        )

        self._spark = builder.getOrCreate()
        return self._spark

    def stop_spark_session(self) -> None:
        """Stop active SparkSession safely."""
        if self._spark is not None:
            try:
                self._spark.stop()
            except Exception:
                pass
            self._spark = None

    def evaluate_distributed(
        self,
        splits: Optional[Sequence[str]] = None,
        field_ids: Optional[Sequence[str]] = None,
        max_patches: Optional[int] = None,
        num_partitions: Optional[int] = None,
        master: Optional[str] = None,
    ) -> Tuple[List[PatchRecord], Dict[str, Any]]:
        """Execute distributed evaluation across Spark partitions.

        Measures startup overhead separately from steady-state distributed processing.

        Args:
            splits: Filter to specific splits (e.g. ['train', 'val']).
            field_ids: Filter to specific field IDs.
            max_patches: Cap on patches to evaluate.
            num_partitions: Number of Spark RDD partitions.
            master: Master URL override (e.g. 'local[4]').

        Returns:
            Tuple of (List of PatchRecord objects, telemetry metrics dict).
        """
        t_init_start = time.perf_counter()

        # 1. Initialize SparkSession (measure startup overhead)
        spark = self.get_or_create_spark_session(master=master)
        sc = spark.sparkContext
        t_spark_ready = time.perf_counter()
        startup_overhead_sec = t_spark_ready - t_init_start

        # 2. Load manifest on driver
        manifest_p = Path(self.config.manifest_path)
        if not manifest_p.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest_p}")

        df = pd.read_csv(manifest_p)
        if splits:
            df = df[df["split"].isin(splits)]
        if field_ids:
            df = df[df["field_id"].isin(field_ids)]
        if max_patches is not None and max_patches > 0:
            df = df.head(max_patches)

        df = df.reset_index(drop=True)
        total_patches_to_eval = len(df)

        if total_patches_to_eval == 0:
            return [], {
                "startup_overhead_sec": round(startup_overhead_sec, 4),
                "steady_state_time_sec": 0.0,
                "total_wall_clock_sec": round(startup_overhead_sec, 4),
                "throughput_patches_per_sec": 0.0,
                "evaluated_patches": 0,
                "partitions": 0,
                "master": spark.conf.get("spark.master"),
            }

        # 3. Determine partitioning
        default_parallelism = sc.defaultParallelism
        parts = num_partitions or self.config.num_partitions or default_parallelism

        # 4. Convert manifest rows to list of dicts and parallelize
        records_manifest = df.to_dict(orient="records")
        rdd = sc.parallelize(records_manifest, numSlices=parts)

        # 5. Broadcast configuration
        broadcast_cfg = sc.broadcast(self.config.to_dict())

        # 6. Execute distributed processing (steady-state measurement)
        t_exec_start = time.perf_counter()
        partitioned_results = rdd.mapPartitions(
            lambda iterator: _evaluate_patch_partition(iterator, broadcast_cfg)
        ).collect()
        t_exec_end = time.perf_counter()

        steady_state_time_sec = t_exec_end - t_exec_start
        total_wall_clock_sec = t_exec_end - t_init_start
        throughput = (
            total_patches_to_eval / steady_state_time_sec
            if steady_state_time_sec > 0
            else 0.0
        )

        # Sort results deterministically by patch_id to ensure exact ordering
        sorted_records = sorted(partitioned_results, key=lambda r: r.patch_id)

        telemetry = {
            "startup_overhead_sec": round(startup_overhead_sec, 4),
            "steady_state_time_sec": round(steady_state_time_sec, 4),
            "total_wall_clock_sec": round(total_wall_clock_sec, 4),
            "throughput_patches_per_sec": round(throughput, 2),
            "evaluated_patches": len(sorted_records),
            "partitions": parts,
            "master": spark.conf.get("spark.master"),
        }

        return sorted_records, telemetry

    def compute_aggregates(self, records: Sequence[PatchRecord]) -> Dict[str, Any]:
        """Compute statistical distributions and aggregates matching single-machine evaluator."""
        evaluator_sm = DatasetWideEvaluator(
            manifest_path=self.config.manifest_path,
            patches_dir=self.config.patches_dir,
            output_dir=self.config.output_dir,
            grid_size=self.config.grid_size,
            adaptive_budget=self.config.adaptive_budget,
            budget_ratio=self.config.budget_ratio,
            ndvi_target=self.config.ndvi_target,
            ndvi_stressed=self.config.ndvi_stressed,
            exploration_rate=self.config.exploration_rate,
            seed=self.config.seed,
            n_trials=self.config.n_trials,
            overlap_threshold=self.config.overlap_threshold,
            alpha=self.config.alpha,
        )
        return evaluator_sm.compute_aggregates(records)

    def compute_statistical_tests(
        self, records: Sequence[PatchRecord]
    ) -> List[HypothesisTestSummary]:
        """Compute paired hypothesis tests matching single-machine evaluator."""
        evaluator_sm = DatasetWideEvaluator(
            manifest_path=self.config.manifest_path,
            patches_dir=self.config.patches_dir,
            output_dir=self.config.output_dir,
            grid_size=self.config.grid_size,
            adaptive_budget=self.config.adaptive_budget,
            budget_ratio=self.config.budget_ratio,
            ndvi_target=self.config.ndvi_target,
            ndvi_stressed=self.config.ndvi_stressed,
            exploration_rate=self.config.exploration_rate,
            seed=self.config.seed,
            n_trials=self.config.n_trials,
            overlap_threshold=self.config.overlap_threshold,
            alpha=self.config.alpha,
        )
        return evaluator_sm.compute_statistical_tests(records)

    def export_csv(
        self, records: Sequence[PatchRecord], path: Optional[Union[str, Path]] = None
    ) -> Path:
        """Export patch records to CSV."""
        p = Path(path) if path else self.output_dir / "spark_patch_metrics.csv"
        rows = [r.to_flat_dict() for r in records]
        df = pd.DataFrame(rows)
        df.to_csv(p, index=False)
        return p

    def export_json(
        self, summary: Dict[str, Any], path: Optional[Union[str, Path]] = None
    ) -> Path:
        """Export summary metrics to JSON."""
        p = Path(path) if path else self.output_dir / "spark_benchmark_summary.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return p

    def run_overlap_sensitivity_analysis(
        self,
        splits: Optional[Sequence[str]] = ("train", "val"),
        max_patches: Optional[int] = None,
        thresholds: Sequence[float] = (0.0, 0.05, 0.10, 0.20),
        master: Optional[str] = "local[2]",
    ) -> Dict[str, Any]:
        """Conduct category-overlap threshold sensitivity analysis.

        Evaluates overall anomaly recall and category-specific recalls across
        overlap thresholds: 0% (any positive pixel), 5%, 10% (default), and 20%.

        Args:
            splits: Splits to evaluate (defaults to train and val containing ground truth).
            max_patches: Optional patch limit.
            thresholds: Sequence of overlap fraction thresholds.
            master: Spark master URL.

        Returns:
            Dictionary containing sensitivity metrics across thresholds.
        """
        results: Dict[str, Any] = {}

        for thresh in thresholds:
            orig_thresh = self.config.overlap_threshold
            self.config.overlap_threshold = thresh

            records, telemetry = self.evaluate_distributed(
                splits=splits,
                max_patches=max_patches,
                master=master,
            )

            summary = self.compute_aggregates(records)

            thresh_key = f"thresh_{int(thresh * 100)}pct"
            rec_dist = summary.get("overall_distributions", {}).get("ground_truth_recall_adaptive_pct", {})
            cat_bd = summary.get("category_breakdown", {})

            results[thresh_key] = {
                "overlap_threshold": thresh,
                "overlap_threshold_pct": f"{int(thresh * 100)}%",
                "mean_adaptive_recall": rec_dist.get("mean", 0.0),
                "std_adaptive_recall": rec_dist.get("std", 0.0),
                "median_adaptive_recall": rec_dist.get("median", 0.0),
                "anomalous_patches_count": summary.get("anomalous_patches_count", 0),
                "categories": {
                    cat: {
                        "patch_count": cat_data.get("count", 0),
                        "qualifying_region_count": cat_data.get("qualifying_regions", 0),
                        "detected_regions_adaptive": cat_data.get("detected_regions_adaptive", 0),
                        "detected_regions_equal": cat_data.get("detected_regions_equal", 0.0),
                        "adaptive_recall_mean": cat_data.get("adaptive_recall", {}).get("mean", 0.0),
                        "equal_recall_mean": cat_data.get("equal_recall", {}).get("mean", 0.0),
                        "adaptive_recall_pooled_pct": cat_data.get("adaptive_recall_pooled_pct", 0.0),
                        "equal_recall_pooled_pct": cat_data.get("equal_recall_pooled_pct", 0.0),
                    }
                    for cat, cat_data in cat_bd.items()
                },
            }

            self.config.overlap_threshold = orig_thresh

        return results

    def benchmark_worker_scaling(
        self,
        worker_configs: Sequence[str] = ("local[1]", "local[2]", "local[4]"),
        splits: Optional[Sequence[str]] = None,
        max_patches: Optional[int] = 160,
    ) -> Dict[str, Any]:
        """Benchmark Spark runtime, throughput, and startup overhead across core configs.

        Compares single-machine execution against Spark local worker configurations.

        Args:
            worker_configs: Sequence of Spark master configurations to evaluate.
            splits: Splits to benchmark.
            max_patches: Number of patches per run.

        Returns:
            Dictionary with benchmark comparisons and speedups.
        """
        results: Dict[str, Any] = {
            "max_patches": max_patches,
            "single_machine": {},
            "spark_configs": {},
        }

        # 1. Single-machine baseline execution
        sm_evaluator = DatasetWideEvaluator(
            manifest_path=self.config.manifest_path,
            patches_dir=self.config.patches_dir,
            output_dir=self.config.output_dir,
            grid_size=self.config.grid_size,
            adaptive_budget=self.config.adaptive_budget,
            budget_ratio=self.config.budget_ratio,
            ndvi_target=self.config.ndvi_target,
            ndvi_stressed=self.config.ndvi_stressed,
            exploration_rate=self.config.exploration_rate,
            seed=self.config.seed,
            n_trials=self.config.n_trials,
            overlap_threshold=self.config.overlap_threshold,
        )

        t_sm_start = time.perf_counter()
        sm_records = sm_evaluator.evaluate_all(splits=splits, max_patches=max_patches)
        t_sm_end = time.perf_counter()
        sm_elapsed = t_sm_end - t_sm_start
        sm_throughput = len(sm_records) / sm_elapsed if sm_elapsed > 0 else 0.0

        results["single_machine"] = {
            "patches_evaluated": len(sm_records),
            "wall_clock_time_sec": round(sm_elapsed, 4),
            "throughput_patches_per_sec": round(sm_throughput, 2),
        }

        # 2. Spark worker configurations
        for master_cfg in worker_configs:
            self.stop_spark_session()
            records, telemetry = self.evaluate_distributed(
                splits=splits,
                max_patches=max_patches,
                master=master_cfg,
            )

            steady_time = telemetry["steady_state_time_sec"]
            total_time = telemetry["total_wall_clock_sec"]
            startup_time = telemetry["startup_overhead_sec"]

            steady_speedup = sm_elapsed / steady_time if steady_time > 0 else 0.0
            total_speedup = sm_elapsed / total_time if total_time > 0 else 0.0

            results["spark_configs"][master_cfg] = {
                "master": master_cfg,
                "patches_evaluated": len(records),
                "startup_overhead_sec": startup_time,
                "steady_state_time_sec": steady_time,
                "total_wall_clock_sec": total_time,
                "throughput_patches_per_sec": telemetry["throughput_patches_per_sec"],
                "speedup_steady_state_vs_single": round(steady_speedup, 2),
                "speedup_total_vs_single": round(total_speedup, 2),
            }

        return results
