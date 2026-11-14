"""Visualization engine for Apache Spark distributed benchmarks and sensitivity analyses.

Generates publication-quality figures illustrating worker scaling performance,
startup overhead separation, processing throughput, and category overlap threshold sensitivity.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class SparkBenchmarkVisualizer:
    """Visualization engine for Spark performance benchmarks and sensitivity analyses."""

    def __init__(self, output_dir: Optional[Union[str, Path]] = None) -> None:
        """Initialize the visualizer."""
        self.output_dir = Path(output_dir) if output_dir else PROJECT_ROOT / "results" / "benchmarks"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Style configuration
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 14,
        })

    def plot_worker_scaling(
        self,
        benchmark_results: Dict[str, Any],
        save_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Generate 4-panel diagnostic chart of Spark worker scaling performance.

        Panels:
        1. Runtime Breakdown: Startup Overhead vs. Steady-State Computation.
        2. Processing Throughput (patches / second).
        3. Steady-State vs. Total Speedup relative to Single-Machine baseline.
        4. Cumulative Runtime vs Number of Worker Cores.

        Args:
            benchmark_results: Dictionary from benchmark_worker_scaling.
            save_path: Optional output path.

        Returns:
            Path to saved figure.
        """
        p = Path(save_path) if save_path else self.output_dir / "spark_worker_scaling_benchmark.png"

        sm_data = benchmark_results.get("single_machine", {})
        spark_data = benchmark_results.get("spark_configs", {})

        configs = ["Single\nMachine"] + list(spark_data.keys())
        startup_times = [0.0] + [spark_data[k].get("startup_overhead_sec", 0.0) for k in spark_data]
        steady_times = [sm_data.get("wall_clock_time_sec", 0.0)] + [
            spark_data[k].get("steady_state_time_sec", 0.0) for k in spark_data
        ]
        throughputs = [sm_data.get("throughput_patches_per_sec", 0.0)] + [
            spark_data[k].get("throughput_patches_per_sec", 0.0) for k in spark_data
        ]
        steady_speedups = [1.0] + [
            spark_data[k].get("speedup_steady_state_vs_single", 1.0) for k in spark_data
        ]
        total_speedups = [1.0] + [
            spark_data[k].get("speedup_total_vs_single", 1.0) for k in spark_data
        ]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "StressSense: Apache Spark Worker Scaling & Runtime Breakdown\n"
            "(Local Mode on Windows 11 AMD64, Preserving Exact Single-Machine Parity)",
            fontweight="bold",
            y=0.98,
        )

        # Panel 1: Stacked Bar Chart of Wall-Clock Runtime
        ax1 = axes[0, 0]
        x = np.arange(len(configs))
        width = 0.55
        p1 = ax1.bar(x, steady_times, width, label="Steady-State Compute", color="#1f77b4", edgecolor="black")
        p2 = ax1.bar(x, startup_times, width, bottom=steady_times, label="Spark Startup Overhead", color="#ff7f0e", edgecolor="black")
        ax1.set_ylabel("Runtime (seconds)")
        ax1.set_title("Runtime Breakdown: Startup vs. Steady-State", fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(configs)
        ax1.legend(loc="upper right")
        ax1.grid(axis="y", linestyle="--", alpha=0.5)

        for i, (st, su) in enumerate(zip(steady_times, startup_times)):
            tot = st + su
            ax1.text(i, tot + (max(steady_times) * 0.02), f"{tot:.1f}s", ha="center", va="bottom", fontsize=9, fontweight="bold")

        # Panel 2: Throughput
        ax2 = axes[0, 1]
        bars2 = ax2.bar(x, throughputs, width, color="#2ca02c", edgecolor="black")
        ax2.set_ylabel("Throughput (patches / second)")
        ax2.set_title("Processing Throughput (Steady-State)", fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(configs)
        ax2.grid(axis="y", linestyle="--", alpha=0.5)

        for bar in bars2:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2.0, h + 0.1, f"{h:.2f} p/s", ha="center", va="bottom", fontsize=9, fontweight="bold")

        # Panel 3: Speedup Comparison
        ax3 = axes[1, 0]
        x_spark = np.arange(len(configs))
        w = 0.35
        ax3.bar(x_spark - w / 2, steady_speedups, w, label="Steady-State Speedup", color="#3b528b", edgecolor="black")
        ax3.bar(x_spark + w / 2, total_speedups, w, label="End-to-End Speedup (inc. JVM init)", color="#5ec962", edgecolor="black")
        ax3.axhline(1.0, color="gray", linestyle="--", alpha=0.7, label="Single-Machine Baseline (1.0x)")
        ax3.set_ylabel("Speedup Factor (x)")
        ax3.set_title("Speedup vs Single-Machine Baseline", fontweight="bold")
        ax3.set_xticks(x_spark)
        ax3.set_xticklabels(configs)
        ax3.legend(loc="upper left")
        ax3.grid(axis="y", linestyle="--", alpha=0.5)

        # Panel 4: Worker Core Scaling Analysis
        ax4 = axes[1, 1]
        cores = [1, 2, 4]
        # Extract spark times for local[1], local[2], local[4]
        steady_by_core = [
            spark_data.get(f"local[{c}]", {}).get("steady_state_time_sec", 0.0) for c in cores
        ]
        ideal_by_core = [
            steady_by_core[0] / c if steady_by_core[0] > 0 else 0.0 for c in cores
        ]
        ax4.plot(cores, steady_by_core, marker="o", linewidth=2.5, color="#d62728", label="Measured Spark Steady-State")
        ax4.plot(cores, ideal_by_core, marker="s", linestyle="--", color="gray", label="Ideal Linear Speedup")
        ax4.set_xlabel("Configured Worker Cores")
        ax4.set_ylabel("Steady-State Time (seconds)")
        ax4.set_title("Core Scaling: Measured vs. Ideal Linear", fontweight="bold")
        ax4.set_xticks(cores)
        ax4.legend(loc="upper right")
        ax4.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        plt.savefig(p, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return p

    def plot_category_sensitivity(
        self,
        sensitivity_data: Dict[str, Any],
        save_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Generate category overlap threshold sensitivity plot across 0%, 5%, 10%, and 20%.

        Args:
            sensitivity_data: Dictionary from run_overlap_sensitivity_analysis.
            save_path: Optional output path.

        Returns:
            Path to saved figure.
        """
        p = Path(save_path) if save_path else self.output_dir / "overlap_threshold_sensitivity.png"

        # Extract thresholds and labels
        thresh_keys = list(sensitivity_data.keys())
        labels = [sensitivity_data[k].get("overlap_threshold_pct", k) for k in thresh_keys]
        thresholds_pct = [sensitivity_data[k].get("overlap_threshold", 0.0) * 100.0 for k in thresh_keys]

        # Extract categories
        all_cats: Set[str] = set()
        for k in thresh_keys:
            cats_dict = sensitivity_data[k].get("categories", {})
            for c in cats_dict:
                all_cats.add(c)
        sorted_cats = sorted(all_cats)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(
            "StressSense: Anomaly Category Overlap Threshold Sensitivity Analysis\n"
            "(Comparing 0% Any-Pixel vs 5%, 10% [Default], and 20% Spatial Overlap Filters)",
            fontweight="bold",
            y=0.98,
        )

        # Plot 1: Overall Anomaly Recall across Thresholds
        overall_recalls = [
            sensitivity_data[k].get("mean_adaptive_recall", 0.0) for k in thresh_keys
        ]
        ax1.plot(thresholds_pct, overall_recalls, marker="D", linewidth=2.5, markersize=8, color="#1f77b4", label="Overall Anomaly Recall (Adaptive)")
        ax1.axvline(10.0, color="#d62728", linestyle="--", linewidth=1.5, label="Default Operational Threshold (10%)")
        ax1.set_xlabel("Category Overlap Threshold (%)")
        ax1.set_ylabel("Ground-Truth Recall (%)")
        ax1.set_title("Overall Anomaly Recall vs. Overlap Filter", fontweight="bold")
        ax1.set_xticks(thresholds_pct)
        ax1.set_xticklabels(labels)
        ax1.set_ylim(0, 100)
        ax1.legend(loc="lower left")
        ax1.grid(True, linestyle="--", alpha=0.5)

        for x_val, y_val in zip(thresholds_pct, overall_recalls):
            ax1.text(x_val, y_val + 2.0, f"{y_val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

        # Plot 2: Per-Category Breakdown
        colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]
        for i, cat in enumerate(sorted_cats):
            cat_recalls = [
                sensitivity_data[k].get("categories", {}).get(cat, {}).get("adaptive_recall_mean", 0.0)
                for k in thresh_keys
            ]
            c = colors[i % len(colors)]
            ax2.plot(thresholds_pct, cat_recalls, marker="o", linewidth=1.8, label=cat, color=c)

        ax2.axvline(10.0, color="#d62728", linestyle="--", linewidth=1.5, label="Default 10%")
        ax2.set_xlabel("Category Overlap Threshold (%)")
        ax2.set_ylabel("Category Recall (%)")
        ax2.set_title("Category-Wise Recall Sensitivity", fontweight="bold")
        ax2.set_xticks(thresholds_pct)
        ax2.set_xticklabels(labels)
        ax2.set_ylim(0, 105)
        ax2.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        ax2.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plt.savefig(p, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return p
