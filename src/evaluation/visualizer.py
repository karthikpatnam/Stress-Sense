"""Diagnostic visualizer for dataset-wide experimental validation.

Generates publication-quality figures summarizing fleet-wide sampling efficiency,
stress-tier coverage distributions, category-wise anomaly recall, and proxy
resource cost reductions across the entire Agriculture-Vision dataset.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.dataset_evaluator import HypothesisTestSummary, PatchRecord

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DatasetVisualizer:
    """Publication-quality visualization engine for dataset-wide benchmarks."""

    def __init__(self, output_dir: Optional[Union[str, Path]] = None) -> None:
        """Initialize the dataset visualizer."""
        self.output_dir = Path(output_dir) if output_dir else PROJECT_ROOT / "results" / "experiments"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Matplotlib styling for high-contrast clarity
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

    def plot_metrics_summary(
        self,
        records: Sequence[PatchRecord],
        summary: Dict[str, Any],
        save_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Generate 4-panel diagnostic summary chart of dataset-wide performance.

        Panels:
        1. Valid Canopy Reduction Distribution (histogram of valid_reduction_pct).
        2. High-Stress Scored Region Recall (Adaptive vs. Equal-Budget).
        3. Ground-Truth Anomaly Recall (Adaptive vs. Equal-Budget).
        4. Cumulative Energy Proxy Expenditure (Full vs. Equal-Budget vs. Adaptive).

        Args:
            records: Evaluated patch records.
            summary: Summary dictionary from compute_aggregates.
            save_path: Optional explicit file path.

        Returns:
            Path to saved figure.
        """
        p = Path(save_path) if save_path else self.output_dir / "dataset_wide_metrics_summary.png"

        fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)
        fig.suptitle(
            f"StressSense: Fleet-Scale Benchmark Summary (N = {len(records)} Patches)",
            fontsize=15,
            fontweight="bold",
            y=0.98,
        )

        # -------------------------------------------------------------
        # Panel 1: Valid Canopy Reduction Distribution
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        valid_reds = [r.valid_reduction_pct for r in records]
        mean_vr = float(np.mean(valid_reds))
        median_vr = float(np.median(valid_reds))

        counts, bins, patches = ax1.hist(
            valid_reds,
            bins=15,
            color="#2b5c8f",
            edgecolor="white",
            alpha=0.85,
        )
        ax1.axvline(
            mean_vr,
            color="#d9534f",
            linestyle="--",
            linewidth=2.0,
            label=f"Mean: {mean_vr:.1f}%",
        )
        ax1.axvline(
            median_vr,
            color="#f0ad4e",
            linestyle=":",
            linewidth=2.0,
            label=f"Median: {median_vr:.1f}%",
        )
        ax1.set_title("A. Canopy Sampling Reduction Distribution", fontweight="bold")
        ax1.set_xlabel("Valid Canopy Sampling Reduction (valid_reduction_pct %)")
        ax1.set_ylabel("Number of Patches")
        ax1.grid(True, linestyle="--", alpha=0.4)
        ax1.legend(loc="upper left")

        # -------------------------------------------------------------
        # Panel 2: High-Stress Scored Region Recall Comparison
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        high_records = [r for r in records if r.high_stress_total > 0]
        ad_high = [r.high_stress_adaptive_pct for r in high_records]
        eq_high = [r.high_stress_equal_pct for r in high_records]

        box_data = [eq_high, ad_high]
        bplot = ax2.boxplot(
            box_data,
            tick_labels=["Equal-Budget\nUniform", "Adaptive\nSampling"],
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.5),
            widths=0.45,
        )
        colors = ["#f0ad4e", "#5cb85c"]
        for patch, color in zip(bplot["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)

        mean_eq_h = float(np.mean(eq_high))
        mean_ad_h = float(np.mean(ad_high))
        ax2.scatter([1, 2], [mean_eq_h, mean_ad_h], color="#d9534f", s=70, zorder=5, label="Mean")
        ax2.text(1.15, mean_eq_h, f"{mean_eq_h:.1f}%", fontweight="bold", color="#8a4b08")
        ax2.text(2.15, mean_ad_h, f"{mean_ad_h:.1f}%", fontweight="bold", color="#1b6d1b")

        ax2.set_title("B. High-Stress Region Recall Comparison", fontweight="bold")
        ax2.set_ylabel("High-Stress Coverage (%)")
        ax2.set_ylim(-5, 105)
        ax2.grid(True, linestyle="--", alpha=0.4)
        ax2.legend(loc="lower left")

        # -------------------------------------------------------------
        # Panel 3: Ground-Truth Anomaly Recall Comparison
        # -------------------------------------------------------------
        ax3 = axes[1, 0]
        gt_records = [r for r in records if r.gt_available and r.gt_anomalous_regions > 0]
        ad_rec = [r.adaptive_recall_pct for r in gt_records]
        eq_rec = [r.equal_recall_pct for r in gt_records]

        bplot_gt = ax3.boxplot(
            [eq_rec, ad_rec],
            tick_labels=["Equal-Budget\nUniform", "Adaptive\nSampling"],
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.5),
            widths=0.45,
        )
        for patch, color in zip(bplot_gt["boxes"], ["#f0ad4e", "#2b5c8f"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)

        mean_eq_r = float(np.mean(eq_rec))
        mean_ad_r = float(np.mean(ad_rec))
        ax3.scatter([1, 2], [mean_eq_r, mean_ad_r], color="#d9534f", s=70, zorder=5, label="Mean")
        ax3.text(1.15, mean_eq_r, f"{mean_eq_r:.1f}%", fontweight="bold", color="#8a4b08")
        ax3.text(2.15, mean_ad_r, f"{mean_ad_r:.1f}%", fontweight="bold", color="#143e69")

        ax3.set_title(f"C. Ground-Truth Anomaly Recall (N = {len(gt_records)} Patches)", fontweight="bold")
        ax3.set_ylabel("Anomaly Recall (%)")
        ax3.set_ylim(-5, 105)
        ax3.grid(True, linestyle="--", alpha=0.4)
        ax3.legend(loc="lower left")

        # -------------------------------------------------------------
        # Panel 4: Cumulative Proxy Energy Expenditure
        # -------------------------------------------------------------
        ax4 = axes[1, 1]
        totals = summary.get("totals", {})
        e_full = totals.get("cumulative_energy_full_j", 0.0)
        e_equal = totals.get("cumulative_energy_equal_j", 0.0)
        e_adapt = totals.get("cumulative_energy_adaptive_j", 0.0)
        savings_pct = totals.get("cumulative_energy_savings_pct", 0.0)

        strategies = ["Full Uniform\n(100% Scan)", "Equal-Budget\nUniform", "Adaptive\nSampling"]
        energies = [e_full, e_equal, e_adapt]
        bar_colors = ["#d9534f", "#f0ad4e", "#5cb85c"]

        bars = ax4.bar(strategies, energies, color=bar_colors, width=0.55, edgecolor="white", alpha=0.88)
        for b, e in zip(bars, energies):
            h = b.get_height()
            ax4.text(b.get_x() + b.get_width() / 2.0, h + (max(energies) * 0.015), f"{e:,.1f} J", ha="center", va="bottom", fontweight="bold")

        ax4.set_title(f"D. Cumulative Proxy Energy Expenditure (-{savings_pct:.1f}% Savings)", fontweight="bold")
        ax4.set_xlabel("Equal-Budget & Adaptive have identical selection budget K and energy", fontsize=9, style="italic")
        ax4.set_ylabel("Total Proxy Energy (Joules)")
        ax4.set_ylim(0, max(energies) * 1.15)
        ax4.grid(True, linestyle="--", alpha=0.4, axis="y")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(p, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return p

    def plot_category_recall_breakdown(
        self,
        summary: Dict[str, Any],
        save_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Generate horizontal bar chart comparing category-wise anomaly recall.

        Args:
            summary: Summary dictionary from compute_aggregates.
            save_path: Optional explicit file path.

        Returns:
            Path to saved figure.
        """
        p = Path(save_path) if save_path else self.output_dir / "category_recall_breakdown.png"

        cat_bd = summary.get("category_breakdown", {})
        categories = []
        ad_means = []
        ad_stds = []
        eq_means = []
        eq_stds = []
        counts = []

        for cat, data in sorted(cat_bd.items()):
            cnt = data.get("count", 0)
            if cnt > 0:
                categories.append(f"{cat}\n(N={cnt})")
                ad = data.get("adaptive_recall", {})
                eq = data.get("equal_recall", {})
                ad_means.append(ad.get("mean", 0.0))
                ad_stds.append(ad.get("std", 0.0))
                eq_means.append(eq.get("mean", 0.0))
                eq_stds.append(eq.get("std", 0.0))
                counts.append(cnt)

        if not categories:
            # Fallback if no anomalies present
            fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
            ax.text(0.5, 0.5, "No anomaly categories present in evaluated split", ha="center", va="center")
            fig.savefig(p, dpi=300)
            plt.close(fig)
            return p

        y = np.arange(len(categories))
        height = 0.35

        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

        rects1 = ax.barh(
            y - height / 2.0,
            eq_means,
            height,
            label="Equal-Budget Uniform",
            color="#f0ad4e",
            edgecolor="white",
            alpha=0.85,
        )
        rects2 = ax.barh(
            y + height / 2.0,
            ad_means,
            height,
            label="Adaptive Sampling",
            color="#2b5c8f",
            edgecolor="white",
            alpha=0.85,
        )

        # Label values on bars
        for r, val in zip(rects1, eq_means):
            ax.text(val + 1.0, r.get_y() + r.get_height() / 2.0, f"{val:.1f}%", va="center", fontsize=8, color="#8a4b08", fontweight="bold")
        for r, val in zip(rects2, ad_means):
            ax.text(val + 1.0, r.get_y() + r.get_height() / 2.0, f"{val:.1f}%", va="center", fontsize=8, color="#143e69", fontweight="bold")

        ax.set_title("Ground-Truth Anomaly Recall by Agriculture-Vision Category", fontweight="bold", fontsize=13)
        ax.set_xlabel("Anomaly Region Recall (%)")
        ax.set_yticks(y)
        ax.set_yticklabels(categories)
        ax.set_xlim(0, 115)
        ax.grid(True, linestyle="--", alpha=0.4, axis="x")
        ax.legend(loc="lower right")

        plt.tight_layout()
        fig.savefig(p, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return p
