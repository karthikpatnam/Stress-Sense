"""Visualization and demonstration engine for StressSense Phase 3: Adaptive Sampling.

Generates publication-quality diagnostic visualizations and benchmark summaries:
1. adaptive_pipeline_showcase.png:
   - Panel 1: Aerial RGB crop patch (128x128)
   - Panel 2: Near-Infrared (NIR) reflectance proxy
   - Panel 3: Heuristic Spectral-Contrast Map (RSDI) with colorbar
   - Panel 4: Spatial Grid & RSDI Priority Tiers (High, Moderate, Healthy, Invalid)
   - Panel 5: Adaptive Sampling Decision Map (Guarantee, Priority, Exploration)
2. strategy_comparison_side_by_side.png:
   - Side-by-side 3-way visual comparison:
     * Full-Coverage Uniform Sampling (100% valid baseline)
     * Equal-Budget Uniform Sampling (Random K regions)
     * Adaptive Sampling (Prioritized K regions)
3. ground_truth_detection_overlay.png:
   - Panel 1: Authentic ground-truth anomaly mask overlaid on crop patch
   - Panel 2: Adaptive sampling detection outcomes (True Positives vs Missed Anomalies)
   - Panel 3: Equal-budget uniform detection outcomes (visualizing random missed anomalies)
4. benchmark_metrics_summary.png:
   - 4-panel diagnostic chart comparing sampling efficiency, stress coverage,
     ground-truth recall/precision, and proxy resource-cost reductions.
5. BENCHMARK_REPORT.md, benchmark_report.json, and benchmark_report.csv:
   - Comprehensive machine-readable and human-readable benchmark deliverables.

SCIENTIFIC INTEGRITY & TERMINOLOGY DISCLOSURES:
- RSDI is an uncalibrated empirical heuristic spectral-contrast index, not calibrated NDVI.
- Stress tiers (High, Moderate, Healthy) are algorithmic priority classes for budget steering,
  not confirmed biological/physiological pathology labels.
- Ground-truth masks are strictly post-hoc diagnostic tools.
- Resource savings are modeled via computational proxies, not physical battery measurements.
"""

import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Non-interactive Matplotlib backend and local cache to avoid Windows permission issues
scratch_dir = PROJECT_ROOT / "scratch" / "matplotlib"
scratch_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(scratch_dir)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from PIL import Image

from src.adaptive.evaluator import (
    AdaptiveEvaluator,
    BenchmarkComparison,
    EvaluationMetrics,
    ResourceCostAssumptions,
)
from src.adaptive.grid_generator import GridGenerator, GridRegion
from src.adaptive.sampling_engine import (
    AdaptiveSampler,
    BudgetOverflowPolicy,
    SamplingDecision,
    SamplingEngine,
    SamplingResult,
    SamplingStrategy,
    UniformSampler,
)
from src.adaptive.stress_scorer import RegionStressScore, StressCategory, StressScorer


OUTPUT_DIR = PROJECT_ROOT / "results" / "adaptive"
DEFAULT_PATCH_ID = "13C8GTY66_1200-2200-1712-2712_p128_128"

# Distinct color palette for priority tiers and decisions
TIER_COLORS: Dict[str, str] = {
    "high_stress": "#D32F2F",     # Crimson Red
    "moderate_stress": "#F57C00", # Amber / Orange
    "healthy": "#388E3C",         # Forest Green
    "invalid": "#424242",         # Dark Charcoal Grey
}

REASON_COLORS: Dict[str, str] = {
    "high_stress_guarantee": "#D32F2F",   # Red
    "moderate_stress_priority": "#F57C00", # Orange
    "healthy_exploration": "#1976D2",      # Blue
    "equal_budget_uniform": "#7B1FA2",    # Purple
    "full_uniform": "#00796B",            # Teal
    "not_selected": "#9E9E9E",            # Light Grey
    "invalid_excluded": "#212121",        # Dark Black
}


class AdaptiveVisualizer:
    """Visualization generator for StressSense Adaptive Sampling."""

    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        default_seed: int = 42,
    ) -> None:
        """Initialize the visualizer.

        Args:
            output_dir: Directory where generated figures and reports are saved.
            default_seed: Pinned random seed for reproducible visual comparisons.
        """
        self.output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.default_seed = int(default_seed)
        self.grid_generator = GridGenerator(grid_size=32, stride=32)
        self.stress_scorer = StressScorer()
        self.sampling_engine = SamplingEngine(default_seed=self.default_seed)
        self.evaluator = AdaptiveEvaluator(default_seed=self.default_seed)

    def load_authentic_patch_data(
        self,
        patch_id: str = DEFAULT_PATCH_ID,
    ) -> Dict[str, Any]:
        """Load authentic RGB, NIR, boundary, mask, and ground-truth anomaly data for a patch.

        Args:
            patch_id: Identifier of the processed patch in data/processed/.

        Returns:
            Dictionary containing numpy arrays for all modalities and metadata.
        """
        patch_dir = PROJECT_ROOT / "data" / "processed" / "patches" / "train"
        manifest_csv = PROJECT_ROOT / "data" / "processed" / "manifests" / "patch_manifest.csv"

        rgb_path = patch_dir / "images" / "rgb" / f"{patch_id}.png"
        nir_path = patch_dir / "images" / "nir" / f"{patch_id}.png"
        msk_path = patch_dir / "masks" / f"{patch_id}.png"
        bnd_path = patch_dir / "boundaries" / f"{patch_id}.png"

        if not rgb_path.is_file():
            raise FileNotFoundError(f"RGB patch not found: {rgb_path}")

        rgb = np.array(Image.open(rgb_path))
        nir = np.array(Image.open(nir_path))
        msk = np.array(Image.open(msk_path)) if msk_path.is_file() else None
        bnd = np.array(Image.open(bnd_path)) if bnd_path.is_file() else None

        # Inspect manifest for category labels
        gt_categories: List[str] = []
        gt_mask: Optional[np.ndarray] = None

        if manifest_csv.is_file():
            df = pd.read_csv(manifest_csv)
            match = df[df["patch_id"] == patch_id]
            if len(match) > 0:
                row = match.iloc[0]
                cats = str(row.get("anomaly_categories", "none"))
                if cats and cats.lower() != "none":
                    gt_categories = [c.strip() for c in cats.split(",") if c.strip()]

        # Load first available category mask
        if gt_categories:
            primary_cat = gt_categories[0]
            cat_mask_path = patch_dir / "labels" / primary_cat / f"{patch_id}.png"
            if cat_mask_path.is_file():
                gt_mask = np.array(Image.open(cat_mask_path))

        return {
            "patch_id": patch_id,
            "rgb": rgb,
            "nir": nir,
            "validity_mask": msk,
            "boundary_mask": bnd,
            "ground_truth_mask": gt_mask,
            "ground_truth_categories": gt_categories,
        }

    def plot_pipeline_showcase(
        self,
        rgb: np.ndarray,
        nir: np.ndarray,
        regions: List[GridRegion],
        scores: List[RegionStressScore],
        adaptive_result: SamplingResult,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Generate 5-panel adaptive sampling pipeline showcase figure.

        Panels:
        1. RGB Aerial Crop Patch
        2. Near-Infrared (NIR) Band
        3. Empirical Heuristic Spectral-Contrast Map (RSDI)
        4. Spatial Grid & RSDI Priority Tiers
        5. Adaptive Sampling Decision Map
        """
        out_path = Path(save_path) if save_path else self.output_dir / "adaptive_pipeline_showcase.png"

        # Compute continuous RSDI map
        red_f, nir_f = self.stress_scorer.validate_and_align_inputs(rgb, nir)
        rsdi_map = self.stress_scorer.compute_ndvi_map(red_f, nir_f)

        fig, axes = plt.subplots(1, 5, figsize=(25, 5.5), dpi=300)

        # Panel 1: RGB Patch
        axes[0].imshow(rgb)
        axes[0].set_title("(1) RGB Aerial Crop Patch\n[128 × 128 px]", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        # Panel 2: NIR Band
        axes[1].imshow(nir, cmap="gray")
        axes[1].set_title("(2) Near-Infrared (NIR) Band\n[8-bit Digital Numbers]", fontsize=12, fontweight="bold")
        axes[1].axis("off")

        # Panel 3: RSDI Contrast Map
        im_rsdi = axes[2].imshow(rsdi_map, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
        axes[2].set_title("(3) Heuristic Spectral-Contrast (RSDI)\n[(NIR - Red) / (NIR + Red)]", fontsize=12, fontweight="bold")
        axes[2].axis("off")
        cbar = plt.colorbar(im_rsdi, ax=axes[2], orientation="horizontal", fraction=0.046, pad=0.08)
        cbar.set_label("RSDI Contrast Value", fontsize=9)

        # Panel 4: Spatial Grid & RSDI Priority Tiers
        axes[3].imshow(rgb)
        score_dict = {s.region_id: s for s in scores}
        for reg in regions:
            s_obj = score_dict.get(reg.region_id)
            cat = s_obj.stress_category.value if s_obj else "invalid"
            color = TIER_COLORS.get(cat, "#424242")
            rect = plt.Rectangle(
                (reg.x_min, reg.y_min),
                reg.width,
                reg.height,
                linewidth=1.8,
                edgecolor=color,
                facecolor=color,
                alpha=0.35 if cat != "invalid" else 0.65,
            )
            axes[3].add_patch(rect)
            # Add short score label in center
            if s_obj and s_obj.is_valid:
                axes[3].text(
                    reg.center_x,
                    reg.center_y,
                    f"{s_obj.combined_stress_score:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    fontweight="bold",
                    color="white",
                )

        axes[3].set_title("(4) Grid & Priority Tiers\n[Red: High | Orange: Mod | Green: Hly]", fontsize=12, fontweight="bold")
        axes[3].axis("off")

        # Panel 5: Adaptive Sampling Decision Map
        axes[4].imshow(rgb)
        selected_map = {d.region_id: d for d in adaptive_result.selected_regions if d.is_selected}

        for reg in regions:
            dec = selected_map.get(reg.region_id)
            if dec:
                reason = dec.selection_reason
                border_color = REASON_COLORS.get(reason, "#00E676")
                rect = plt.Rectangle(
                    (reg.x_min, reg.y_min),
                    reg.width,
                    reg.height,
                    linewidth=2.5,
                    edgecolor=border_color,
                    facecolor="none",
                )
                axes[4].add_patch(rect)
                # Label selection reason
                lbl = "HIGH" if "high" in reason else ("MOD" if "moderate" in reason else "EXPL")
                axes[4].text(
                    reg.center_x,
                    reg.center_y,
                    lbl,
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="heavy",
                    color="white",
                    bbox=dict(boxstyle="square,pad=0.15", facecolor=border_color, alpha=0.85, edgecolor="none"),
                )
            else:
                # Dim unselected regions
                dim_rect = plt.Rectangle(
                    (reg.x_min, reg.y_min),
                    reg.width,
                    reg.height,
                    facecolor="black",
                    alpha=0.60,
                )
                axes[4].add_patch(dim_rect)

        axes[4].set_title(
            f"(5) Adaptive Sampling Decisions\n[K={adaptive_result.actual_selected_count} Selected | {adaptive_result.valid_reduction_pct:.1f}% Savings]",
            fontsize=12,
            fontweight="bold",
        )
        axes[4].axis("off")

        # Add legend underneath panel 5
        legend_patches = [
            mpatches.Patch(color=REASON_COLORS["high_stress_guarantee"], label="High-Stress Guarantee"),
            mpatches.Patch(color=REASON_COLORS["moderate_stress_priority"], label="Moderate-Stress Priority"),
            mpatches.Patch(color=REASON_COLORS["healthy_exploration"], label="Healthy Exploration (5%)"),
            mpatches.Patch(color="#9E9E9E", label="Unselected (Transmission Dropped)"),
        ]
        axes[4].legend(
            handles=legend_patches,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=2,
            fontsize=8,
            frameon=True,
        )

        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def plot_strategy_comparison(
        self,
        rgb: np.ndarray,
        regions: List[GridRegion],
        benchmark: BenchmarkComparison,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Generate side-by-side 3-way visual comparison of all sampling strategies.

        Panels:
        1. Full-Coverage Uniform Sampling (100% valid baseline)
        2. Equal-Budget Uniform Sampling (Random K regions)
        3. Adaptive Sampling (RSDI-prioritized K regions)
        """
        out_path = Path(save_path) if save_path else self.output_dir / "strategy_comparison_side_by_side.png"

        full_m = benchmark.strategy_metrics["full_uniform"]
        equal_m = benchmark.strategy_metrics["equal_budget_uniform"]
        ad_m = benchmark.strategy_metrics["adaptive"]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), dpi=300)

        # Helper to draw strategy panel
        def _draw_strategy_panel(ax, metrics: EvaluationMetrics, title_prefix: str, primary_color: str):
            ax.imshow(rgb)
            selected_ids = {
                d["region_id"] if isinstance(d, dict) else d.region_id
                for d in metrics.metadata.get("selected_decisions", [])
                if (d.get("is_selected", False) if isinstance(d, dict) else d.is_selected)
            }
            # Fallback if selected_decisions not in metadata: derive from evaluator selected count
            # Use raw sampling decisions if available
            reg_dict = {r.region_id: r for r in regions}

            for reg in regions:
                is_sel = reg.region_id in selected_ids

                if is_sel:
                    rect = plt.Rectangle(
                        (reg.x_min, reg.y_min),
                        reg.width,
                        reg.height,
                        linewidth=2.2,
                        edgecolor=primary_color,
                        facecolor=primary_color,
                        alpha=0.30,
                    )
                    ax.add_patch(rect)
                    ax.text(
                        reg.center_x,
                        reg.center_y,
                        "✓",
                        ha="center",
                        va="center",
                        fontsize=12,
                        fontweight="bold",
                        color="white",
                    )
                else:
                    dim_rect = plt.Rectangle(
                        (reg.x_min, reg.y_min),
                        reg.width,
                        reg.height,
                        facecolor="black",
                        alpha=0.65,
                    )
                    ax.add_patch(dim_rect)

            ax.set_title(
                f"{title_prefix}\nSelected: {metrics.selected_regions_count}/{metrics.valid_regions} | "
                f"Canopy Savings: {metrics.valid_reduction_pct:.1f}%\n"
                f"High-Stress Coverage: {metrics.high_stress_coverage_pct:.1f}% | "
                f"Payload: {metrics.estimated_data_kb:.1f} KB",
                fontsize=11,
                fontweight="bold",
            )
            ax.axis("off")

        # 1. Full Uniform
        _draw_strategy_panel(
            axes[0],
            full_m,
            "(A) Full-Coverage Uniform (Reference)",
            primary_color="#00897B",
        )

        # 2. Equal-Budget Uniform
        _draw_strategy_panel(
            axes[1],
            equal_m,
            f"(B) Equal-Budget Uniform [Seed={equal_m.seed}]",
            primary_color="#8E24AA",
        )

        # 3. Adaptive Sampling
        _draw_strategy_panel(
            axes[2],
            ad_m,
            f"(C) Adaptive Sampling [Seed={ad_m.seed}]",
            primary_color="#D81B60",
        )

        plt.suptitle(
            f"StressSense Sampling Strategy Comparison – Patch {benchmark.patch_id}\n"
            f"Fair Budget Matched: K={ad_m.selected_regions_count} | High-Stress Delta: +{benchmark.comparison_summary.get('high_stress_coverage_delta', 0.0):.1f}% vs. Random",
            fontsize=13,
            fontweight="bold",
            y=0.98,
        )

        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def plot_ground_truth_detection_overlay(
        self,
        rgb: np.ndarray,
        regions: List[GridRegion],
        scores: List[RegionStressScore],
        gt_mask: np.ndarray,
        benchmark: BenchmarkComparison,
        category_name: str = "weed_cluster",
        save_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Generate ground-truth diagnostic visualization showing true positives vs missed anomalies.

        Panels:
        1. Authentic Ground-Truth Mask overlaid on RGB patch
        2. Adaptive Sampling Detection Outcomes (True Positives vs Missed Anomalies)
        3. Equal-Budget Uniform Detection Outcomes (Visualizing random missed anomalies)
        """
        out_path = Path(save_path) if save_path else self.output_dir / "ground_truth_detection_overlay.png"

        ad_m = benchmark.strategy_metrics["adaptive"]
        eq_m = benchmark.strategy_metrics["equal_budget_uniform"]

        # Anomaly region set
        gt_score_dict = {s.region_id: s for s in scores if s.is_valid and s.gt_has_anomaly}
        gt_region_ids = set(gt_score_dict.keys())

        fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), dpi=300)

        # Panel 1: Authentic Ground-Truth Mask Overlay
        axes[0].imshow(rgb)
        if gt_mask is not None and np.any(gt_mask > 0):
            # Create yellow/red translucent mask
            colored_mask = np.zeros((*gt_mask.shape[:2], 4), dtype=np.float32)
            colored_mask[gt_mask > 0] = [1.0, 0.2, 0.2, 0.65]  # Red overlay
            axes[0].imshow(colored_mask)

        # Draw grid outlines
        for reg in regions:
            rect = plt.Rectangle(
                (reg.x_min, reg.y_min),
                reg.width,
                reg.height,
                linewidth=1.0,
                edgecolor="white",
                facecolor="none",
                alpha=0.5,
            )
            axes[0].add_patch(rect)

        axes[0].set_title(
            f"(1) Authentic Anomaly Mask\nCategory: {category_name}\n"
            f"Anomalous Regions: {len(gt_region_ids)} / {len(regions)} cells",
            fontsize=11,
            fontweight="bold",
        )
        axes[0].axis("off")

        # Helper to plot anomaly detection outcome
        def _plot_detection_outcome(ax, metrics: EvaluationMetrics, title: str):
            ax.imshow(rgb)
            selected_ids = {
                d["region_id"] if isinstance(d, dict) else d.region_id
                for d in metrics.metadata.get("selected_decisions", [])
                if (d.get("is_selected", False) if isinstance(d, dict) else d.is_selected)
            }

            tp_count = 0
            fn_count = 0

            for reg in regions:
                has_gt = reg.region_id in gt_region_ids
                is_sel = reg.region_id in selected_ids

                if has_gt and is_sel:
                    # True Positive: Detected anomaly region!
                    tp_count += 1
                    rect = plt.Rectangle(
                        (reg.x_min, reg.y_min),
                        reg.width,
                        reg.height,
                        linewidth=2.8,
                        edgecolor="#00E676",  # Bright Green
                        facecolor="#00E676",
                        alpha=0.40,
                    )
                    ax.add_patch(rect)
                    ax.text(
                        reg.center_x,
                        reg.center_y,
                        "DETECTED",
                        ha="center",
                        va="center",
                        fontsize=7,
                        fontweight="heavy",
                        color="white",
                        bbox=dict(boxstyle="square,pad=0.15", facecolor="#00C853", alpha=0.9, edgecolor="none"),
                    )

                elif has_gt and not is_sel:
                    # False Negative: Missed anomaly!
                    fn_count += 1
                    rect = plt.Rectangle(
                        (reg.x_min, reg.y_min),
                        reg.width,
                        reg.height,
                        linewidth=3.0,
                        edgecolor="#FF1744",  # Bold Red Alert
                        facecolor="#FF1744",
                        alpha=0.45,
                        linestyle="--",
                    )
                    ax.add_patch(rect)
                    ax.text(
                        reg.center_x,
                        reg.center_y,
                        "MISSED!",
                        ha="center",
                        va="center",
                        fontsize=7,
                        fontweight="heavy",
                        color="white",
                        bbox=dict(boxstyle="square,pad=0.15", facecolor="#D50000", alpha=0.9, edgecolor="none"),
                    )

                elif not has_gt and is_sel:
                    # Clean region selected
                    rect = plt.Rectangle(
                        (reg.x_min, reg.y_min),
                        reg.width,
                        reg.height,
                        linewidth=1.8,
                        edgecolor="#29B6F6",
                        facecolor="none",
                    )
                    ax.add_patch(rect)

                else:
                    # Clean unselected -> dimmed
                    dim_rect = plt.Rectangle(
                        (reg.x_min, reg.y_min),
                        reg.width,
                        reg.height,
                        facecolor="black",
                        alpha=0.60,
                    )
                    ax.add_patch(dim_rect)

            rec_val = metrics.gt_anomalous_region_recall_pct or 0.0
            prec_val = metrics.gt_anomalous_region_precision_pct or 0.0
            ax.set_title(
                f"{title}\nDetected: {tp_count}/{len(gt_region_ids)} | Missed: {fn_count}\n"
                f"Region Recall: {rec_val:.1f}% | Precision: {prec_val:.1f}%",
                fontsize=11,
                fontweight="bold",
            )
            ax.axis("off")

        # Panel 2: Adaptive Sampling Anomaly Detection
        _plot_detection_outcome(
            axes[1],
            ad_m,
            "(2) Adaptive Sampling Detection Outcome",
        )

        # Panel 3: Equal-Budget Uniform Detection
        _plot_detection_outcome(
            axes[2],
            eq_m,
            "(3) Equal-Budget Uniform Detection Outcome",
        )

        # Custom Legend
        legend_handles = [
            mpatches.Patch(color="#00C853", label="Detected Anomaly Region (True Positive)"),
            mpatches.Patch(color="#D50000", label="Missed Anomaly Region (False Negative)"),
            mpatches.Patch(edgecolor="#29B6F6", facecolor="none", label="Surveillance Selection (Clean)"),
            mpatches.Patch(color="black", alpha=0.6, label="Unselected Normal Region"),
        ]
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.04),
            ncol=4,
            fontsize=9,
            frameon=True,
        )

        plt.suptitle(
            f"Ground-Truth Anomaly Detection Benchmark – Patch {benchmark.patch_id}",
            fontsize=13,
            fontweight="bold",
            y=0.98,
        )

        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def plot_benchmark_metrics_summary(
        self,
        benchmark: BenchmarkComparison,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Generate 4-panel diagnostic bar chart comparing all benchmark metrics.

        Panels:
        A. Sampling Efficiency Reductions (Valid Canopy % vs Total Grid %)
        B. Stress Tier RSDI Coverage % (High Stress vs Moderate Stress vs Healthy)
        C. Ground-Truth Anomaly Region Recall & Precision %
        D. Proxy Resource Costs (Payload in KB, Compute Workload in NCU, Energy Proxy in J)
        """
        out_path = Path(save_path) if save_path else self.output_dir / "benchmark_metrics_summary.png"

        ad = benchmark.strategy_metrics["adaptive"]
        eq = benchmark.strategy_metrics["equal_budget_uniform"]
        fu = benchmark.strategy_metrics["full_uniform"]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
        strategies = ["Full Uniform", "Equal-Budget", "Adaptive"]
        colors = ["#00796B", "#7B1FA2", "#D81B60"]

        # Panel A: Sampling Efficiency
        ax_a = axes[0, 0]
        x = np.arange(len(strategies))
        width = 0.35
        v_reds = [fu.valid_reduction_pct, eq.valid_reduction_pct, ad.valid_reduction_pct]
        t_reds = [fu.sampling_reduction_pct, eq.sampling_reduction_pct, ad.sampling_reduction_pct]

        r1 = ax_a.bar(x - width / 2, v_reds, width, label="Valid Canopy Savings (Primary)", color="#1976D2")
        r2 = ax_a.bar(x + width / 2, t_reds, width, label="Total-Grid Reduction (Secondary)", color="#90CAF9")
        ax_a.set_ylabel("Reduction Percentage (%)", fontsize=10, fontweight="bold")
        ax_a.set_title("(A) Sampling Efficiency Reductions", fontsize=11, fontweight="bold")
        ax_a.set_xticks(x)
        ax_a.set_xticklabels(strategies, fontsize=9, fontweight="bold")
        ax_a.set_ylim(0, 105)
        ax_a.legend(fontsize=8, loc="upper right")
        ax_a.grid(axis="y", linestyle="--", alpha=0.5)

        for rect in r1 + r2:
            h = rect.get_height()
            ax_a.annotate(
                f"{h:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # Panel B: Stress Tier Coverage
        ax_b = axes[0, 1]
        width_b = 0.25
        high_cov = [fu.high_stress_coverage_pct, eq.high_stress_coverage_pct, ad.high_stress_coverage_pct]
        mod_cov = [fu.moderate_stress_coverage_pct, eq.moderate_stress_coverage_pct, ad.moderate_stress_coverage_pct]
        hly_samp = [fu.healthy_sampling_pct, eq.healthy_sampling_pct, ad.healthy_sampling_pct]

        b1 = ax_b.bar(x - width_b, high_cov, width_b, label="High-Stress Coverage", color=TIER_COLORS["high_stress"])
        b2 = ax_b.bar(x, mod_cov, width_b, label="Moderate-Stress Coverage", color=TIER_COLORS["moderate_stress"])
        b3 = ax_b.bar(x + width_b, hly_samp, width_b, label="Healthy Baseline Rate", color=TIER_COLORS["healthy"])
        ax_b.set_ylabel("Coverage Percentage (%)", fontsize=10, fontweight="bold")
        ax_b.set_title("(B) RSDI Stress Tier Coverage", fontsize=11, fontweight="bold")
        ax_b.set_xticks(x)
        ax_b.set_xticklabels(strategies, fontsize=9, fontweight="bold")
        ax_b.set_ylim(0, 115)
        ax_b.legend(fontsize=8, loc="upper right")
        ax_b.grid(axis="y", linestyle="--", alpha=0.5)

        for rect in b1:
            h = rect.get_height()
            ax_b.annotate(
                f"{h:.0f}%",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

        # Panel C: Ground-Truth Anomaly Metrics
        ax_c = axes[1, 0]
        if ad.gt_available:
            rec = [fu.gt_anomalous_region_recall_pct or 0.0, eq.gt_anomalous_region_recall_pct or 0.0, ad.gt_anomalous_region_recall_pct or 0.0]
            prec = [fu.gt_anomalous_region_precision_pct or 0.0, eq.gt_anomalous_region_precision_pct or 0.0, ad.gt_anomalous_region_precision_pct or 0.0]
            c1 = ax_c.bar(x - width / 2, rec, width, label="Region-Level Anomaly Recall", color="#00897B")
            c2 = ax_c.bar(x + width / 2, prec, width, label="Region-Level Anomaly Precision", color="#80CBC4")
            ax_c.set_ylabel("Percentage (%)", fontsize=10, fontweight="bold")
            ax_c.set_title("(C) Ground-Truth Anomaly Region Metrics", fontsize=11, fontweight="bold")
            ax_c.set_xticks(x)
            ax_c.set_xticklabels(strategies, fontsize=9, fontweight="bold")
            ax_c.set_ylim(0, 115)
            ax_c.legend(fontsize=8, loc="upper right")
            ax_c.grid(axis="y", linestyle="--", alpha=0.5)
            for rect in c1 + c2:
                h = rect.get_height()
                ax_c.annotate(
                    f"{h:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        else:
            ax_c.text(0.5, 0.5, "Ground-Truth Not Available", ha="center", va="center", fontsize=12)
            ax_c.set_title("(C) Ground-Truth Anomaly Detection", fontsize=11, fontweight="bold")
            ax_c.axis("off")

        # Panel D: Proxy Resource Costs (KB & Energy)
        ax_d = axes[1, 1]
        data_kb = [fu.estimated_data_kb, eq.estimated_data_kb, ad.estimated_data_kb]
        energy_j = [fu.estimated_energy_joules_proxy, eq.estimated_energy_joules_proxy, ad.estimated_energy_joules_proxy]

        ax_d_energy = ax_d.twinx()
        d1 = ax_d.bar(x - width / 2, data_kb, width, label="Transmitted Payload (KB)", color="#E65100")
        d2 = ax_d_energy.bar(x + width / 2, energy_j, width, label="Active Energy Proxy (J)", color="#FFB74D")

        ax_d.set_ylabel("Payload Size (KB)", fontsize=10, fontweight="bold", color="#E65100")
        ax_d_energy.set_ylabel("Proxy Energy (Joules)", fontsize=10, fontweight="bold", color="#B26A00")
        ax_d.set_title("(D) Resource Payload & Computational Proxies", fontsize=11, fontweight="bold")
        ax_d.set_xticks(x)
        ax_d.set_xticklabels(strategies, fontsize=9, fontweight="bold")
        ax_d.grid(axis="y", linestyle="--", alpha=0.5)

        for rect in d1:
            h = rect.get_height()
            ax_d.annotate(
                f"{h:.1f} KB",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.suptitle(
            f"StressSense Benchmark Summary Telemetry – Patch {benchmark.patch_id}",
            fontsize=13,
            fontweight="bold",
        )

        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def generate_all_visualizations(
        self,
        patch_id: str = DEFAULT_PATCH_ID,
        adaptive_budget: int = 5,
        grid_size: int = 32,
        seed: int = 42,
    ) -> Dict[str, Path]:
        """Generate complete suite of Phase 3 Stage 5 visual and report artifacts.

        Args:
            patch_id: Patch ID to analyze.
            adaptive_budget: Desired sampling budget cap for demonstration.
            grid_size: Grid cell width/height (default 32).
            seed: Pinned random seed.

        Returns:
            Dictionary mapping artifact names to their created Path locations.
        """
        # 1. Load authentic multi-modal patch
        data = self.load_authentic_patch_data(patch_id)
        rgb = data["rgb"]
        nir = data["nir"]
        msk = data["validity_mask"]
        bnd = data["boundary_mask"]
        gt = data["ground_truth_mask"]
        gt_cats = data["ground_truth_categories"]

        # 2. Grid Generation
        gen = GridGenerator(grid_size=grid_size, stride=grid_size)
        regions = gen.generate_grid(
            image_shape=rgb.shape,
            image_id=patch_id,
            validity_mask=msk,
            boundary_mask=bnd,
        )

        # 3. Stress Scoring
        scorer = StressScorer()
        scores = scorer.score_all_regions(
            rgb=rgb,
            nir=nir,
            regions=regions,
            validity_mask=msk,
            boundary_mask=bnd,
            ground_truth_mask=gt,
            ground_truth_categories=gt_cats,
        )

        # 4. Sampling & Evaluation Benchmark
        evaluator = AdaptiveEvaluator(default_seed=seed)
        benchmark = evaluator.run_repeated_trials(
            scores=scores,
            adaptive_budget=adaptive_budget,
            seeds=[seed, seed + 1, seed + 2],
            patch_id=patch_id,
            pixels_per_region=grid_size * grid_size,
        )

        # In benchmark.strategy_metrics, ensure selected_decisions are embedded in metadata for plotting
        raw_results = evaluator.sampling_engine.compare_all_strategies(
            scores=scores,
            adaptive_budget=adaptive_budget,
            seed=seed,
        )
        for k in ["full_uniform", "equal_budget_uniform", "adaptive"]:
            benchmark.strategy_metrics[k].metadata["selected_decisions"] = [
                d.to_dict() for d in raw_results[k].selected_regions
            ]

        ad_res = raw_results["adaptive"]

        # 5. Generate Figures
        p1 = self.plot_pipeline_showcase(
            rgb=rgb,
            nir=nir,
            regions=regions,
            scores=scores,
            adaptive_result=ad_res,
        )

        p2 = self.plot_strategy_comparison(
            rgb=rgb,
            regions=regions,
            benchmark=benchmark,
        )

        primary_cat = gt_cats[0] if gt_cats else "none"
        p3 = self.plot_ground_truth_detection_overlay(
            rgb=rgb,
            regions=regions,
            scores=scores,
            gt_mask=gt,
            benchmark=benchmark,
            category_name=primary_cat,
        )

        p4 = self.plot_benchmark_metrics_summary(benchmark=benchmark)

        # 6. Export Machine-Readable Benchmark Outputs
        json_path = evaluator.export_json(benchmark, self.output_dir / "benchmark_report.json")
        csv_path = evaluator.export_csv(benchmark, self.output_dir / "benchmark_report.csv")

        # 7. Generate Formatted Markdown Report
        md_content = evaluator.generate_markdown_report(benchmark)
        md_path = self.output_dir / "BENCHMARK_REPORT.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        return {
            "pipeline_showcase": p1,
            "strategy_comparison": p2,
            "ground_truth_overlay": p3,
            "metrics_summary": p4,
            "benchmark_json": json_path,
            "benchmark_csv": csv_path,
            "benchmark_report_md": md_path,
        }


def main():
    """Command-line entry point to generate all Phase 3 visualizations."""
    visualizer = AdaptiveVisualizer()
    print("Generating StressSense Phase 3 Adaptive Sampling Visualizations...")
    artifacts = visualizer.generate_all_visualizations()
    print(f"Artifacts successfully created in: {visualizer.output_dir}")
    for name, path in artifacts.items():
        print(f" - {name}: {path.name} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
