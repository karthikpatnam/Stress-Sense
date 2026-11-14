"""Visualization Generator for StressSense Image Preprocessing and Patch Generation.

Generates:
1. preprocessing_pipeline_showcase.png:
   - Original RGB image (512x512)
   - Original NIR image (512x512)
   - Generated RGB patch (128x128)
   - Corresponding annotation mask (128x128)
   - Overlay of annotation mask on RGB patch (128x128)
2. patch_grid_decomposition.png:
   - Original image with 4x4 patch grid tiling and corresponding 16 extracted patches
3. multilabel_patch_overlays.png:
   - Detailed multi-label anomaly overlays on 128x128 patches
4. patch_statistics_summary.png:
   - Distribution of patches, anomalies, and splits
"""

import json
import os
from pathlib import Path
import sys
from typing import Dict, List, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Use scratch directory for Matplotlib cache to avoid Windows permission issues
scratch_dir = PROJECT_ROOT / "scratch" / "matplotlib"
scratch_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(scratch_dir)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.dataset.annotation_loader import OFFICIAL_ANOMALY_CATEGORIES


OUTPUT_DIR = PROJECT_ROOT / "results" / "preprocessing"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "Agriculture-Vision-2021"

# Distinct color palette for the 6 anomaly categories
CATEGORY_COLORS: Dict[str, Tuple[int, int, int]] = {
    "cloud_shadow": (60, 60, 60),      # Charcoal
    "double_plant": (0, 180, 255),     # Sky Blue
    "planter_skip": (255, 140, 0),     # Orange
    "standing_water": (30, 90, 240),   # Deep Blue
    "waterway": (0, 200, 120),         # Teal / Turquoise
    "weed_cluster": (220, 20, 60),     # Crimson Red
}


def create_patch_overlay(
    rgb_patch: np.ndarray,
    cat_masks: Dict[str, np.ndarray],
    alpha: float = 0.5,
) -> np.ndarray:
    """Blend multi-label color-coded anomaly masks onto an RGB patch."""
    overlay = rgb_patch.copy().astype(np.float32)
    colored_mask = np.zeros_like(overlay)
    has_any = np.zeros(rgb_patch.shape[:2], dtype=bool)

    for cat, mask in cat_masks.items():
        if np.any(mask > 0):
            color = CATEGORY_COLORS.get(cat, (255, 255, 0))
            active = mask > 0
            has_any = has_any | active
            for c in range(3):
                colored_mask[active, c] = color[c]

    overlay[has_any] = (1.0 - alpha) * overlay[has_any] + alpha * colored_mask[has_any]
    return np.clip(overlay, 0, 255).astype(np.uint8)


def plot_preprocessing_pipeline_showcase(sample_id: str, patch_y: int = 128, patch_x: int = 128) -> None:
    """Generate the required 5-panel pipeline visualization."""
    split_dir = RAW_DIR / "train"
    rgb_orig_path = split_dir / "images" / "rgb" / f"{sample_id}.jpg"
    nir_orig_path = split_dir / "images" / "nir" / f"{sample_id}.jpg"

    # Read original full images (512x512)
    bgr_orig = cv2.imread(str(rgb_orig_path))
    rgb_orig = cv2.cvtColor(bgr_orig, cv2.COLOR_BGR2RGB)
    nir_orig = cv2.imread(str(nir_orig_path), cv2.IMREAD_GRAYSCALE)

    # Patch paths
    patch_id = f"{sample_id}_p{patch_y:03d}_{patch_x:03d}"
    patch_split_dir = PROCESSED_DIR / "patches" / "train"
    rgb_patch_path = patch_split_dir / "images" / "rgb" / f"{patch_id}.png"
    nir_patch_path = patch_split_dir / "images" / "nir" / f"{patch_id}.png"

    bgr_patch = cv2.imread(str(rgb_patch_path))
    rgb_patch = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2RGB)
    nir_patch = cv2.imread(str(nir_patch_path), cv2.IMREAD_GRAYSCALE)

    # Load all category masks for this patch
    cat_masks: Dict[str, np.ndarray] = {}
    for cat in OFFICIAL_ANOMALY_CATEGORIES:
        c_path = patch_split_dir / "labels" / cat / f"{patch_id}.png"
        if c_path.is_file():
            cat_masks[cat] = cv2.imread(str(c_path), cv2.IMREAD_GRAYSCALE)

    # Find the primary active category or composite mask
    composite_mask = np.zeros((128, 128), dtype=np.uint8)
    active_labels = []
    for cat, m in cat_masks.items():
        if np.any(m > 0):
            composite_mask = np.maximum(composite_mask, m)
            active_labels.append(cat)

    active_title = ", ".join(active_labels) if active_labels else "Clean Farmland"

    # Create overlay
    overlay_patch = create_patch_overlay(rgb_patch, cat_masks, alpha=0.5)

    # 5-Panel Showcase Figure
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.8))

    # 1. Original RGB (with patch bounding box)
    rgb_with_box = rgb_orig.copy()
    cv2.rectangle(
        rgb_with_box,
        (patch_x, patch_y),
        (patch_x + 128, patch_y + 128),
        (255, 230, 0),
        3,
    )
    axes[0].imshow(rgb_with_box)
    axes[0].set_title("1. Original RGB Image (512x512)\n[Yellow Box: 128x128 Patch]", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    # 2. Original NIR
    im_nir = axes[1].imshow(nir_orig, cmap="viridis")
    axes[1].set_title("2. Original NIR Image (512x512)\n[Near-Infrared Reflectance]", fontsize=10, fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im_nir, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Generated RGB Patch
    axes[2].imshow(rgb_patch)
    axes[2].set_title(f"3. Generated RGB Patch\n(128x128, y={patch_y}, x={patch_x})", fontsize=10, fontweight="bold")
    axes[2].axis("off")

    # 4. Corresponding Annotation Mask
    im_mask = axes[3].imshow(composite_mask, cmap="plasma")
    axes[3].set_title(f"4. Annotation Mask (128x128)\n[{active_title}]", fontsize=10, fontweight="bold")
    axes[3].axis("off")
    plt.colorbar(im_mask, ax=axes[3], fraction=0.046, pad=0.04)

    # 5. Overlay on RGB Patch
    axes[4].imshow(overlay_patch)
    axes[4].set_title("5. Overlay on RGB Patch\n[Color-Coded Semantic Localization]", fontsize=10, fontweight="bold")
    axes[4].axis("off")

    plt.tight_layout()
    out_file = OUTPUT_DIR / "preprocessing_pipeline_showcase.png"
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_patch_grid_decomposition(sample_id: str) -> None:
    """Visualizes the 4x4 non-overlapping patch tiling of a 512x512 image."""
    split_dir = RAW_DIR / "train"
    rgb_orig_path = split_dir / "images" / "rgb" / f"{sample_id}.jpg"

    bgr_orig = cv2.imread(str(rgb_orig_path))
    rgb_orig = cv2.cvtColor(bgr_orig, cv2.COLOR_BGR2RGB)

    patch_split_dir = PROCESSED_DIR / "patches" / "train"

    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(4, 8)

    # Left: Full 512x512 image with grid lines
    ax_full = fig.add_subplot(gs[:, 0:4])
    grid_img = rgb_orig.copy()
    for coord in [128, 256, 384]:
        cv2.line(grid_img, (0, coord), (512, coord), (255, 255, 0), 2)
        cv2.line(grid_img, (coord, 0), (coord, 512), (255, 255, 0), 2)

    # Label grid indices
    for r, y in enumerate([0, 128, 256, 384]):
        for c, x in enumerate([0, 128, 256, 384]):
            cv2.putText(
                grid_img,
                f"P({r},{c})",
                (x + 10, y + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

    ax_full.imshow(grid_img)
    ax_full.set_title(f"Full Orthomosaic Tile (512x512)\n4x4 Non-Overlapping Grid (stride=128)\n{sample_id}", fontsize=12, fontweight="bold")
    ax_full.axis("off")

    # Right 4x4 subplots: Individual 16 patches
    for r, y in enumerate([0, 128, 256, 384]):
        for c, x in enumerate([0, 128, 256, 384]):
            pid = f"{sample_id}_p{y:03d}_{x:03d}"
            p_path = patch_split_dir / "images" / "rgb" / f"{pid}.png"
            ax_patch = fig.add_subplot(gs[r, 4 + c])
            if p_path.is_file():
                p_bgr = cv2.imread(str(p_path))
                p_rgb = cv2.cvtColor(p_bgr, cv2.COLOR_BGR2RGB)
                ax_patch.imshow(p_rgb)
            ax_patch.set_title(f"P({r},{c}): y={y},x={x}", fontsize=8)
            ax_patch.axis("off")

    plt.tight_layout()
    out_file = OUTPUT_DIR / "patch_grid_decomposition.png"
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_multilabel_patch_overlays(manifest_path: Path) -> None:
    """Showcases diverse anomaly patches including multi-label overlaps and healthy farmland."""
    df = pd.read_csv(manifest_path)
    train_df = df[df["split"] == "train"]

    # Select representative patches:
    # 1. Planter skip
    ps_patch = train_df[train_df["planter_skip_fraction"] > 0].iloc[0]
    # 2. Standing water
    sw_patch = train_df[train_df["standing_water_fraction"] > 0].iloc[0]
    # 3. Weed cluster
    wc_patch = train_df[train_df["weed_cluster_fraction"] > 0].iloc[0]
    # 4. Multi-label overlap
    overlap_patch = train_df[train_df["has_overlap"]].iloc[0]
    # 5. Healthy clean crop
    healthy_patch = train_df[(~train_df["has_anomaly"]) & (train_df["valid_pixel_fraction"] > 0.9)].iloc[0]

    selected = [
        ("Planter Skip", ps_patch),
        ("Standing Water", sw_patch),
        ("Weed Cluster", wc_patch),
        (f"Multi-Label: {overlap_patch['anomaly_categories']}", overlap_patch),
        ("Healthy Farmland", healthy_patch),
    ]

    fig, axes = plt.subplots(3, len(selected), figsize=(19, 9))

    for idx, (title, row) in enumerate(selected):
        pid = row["patch_id"]
        split = row["split"]
        patch_dir = PROCESSED_DIR / "patches" / split

        # RGB
        rgb_bgr = cv2.imread(str(patch_dir / "images" / "rgb" / f"{pid}.png"))
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

        # NIR
        nir = cv2.imread(str(patch_dir / "images" / "nir" / f"{pid}.png"), cv2.IMREAD_GRAYSCALE)

        # Labels
        cat_masks = {}
        for cat in OFFICIAL_ANOMALY_CATEGORIES:
            c_p = patch_dir / "labels" / cat / f"{pid}.png"
            if c_p.is_file():
                cat_masks[cat] = cv2.imread(str(c_p), cv2.IMREAD_GRAYSCALE)

        overlay = create_patch_overlay(rgb, cat_masks, alpha=0.55)

        # Row 0: RGB
        axes[0, idx].imshow(rgb)
        axes[0, idx].set_title(f"{title}\nRGB Patch (128x128)", fontsize=9, fontweight="bold")
        axes[0, idx].axis("off")

        # Row 1: NIR
        axes[1, idx].imshow(nir, cmap="viridis")
        axes[1, idx].set_title("NIR Reflectance", fontsize=9)
        axes[1, idx].axis("off")

        # Row 2: Overlay
        axes[2, idx].imshow(overlay)
        axes[2, idx].set_title("Multi-Label Overlay", fontsize=9, fontweight="bold")
        axes[2, idx].axis("off")

    plt.tight_layout()
    out_file = OUTPUT_DIR / "multilabel_patch_overlays.png"
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_patch_statistics_summary(manifest_path: Path) -> None:
    """Generate statistical publication-style bar chart of generated patches."""
    df = pd.read_csv(manifest_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # Split breakdown: total, anomaly, healthy
    splits = ["train", "val", "test"]
    x = np.arange(len(splits))
    width = 0.28

    total_counts = [len(df[df["split"] == s]) for s in splits]
    anomaly_counts = [len(df[(df["split"] == s) & (df["has_anomaly"])]) for s in splits]
    overlap_counts = [len(df[(df["split"] == s) & (df["has_overlap"])]) for s in splits]

    r1 = ax1.bar(x - width, total_counts, width, label="Total Patches", color="#1976D2")
    r2 = ax1.bar(x, anomaly_counts, width, label="Anomaly-Bearing", color="#FB8C00")
    r3 = ax1.bar(x + width, overlap_counts, width, label="Multi-Label Overlap", color="#E53935")

    ax1.set_ylabel("Number of Patches", fontsize=11, fontweight="bold")
    ax1.set_title("Generated Patches Breakdown by Split (128x128)", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.title() for s in splits], fontsize=11)
    ax1.legend(frameon=True)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in r1 + r2 + r3:
        h = bar.get_height()
        ax1.annotate(f"{int(h)}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    # Category frequencies across train and val
    cat_counts = [
        int((df[f"{cat}_fraction"] > 0).sum())
        for cat in OFFICIAL_ANOMALY_CATEGORIES
    ]
    clean_cat_labels = [c.replace("_", " ").title() for c in OFFICIAL_ANOMALY_CATEGORIES]
    colors = [
        f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"
        for c in [CATEGORY_COLORS[cat] for cat in OFFICIAL_ANOMALY_CATEGORIES]
    ]

    bars = ax2.bar(clean_cat_labels, cat_counts, color=colors, edgecolor="black", linewidth=0.8)
    ax2.set_ylabel("Number of Patches Containing Anomaly", fontsize=11, fontweight="bold")
    ax2.set_title("Anomaly Category Occurrences Across Patches", fontsize=12, fontweight="bold")
    ax2.set_xticklabels(clean_cat_labels, rotation=25, ha="right", fontsize=10)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax2.annotate(f"{int(h)}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out_file = OUTPUT_DIR / "patch_statistics_summary.png"
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_csv = PROCESSED_DIR / "manifests" / "patch_manifest.csv"

    if not manifest_csv.is_file():
        print(f"Error: Manifest {manifest_csv} not found. Run patch generation first.")
        return

    print("Generating preprocessing visualizations...", flush=True)
    sample_id = "KEP4MBVE7_1000-2000-1512-2512"

    plot_preprocessing_pipeline_showcase(sample_id, patch_y=128, patch_x=256)
    plot_patch_grid_decomposition(sample_id)
    plot_multilabel_patch_overlays(manifest_csv)
    plot_patch_statistics_summary(manifest_csv)

    print("\nAll preprocessing visualizations generated successfully!")


if __name__ == "__main__":
    main()
