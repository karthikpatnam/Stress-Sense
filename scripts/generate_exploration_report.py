"""Dataset exploration and visualization script for StressSense.

Generates:
- Statistical summary tables (dimensions, samples per split, annotation distribution)
- Visualizations saved to `results/dataset_report/`:
  1. sample_modalities_grid.png: Side-by-side comparison of RGB, NIR, Boundary, and Validity masks
  2. anomaly_categories_showcase.png: Visualizing each of the 6 official anomaly categories
  3. rgb_annotation_overlays.png: Color-coded segmentation masks overlaid onto aerial RGB imagery
  4. overlapping_anomalies_showcase.png: Demonstration of multi-label overlapping anomaly preservation
  5. class_distribution_chart.png: Publication-ready distribution plot of categories and overlap
"""

import json
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Use scratch directory for Matplotlib cache to prevent permission warnings
scratch_dir = PROJECT_ROOT / "scratch" / "matplotlib"
scratch_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(scratch_dir)

from typing import Dict, List, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.dataset.image_loader import load_rgb_image, load_nir_image
from src.dataset.annotation_loader import (
    load_boundary_mask,
    load_validity_mask,
    load_all_category_masks,
    OFFICIAL_ANOMALY_CATEGORIES,
)
from src.dataset.dataset_validator import validate_dataset_split


OUTPUT_DIR = Path("results/dataset_report")
DATA_DIR = Path("data/raw/Agriculture-Vision-2021")

# High-contrast color palette for the 6 anomaly classes (RGB)
CATEGORY_COLORS: Dict[str, Tuple[int, int, int]] = {
    "cloud_shadow": (60, 60, 60),      # Dark Charcoal
    "double_plant": (0, 180, 255),     # Sky Blue / Cyan
    "planter_skip": (255, 140, 0),     # Orange
    "standing_water": (30, 90, 240),   # Deep Blue
    "waterway": (0, 200, 120),         # Turquoise / Teal
    "weed_cluster": (220, 20, 60),     # Crimson Red
}


def create_overlay(
    rgb_img: np.ndarray,
    cat_masks: Dict[str, np.ndarray],
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend color-coded anomaly masks onto an RGB image."""
    overlay = rgb_img.copy().astype(np.float32)
    colored_mask = np.zeros_like(overlay)
    has_any = np.zeros(rgb_img.shape[:2], dtype=bool)

    for cat, mask in cat_masks.items():
        if np.any(mask > 0):
            color = CATEGORY_COLORS.get(cat, (255, 255, 0))
            active = mask > 0
            has_any = has_any | active
            for c in range(3):
                colored_mask[active, c] = color[c]

    # Blend where masks exist
    overlay[has_any] = (
        (1.0 - alpha) * overlay[has_any] + alpha * colored_mask[has_any]
    )
    return np.clip(overlay, 0, 255).astype(np.uint8)


def plot_modalities_grid(sample_id: str, split: str = "train") -> None:
    """Generate figure comparing RGB, NIR, Boundary, and Validity mask for a single sample."""
    split_dir = DATA_DIR / split
    rgb_path = split_dir / "images" / "rgb" / f"{sample_id}.jpg"
    nir_path = split_dir / "images" / "nir" / f"{sample_id}.jpg"
    bnd_path = split_dir / "boundaries" / f"{sample_id}.png"
    msk_path = split_dir / "masks" / f"{sample_id}.png"

    rgb = load_rgb_image(rgb_path)
    nir = load_nir_image(nir_path)
    bnd = load_boundary_mask(bnd_path)
    msk = load_validity_mask(msk_path)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    axes[0].imshow(rgb)
    axes[0].set_title(f"RGB Imagery (512x512)\n{sample_id}", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    im_nir = axes[1].imshow(nir, cmap="viridis")
    axes[1].set_title("Near-Infrared (NIR)\nHigh Foliage Reflectance", fontsize=11, fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im_nir, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(bnd, cmap="gray")
    axes[2].set_title("Field Boundary Mask\n(Valid Farmland Region)", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    axes[3].imshow(msk, cmap="gray")
    axes[3].set_title("Validity Mask\n(Sensor Coverage Swath)", fontsize=11, fontweight="bold")
    axes[3].axis("off")

    plt.tight_layout()
    out_file = OUTPUT_DIR / "sample_modalities_grid.png"
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_anomaly_categories_showcase(split: str = "train") -> None:
    """Visualizes each of the 6 individual official anomaly categories with matching masks."""
    split_dir = DATA_DIR / split
    labels_dir = split_dir / "labels"
    rgb_dir = split_dir / "images" / "rgb"

    # Find a sample containing each category
    samples_per_cat: Dict[str, str] = {}
    for cat in OFFICIAL_ANOMALY_CATEGORIES:
        cat_dir = labels_dir / cat
        for f in cat_dir.iterdir():
            if f.is_file():
                mask = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                if mask is not None and np.any(mask > 0):
                    samples_per_cat[cat] = f.stem
                    break

    fig, axes = plt.subplots(2, 6, figsize=(22, 7.5))

    for idx, cat in enumerate(OFFICIAL_ANOMALY_CATEGORIES):
        sid = samples_per_cat.get(cat)
        if sid:
            rgb = load_rgb_image(rgb_dir / f"{sid}.jpg")
            mask = cv2.imread(str(labels_dir / cat / f"{sid}.png"), cv2.IMREAD_GRAYSCALE)
        else:
            rgb = np.zeros((512, 512, 3), dtype=np.uint8)
            mask = np.zeros((512, 512), dtype=np.uint8)

        # Row 0: Aerial RGB Image
        axes[0, idx].imshow(rgb)
        axes[0, idx].set_title(f"{cat.replace('_', ' ').title()}\n(RGB Context)", fontsize=10, fontweight="bold")
        axes[0, idx].axis("off")

        # Row 1: Binary Ground Truth Mask
        axes[1, idx].imshow(mask, cmap="plasma")
        axes[1, idx].set_title(f"Ground Truth Mask\n({cat})", fontsize=10)
        axes[1, idx].axis("off")

    plt.tight_layout()
    out_file = OUTPUT_DIR / "anomaly_categories_showcase.png"
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_overlays_and_multilabel(split: str = "train") -> None:
    """Generate side-by-side color-coded overlays showcasing multi-label overlapping anomalies."""
    split_dir = DATA_DIR / split
    labels_dir = split_dir / "labels"
    rgb_dir = split_dir / "images" / "rgb"

    # Select representative samples: 2 single-label, 2 multi-label overlapping, 1 healthy
    selected_samples = [
        ("Single: Cloud Shadow", "KEP4MBVE7_1000-2000-1512-2512"),
        ("Single: Standing Water", "911VK8KRX_4000-5000-4512-5512"),
        ("Overlapping: Cloud Shadow + Weed", "8VBADR4KG_1100-2100-1612-2612"),
        ("Overlapping: Water + Waterway", "JYR8H78YW_2100-3100-2612-3612"),
        ("Healthy Farmland (No Anomaly)", "LLV7JDILE_1500-2500-2012-3012"),
    ]

    fig, axes = plt.subplots(2, len(selected_samples), figsize=(20, 8))

    for idx, (label_title, sid) in enumerate(selected_samples):
        rgb_file = rgb_dir / f"{sid}.jpg"
        if not rgb_file.is_file():
            continue

        rgb = load_rgb_image(rgb_file)
        masks = load_all_category_masks(labels_dir, sid)
        overlay = create_overlay(rgb, masks, alpha=0.5)

        # Raw RGB
        axes[0, idx].imshow(rgb)
        axes[0, idx].set_title(f"{label_title}\nRGB Imagery", fontsize=10, fontweight="bold")
        axes[0, idx].axis("off")

        # Overlay
        axes[1, idx].imshow(overlay)
        active_cats = [c for c, m in masks.items() if np.any(m > 0)]
        active_str = ", ".join(active_cats) if active_cats else "None (Healthy)"
        axes[1, idx].set_title(f"Annotated Overlay\n[{active_str}]", fontsize=9)
        axes[1, idx].axis("off")

    plt.tight_layout()
    out_file = OUTPUT_DIR / "rgb_annotation_overlays.png"
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_class_distribution(report_data: Dict) -> None:
    """Generate statistical publication-style bar chart of anomaly distribution."""
    train_cats = report_data["train"]["categories"]
    val_cats = report_data["val"]["categories"]

    x = np.arange(len(OFFICIAL_ANOMALY_CATEGORIES))
    width = 0.35

    train_vals = [train_cats.get(c, 0) for c in OFFICIAL_ANOMALY_CATEGORIES]
    val_vals = [val_cats.get(c, 0) for c in OFFICIAL_ANOMALY_CATEGORIES]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # Anomaly categories bar chart
    clean_labels = [c.replace("_", " ").title() for c in OFFICIAL_ANOMALY_CATEGORIES]
    r1 = ax1.bar(x - width/2, train_vals, width, label="Train Split (N=20)", color="#2E7D32")
    r2 = ax1.bar(x + width/2, val_vals, width, label="Validation Split (N=10)", color="#1976D2")

    ax1.set_ylabel("Number of Samples", fontsize=11, fontweight="bold")
    ax1.set_title("Agriculture-Vision Anomaly Class Distribution", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(clean_labels, rotation=25, ha="right", fontsize=10)
    ax1.legend(frameon=True)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Add count labels on bars
    for bar in r1 + r2:
        height = bar.get_height()
        ax1.annotate(f"{int(height)}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    # Anomaly Overlap & Clean Distribution
    splits = ["Train", "Val", "Test"]
    single_counts = [
        report_data["train"]["total"] - report_data["train"]["overlap"] - report_data["train"]["healthy"],
        report_data["val"]["total"] - report_data["val"]["overlap"] - report_data["val"]["healthy"],
        0,
    ]
    overlap_counts = [
        report_data["train"]["overlap"],
        report_data["val"]["overlap"],
        0,
    ]
    healthy_counts = [
        report_data["train"]["healthy"],
        report_data["val"]["healthy"],
        report_data["test"]["total"],  # Test is unannotated / evaluation
    ]

    p1 = ax2.bar(splits, healthy_counts, width=0.45, label="Healthy / Unannotated", color="#81C784")
    p2 = ax2.bar(splits, single_counts, width=0.45, bottom=healthy_counts, label="Single Anomaly", color="#FFB74D")
    bottom_overlap = [h + s for h, s in zip(healthy_counts, single_counts)]
    p3 = ax2.bar(splits, overlap_counts, width=0.45, bottom=bottom_overlap, label="Multi-Label Overlap", color="#E57373")

    ax2.set_ylabel("Total Number of Tiles", fontsize=11, fontweight="bold")
    ax2.set_title("Composition & Multi-Label Overlap Preservation", fontsize=12, fontweight="bold")
    ax2.legend(frameon=True)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    for i, total in enumerate([report_data["train"]["total"], report_data["val"]["total"], report_data["test"]["total"]]):
        ax2.annotate(f"Total: {total}", xy=(i, total), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    out_file = OUTPUT_DIR / "class_distribution_chart.png"
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Validating dataset splits for exploration...", flush=True)

    report_data = {}
    for split in ["train", "val", "test"]:
        has_labels = (split != "test")
        vrep = validate_dataset_split(DATA_DIR / split, split_name=split, has_labels=has_labels)
        # Compute healthy count
        healthy_count = sum(1 for res in vrep.sample_results.values() if len(res.categories_present) == 0)
        report_data[split] = {
            "total": vrep.total_samples,
            "valid": vrep.valid_samples,
            "invalid": vrep.invalid_samples,
            "overlap": vrep.overlap_samples_count,
            "healthy": healthy_count,
            "categories": vrep.category_counts,
        }

    print("Generating visual reports...", flush=True)
    sample_id = "KEP4MBVE7_1000-2000-1512-2512"
    plot_modalities_grid(sample_id, split="train")
    plot_anomaly_categories_showcase(split="train")
    plot_overlays_and_multilabel(split="train")
    plot_class_distribution(report_data)

    print("\nAll exploration visualizations generated successfully.")


if __name__ == "__main__":
    main()
