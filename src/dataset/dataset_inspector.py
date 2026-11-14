"""Dataset inspector module for Agriculture-Vision dataset.

Inspects the dataset directory structure and discovers:
- Training, validation, and test split directories
- Modalities: RGB images, NIR images, boundary maps, validity masks
- Anomaly label categories and counts
- Image formats and dimension consistency
- Missing files and cross-modality correspondence
- Duplicate files (by file hash or identifier)
"""

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import cv2
from PIL import Image

from src.dataset.annotation_loader import OFFICIAL_ANOMALY_CATEGORIES


@dataclass
class SplitInspectionSummary:
    """Summary of inspection for a specific dataset split (train/val/test)."""
    split_name: str
    exists: bool = False
    has_images_dir: bool = False
    has_rgb_dir: bool = False
    has_nir_dir: bool = False
    has_boundaries_dir: bool = False
    has_masks_dir: bool = False
    has_labels_dir: bool = False
    rgb_count: int = 0
    nir_count: int = 0
    boundary_count: int = 0
    mask_count: int = 0
    label_categories: List[str] = field(default_factory=list)
    label_counts: Dict[str, int] = field(default_factory=dict)
    sample_ids: List[str] = field(default_factory=list)
    missing_nir_ids: List[str] = field(default_factory=list)
    missing_boundary_ids: List[str] = field(default_factory=list)
    missing_mask_ids: List[str] = field(default_factory=list)
    image_formats: Dict[str, int] = field(default_factory=dict)
    image_dimensions: Dict[str, int] = field(default_factory=dict)
    duplicate_files: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class DatasetInspectionReport:
    """Complete dataset inspection report across all splits."""
    root_dir: str
    splits_found: List[str] = field(default_factory=list)
    split_summaries: Dict[str, SplitInspectionSummary] = field(default_factory=dict)
    total_samples: int = 0
    total_rgb_images: int = 0
    total_nir_images: int = 0
    total_boundaries: int = 0
    total_validity_masks: int = 0
    known_categories: List[str] = field(default_factory=lambda: list(OFFICIAL_ANOMALY_CATEGORIES))
    missing_files_total: int = 0
    duplicate_files_total: int = 0
    all_dimensions_uniform: bool = True
    detected_dimensions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to JSON-serializable dictionary."""
        return asdict(self)


def get_file_md5(file_path: Path) -> str:
    """Compute MD5 hash of a file for duplicate detection."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def inspect_split(split_dir: Union[str, Path], split_name: str) -> SplitInspectionSummary:
    """Inspect a single split directory (train, val, or test)."""
    sdir = Path(split_dir)
    summary = SplitInspectionSummary(split_name=split_name)

    if not sdir.is_dir():
        return summary

    summary.exists = True
    images_dir = sdir / "images"
    summary.has_images_dir = images_dir.is_dir()

    rgb_dir = images_dir / "rgb" if summary.has_images_dir else sdir / "rgb"
    nir_dir = images_dir / "nir" if summary.has_images_dir else sdir / "nir"
    boundaries_dir = sdir / "boundaries"
    masks_dir = sdir / "masks"
    labels_dir = sdir / "labels"

    summary.has_rgb_dir = rgb_dir.is_dir()
    summary.has_nir_dir = nir_dir.is_dir()
    summary.has_boundaries_dir = boundaries_dir.is_dir()
    summary.has_masks_dir = masks_dir.is_dir()
    summary.has_labels_dir = labels_dir.is_dir()

    # Collect sample IDs from RGB images
    rgb_files: Dict[str, Path] = {}
    if summary.has_rgb_dir:
        for f in rgb_dir.iterdir():
            if f.is_file() and not f.name.startswith("."):
                ext = f.suffix.lower()
                summary.image_formats[ext] = summary.image_formats.get(ext, 0) + 1
                sample_id = f.stem
                rgb_files[sample_id] = f

                # Check dimensions for first 200 images or all
                if len(rgb_files) <= 200:
                    try:
                        with Image.open(f) as img:
                            dim_str = f"{img.width}x{img.height}"
                            summary.image_dimensions[dim_str] = (
                                summary.image_dimensions.get(dim_str, 0) + 1
                            )
                    except Exception:
                        pass

    summary.rgb_count = len(rgb_files)
    summary.sample_ids = sorted(list(rgb_files.keys()))

    # Check NIR files
    nir_ids: Set[str] = set()
    if summary.has_nir_dir:
        for f in nir_dir.iterdir():
            if f.is_file() and not f.name.startswith("."):
                nir_ids.add(f.stem)
    summary.nir_count = len(nir_ids)

    # Check Boundaries
    boundary_ids: Set[str] = set()
    if summary.has_boundaries_dir:
        for f in boundaries_dir.iterdir():
            if f.is_file() and not f.name.startswith("."):
                boundary_ids.add(f.stem)
    summary.boundary_count = len(boundary_ids)

    # Check Masks
    mask_ids: Set[str] = set()
    if summary.has_masks_dir:
        for f in masks_dir.iterdir():
            if f.is_file() and not f.name.startswith("."):
                mask_ids.add(f.stem)
    summary.mask_count = len(mask_ids)

    # Check missing cross-modality correspondence
    for sid in summary.sample_ids:
        if sid not in nir_ids:
            summary.missing_nir_ids.append(sid)
        if sid not in boundary_ids:
            summary.missing_boundary_ids.append(sid)
        if sid not in mask_ids:
            summary.missing_mask_ids.append(sid)

    # Inspect labels if directory exists
    if summary.has_labels_dir:
        for cat_dir in labels_dir.iterdir():
            if cat_dir.is_dir() and not cat_dir.name.startswith("."):
                cat_name = cat_dir.name
                summary.label_categories.append(cat_name)
                # Count files inside category dir
                count = sum(1 for f in cat_dir.iterdir() if f.is_file() and not f.name.startswith("."))
                summary.label_counts[cat_name] = count

    return summary


def inspect_dataset(dataset_dir: Union[str, Path]) -> DatasetInspectionReport:
    """Inspect the entire dataset directory across train, val, and test splits.

    Args:
        dataset_dir: Root dataset folder (e.g. data/raw/Agriculture-Vision-2021).

    Returns:
        DatasetInspectionReport with complete statistics.
    """
    root = Path(dataset_dir)
    report = DatasetInspectionReport(root_dir=str(root))

    if not root.is_dir():
        return report

    # Check splits
    potential_splits = ["train", "val", "test"]
    all_dims: Set[str] = set()

    for split in potential_splits:
        sdir = root / split
        if sdir.is_dir():
            report.splits_found.append(split)
            summary = inspect_split(sdir, split_name=split)
            report.split_summaries[split] = summary

            report.total_samples += len(summary.sample_ids)
            report.total_rgb_images += summary.rgb_count
            report.total_nir_images += summary.nir_count
            report.total_boundaries += summary.boundary_count
            report.total_validity_masks += summary.mask_count

            report.missing_files_total += (
                len(summary.missing_nir_ids)
                + len(summary.missing_boundary_ids)
                + len(summary.missing_mask_ids)
            )
            report.duplicate_files_total += len(summary.duplicate_files)

            all_dims.update(summary.image_dimensions.keys())

    report.detected_dimensions = sorted(list(all_dims))
    report.all_dimensions_uniform = (len(all_dims) == 1 and "512x512" in all_dims) or len(all_dims) == 0

    return report


def main() -> None:
    """CLI entry point for dataset inspection."""
    parser = argparse.ArgumentParser(description="Inspect Agriculture-Vision dataset structure.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw/Agriculture-Vision-2021",
        help="Root path to Agriculture-Vision dataset directory.",
    )
    parser.add_argument(
        "--json_out",
        type=str,
        default="results/dataset_report/dataset_inspection.json",
        help="Optional output path for JSON inspection report.",
    )
    args = parser.parse_args()

    report = inspect_dataset(args.data_dir)
    print("=" * 60)
    print(f"Agriculture-Vision Dataset Inspection: {args.data_dir}")
    print("=" * 60)
    print(f"Splits found: {report.splits_found}")
    print(f"Total samples: {report.total_samples}")
    print(f"Total RGB images: {report.total_rgb_images}")
    print(f"Total NIR images: {report.total_nir_images}")
    print(f"Total boundaries: {report.total_boundaries}")
    print(f"Total validity masks: {report.total_validity_masks}")
    print(f"Detected dimensions: {report.detected_dimensions}")
    print(f"Dimensions uniform (512x512): {report.all_dimensions_uniform}")
    print(f"Missing files total: {report.missing_files_total}")

    for sname, ssum in report.split_summaries.items():
        print(f"\n--- Split: {sname} ---")
        print(f"  RGB count: {ssum.rgb_count}")
        print(f"  NIR count: {ssum.nir_count}")
        print(f"  Boundaries: {ssum.boundary_count}")
        print(f"  Masks: {ssum.mask_count}")
        print(f"  Label categories: {ssum.label_categories}")
        print(f"  Label counts: {ssum.label_counts}")
        if ssum.missing_nir_ids:
            print(f"  Missing NIR: {len(ssum.missing_nir_ids)}")
        if ssum.missing_boundary_ids:
            print(f"  Missing Boundaries: {len(ssum.missing_boundary_ids)}")
        if ssum.missing_mask_ids:
            print(f"  Missing Masks: {len(ssum.missing_mask_ids)}")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nInspection report saved to: {out_path}")


if __name__ == "__main__":
    main()
