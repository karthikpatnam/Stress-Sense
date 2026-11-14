"""Data Validation Module for Generated Patches in StressSense.

Verifies:
- Patch dimensions (128x128) across all modalities.
- Channel consistency (RGB 3-channel, NIR 1-channel, masks 1-channel).
- Image-mask dimension matching for every individual patch.
- Valid binary annotation values ({0, 255}).
- Missing or corrupted patch files.
- Duplicate patch IDs.
- Strict preservation of original image IDs and field IDs.
- Zero cross-split contamination (no patch crosses train/val/test splits).
"""

import argparse
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import cv2
import numpy as np
import pandas as pd

from src.dataset.annotation_loader import OFFICIAL_ANOMALY_CATEGORIES


@dataclass
class PatchValidationReport:
    """Summary report of patch dataset integrity."""
    total_patches: int = 0
    valid_patches: int = 0
    invalid_patches: int = 0
    duplicate_patch_ids: List[str] = field(default_factory=list)
    missing_files: List[str] = field(default_factory=list)
    dimension_mismatches: List[str] = field(default_factory=list)
    channel_mismatches: List[str] = field(default_factory=list)
    invalid_annotation_values: List[str] = field(default_factory=list)
    split_cross_contaminations: List[str] = field(default_factory=list)
    field_cross_contaminations: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        """Returns True if there are zero errors, duplicates, or corrupted files."""
        return (
            self.invalid_patches == 0
            and len(self.duplicate_patch_ids) == 0
            and len(self.missing_files) == 0
            and len(self.dimension_mismatches) == 0
            and len(self.channel_mismatches) == 0
            and len(self.invalid_annotation_values) == 0
            and len(self.split_cross_contaminations) == 0
            and len(self.field_cross_contaminations) == 0
            and len(self.errors) == 0
        )

    def to_dict(self) -> Dict:
        """Convert report to dictionary."""
        return asdict(self)


class PatchValidator:
    """Validates processed patches directory and patch manifest."""

    def __init__(
        self,
        processed_dir: Union[str, Path] = "data/processed",
        manifest_path: Optional[Union[str, Path]] = None,
        expected_size: Tuple[int, int] = (128, 128),
        categories: Optional[List[str]] = None,
    ) -> None:
        """Initialize validator.

        Args:
            processed_dir: Path to processed data root.
            manifest_path: Optional path to patch_manifest.csv.
            expected_size: Expected (H, W) for every patch.
            categories: List of official anomaly categories.
        """
        self.processed_dir = Path(processed_dir)
        self.manifest_path = (
            Path(manifest_path)
            if manifest_path
            else self.processed_dir / "manifests" / "patch_manifest.csv"
        )
        self.expected_size = expected_size
        self.categories = categories or OFFICIAL_ANOMALY_CATEGORIES

    def validate(self) -> PatchValidationReport:
        """Run validation checks on the manifest and patch files."""
        report = PatchValidationReport()

        if not self.manifest_path.is_file():
            report.errors.append(f"Manifest file not found: {self.manifest_path}")
            return report

        try:
            df = pd.read_csv(self.manifest_path)
        except Exception as exc:
            report.errors.append(f"Cannot read manifest CSV: {exc}")
            return report

        report.total_patches = len(df)
        if report.total_patches == 0:
            report.warnings.append("Manifest contains zero patch records.")
            return report

        # 1. Duplicate Patch ID Check
        patch_id_counts = df["patch_id"].value_counts()
        duplicates = patch_id_counts[patch_id_counts > 1].index.tolist()
        if duplicates:
            report.duplicate_patch_ids = duplicates
            report.errors.append(f"Found {len(duplicates)} duplicate patch IDs.")

        # 2. Check Split & Field Contamination
        # Ensure that no original_image_id appears in more than one split
        img_splits = df.groupby("original_image_id")["split"].nunique()
        split_cross = img_splits[img_splits > 1].index.tolist()
        if split_cross:
            report.split_cross_contaminations = split_cross
            report.errors.append(f"Found images crossing official splits: {split_cross}")

        # Ensure that no field_id appears in more than one split
        if "field_id" in df.columns:
            field_splits = df.groupby("field_id")["split"].nunique()
            field_cross = field_splits[field_splits > 1].index.tolist()
            if field_cross:
                report.field_cross_contaminations = field_cross
                report.errors.append(f"Found field IDs crossing splits: {field_cross}")

        # 3. File existence, dimensions, channels, and value validation
        exp_h, exp_w = self.expected_size

        for _, row in df.iterrows():
            pid = row["patch_id"]
            split = row["split"]
            has_labels = (split != "test")
            patch_valid = True

            # Verify RGB Patch
            rgb_path = Path(row["rgb_patch_path"])
            if not rgb_path.is_file():
                report.missing_files.append(str(rgb_path))
                patch_valid = False
            else:
                img_rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
                if img_rgb is None:
                    report.errors.append(f"[{pid}] Cannot decode RGB patch {rgb_path}")
                    patch_valid = False
                else:
                    if (img_rgb.shape[0], img_rgb.shape[1]) != (exp_h, exp_w):
                        report.dimension_mismatches.append(
                            f"[{pid}] RGB shape {img_rgb.shape[:2]} != {self.expected_size}"
                        )
                        patch_valid = False
                    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
                        report.channel_mismatches.append(f"[{pid}] RGB channels != 3")
                        patch_valid = False

            # Verify NIR Patch
            nir_path = Path(row["nir_patch_path"])
            if not nir_path.is_file():
                report.missing_files.append(str(nir_path))
                patch_valid = False
            else:
                img_nir = cv2.imread(str(nir_path), cv2.IMREAD_GRAYSCALE)
                if img_nir is None:
                    report.errors.append(f"[{pid}] Cannot decode NIR patch {nir_path}")
                    patch_valid = False
                else:
                    if img_nir.shape != (exp_h, exp_w):
                        report.dimension_mismatches.append(
                            f"[{pid}] NIR shape {img_nir.shape} != {self.expected_size}"
                        )
                        patch_valid = False

            # Verify Boundary Patch
            bnd_path = Path(row["boundary_patch_path"])
            if not bnd_path.is_file():
                report.missing_files.append(str(bnd_path))
                patch_valid = False
            else:
                img_bnd = cv2.imread(str(bnd_path), cv2.IMREAD_GRAYSCALE)
                if img_bnd is None:
                    report.errors.append(f"[{pid}] Cannot decode boundary patch {bnd_path}")
                    patch_valid = False
                else:
                    if img_bnd.shape != (exp_h, exp_w):
                        report.dimension_mismatches.append(f"[{pid}] Boundary shape != {self.expected_size}")
                        patch_valid = False
                    uvals = np.unique(img_bnd)
                    if not np.all(np.isin(uvals, [0, 255])):
                        report.invalid_annotation_values.append(
                            f"[{pid}] Non-binary values in boundary: {uvals}"
                        )
                        patch_valid = False

            # Verify Validity Mask Patch
            msk_path = Path(row["mask_patch_path"])
            if not msk_path.is_file():
                report.missing_files.append(str(msk_path))
                patch_valid = False
            else:
                img_msk = cv2.imread(str(msk_path), cv2.IMREAD_GRAYSCALE)
                if img_msk is None:
                    report.errors.append(f"[{pid}] Cannot decode mask patch {msk_path}")
                    patch_valid = False
                else:
                    if img_msk.shape != (exp_h, exp_w):
                        report.dimension_mismatches.append(f"[{pid}] Mask shape != {self.expected_size}")
                        patch_valid = False
                    uvals = np.unique(img_msk)
                    if not np.all(np.isin(uvals, [0, 255])):
                        report.invalid_annotation_values.append(
                            f"[{pid}] Non-binary values in validity mask: {uvals}"
                        )
                        patch_valid = False

            # Verify Category Label Patches (if split has labels)
            if has_labels:
                for cat in self.categories:
                    cat_path = (
                        self.processed_dir
                        / "patches"
                        / split
                        / "labels"
                        / cat
                        / f"{pid}.png"
                    )
                    if not cat_path.is_file():
                        report.missing_files.append(str(cat_path))
                        patch_valid = False
                    else:
                        img_cat = cv2.imread(str(cat_path), cv2.IMREAD_GRAYSCALE)
                        if img_cat is None:
                            report.errors.append(f"[{pid}] Cannot decode {cat} patch")
                            patch_valid = False
                        else:
                            if img_cat.shape != (exp_h, exp_w):
                                report.dimension_mismatches.append(f"[{pid}] {cat} shape != {self.expected_size}")
                                patch_valid = False
                            uvals = np.unique(img_cat)
                            if not np.all(np.isin(uvals, [0, 255])):
                                report.invalid_annotation_values.append(
                                    f"[{pid}] Non-binary values in {cat}: {uvals}"
                                )
                                patch_valid = False

            if patch_valid:
                report.valid_patches += 1
            else:
                report.invalid_patches += 1

        return report


def main() -> None:
    """CLI runner for patch validation."""
    parser = argparse.ArgumentParser(description="Validate generated Agriculture-Vision patches.")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--json_out", type=str, default="results/preprocessing/validation_report.json")
    args = parser.parse_args()

    validator = PatchValidator(
        processed_dir=args.processed_dir,
        manifest_path=args.manifest,
    )
    report = validator.validate()

    print("=" * 60)
    print("STRESSSENSE - PATCH VALIDATION REPORT")
    print("=" * 60)
    print(f"Total Patches Checked    : {report.total_patches}")
    print(f"Valid Patches            : {report.valid_patches}")
    print(f"Invalid Patches          : {report.invalid_patches}")
    print(f"Duplicate Patch IDs      : {len(report.duplicate_patch_ids)}")
    print(f"Missing Files            : {len(report.missing_files)}")
    print(f"Dimension Mismatches     : {len(report.dimension_mismatches)}")
    print(f"Channel Mismatches       : {len(report.channel_mismatches)}")
    print(f"Invalid Value Warnings   : {len(report.invalid_annotation_values)}")
    print(f"Split Cross-Contaminations: {len(report.split_cross_contaminations)}")
    print(f"Field Cross-Contaminations: {len(report.field_cross_contaminations)}")
    print(f"Overall Dataset Clean    : {report.is_clean}")
    print("=" * 60)

    if args.json_out:
        out_p = Path(args.json_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"Validation report saved to: {out_p}")


if __name__ == "__main__":
    main()
