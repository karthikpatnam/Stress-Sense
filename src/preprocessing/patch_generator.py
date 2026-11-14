"""End-to-End Patch Generation Pipeline for StressSense.

Generates 128x128 patches across RGB, NIR, boundary, validity mask, and all
6 official anomaly categories, preserving field IDs and split boundaries without data leakage.
Outputs to `data/processed/patches/` and registers metadata in `data/processed/manifests/patch_manifest.csv`.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import pandas as pd

from src.dataset.image_loader import (
    load_rgb_image,
    load_nir_image,
    DEFAULT_EXPECTED_SHAPE,
)
from src.dataset.annotation_loader import (
    OFFICIAL_ANOMALY_CATEGORIES,
    load_boundary_mask,
    load_validity_mask,
    load_all_category_masks,
)
from src.preprocessing.patch_extractor import PatchExtractor, ExtractedPatch


DEFAULT_RAW_DIR = Path("data/raw/Agriculture-Vision-2021")
DEFAULT_PROCESSED_DIR = Path("data/processed")


class PatchGenerator:
    """Orchestrates patch extraction, disk persistence, and manifest generation."""

    def __init__(
        self,
        raw_dir: Union[str, Path] = DEFAULT_RAW_DIR,
        processed_dir: Union[str, Path] = DEFAULT_PROCESSED_DIR,
        patch_size: int = 128,
        stride: Optional[int] = None,
        categories: Optional[List[str]] = None,
    ) -> None:
        """Initialize PatchGenerator.

        Args:
            raw_dir: Path to raw dataset root.
            processed_dir: Path to processed output root.
            patch_size: Spatial height and width of each patch (default 128).
            stride: Step size (default 128, non-overlapping).
            categories: List of official anomaly categories.
        """
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.patch_size = patch_size
        self.stride = stride or patch_size
        self.categories = categories or OFFICIAL_ANOMALY_CATEGORIES
        self.extractor = PatchExtractor(
            patch_size=(self.patch_size, self.patch_size),
            stride=(self.stride, self.stride),
            pad_if_needed=True,
            pad_mode="reflect",
        )

    def extract_field_id(self, image_id: str) -> str:
        """Parse field identifier from image ID (format: <flightcode>_<coords>)."""
        return image_id.split("_")[0] if "_" in image_id else image_id

    def setup_directories(self, split: str, has_labels: bool = True) -> Dict[str, Path]:
        """Create target directory hierarchy for a given split."""
        split_root = self.processed_dir / "patches" / split
        dirs = {
            "root": split_root,
            "rgb": split_root / "images" / "rgb",
            "nir": split_root / "images" / "nir",
            "boundaries": split_root / "boundaries",
            "masks": split_root / "masks",
        }
        for p in dirs.values():
            p.mkdir(parents=True, exist_ok=True)

        if has_labels:
            for cat in self.categories:
                cat_dir = split_root / "labels" / cat
                cat_dir.mkdir(parents=True, exist_ok=True)
                dirs[f"label_{cat}"] = cat_dir

        return dirs

    def process_sample(
        self,
        sample_id: str,
        split: str,
        target_dirs: Dict[str, Path],
        has_labels: bool = True,
    ) -> List[Dict]:
        """Generate, save, and catalog patches for a single 512x512 tile.

        Returns:
            List of metadata dictionary records for each extracted patch.
        """
        split_raw = self.raw_dir / split
        rgb_file = split_raw / "images" / "rgb" / f"{sample_id}.jpg"
        if not rgb_file.is_file():
            rgb_file = split_raw / "images" / "rgb" / f"{sample_id}.png"

        nir_file = split_raw / "images" / "nir" / f"{sample_id}.jpg"
        if not nir_file.is_file():
            nir_file = split_raw / "images" / "nir" / f"{sample_id}.png"

        bnd_file = split_raw / "boundaries" / f"{sample_id}.png"
        msk_file = split_raw / "masks" / f"{sample_id}.png"

        # Load modalities using Phase 1 verified loaders
        rgb = load_rgb_image(rgb_file)
        nir = load_nir_image(nir_file)
        bnd = load_boundary_mask(bnd_file)
        msk = load_validity_mask(msk_file)

        # Extract patches across primary modalities
        rgb_patches = self.extractor.extract_patches(rgb)
        nir_patches = self.extractor.extract_patches(nir)
        bnd_patches = self.extractor.extract_patches(bnd)
        msk_patches = self.extractor.extract_patches(msk)

        # Load category masks if labels are available (train and val)
        cat_patches_map: Dict[str, List[ExtractedPatch]] = {}
        if has_labels:
            labels_dir = split_raw / "labels"
            cat_masks = load_all_category_masks(
                labels_dir=labels_dir,
                sample_id=sample_id,
                categories=self.categories,
                to_boolean=False,  # uint8 0 or 255
            )
            for cat, cmask in cat_masks.items():
                cat_patches_map[cat] = self.extractor.extract_patches(cmask)

        field_id = self.extract_field_id(sample_id)
        records: List[Dict] = []
        num_patches = len(rgb_patches)
        total_pixels = float(self.patch_size * self.patch_size)

        for i in range(num_patches):
            p_rgb = rgb_patches[i]
            p_nir = nir_patches[i]
            p_bnd = bnd_patches[i]
            p_msk = msk_patches[i]

            # Deterministic, unique patch identifier
            patch_id = f"{sample_id}_p{p_rgb.y:03d}_{p_rgb.x:03d}"

            # File destinations (saved as lossless PNG to prevent compression artifacts)
            rgb_out = target_dirs["rgb"] / f"{patch_id}.png"
            nir_out = target_dirs["nir"] / f"{patch_id}.png"
            bnd_out = target_dirs["boundaries"] / f"{patch_id}.png"
            msk_out = target_dirs["masks"] / f"{patch_id}.png"

            # Save imagery (OpenCV expects BGR for RGB saving)
            cv2.imwrite(str(rgb_out), cv2.cvtColor(p_rgb.data, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(nir_out), p_nir.data)
            cv2.imwrite(str(bnd_out), p_bnd.data)
            cv2.imwrite(str(msk_out), p_msk.data)

            # Compute spatial coverage metrics
            valid_frac = float(np.count_nonzero(p_msk.data > 0)) / total_pixels
            bnd_frac = float(np.count_nonzero(p_bnd.data > 0)) / total_pixels

            # Compute category fractions and presence
            cat_fractions: Dict[str, float] = {}
            active_cats: List[str] = []

            if has_labels:
                for cat in self.categories:
                    p_cat = cat_patches_map[cat][i]
                    cat_out = target_dirs[f"label_{cat}"] / f"{patch_id}.png"
                    cv2.imwrite(str(cat_out), p_cat.data)

                    frac = float(np.count_nonzero(p_cat.data > 0)) / total_pixels
                    cat_fractions[f"{cat}_fraction"] = round(frac, 5)
                    if frac > 0.0:
                        active_cats.append(cat)
            else:
                for cat in self.categories:
                    cat_fractions[f"{cat}_fraction"] = 0.0

            has_anomaly = len(active_cats) > 0
            is_overlap = len(active_cats) > 1

            record = {
                "patch_id": patch_id,
                "original_image_id": sample_id,
                "field_id": field_id,
                "split": split,
                "y_coordinate": p_rgb.y,
                "x_coordinate": p_rgb.x,
                "patch_size": self.patch_size,
                "stride": self.stride,
                "valid_pixel_fraction": round(valid_frac, 5),
                "boundary_pixel_fraction": round(bnd_frac, 5),
                "has_anomaly": has_anomaly,
                "has_overlap": is_overlap,
                "anomaly_count": len(active_cats),
                "anomaly_categories": ",".join(active_cats) if active_cats else "none",
                "rgb_patch_path": str(rgb_out).replace("\\", "/"),
                "nir_patch_path": str(nir_out).replace("\\", "/"),
                "boundary_patch_path": str(bnd_out).replace("\\", "/"),
                "mask_patch_path": str(msk_out).replace("\\", "/"),
            }
            record.update(cat_fractions)
            records.append(record)

        return records

    def run(self) -> pd.DataFrame:
        """Execute patch generation across all splits (train, val, test).

        Returns:
            DataFrame representing the full patch manifest.
        """
        all_records: List[Dict] = []
        splits = ["train", "val", "test"]

        print(f"Starting patch generation (patch_size={self.patch_size}, stride={self.stride})...", flush=True)

        for split in splits:
            split_dir = self.raw_dir / split
            if not split_dir.is_dir():
                print(f"Warning: Split directory {split_dir} does not exist, skipping.", flush=True)
                continue

            has_labels = (split != "test")
            target_dirs = self.setup_directories(split, has_labels=has_labels)

            rgb_dir = split_dir / "images" / "rgb"
            sample_ids = sorted(
                [f.stem for f in rgb_dir.iterdir() if f.suffix.lower() in [".jpg", ".png", ".jpeg"]]
            )

            print(f"Split [{split}]: Processing {len(sample_ids)} full tiles...", flush=True)
            for sid in sample_ids:
                recs = self.process_sample(sid, split, target_dirs, has_labels=has_labels)
                all_records.extend(recs)

        # Build and save manifest
        manifest_df = pd.DataFrame(all_records)
        manifest_dir = self.processed_dir / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)

        csv_path = manifest_dir / "patch_manifest.csv"
        json_path = manifest_dir / "patch_manifest.json"

        manifest_df.to_csv(csv_path, index=False)
        print(f"Saved patch manifest CSV to: {csv_path} ({len(manifest_df)} patches)")

        # Save machine-readable JSON summary of manifest
        summary_stats = {
            "total_patches": len(manifest_df),
            "patch_size": self.patch_size,
            "stride": self.stride,
            "splits": {
                s: {
                    "total_patches": int((manifest_df["split"] == s).sum()),
                    "anomaly_patches": int(((manifest_df["split"] == s) & (manifest_df["has_anomaly"])).sum()),
                    "overlap_patches": int(((manifest_df["split"] == s) & (manifest_df["has_overlap"])).sum()),
                    "healthy_patches": int(((manifest_df["split"] == s) & (~manifest_df["has_anomaly"])).sum()),
                }
                for s in splits if (manifest_df["split"] == s).any()
            },
            "category_frequencies": {
                cat: int((manifest_df[f"{cat}_fraction"] > 0).sum())
                for cat in self.categories
            },
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_stats, f, indent=2)
        print(f"Saved patch manifest JSON summary to: {json_path}")

        return manifest_df


def main() -> None:
    """CLI runner for patch generation."""
    parser = argparse.ArgumentParser(description="Generate image patches for StressSense.")
    parser.add_argument("--raw_dir", type=str, default="data/raw/Agriculture-Vision-2021")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=128)
    args = parser.parse_args()

    gen = PatchGenerator(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        patch_size=args.patch_size,
        stride=args.stride,
    )
    df = gen.run()
    print("Patch generation complete!")


if __name__ == "__main__":
    main()
