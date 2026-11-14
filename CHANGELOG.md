# Changelog

All notable changes to the StressSense project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-11-14

### Added - Phase 2: Image Preprocessing and Patch Generation
- Implemented `ImageNormalizer` in `src/preprocessing/normalizer.py`:
  - Min-Max normalization with numerical stability constant ($\epsilon=10^{-7}$).
  - Z-Score standardization with exact inverse de-normalization.
  - Zero-centered normalization $[-1.0, 1.0]$.
  - Normalized Difference Vegetation Index (NDVI) computation combining NIR and Red bands.
- Implemented `PatchExtractor` in `src/preprocessing/patch_extractor.py`:
  - Configurable patch extraction geometry (default $128 \times 128$ pixels, stride $128$).
  - Boundary padding handling (`reflect`, `constant`) for arbitrary dimensions.
  - Reversible round-trip image tile reconstruction from patches with bit-exact fidelity (MAE $= 0.0$).
  - 0-indexed coordinate indexing: `(x_min, y_min, x_max, y_max)`.
- Implemented `PatchGenerator` in `src/preprocessing/patch_generator.py`:
  - End-to-end multi-spectral patch generation across RGB, NIR, boundary, validity, and 6 ground-truth anomaly masks.
  - Multi-label anomaly preservation without merging distinct classes.
  - Generated $640$ total patches across $40$ Agriculture-Vision tiles ($320$ train, $160$ val, $160$ test).
  - Persisted structured metadata catalog into `data/processed/manifests/patch_manifest.csv` and `patch_manifest.json` ($23$ metadata fields).
- Implemented `PatchValidator` in `src/preprocessing/patch_validator.py`:
  - Comprehensive data integrity audit checking patch dimensions, channel counts, $\{0, 255\}$ binary mask values, missing files, duplicate IDs, and zero cross-split or cross-field data leakage.
  - Output validation report saved to `results/preprocessing/validation_report.json` ($640/640$ valid, clean status `true`).
- Implemented `PreprocessingVisualizer` in `src/preprocessing/visualizer.py`:
  - Generated publication-ready figures in `results/preprocessing/`:
    - `preprocessing_pipeline_showcase.png` (5-panel raw-to-overlay showcase)
    - `patch_grid_decomposition.png` ($4 \times 4$ tiled grid overlay)
    - `multilabel_patch_overlays.png` (overlaid patches across anomaly categories and healthy farmland)
    - `patch_statistics_summary.png` (split distributions and anomaly frequencies)
- Implemented unit test suite in `tests/test_preprocessing.py`:
  - $9/9$ unit tests passing ($100\%$ pass rate).
- Updated cumulative project test suite: $36/36$ tests passing across Phase 0, Phase 1, and Phase 2.
- Authored comprehensive documentation in `docs/PREPROCESSING_DOCUMENTATION.md`.

## [0.2.0] - 2026-11-13

### Added - Phase 1: Dataset Acquisition and Exploration
- Verified official dataset source: `s3://intelinair-data-releases/agriculture-vision/cvpr_challenge_2021/supervised/Agriculture-Vision-2021.tar.gz` (21.03 GB compressed, ~60.5 GB uncompressed).
- Conducted storage assessment on `P:\` (30.82 GB free space available; confirmed full extraction requires ~81.5 GB, justifying a verified representative subset for initial exploration).
- Downloaded official field manifest `data/raw/data_manifest.csv` containing 8,515 flightcode and field mappings.
- Downloaded official terms of use and research publication PDFs to `docs/`:
  - `docs/Agriculture-Vision_Dataset_Terms_of_Use.pdf`
  - `docs/Agriculture-Vision_Paper_2020.pdf`
- Acquired complete ground-truth anomaly file rosters to `data/annotations/` for all 6 official challenge categories (12,901 unique anomaly samples).
- Generated verified representative dataset subset in `data/raw/Agriculture-Vision-2021/`:
  - 40 tiles across `train/` (20), `val/` (10), and `test/` (10)
  - 4 spectral channels (RGB 3-channel + NIR 1-channel, 512x512 resolution)
  - Binary field boundary masks and validity masks (512x512)
  - Ground-truth binary segmentation masks for 6 categories with multi-label overlapping anomalies preserved without class merging
  - Test split with withheld labels adhering to competition evaluation rules
- Implemented `src/dataset/` package:
  - `image_loader.py`: Loading and normalization of RGB, NIR, and stacked 4-channel arrays.
  - `annotation_loader.py`: Loading boundary masks, validity masks, category masks, and multi-hot (512x512x6) arrays.
  - `dataset_validator.py`: Automated 1-to-1 modality correspondence and data integrity verification.
  - `dataset_inspector.py`: Discovery and audit of splits, channels, dimensions, formats, and duplicates.
- Generated comprehensive dataset exploration report and visualizations in `results/dataset_report/`:
  - `sample_modalities_grid.png` (RGB, NIR, boundary, validity mask comparison)
  - `anomaly_categories_showcase.png` (12-panel showcase of all 6 anomaly categories)
  - `rgb_annotation_overlays.png` (color-coded anomaly segmentation overlays)
  - `class_distribution_chart.png` (class occurrence and multi-label overlap frequency chart)
  - `dataset_inspection.json` (machine-readable inspection metadata)
- Published machine-readable `dataset_summary.json` and comprehensive documentation in `docs/DATASET_DOCUMENTATION.md`.
- Implemented automated pytest suite in `tests/test_dataset.py` with 12/12 unit tests passing (100% pass rate).
- Verified zero data leakage: field ID separation confirmed between train, val, and test splits.

## [0.1.0] - 2026-11-12

### Added - Phase 0: Environment and Version Configuration
- Verified baseline host environment on Windows 11 with Python 3.11.9 and OpenJDK 17.0.21.0.2 LTS (`C:\Java\jdk-17`).
- Detected NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM) with Driver 576.93 supporting CUDA 12.9 compute.
- Created isolated virtual environment `.venv` using Python 3.11.9.
- Defined pinned dependencies in `requirements.txt`:
  - `numpy==2.2.6`
  - `pandas==2.2.3`
  - `opencv-python==4.12.0.88`
  - `matplotlib==3.10.1`
  - `scikit-learn==1.7.2`
  - `pyspark==4.3.0`
  - `pytest==8.3.5`
  - `Pillow==11.1.0`
- Configured dedicated CUDA 12.6 PyTorch builds:
  - `torch==2.14.1+cu126`
  - `torchvision==0.29.1+cu126`
- Created standard project folder structure:
  - `data/` (`raw`, `processed`, `annotations`)
  - `src/` (`preprocessing`, `models`, `adaptive`, `spark`, `evaluation`)
  - `tests/`
  - `notebooks/`
  - `experiments/`
  - `results/`
  - `docs/`
  - `scripts/`
- Implemented diagnostic script `scripts/verify_environment.py` for automated health checks across Python, Java, PySpark, PyTorch (CPU and CUDA), and data science libraries.
- Implemented automated pytest test suite in `tests/test_environment.py` with Spark DataFrame transformation tests and GPU/CPU tensor tests.
- Documented full technology stack and compatibility matrix in `docs/TECH_STACK.md`.
- Published initial `TEST_REPORT.md` recording all Phase 0 acceptance criteria.
