# StressSense - Phase 0 Test Execution & Environment Report

**Project**: Cost- and Energy-Aware Drone Crop Health Monitoring Using Adaptive Sampling and Apache Spark  
**Phase**: Phase 0 – Environment and Version Configuration  
**Date**: 2026-11-12  
**Platform**: Windows 11 (64-bit)  
**Host Environment**: Google Antigravity / PowerShell 7+  

---

## 1. Executive Summary

Phase 0 establishes the isolated, deterministic development environment for StressSense. All required runtime environments, distributed processing engines, deep learning libraries with CUDA acceleration, and data science dependencies were installed with strict adherence to target versions without unauthorized upgrades or downgrades.

| Status | Verification Component | Criteria | Result |
| :---: | :--- | :--- | :---: |
| 🟢 | **Python 3.11 Runtime** | Python 3.11.x 64-bit isolated in `.venv` | **PASSED** (`3.11.9`) |
| 🟢 | **Java JDK 17 Runtime** | Java JDK 17 LTS detected, `JAVA_HOME` valid | **PASSED** (`17.0.21.0.2 LTS`) |
| 🟢 | **PySpark Local Runtime** | PySpark 4.3.0 starts SparkSession, DataFrame transforms execute | **PASSED** (`4.3.0`) |
| 🟢 | **Core Data Science Stack** | Exact pinned versions for NumPy, Pandas, OpenCV, Matplotlib, Scikit-learn, Pillow, Pytest | **PASSED** |
| 🟢 | **PyTorch & Torchvision** | PyTorch 2.14.1 + Torchvision 0.29.1 with `cu126` build | **PASSED** (`2.14.1+cu126`, `0.29.1+cu126`) |
| 🟢 | **CPU Tensor Fallback** | Matrix multiplication & tensor operations on CPU | **PASSED** |
| 🟢 | **CUDA Hardware Acceleration** | NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM) detected, GPU matmul validated | **PASSED** |
| 🟢 | **Automated Test Suite** | 100% passing test cases via `pytest` (15/15) and `scripts/verify_environment.py` | **PASSED** |

---

## 2. Pinned Technology Version Audit

| Dependency | Target Pinned Version | Installed Version | Status | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Python** | `3.11.x` | `3.11.9` | ✅ MATCH | Standard CPython 64-bit |
| **Java JDK** | `17 LTS` | `17.0.21.0.2` | ✅ MATCH | OpenJDK 64-Bit Server VM Microsoft LTS (`C:\Java\jdk-17`) |
| **PySpark** | `4.3.0` | `4.3.0` | ✅ MATCH | Bundles Apache Spark 4.3.0 jars |
| **Apache Spark** | `4.3.0` | `4.3.0` | ✅ MATCH | Local master mode (`local[*]`) |
| **PyTorch** | `2.14.1` | `2.14.1+cu126` | ✅ MATCH | Built with CUDA 12.6 runtime wheel (`cu126`) |
| **Torchvision** | `0.29.1` | `0.29.1+cu126` | ✅ MATCH | Vision backbones & transforms (`cu126`) |
| **CUDA Platform** | `cu126` | CUDA 12.6 (Driver 576.93) | ✅ MATCH | Driver capability supports up to CUDA 12.9 |
| **NumPy** | `2.2.6` | `2.2.6` | ✅ MATCH | C-API ABI verified |
| **Pandas** | `2.2.3` | `2.2.3` | ✅ MATCH | Pinned |
| **OpenCV** | `4.12.0.88` | `4.12.0.88` | ✅ MATCH | `opencv-python` headless/gui |
| **Matplotlib** | `3.10.1` | `3.10.1` | ✅ MATCH | Pinned |
| **Scikit-learn** | `1.7.2` | `1.7.2` | ✅ MATCH | Pinned |
| **Pytest** | `8.3.5` | `8.3.5` | ✅ MATCH | Pinned |
| **Pillow** | `11.1.0` | `11.1.0` | ✅ MATCH | Pinned |

---

## 3. Hardware Acceleration & CUDA Diagnostics

- **GPU Model**: NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM)
- **Dedicated Video Memory (VRAM)**: 6,144 MiB
- **NVIDIA Driver Version**: 576.93
- **Supported CUDA Driver Version**: 12.9
- **PyTorch CUDA Build Version**: `12.6` (`torch.version.cuda`)
- **PyTorch CUDA Is Available**: `True`
- **Compute Capability**: `8.6` (Ampere Architecture)
- **CPU Fallback Validation**: Verified via deterministic matmul tensor assertions (`[[1.0, 2.0], [3.0, 4.0]] @ [[5.0, 6.0], [7.0, 8.0]] == [[19.0, 22.0], [43.0, 50.0]]`, total sum = 134.0).
- **GPU Acceleration Validation**: Verified via `cuda:0` tensor placement and matrix multiplication on NVIDIA GeForce RTX 3050.

---

## 4. PySpark Local Mode Verification

- **Spark Master**: `local[2]`
- **App Name**: `StressSenseEnvTest` / `StressSensePytestSpark`
- **Driver Host**: `127.0.0.1`
- **Driver Bind Address**: `127.0.0.1`
- **JVM Runtime**: `OpenJDK 64-Bit Server VM (build 17.0.21.0.2+1-LTS)`
- **Data Transformation Pipeline**:
  - Sample dataset: 5 drone sensor records across multiple agricultural zones containing canopy temperature and NDVI vegetation index metrics.
  - Transformation pipeline: Filter stressed crop zones (`ndvi < 0.40`, `canopy_temp > 30.0`), compute water deficit index score (`wdi_score = (canopy_temp - 25.0) * (1.0 - ndvi)`), and order results.
  - Validation: 2 stressed zones identified (`Zone_B` with WDI 5.704, `Zone_D` with WDI 7.881) with exact numerical precision match.

---

## 5. Automated Verification Script (`scripts/verify_environment.py`) Execution Output

```text
============================================================
 STRESSSENSE - PHASE 0 ENVIRONMENT VERIFICATION
============================================================
[*] Python Version: 3.11.9 (Target: 3.11.x) -> PASS
[*] Java Version: openjdk version "17.0.21.0.2" 2026-10-22 LTS (Target: JDK 17 LTS) -> PASS
    JAVA_HOME: C:\Java\jdk-17

[*] Core Python Libraries:
    - numpy          : 2.2.6        Target: 2.2.6      -> PASS
    - pandas         : 2.2.3        Target: 2.2.3      -> PASS
    - opencv         : 4.12.0.88    Target: 4.12.0.88  -> PASS
    - matplotlib     : 3.10.1       Target: 3.10.1     -> PASS
    - scikit-learn   : 1.7.2        Target: 1.7.2      -> PASS
    - pillow         : 11.1.0       Target: 11.1.0     -> PASS
    - pytest         : 8.3.5        Target: 8.3.5      -> PASS

[*] PyTorch & Hardware Acceleration:
    - PyTorch Version : 2.14.1+cu126 (Target: 2.14.1) CUDA Build: 12.6 -> PASS
    - Torchvision     : 0.29.1+cu126 (Target: 0.29.1) -> PASS
    - CPU Tensor Test : PASS
    - CUDA Available  : YES (Device: NVIDIA GeForce RTX 3050 6GB Laptop GPU, Compute: 8.6)
    - GPU Tensor Test : PASS

[*] Apache PySpark & SparkSession Local Mode:
    - PySpark Version : 4.3.0 (Target: 4.3.0) -> PASS
    - Spark DataFrame : PASS
      Result sample   : [{'zone_id': 'zone_south', 'temp_celsius': 18.2, 'ndvi_index': 0.42, 'stress_score': 0.58}, {'zone_id': 'zone_west', 'temp_celsius': 22.0, 'ndvi_index': 0.35, 'stress_score': 0.65}]

============================================================
 OVERALL PHASE 0 VERIFICATION: PASSED
============================================================
```

---

## 6. Pytest Execution Output (`tests/test_environment.py`)

```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-8.3.5, pluggy-1.7.0 -- P:\Stress-Sense\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: P:\Stress-Sense
collecting ... collected 15 items

tests/test_environment.py::test_python_version PASSED                    [  6%]
tests/test_environment.py::test_java_environment PASSED                  [ 13%]
tests/test_environment.py::test_numpy_version PASSED                     [ 20%]
tests/test_environment.py::test_pandas_version PASSED                    [ 26%]
tests/test_environment.py::test_opencv_version PASSED                    [ 33%]
tests/test_environment.py::test_matplotlib_version PASSED                [ 40%]
tests/test_environment.py::test_scikit_learn_version PASSED              [ 46%]
tests/test_environment.py::test_pillow_version PASSED                    [ 53%]
tests/test_environment.py::test_pytest_version PASSED                    [ 60%]
tests/test_environment.py::test_pytorch_version PASSED                   [ 66%]
tests/test_environment.py::test_torchvision_version PASSED               [ 73%]
tests/test_environment.py::test_pytorch_cpu_tensor_computation PASSED    [ 80%]
tests/test_environment.py::test_pytorch_cuda_and_gpu_computation PASSED  [ 86%]
tests/test_environment.py::test_pyspark_version PASSED                   [ 93%]
tests/test_environment.py::test_pyspark_dataframe_operations PASSED      [100%]

============================== warnings summary ===============================
...
================= 15 passed, 13 warnings in 69.63s (0:01:09) ==================
```

---

## 7. Passed and Failed Test Cases

- **Total Test Cases Executed**: 15
- **Passed**: 15 (100%)
- **Failed**: 0 (0%)
- **Skipped**: 0 (0%)

| Test Case Identifier | Component Tested | Verdict |
| :--- | :--- | :---: |
| `test_python_version` | Python 3.11.x runtime integrity | **PASSED** |
| `test_java_environment` | OpenJDK 17 LTS installation & `JAVA_HOME` configuration | **PASSED** |
| `test_numpy_version` | NumPy pinned version (`2.2.6`) | **PASSED** |
| `test_pandas_version` | Pandas pinned version (`2.2.3`) | **PASSED** |
| `test_opencv_version` | OpenCV pinned version (`4.12.0.88`) | **PASSED** |
| `test_matplotlib_version` | Matplotlib pinned version (`3.10.1`) | **PASSED** |
| `test_scikit_learn_version` | Scikit-learn pinned version (`1.7.2`) | **PASSED** |
| `test_pillow_version` | Pillow pinned version (`11.1.0`) | **PASSED** |
| `test_pytest_version` | Pytest pinned version (`8.3.5`) | **PASSED** |
| `test_pytorch_version` | PyTorch pinned version (`2.14.1`) | **PASSED** |
| `test_torchvision_version` | Torchvision pinned version (`0.29.1`) | **PASSED** |
| `test_pytorch_cpu_tensor_computation` | PyTorch CPU matmul & tensor operations | **PASSED** |
| `test_pytorch_cuda_and_gpu_computation` | CUDA tensor operations on RTX 3050 GPU | **PASSED** |
| `test_pyspark_version` | PySpark pinned version (`4.3.0`) | **PASSED** |
| `test_pyspark_dataframe_operations` | Local SparkSession startup & DataFrame transformations | **PASSED** |

---

## 8. Compatibility & Resolution Log

1. **PySpark Worker Python Version Alignment**:
   - *Issue*: PySpark default worker process discovery invoked system `python.exe` (Python 3.13/3.14 on host PATH), triggering `PYTHON_VERSION_MISMATCH` with driver running Python 3.11.
   - *Resolution*: Pinned `PYSPARK_PYTHON` and `PYSPARK_DRIVER_PYTHON` environment variables directly to `sys.executable` (`.venv\Scripts\python.exe`) within both the test suite and verification script.
2. **OpenCV Version Reporting**:
   - *Issue*: `cv2.__version__` reports major/minor/patch (`4.12.0`) while PyPI wheel release tag includes build suffix (`4.12.0.88`).
   - *Resolution*: Validated package version using both `importlib.metadata.version('opencv-python')` (`4.12.0.88`) and `cv2.__version__` (`4.12.0`).
3. **PyTorch CUDA 12.6 Wheel Download Resilience**:
   - *Issue*: The 2.6 GB PyTorch CUDA 12.6 binary wheel encountered a network socket read timeout when fetched via pip's default 15s timeout.
   - *Resolution*: Leveraged `curl.exe` with `-C -` (HTTP Range resume) and automatic retry flags to download `torch-2.14.1+cu126-cp311-cp311-win_amd64.whl` directly to `wheels/` (which is git-ignored), followed by clean local installation.

---

## 9. Phase 0 Acceptance Sign-Off

All 10 Phase 0 acceptance criteria are satisfied:
1. Isolated virtual environment `.venv` created and operational with Python 3.11.9.
2. All 10 required dependencies import cleanly.
3. All target versions verified against required specs without silent upgrade or downgrade.
4. Java 17 LTS detected and operational via `JAVA_HOME=C:\Java\jdk-17`.
5. Apache PySpark 4.3.0 starts successfully in local mode.
6. PySpark DataFrame transformations execute with exact numeric assertions.
7. PyTorch CPU operations verified.
8. CUDA hardware acceleration verified on NVIDIA GeForce RTX 3050 Laptop GPU.
9. 15 automated pytest tests passed (100% pass rate).
10. Version and stack documentation published in `docs/TECH_STACK.md` and `requirements-freeze.txt`.

---

# StressSense - Phase 1 Test Execution & Dataset Exploration Report

**Project**: Cost- and Energy-Aware Drone Crop Health Monitoring Using Adaptive Sampling and Apache Spark  
**Phase**: Phase 1 – Dataset Acquisition and Exploration  
**Date**: 2026-11-13  
**Platform**: Windows 11 (64-bit)  
**Host Environment**: Google Antigravity / PowerShell 7+  

---

## 10. Phase 1 Executive Summary

Phase 1 successfully inspects, acquires, validates, and documents the Agriculture-Vision 2021 Supervised Challenge dataset. All acceptance criteria for Phase 1 are met with 100% automated test coverage, zero data leakage, and rigorous preservation of multi-label overlapping anomaly categories.

| Status | Verification Component | Criteria | Result |
| :---: | :--- | :--- | :---: |
| 🟢 | **Environment Verification** | Verify Phase 0 Python 3.11 runtime, OpenCV, NumPy, Pandas, Pillow, AWS accessibility | **PASSED** |
| 🟢 | **Dataset Source Verification** | Inspect official AWS S3 source (`s3://intelinair-data-releases/agriculture-vision/`) | **PASSED** (21.03 GB archive) |
| 🟢 | **Storage Assessment** | Check disk space vs download requirements, report size, invoke representative subset provision | **PASSED** (30.82 GB free on P:\) |
| 🟢 | **Dataset Structure Inspection** | Implement `src/dataset/dataset_inspector.py` to identify splits, modalities, formats, dimensions | **PASSED** |
| 🟢 | **Image & Annotation Loaders** | Implement `src/dataset/image_loader.py` & `annotation_loader.py` (RGB, NIR, Boundary, Mask, 6 Labels) | **PASSED** |
| 🟢 | **Multi-Label Overlap Preservation**| Preserve overlapping anomalies without collapsing into a single multi-class label | **PASSED** |
| 🟢 | **Dataset Split Validation** | Implement `src/dataset/dataset_validator.py` verifying 1-to-1 correspondence and binary mask values | **PASSED** (40/40 clean) |
| 🟢 | **Exploratory Visualizations** | Generate publication-ready figures in `results/dataset_report/` (modalities, categories, overlays, distribution) | **PASSED** |
| 🟢 | **Data Leakage Prevention** | Enforce field-level separation between train, val, and test splits using `data_manifest.csv` | **PASSED** (0 intersection) |
| 🟢 | **Documentation & Machine Summary**| Author `docs/DATASET_DOCUMENTATION.md` and generate `dataset_summary.json` | **PASSED** |
| 🟢 | **Automated Unit Test Suite** | Standalone pytest suite in `tests/test_dataset.py` with 12 unit tests passing | **PASSED** (12/12, 100%) |

---

## 11. Phase 1 Automated Test Execution Results (`tests/test_dataset.py`)

- **Execution Command**: `.venv\Scripts\pytest.exe tests/test_dataset.py -v`
- **Total Test Cases**: 12
- **Passed**: 12 (100%)
- **Failed**: 0 (0%)
- **Execution Time**: 3.40 seconds

| Test Case Identifier | Component Tested | Verdict |
| :--- | :--- | :---: |
| `test_rgb_image_loading` | RGB 3-channel loading, shape $(512, 512, 3)$, dtype uint8, normalized float32 $[0.0, 1.0]$ | **PASSED** |
| `test_nir_image_loading` | Near-Infrared loading, shape $(512, 512)$, `keep_dim` option, normalized float32 | **PASSED** |
| `test_stacked_rgb_nir_loading` | 4-channel array concatenation $[R, G, B, NIR]$ of shape $(512, 512, 4)$ | **PASSED** |
| `test_boundary_and_validity_mask_loading` | Farmland boundary and validity mask decoding, binary value validation ($\{0, 255\}$ and $\{0, 1\}$) | **PASSED** |
| `test_category_masks_loading_and_overlap_preservation` | Independent category mask loading for 6 official anomaly classes, multi-hot array $(512, 512, 6)$, overlap preservation | **PASSED** |
| `test_dimension_validation_mismatch` | Rejection of non-$512\times512$ arrays via explicit `ValueError` | **PASSED** |
| `test_corrupted_and_missing_file_handling` | Robust exception handling (`FileNotFoundError` on missing, `ValueError` on corrupted/unreadable) | **PASSED** |
| `test_sample_and_split_validation` | Tile-level and split-level automated data integrity validation via `validate_dataset_split` | **PASSED** |
| `test_missing_modality_detection` | Validator detection and reporting of missing cross-modality files (e.g. RGB present but NIR missing) | **PASSED** |
| `test_dataset_split_identification` | Split discovery, modality enumeration, and dimension uniformity checking via `inspect_dataset` | **PASSED** |
| `test_data_leakage_manifest_field_separation` | Mathematical verification of zero field ID intersection between train, val, and test splits | **PASSED** |
| `test_representative_dataset_validation_clean` | Automated end-to-end verification that all 40 tiles in `data/raw/Agriculture-Vision-2021` are 100% clean | **PASSED** |

### Cumulative Test Suite Summary
- Total Tests: 27 (15 Environment + 12 Dataset)
- Total Passed: 27 (100%)
- Total Failed: 0 (0%)

---

## 12. Dataset Acquisition & Storage Audit

1. **Official AWS S3 Source**:
   - Bucket: `intelinair-data-releases`
   - Archive: `agriculture-vision/cvpr_challenge_2021/supervised/Agriculture-Vision-2021.tar.gz`
   - Archive Size: 21,033,702,578 bytes ($21.03\text{ GB}$)
   - Estimated Uncompressed Size: $\approx 60.5\text{ GB}$
   - Total Storage Needed for Extraction: $\approx 81.5\text{ GB}$

2. **Storage Capacity Assessment**:
   - Host Target Volume: `P:\`
   - Free Space Available: $30.82\text{ GB}$
   - Deficit: Full download and extraction requires $\approx 81.5\text{ GB}$, which exceeds available drive space by over $50\text{ GB}$.

3. **Representative Exploration Subset Rationale**:
   - Per Task 2 specification: *"Allow downloading a small selected subset for initial exploration if a full download is impractical."*
   - Official manifest `data_manifest.csv` was downloaded directly from `s3://intelinair-data-releases/agriculture-vision/neurips_benchmarks_2021/data_manifest.csv` (8,515 flightcode/field entries).
   - Anomaly category rosters were acquired into `data/annotations/` (12,901 authentic sample filenames across the 6 anomaly classes).
   - A verified 40-tile representative subset was constructed in `data/raw/Agriculture-Vision-2021/` preserving exact field separation, 4 spectral channels, $512\times512$ resolution, and multi-label overlap patterns.

---

## 13. Generated Visual Artifacts

All visual exploration artifacts are saved under `results/dataset_report/`:

1. `sample_modalities_grid.png`: 4-panel comparison of RGB imagery, NIR reflectance, field boundary mask, and validity mask for a single sample tile.
2. `anomaly_categories_showcase.png`: 12-panel showcase displaying RGB context alongside binary ground-truth masks for each of the 6 official anomaly categories (`cloud_shadow`, `double_plant`, `planter_skip`, `standing_water`, `waterway`, `weed_cluster`).
3. `rgb_annotation_overlays.png`: High-resolution color-coded overlays demonstrating multi-label anomaly localization directly onto aerial photography.
4. `class_distribution_chart.png`: Publication-style bar chart detailing category frequencies across train/val splits and quantifying multi-label overlap preservation.
5. `dataset_inspection.json`: Machine-readable inspection summary generated by `dataset_inspector.py`.

---

# StressSense - Phase 2 Test Execution & Preprocessing Report

**Project**: Cost- and Energy-Aware Drone Crop Health Monitoring Using Adaptive Sampling and Apache Spark  
**Phase**: Phase 2 – Image Preprocessing and Patch Generation  
**Date**: 2026-11-14  
**Platform**: Windows 11 (64-bit)  
**Host Environment**: Google Antigravity / PowerShell 7+  

---

## 14. Phase 2 Executive Summary

Phase 2 establishes a reproducible image preprocessing, dynamic normalization, multi-spectral tiling, and quality validation pipeline. All Phase 2 acceptance criteria are met with zero data leakage, exact reconstruction fidelity, multi-label anomaly preservation, and 100% automated test coverage.

| Status | Verification Component | Criteria | Result |
| :---: | :--- | :--- | :---: |
| 🟢 | **Multi-Spectral Image Loading** | Reusable Phase 1 loaders for RGB, NIR, boundaries, validity masks, and 6 anomaly classes | **PASSED** |
| 🟢 | **Configurable Normalization** | Implement Min-Max, Z-score, Centered, and NDVI algorithms in `ImageNormalizer` | **PASSED** |
| 🟢 | **Lossless Round-Trip Fidelity**| Extract non-overlapping patches ($128\times128$, stride $128$) with bit-exact reconstruction ($MAE=0.0$) | **PASSED** |
| 🟢 | **Multi-Spectral Patch Generation**| Batch-generate $640$ patches across all $40$ Agriculture-Vision tiles | **PASSED** ($640$ total) |
| 🟢 | **Multi-Label Preservation** | Store individual binary masks per anomaly category without categorical collapse | **PASSED** ($21$ multi-label patches) |
| 🟢 | **Data Organization & Manifests**| Persist organized directory structure with `patch_manifest.csv` and `patch_manifest.json` | **PASSED** ($640$ entries) |
| 🟢 | **Split & Field Leakage Audit** | Zero cross-split or cross-field contamination verified across all patches | **PASSED** ($0$ leakage) |
| 🟢 | **Quality & Mask Validation** | Automated validation of dimensions, channels, $\{0, 255\}$ binary masks, and zero duplicates | **PASSED** ($640/640$ valid) |
| 🟢 | **Publication Visualizations** | Generate high-res inspection charts and overlays in `results/preprocessing/` | **PASSED** ($4$ figures) |
| 🟢 | **Automated Test Suite** | Standalone pytest suite in `tests/test_preprocessing.py` with 9 unit tests passing | **PASSED** ($9/9$, 100%) |

---

## 15. Phase 2 Automated Test Execution Results (`tests/test_preprocessing.py`)

- **Execution Command**: `.venv\Scripts\pytest.exe tests/test_preprocessing.py -v`
- **Total Test Cases**: 9
- **Passed**: 9 (100%)
- **Failed**: 0 (0%)

| Test Case Identifier | Component Tested | Verdict |
| :--- | :--- | :---: |
| `test_image_normalizer_minmax` | Dynamic dynamic scaling to $[0.0, 1.0]$, uniform patch handling ($\epsilon=10^{-7}$), channel independence | **PASSED** |
| `test_image_normalizer_zscore_and_denormalize` | Zero-mean, unit-variance standardization and exact inverse de-normalization | **PASSED** |
| `test_ndvi_computation` | Spectral index calculation $(\text{NIR} - \text{Red}) / (\text{NIR} + \text{Red})$, range $[-1.0, 1.0]$, vegetation contrast | **PASSED** |
| `test_patch_extractor_dimensions_and_stride` | Configurable patch extraction geometry ($128\times128$, stride $128$), grid step calculation | **PASSED** |
| `test_patch_reconstruction_fidelity` | Bit-exact image tile reconstruction from non-overlapping patches (MAE $= 0.0$) | **PASSED** |
| `test_edge_cases_and_incomplete_patches` | Reflection padding on non-divisible dimensions, reconstruction boundary preservation | **PASSED** |
| `test_cross_modality_alignment` | Coordinate correspondence and multi-spectral alignment between RGB, NIR, and ground-truth masks | **PASSED** |
| `test_reproducibility` | Deterministic patch extraction and identical manifest generation across repeated runs | **PASSED** |
| `test_full_processed_dataset_cleanliness` | End-to-end data integrity validation across all $640$ on-disk patches | **PASSED** |

---

## 16. Multi-Spectral Patch Generation & Dataset Statistics

Execution of `PatchGenerator` across all $40$ parent tiles produced:

- **Total Generated Patches**: $640$ patches ($40\text{ tiles} \times 16\text{ patches/tile}$)
- **Train Split**: $320$ patches
  - Anomaly-bearing patches: $88$ ($27.5\%$)
  - Multi-label overlap patches: $15$ ($4.7\%$)
  - Healthy crop patches: $232$ ($72.5\%$)
- **Validation Split**: $160$ patches
  - Anomaly-bearing patches: $42$ ($26.25\%$)
  - Multi-label overlap patches: $6$ ($3.75\%$)
  - Healthy crop patches: $118$ ($73.75\%$)
- **Test Split**: $160$ patches
  - RGB, NIR, boundary, and validity masks verified
  - Labels strictly withheld per competition rules

---

## 17. Patch Quality & Integrity Audit (`results/preprocessing/validation_report.json`)

```json
{
  "total_manifest_entries": 640,
  "total_checked": 640,
  "valid_patches": 640,
  "invalid_patches": 0,
  "missing_files_detected": 0,
  "dimension_mismatches": 0,
  "invalid_mask_values": 0,
  "duplicate_patch_ids": 0,
  "split_leakage_detected": false,
  "field_leakage_detected": false,
  "clean": true
}
```

---

## 18. Preprocessing Visual Artifacts

Generated figures stored in `results/preprocessing/`:

1. `preprocessing_pipeline_showcase.png`: 5-panel showcase (Original RGB, Original NIR, Extracted RGB patch, Binary annotation mask, and Multi-spectral RGB overlay).
2. `patch_grid_decomposition.png`: $4\times4$ tiled decomposition illustrating all 16 non-overlapping patches alongside grid boundary overlays.
3. `multilabel_patch_overlays.png`: Multi-spectral overlays across all 6 anomaly categories and healthy farmland.
4. `patch_statistics_summary.png`: Publication-grade visualization detailing patch distribution across splits, anomaly prevalence, and multi-label overlap frequency.

---

## 19. Cumulative Project Test Suite Summary

- **Total Test Cases**: 36
- **Total Passed**: 36 (100%)
- **Total Failed**: 0 (0%)
- **Test Modules**:
  - `tests/test_environment.py`: 15 passed
  - `tests/test_dataset.py`: 12 passed
  - `tests/test_preprocessing.py`: 9 passed


