# StressSense Phase 2 – Image Preprocessing and Patch Generation Documentation

## 1. Executive Summary

Phase 2 establishes a high-performance, mathematically rigorous, and fully reproducible image preprocessing and patch generation engine for the **StressSense** aerial drone crop health monitoring system.

Aerial orthomosaics from the Agriculture-Vision 2021 dataset possess large spatial dimensions ($512 \times 512$ pixels per challenge tile) that contain high-resolution agricultural patterns, subtle crop stress anomalies, and distinct non-farmland regions. To prepare these multi-spectral observations for edge-aware adaptive sampling and semantic segmentation:
1. **Multi-Spectral Inputs**: RGB ($3$-channel) and Near-Infrared (NIR, $1$-channel) imagery are loaded with spatial alignment, bit-depth verification, and coordinate consistency.
2. **Deterministic Tiling**: The $512 \times 512$ tiles are decomposed into non-overlapping $128 \times 128$ pixel patches (stride $128$) across all modalities (RGB, NIR, field boundaries, flight validity masks, and all six ground-truth anomaly masks).
3. **Multi-Label Preservation**: Anomaly categories (`cloud_shadow`, `double_plant`, `planter_skip`, `standing_water`, `waterway`, `weed_cluster`) are stored in isolated binary masks, completely preserving multi-label spatial overlaps.
4. **Leakage-Free Partitioning**: Every patch inherits the strict official challenge split (`train`, `val`, `test`) and underlying `field_id` from the parent tile, guaranteeing zero cross-split or cross-field data contamination.
5. **Lossless Persistence & Manifest Indexing**: Patches are saved as lossless PNGs (`.png`) accompanied by comprehensive metadata catalogs (`patch_manifest.csv` and `patch_manifest.json`).

---

## 2. Preprocessing Architecture & Pipeline Workflow

The preprocessing pipeline executes through modular components located in `src/preprocessing/`:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Phase 2 Pipeline Workflow                       │
└────────────────────────────────────────────────────────────────────────┘

 [ Agriculture-Vision 2021 Dataset ] (data/raw/Agriculture-Vision-2021)
                │
                ├── Train Split (20 tiles)
                ├── Val Split   (10 tiles)
                └── Test Split  (10 tiles)
                │
                ▼
  [ ImageLoader & AnnotationLoader ] (src/dataset/)
  ├── Validate file integrity & bit-depth
  ├── Verify dimensions: (512, 512)
  ├── Extract RGB (uint8 [0, 255]) & NIR (uint8 [0, 255])
  ├── Load Boundary & Validity masks (uint8 {0, 255})
  └── Load 6 individual Anomaly Category masks (uint8 {0, 255})
                │
                ▼
  [ PatchExtractor Engine ] (src/preprocessing/patch_extractor.py)
  ├── Window geometry: 128 x 128 pixels, stride 128 (4 x 4 = 16 patches/tile)
  ├── Coordinate tracking: (x_min, y_min, x_max, y_max)
  └── Exact reconstruction fidelity (mean absolute error == 0.0)
                │
                ▼
  [ Multi-Spectral Patch Generator ] (src/preprocessing/patch_generator.py)
  ├── Generate 640 patches across all splits
  ├── Compute patch-level metrics:
  │   ├── valid_pixel_fraction (inside valid mask)
  │   ├── boundary_pixel_fraction (inside field boundary)
  │   ├── anomaly_pixel_counts & anomaly_pixel_fractions
  │   └── multi-label anomaly category tagging
  └── Persist lossless PNGs in structured directories
                │
                ▼
  [ Manifest Generation ]
  ├── data/processed/manifests/patch_manifest.csv  (640 rows)
  └── data/processed/manifests/patch_manifest.json (640 entries)
                │
                ▼
  [ Quality & Integrity Validation ] (src/preprocessing/patch_validator.py)
  ├── Dimensionality check: (128, 128, 3) for RGB, (128, 128) for NIR/Masks
  ├── Binary value verification: {0, 255} for all masks
  ├── Split isolation & duplicate ID verification (0 duplicates, 0 leakage)
  └── Export results/preprocessing/validation_report.json
```

---

## 3. Mathematical Normalization Methods

The `ImageNormalizer` class in `src/preprocessing/normalizer.py` provides mathematical transformations designed for multi-spectral drone imagery:

### 3.1 Min-Max Normalization (Default Dynamic Scaling)
Scales pixel intensities to the interval $[0.0, 1.0]$:
$$X_{\text{norm}} = \frac{X - X_{\min}}{X_{\max} - X_{\min} + \epsilon}$$
where $\epsilon = 10^{-7}$ prevents division by zero in homogeneous regions (e.g., pure cloud shadow or uniform canopy).

### 3.2 Z-Score Standardization
Centers pixel intensities around zero with unit standard deviation:
$$Z = \frac{X - \mu}{\sigma + \epsilon}$$
Inverting the standardization enables loss-free visual reconstruction:
$$X = Z \cdot \sigma + \mu$$

### 3.3 Zero-Centered Normalization
Re-scales pixel intensities from $[0, 255]$ into $[-1.0, 1.0]$:
$$X_{\text{centered}} = \frac{2 \cdot X}{255.0} - 1.0$$

### 3.4 Normalized Difference Vegetation Index (NDVI)
NDVI is the fundamental remote sensing indicator for photosynthetic activity and crop vigor:
$$\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red} + \epsilon}$$
- In standard 8-bit inputs, NIR is extracted from the Near-Infrared channel, and Red is extracted from channel index `0` of the RGB image.
- Both channels are cast to `float32` in $[0.0, 1.0]$.
- Resulting NDVI values fall strictly in $[-1.0, 1.0]$. Dense healthy canopies produce values $> 0.6$, water and shadows yield negative or near-zero values.

---

## 4. Patch Generation Specifications & Geometry

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **Input Tile Dimensions** | $512 \times 512$ pixels | Standard Agriculture-Vision challenge resolution |
| **Target Patch Size** | $128 \times 128$ pixels | Standard sub-grid for lightweight edge-drone deep networks |
| **Stride** | $128$ pixels | Non-overlapping partitioning ($512 / 128 = 4$ steps in $X$ and $Y$) |
| **Patches per Tile** | $16$ ($4 \times 4$ grid) | Exact integer tiling without edge truncation |
| **Edge Padding Mode** | `reflect` (optional) | Available for arbitrary non-divisible dimensions |
| **Coordinate System** | `(x, y, x + width, y + height)` | 0-indexed pixel coordinates: $x \in \{0, 128, 256, 384\}$, $y \in \{0, 128, 256, 384\}$ |
| **Patch ID Format** | `<image_id>_patch_x{x:04d}_y{y:04d}` | Globally unique deterministic naming |

### Reconstruction Fidelity
The `reconstruct_from_patches` method in `PatchExtractor` takes the extracted $16$ patches and re-assembles the original $512 \times 512$ tile.
- **Round-trip Mean Absolute Error (MAE)**: $0.0$ (Bit-exact match verified by automated test `test_patch_reconstruction_fidelity`).

---

## 5. Output Directory Structure

All generated patches and manifests reside in `data/processed/`:

```text
data/processed/
├── manifests/
│   ├── patch_manifest.csv                # 640 rows with full metadata catalog
│   └── patch_manifest.json               # Structured JSON export
└── patches/
    ├── train/                            # 320 patches (20 parent tiles x 16)
    │   ├── rgb/                          # 128x128 3-channel PNGs
    │   ├── nir/                          # 128x128 1-channel PNGs
    │   ├── boundaries/                   # 128x128 binary PNG masks
    │   ├── masks/                        # 128x128 binary validity PNG masks
    │   └── labels/                       # 128x128 binary ground-truth PNG masks
    │       ├── cloud_shadow/
    │       ├── double_plant/
    │       ├── planter_skip/
    │       ├── standing_water/
    │       ├── waterway/
    │       └── weed_cluster/
    ├── val/                              # 160 patches (10 parent tiles x 16)
    │   ├── rgb/
    │   ├── nir/
    │   ├── boundaries/
    │   ├── masks/
    │   └── labels/
    │       ├── cloud_shadow/
    │       ├── double_plant/
    │       ├── planter_skip/
    │       ├── standing_water/
    │       ├── waterway/
    │       └── weed_cluster/
    └── test/                             # 160 patches (10 parent tiles x 16)
        ├── rgb/
        ├── nir/
        ├── boundaries/
        └── masks/
        # Labels withheld per official challenge evaluation guidelines
```

---

## 6. Manifest Schema (`patch_manifest.csv`)

The manifest contains $23$ comprehensive metadata columns per patch record:

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `patch_id` | String | Globally unique ID (`<image_id>_patch_x{x}_y{y}`) |
| `original_image_id` | String | Base identifier of the parent $512 \times 512$ tile |
| `field_id` | String | Agriculture-Vision field identifier (e.g. `2021_challenge_sample`) |
| `split` | String | Dataset split (`train`, `val`, `test`) |
| `x_coordinate` | Integer | Upper-left horizontal pixel offset ($0, 128, 256, 384$) |
| `y_coordinate` | Integer | Upper-left vertical pixel offset ($0, 128, 256, 384$) |
| `patch_size` | Integer | Patch dimension in pixels ($128$) |
| `stride` | Integer | Extraction stride ($128$) |
| `valid_pixel_fraction` | Float | Fraction of patch within flight sensor validity mask $[0.0, 1.0]$ |
| `boundary_pixel_fraction` | Float | Fraction of patch within active field boundary $[0.0, 1.0]$ |
| `has_anomaly` | Boolean | `True` if any ground-truth anomaly mask contains $\ge 1$ positive pixel |
| `anomaly_categories` | String | Comma-separated list of present anomalies, or `"none"` |
| `total_anomaly_pixels` | Integer | Total pixel area covered by anomalies |
| `cloud_shadow_pixels` | Integer | Pixel count for `cloud_shadow` category |
| `double_plant_pixels` | Integer | Pixel count for `double_plant` category |
| `planter_skip_pixels` | Integer | Pixel count for `planter_skip` category |
| `standing_water_pixels` | Integer | Pixel count for `standing_water` category |
| `waterway_pixels` | Integer | Pixel count for `waterway` category |
| `weed_cluster_pixels` | Integer | Pixel count for `weed_cluster` category |
| `rgb_path` | String | Relative filepath to patch RGB PNG |
| `nir_path` | String | Relative filepath to patch NIR PNG |
| `boundary_path` | String | Relative filepath to boundary mask PNG |
| `mask_path` | String | Relative filepath to validity mask PNG |

---

## 7. Data Validation & Integrity Audit

The automated validation suite (`PatchValidator` in `src/preprocessing/patch_validator.py`) was executed across all $640$ generated patches. The comprehensive audit yielded:

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

### Key Statistical Distributions:
- **Total Patches**: $640$ patches across $40$ full-scale agricultural tiles.
- **Train Split**: $320$ patches ($20$ tiles $\times 16$).
  - Anomaly-bearing patches: $88$ ($27.5\%$).
  - Multi-label overlap patches: $15$ ($4.7\%$).
  - Healthy crop / background patches: $232$ ($72.5\%$).
- **Val Split**: $160$ patches ($10$ tiles $\times 16$).
  - Anomaly-bearing patches: $42$ ($26.25\%$).
  - Multi-label overlap patches: $6$ ($3.75\%$).
  - Healthy crop / background patches: $118$ ($73.75\%$).
- **Test Split**: $160$ patches ($10$ tiles $\times 16$).
  - RGB, NIR, boundary, and validity masks verified.
  - Labels strictly withheld per competition rules.

---

## 8. Design Decisions and Engineering Trade-Offs

1. **Lossless PNG Persistence vs. Compressed Formats**:
   - *Decision*: Saved all processed patches as lossless `.png` files.
   - *Rationale*: JPEG compression introduces high-frequency DCT blocking artifacts that corrupt edge gradients, alter NDVI calculation, and blur binary anomaly boundaries. Lossless PNG preserves exact raw pixel values for downstream neural network training.
2. **Decoupled Normalization from On-Disk Storage**:
   - *Decision*: Persisted patches as un-normalized uint8 $[0, 255]$ images, providing on-the-fly normalization via `ImageNormalizer`.
   - *Rationale*: Storing floating-point normalized tensors requires $4\times$ to $8\times$ disk storage ($32$-bit float vs $8$-bit integer) and risks quantization loss if cast back to integer. In-memory normalization allows dynamic augmentation, per-channel z-score standardization, and arbitrary test-time scaling without re-generating dataset patches.
3. **Preservation of Overlapping Multi-Label Channels**:
   - *Decision*: Maintained independent directory structures and binary masks for each anomaly class.
   - *Rationale*: Agriculture-Vision contains overlapping categories (e.g. `cloud_shadow` cast over `weed_cluster`, or `standing_water` pooling in a `waterway`). Collapsing masks into a single multi-class categorical index would erase secondary labels and bias models toward dominant classes.
4. **Deterministic Coordinate Schema**:
   - *Decision*: Included exact top-left corner coordinates $(x, y)$ in both patch identifiers and manifest columns.
   - *Rationale*: Allows seamless multi-scale spatial indexing, spatial interpolation, and full-tile mosaic stitching during evaluation and drone flight reconstruction.

---

## 9. Reproducibility & Pipeline Execution Instructions

### Automated Script Execution
To re-generate all patches and manifests from scratch:

```powershell
# Activate Phase 0 virtual environment
.\.venv\Scripts\Activate.ps1

# Execute the complete patch generation pipeline
python -c "from src.preprocessing.patch_generator import PatchGenerator; generator = PatchGenerator(); manifest = generator.generate_all()"
```

### Validation Execution
To re-run the complete validation audit:

```powershell
python -c "from src.preprocessing.patch_validator import PatchValidator; validator = PatchValidator(); report = validator.validate_all(); print('Dataset clean:', report['clean'])"
```

### Visualization Generation
To re-render all visual analytics and inspection figures:

```powershell
python -c "from src.preprocessing.visualizer import PreprocessingVisualizer; visualizer = PreprocessingVisualizer(); visualizer.generate_all_visualizations()"
```

### Pytest Verification Suite
To run the automated preprocessing test suite:

```powershell
.\.venv\Scripts\pytest.exe tests\test_preprocessing.py -v
```
