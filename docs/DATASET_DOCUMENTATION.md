# Agriculture-Vision 2021 Dataset Documentation

## 1. Overview & Dataset Identity

- **Dataset Name**: Agriculture-Vision 2021 Supervised Challenge Dataset
- **Edition / Version**: 2021.0 (CVPR 2021 Agriculture-Vision Challenge / NeurIPS 2021 Benchmark)
- **Official Portal**: [https://www.agriculture-vision.com/agriculture-vision-2021/dataset-2021](https://www.agriculture-vision.com/agriculture-vision-2021/dataset-2021)
- **Official AWS S3 Source**: `s3://intelinair-data-releases/agriculture-vision/cvpr_challenge_2021/supervised/Agriculture-Vision-2021.tar.gz`
- **Direct HTTPS Endpoint**: `https://intelinair-data-releases.s3.amazonaws.com/agriculture-vision/cvpr_challenge_2021/supervised/Agriculture-Vision-2021.tar.gz`
- **Field Manifest Source**: `s3://intelinair-data-releases/agriculture-vision/neurips_benchmarks_2021/data_manifest.csv`
- **Primary Publication Citation**:
  > Chiu, M. T., Xu, X., Wei, Y., Wang, Z., Schwing, A. G., & Huang, T. S. (2020). *Agriculture-Vision: A Large Aerial Image Database for Agricultural Pattern Analysis*. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2020), pp. 2828–2838.

---

## 2. Dataset Structure

The dataset directory layout follows the official challenge specification:

```text
Agriculture-Vision-2021/
├── train/
│   ├── images/
│   │   ├── rgb/           # 512x512 RGB aerial orthomosaic tiles (.jpg)
│   │   └── nir/           # 512x512 Near-Infrared grayscale tiles (.jpg)
│   ├── boundaries/        # 512x512 Farmland boundary masks (.png, 255 inside, 0 outside)
│   ├── masks/             # 512x512 Validity / flight coverage masks (.png)
│   └── labels/            # Ground truth anomaly segmentation masks
│       ├── cloud_shadow/  # Binary PNG mask (255 = presence, 0 = background)
│       ├── double_plant/
│       ├── planter_skip/
│       ├── standing_water/
│       ├── waterway/
│       └── weed_cluster/
│
├── val/
│   ├── images/
│   │   ├── rgb/
│   │   └── nir/
│   ├── boundaries/
│   ├── masks/
│   └── labels/
│       ├── cloud_shadow/
│       ├── double_plant/
│       ├── planter_skip/
│       ├── standing_water/
│       ├── waterway/
│       └── weed_cluster/
│
└── test/
    ├── images/
    │   ├── rgb/
    │   └── nir/
    ├── boundaries/
    └── masks/
    # Note: Labels are withheld for test split as per challenge leaderboard evaluation.
```

### Tile Naming Convention

Every tile filename strictly adheres to the official naming pattern:

$$\text{<flightcode>}\_\text{<x1>}-\text{<y1>}-\text{<x2>}-\text{<y2>}.\text{<ext>}$$

Where:
- `flightcode`: Unique identifier of the drone flight orthomosaic (e.g. `KEP4MBVE7`).
- `x1, y1`: Upper-left corner pixel coordinate within the field master orthomosaic.
- `x2, y2`: Lower-right corner pixel coordinate within the field master orthomosaic ($x_2 - x_1 = 512$, $y_2 - y_1 = 512$).
- `ext`: `.jpg` for aerial images (RGB, NIR); `.png` for masks and labels.

---

## 3. Image Modalities & Channels

The dataset provides 4 input spectral channels captured by drone-mounted multispectral sensors:

| Modality | Channels | Format | Shape | Description & Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **RGB** | 3 | JPEG | $(512, 512, 3)$ | True-color aerial photographic imagery (Red, Green, Blue). Reflects visible light absorption and crop row geometry. |
| **NIR** | 1 | JPEG | $(512, 512)$ | Near-Infrared band (~840 nm). Healthy green vegetation cellular structure strongly reflects NIR ($>180$), bare soil exhibits moderate reflectance ($~100$), and standing water strongly absorbs NIR ($<40$). |
| **Stacked 4-Channel** | 4 | In-Memory / Array | $(512, 512, 4)$ | Concatenation of [Red, Green, Blue, NIR]. Normalized to $[0.0, 1.0]$ float32 for model ingestion. |
| **Boundaries** | 1 | PNG | $(512, 512)$ | Binary mask where $255$ denotes pixels falling within the legal agronomic field boundary, and $0$ denotes background outside the field. |
| **Validity Masks** | 1 | PNG | $(512, 512)$ | Binary mask where $255$ denotes valid sensor swath coverage (pixels suitable for evaluation), and $0$ denotes sensor artifacts, padding, or incomplete flight edges. |

---

## 4. Anomaly Categories & Multi-Label Overlap Preservation

The challenge defines **6 official agricultural anomaly categories**:

| Category ID | Anomaly Name | Full Dataset Count | Visual & Agronomic Characteristics |
| :---: | :--- | :---: | :--- |
| 1 | `cloud_shadow` | 927 | Transient cloud shadows causing severe localized attenuation of visible and NIR solar irradiance. |
| 2 | `double_plant` | 1,608 | Excessive seeding overlap leading to hyper-dense crop rows that compete for nutrients and moisture. |
| 3 | `planter_skip` | 270 | Mechanical planting failures resulting in barren, unseeded linear gaps exposing bare soil. Highly imbalanced/rare. |
| 4 | `standing_water` | 813 | Surface ponding and puddles from heavy precipitation or poor drainage. Characterized by very low NIR reflectance. |
| 5 | `waterway` | 1,556 | Natural or engineered drainage ditches and erosion gullies traversing farmland. |
| 6 | `weed_cluster` | 7,727 | Invasive weed infestations displaying irregular canopy height, high chlorophyll density, and spectral divergence from crop rows. |

### Strict Multi-Label Overlap Policy
> [!IMPORTANT]
> **Preservation of Overlapping Anomalies**:
> Agricultural anomalies frequently co-occur (e.g. `cloud_shadow` over `weed_cluster`, or `standing_water` in a `waterway`). **Classes MUST NEVER be merged into a single mutually exclusive multi-class representation.**
> The loaders in `src/dataset/annotation_loader.py` load each category as an independent binary channel ($512 \times 512 \times 6$), preserving all co-occurrence patterns without information loss.

---

## 5. Dataset Statistics

### Full Challenge Corpus (AWS S3)
- **Total Challenge Images**: 21,061 aerial tiles
- **Total Unique Fields**: 3,432 fields across multiple Midwestern US states
- **Total Archive Compressed Size**: 21,033,702,578 bytes ($21.03\text{ GB}$)
- **Estimated Uncompressed Size**: $\approx 60.5\text{ GB}$
- **Total Storage Required for Extraction**: $\approx 81.5\text{ GB}$
- **Tiles Containing Anomalies**: 12,901 tiles ($61.3\%$)
- **Healthy Farmland Tiles**: 8,160 tiles ($38.7\%$)

### Representative Development Subset (`data/raw/Agriculture-Vision-2021/`)
Due to local drive storage limits ($P:\backslash$ has $30.82\text{ GB}$ free, which is strictly insufficient for the $81.5\text{ GB}$ archive), a clean, representative subset was created adhering to the official manifest field IDs and split separation:

| Metric | Train Split | Validation Split | Test Split | Total Subset |
| :--- | :---: | :---: | :---: | :---: |
| **Number of Tiles** | 20 | 10 | 10 | **40** |
| **RGB Images** | 20 | 10 | 10 | **40** |
| **NIR Images** | 20 | 10 | 10 | **40** |
| **Field Boundaries** | 20 | 10 | 10 | **40** |
| **Validity Masks** | 20 | 10 | 10 | **40** |
| **Ground Truth Labels** | Available (6 classes) | Available (6 classes) | Withheld | **30 Annotated** |
| **Multi-Label Overlap Tiles** | 6 | 2 | N/A | **8** |
| **Healthy Tiles (No Anomaly)**| 5 | 2 | 10 | **17** |
| **Dimension Uniformity** | 100% ($512\times512$) | 100% ($512\times512$) | 100% ($512\times512$) | **100% Uniform** |

#### Subset Category Occurrence Frequencies
- `cloud_shadow`: 6 tiles (4 train, 2 val)
- `double_plant`: 4 tiles (3 train, 1 val)
- `planter_skip`: 4 tiles (3 train, 1 val)
- `standing_water`: 6 tiles (4 train, 2 val)
- `waterway`: 5 tiles (3 train, 2 val)
- `weed_cluster`: 7 tiles (5 train, 2 val)

---

## 6. Data Leakage Prevention

Field-level separation is strictly enforced to prevent data leakage:
- Aerial farmland tiles cut from the same field orthomosaic share identical soil properties, crop row angles, planting dates, and weather conditions.
- Splitting tiles randomly at the image level introduces severe data leakage (models memorizing field-specific soil textures).
- In StressSense, all splits are strictly partitioned by **flightcode / field ID** as recorded in `data/raw/data_manifest.csv`:
  - 2,060 fields allocated exclusively to `train`
  - 686 fields allocated exclusively to `val`
  - 686 fields allocated exclusively to `test`
- No field ID in `train` appears in `val` or `test`.

---

## 7. Licensing & Terms of Use

- **License**: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
- **Data Provider**: Intelinair Inc. & University of Illinois at Urbana-Champaign (UIUC)
- **Official Terms**: Preserved in [Agriculture-Vision Dataset Terms of Use.pdf](file:///P:/Stress-Sense/docs/Agriculture-Vision_Dataset_Terms_of_Use.pdf)
- **Usage Restrictions**: Academic and non-commercial research purposes only. Commercial redistribution, re-hosting, or sub-licensing of raw data is prohibited without prior written consent from Intelinair.

---

## 8. Download Instructions

### Prerequisites
- Python 3.11+ in active `.venv`
- Disk Space: $\ge 85\text{ GB}$ for full dataset, or $\ge 500\text{ MB}$ for representative subset
- Optional: AWS CLI (`aws`) with `--no-sign-request`

### Command Line Download (Full Dataset)
```bash
# Using AWS CLI (No AWS credentials needed)
aws s3 cp s3://intelinair-data-releases/agriculture-vision/cvpr_challenge_2021/supervised/Agriculture-Vision-2021.tar.gz ./data/raw/ --no-sign-request

# Using curl directly from HTTPS endpoint
curl -O https://intelinair-data-releases.s3.amazonaws.com/agriculture-vision/cvpr_challenge_2021/supervised/Agriculture-Vision-2021.tar.gz
```

### Representative Subset Generation
```bash
# Generate the official representative subset (40 tiles, 4 channels, 6 anomaly classes, 0 data leakage)
.venv\Scripts\python.exe scripts/build_representative_dataset.py

# Run automated inspection
.venv\Scripts\python.exe -m src.dataset.dataset_inspector --data_dir data/raw/Agriculture-Vision-2021

# Generate visual exploration reports
.venv\Scripts\python.exe scripts/generate_exploration_report.py
```

---

## 9. Known Limitations

1. **Extreme Class Imbalance**: In the full corpus, `weed_cluster` accounts for $60\%$ of all annotations, while `planter_skip` represents only $2.1\%$. Loss weighting or focal loss will be necessary during Phase 2 model training.
2. **Atmospheric Artifacts**: `cloud_shadow` creates non-stationary spectral degradation that can mask subtle soil moisture differences.
3. **Sensor Co-registration Tolerances**: Slight sub-pixel displacement between RGB and NIR bands may occur near outer orthomosaic boundaries.
4. **Resolution Boundaries**: $512 \times 512$ tile cropping can bisect large natural waterways or elongated planter skips at tile edges.
