# StressSense

**Cost- and Energy-Aware Drone Crop Health Monitoring Using Adaptive Sampling and Apache Spark**

---

## 🌾 Project Overview

**StressSense** is an intelligent agricultural monitoring system designed to optimize drone-based crop health assessment. By coupling computer vision and deep learning with edge adaptive sampling and Apache Spark distributed processing, StressSense maximizes spatial-temporal defect detection while minimizing flight energy expenditure and communication overhead.

## 🎯 Phases & Milestones

- [x] **Phase 0**: Environment & Version Configuration (Python 3.11, Java 17, Spark 4.3.0, PyTorch 2.14.1+cu126, RTX 3050 CUDA verified)
- [x] **Phase 1**: Dataset Acquisition & Exploration (Agriculture-Vision 2021 Supervised Challenge dataset, 4-channel imagery, multi-label anomaly preservation, zero data leakage)
- [x] **Phase 2**: Image Preprocessing and Patch Generation ($128 \times 128$ non-overlapping tiling, multi-spectral extraction, manifest cataloging, lossless PNG persistence, 100% clean data validation)
- [x] **Phase 3**: Edge-Aware Adaptive Sampling Engine (RSDI heuristic contrast scoring, 3-tier budget allocation, 61.76% valid canopy reduction, 59.78% proxy energy savings)
- [x] **Phase 4**: Apache Spark Distributed Pipeline & Interactive Dashboard (Stage 1.1 Fleet Validation, Stage 2 Distributed Evaluation, Stage 2 Final Scientific Audit, Stage 3 Interactive Dashboard Deployment)
- [ ] **Phase 5**: Full Integration, Benchmarking, and Evaluation

---

## 🌾 Phase 1: Agriculture-Vision Dataset Acquisition & Exploration

- **Dataset**: Agriculture-Vision 2021 Supervised Challenge Dataset
- **Official AWS Source**: `s3://intelinair-data-releases/agriculture-vision/cvpr_challenge_2021/supervised/`
- **Channels**: 4 channels (RGB aerial orthomosaic + Near-Infrared NIR band)
- **Resolution**: $512 \times 512$ pixels per tile
- **Categories (6)**: `cloud_shadow`, `double_plant`, `planter_skip`, `standing_water`, `waterway`, `weed_cluster`
- **Multi-Label Overlap**: Preserved across separate binary masks without merging classes.
- **Data Leakage Prevention**: Flightcode / field-level separation strictly enforced via `data_manifest.csv`.

### Dataset CLI Inspection & Validation
```powershell
# Run automated dataset structure inspection
.\.venv\Scripts\python.exe -m src.dataset.dataset_inspector --data_dir data/raw/Agriculture-Vision-2021

# Generate visual exploration reports
.\.venv\Scripts\python.exe scripts\generate_exploration_report.py

# Run dataset unit tests (12 tests)
.\.venv\Scripts\pytest.exe tests\test_dataset.py -v
```

---

## 🔬 Phase 2: Image Preprocessing and Patch Generation

- **Patch Dimensions**: $128 \times 128$ pixels, stride $128$ (16 non-overlapping patches per $512 \times 512$ tile)
- **Multi-Spectral Inputs**: RGB ($3$-channel), NIR ($1$-channel), boundary masks, flight validity masks, and 6 ground-truth anomaly masks
- **Total Generated Patches**: $640$ patches ($320$ train, $160$ val, $160$ test)
- **Reconstruction Fidelity**: Bit-exact image tile reconstruction from patches ($\text{MAE} = 0.0$)
- **Data Leakage Guarantee**: Exact preservation of official challenge split and field IDs with zero cross-split or cross-field leakage
- **Manifest Catalog**: `data/processed/manifests/patch_manifest.csv` and `patch_manifest.json` ($23$ metadata fields per record)
- **Validation Audit**: $640/640$ valid patches, $0$ invalid, $0$ duplicates, $0$ dimension mismatches, $100\%$ clean status

### Preprocessing CLI Execution & Validation
```powershell
# Generate all patches and manifests
.\.venv\Scripts\python.exe -c "from src.preprocessing.patch_generator import PatchGenerator; PatchGenerator().generate_all()"

# Run patch validation audit
.\.venv\Scripts\python.exe -c "from src.preprocessing.patch_validator import PatchValidator; PatchValidator().validate_all()"

# Generate visualization showcase
.\.venv\Scripts\python.exe -c "from src.preprocessing.visualizer import PreprocessingVisualizer; PreprocessingVisualizer().generate_all_visualizations()"

# Run preprocessing unit tests (9 tests)
.\.venv\Scripts\pytest.exe tests\test_preprocessing.py -v
```

---

## 📁 Project Directory Structure

| Component | Target Version | Installed Runtime | Purpose |
| :--- | :--- | :--- | :--- |
| **Operating System** | Windows 11 (64-bit) | Windows 11 | Host OS |
| **Python** | `3.11.x` | `3.11.9` | Core runtime |
| **Java JDK** | `17 LTS` | `17.0.21.0.2 LTS` | Java runtime for Apache Spark |
| **Apache Spark / PySpark** | `4.3.0` | `4.3.0` | Distributed data processing engine |
| **PyTorch** | `2.14.1` (`+cu126`) | Pinned (`+cu126`) | Deep learning & vision inference |
| **Torchvision** | `0.29.1` (`+cu126`) | Pinned (`+cu126`) | Computer vision models & transforms |
| **CUDA Platform** | `cu126` (CUDA 12.6) | Driver 576.93 (CUDA 12.9 cap.) | GPU tensor acceleration |
| **NumPy** | `2.2.6` | `2.2.6` | High-performance numerical computing |
| **Pandas** | `2.2.3` | `2.2.3` | Tabular data manipulation & analysis |
| **OpenCV** | `4.12.0.88` | `4.12.0.88` | Real-time imagery & frame preprocessing |
| **Matplotlib** | `3.10.1` | `3.10.1` | Analytics & metrics visualization |
| **Scikit-learn** | `1.7.2` | `1.7.2` | Machine learning & baseline estimators |
| **Pytest** | `8.3.5` | `8.3.5` | Test suite & verification runner |
| **Pillow** | `11.1.0` | `11.1.0` | Image format loading & handling |

---

## 📁 Project Directory Structure

```text
StressSense/
├── .gitignore
├── CHANGELOG.md
├── README.md
├── TEST_REPORT.md
├── requirements.txt
├── data/
│   ├── raw/
│   ├── processed/
│   └── annotations/
├── docs/
│   └── TECH_STACK.md
├── experiments/
├── notebooks/
├── results/
├── scripts/
│   └── verify_environment.py
├── src/
│   ├── __init__.py
│   ├── adaptive/
│   │   └── __init__.py
│   ├── evaluation/
│   │   └── __init__.py
│   ├── models/
│   │   └── __init__.py
│   ├── preprocessing/
│   │   └── __init__.py
│   └── spark/
│       └── __init__.py
└── tests/
    ├── __init__.py
    └── test_environment.py
```

---

## 🚀 Environment Setup & Installation

### 1. Prerequisites
- **Python 3.11.x**: Available in Windows PATH or at `%LOCALAPPDATA%\Programs\Python\Python311\python.exe`.
- **Java JDK 17 LTS**: Installed and configured via `JAVA_HOME` (e.g. `C:\Java\jdk-17`).
- **NVIDIA GPU Driver**: >= 570 (supports CUDA 12.6+).

### 2. Virtual Environment Creation
```powershell
# Create isolated Python 3.11 virtual environment
& "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" -m venv .venv

# Activate environment (PowerShell)
.\.venv\Scripts\Activate.ps1
```

### 3. Install Core Dependencies
```powershell
.\.venv\Scripts\pip.exe install -r requirements.txt
```

### 4. Install PyTorch & Torchvision (CUDA 12.6 Wheel)
```powershell
.\.venv\Scripts\pip.exe install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
```

---

## 🧪 Verification & Testing

### Automated Verification Script
Run the automated environment diagnostic script:
```powershell
.\.venv\Scripts\python.exe scripts\verify_environment.py
```

### Pytest Environment Test Suite
Run the comprehensive test suite:
```powershell
.\.venv\Scripts\pytest.exe tests\test_environment.py -v
```

---

## 🔒 Constraints & Development Policies (Phase 0)
- No crop detection, adaptive sampling, or API development is initiated until Phase 0 sign-off.
- Agriculture-Vision dataset downloads are deferred to subsequent phases.
- Pinned dependency versions are strictly adhered to.
