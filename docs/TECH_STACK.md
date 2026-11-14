# StressSense - Technology Stack & Version Matrix

This document defines the complete technology stack, verified versions, hardware acceleration interfaces, and installation procedures for **StressSense: Cost- and Energy-Aware Drone Crop Health Monitoring Using Adaptive Sampling and Apache Spark**.

---

## 1. System & Non-Python Runtimes

*Note: Non-Python runtimes are managed at the system level and are not tracked inside Python wheels/freeze files.*

| Technology | Target Version | Installed Version | Purpose | Compatibility Notes | Setup / Installation Command |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Operating System** | Windows 11 (64-bit) | Windows 11 Home/Pro (Build 26100+) | Host execution environment | Fully compatible with PySpark local mode and Win64 PyTorch CUDA wheels | Native OS |
| **Java JDK** | 17 LTS | `17.0.21.0.2 LTS` (OpenJDK 64-bit) | JVM execution runtime required by Apache Spark / PySpark | Configured via `JAVA_HOME=C:\Java\jdk-17`. Fully compliant with Spark 4.x JVM requirements. | Pre-installed at `C:\Java\jdk-17`. Configured in environment variables. |
| **NVIDIA Driver** | >= 550.00 | `576.93` | Host GPU kernel driver | Driver supports CUDA runtime versions up to 12.9, ensuring full backward compatibility with `cu126`. | Installed via NVIDIA GeForce Experience / Enterprise Display Driver. |
| **NVIDIA GPU Hardware**| Compute Cap >= 7.0 | NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM, Ampere GA107, Compute 8.6) | Edge and workstation neural network acceleration | Supports FP16, Tensor Cores, and INT8 inference pipelines. | Hardware device. |

---

## 2. Core Python Dependencies

All packages are isolated within the `.venv` virtual environment running **Python 3.11.9**.

| Technology | Required Version | Installed Version | Purpose | Compatibility Notes | Installation Command |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Python** | `3.11.x` | `3.11.9` | Core project language runtime | Stable support across PySpark 4.3.0, PyTorch 2.14.1, NumPy 2.x, and Pandas 2.x. | System Python at `C:\Users\Karthik\AppData\Local\Programs\Python\Python311\python.exe` |
| **PyTorch** | `2.14.1` | `2.14.1+cu126` | Deep learning tensor engine, defect detection, and model inference | Built with CUDA 12.6 support (`cu126`). Supports both CUDA acceleration and automatic CPU fallback. | `pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126` |
| **Torchvision** | `0.29.1` | `0.29.1+cu126` | Drone imagery computer vision backbones and transform pipelines | Pinned to match PyTorch 2.14.1 ABI. Compatible with Pillow 11.1.0 and NumPy 2.2.6. | `pip install torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126` |
| **PySpark** | `4.3.0` | `4.3.0` | Distributed data processing engine for crop metrics, telemetry, and spatial aggregation | Runs in local mode (`local[*]`). Requires JDK 17 (`JAVA_HOME`). Bundles Spark 4.3.0 runtime jars. | `pip install pyspark==4.3.0` |
| **NumPy** | `2.2.6` | `2.2.6` | Fundamental numerical matrix operations and tensor array buffers | C-API ABI compatible with OpenCV 4.12.0.88 and PyTorch 2.14.1. | `pip install numpy==2.2.6` |
| **Pandas** | `2.2.3` | `2.2.3` | Tabular data structures and local telemetry analysis | Integrates with PySpark Arrow conversions and PyTorch Dataset loaders. | `pip install pandas==2.2.3` |
| **OpenCV** | `4.12.0.88` | `4.12.0.88` (`opencv-python`) | High-throughput image processing, orthomosaic cropping, NDVI color transforms | Includes precompiled headless/GUI Win64 binaries with SIMD acceleration. | `pip install opencv-python==4.12.0.88` |
| **Matplotlib** | `3.10.1` | `3.10.1` | Scientific visualization, stress heatmaps, and flight path telemetry plotting | Integrates with Pillow and NumPy for plot rendering. | `pip install matplotlib==3.10.1` |
| **Scikit-learn** | `1.7.2` | `1.7.2` | Baseline classifiers, feature scaling, clustering, and evaluation metrics | Interoperable with NumPy 2.2.6 and SciPy. | `pip install scikit-learn==1.7.2` |
| **Pytest** | `8.3.5` | `8.3.5` | Automated test suite and unit verification framework | Executes environment diagnostics and pipeline unit tests. | `pip install pytest==8.3.5` |
| **Pillow** | `11.1.0` | `11.1.0` | Raw multi-band image loading and format decoding | Compatible with Torchvision 0.29.1 and OpenCV buffer exchanges. | `pip install Pillow==11.1.0` |

---

## 3. Hardware Acceleration & CUDA Configuration

- **CUDA Wheel Tag**: `cu126`
- **Driver CUDA Capability**: NVIDIA Driver 576.93 supports up to CUDA 12.9.
- **PyTorch CUDA Build**: PyTorch 2.14.1 is installed from `https://download.pytorch.org/whl/cu126`.
- **CPU Fallback**: Verified and enabled. If CUDA GPU is unavailable, all tensor operations gracefully execute on CPU without exceptions.

---

## 4. Local PySpark Execution Configuration

PySpark is configured for local development and processing:
- **Master**: `local[*]` (or `local[2]` for controlled concurrency tests)
- **Driver Host**: `127.0.0.1`
- **UI**: Optional, disabled during test runs via `spark.ui.enabled=false`.
- **Java Home**: `C:\Java\jdk-17`

---

## 5. Frozen Dependency Record

A full pip freeze snapshot is recorded in `requirements-freeze.txt` upon installation completion.
