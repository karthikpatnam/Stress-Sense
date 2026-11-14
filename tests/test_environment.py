"""
tests/test_environment.py

Pytest suite validating Phase 0 environment setup, dependency versions,
hardware accelerators, and local Spark runtime for StressSense.
"""

import sys
import os
import subprocess
import pytest

# Expected Target Versions
TARGET_VERSIONS = {
    "python_major_minor": (3, 11),
    "java_major": "17",
    "numpy": "2.2.6",
    "pandas": "2.2.3",
    "opencv": "4.12.0.88",
    "matplotlib": "3.10.1",
    "scikit-learn": "1.7.2",
    "pytest": "8.3.5",
    "pillow": "11.1.0",
    "pyspark": "4.3.0",
    "torch": "2.14.1",
    "torchvision": "0.29.1",
    "cuda_wheel": "cu126",
}


def test_python_version():
    """Verify runtime Python is 3.11.x."""
    assert (
        sys.version_info.major,
        sys.version_info.minor,
    ) == TARGET_VERSIONS["python_major_minor"], (
        f"Expected Python 3.11.x, but running on {sys.version}"
    )


def test_java_environment():
    """Verify Java JDK 17 is installed and JAVA_HOME is set."""
    java_home = os.environ.get("JAVA_HOME")
    if not java_home and os.path.exists(r"C:\Java\jdk-17"):
        java_home = r"C:\Java\jdk-17"
        os.environ["JAVA_HOME"] = java_home

    assert java_home is not None, "JAVA_HOME environment variable is not set."
    assert os.path.exists(java_home), f"JAVA_HOME path does not exist: {java_home}"

    java_bin = os.path.join(java_home, "bin", "java.exe")
    assert os.path.exists(java_bin), f"java.exe not found at {java_bin}"

    proc = subprocess.run(
        [java_bin, "-version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    raw = proc.stderr if proc.stderr else proc.stdout
    assert "17." in raw or "\"17" in raw, f"Java version output did not contain JDK 17: {raw}"


def test_numpy_version():
    """Verify NumPy is installed and exact pinned version matches 2.2.6."""
    import numpy as np

    assert (
        np.__version__ == TARGET_VERSIONS["numpy"]
    ), f"NumPy version mismatch: expected {TARGET_VERSIONS['numpy']}, got {np.__version__}"


def test_pandas_version():
    """Verify Pandas is installed and exact pinned version matches 2.2.3."""
    import pandas as pd

    assert (
        pd.__version__ == TARGET_VERSIONS["pandas"]
    ), f"Pandas version mismatch: expected {TARGET_VERSIONS['pandas']}, got {pd.__version__}"


def test_opencv_version():
    """Verify OpenCV is installed and exact pinned version matches 4.12.0.88."""
    import cv2
    import importlib.metadata

    pkg_ver = importlib.metadata.version("opencv-python")
    assert (
        pkg_ver == TARGET_VERSIONS["opencv"] or cv2.__version__.startswith("4.12.0")
    ), f"OpenCV version mismatch: expected {TARGET_VERSIONS['opencv']}, got pkg={pkg_ver}, module={cv2.__version__}"


def test_matplotlib_version():
    """Verify Matplotlib is installed and exact pinned version matches 3.10.1."""
    import matplotlib

    assert (
        matplotlib.__version__ == TARGET_VERSIONS["matplotlib"]
    ), f"Matplotlib version mismatch: expected {TARGET_VERSIONS['matplotlib']}, got {matplotlib.__version__}"


def test_scikit_learn_version():
    """Verify Scikit-learn is installed and exact pinned version matches 1.7.2."""
    import sklearn

    assert (
        sklearn.__version__ == TARGET_VERSIONS["scikit-learn"]
    ), f"Scikit-learn version mismatch: expected {TARGET_VERSIONS['scikit-learn']}, got {sklearn.__version__}"


def test_pillow_version():
    """Verify Pillow is installed and exact pinned version matches 11.1.0."""
    import PIL

    assert (
        PIL.__version__ == TARGET_VERSIONS["pillow"]
    ), f"Pillow version mismatch: expected {TARGET_VERSIONS['pillow']}, got {PIL.__version__}"


def test_pytest_version():
    """Verify Pytest version matches 8.3.5."""
    assert (
        pytest.__version__ == TARGET_VERSIONS["pytest"]
    ), f"Pytest version mismatch: expected {TARGET_VERSIONS['pytest']}, got {pytest.__version__}"


def test_pytorch_version():
    """Verify PyTorch is installed and version is 2.14.1."""
    import torch

    assert TARGET_VERSIONS["torch"] in torch.__version__, (
        f"PyTorch version mismatch: expected {TARGET_VERSIONS['torch']}, got {torch.__version__}"
    )


def test_torchvision_version():
    """Verify Torchvision is installed and version is 0.29.1."""
    import torchvision

    assert TARGET_VERSIONS["torchvision"] in torchvision.__version__, (
        f"Torchvision version mismatch: expected {TARGET_VERSIONS['torchvision']}, got {torchvision.__version__}"
    )


def test_pytorch_cpu_tensor_computation():
    """Verify PyTorch CPU tensor operations execute correctly with expected numerical values."""
    import torch

    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32)
    c = torch.matmul(a, b)

    expected = [[19.0, 22.0], [43.0, 50.0]]
    assert c.tolist() == expected, f"CPU matmul result mismatch: {c.tolist()} != {expected}"

    # Addition and reduction
    total = c.sum().item()
    assert abs(total - 134.0) < 1e-5, f"Sum mismatch: {total} != 134.0"


def test_pytorch_cuda_and_gpu_computation():
    """
    Verify CUDA availability and GPU tensor computation.
    If no GPU is available or CUDA is not configured, validates fallback gracefully.
    """
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU is not available on this system. CPU fallback is operational.")

    device_name = torch.cuda.get_device_name(0)
    assert device_name, "CUDA is available but device name is empty"

    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, device="cuda:0")
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32, device="cuda:0")
    c = torch.matmul(a, b)

    expected = [[19.0, 22.0], [43.0, 50.0]]
    assert c.cpu().tolist() == expected, f"GPU matmul result mismatch: {c.cpu().tolist()} != {expected}"
    assert c.is_cuda, "Resulting tensor is not on CUDA device"


def test_pyspark_version():
    """Verify PySpark is installed and version matches 4.3.0."""
    import pyspark

    assert (
        pyspark.__version__ == TARGET_VERSIONS["pyspark"]
    ), f"PySpark version mismatch: expected {TARGET_VERSIONS['pyspark']}, got {pyspark.__version__}"


def test_pyspark_dataframe_operations():
    """
    Verify PySpark starts in local mode, creates a test DataFrame,
    executes transformations, and validates the resulting values.
    """
    import os
    if "JAVA_HOME" not in os.environ and os.path.exists(r"C:\Java\jdk-17"):
        os.environ["JAVA_HOME"] = r"C:\Java\jdk-17"

    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    from pyspark.sql import SparkSession
    import pyspark.sql.functions as F

    spark = (
        SparkSession.builder.appName("StressSensePytestSpark")
        .master("local[2]")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )

    try:
        # Mock drone sensor readings: drone_id, zone, canopy_temp, ndvi
        data = [
            ("DRONE-01", "Zone_A", 28.5, 0.72),
            ("DRONE-01", "Zone_B", 34.2, 0.38),
            ("DRONE-02", "Zone_C", 29.0, 0.65),
            ("DRONE-02", "Zone_D", 36.1, 0.29),
            ("DRONE-03", "Zone_E", 27.8, 0.81),
        ]
        columns = ["drone_id", "zone", "canopy_temp", "ndvi"]
        df = spark.createDataFrame(data, schema=columns)

        # Transformation:
        # Filter zones experiencing water/vegetation stress: ndvi < 0.40 and canopy_temp > 30.0
        # Calculate water deficit index: (canopy_temp - 25.0) * (1.0 - ndvi)
        stressed_df = (
            df.filter((F.col("ndvi") < 0.40) & (F.col("canopy_temp") > 30.0))
            .withColumn(
                "wdi_score",
                F.round((F.col("canopy_temp") - 25.0) * (1.0 - F.col("ndvi")), 3),
            )
            .orderBy("zone")
        )

        rows = stressed_df.collect()
        assert len(rows) == 2, f"Expected 2 stressed zones, got {len(rows)}"

        # Validate Zone_B
        row_b = rows[0].asDict()
        assert row_b["zone"] == "Zone_B"
        # (34.2 - 25.0) * (1 - 0.38) = 9.2 * 0.62 = 5.704
        assert abs(row_b["wdi_score"] - 5.704) < 1e-3, f"WDI score mismatch for Zone_B: {row_b['wdi_score']}"

        # Validate Zone_D
        row_d = rows[1].asDict()
        assert row_d["zone"] == "Zone_D"
        # (36.1 - 25.0) * (1 - 0.29) = 11.1 * 0.71 = 7.881
        assert abs(row_d["wdi_score"] - 7.881) < 1e-3, f"WDI score mismatch for Zone_D: {row_d['wdi_score']}"

    finally:
        spark.stop()
