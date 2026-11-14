#!/usr/bin/env python3
"""
scripts/verify_environment.py

StressSense - Phase 0 Environment & Dependency Verification Script
Validates runtime versions, hardware accelerators (CUDA/CPU fallback),
and Apache Spark local mode execution.
"""

import sys
import os
import subprocess
import json
from typing import Dict, Any, Tuple

# Required technology version targets
TARGET_VERSIONS = {
    "python": "3.11",
    "java": "17",
    "pyspark": "4.3.0",
    "numpy": "2.2.6",
    "pandas": "2.2.3",
    "opencv": "4.12.0.88",
    "matplotlib": "3.10.1",
    "scikit-learn": "1.7.2",
    "pytest": "8.3.5",
    "pillow": "11.1.0",
    "torch": "2.14.1",
    "torchvision": "0.29.1",
    "cuda_wheel": "cu126",
}


def check_python_version() -> Tuple[bool, str]:
    ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    passed = sys.version_info.major == 3 and sys.version_info.minor == 11
    return passed, ver


def check_java_version() -> Tuple[bool, Dict[str, Any]]:
    java_home = os.environ.get("JAVA_HOME", "")
    java_bin = os.path.join(java_home, "bin", "java.exe") if java_home else "java"
    if not os.path.exists(java_bin):
        java_bin = "java"

    try:
        proc = subprocess.run(
            [java_bin, "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        output = proc.stderr if proc.stderr else proc.stdout
        # Example output: openjdk version "17.0.21.0.2" ...
        first_line = output.splitlines()[0] if output else "Unknown"
        is_java_17 = "17." in first_line or "\"17" in first_line
        return is_java_17, {
            "version_string": first_line.strip(),
            "java_home": java_home,
            "detected_binary": java_bin,
            "raw_output": output.strip(),
        }
    except Exception as e:
        return False, {"error": str(e), "java_home": java_home}


def check_python_libraries() -> Dict[str, Dict[str, Any]]:
    libs = {}

    # NumPy
    try:
        import numpy as np
        libs["numpy"] = {
            "installed": True,
            "version": np.__version__,
            "expected": TARGET_VERSIONS["numpy"],
            "matches": np.__version__ == TARGET_VERSIONS["numpy"],
        }
    except ImportError as e:
        libs["numpy"] = {"installed": False, "error": str(e)}

    # Pandas
    try:
        import pandas as pd
        libs["pandas"] = {
            "installed": True,
            "version": pd.__version__,
            "expected": TARGET_VERSIONS["pandas"],
            "matches": pd.__version__ == TARGET_VERSIONS["pandas"],
        }
    except ImportError as e:
        libs["pandas"] = {"installed": False, "error": str(e)}

    # OpenCV
    try:
        import cv2
        import importlib.metadata
        pkg_version = importlib.metadata.version("opencv-python")
        libs["opencv"] = {
            "installed": True,
            "version": pkg_version,
            "module_version": cv2.__version__,
            "expected": TARGET_VERSIONS["opencv"],
            "matches": pkg_version == TARGET_VERSIONS["opencv"] or cv2.__version__.startswith("4.12.0"),
        }
    except ImportError as e:
        libs["opencv"] = {"installed": False, "error": str(e)}

    # Matplotlib
    try:
        import matplotlib
        libs["matplotlib"] = {
            "installed": True,
            "version": matplotlib.__version__,
            "expected": TARGET_VERSIONS["matplotlib"],
            "matches": matplotlib.__version__ == TARGET_VERSIONS["matplotlib"],
        }
    except ImportError as e:
        libs["matplotlib"] = {"installed": False, "error": str(e)}

    # Scikit-learn
    try:
        import sklearn
        libs["scikit-learn"] = {
            "installed": True,
            "version": sklearn.__version__,
            "expected": TARGET_VERSIONS["scikit-learn"],
            "matches": sklearn.__version__ == TARGET_VERSIONS["scikit-learn"],
        }
    except ImportError as e:
        libs["scikit-learn"] = {"installed": False, "error": str(e)}

    # Pillow
    try:
        import PIL
        libs["pillow"] = {
            "installed": True,
            "version": PIL.__version__,
            "expected": TARGET_VERSIONS["pillow"],
            "matches": PIL.__version__ == TARGET_VERSIONS["pillow"],
        }
    except ImportError as e:
        libs["pillow"] = {"installed": False, "error": str(e)}

    # Pytest
    try:
        import pytest
        libs["pytest"] = {
            "installed": True,
            "version": pytest.__version__,
            "expected": TARGET_VERSIONS["pytest"],
            "matches": pytest.__version__ == TARGET_VERSIONS["pytest"],
        }
    except ImportError as e:
        libs["pytest"] = {"installed": False, "error": str(e)}

    return libs


def check_pytorch_and_cuda() -> Dict[str, Any]:
    res = {}
    try:
        import torch
        res["torch_installed"] = True
        res["torch_version"] = torch.__version__
        res["expected_torch"] = TARGET_VERSIONS["torch"]
        res["torch_version_matches"] = TARGET_VERSIONS["torch"] in torch.__version__
        res["torch_cuda_build"] = torch.version.cuda
    except ImportError as e:
        res["torch_installed"] = False
        res["error"] = str(e)
        return res

    try:
        import torchvision
        res["torchvision_installed"] = True
        res["torchvision_version"] = torchvision.__version__
        res["expected_torchvision"] = TARGET_VERSIONS["torchvision"]
        res["torchvision_version_matches"] = (
            TARGET_VERSIONS["torchvision"] in torchvision.__version__
        )
    except ImportError as e:
        res["torchvision_installed"] = False
        res["torchvision_error"] = str(e)

    # CPU Tensor Computation Test
    try:
        t_cpu_a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        t_cpu_b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32)
        t_cpu_c = torch.matmul(t_cpu_a, t_cpu_b)
        expected_cpu_c = [[19.0, 22.0], [43.0, 50.0]]
        cpu_passed = t_cpu_c.tolist() == expected_cpu_c
        res["cpu_tensor_test"] = {
            "passed": cpu_passed,
            "result": t_cpu_c.tolist(),
        }
    except Exception as e:
        res["cpu_tensor_test"] = {"passed": False, "error": str(e)}

    # CUDA & GPU Computation Test
    cuda_avail = torch.cuda.is_available()
    res["cuda_available"] = cuda_avail

    if cuda_avail:
        try:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            capability = torch.cuda.get_device_capability(0)
            res["gpu_info"] = {
                "device_count": device_count,
                "device_name": device_name,
                "compute_capability": f"{capability[0]}.{capability[1]}",
                "cuda_arch_list": torch.cuda.get_arch_list()
                if hasattr(torch.cuda, "get_arch_list")
                else [],
            }

            # GPU Tensor computation
            t_gpu_a = t_cpu_a.to("cuda:0")
            t_gpu_b = t_cpu_b.to("cuda:0")
            t_gpu_c = torch.matmul(t_gpu_a, t_gpu_b)
            gpu_res = t_gpu_c.cpu().tolist()
            gpu_passed = gpu_res == expected_cpu_c
            res["gpu_tensor_test"] = {
                "passed": gpu_passed,
                "result": gpu_res,
                "device": str(t_gpu_c.device),
            }
        except Exception as e:
            res["gpu_tensor_test"] = {"passed": False, "error": str(e)}
    else:
        res["gpu_info"] = None
        res["gpu_tensor_test"] = {
            "passed": None,
            "notice": "CUDA is unavailable. CPU fallback is operational.",
        }

    return res


def check_pyspark() -> Dict[str, Any]:
    res = {}
    try:
        import pyspark
        from pyspark.sql import SparkSession
        import pyspark.sql.functions as F

        res["pyspark_installed"] = True
        res["pyspark_version"] = pyspark.__version__
        res["expected_pyspark"] = TARGET_VERSIONS["pyspark"]
        res["matches"] = pyspark.__version__ == TARGET_VERSIONS["pyspark"]

        # Ensure JAVA_HOME and PySpark worker python executables are configured
        if "JAVA_HOME" not in os.environ and os.path.exists(r"C:\Java\jdk-17"):
            os.environ["JAVA_HOME"] = r"C:\Java\jdk-17"

        os.environ["PYSPARK_PYTHON"] = sys.executable
        os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

        # Initialize SparkSession in local mode
        spark = (
            SparkSession.builder.appName("StressSenseEnvTest")
            .master("local[2]")
            .config("spark.driver.host", "127.0.0.1")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )

        try:
            # Create test data
            data = [
                ("zone_north", 12.5, 0.85),
                ("zone_south", 18.2, 0.42),
                ("zone_east", 9.8, 0.91),
                ("zone_west", 22.0, 0.35),
            ]
            columns = ["zone_id", "temp_celsius", "ndvi_index"]
            df = spark.createDataFrame(data, schema=columns)

            # Transformation: filter stressed zones (ndvi_index < 0.5) and calculate stress score
            stressed_df = df.filter(F.col("ndvi_index") < 0.5).withColumn(
                "stress_score", F.round(1.0 - F.col("ndvi_index"), 2)
            )

            results = [row.asDict() for row in stressed_df.collect()]

            # Validate results
            expected_zones = {"zone_south", "zone_west"}
            actual_zones = {r["zone_id"] for r in results}
            passed = (actual_zones == expected_zones) and (len(results) == 2)

            res["spark_session_test"] = {
                "passed": passed,
                "app_name": spark.sparkContext.appName,
                "spark_version": spark.version,
                "master": spark.sparkContext.master,
                "transformation_result": results,
            }
        finally:
            spark.stop()

    except Exception as e:
        res["pyspark_installed"] = False
        res["error"] = str(e)

    return res


def run_full_verification() -> Dict[str, Any]:
    print("=" * 60)
    print(" STRESSSENSE - PHASE 0 ENVIRONMENT VERIFICATION")
    print("=" * 60)

    # 1. Python Check
    py_ok, py_ver = check_python_version()
    print(f"[*] Python Version: {py_ver} (Target: 3.11.x) -> {'PASS' if py_ok else 'FAIL'}")

    # 2. Java Check
    java_ok, java_info = check_java_version()
    print(
        f"[*] Java Version: {java_info.get('version_string', 'Not Found')} "
        f"(Target: JDK 17 LTS) -> {'PASS' if java_ok else 'FAIL'}"
    )
    if "java_home" in java_info:
        print(f"    JAVA_HOME: {java_info['java_home']}")

    # 3. Core Libraries Check
    print("\n[*] Core Python Libraries:")
    lib_results = check_python_libraries()
    all_libs_ok = True
    for name, info in lib_results.items():
        if info.get("installed") and info.get("matches"):
            status = "PASS"
        elif info.get("installed"):
            status = f"WARN (version mismatch: actual={info['version']}, target={info['expected']})"
            all_libs_ok = False
        else:
            status = f"FAIL (not installed: {info.get('error')})"
            all_libs_ok = False
        print(f"    - {name:<15}: {info.get('version', 'N/A'):<12} Target: {TARGET_VERSIONS.get(name, 'N/A'):<10} -> {status}")

    # 4. PyTorch & CUDA Check
    print("\n[*] PyTorch & Hardware Acceleration:")
    torch_info = check_pytorch_and_cuda()
    if torch_info.get("torch_installed"):
        torch_ok = torch_info.get("torch_version_matches", False)
        print(
            f"    - PyTorch Version : {torch_info.get('torch_version')} (Target: {TARGET_VERSIONS['torch']}) "
            f"CUDA Build: {torch_info.get('torch_cuda_build')} -> {'PASS' if torch_ok else 'WARN'}"
        )
        print(
            f"    - Torchvision     : {torch_info.get('torchvision_version')} (Target: {TARGET_VERSIONS['torchvision']}) "
            f"-> {'PASS' if torch_info.get('torchvision_version_matches') else 'WARN'}"
        )
        cpu_test = torch_info.get("cpu_tensor_test", {})
        print(f"    - CPU Tensor Test : {'PASS' if cpu_test.get('passed') else 'FAIL'}")

        if torch_info.get("cuda_available"):
            gpu = torch_info.get("gpu_info", {})
            print(f"    - CUDA Available  : YES (Device: {gpu.get('device_name')}, Compute: {gpu.get('compute_capability')})")
            gpu_test = torch_info.get("gpu_tensor_test", {})
            print(f"    - GPU Tensor Test : {'PASS' if gpu_test.get('passed') else 'FAIL'}")
        else:
            print("    - CUDA Available  : NO (CPU fallback operational)")
    else:
        print(f"    - PyTorch         : FAIL ({torch_info.get('error')})")

    # 5. PySpark Check
    print("\n[*] Apache PySpark & SparkSession Local Mode:")
    spark_info = check_pyspark()
    if spark_info.get("pyspark_installed"):
        sp_matches = spark_info.get("matches", False)
        print(
            f"    - PySpark Version : {spark_info.get('pyspark_version')} (Target: {TARGET_VERSIONS['pyspark']}) "
            f"-> {'PASS' if sp_matches else 'WARN'}"
        )
        s_test = spark_info.get("spark_session_test", {})
        print(f"    - Spark DataFrame : {'PASS' if s_test.get('passed') else 'FAIL'}")
        if s_test.get("transformation_result"):
            print(f"      Result sample   : {s_test['transformation_result']}")
    else:
        print(f"    - PySpark         : FAIL ({spark_info.get('error')})")

    overall_pass = (
        py_ok
        and java_ok
        and all_libs_ok
        and torch_info.get("torch_installed", False)
        and torch_info.get("cpu_tensor_test", {}).get("passed", False)
        and spark_info.get("pyspark_installed", False)
        and spark_info.get("spark_session_test", {}).get("passed", False)
    )

    print("\n" + "=" * 60)
    print(f" OVERALL PHASE 0 VERIFICATION: {'PASSED' if overall_pass else 'FAILED'}")
    print("=" * 60)

    report = {
        "overall_pass": overall_pass,
        "python": {"passed": py_ok, "version": py_ver},
        "java": {"passed": java_ok, "info": java_info},
        "libraries": lib_results,
        "torch_acceleration": torch_info,
        "pyspark": spark_info,
    }
    return report


if __name__ == "__main__":
    report = run_full_verification()
    sys.exit(0 if report["overall_pass"] else 1)
