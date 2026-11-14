"""Diagnostic preflight script for Apache Spark on Windows 11."""
import os
import sys
import socket
import platform

print("=== Apache Spark Preflight Diagnostic ===")
print(f"OS: {platform.system()} {platform.release()} ({platform.version()})")
print(f"Python: {sys.version}")
print(f"Python Executable: {sys.executable}")
java_home = os.environ.get("JAVA_HOME", r"C:\Java\jdk-17")
print(f"JAVA_HOME: {java_home}")
print(f"JAVA_HOME exists: {os.path.exists(java_home)}")

os.environ["JAVA_HOME"] = java_home
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# Step 1: Test basic loopback TCP socket binding and connecting
print("\n--- Step 1: Testing local loopback TCP socket ---")
try:
    s_srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s_srv.bind(("127.0.0.1", 0))
    port = s_srv.getsockname()[1]
    s_srv.listen(1)
    print(f"Bound server socket on 127.0.0.1:{port} successfully.")

    s_cli = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s_cli.connect(("127.0.0.1", port))
    print(f"Client connected to 127.0.0.1:{port} successfully.")
    
    conn, addr = s_srv.accept()
    print(f"Server accepted connection from {addr}.")
    conn.close()
    s_cli.close()
    s_srv.close()
    print("Loopback socket test PASSED.")
except Exception as e:
    print(f"Loopback socket test FAILED: {type(e).__name__}: {e}")

# Step 2: Test Py4J JavaGateway directly
print("\n--- Step 2: Testing Py4J Gateway Launch ---")
try:
    from py4j.java_gateway import JavaGateway, GatewayParameters, launch_gateway
    print("Attempting launch_gateway...")
    port = launch_gateway(java_path=os.path.join(java_home, "bin", "java.exe"))
    print(f"Py4J Gateway launched on port: {port}")
    gateway = JavaGateway(gateway_parameters=GatewayParameters(port=port))
    jvm = gateway.jvm
    print(f"JVM System Property java.version: {jvm.java.lang.System.getProperty('java.version')}")
    gateway.shutdown()
    print("Py4J standalone gateway test PASSED.")
except Exception as e:
    print(f"Py4J gateway test FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Step 3: Test PySpark SparkSession
print("\n--- Step 3: Testing PySpark SparkSession ---")
try:
    import pyspark
    print(f"PySpark version: {pyspark.__version__}")
    from pyspark.sql import SparkSession
    
    spark = (
        SparkSession.builder.appName("StressSensePreflight")
        .master("local[2]")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    print("SparkSession created successfully!")
    df = spark.createDataFrame([(1, "Canopy_A"), (2, "Canopy_B")], ["id", "zone"])
    rows = df.collect()
    print(f"DataFrame collect() succeeded with {len(rows)} rows: {rows}")
    spark.stop()
    print("PySpark SparkSession test PASSED.")
except Exception as e:
    print(f"PySpark SparkSession test FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
