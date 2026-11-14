"""Audit script to recalculate raw unrounded throughput and reconcile timings."""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

with open(PROJECT_ROOT / "results" / "benchmarks" / "worker_scaling_results.json") as f:
    ws = json.load(f)

print("=== PART A: Worker Scaling Benchmark (160 Patches) ===")
sm = ws["single_machine"]
sm_wall = sm["wall_clock_time_sec"]
sm_n = sm["patches_evaluated"]
sm_tp_raw = sm_n / sm_wall
print(f"Single Machine Baseline:")
print(f"  Patches Evaluated: {sm_n}")
print(f"  Wall-Clock Time:   {sm_wall:.6f} s (stored: {sm['wall_clock_time_sec']})")
print(f"  Raw Throughput:    {sm_tp_raw:.6f} patches/s (stored: {sm['throughput_patches_per_sec']})")

print("\nSpark Local Worker Configurations:")
for cfg, d in ws["spark_configs"].items():
    su = d["startup_overhead_sec"]
    st = d["steady_state_time_sec"]
    tot = d["total_wall_clock_sec"]
    n = d["patches_evaluated"]
    steady_tp_raw = n / st if st > 0 else 0
    total_tp_raw = n / tot if tot > 0 else 0
    sum_components = su + st
    driver_overhead = tot - sum_components
    steady_speedup_raw = sm_wall / st if st > 0 else 0
    total_speedup_raw = sm_wall / tot if tot > 0 else 0

    print(f"\nConfiguration: {cfg}")
    print(f"  Configured Cores:       {cfg.replace('local[', '').replace(']', '')}")
    print(f"  Startup Overhead:       {su:.6f} s (stored: {su})")
    print(f"  Steady-State Compute:   {st:.6f} s (stored: {st})")
    print(f"  Startup + Steady Sum:   {sum_components:.6f} s")
    print(f"  Total Wall-Clock Time:  {tot:.6f} s (stored: {tot})")
    print(f"  Driver Framework Delta: {driver_overhead:.6f} s ({(driver_overhead/tot)*100:.2f}% of total)")
    print(f"  Raw Steady Throughput:  {steady_tp_raw:.6f} patches/s (stored: {d['throughput_patches_per_sec']})")
    print(f"  Raw Total Throughput:   {total_tp_raw:.6f} patches/s")
    print(f"  Raw Steady Speedup:     {steady_speedup_raw:.6f}x (stored: {d['speedup_steady_state_vs_single']}x)")
    print(f"  Raw Total Speedup:      {total_speedup_raw:.6f}x (stored: {d['speedup_total_vs_single']}x)")

print("\n=== PART B: Full Fleet Distributed Telemetry (640 Patches) ===")
with open(PROJECT_ROOT / "results" / "benchmarks" / "spark_benchmark_summary.json") as f:
    sb = json.load(f)

tel = sb["telemetry"]
f_su = tel["startup_overhead_sec"]
f_st = tel["steady_state_time_sec"]
f_tot = tel["total_wall_clock_sec"]
f_n = tel["evaluated_patches"]
f_sum = f_su + f_st
f_diff = f_tot - f_sum
f_steady_tp = f_n / f_st if f_st > 0 else 0
f_total_tp = f_n / f_tot if f_tot > 0 else 0

print(f"Fleet Scale Allocation:   {tel['master']} ({tel['partitions']} partitions)")
print(f"  Total Evaluated:        {f_n} patches")
print(f"  Cold Startup Overhead:  {f_su:.6f} s (stored: {f_su})")
print(f"  Steady-State Compute:   {f_st:.6f} s (stored: {f_st})")
print(f"  Startup + Steady Sum:   {f_sum:.6f} s")
print(f"  Total Wall-Clock Time:  {f_tot:.6f} s (stored: {f_tot})")
print(f"  Driver Overhead Delta:  {f_diff:.6f} s ({(f_diff/f_tot)*100:.2f}% of total)")
print(f"  Raw Steady Throughput:  {f_steady_tp:.6f} patches/s (stored: {tel['throughput_patches_per_sec']})")
print(f"  Raw Total Throughput:   {f_total_tp:.6f} patches/s")
