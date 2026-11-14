import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.distributed.spark_evaluator import SparkDistributedEvaluator, SparkEvaluatorConfig
from src.distributed.benchmark_visualizer import SparkBenchmarkVisualizer

def main():
    benchmarks_dir = PROJECT_ROOT / "results" / "benchmarks"
    config = SparkEvaluatorConfig(
        output_dir=benchmarks_dir,
        master="local[4]",
        grid_size=32,
        budget_ratio=0.35,
        ndvi_target=0.32,
        ndvi_stressed=0.20,
        exploration_rate=0.05,
        seed=42,
        n_trials=5,
        overlap_threshold=0.10,
    )

    evaluator = SparkDistributedEvaluator(config=config)
    print("Running distributed overlap sensitivity analysis across 0%, 5%, 10%, 20%...")
    sens_results = evaluator.run_overlap_sensitivity_analysis(
        splits=["train", "val"],
        thresholds=[0.0, 0.05, 0.10, 0.20],
        master="local[4]",
    )
    evaluator.stop_spark_session()

    out_file = benchmarks_dir / "category_sensitivity_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(sens_results, f, indent=2)

    print(f"Saved {out_file} successfully!")
    for k, v in sens_results.items():
        print(f"=== {v['overlap_threshold_pct']} Overlap ===")
        for cat, cd in v["categories"].items():
            print(
                f"{cat:15s} | Patches: {cd.get('patch_count'):2d} | "
                f"Regions: {cd.get('qualifying_region_count'):3d} | "
                f"Det Ad: {cd.get('detected_regions_adaptive'):3d} | "
                f"Ad Macro: {cd.get('adaptive_recall_mean'):6.2f}% | "
                f"Ad Pooled: {cd.get('adaptive_recall_pooled_pct'):6.2f}% | "
                f"Eq Macro: {cd.get('equal_recall_mean'):6.2f}% | "
                f"Eq Pooled: {cd.get('equal_recall_pooled_pct'):6.2f}%"
            )

    # Re-plot diagnostic sensitivity curve
    vis = SparkBenchmarkVisualizer(output_dir=benchmarks_dir)
    vis.plot_category_sensitivity(sens_results)
    print("Re-plotted overlap_threshold_sensitivity.png successfully!")

if __name__ == "__main__":
    main()
