"""StressSense Evaluation Package.

This package provides dataset-wide experimental benchmarking, statistical
evaluation, and visualization engines for validating adaptive sampling against
uniform coverage baselines across the Agriculture-Vision dataset.
"""

from src.evaluation.dataset_evaluator import (
    DatasetWideEvaluator,
    HypothesisTestSummary,
    MetricDistribution,
    PatchRecord,
)
from src.evaluation.visualizer import DatasetVisualizer

__all__ = [
    "DatasetWideEvaluator",
    "DatasetVisualizer",
    "HypothesisTestSummary",
    "MetricDistribution",
    "PatchRecord",
]
