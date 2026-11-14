"""Distributed Apache Spark evaluation module for StressSense."""

from src.distributed.spark_evaluator import (
    SparkDistributedEvaluator,
    SparkEvaluatorConfig,
)

__all__ = [
    "SparkDistributedEvaluator",
    "SparkEvaluatorConfig",
]
