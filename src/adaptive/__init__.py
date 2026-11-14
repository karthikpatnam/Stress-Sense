"""Adaptive sampling package for StressSense.

Provides spatial grid generation, multi-spectral crop stress scoring,
uniform and adaptive sampling engines, and comparative performance evaluation.
"""

from src.adaptive.evaluator import (
    AdaptiveEvaluator,
    BenchmarkComparison,
    EvaluationMetrics,
    ResourceCostAssumptions,
    TrialAggregateMetrics,
)
from src.adaptive.grid_generator import GridGenerator, GridRegion
from src.adaptive.sampling_engine import (
    AdaptiveSampler,
    BudgetOverflowPolicy,
    SamplingDecision,
    SamplingEngine,
    SamplingResult,
    SamplingStrategy,
    UniformSampler,
)
from src.adaptive.stress_scorer import RegionStressScore, StressCategory, StressScorer
from src.adaptive.visualizer import AdaptiveVisualizer

__all__ = [
    "GridGenerator",
    "GridRegion",
    "RegionStressScore",
    "StressCategory",
    "StressScorer",
    "SamplingStrategy",
    "BudgetOverflowPolicy",
    "SamplingDecision",
    "SamplingResult",
    "UniformSampler",
    "AdaptiveSampler",
    "SamplingEngine",
    "AdaptiveEvaluator",
    "BenchmarkComparison",
    "EvaluationMetrics",
    "ResourceCostAssumptions",
    "TrialAggregateMetrics",
    "AdaptiveVisualizer",
]
