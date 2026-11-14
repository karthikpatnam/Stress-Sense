"""Sampling engine for StressSense adaptive aerial monitoring.

This module implements three sampling strategies:
1. Full-Coverage Uniform: Baseline 100% scan of all valid regions.
2. Equal-Budget Uniform: Fair comparison baseline selecting exactly K valid
   regions via reproducible pseudo-random sampling.
3. Adaptive Sampling: Prioritizes regions with high Relative Spectral Difference
   Index (RSDI) scores, allocates budget to moderate-stress regions, reserves
   an exploration fraction for healthy regions, and handles budget overflows.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.adaptive.stress_scorer import RegionStressScore, StressCategory


class SamplingStrategy(str, Enum):
    """Identifier for the sampling strategy employed."""

    FULL_UNIFORM = "full_uniform"
    EQUAL_BUDGET_UNIFORM = "equal_budget_uniform"
    ADAPTIVE = "adaptive"


class BudgetOverflowPolicy(str, Enum):
    """Policy governing behavior when mandatory high-score regions exceed requested budget.

    GUARANTEE_HIGH_STRESS:
        Safety-critical mode (default). High-RSDI regions are mandatorily selected,
        even if the total selected count exceeds the requested budget. A documented
        overflow warning is logged.

    STRICT_CAP:
        Hard-budget enforcement. High-RSDI regions are ranked by score descending
        and strictly capped at the requested budget. A deficit alert is logged.
    """

    GUARANTEE_HIGH_STRESS = "guarantee_high_stress"
    STRICT_CAP = "strict_cap"


@dataclass
class SamplingDecision:
    """Individual sampling decision and telemetry for a single grid region.

    Attributes:
        region_id: Unique grid region identifier.
        row: Row index in the 2D grid.
        col: Column index in the 2D grid.
        is_selected: True if selected for drone transmission/deep inference.
        selection_reason: Descriptive rationale for the decision:
            'full_uniform', 'equal_budget_uniform', 'high_stress_guarantee',
            'moderate_stress_priority', 'healthy_exploration', 'not_selected',
            or 'invalid_excluded'.
        combined_stress_score: Empirical RSDI score in [0.0, 1.0].
        stress_category: Categorical classification (HEALTHY, MODERATE_STRESS, HIGH_STRESS, INVALID).
        is_valid: False if sensor coverage or boundary validity is insufficient.
    """

    region_id: str
    row: int
    col: int
    is_selected: bool
    selection_reason: str
    combined_stress_score: float
    stress_category: str
    is_valid: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert decision to a serializable dictionary."""
        return {
            "region_id": self.region_id,
            "row": self.row,
            "col": self.col,
            "is_selected": self.is_selected,
            "selection_reason": self.selection_reason,
            "combined_stress_score": round(self.combined_stress_score, 5),
            "stress_category": self.stress_category,
            "is_valid": self.is_valid,
        }


@dataclass
class SamplingResult:
    """Aggregate result and metrics from executing a sampling strategy.

    Attributes:
        strategy: SamplingStrategy enum used.
        total_regions: Total grid regions in the aerial image/tile.
        valid_regions: Number of valid agricultural regions.
        invalid_regions: Number of invalid/obscured regions excluded.
        requested_budget: Target number of regions requested, or None if threshold-driven.
        actual_selected_count: Exact number of regions chosen for sampling.
        sampling_reduction_pct: Percentage reduction relative to total regions:
            ((total_regions - actual_selected_count) / total_regions) * 100.
            Reports total-grid reduction across all regions including non-crop/invalid areas.
        valid_reduction_pct: Primary sampling-efficiency metric relative to valid regions:
            ((valid_regions - actual_selected_count) / valid_regions) * 100.
            Measures true drone transmission and compute savings strictly over candidate
            agricultural canopy. Inherently invalid regions are never treated as avoided sampling.
        seed: Pinned random seed used for reproducibility.
        overflow_policy: Policy used if high-stress regions exceeded budget.
        budget_exceeded: True if actual_selected_count > requested_budget.
        overflow_count: Number of regions selected beyond requested_budget.
        deficit_count: Number of high-stress regions omitted under STRICT_CAP.
        selected_regions: List of SamplingDecision objects that were selected.
        all_decisions: List of SamplingDecision objects for every region in the grid.
        metadata: Telemetry dict detailing tier breakdowns and warnings.
    """

    strategy: SamplingStrategy
    total_regions: int
    valid_regions: int
    invalid_regions: int
    requested_budget: Optional[int]
    actual_selected_count: int
    sampling_reduction_pct: float
    valid_reduction_pct: float
    seed: int
    overflow_policy: Optional[BudgetOverflowPolicy] = None
    budget_exceeded: bool = False
    overflow_count: int = 0
    deficit_count: int = 0
    selected_regions: List[SamplingDecision] = field(default_factory=list)
    all_decisions: List[SamplingDecision] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert summary metrics to dictionary format."""
        return {
            "strategy": self.strategy.value,
            "total_regions": self.total_regions,
            "valid_regions": self.valid_regions,
            "invalid_regions": self.invalid_regions,
            "requested_budget": self.requested_budget,
            "actual_selected_count": self.actual_selected_count,
            "sampling_reduction_pct": round(self.sampling_reduction_pct, 2),
            "valid_reduction_pct": round(self.valid_reduction_pct, 2),
            "seed": self.seed,
            "overflow_policy": self.overflow_policy.value if self.overflow_policy else None,
            "budget_exceeded": self.budget_exceeded,
            "overflow_count": self.overflow_count,
            "deficit_count": self.deficit_count,
            "metadata": self.metadata,
        }


class UniformSampler:
    """Implements Full-Coverage and Equal-Budget uniform sampling strategies."""

    @staticmethod
    def sample_full_coverage(
        scores: List[RegionStressScore],
        seed: int = 42,
    ) -> SamplingResult:
        """Select every valid region to establish the full-coverage reference baseline.

        Args:
            scores: List of RegionStressScore objects across the grid.
            seed: Pinned random seed.

        Returns:
            SamplingResult with all valid regions selected (0% valid reduction).
        """
        total = len(scores)
        valid_scores = [s for s in scores if s.is_valid]
        invalid_count = total - len(valid_scores)

        all_decisions: List[SamplingDecision] = []
        selected_decisions: List[SamplingDecision] = []

        for s in scores:
            if s.is_valid:
                decision = SamplingDecision(
                    region_id=s.region_id,
                    row=s.row,
                    col=s.col,
                    is_selected=True,
                    selection_reason="full_uniform",
                    combined_stress_score=s.combined_stress_score,
                    stress_category=s.stress_category.value,
                    is_valid=True,
                )
                selected_decisions.append(decision)
            else:
                decision = SamplingDecision(
                    region_id=s.region_id,
                    row=s.row,
                    col=s.col,
                    is_selected=False,
                    selection_reason="invalid_excluded",
                    combined_stress_score=0.0,
                    stress_category=StressCategory.INVALID.value,
                    is_valid=False,
                )
            all_decisions.append(decision)

        n_selected = len(selected_decisions)
        n_valid = len(valid_scores)

        reduction_total = ((total - n_selected) / total * 100.0) if total > 0 else 0.0
        reduction_valid = ((n_valid - n_selected) / n_valid * 100.0) if n_valid > 0 else 0.0

        return SamplingResult(
            strategy=SamplingStrategy.FULL_UNIFORM,
            total_regions=total,
            valid_regions=n_valid,
            invalid_regions=invalid_count,
            requested_budget=None,
            actual_selected_count=n_selected,
            sampling_reduction_pct=reduction_total,
            valid_reduction_pct=reduction_valid,
            seed=seed,
            overflow_policy=None,
            budget_exceeded=False,
            overflow_count=0,
            deficit_count=0,
            selected_regions=selected_decisions,
            all_decisions=all_decisions,
            metadata={"description": "Full-coverage scan of all valid regions."},
        )

    @staticmethod
    def sample_equal_budget(
        scores: List[RegionStressScore],
        budget: int,
        seed: int = 42,
    ) -> SamplingResult:
        """Randomly select exactly K valid regions to establish an equal-resource baseline.

        Args:
            scores: List of RegionStressScore objects across the grid.
            budget: Number of regions K to select (matches adaptive selected count).
            seed: Pinned random seed for reproducible pseudo-random selection.

        Returns:
            SamplingResult with exactly min(budget, valid_regions) selected.
        """
        if budget < 0:
            raise ValueError(f"budget must be non-negative, got {budget}")

        total = len(scores)
        valid_scores = [s for s in scores if s.is_valid]
        n_valid = len(valid_scores)
        invalid_count = total - n_valid

        k = min(budget, n_valid)

        rng = np.random.RandomState(seed)
        if n_valid > 0 and k > 0:
            chosen_indices = set(rng.choice(n_valid, size=k, replace=False))
        else:
            chosen_indices = set()

        chosen_ids = {valid_scores[idx].region_id for idx in chosen_indices}

        all_decisions: List[SamplingDecision] = []
        selected_decisions: List[SamplingDecision] = []

        for s in scores:
            if not s.is_valid:
                decision = SamplingDecision(
                    region_id=s.region_id,
                    row=s.row,
                    col=s.col,
                    is_selected=False,
                    selection_reason="invalid_excluded",
                    combined_stress_score=0.0,
                    stress_category=StressCategory.INVALID.value,
                    is_valid=False,
                )
            elif s.region_id in chosen_ids:
                decision = SamplingDecision(
                    region_id=s.region_id,
                    row=s.row,
                    col=s.col,
                    is_selected=True,
                    selection_reason="equal_budget_uniform",
                    combined_stress_score=s.combined_stress_score,
                    stress_category=s.stress_category.value,
                    is_valid=True,
                )
                selected_decisions.append(decision)
            else:
                decision = SamplingDecision(
                    region_id=s.region_id,
                    row=s.row,
                    col=s.col,
                    is_selected=False,
                    selection_reason="not_selected",
                    combined_stress_score=s.combined_stress_score,
                    stress_category=s.stress_category.value,
                    is_valid=True,
                )
            all_decisions.append(decision)

        n_selected = len(selected_decisions)
        reduction_total = ((total - n_selected) / total * 100.0) if total > 0 else 0.0
        reduction_valid = ((n_valid - n_selected) / n_valid * 100.0) if n_valid > 0 else 0.0

        return SamplingResult(
            strategy=SamplingStrategy.EQUAL_BUDGET_UNIFORM,
            total_regions=total,
            valid_regions=n_valid,
            invalid_regions=invalid_count,
            requested_budget=budget,
            actual_selected_count=n_selected,
            sampling_reduction_pct=reduction_total,
            valid_reduction_pct=reduction_valid,
            seed=seed,
            overflow_policy=None,
            budget_exceeded=False,
            overflow_count=0,
            deficit_count=0,
            selected_regions=selected_decisions,
            all_decisions=all_decisions,
            metadata={
                "requested_k": budget,
                "selected_k": n_selected,
                "insufficient_valid_regions": budget > n_valid,
            },
        )


class AdaptiveSampler:
    """Implements Adaptive Sampling with RSDI prioritization and exploration."""

    def __init__(
        self,
        overflow_policy: BudgetOverflowPolicy = BudgetOverflowPolicy.GUARANTEE_HIGH_STRESS,
        healthy_exploration_rate: float = 0.05,
        default_seed: int = 42,
    ) -> None:
        """Initialize AdaptiveSampler.

        Args:
            overflow_policy: Policy when high-stress regions exceed requested budget.
                             Defaults to GUARANTEE_HIGH_STRESS.
            healthy_exploration_rate: Fraction of healthy regions (or budget) allocated
                                      for exploratory monitoring (default 0.05 / 5%).
            default_seed: Default random seed for healthy exploration selection.
        """
        if not (0.0 <= healthy_exploration_rate <= 1.0):
            raise ValueError(
                f"healthy_exploration_rate must be in [0.0, 1.0], got {healthy_exploration_rate}"
            )
        self.overflow_policy = overflow_policy
        self.healthy_exploration_rate = float(healthy_exploration_rate)
        self.default_seed = int(default_seed)

    def sample(
        self,
        scores: List[RegionStressScore],
        budget: Optional[int] = None,
        overflow_policy: Optional[BudgetOverflowPolicy] = None,
        healthy_exploration_rate: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> SamplingResult:
        """Execute adaptive sampling over scored grid regions.

        Prioritizes HIGH_STRESS regions, allocates budget to MODERATE_STRESS regions,
        and samples a 5% exploration baseline of HEALTHY regions. Excludes INVALID regions.

        CATEGORY CLASSIFICATION (Single Source of Truth):
        AdaptiveSampler consumes the categorical classification (StressCategory) assigned
        by StressScorer on each RegionStressScore object. It does not independently reclassify
        or apply duplicate threshold logic.

        EXPLORATION PRECEDENCE & TRADEOFF:
        In budget-constrained mode, healthy exploration slots are evaluated and reserved
        BEFORE allocating remaining slots to moderate-stress regions.
        Agronomic Trade-off:
        When available budget slots are scarce (e.g., only 1 remaining slot after high-stress
        selection), dedicating that slot to healthy exploration prevents surveillance blind
        spots across vast healthy canopy. This trades off secondary confirmation of known
        moderate stress in favor of discovering unexpected, nascent anomaly emergence elsewhere.

        Args:
            scores: List of RegionStressScore objects across the grid.
            budget: Optional integer cap on selected regions. If None, runs in
                    threshold-driven mode (all high-stress + score-proportional moderate).
            overflow_policy: Overrides default policy when high-stress count > budget.
            healthy_exploration_rate: Overrides default exploration fraction.
            seed: Pinned random seed for reproducible healthy exploration sampling.

        Returns:
            SamplingResult containing selected regions, counts, reduction %, and telemetry.
        """
        if budget is not None and budget < 0:
            raise ValueError(f"budget must be non-negative, got {budget}")

        policy = overflow_policy or self.overflow_policy
        p_explore = (
            healthy_exploration_rate
            if healthy_exploration_rate is not None
            else self.healthy_exploration_rate
        )
        rng_seed = seed if seed is not None else self.default_seed
        rng = np.random.RandomState(rng_seed)

        total = len(scores)
        valid_scores = [s for s in scores if s.is_valid]
        n_valid = len(valid_scores)
        invalid_count = total - n_valid

        # Partition valid regions by RSDI category
        high_tier = [s for s in valid_scores if s.stress_category == StressCategory.HIGH_STRESS]
        mod_tier = [s for s in valid_scores if s.stress_category == StressCategory.MODERATE_STRESS]
        healthy_tier = [s for s in valid_scores if s.stress_category == StressCategory.HEALTHY]

        # Sort high and moderate tiers by RSDI score descending
        high_tier.sort(key=lambda x: x.combined_stress_score, reverse=True)
        mod_tier.sort(key=lambda x: x.combined_stress_score, reverse=True)

        selected_map: Dict[str, str] = {}  # region_id -> reason
        budget_exceeded = False
        overflow_count = 0
        deficit_count = 0

        # ======================================================================
        # BRANCH A: THRESHOLD-DRIVEN MODE (budget is None)
        # ======================================================================
        if budget is None:
            # 1. 100% of HIGH_STRESS regions are mandatorily selected
            for s in high_tier:
                selected_map[s.region_id] = "high_stress_guarantee"

            # 2. All valid MODERATE_STRESS regions are selected deterministically
            for s in mod_tier:
                selected_map[s.region_id] = "moderate_stress_priority"

            # 3. HEALTHY regions: sampled via explicitly randomized exploration at rate p_explore
            n_healthy_explored = 0
            for s in healthy_tier:
                if rng.rand() < p_explore:
                    selected_map[s.region_id] = "healthy_exploration"
                    n_healthy_explored += 1

            target_explore_count = int(np.round(len(healthy_tier) * p_explore))
            actual_explore_count = n_healthy_explored
            exploration_status = (
                "fulfilled_probabilistic_threshold"
                if actual_explore_count >= target_explore_count
                else "probabilistic_below_expected"
            )
            moderate_status = (
                "all_moderate_selected" if len(mod_tier) > 0 else "no_moderate_regions_exist"
            )
            is_discretionary_min = False

        # ======================================================================
        # BRANCH B: BUDGET-CONSTRAINED MODE (budget is an integer)
        # ======================================================================
        else:
            effective_budget = min(budget, n_valid)
            n_high = len(high_tier)
            n_mod = len(mod_tier)
            n_healthy = len(healthy_tier)

            actual_explore_count = 0
            is_discretionary_min = False

            if effective_budget == 0:
                target_explore_count = 0
                exploration_status = "unfulfilled_zero_budget"
                moderate_status = "zero_budget"

            elif n_high > effective_budget:
                target_explore_count = int(np.round(effective_budget * p_explore))
                if policy == BudgetOverflowPolicy.GUARANTEE_HIGH_STRESS:
                    # Mandatory guarantee overrides requested budget cap
                    for s in high_tier:
                        selected_map[s.region_id] = "high_stress_guarantee"
                    budget_exceeded = True
                    overflow_count = n_high - effective_budget
                    exploration_status = "unfulfilled_high_stress_overflow"
                    moderate_status = "unfulfilled_high_stress_overflow"
                else:  # STRICT_CAP
                    # Enforce strict budget cap, selecting top-B high-stress regions
                    for s in high_tier[:effective_budget]:
                        selected_map[s.region_id] = "high_stress_guarantee"
                    deficit_count = n_high - effective_budget
                    exploration_status = "unfulfilled_strict_cap_exhausted_by_high_stress"
                    moderate_status = "unfulfilled_strict_cap_exhausted_by_high_stress"

            elif n_high == effective_budget:
                target_explore_count = int(np.round(effective_budget * p_explore))
                # All high-stress selected, exactly consumes entire budget
                for s in high_tier:
                    selected_map[s.region_id] = "high_stress_guarantee"
                exploration_status = "unfulfilled_budget_exhausted_by_high_stress"
                moderate_status = "unfulfilled_budget_exhausted_by_high_stress"

            else:
                # High-stress regions fit within budget with remaining slots
                for s in high_tier:
                    selected_map[s.region_id] = "high_stress_guarantee"

                slots_remaining = effective_budget - n_high

                # Target exploration count: round(effective_budget * p_explore)
                # When budget is small (e.g. B < 10) such that round(B * 0.05) == 0,
                # but healthy regions exist and slots_remaining > 0, allocate 1 slot
                # as a discrete discretionary minimum to permit baseline surveillance.
                calc_explore = int(np.round(effective_budget * p_explore))
                if calc_explore == 0 and p_explore > 0.0 and slots_remaining > 0 and n_healthy > 0:
                    target_explore_count = 1
                    is_discretionary_min = True
                else:
                    target_explore_count = calc_explore
                    is_discretionary_min = False

                slots_for_explore = min(target_explore_count, slots_remaining, n_healthy)

                if slots_for_explore > 0:
                    chosen_healthy_idx = rng.choice(n_healthy, size=slots_for_explore, replace=False)
                    for idx in chosen_healthy_idx:
                        selected_map[healthy_tier[idx].region_id] = "healthy_exploration"
                    actual_explore_count = slots_for_explore
                    slots_remaining -= slots_for_explore

                    if actual_explore_count == target_explore_count:
                        exploration_status = (
                            "fulfilled_discretionary_minimum"
                            if is_discretionary_min
                            else "fulfilled"
                        )
                    else:
                        exploration_status = "partial_budget_capped"
                else:
                    if n_healthy == 0:
                        exploration_status = "unfulfilled_no_healthy_regions"
                    elif slots_remaining == 0:
                        exploration_status = "unfulfilled_no_slots_remaining"
                    else:
                        exploration_status = "unfulfilled_zero_target"

                # Allocate remaining slots to moderate-stress regions (ranked by RSDI descending)
                if n_mod == 0:
                    moderate_status = "no_moderate_regions_exist"
                elif slots_remaining == 0:
                    moderate_status = "unfulfilled_no_slots_remaining"
                elif n_mod <= slots_remaining:
                    for s in mod_tier:
                        selected_map[s.region_id] = "moderate_stress_priority"
                    slots_remaining -= n_mod
                    moderate_status = "all_moderate_selected"
                else:  # n_mod > slots_remaining
                    for s in mod_tier[:slots_remaining]:
                        selected_map[s.region_id] = "moderate_stress_priority"
                    moderate_status = f"partial_moderate_selected_{slots_remaining}_of_{n_mod}"
                    slots_remaining = 0

                # Note: Any remaining unfilled budget slots are intentionally left unallocated
                # to maximize drone energy/bandwidth reduction rather than over-sampling healthy regions.

        # Build final decision objects for every region
        all_decisions: List[SamplingDecision] = []
        selected_decisions: List[SamplingDecision] = []

        for s in scores:
            if not s.is_valid:
                decision = SamplingDecision(
                    region_id=s.region_id,
                    row=s.row,
                    col=s.col,
                    is_selected=False,
                    selection_reason="invalid_excluded",
                    combined_stress_score=0.0,
                    stress_category=StressCategory.INVALID.value,
                    is_valid=False,
                )
            elif s.region_id in selected_map:
                decision = SamplingDecision(
                    region_id=s.region_id,
                    row=s.row,
                    col=s.col,
                    is_selected=True,
                    selection_reason=selected_map[s.region_id],
                    combined_stress_score=s.combined_stress_score,
                    stress_category=s.stress_category.value,
                    is_valid=True,
                )
                selected_decisions.append(decision)
            else:
                decision = SamplingDecision(
                    region_id=s.region_id,
                    row=s.row,
                    col=s.col,
                    is_selected=False,
                    selection_reason="not_selected",
                    combined_stress_score=s.combined_stress_score,
                    stress_category=s.stress_category.value,
                    is_valid=True,
                )
            all_decisions.append(decision)

        # Invariant checks: ensure zero duplicates and valid counts match
        selected_ids = [d.region_id for d in selected_decisions]
        assert len(selected_ids) == len(set(selected_ids)), "Duplicate region IDs detected in selection"
        assert all(d.is_valid for d in selected_decisions), "Invalid region detected in selected decisions"

        n_selected = len(selected_decisions)
        reduction_total = ((total - n_selected) / total * 100.0) if total > 0 else 0.0
        reduction_valid = ((n_valid - n_selected) / n_valid * 100.0) if n_valid > 0 else 0.0

        # Calculate actual exploration percentages safely
        actual_explore_pct_of_selected = (
            round((actual_explore_count / n_selected) * 100.0, 2) if n_selected > 0 else 0.0
        )
        eff_b = min(budget, n_valid) if budget is not None else 0
        if eff_b > 0:
            actual_explore_pct_of_budget = round(
                (actual_explore_count / eff_b) * 100.0, 2
            )
        else:
            actual_explore_pct_of_budget = 0.0

        metadata = {
            "tier_counts": {
                "high_stress": len(high_tier),
                "moderate_stress": len(mod_tier),
                "healthy": len(healthy_tier),
                "invalid": invalid_count,
            },
            "selected_tier_counts": {
                "high_stress": sum(1 for d in selected_decisions if d.stress_category == "high_stress"),
                "moderate_stress": sum(1 for d in selected_decisions if d.stress_category == "moderate_stress"),
                "healthy": sum(1 for d in selected_decisions if d.stress_category == "healthy"),
            },
            "exploration": {
                "target_exploration_rate": p_explore,
                "target_exploration_count": target_explore_count,
                "actual_exploration_count": actual_explore_count,
                "actual_exploration_pct_of_selected": actual_explore_pct_of_selected,
                "actual_exploration_pct_of_budget": actual_explore_pct_of_budget,
                "is_discretionary_minimum": is_discretionary_min,
                "exploration_status": exploration_status,
                "exploration_fulfilled": (
                    actual_explore_count >= target_explore_count
                    and target_explore_count > 0
                ) or (target_explore_count == 0 and actual_explore_count == 0),
            },
            "moderate_status": moderate_status,
            "policy": policy.value,
            "insufficient_valid_regions": (budget > n_valid) if budget is not None else False,
        }

        return SamplingResult(
            strategy=SamplingStrategy.ADAPTIVE,
            total_regions=total,
            valid_regions=n_valid,
            invalid_regions=invalid_count,
            requested_budget=budget,
            actual_selected_count=n_selected,
            sampling_reduction_pct=reduction_total,
            valid_reduction_pct=reduction_valid,
            seed=rng_seed,
            overflow_policy=policy,
            budget_exceeded=budget_exceeded,
            overflow_count=overflow_count,
            deficit_count=deficit_count,
            selected_regions=selected_decisions,
            all_decisions=all_decisions,
            metadata=metadata,
        )


class SamplingEngine:
    """Unified facade coordinating scoring and multi-strategy sampling."""

    def __init__(
        self,
        overflow_policy: BudgetOverflowPolicy = BudgetOverflowPolicy.GUARANTEE_HIGH_STRESS,
        healthy_exploration_rate: float = 0.05,
        default_seed: int = 42,
    ) -> None:
        """Initialize SamplingEngine facade."""
        self.adaptive_sampler = AdaptiveSampler(
            overflow_policy=overflow_policy,
            healthy_exploration_rate=healthy_exploration_rate,
            default_seed=default_seed,
        )
        self.uniform_sampler = UniformSampler()
        self.default_seed = default_seed

    def sample_full_uniform(
        self,
        scores: List[RegionStressScore],
        seed: Optional[int] = None,
    ) -> SamplingResult:
        """Execute Full-Coverage Uniform Sampling (100% valid regions)."""
        return self.uniform_sampler.sample_full_coverage(
            scores=scores,
            seed=seed if seed is not None else self.default_seed,
        )

    def sample_equal_budget_uniform(
        self,
        scores: List[RegionStressScore],
        budget: int,
        seed: Optional[int] = None,
    ) -> SamplingResult:
        """Execute Equal-Budget Uniform Sampling (randomly selects exactly K regions)."""
        return self.uniform_sampler.sample_equal_budget(
            scores=scores,
            budget=budget,
            seed=seed if seed is not None else self.default_seed,
        )

    def sample_adaptive(
        self,
        scores: List[RegionStressScore],
        budget: Optional[int] = None,
        overflow_policy: Optional[BudgetOverflowPolicy] = None,
        healthy_exploration_rate: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> SamplingResult:
        """Execute Adaptive Sampling with RSDI prioritization and 5% exploration."""
        return self.adaptive_sampler.sample(
            scores=scores,
            budget=budget,
            overflow_policy=overflow_policy,
            healthy_exploration_rate=healthy_exploration_rate,
            seed=seed if seed is not None else self.default_seed,
        )

    def compare_all_strategies(
        self,
        scores: List[RegionStressScore],
        adaptive_budget: Optional[int] = None,
        overflow_policy: Optional[BudgetOverflowPolicy] = None,
        healthy_exploration_rate: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, SamplingResult]:
        """Execute all three sampling strategies on the identical grid for fair comparison.

        Runs Adaptive Sampling first to establish the selected count K, then runs
        Equal-Budget Uniform Sampling with budget K, and Full-Coverage Uniform.

        Args:
            scores: List of RegionStressScore objects across the grid.
            adaptive_budget: Optional budget constraint for Adaptive Sampling.
            overflow_policy: Overflow policy if high-stress count > budget.
            healthy_exploration_rate: Exploration fraction for healthy regions.
            seed: Pinned random seed for both adaptive exploration and equal-budget selection.

        Returns:
            Dict containing 'adaptive', 'equal_budget_uniform', and 'full_uniform' results.
        """
        rng_seed = seed if seed is not None else self.default_seed

        # 1. Run Adaptive Sampling
        adaptive_res = self.sample_adaptive(
            scores=scores,
            budget=adaptive_budget,
            overflow_policy=overflow_policy,
            healthy_exploration_rate=healthy_exploration_rate,
            seed=rng_seed,
        )

        # 2. Run Equal-Budget Uniform Sampling with K = adaptive.actual_selected_count
        k_target = adaptive_res.actual_selected_count
        equal_budget_res = self.sample_equal_budget_uniform(
            scores=scores,
            budget=k_target,
            seed=rng_seed,
        )

        # 3. Run Full-Coverage Uniform Sampling
        full_uniform_res = self.sample_full_uniform(
            scores=scores,
            seed=rng_seed,
        )

        return {
            "adaptive": adaptive_res,
            "equal_budget_uniform": equal_budget_res,
            "full_uniform": full_uniform_res,
        }
