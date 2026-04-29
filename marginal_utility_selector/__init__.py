"""Public API for marginal-utility-selector."""

from marginal_utility_selector.scoring import (
    RankerWeights,
    default_blend,
    default_redundancy,
    line_ranges_touch,
)
from marginal_utility_selector.selector import (
    MarginalUtilitySelector,
    compare_strategies,
    fuse_candidate_pools,
    greedy_fill,
    select_by_marginal_utility,
)
from marginal_utility_selector.types import (
    BudgetConfig,
    Candidate,
    Rejection,
    SelectionResult,
    StopReason,
)

__all__ = [
    "BudgetConfig",
    "Candidate",
    "MarginalUtilitySelector",
    "RankerWeights",
    "Rejection",
    "SelectionResult",
    "StopReason",
    "compare_strategies",
    "default_blend",
    "default_redundancy",
    "fuse_candidate_pools",
    "greedy_fill",
    "line_ranges_touch",
    "select_by_marginal_utility",
]
