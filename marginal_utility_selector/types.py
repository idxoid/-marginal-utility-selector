"""Typed models for budget-aware marginal utility selection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any


class StopReason(StrEnum):
    """Why selection stopped."""

    POOL_EXHAUSTED = "pool_exhausted"
    UTILITY_THRESHOLD = "utility_threshold"
    HARD_CAP_REACHED = "hard_cap_reached"
    EMPTY_POOL = "empty_pool"


@dataclass(frozen=True)
class Candidate:
    """A graph, vector, or fused context candidate.

    The model is intentionally generic enough for code symbols, doc chunks, API
    contracts, traces, or any other context item competing for the same token
    budget.
    """

    uid: str
    kind: str = "symbol"
    token_cost: int = 0
    graph_score: float = 0.0
    semantic_score: float = 0.0
    intent_weight: float = 0.0
    source_id: str = ""
    line_range: tuple[int, int] | None = None
    covers_uids: frozenset[str] = field(default_factory=frozenset)
    role: str = ""
    depth: int = 0
    provenance: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def overlap(self) -> bool:
        """True when graph and semantic tracks both found this candidate."""

        return self.graph_score > 0.0 and self.semantic_score > 0.0

    def with_updates(self, **changes: Any) -> "Candidate":
        """Return a copy with selected fields changed."""

        return replace(self, **changes)


@dataclass(frozen=True)
class BudgetConfig:
    """Selection budget and stop-condition knobs."""

    hard_cap: int
    reserved_tokens: int = 0
    base_budget: int | None = None
    trust_credit: int = 0
    min_utility: float = 0.12
    token_lambda: float = 0.005
    stop_after_threshold: bool = True
    allow_signature_fallback: bool = False
    signature_token_cost: int = 80

    @property
    def effective_cap(self) -> int:
        """Budget ceiling, optionally above the base budget via trust credit."""

        cap = self.base_budget if self.base_budget is not None else self.hard_cap
        return min(self.hard_cap, cap + self.trust_credit)


@dataclass(frozen=True)
class Rejection:
    """Candidate that was skipped or caused selection to stop."""

    candidate: Candidate
    reason: str
    utility: float
    token_cost: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SelectionResult:
    """Selected candidates plus budget and observability metadata."""

    selected: list[Candidate]
    rejected: list[Rejection]
    spent: int
    remaining: int
    stopped_reason: StopReason
    utilities: dict[str, float]

    @property
    def selected_uids(self) -> list[str]:
        return [candidate.uid for candidate in self.selected]

    @property
    def token_count(self) -> int:
        return sum(candidate.token_cost for candidate in self.selected)
