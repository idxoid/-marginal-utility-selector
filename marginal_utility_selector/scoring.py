"""Scoring and redundancy helpers for marginal utility selection."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from marginal_utility_selector.types import Candidate

BlendFunction = Callable[[Candidate], float]
RedundancyFunction = Callable[[Candidate, Sequence[Candidate]], float]


@dataclass(frozen=True)
class RankerWeights:
    """Default blend weights copied from the unified-ranking design."""

    alpha: float = 1.0
    beta: float = 0.8
    gamma: float = 0.4
    delta: float = 0.5
    epsilon: float = 0.0


def default_blend(candidate: Candidate, weights: RankerWeights = RankerWeights()) -> float:
    """Blend graph, semantic, intent, and overlap signals.

    Token cost is usually handled by marginal utility's `token_lambda`, so the
    default `epsilon` is zero. Set it when you need backward-compatible blended
    scores that already include a cost penalty.
    """

    overlap_bonus = weights.delta if candidate.overlap else 0.0
    positive = (
        weights.alpha * candidate.graph_score
        + weights.beta * candidate.semantic_score
        + weights.gamma * candidate.intent_weight
        + overlap_bonus
    )
    return positive - weights.epsilon * candidate.token_cost / 100


def default_redundancy(candidate: Candidate, chosen: Sequence[Candidate]) -> float:
    """Deterministic redundancy penalty against already-selected candidates.

    This starts with cheap checks:
    - same UID: duplicate, effectively never useful;
    - overlapping same-source line ranges;
    - doc chunk covers an already selected symbol;
    - repeated same source/file with diminishing returns.
    """

    if not chosen:
        return 0.0

    penalty = 0.0
    for selected in chosen:
        if candidate.uid and candidate.uid == selected.uid:
            penalty = max(penalty, 1.0)

        if (
            candidate.source_id
            and candidate.source_id == selected.source_id
            and line_ranges_touch(candidate.line_range, selected.line_range)
        ):
            penalty = max(penalty, 0.5)

        if candidate.kind == "doc" and selected.uid in candidate.covers_uids:
            penalty = max(penalty, 0.35)
        if selected.kind == "doc" and candidate.uid in selected.covers_uids:
            penalty = max(penalty, 0.35)

    same_source_count = sum(
        1
        for selected in chosen
        if candidate.source_id and selected.source_id == candidate.source_id
    )
    penalty += min(0.3, 0.08 * same_source_count)
    return min(1.25, penalty)


def line_ranges_touch(
    left: tuple[int, int] | None,
    right: tuple[int, int] | None,
) -> bool:
    """Return true when line ranges overlap or are adjacent."""

    if left is None or right is None:
        return False
    return max(left[0], right[0]) <= min(left[1], right[1]) + 1


def normalize_scores(candidates: Sequence[Candidate]) -> list[Candidate]:
    """Return candidates with graph and semantic scores min-max normalized."""

    graph_values = [candidate.graph_score for candidate in candidates if candidate.graph_score > 0]
    semantic_values = [
        candidate.semantic_score for candidate in candidates if candidate.semantic_score > 0
    ]
    graph_min, graph_max = _bounds(graph_values)
    semantic_min, semantic_max = _bounds(semantic_values)

    normalized: list[Candidate] = []
    for candidate in candidates:
        graph_score = _normalize(candidate.graph_score, graph_min, graph_max)
        semantic_score = _normalize(candidate.semantic_score, semantic_min, semantic_max)
        normalized.append(
            candidate.with_updates(graph_score=graph_score, semantic_score=semantic_score)
        )
    return normalized


def cost_aware_rank_score(candidate: Candidate, blend: BlendFunction) -> float:
    """Greedy baseline rank score: blended score divided by token-cost drag."""

    return blend(candidate) / math.log(candidate.token_cost + 10)


def _bounds(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return (0.0, 1.0)
    unique = set(values)
    if len(unique) == 1:
        value = unique.pop()
        return (0.0, value)
    return (min(values), max(values))


def _normalize(value: float, low: float, high: float) -> float:
    if value <= 0:
        return 0.0
    return (value - low) / (high - low)
