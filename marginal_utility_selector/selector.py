"""Selector implementation and graph/vector integration helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from marginal_utility_selector.scoring import (
    BlendFunction,
    RedundancyFunction,
    cost_aware_rank_score,
    default_blend,
    default_redundancy,
    normalize_scores,
)
from marginal_utility_selector.types import (
    BudgetConfig,
    Candidate,
    Rejection,
    SelectionResult,
    StopReason,
)

RoleBonusFunction = Callable[[Candidate, Sequence[Candidate]], float]


class MarginalUtilitySelector:
    """Select context while each additional candidate still pays for its tokens."""

    def __init__(
        self,
        *,
        blend: BlendFunction = default_blend,
        redundancy: RedundancyFunction = default_redundancy,
        role_bonus: RoleBonusFunction | None = None,
        normalize: bool = True,
    ) -> None:
        self.blend = blend
        self.redundancy = redundancy
        self.role_bonus = role_bonus or _no_role_bonus
        self.normalize = normalize

    def select(
        self,
        candidates: Iterable[Candidate],
        budget: BudgetConfig,
        *,
        already_selected: Sequence[Candidate] = (),
    ) -> SelectionResult:
        return select_by_marginal_utility(
            candidates,
            budget,
            blend=self.blend,
            redundancy=self.redundancy,
            role_bonus=self.role_bonus,
            normalize=self.normalize,
            already_selected=already_selected,
        )


def select_by_marginal_utility(
    candidates: Iterable[Candidate],
    budget: BudgetConfig,
    *,
    blend: BlendFunction = default_blend,
    redundancy: RedundancyFunction = default_redundancy,
    role_bonus: RoleBonusFunction | None = None,
    normalize: bool = True,
    already_selected: Sequence[Candidate] = (),
) -> SelectionResult:
    """Incrementally select candidates by marginal utility.

    Utility:

    `blend(candidate) + role_bonus - redundancy(candidate, chosen) - λ * token_cost`
    """

    pool = list(candidates)
    if normalize:
        pool = normalize_scores(pool)
    pool.sort(key=blend, reverse=True)

    chosen = list(already_selected)
    selected: list[Candidate] = []
    rejected: list[Rejection] = []
    utilities: dict[str, float] = {}
    spent = budget.reserved_tokens + sum(max(0, item.token_cost) for item in chosen)
    role_bonus = role_bonus or _no_role_bonus

    if not pool:
        return _result(selected, rejected, spent, budget, StopReason.EMPTY_POOL, utilities)

    stopped_reason = StopReason.POOL_EXHAUSTED
    for candidate in pool:
        token_cost = _effective_token_cost(candidate, budget)
        blended_score = blend(candidate)
        bonus = role_bonus(candidate, chosen)
        redundancy_penalty = redundancy(candidate, chosen)
        utility = blended_score + bonus - redundancy_penalty - budget.token_lambda * token_cost
        utilities[candidate.uid] = utility

        if utility < budget.min_utility:
            rejection = Rejection(
                candidate=candidate,
                reason="utility_threshold",
                utility=utility,
                token_cost=token_cost,
                metadata={
                    "blended_score": blended_score,
                    "role_bonus": bonus,
                    "redundancy": redundancy_penalty,
                },
            )
            rejected.append(rejection)
            if budget.stop_after_threshold and redundancy_penalty <= 0:
                stopped_reason = StopReason.UTILITY_THRESHOLD
                break
            continue

        if spent + token_cost > budget.effective_cap:
            rejected.append(
                Rejection(
                    candidate=candidate,
                    reason="over_budget",
                    utility=utility,
                    token_cost=token_cost,
                    metadata={"spent": spent, "cap": budget.effective_cap},
                )
            )
            if spent >= budget.effective_cap:
                stopped_reason = StopReason.HARD_CAP_REACHED
                break
            continue

        selected_candidate = candidate
        if token_cost != candidate.token_cost:
            selected_candidate = candidate.with_updates(
                token_cost=token_cost,
                metadata={**dict(candidate.metadata), "render_mode": "signature_only"},
            )

        selected.append(selected_candidate)
        chosen.append(selected_candidate)
        spent += token_cost

    return _result(selected, rejected, spent, budget, stopped_reason, utilities)


def greedy_fill(
    candidates: Iterable[Candidate],
    budget: BudgetConfig,
    *,
    blend: BlendFunction = default_blend,
    normalize: bool = True,
    cost_aware: bool = False,
) -> SelectionResult:
    """Baseline selector: sort once and fill until the budget is exhausted."""

    pool = list(candidates)
    if normalize:
        pool = normalize_scores(pool)
    key = (lambda candidate: cost_aware_rank_score(candidate, blend)) if cost_aware else blend
    pool.sort(key=key, reverse=True)

    selected: list[Candidate] = []
    rejected: list[Rejection] = []
    utilities: dict[str, float] = {}
    spent = budget.reserved_tokens
    for candidate in pool:
        score = blend(candidate)
        utilities[candidate.uid] = score
        if spent + candidate.token_cost > budget.effective_cap:
            rejected.append(
                Rejection(
                    candidate=candidate,
                    reason="over_budget",
                    utility=score,
                    token_cost=candidate.token_cost,
                    metadata={"spent": spent, "cap": budget.effective_cap},
                )
            )
            continue
        selected.append(candidate)
        spent += candidate.token_cost

    return _result(selected, rejected, spent, budget, StopReason.POOL_EXHAUSTED, utilities)


def fuse_candidate_pools(
    *pools: Iterable[Candidate | Mapping[str, Any]],
    normalize: bool = False,
) -> list[Candidate]:
    """Merge graph/vector result pools by UID.

    This is the small integration adapter you need when graph traversal and
    vector search return separate rows for the same symbol or doc chunk.
    """

    fused: dict[str, Candidate] = {}
    for pool in pools:
        for raw in pool:
            candidate = _coerce_candidate(raw)
            existing = fused.get(candidate.uid)
            if existing is None:
                fused[candidate.uid] = candidate
                continue
            fused[candidate.uid] = _merge_candidates(existing, candidate)

    candidates = list(fused.values())
    return normalize_scores(candidates) if normalize else candidates


@dataclass(frozen=True)
class StrategyComparison:
    greedy: SelectionResult
    marginal: SelectionResult

    @property
    def token_delta(self) -> int:
        return self.marginal.spent - self.greedy.spent

    @property
    def selected_overlap(self) -> set[str]:
        return set(self.greedy.selected_uids) & set(self.marginal.selected_uids)


def compare_strategies(
    candidates: Iterable[Candidate],
    budget: BudgetConfig,
    *,
    blend: BlendFunction = default_blend,
    redundancy: RedundancyFunction = default_redundancy,
    normalize: bool = True,
) -> StrategyComparison:
    """Run greedy filling and marginal utility selection on the same pool."""

    pool = list(candidates)
    greedy = greedy_fill(pool, budget, blend=blend, normalize=normalize)
    marginal = select_by_marginal_utility(
        pool,
        budget,
        blend=blend,
        redundancy=redundancy,
        normalize=normalize,
    )
    return StrategyComparison(greedy=greedy, marginal=marginal)


def _result(
    selected: list[Candidate],
    rejected: list[Rejection],
    spent: int,
    budget: BudgetConfig,
    stopped_reason: StopReason,
    utilities: dict[str, float],
) -> SelectionResult:
    remaining = max(0, budget.effective_cap - spent)
    return SelectionResult(
        selected=selected,
        rejected=rejected,
        spent=spent,
        remaining=remaining,
        stopped_reason=stopped_reason,
        utilities=utilities,
    )


def _effective_token_cost(candidate: Candidate, budget: BudgetConfig) -> int:
    if budget.allow_signature_fallback and candidate.depth >= 2:
        return min(candidate.token_cost, budget.signature_token_cost)
    return max(0, candidate.token_cost)


def _no_role_bonus(candidate: Candidate, chosen: Sequence[Candidate]) -> float:
    return 0.0


def _coerce_candidate(raw: Candidate | Mapping[str, Any]) -> Candidate:
    if isinstance(raw, Candidate):
        return raw
    source_id = raw.get("source_id", raw.get("file_path", ""))
    line_range = raw.get("line_range", raw.get("range"))
    return Candidate(
        uid=str(raw["uid"]),
        kind=str(raw.get("kind", "symbol")),
        token_cost=int(raw.get("token_cost", raw.get("token_estimate", 0)) or 0),
        graph_score=float(raw.get("graph_score", 0.0) or 0.0),
        semantic_score=float(raw.get("semantic_score", raw.get("score", 0.0)) or 0.0),
        intent_weight=float(raw.get("intent_weight", 0.0) or 0.0),
        source_id=str(source_id or ""),
        line_range=_coerce_line_range(line_range),
        covers_uids=frozenset(str(uid) for uid in raw.get("covers_uids", ())),
        role=str(raw.get("role", raw.get("evidence_role", "")) or ""),
        depth=int(raw.get("depth", 0) or 0),
        provenance=tuple(str(item) for item in raw.get("provenance", ())),
        metadata=dict(raw),
    )


def _merge_candidates(left: Candidate, right: Candidate) -> Candidate:
    provenance = tuple(dict.fromkeys([*left.provenance, *right.provenance]))
    metadata = {**dict(left.metadata), **dict(right.metadata)}
    return left.with_updates(
        token_cost=max(left.token_cost, right.token_cost),
        graph_score=max(left.graph_score, right.graph_score),
        semantic_score=max(left.semantic_score, right.semantic_score),
        intent_weight=max(left.intent_weight, right.intent_weight),
        source_id=left.source_id or right.source_id,
        line_range=left.line_range or right.line_range,
        covers_uids=left.covers_uids | right.covers_uids,
        role=left.role or right.role,
        depth=min(left.depth, right.depth),
        provenance=provenance,
        metadata=metadata,
    )


def _coerce_line_range(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    try:
        start, end = value
    except (TypeError, ValueError):
        return None
    start_int = int(start)
    end_int = int(end)
    if start_int <= end_int:
        return (start_int, end_int)
    return (end_int, start_int)
