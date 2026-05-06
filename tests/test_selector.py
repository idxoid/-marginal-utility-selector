from marginal_utility_selector import (
    BudgetConfig,
    Candidate,
    StopReason,
    compare_strategies,
    default_blend,
    fuse_candidate_pools,
    greedy_fill,
    select_by_marginal_utility,
)


def c(uid, *, tokens=100, graph=0.0, sem=0.0, source="", lines=None, kind="symbol", **kw):
    return Candidate(
        uid=uid,
        kind=kind,
        token_cost=tokens,
        graph_score=graph,
        semantic_score=sem,
        source_id=source,
        line_range=lines,
        **kw,
    )


def test_selects_high_utility_until_threshold() -> None:
    budget = BudgetConfig(hard_cap=1000, min_utility=0.2, token_lambda=0.005)
    pool = [
        c("good", tokens=100, graph=1.0),
        c("weak", tokens=100, graph=0.3),
        c("bad", tokens=100, graph=0.1),
    ]

    result = select_by_marginal_utility(pool, budget, normalize=False)

    assert result.selected_uids == ["good"]
    assert result.stopped_reason == StopReason.UTILITY_THRESHOLD


def test_over_budget_skips_and_keeps_trying() -> None:
    budget = BudgetConfig(hard_cap=300, reserved_tokens=50, min_utility=0.0, token_lambda=0.0)
    pool = [
        c("huge", tokens=400, graph=1.0),
        c("small", tokens=100, graph=0.8),
    ]

    result = select_by_marginal_utility(pool, budget, normalize=False)

    assert result.selected_uids == ["small"]
    assert result.rejected[0].reason == "over_budget"


def test_redundant_same_file_range_is_penalized() -> None:
    budget = BudgetConfig(hard_cap=1000, min_utility=0.2, token_lambda=0.0)
    pool = [
        c("a", tokens=100, graph=0.8, source="/app.py", lines=(1, 20)),
        c("b", tokens=100, graph=0.7, source="/app.py", lines=(18, 30)),
        c("c", tokens=100, graph=0.65, source="/other.py", lines=(1, 5)),
    ]

    result = select_by_marginal_utility(pool, budget, normalize=False)

    assert result.selected_uids == ["a", "c"]
    assert result.rejected[0].candidate.uid == "b"


def test_custom_blend_can_replace_scoring() -> None:
    budget = BudgetConfig(hard_cap=500, min_utility=0.0, token_lambda=0.0)
    pool = [
        c("graph", tokens=50, graph=1.0, sem=0.1),
        c("semantic", tokens=50, graph=0.0, sem=0.9),
    ]

    result = select_by_marginal_utility(
        pool,
        budget,
        blend=lambda item: item.semantic_score,
        normalize=False,
    )

    assert result.selected_uids[0] == "semantic"


def test_fuse_candidate_pools_merges_graph_and_vector_rows() -> None:
    fused = fuse_candidate_pools(
        [{"uid": "validate", "token_cost": 90, "graph_score": 0.8, "provenance": ["graph"]}],
        [{"uid": "validate", "semantic_score": 0.72, "provenance": ["vector"]}],
    )

    assert len(fused) == 1
    assert fused[0].overlap is True
    assert fused[0].graph_score == 0.8
    assert fused[0].semantic_score == 0.72
    assert fused[0].provenance == ("graph", "vector")


def test_doc_covering_selected_symbol_is_redundant() -> None:
    budget = BudgetConfig(hard_cap=1000, min_utility=0.2, token_lambda=0.0)
    pool = [
        c("validate", tokens=100, graph=0.9),
        c("doc-validate", kind="doc", tokens=100, sem=0.65, covers_uids=frozenset({"validate"})),
    ]

    result = select_by_marginal_utility(pool, budget, normalize=False)

    assert result.selected_uids == ["validate"]


def test_signature_fallback_reduces_deep_candidate_cost() -> None:
    budget = BudgetConfig(
        hard_cap=250,
        min_utility=0.0,
        token_lambda=0.0,
        allow_signature_fallback=True,
        signature_token_cost=80,
    )
    pool = [c("deep", tokens=300, graph=0.9, depth=2)]

    result = select_by_marginal_utility(pool, budget, normalize=False)

    assert result.selected[0].token_cost == 80
    assert result.selected[0].metadata["render_mode"] == "signature_only"


def test_compare_greedy_vs_marginal_stops_before_filling_junk() -> None:
    budget = BudgetConfig(hard_cap=1000, min_utility=0.2, token_lambda=0.005)
    pool = [
        c("strong", tokens=100, graph=1.0),
        c("junk-1", tokens=100, graph=0.22),
        c("junk-2", tokens=100, graph=0.21),
    ]

    comparison = compare_strategies(pool, budget, normalize=False)

    assert greedy_fill(pool, budget, normalize=False).selected_uids == [
        "strong",
        "junk-1",
        "junk-2",
    ]
    assert comparison.marginal.selected_uids == ["strong"]
    assert comparison.marginal.spent < comparison.greedy.spent
    assert default_blend(pool[0]) > default_blend(pool[1])


def test_single_positive_score_normalizes_to_one() -> None:
    budget = BudgetConfig(hard_cap=500, min_utility=0.1, token_lambda=0.0)
    pool = [c("only", tokens=100, graph=0.42)]

    result = select_by_marginal_utility(pool, budget)

    assert result.selected_uids == ["only"]
    assert result.utilities["only"] >= 1.0
