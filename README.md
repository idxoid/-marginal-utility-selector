# marginal-utility-selector

Standalone Python library for selecting LLM/RAG context by marginal utility
instead of filling the token budget greedily.

It extracts the idea from `docs/spec_unified_ranking.md`:

```text
pool -> sort by blended_score -> incremental selection with stop condition

utility(c) =
    blended_score(c)
  - redundancy(c, chosen)
  - lambda * token_cost(c)
```

The point is token economics: additional context should be included only while
it still pays for itself.

## Install

```bash
pip install -e ./marginal-utility-selector
```

No runtime dependencies beyond the Python standard library.

## Quick Start

```python
from marginal_utility_selector import BudgetConfig, Candidate, select_by_marginal_utility

pool = [
    Candidate(uid="validate_amount", token_cost=120, graph_score=0.9, semantic_score=0.72),
    Candidate(uid="doc-negative-amounts", kind="doc", token_cost=180, semantic_score=0.91),
    Candidate(uid="distant_test_helper", token_cost=220, graph_score=0.18),
]

result = select_by_marginal_utility(
    pool,
    BudgetConfig(hard_cap=1200, reserved_tokens=100, min_utility=0.12, token_lambda=0.005),
)

print(result.selected_uids)
print(result.spent, result.stopped_reason)
```

## Graph + Vector Integration

Merge graph traversal candidates with vector-search candidates before selection:

```python
from marginal_utility_selector import BudgetConfig, fuse_candidate_pools, select_by_marginal_utility

graph_results = [
    {
        "uid": "validate_amount",
        "kind": "symbol",
        "token_cost": 120,
        "graph_score": 0.9,
        "file_path": "payments/validators.py",
        "range": [10, 45],
        "provenance": ["graph:CALLS_DIRECT,depth=1"],
    }
]

vector_results = [
    {
        "uid": "validate_amount",
        "kind": "symbol",
        "semantic_score": 0.72,
        "provenance": ["vector:symbols,sim=0.72"],
    },
    {
        "uid": "doc-negative-amounts",
        "kind": "doc",
        "token_cost": 180,
        "semantic_score": 0.91,
        "covers_uids": ["validate_amount"],
        "provenance": ["vector:docs,sim=0.91"],
    },
]

pool = fuse_candidate_pools(graph_results, vector_results)
result = select_by_marginal_utility(pool, BudgetConfig(hard_cap=1500))
```

When the same UID appears in both tracks, the merged candidate keeps the max
graph score, max semantic score, union provenance, and is eligible for the
default overlap bonus.

## Custom Scoring

Swap the blend function when your retrieval signals differ:

```python
from marginal_utility_selector import MarginalUtilitySelector

selector = MarginalUtilitySelector(
    blend=lambda c: 0.3 * c.graph_score + 1.2 * c.semantic_score + 0.2 * c.intent_weight
)
```

You can also replace the redundancy function to use embeddings or domain
coverage rules.

## Greedy Fill vs Marginal Utility

Greedy fill:

```text
sort by blended_score
take candidates while they fit
goal: spend the budget
```

Marginal utility:

```text
sort by blended_score
estimate incremental utility against already-selected context
stop when the next candidate is not worth its token cost
goal: buy useful evidence, not tokens
```

Example behavior:

| Pool | Greedy fill | Marginal utility |
| --- | --- | --- |
| One strong symbol + two weak neighbors | includes all if budget fits | stops after the strong item |
| Repeated same-file ranges | often includes both | penalizes overlap |
| Doc covering selected symbol | may include redundant explanation | keeps only if utility remains positive |
| Complex query with strong graph and vector hits | continues selecting | continues until utility threshold |

Run a direct comparison:

```python
from marginal_utility_selector import compare_strategies

comparison = compare_strategies(pool, BudgetConfig(hard_cap=4000))
print(comparison.greedy.selected_uids)
print(comparison.marginal.selected_uids)
print(comparison.token_delta)
```

## Tuning Notes

- `token_lambda`: default `0.005`; practical range `0.003-0.01`.
- `min_utility`: default `0.12`; raise for terse answers, lower for deep exploration.
- `trust_credit`: lets high-trust flows exceed a base budget while respecting `hard_cap`.
- `allow_signature_fallback`: shrinks deep candidates to signature-only cost when useful but expensive.
# -marginal-utility-selector
