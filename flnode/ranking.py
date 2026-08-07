"""Deterministic, context-aware ranking of nodes for the FL aggregator role.

This module implements the "controlled aggregator re-assignment" building block
(TAAS article, Section 5.4). Its defining requirement is stated there:

    "A key requirement for this ranking policy is that it must produce
    unambiguous and deterministic results so that each node, when performing
    its independent assessment, arrives at the exact same ranking."

Every node runs this code on its own replicated copy of the context table and
must reach the same answer without any further coordination. Three things are
needed to actually guarantee that, and all three are easy to get wrong:

1. **Staleness filtering.** A crashed node's context record stays in the
   replicated dictionary forever -- nothing removes it. Without a TTL, a node
   that has just died remains a ranking candidate and can be "elected" to
   replace itself. Records older than `ttl` are dropped.

2. **Explicit tie-breaking.** Sorting float scores alone is only deterministic
   when there are no ties. Ranking is on `(score, node)`, so equal scores fall
   back to the node address, which is unique and totally ordered.

3. **Order-independent arithmetic.** Two nodes may iterate the replicated
   dictionary in different orders, and floating-point addition is not
   associative, so a naive `sum(...) / len(...)` can yield results that differ
   in the last bits on different nodes -- enough to flip a ranking. `mean_of`
   uses `math.fsum`, which is correctly rounded and therefore order-independent.

This module is deliberately pure: no I/O, no RAFT, no Flower. That is what
makes the determinism claim testable (see tests/test_ranking.py).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from typing import Any, NamedTuple

# A context record as published by flnode.context.ContextMonitor.
ContextRecord = Mapping[str, Any]
ContextTable = Mapping[str, ContextRecord]

#: Default staleness window. Records not refreshed within this many seconds are
#: treated as belonging to a dead node. Should be a small multiple of the
#: context monitor's publication period.
DEFAULT_TTL = 30.0


class Ranked(NamedTuple):
    """One entry of a ranking. Lower `score` is better."""

    node: str
    score: float


def mean_of(values: Iterable[float]) -> float:
    """Order-independent arithmetic mean.

    `math.fsum` tracks an exact running sum and rounds once at the end, so the
    result does not depend on the order in which values arrive. Plain `sum`
    would make the ranking dependent on dictionary iteration order, which is
    not guaranteed to agree across nodes.
    """
    vals = list(values)
    if not vals:
        return math.inf
    return math.fsum(vals) / len(vals)


# --------------------------------------------------------------------------
# Scoring policies
#
# A policy maps one node's context record to a score, where lower is better.
# `math.inf` means "unrankable" and pushes the node to the end.
# --------------------------------------------------------------------------

Policy = Callable[[ContextRecord], float]


def mean_latency_policy(record: ContextRecord) -> float:
    """Rank by mean network latency towards all peers (lower is better).

    This is the policy used in the article's proof of concept (Section 6.1.3):
    "the node with the lowest average latency across the network steps forward
    as the new aggregator".

    Prefers the pre-computed `mean_latency` scalar published by the node
    itself; falls back to recomputing from the per-peer table so that the
    ranking still works if a node published only raw measurements.
    """
    mean = record.get("mean_latency")
    if isinstance(mean, (int, float)) and math.isfinite(mean):
        return float(mean)

    latency = record.get("latency") or {}
    if not latency:
        return math.inf
    # Sort keys so the fallback path is order-independent too.
    finite = [
        float(latency[peer])
        for peer in sorted(latency)
        if isinstance(latency[peer], (int, float)) and math.isfinite(latency[peer])
    ]
    return mean_of(finite) if finite else math.inf


def weighted_policy(weights: Mapping[str, float]) -> Policy:
    """Build a multi-criteria policy: a weighted sum of normalised fields.

    The article notes (Section 8.3) that the ranking "may involve a
    multi-criteria decision-making algorithm that takes into account multiple
    contextual factors, such as network latency, processing power and
    reputation scores". This is the hook for that: each key is read from the
    context record and multiplied by its weight, with lower still meaning
    better. Use a negative weight for fields where higher is better
    (e.g. battery level).

    Fields are summed in sorted key order via `fsum`, so the result is
    order-independent for the same reason `mean_of` is.
    """

    def policy(record: ContextRecord) -> float:
        terms = []
        for key in sorted(weights):
            value = record.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                return math.inf
            terms.append(weights[key] * float(value))
        return math.fsum(terms)

    return policy


# --------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------


def live_nodes(
    contexts: ContextTable,
    *,
    now: float,
    ttl: float = DEFAULT_TTL,
    exclude: Iterable[str] = (),
) -> dict[str, ContextRecord]:
    """Return the context records that are fresh enough to be trusted.

    This is the guard that keeps a just-crashed aggregator from being ranked as
    its own replacement: it stops refreshing its record the moment it dies, so
    it drops out once `ttl` elapses.
    """
    excluded = frozenset(exclude)
    fresh: dict[str, ContextRecord] = {}
    for node, record in contexts.items():
        if node in excluded:
            continue
        ts = record.get("ts")
        if not isinstance(ts, (int, float)):
            continue
        if now - float(ts) > ttl:
            continue
        fresh[node] = record
    return fresh


def rank_nodes(
    contexts: ContextTable,
    *,
    now: float,
    ttl: float = DEFAULT_TTL,
    exclude: Iterable[str] = (),
    policy: Policy = mean_latency_policy,
) -> list[Ranked]:
    """Rank candidate nodes for the aggregator role, best first.

    Given the same context table, this returns the same list on every node --
    that is the whole point. See the module docstring for what "the same"
    depends on.
    """
    fresh = live_nodes(contexts, now=now, ttl=ttl, exclude=exclude)
    scored = [Ranked(node, policy(record)) for node, record in fresh.items()]
    # (score, node): the node address breaks ties deterministically.
    scored.sort(key=lambda entry: (entry.score, entry.node))
    return scored


def select_aggregator(
    contexts: ContextTable,
    *,
    now: float,
    ttl: float = DEFAULT_TTL,
    incumbent: str | None = None,
    margin: float = 0.0,
    exclude: Iterable[str] = (),
    policy: Policy = mean_latency_policy,
    fallback: bool = False,
) -> str | None:
    """Decide which node should hold the aggregator role. `None` if nobody can.

    `fallback` covers the case where every live node is unrankable -- no
    finite score anywhere. That happens legitimately at startup, before the
    first probes complete, and in a single-node cluster with no peers to
    measure against. With `fallback=True` the lowest-addressed live node is
    chosen, which is still deterministic and identical on every node, so the
    cluster makes progress instead of deadlocking. With `fallback=False` the
    caller gets `None` and can decide to keep waiting.

    `margin` adds hysteresis for the *proactive* re-assignment path, where the
    incumbent is alive and we are only considering whether a better node
    exists. The challenger must beat the incumbent by a relative margin before
    the role moves:

        challenger < incumbent * (1 - margin)

    Without this, any measurement jitter flips the role and the cluster spends
    its time handing the aggregator back and forth instead of training.
    `margin=0.0` reproduces the strict-inequality behaviour.

    During *recovery* the incumbent has crashed, so it has already been dropped
    by the staleness filter and `margin` has no effect -- which is correct.
    """
    ranking = rank_nodes(
        contexts, now=now, ttl=ttl, exclude=exclude, policy=policy
    )
    if not ranking:
        return None

    best = ranking[0]
    if not math.isfinite(best.score):
        # Ranking already sorts on (score, node), so with all scores equal to
        # infinity ranking[0] is the lowest-addressed live node -- the same
        # answer on every node.
        return best.node if fallback else None

    if incumbent is None or margin <= 0.0:
        return best.node

    incumbent_entry = next((r for r in ranking if r.node == incumbent), None)
    if incumbent_entry is None or not math.isfinite(incumbent_entry.score):
        # Incumbent is gone or unrankable -> no hysteresis to apply.
        return best.node
    if best.node == incumbent:
        return incumbent

    if best.score < incumbent_entry.score * (1.0 - margin):
        return best.node
    return incumbent
