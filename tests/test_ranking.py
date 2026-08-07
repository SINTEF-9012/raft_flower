"""Tests for flnode.ranking.

The first test class is the important one: it enforces the TAAS article's
Section 5.4 claim that every node, ranking independently, reaches the same
conclusion. That claim was previously neither implemented nor tested.
"""

from __future__ import annotations

import math
import random

import pytest

from flnode.ranking import (
    Ranked,
    live_nodes,
    mean_latency_policy,
    mean_of,
    rank_nodes,
    select_aggregator,
    weighted_policy,
)

NOW = 1_000_000.0


def ctx(node: str, mean_latency: float, *, age: float = 0.0, **extra):
    """Build a context record as ContextMonitor would publish it."""
    return {
        "node": node,
        "ts": NOW - age,
        "mean_latency": mean_latency,
        "latency": {},
        **extra,
    }


# ---------------------------------------------------------------------------
# Determinism -- the article's core correctness requirement
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_identical_ranking_under_shuffled_iteration_order(self):
        """Every node must rank identically regardless of dict order.

        Nodes receive the same logical context table but may iterate it in
        different orders. The ranking must not depend on that.
        """
        base = {
            f"10.0.0.{i}:5000": ctx(f"10.0.0.{i}:5000", 40.0 + i * 0.5)
            for i in range(1, 12)
        }
        expected = rank_nodes(base, now=NOW)

        rng = random.Random(1234)
        for _ in range(100):
            items = list(base.items())
            rng.shuffle(items)
            shuffled = dict(items)
            assert rank_nodes(shuffled, now=NOW) == expected

    def test_scores_are_bit_identical_under_shuffled_peer_order(self):
        """Recomputed means must be bit-identical, not merely close.

        A difference in the last bits is enough to flip two adjacent nodes in
        the ranking, which would let two nodes both claim the aggregator role.
        Values are chosen to be unrepresentable in binary floating point so
        that naive left-to-right summation genuinely differs by order.
        """
        peers = {f"peer{i}": 0.1 * i + 0.7 for i in range(1, 25)}

        rng = random.Random(99)
        reference = mean_latency_policy({"latency": peers, "ts": NOW})
        for _ in range(100):
            items = list(peers.items())
            rng.shuffle(items)
            record = {"latency": dict(items), "ts": NOW}
            assert mean_latency_policy(record).hex() == reference.hex()

    def test_mean_of_is_order_independent(self):
        values = [0.1, 0.2, 0.3, 1e16, -1e16, 0.7]
        rng = random.Random(7)
        reference = mean_of(values)
        for _ in range(50):
            shuffled = values[:]
            rng.shuffle(shuffled)
            assert mean_of(shuffled).hex() == reference.hex()

    def test_ties_broken_by_node_address(self):
        """Equal scores must still produce a total, stable order."""
        contexts = {
            "b:5000": ctx("b:5000", 50.0),
            "a:5000": ctx("a:5000", 50.0),
            "c:5000": ctx("c:5000", 50.0),
        }
        assert [r.node for r in rank_nodes(contexts, now=NOW)] == [
            "a:5000",
            "b:5000",
            "c:5000",
        ]

    def test_all_nodes_agree_on_winner_with_ties(self):
        contexts = {f"node{i}:5000": ctx(f"node{i}:5000", 12.5) for i in range(20)}
        rng = random.Random(5)
        winners = set()
        for _ in range(50):
            items = list(contexts.items())
            rng.shuffle(items)
            winners.add(select_aggregator(dict(items), now=NOW))
        assert len(winners) == 1


# ---------------------------------------------------------------------------
# Staleness -- stops a crashed node being elected to replace itself
# ---------------------------------------------------------------------------


class TestStaleness:
    def test_stale_record_is_dropped(self):
        contexts = {
            "dead:5000": ctx("dead:5000", 1.0, age=120.0),  # best score, but stale
            "alive:5000": ctx("alive:5000", 90.0),
        }
        assert [r.node for r in rank_nodes(contexts, now=NOW, ttl=30.0)] == [
            "alive:5000"
        ]

    def test_crashed_aggregator_is_not_reelected(self):
        """The regression this whole mechanism exists to prevent.

        The crashed node had the lowest latency, so a ranking without
        staleness filtering would hand the role straight back to it.
        """
        contexts = {
            "aggr:5000": ctx("aggr:5000", 2.0, age=45.0),
            "w1:5000": ctx("w1:5000", 30.0),
            "w2:5000": ctx("w2:5000", 20.0),
        }
        assert select_aggregator(contexts, now=NOW, ttl=30.0) == "w2:5000"

    def test_record_exactly_at_ttl_is_kept(self):
        contexts = {"n:5000": ctx("n:5000", 5.0, age=30.0)}
        assert live_nodes(contexts, now=NOW, ttl=30.0)

    def test_record_without_timestamp_is_dropped(self):
        contexts = {"n:5000": {"node": "n:5000", "mean_latency": 5.0}}
        assert rank_nodes(contexts, now=NOW) == []

    def test_all_stale_yields_no_aggregator(self):
        contexts = {"a:5000": ctx("a:5000", 5.0, age=999.0)}
        assert select_aggregator(contexts, now=NOW) is None


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


class TestPolicies:
    def test_lowest_mean_latency_wins(self):
        contexts = {
            "a:5000": ctx("a:5000", 245.0),
            "b:5000": ctx("b:5000", 108.0),
            "c:5000": ctx("c:5000", 110.0),
            "d:5000": ctx("d:5000", 125.0),
        }
        # Matches Table 2 of the article, where Node 2 is the suitable one.
        assert select_aggregator(contexts, now=NOW) == "b:5000"

    def test_falls_back_to_per_peer_table(self):
        record = {"ts": NOW, "latency": {"x": 10.0, "y": 20.0, "z": 30.0}}
        assert mean_latency_policy(record) == pytest.approx(20.0)

    def test_unrankable_node_sorts_last(self):
        contexts = {
            "good:5000": ctx("good:5000", 50.0),
            "empty:5000": {"node": "empty:5000", "ts": NOW, "latency": {}},
        }
        ranking = rank_nodes(contexts, now=NOW)
        assert ranking[0].node == "good:5000"
        assert math.isinf(ranking[-1].score)

    def test_no_finite_candidate_yields_none(self):
        contexts = {"empty:5000": {"node": "empty:5000", "ts": NOW, "latency": {}}}
        assert select_aggregator(contexts, now=NOW) is None

    def test_weighted_multi_criteria(self):
        """Negative weight for a higher-is-better field (battery)."""
        policy = weighted_policy({"mean_latency": 1.0, "battery": -100.0})
        contexts = {
            "fast_flat:5000": ctx("fast_flat:5000", 10.0, battery=0.1),
            "slow_full:5000": ctx("slow_full:5000", 50.0, battery=1.0),
        }
        # 10 - 10 = 0  vs  50 - 100 = -50 -> the charged node wins.
        assert select_aggregator(contexts, now=NOW, policy=policy) == "slow_full:5000"

    def test_weighted_policy_is_order_independent(self):
        policy = weighted_policy({f"m{i}": 0.1 * i for i in range(1, 20)})
        record = {"ts": NOW, **{f"m{i}": 0.1 * i + 0.3 for i in range(1, 20)}}
        rng = random.Random(3)
        reference = policy(record)
        for _ in range(50):
            items = list(record.items())
            rng.shuffle(items)
            assert policy(dict(items)).hex() == reference.hex()


# ---------------------------------------------------------------------------
# Hysteresis and exclusion
# ---------------------------------------------------------------------------


class TestHysteresis:
    def test_marginal_improvement_does_not_move_the_role(self):
        contexts = {
            "incumbent:5000": ctx("incumbent:5000", 100.0),
            "challenger:5000": ctx("challenger:5000", 98.0),
        }
        assert (
            select_aggregator(
                contexts, now=NOW, incumbent="incumbent:5000", margin=0.10
            )
            == "incumbent:5000"
        )

    def test_decisive_improvement_moves_the_role(self):
        contexts = {
            "incumbent:5000": ctx("incumbent:5000", 100.0),
            "challenger:5000": ctx("challenger:5000", 50.0),
        }
        assert (
            select_aggregator(
                contexts, now=NOW, incumbent="incumbent:5000", margin=0.10
            )
            == "challenger:5000"
        )

    def test_zero_margin_is_strict_inequality(self):
        contexts = {
            "incumbent:5000": ctx("incumbent:5000", 100.0),
            "challenger:5000": ctx("challenger:5000", 99.9),
        }
        assert (
            select_aggregator(
                contexts, now=NOW, incumbent="incumbent:5000", margin=0.0
            )
            == "challenger:5000"
        )

    def test_dead_incumbent_ignores_margin(self):
        """On the recovery path the incumbent is stale, so hysteresis must not
        keep a crashed node in the role."""
        contexts = {
            "incumbent:5000": ctx("incumbent:5000", 1.0, age=120.0),
            "challenger:5000": ctx("challenger:5000", 500.0),
        }
        assert (
            select_aggregator(
                contexts, now=NOW, incumbent="incumbent:5000", margin=0.99
            )
            == "challenger:5000"
        )

    def test_explicit_exclusion(self):
        contexts = {
            "a:5000": ctx("a:5000", 1.0),
            "b:5000": ctx("b:5000", 2.0),
        }
        assert select_aggregator(contexts, now=NOW, exclude={"a:5000"}) == "b:5000"

    def test_empty_table(self):
        assert rank_nodes({}, now=NOW) == []
        assert select_aggregator({}, now=NOW) is None

    def test_single_node_cluster(self):
        contexts = {"solo:5000": ctx("solo:5000", 0.0)}
        assert select_aggregator(contexts, now=NOW) == "solo:5000"
        assert rank_nodes(contexts, now=NOW) == [Ranked("solo:5000", 0.0)]
