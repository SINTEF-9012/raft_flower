"""Tests for the supervisor's client-threshold sizing.

`effective_min_nodes` is what keeps a shrinking cluster training instead of
waiting for nodes that no longer exist. It is pure so that the case which
actually deadlocked can be pinned down without standing up a cluster.
"""

from __future__ import annotations

import pytest

from flnode.node import effective_min_nodes


class TestEffectiveMinNodes:
    def test_the_three_node_deadlock(self):
        """The exact configuration that hung.

        3 nodes, --min-nodes 2. Before the failure there are 2 workers and the
        threshold is satisfiable. After the aggregator dies, 2 nodes remain --
        one aggregator, one worker -- so a fixed threshold of 2 can never be
        met and `flwr run` waits forever.
        """
        assert effective_min_nodes(2, live_members=3) == 2  # healthy: unchanged
        assert effective_min_nodes(2, live_members=2) == 1  # after failure: clamped

    def test_leaves_a_satisfiable_threshold_alone(self):
        assert effective_min_nodes(2, live_members=4) == 2
        assert effective_min_nodes(3, live_members=11) == 3

    def test_clamps_to_available_workers(self):
        # The aggregator runs no SuperNode, so workers == live - 1.
        assert effective_min_nodes(10, live_members=4) == 3
        assert effective_min_nodes(5, live_members=3) == 2

    def test_never_below_one(self):
        """FL needs a client; a threshold of zero would be meaningless."""
        assert effective_min_nodes(2, live_members=1) == 1
        assert effective_min_nodes(0, live_members=4) == 1
        assert effective_min_nodes(-3, live_members=4) == 1
        assert effective_min_nodes(2, live_members=0) == 1

    def test_single_node_cluster(self):
        assert effective_min_nodes(1, live_members=1) == 1

    def test_strict_mode_keeps_the_configured_value(self):
        """Opt-out for experiments that would rather stall than vary the
        client count mid-run."""
        assert effective_min_nodes(2, live_members=2, strict=True) == 2
        assert effective_min_nodes(5, live_members=2, strict=True) == 5

    def test_strict_mode_still_floors_at_one(self):
        assert effective_min_nodes(0, live_members=4, strict=True) == 1

    @pytest.mark.parametrize("nodes", [3, 4, 5, 7, 11])
    def test_survives_successive_failures(self, nodes):
        """Chained failures must keep yielding a runnable threshold.

        RAFT tolerates floor((N-1)/2) losses; across every survivable cluster
        size the threshold must stay at least 1 and never exceed the workers
        actually present.
        """
        configured = nodes - 1
        tolerable = (nodes - 1) // 2
        for failures in range(tolerable + 1):
            live = nodes - failures
            got = effective_min_nodes(configured, live)
            assert got >= 1
            assert got <= max(1, live - 1)
