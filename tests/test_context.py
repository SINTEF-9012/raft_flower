"""Tests for flnode.context.

The unreachable-peer handling is the important part: `tcppinglib` reports
`avg_rtt == 0.0` for a host it cannot reach, and treating that as a real
measurement makes dead nodes look like the fastest nodes in the cluster.
"""

from __future__ import annotations

import math
import socket
import threading
import time

import pytest

from flnode.context import (
    ContextMonitor,
    ProbeConfig,
    ProbeListener,
    context_key,
    probe_port_of,
    read_context_table,
)
from flnode.ranking import select_aggregator


class FakeState(dict):
    def set(self, key, value, **kwargs):  # noqa: A003 - matches ReplDict
        self[key] = value

    def get(self, key, default=None):
        return super().get(key, default)


def monitor(state, node, peers, probes=None):
    """Build a monitor whose probe results are scripted."""
    mon = ContextMonitor(
        state, node, peers, fl_port=9092, config=ProbeConfig(period=0.1)
    )
    if probes is not None:
        mon.probe = lambda peer: probes.get(peer)  # type: ignore[method-assign]
    return mon


class TestSampling:
    def test_record_shape(self):
        state = FakeState()
        mon = monitor(
            state, "a:5000", ["b:5000", "c:5000"], {"b:5000": 10.0, "c:5000": 20.0}
        )
        record = mon.sample()
        assert record["node"] == "a:5000"
        assert record["latency"] == {"b:5000": 10.0, "c:5000": 20.0}
        assert record["mean_latency"] == pytest.approx(15.0)
        assert record["unreachable"] == []
        assert isinstance(record["ts"], float)

    def test_unreachable_peer_excluded_from_mean(self):
        """The regression. A dead peer must not contribute 0.0 ms."""
        state = FakeState()
        mon = monitor(
            state,
            "a:5000",
            ["b:5000", "dead:5000"],
            {"b:5000": 40.0, "dead:5000": None},
        )
        record = mon.sample()
        assert record["unreachable"] == ["dead:5000"]
        assert "dead:5000" not in record["latency"]
        # 40.0, not (40.0 + 0.0) / 2 == 20.0
        assert record["mean_latency"] == pytest.approx(40.0)

    def test_dead_peer_does_not_improve_neighbour_ranking(self):
        """End-to-end version of the same bug.

        Two nodes are equally fast to the survivors. If the one that also
        neighboured the crashed node counted it as 0 ms, it would win.
        """
        state = FakeState()
        near = monitor(
            state, "near:5000", ["far:5000", "dead:5000"],
            {"far:5000": 100.0, "dead:5000": None},
        )
        far = monitor(
            state, "far:5000", ["near:5000", "dead:5000"],
            {"near:5000": 100.0, "dead:5000": None},
        )
        near.publish(near.sample())
        far.publish(far.sample())

        table = read_context_table(state)
        assert table["near:5000"]["mean_latency"] == pytest.approx(100.0)
        assert table["far:5000"]["mean_latency"] == pytest.approx(100.0)
        # A tie, broken deterministically by address -- not a spurious win.
        assert select_aggregator(table, now=time.time()) == "far:5000"

    def test_all_peers_unreachable_is_unrankable(self):
        state = FakeState()
        mon = monitor(state, "lonely:5000", ["x:5000"], {"x:5000": None})
        record = mon.sample()
        assert math.isinf(record["mean_latency"])
        mon.publish(record)
        # An isolated node must not be handed the aggregator role.
        assert select_aggregator(read_context_table(state), now=time.time()) is None

    def test_solo_node_has_no_peers(self):
        state = FakeState()
        record = monitor(state, "solo:5000", []).sample()
        assert record["latency"] == {}
        assert math.isinf(record["mean_latency"])


class TestPublishing:
    def test_publishes_under_own_key_only(self):
        state = FakeState()
        mon = monitor(state, "a:5000", ["b:5000"], {"b:5000": 5.0})
        mon.publish(mon.sample())
        assert list(state.keys()) == [context_key("a:5000")]

    def test_publish_reports_size(self):
        state = FakeState()
        mon = monitor(state, "a:5000", ["b:5000"], {"b:5000": 5.0})
        assert mon.publish(mon.sample()) > 0

    def test_read_context_table_ignores_other_keys(self):
        state = FakeState()
        state.set("fl:num_rounds", 3)
        state.set("ckpt:latest", {"round": 1})
        mon = monitor(state, "a:5000", ["b:5000"], {"b:5000": 5.0})
        mon.publish(mon.sample())
        assert set(read_context_table(state)) == {"a:5000"}

    def test_read_context_table_skips_non_dict_values(self):
        state = FakeState()
        state.set(context_key("bad:5000"), "not-a-record")
        assert read_context_table(state) == {}


class TestProbing:
    def test_closed_port_is_unreachable(self):
        """Real probe against a port nothing is listening on."""
        mon = ContextMonitor(
            FakeState(), "a:5000", [], fl_port=9092, probe_offset=54998,
            config=ProbeConfig(timeout=1, count=1, interval=0.1),
        )
        assert mon.probe("127.0.0.1:5000") is None

    def test_probe_listener_makes_a_node_measurable(self):
        """Real probe against this project's own ProbeListener."""
        listener = ProbeListener(0)
        # Bind an ephemeral port, then aim the probe at it.
        listener.port = 0
        listener.start()
        port = listener._sock.getsockname()[1]  # noqa: SLF001 - test only
        try:
            mon = ContextMonitor(
                FakeState(), "a:5000", [],
                fl_port=9092, probe_offset=port - 5000,
                config=ProbeConfig(timeout=2, count=2, interval=0.1),
            )
            rtt = mon.probe("127.0.0.1:5000")
            assert rtt is not None and rtt > 0.0
        finally:
            listener.stop()

    def test_probe_targets_dedicated_port_not_fl_or_raft(self):
        """Probes must ride neither the RAFT port (which would pollute the
        Section 7.2.2 consensus measurement) nor the Fleet port (which only
        exists while that node happens to be the aggregator)."""
        seen = {}

        mon = ContextMonitor(FakeState(), "a:5000", [], fl_port=9092)

        def fake_tcpping(host, port=80, **kwargs):
            seen["host"], seen["port"] = host, port
            raise RuntimeError("stop here")

        import flnode.context as ctx_mod

        original = ctx_mod.tcpping
        ctx_mod.tcpping = fake_tcpping
        try:
            mon.probe("10.0.0.7:5000")
        finally:
            ctx_mod.tcpping = original

        assert seen["host"] == "10.0.0.7"
        assert seen["port"] == 6000  # 5000 + PROBE_PORT_OFFSET
        assert seen["port"] not in (5000, 9092)

    def test_peer_advertised_probe_port_wins(self):
        state = FakeState()
        state.set(context_key("10.0.0.7:5000"), {"probe_port": 7777, "ts": time.time()})
        mon = ContextMonitor(state, "a:5000", [], fl_port=9092)
        assert mon.peer_probe_port("10.0.0.7:5000") == 7777

    def test_probe_port_derivation(self):
        assert probe_port_of("10.0.0.1:5000") == 6000
        assert probe_port_of("10.0.0.2:5003") == 6003

    def test_probe_survives_exceptions(self):
        mon = ContextMonitor(FakeState(), "a:5000", [], fl_port=9092)
        import flnode.context as ctx_mod

        original = ctx_mod.tcpping
        ctx_mod.tcpping = lambda *a, **k: (_ for _ in ()).throw(OSError("boom"))
        try:
            assert mon.probe("x:5000") is None
        finally:
            ctx_mod.tcpping = original


class TestThreadLifecycle:
    def test_publishes_repeatedly_then_stops(self):
        state = FakeState()
        mon = monitor(state, "a:5000", ["b:5000"], {"b:5000": 5.0})
        mon.start()
        try:
            deadline = time.time() + 5
            while time.time() < deadline and context_key("a:5000") not in state:
                time.sleep(0.05)
            assert context_key("a:5000") in state
            first_ts = state[context_key("a:5000")]["ts"]

            deadline = time.time() + 5
            while (
                time.time() < deadline
                and state[context_key("a:5000")]["ts"] == first_ts
            ):
                time.sleep(0.05)
            assert state[context_key("a:5000")]["ts"] > first_ts
        finally:
            mon.stop()
        assert not mon.is_alive()

    def test_stop_is_idempotent(self):
        mon = monitor(FakeState(), "a:5000", [])
        mon.stop()
        mon.stop()
