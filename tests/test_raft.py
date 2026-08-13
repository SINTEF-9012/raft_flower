"""Tests for leadership and liveness reads.

The stale-leader case is the one that matters: pysyncobj keeps naming a dead
leader for a short window after it fails, and accepting that answer both names
the wrong node and makes the measured RAFT re-election interval shorter than it
really was.
"""

from __future__ import annotations

import time

import pytest

from flnode.raft import (
    PARTNER_CONNECTED,
    PARTNER_DISCONNECTED,
    default_conf,
    has_quorum,
    is_alive,
    leader_of,
    live_members,
    partner_states,
    wait_for_leader,
)


class FakeSyncObj:
    """Minimal stand-in exposing only getStatus(), which is all we read.

    `script` is a list of status dicts returned one per call, the last
    repeating -- enough to model a cluster mid-election.
    """

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def getStatus(self):  # noqa: N802 - matches pysyncobj
        self.calls += 1
        index = min(self.calls - 1, len(self.script) - 1)
        return self.script[index]


def status(leader, partners, quorum=True):
    out = {"leader": leader, "has_quorum": quorum}
    for addr, state in partners.items():
        out[f"partner_node_status_server_{addr}"] = state
    return out


A, B, C = "n1:5000", "n2:5000", "n3:5000"


class TestReads:
    def test_leader_and_quorum(self):
        obj = FakeSyncObj([status(A, {B: PARTNER_CONNECTED})])
        assert leader_of(obj) == A
        assert has_quorum(obj)

    def test_no_leader_during_election(self):
        obj = FakeSyncObj([status(None, {B: PARTNER_CONNECTED})])
        assert leader_of(obj) is None

    def test_empty_leader_string_is_none(self):
        obj = FakeSyncObj([status("", {B: PARTNER_CONNECTED})])
        assert leader_of(obj) is None

    def test_partner_states_parsed(self):
        obj = FakeSyncObj([
            status(A, {B: PARTNER_CONNECTED, C: PARTNER_DISCONNECTED})
        ])
        assert partner_states(obj) == {B: PARTNER_CONNECTED, C: PARTNER_DISCONNECTED}

    def test_is_alive(self):
        obj = FakeSyncObj([
            status(A, {B: PARTNER_CONNECTED, C: PARTNER_DISCONNECTED})
        ])
        assert is_alive(obj, B)
        assert not is_alive(obj, C)

    def test_a_node_is_alive_to_itself(self):
        obj = FakeSyncObj([status(A, {B: PARTNER_CONNECTED})])
        assert is_alive(obj, A, self_addr=A)

    def test_unknown_partner_is_not_alive(self):
        obj = FakeSyncObj([status(A, {B: PARTNER_CONNECTED})])
        assert not is_alive(obj, "ghost:5000")

    def test_live_members_includes_self(self):
        obj = FakeSyncObj([
            status(A, {B: PARTNER_CONNECTED, C: PARTNER_DISCONNECTED})
        ])
        assert live_members(obj, self_addr=A) == {A, B}


class TestWaitForLeader:
    def test_rejects_a_leader_that_is_no_longer_live(self):
        """The observed failure: pysyncobj still names the node that just died.

        A is the dead former leader; only B and C remain. Waiting must not
        return A, and must return B once B is reported as leader.
        """
        script = [
            # A is gone but still reported as leader, and quorum holds among B/C.
            status(A, {A: PARTNER_DISCONNECTED, C: PARTNER_CONNECTED}),
            status(A, {A: PARTNER_DISCONNECTED, C: PARTNER_CONNECTED}),
            # Election completes; B (this node) takes over.
            status(B, {A: PARTNER_DISCONNECTED, C: PARTNER_CONNECTED}),
        ]
        obj = FakeSyncObj(script)
        assert wait_for_leader(obj, timeout=5, poll=0.01, self_addr=B) == B

    def test_returns_a_live_peer_leader(self):
        obj = FakeSyncObj([
            status(C, {A: PARTNER_DISCONNECTED, C: PARTNER_CONNECTED})
        ])
        assert wait_for_leader(obj, timeout=5, poll=0.01, self_addr=B) == C

    def test_waits_through_an_election(self):
        script = [
            status(None, {C: PARTNER_CONNECTED}),
            status(None, {C: PARTNER_CONNECTED}),
            status(C, {C: PARTNER_CONNECTED}),
        ]
        obj = FakeSyncObj(script)
        assert wait_for_leader(obj, timeout=5, poll=0.01, self_addr=B) == C

    def test_does_not_return_without_quorum(self):
        obj = FakeSyncObj([
            status(C, {C: PARTNER_CONNECTED}, quorum=False)
        ])
        started = time.monotonic()
        wait_for_leader(obj, timeout=0.3, poll=0.01, self_addr=B)
        assert time.monotonic() - started >= 0.3  # waited out the timeout

    def test_timeout_returns_last_known_leader(self):
        """Best-effort on timeout, so the caller can still log something."""
        obj = FakeSyncObj([status(A, {A: PARTNER_DISCONNECTED})])
        assert wait_for_leader(obj, timeout=0.2, poll=0.01, self_addr=B) == A


class TestConfig:
    def test_defaults_validate(self):
        default_conf().validate()

    def test_lowering_connection_timeout_scales_the_window(self):
        """pysyncobj cross-validates these; lowering one alone is invalid."""
        conf = default_conf(connectionTimeout=1.0)
        conf.validate()
        assert conf.connectionTimeout == 1.0
        assert conf.raftMaxTimeout <= conf.connectionTimeout
        assert conf.raftMinTimeout > conf.appendEntriesPeriod * 3

    def test_explicit_values_are_respected(self):
        conf = default_conf(connectionTimeout=5.0, raftMaxTimeout=2.0)
        assert conf.raftMaxTimeout == 2.0

    def test_impossible_combination_is_rejected_clearly(self):
        with pytest.raises(ValueError, match="RAFT timing"):
            default_conf(connectionTimeout=1.0, raftMaxTimeout=99.0)

    def test_none_overrides_are_ignored(self):
        conf = default_conf(connectionTimeout=None)
        assert conf.connectionTimeout == 3.5
