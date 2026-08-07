"""Thin wrapper over PySyncObj: cluster membership, leadership, liveness.

Everything here goes through PySyncObj's public `getStatus()` surface. The
original prototype reached into private methods instead -- `sync_obj._isLeader()`
and `sync_obj._getLeader()` (`legacy/main.py:171,196,200,202`) -- which is both
fragile across library versions and unable to answer the one question the
recovery path actually needs: *is a specific other node still alive?*

`getStatus()` answers it directly, via the per-partner connection state that
RAFT already maintains from its own heartbeats. That matters for the traffic
measurements in Section 7.2.2: aggregator liveness detection adds no packets of
its own, because it reads state the consensus layer is already keeping.

Verified against pysyncobj 0.3.15 on Python 3.13; `getStatus()` returns keys
`leader`, `has_quorum`, `state`, `raft_term`, and one
`partner_node_status_server_<addr>` per partner.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from typing import Any

from pysyncobj import SyncObj, SyncObjConf
from pysyncobj.batteries import ReplDict

#: PySyncObj connection states for a partner node.
PARTNER_DISCONNECTED = 0
PARTNER_CONNECTING = 1
PARTNER_CONNECTED = 2

_PARTNER_PREFIX = "partner_node_status_server_"


def default_conf(**overrides: Any) -> SyncObjConf:
    """Cluster configuration.

    `appendEntriesUseBatch=False` and the enlarged buffers are inherited from
    the original prototype, which sized them for ~250 kB checkpoints.

    **`connectionTimeout` dominates measured recovery time.** It is how long a
    node waits before declaring a silent partner disconnected, and therefore
    how long after a crash the survivors take to *notice*. Measured on a local
    4-node cluster, that detection lag was around 7 s with pysyncobj's default
    of 3.5 s, against ~0.3 s for RAFT re-election plus controlled
    re-assignment combined. Anyone reproducing the recovery-time figures should
    report this setting alongside them, because it, not the consensus
    mechanism, sets the floor.

    Note that no `journalFile` or `fullDumpFile` is set, so the replicated log
    lives in memory only. This matches the published behaviour and is stated
    in the SEAMS paper (Section 6.2.1): state survives as long as at least one
    node of the original cluster is still running, but not a full cluster
    restart. Pass `journalFile=...` here to change that.
    """
    params: dict[str, Any] = {
        "appendEntriesUseBatch": False,
        "appendEntriesBatchSizeBytes": 2**20,
        "logCompactionBatchSize": 2**20,
        "recvBufferSize": 2**20,
        "sendBufferSize": 2**20,
        "dynamicMembershipChange": True,
    }
    supplied = {k: v for k, v in overrides.items() if v is not None}

    # pysyncobj cross-validates these four settings:
    #     connectionTimeout >= raftMaxTimeout
    #     raftMinTimeout    >  appendEntriesPeriod * 3
    # so lowering the connection timeout on its own -- the natural thing to do
    # when tuning for faster failure detection -- fails at construction. The
    # whole window is scaled together instead, preserving pysyncobj's own
    # default ratios (0.1 / 0.4 / 1.4 against 3.5), unless the caller sets a
    # value explicitly.
    connection_timeout = supplied.get("connectionTimeout")
    if connection_timeout is not None:
        scale = connection_timeout / 3.5
        supplied.setdefault("appendEntriesPeriod", 0.1 * scale)
        supplied.setdefault("raftMinTimeout", 0.4 * scale)
        supplied.setdefault("raftMaxTimeout", 1.4 * scale)

    params.update(supplied)
    conf = SyncObjConf(**params)
    # Fail here, with the offending values in view, rather than later inside
    # SyncObj where the error surfaces as a bare AssertionError.
    try:
        conf.validate()
    except AssertionError as exc:
        raise ValueError(
            "Invalid RAFT timing configuration: "
            f"appendEntriesPeriod={conf.appendEntriesPeriod}, "
            f"raftMinTimeout={conf.raftMinTimeout}, "
            f"raftMaxTimeout={conf.raftMaxTimeout}, "
            f"connectionTimeout={conf.connectionTimeout}. "
            "Requires connectionTimeout >= raftMaxTimeout > raftMinTimeout > "
            "appendEntriesPeriod * 3."
        ) from exc
    return conf


def join_cluster(
    self_addr: str | None,
    partners: Sequence[str],
    *,
    conf: SyncObjConf | None = None,
) -> tuple[SyncObj, ReplDict]:
    """Join the RAFT cluster and return the object plus its replicated dict.

    Passing `self_addr=None` joins as a **read-only client**: the node
    replicates state and may issue writes (they are forwarded to the leader),
    but does not vote and is not counted towards quorum. The Flower ServerApp
    runs in its own process and joins this way, so that it can read and write
    checkpoints without perturbing the consensus group. Verified against
    pysyncobj 0.3.15.
    """
    state = ReplDict()
    sync_obj = SyncObj(
        self_addr,
        list(partners),
        consumers=[state],
        conf=conf or default_conf(),
    )
    return sync_obj, state


def wait_ready(sync_obj: SyncObj, timeout: float = 60.0, poll: float = 0.2) -> bool:
    """Block until this node has caught up with the cluster."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if sync_obj.isReady():
            return True
        time.sleep(poll)
    return sync_obj.isReady()


def leader_of(sync_obj: SyncObj) -> str | None:
    """Address of the current RAFT leader, or `None` during an election."""
    leader = sync_obj.getStatus().get("leader")
    if leader is None:
        return None
    text = str(leader)
    return text or None


def is_leader(sync_obj: SyncObj, self_addr: str) -> bool:
    return leader_of(sync_obj) == self_addr


def has_quorum(sync_obj: SyncObj) -> bool:
    return bool(sync_obj.getStatus().get("has_quorum"))


def wait_for_leader(
    sync_obj: SyncObj, timeout: float = 30.0, poll: float = 0.05
) -> str | None:
    """Block until RAFT has settled on a leader. Returns its address.

    This is the article's intermediate recovery phase (Section 5.3): after the
    aggregator dies, the cluster first re-elects a RAFT leader "aiming to
    quickly restore the system to a stable state rather than resume active FL
    operations", and only then ranks nodes for the aggregator role.

    Polled finely because the interval this measures is reported directly as
    the RAFT re-election time in Figure 6.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        leader = leader_of(sync_obj)
        if leader is not None and has_quorum(sync_obj):
            return leader
        time.sleep(poll)
    return leader_of(sync_obj)


def partner_states(sync_obj: SyncObj) -> dict[str, int]:
    """Connection state of every partner, keyed by address."""
    states: dict[str, int] = {}
    for key, value in sync_obj.getStatus().items():
        text = str(key)
        if text.startswith(_PARTNER_PREFIX):
            states[text[len(_PARTNER_PREFIX) :]] = int(value)
    return states


def is_alive(sync_obj: SyncObj, addr: str, self_addr: str | None = None) -> bool:
    """Is `addr` currently reachable?

    This is the check the article requires of every worker after each round:
    "worker nodes check whether the aggregator is still alive. If not, they
    transition back to the very initial phase" (Section 6.2). The original
    prototype had no such check -- it discovered failures only as an
    exception from a gRPC call, swallowed by a bare `except:`.

    A node is trivially alive to itself.
    """
    if self_addr is not None and addr == self_addr:
        return True
    return partner_states(sync_obj).get(addr, PARTNER_DISCONNECTED) == PARTNER_CONNECTED


def live_members(
    sync_obj: SyncObj, self_addr: str | None = None
) -> set[str]:
    """Every node currently reachable, including this one."""
    live = {
        addr
        for addr, state in partner_states(sync_obj).items()
        if state == PARTNER_CONNECTED
    }
    if self_addr:
        live.add(self_addr)
    return live


def wait_until_dead(
    sync_obj: SyncObj,
    addr: str,
    self_addr: str | None = None,
    *,
    poll: float = 0.05,
    stop: Iterable[Any] = (),
) -> None:
    """Block while `addr` remains reachable."""
    stops = list(stop)
    while is_alive(sync_obj, addr, self_addr):
        if any(getattr(s, "is_set", lambda: False)() for s in stops):
            return
        time.sleep(poll)
