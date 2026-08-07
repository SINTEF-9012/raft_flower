"""Continuous contextual monitoring, replicated across the cluster.

This is the ground layer of the TAAS architecture (Section 5.1): "each node
continuously performs contextual monitoring, i.e., collecting various runtime
metrics about its own resources and environmental conditions ... which are then
replicated across all nodes to ensure a consistent state of knowledge".

Each node publishes exactly one record, under `ctx:<its own RAFT address>`, and
never writes another node's record. Combined with the per-peer latency table
kept inside each record, the cluster-wide matrix of Table 2 can be
reconstructed by reading every `ctx:*` key.

Three differences from the original prototype, each of which mattered:

**Unreachable peers are excluded, not scored as zero.** `tcppinglib` returns
`avg_rtt == 0.0` with `is_alive == False` for a host it cannot reach. The
original appended that value straight into the latency list, so the moment a
node died, its neighbours' mean latency fell *towards zero* -- making the nodes
closest to the failure look like the best aggregator candidates. Unreachable
peers are recorded separately and left out of the mean.

**Probes target the FL port, not the RAFT port.** The original pinged port 5000,
the same port RAFT uses. That inflates the port-5000 byte count that Section
7.2.2 attributes to consensus traffic, and makes the traffic breakdown in
Figure 7 impossible to reproduce cleanly. Probing the Flower Fleet API port
keeps the two measurements separate, and also measures the path that FL traffic
will actually take.

**One key per node, not one per pair.** N keys instead of N^2, refreshed as a
single value so a reader never sees half of one node's measurements.

Records are written with `sync=False`: losing one sample of soft, continuously
refreshed telemetry is harmless, and blocking the monitor thread on a RAFT
majority every period would be a needless cost. Checkpoints, by contrast, use
`sync=True`.
"""

from __future__ import annotations

import math
import os
import socket
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import objsize
from tcppinglib import tcpping

from flnode.events import CONTEXT_WRITTEN, NULL_LOG
from flnode.ranking import mean_of

#: Prefix for per-node context records in the replicated dictionary.
CTX_PREFIX = "ctx:"

#: Offset from a node's RAFT port to its latency probe port.
#:
#: Latency probes need a target that is listening for the whole life of the
#: node. The Flower Fleet port is not: it exists only while that node happens
#: to be the aggregator, so probing it deadlocks the cluster at startup --
#: every node scores infinity, nobody can be ranked, so no aggregator starts
#: and no Fleet port ever opens. The RAFT port is always up, but probing it
#: would fold probe traffic into the port-5000 measurement that Section 7.2.2
#: attributes to consensus.
#:
#: A dedicated port solves both, and improves on the article's setup: probe
#: traffic becomes separately measurable instead of being conflated with either
#: FL or RAFT traffic. Being a fixed offset, any node can compute any peer's
#: probe port from the RAFT address it already knows.
PROBE_PORT_OFFSET = 1000


def context_key(node: str) -> str:
    return f"{CTX_PREFIX}{node}"


def probe_port_of(raft_addr: str, offset: int = PROBE_PORT_OFFSET) -> int:
    """Latency probe port for a node, derived from its RAFT address."""
    return int(raft_addr.rsplit(":", 1)[1]) + offset


class ProbeListener(threading.Thread):
    """Accept-and-close TCP listener so peers can time a round trip to us.

    Deliberately trivial: the measurement is the TCP handshake, so the
    connection is closed immediately and nothing is ever read or written.
    """

    def __init__(self, port: int, host: str = "0.0.0.0", backlog: int = 128) -> None:
        super().__init__(name=f"probe-listener:{port}", daemon=True)
        self.port = port
        self.host = host
        self.backlog = backlog
        self._stop = threading.Event()
        self._sock: socket.socket | None = None

    def start(self) -> None:  # type: ignore[override]
        # Bind before the thread starts so a port clash surfaces immediately
        # rather than silently leaving the node unprobeable (and so unrankable).
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.listen(self.backlog)
        sock.settimeout(0.5)
        self._sock = sock
        super().start()

    def run(self) -> None:
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
                conn.close()
            except TimeoutError:
                continue
            except OSError:
                if self._stop.is_set():
                    return
                continue

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        if self.is_alive():
            self.join(timeout=timeout)


def read_context_table(state: Any) -> dict[str, dict]:
    """Collect every node's context record from the replicated dictionary.

    Returns a plain mapping of node address -> record, which is exactly what
    `flnode.ranking.rank_nodes` consumes.
    """
    table: dict[str, dict] = {}
    for key in list(state.keys()):
        if not isinstance(key, str) or not key.startswith(CTX_PREFIX):
            continue
        record = state.get(key, None)
        if isinstance(record, dict):
            table[key[len(CTX_PREFIX) :]] = record
    return table


@dataclass
class ProbeConfig:
    """Latency probe tuning.

    Defaults are sized for the article's LAN testbed (2-5 ms between nodes) and
    to keep one full sweep comfortably shorter than `period`.
    """

    period: float = 10.0
    timeout: float = 2.0
    count: int = 3
    interval: float = 0.2


#: A provider adds extra fields to the published record. This is the hook for
#: the additional contextual properties the article names -- battery level,
#: reputation, computational capacity (Sections 5.1 and 8.3) -- which the
#: multi-criteria `weighted_policy` can then rank on.
Provider = Callable[[], Mapping[str, Any]]


def cpu_provider() -> Mapping[str, Any]:
    """Normalised 1-minute load average, where available.

    `os.getloadavg` does not exist on Windows, so this degrades to an empty
    contribution rather than failing. Absent fields simply do not participate
    in ranking.
    """
    try:
        load1 = os.getloadavg()[0]
    except (OSError, AttributeError):
        return {}
    cpus = os.cpu_count() or 1
    return {"cpu_load": load1 / cpus}


class ContextMonitor(threading.Thread):
    """Background thread publishing this node's context record periodically."""

    def __init__(
        self,
        state: Any,
        node: str,
        peers: Iterable[str],
        *,
        fl_port: int,
        probe_offset: int = PROBE_PORT_OFFSET,
        config: ProbeConfig | None = None,
        providers: Iterable[Provider] = (cpu_provider,),
        log=NULL_LOG,
    ) -> None:
        super().__init__(name="context-monitor", daemon=True)
        self.state = state
        self.node = node
        self.peers = list(peers)
        self.fl_port = fl_port
        self.probe_offset = probe_offset
        self.config = config or ProbeConfig()
        self.providers = list(providers)
        self.log = log
        self._stop = threading.Event()
        self._started_at = time.time()

    # -- measurement --------------------------------------------------------

    def peer_probe_port(self, peer: str) -> int:
        """Which port to probe on `peer`.

        Prefers the port the peer advertises in its own record; falls back to
        the fixed offset from its RAFT address, which is what makes the very
        first measurement possible before any record has been published.
        """
        record = self.state.get(context_key(peer), None)
        if isinstance(record, dict):
            port = record.get("probe_port")
            if isinstance(port, int) and 0 < port < 65536:
                return port
        return probe_port_of(peer, self.probe_offset)

    def probe(self, peer: str) -> float | None:
        """Round-trip latency to one peer in ms, or `None` if unreachable.

        `peer` is a RAFT address (`host:5000`); only the host part is used, and
        the probe targets that node's dedicated probe port.
        """
        host = peer.rsplit(":", 1)[0]
        try:
            result = tcpping(
                host,
                port=self.peer_probe_port(peer),
                timeout=self.config.timeout,
                count=self.config.count,
                interval=self.config.interval,
            )
        except Exception:
            # Probing must never take the monitor thread down: an unreachable
            # peer is normal operation here, not an error.
            return None

        # A closed or filtered port yields avg_rtt == 0.0 with is_alive False.
        # Treating that as zero latency would rank a dead node as the best
        # possible aggregator, so it is reported as unreachable instead.
        if not getattr(result, "is_alive", False):
            return None
        rtt = getattr(result, "avg_rtt", None)
        if not isinstance(rtt, (int, float)) or not math.isfinite(rtt) or rtt <= 0.0:
            return None
        return float(rtt)

    def sample(self) -> dict[str, Any]:
        """Take one full measurement pass and build the record to publish.

        Separated from `run` so it can be tested without a thread or a cluster.
        """
        latency: dict[str, float] = {}
        unreachable: list[str] = []
        for peer in self.peers:
            rtt = self.probe(peer)
            if rtt is None:
                unreachable.append(peer)
            else:
                latency[peer] = rtt

        # Mean over reachable peers only. With no reachable peers this is
        # `inf`, which `rank_nodes` treats as unrankable -- correct, since a
        # node that can reach nobody must not be made the aggregator.
        mean_latency = mean_of(latency[p] for p in sorted(latency))

        record: dict[str, Any] = {
            "node": self.node,
            "ts": time.time(),
            "latency": latency,
            "unreachable": unreachable,
            "mean_latency": mean_latency,
            "uptime": time.time() - self._started_at,
            # Advertised so a worker knows which port to point its SuperNode at
            # once this node is ranked into the aggregator role, and so peers
            # know where to send latency probes.
            "fl_port": self.fl_port,
            "probe_port": probe_port_of(self.node, self.probe_offset),
        }
        for provider in self.providers:
            try:
                record.update(provider())
            except Exception:
                continue
        return record

    def publish(self, record: Mapping[str, Any]) -> int:
        """Replicate the record. Returns its size in bytes."""
        self.state.set(context_key(self.node), dict(record), sync=False)
        size = objsize.get_deep_size(record)
        self.log.emit(
            CONTEXT_WRITTEN,
            bytes=size,
            peers=len(record.get("latency") or {}),
            unreachable=len(record.get("unreachable") or []),
            mean_latency=record.get("mean_latency"),
        )
        return size

    # -- thread lifecycle ---------------------------------------------------

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                self.publish(self.sample())
            except Exception:
                # Never let a transient failure kill contextual monitoring;
                # a node that stops publishing is treated as dead by every
                # other node's staleness filter.
                pass
            self._stop.wait(self.config.period)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self.is_alive():
            self.join(timeout=timeout)
