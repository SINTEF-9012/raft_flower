"""Structured event log for reproducing the article's measurements.

The published figures report an aggregator recovery time split into RAFT
re-election plus controlled re-assignment (Figures 6 and 8), and a traffic
breakdown (Figure 7). The original prototype emitted only `print()` output and
relied on reading `iftop` by hand, so none of those figures could be
regenerated from the repository. This module records the timestamps needed to
rebuild them.

One JSONL file per node, one JSON object per line, appended and flushed
immediately so a `SIGKILL` of the aggregator (which is how failures are
injected) cannot lose buffered events.

The recovery sequence, and how the article's quantities map onto it:

    aggregator_down_detected   t0
    raft_leader_elected        t1   RAFT re-election time      = t1 - t0
    ranking_computed           t2   controlled re-assignment   = t2 - t1
    state_restored             t3   checkpoint restore         = t3 - t2
    aggregator_serving         t4   total recovery time        = t4 - t0

Both a wall-clock and a monotonic timestamp are written on every event.
Monotonic is correct for durations within one process; wall-clock is the only
thing that can align events *across* nodes, which the recovery sequence
requires because t0 and t4 are observed on different machines.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

# Event names used by the recovery path. Defined as constants so a typo shows
# up as an ImportError rather than as a silently missing row in the results.
ROUND_STARTED = "round_started"
ROUND_COMPLETED = "round_completed"
CHECKPOINT_WRITTEN = "checkpoint_written"
CHECKPOINT_RESTORED = "state_restored"
CONTEXT_WRITTEN = "context_written"
AGGREGATOR_DOWN_DETECTED = "aggregator_down_detected"
RAFT_LEADER_ELECTED = "raft_leader_elected"
RANKING_COMPUTED = "ranking_computed"
ROLE_DECIDED = "role_decided"
#: Emitted by the supervisor when it takes the aggregator role and starts its
#: SuperLink. This is the end of the recovery interval the article reports.
AGGREGATOR_SERVING = "aggregator_serving"
#: Emitted by the ServerApp process once it has restored the checkpoint and is
#: about to drive rounds. Distinct from AGGREGATOR_SERVING because it happens
#: in a different process, and only after enough SuperNodes have reconnected --
#: so folding the two together would charge client reconnection time to the
#: recovery measurement.
SERVERAPP_READY = "serverapp_ready"
WORKER_SERVING = "worker_serving"
NODE_STARTED = "node_started"
NODE_STOPPING = "node_stopping"
CHILD_SPAWNED = "child_spawned"
CHILD_EXITED = "child_exited"
#: The configured client threshold exceeded the surviving worker count and was
#: reduced. Recorded so a run that trained with fewer clients than requested
#: says so in its results instead of changing silently.
MIN_NODES_CLAMPED = "min_nodes_clamped"


class EventLog:
    """Append-only JSONL event log, safe to share across threads."""

    def __init__(self, path: str | os.PathLike[str], node: str, *, run_id: str = "") -> None:
        self.path = Path(path)
        self.node = node
        self.run_id = run_id
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Line-buffered text append; every emit is flushed as well, because the
        # aggregator is killed with SIGKILL and gets no chance to close this.
        self._fh = self.path.open("a", buffering=1, encoding="utf-8")

    def emit(self, event: str, **fields: Any) -> dict[str, Any]:
        """Write one event. Extra keyword arguments become JSON fields."""
        record = {
            "ts": time.time(),
            "mono": time.monotonic(),
            "node": self.node,
            "event": event,
            **fields,
        }
        if self.run_id:
            record["run"] = self.run_id
        line = json.dumps(record, default=str)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()
            os.fsync(self._fh.fileno())
        return record

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()

    def __enter__(self) -> EventLog:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class _NullEventLog:
    """No-op log, so library code can always call `emit` unconditionally."""

    node = ""
    run_id = ""
    path = Path(os.devnull)

    def emit(self, event: str, **fields: Any) -> dict[str, Any]:  # noqa: ARG002
        return {}

    def close(self) -> None:
        return

    def __enter__(self) -> _NullEventLog:
        return self

    def __exit__(self, *exc: object) -> None:
        return


NULL_LOG = _NullEventLog()


def default_path(node: str, directory: str | os.PathLike[str] = "results") -> Path:
    """Per-node log path. The node address is sanitised for use as a filename."""
    safe = node.replace(":", "_").replace("/", "_")
    return Path(directory) / f"events-{safe}.jsonl"
