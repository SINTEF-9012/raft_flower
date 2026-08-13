"""Replicated FL checkpoint: the state a new aggregator resumes from.

The TAAS article (Section 5.2) describes the replicated training state as "the
most recent global model parameters, the number of completed rounds and a log
of participating clients". This module stores exactly that, and nothing else.

Two deliberate differences from the original prototype:

**One key, one RAFT entry, per round.** The original replicated four separate
tensor families every round (`initialize_parameters_*`, `configure_fit_*`,
`configure_evaluate_*`, `aggregate_fit_*`), one RAFT log entry *per tensor*,
of which only `configure_fit_*` was ever read back during recovery. Beyond the
wasted traffic, splitting a checkpoint across many entries is not atomic: a
crash partway through leaves a torn checkpoint whose model parameters and round
counter disagree. Here the entire checkpoint is a single value under a single
key, so RAFT commits it in one entry -- a reader sees either the whole previous
checkpoint or the whole new one.

**The round counter is derived, not tracked separately.** The original computed
`rounds_left = num_rounds - max(0, configure_fit_round - 1)`, where
`configure_fit_round` is the *restarted* server's local round index. That index
resets to 1 on every restart, so the arithmetic is right for the first recovery
and wrong for every one after it -- meaning chained aggregator failures were not
actually supported. Here `completed_rounds` reads the round number stored inside
the last committed checkpoint, which is global and monotonic because a
checkpoint is only ever written after a round has genuinely finished.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from flwr.app import Array, ArrayRecord

from flnode.events import CHECKPOINT_WRITTEN, NULL_LOG

# --- Replicated key schema -------------------------------------------------

#: Total rounds requested for this run. Written once, at bootstrap.
KEY_NUM_ROUNDS = "fl:num_rounds"
#: The node currently holding the aggregator role. This is an *observability*
#: record written by the RAFT leader, not the source of truth: every node
#: decides its own role by running flnode.ranking locally, exactly as the
#: article requires (Section 5.4). Recovery must not depend on this key,
#: because the node that would have written it may be the one that crashed.
KEY_AGGREGATOR = "fl:aggregator"
#: The checkpoint itself: a single dict, committed as one RAFT log entry.
KEY_CHECKPOINT = "ckpt:latest"
#: Incremented every time an aggregator starts a Flower run.
#:
#: A SuperNode is bound to the SuperLink it was launched against. When the
#: aggregator restarts its run -- to apply a reduced client threshold, or after
#: taking over from a failed peer -- the old SuperLink goes away and the
#: workers' SuperNodes die with it, without the aggregator itself ever having
#: looked unhealthy. Workers therefore cannot detect this by watching
#: aggregator liveness; they watch this counter instead and relaunch.
KEY_RUN_GENERATION = "fl:run_generation"


class ReplicatedState(Protocol):
    """The subset of pysyncobj's ReplDict that this module uses."""

    def set(self, key: str, value: Any, **kwargs: Any) -> Any: ...
    def get(self, key: str, default: Any = None) -> Any: ...


# --- Model (de)serialisation ----------------------------------------------
#
# `ArrayRecord.deflate()` is deliberately *not* used here. It returns a ~1.4 kB
# manifest that references the individual arrays as separate content-addressed
# child objects; the array data is not in the returned bytes, and
# `ArrayRecord.inflate()` refuses to reconstruct without those children. That
# is the right design for Flower's own transport, but it is not a standalone
# serialisation, which is what a replicated checkpoint needs.
#
# `.npz` is used instead: one self-contained blob, no pickle (so a checkpoint
# read from the replicated log cannot execute code), dtypes and shapes
# preserved exactly. For the article's CIFAR-10 CNN this produces ~250 kB,
# matching the "approximately 250 kB per round" figure reported in Section 7.1.


#: Entry holding the ArrayRecord's keys, so they survive the round trip.
_NAMES_ENTRY = "__names__"


def dump_arrays(record: ArrayRecord) -> bytes:
    """Serialise a model to a single self-contained blob.

    The record's **keys are preserved**, not just its values. An ArrayRecord
    built from a PyTorch model is keyed by parameter name ("conv1.weight",
    ...), and `load_state_dict` matches on those names. Rebuilding with
    `ArrayRecord.from_numpy_ndarrays` alone would renumber them "0".."9" and
    every client would then reject the restored model with "Missing key(s) in
    state_dict" -- which only ever shows up *after* a recovery, since the very
    first round uses the freshly constructed model rather than a checkpoint.
    """
    names = list(record.keys())
    arrays = record.to_numpy_ndarrays()
    payload = {f"a{i:04d}": arr for i, arr in enumerate(arrays)}
    # Stored as a separate entry rather than encoded into the array keys, so
    # no escaping is needed for names containing dots or other punctuation.
    payload[_NAMES_ENTRY] = np.array(names, dtype=np.str_)
    buf = io.BytesIO()
    np.savez(buf, **payload)
    return buf.getvalue()


def load_arrays(blob: bytes) -> ArrayRecord:
    """Inverse of `dump_arrays`, keys included."""
    with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
        names = (
            [str(n) for n in archive[_NAMES_ENTRY]]
            if _NAMES_ENTRY in archive.files
            else []
        )
        # Zero-padded keys make lexicographic order equal to positional order.
        ordered = sorted(f for f in archive.files if f != _NAMES_ENTRY)
        arrays = [archive[key] for key in ordered]

    if len(names) != len(arrays):
        # No usable names recorded: fall back to positional keys.
        return ArrayRecord.from_numpy_ndarrays(arrays)
    return ArrayRecord.from_array_dict(
        {name: Array(array) for name, array in zip(names, arrays)}
    )


@dataclass(frozen=True)
class Checkpoint:
    """A committed training checkpoint."""

    #: Serialised global model (`ArrayRecord.deflate()`).
    arrays: bytes
    #: Global round number this checkpoint completed. 1-based.
    server_round: int
    #: Identifiers of the nodes that contributed to this round -- the article's
    #: "log of participating clients" (Section 5.2).
    clients: list[str] = field(default_factory=list)
    #: Wall-clock time the checkpoint was written.
    ts: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "arrays": self.arrays,
            "round": self.server_round,
            "clients": list(self.clients),
            "ts": self.ts,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> Checkpoint | None:
        if not isinstance(raw, dict):
            return None
        arrays = raw.get("arrays")
        server_round = raw.get("round")
        if not isinstance(arrays, (bytes, bytearray)):
            return None
        if not isinstance(server_round, int):
            return None
        return cls(
            arrays=bytes(arrays),
            server_round=server_round,
            clients=list(raw.get("clients") or []),
            ts=float(raw.get("ts") or 0.0),
        )

    @property
    def size_bytes(self) -> int:
        return len(self.arrays)

    def to_array_record(self) -> ArrayRecord:
        """Rebuild the global model this checkpoint carries."""
        return load_arrays(self.arrays)


# --- Write / read ----------------------------------------------------------


def save_checkpoint(
    state: ReplicatedState,
    *,
    arrays: ArrayRecord | bytes,
    server_round: int,
    clients: list[str] | None = None,
    log=NULL_LOG,
) -> Checkpoint:
    """Replicate a checkpoint to every node. Blocks until RAFT commits it.

    `sync=True` is intentional and is the point of the whole exercise: the
    round is not considered durable until a majority of nodes have the
    checkpoint. Contextual metrics use `sync=False` because losing one
    measurement sample is harmless, but losing a round of training is not.
    """
    blob = arrays if isinstance(arrays, (bytes, bytearray)) else dump_arrays(arrays)
    checkpoint = Checkpoint(
        arrays=bytes(blob),
        server_round=server_round,
        clients=list(clients or []),
        ts=time.time(),
    )
    started = time.monotonic()
    state.set(KEY_CHECKPOINT, checkpoint.to_dict(), sync=True)
    elapsed = time.monotonic() - started

    log.emit(
        CHECKPOINT_WRITTEN,
        round=server_round,
        bytes=checkpoint.size_bytes,
        clients=len(checkpoint.clients),
        duration_s=elapsed,
    )
    return checkpoint


def load_checkpoint(state: ReplicatedState) -> Checkpoint | None:
    """Read the latest committed checkpoint, or `None` on a cold start."""
    return Checkpoint.from_dict(state.get(KEY_CHECKPOINT, None))


# --- Round bookkeeping -----------------------------------------------------


def set_num_rounds(state: ReplicatedState, rounds: int) -> None:
    state.set(KEY_NUM_ROUNDS, int(rounds), sync=True)


def num_rounds(state: ReplicatedState, default: int = 0) -> int:
    """Total rounds requested for the run.

    Coerced to `int` because the original prototype wrote this straight from
    `argparse`, i.e. as a string, and then compared it against integers
    elsewhere.
    """
    value = state.get(KEY_NUM_ROUNDS, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def completed_rounds(state: ReplicatedState) -> int:
    """Rounds finished so far, globally, across every aggregator that has held
    the role. Zero before the first checkpoint."""
    checkpoint = load_checkpoint(state)
    return checkpoint.server_round if checkpoint else 0


def remaining_rounds(state: ReplicatedState) -> int:
    """Rounds still to run. Never negative."""
    return max(0, num_rounds(state) - completed_rounds(state))


def is_training_complete(state: ReplicatedState) -> bool:
    return num_rounds(state) > 0 and remaining_rounds(state) == 0


# --- Aggregator record (observability only) --------------------------------


def set_aggregator(state: ReplicatedState, node: str, *, sync: bool = True) -> bool:
    """Record who holds the aggregator role. Returns whether the write landed.

    Best-effort by design. This key is a trace, not the source of truth --
    nothing reads it to make a role decision (see the note on KEY_AGGREGATOR).

    The write is attempted at exactly the moment it is most likely to fail:
    just after a failure, while RAFT is still settling. pysyncobj then raises
    `SyncObjException(LEADER_CHANGED)`, which is entirely routine. Letting that
    propagate would kill the node that was trying to take over -- turning one
    aggregator failure into two, and in a small cluster into loss of quorum.
    So it is caught here and reported as a return value.
    """
    try:
        state.set(KEY_AGGREGATOR, node, sync=sync)
        return True
    except Exception:
        return False


def get_aggregator(state: ReplicatedState) -> str | None:
    value = state.get(KEY_AGGREGATOR, None)
    return value if isinstance(value, str) else None


# --- Run generation --------------------------------------------------------


def run_generation(state: ReplicatedState) -> int:
    value = state.get(KEY_RUN_GENERATION, 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def bump_run_generation(state: ReplicatedState) -> int:
    """Announce a new Flower run. Returns the new generation.

    Best-effort like `set_aggregator`: this is written while the cluster may
    still be settling after a failure, and a refused write must not take the
    incoming aggregator down. A missed bump only costs the workers a delayed
    relaunch, which their child-exit watchdog also covers.
    """
    generation = run_generation(state) + 1
    try:
        state.set(KEY_RUN_GENERATION, generation, sync=True)
    except Exception:
        return run_generation(state)
    return generation
