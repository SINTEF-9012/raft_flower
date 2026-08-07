"""FedAvg with a RAFT-replicated checkpoint after every completed round.

This is the "state replication" building block (TAAS Section 5.2). The
aggregation itself is untouched -- Section 7.2.1 is explicit that the recovery
mechanism "does not affect the internal ML training logic", so this subclasses
Flower's `FedAvg` and overrides exactly one method.

The original prototype instead vendored two ~400-line copies of the whole
`FedAvg` class in order to sprinkle replication calls through four of its
methods. Those copies are in `legacy/` and are replaced by the ~30 lines below.

**Why the checkpoint is written here and nowhere else.** `aggregate_train` is
the single point at which a round has genuinely finished: clients have replied
and their updates have been folded into a new global model. Replicating in
`configure_train` (as the original did) persists a model the round has not
produced yet, and makes the recorded round number one ahead of reality --
which is exactly why the original's recovery arithmetic needed its
`max(0, configure_fit_round - 1)` correction, and why that correction broke on
the second recovery.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any

from flwr.app import ArrayRecord, Message, MetricRecord
from flwr.serverapp.strategy import FedAvg

from flnode.checkpoint import save_checkpoint
from flnode.events import ROUND_COMPLETED, ROUND_STARTED, NULL_LOG


class RaftFedAvg(FedAvg):
    """FedAvg that replicates a checkpoint through RAFT each round.

    Parameters
    ----------
    state:
        The replicated dictionary (pysyncobj `ReplDict`) to checkpoint into.
    completed_rounds:
        Rounds already finished by *previous* aggregators. Flower restarts its
        own `server_round` counter at 1 whenever a run starts, so this offset
        is what turns a local round index into the global one that recovery
        depends on.
    log:
        Event log used to record round and checkpoint timings.
    """

    def __init__(
        self,
        *args: Any,
        state: Any,
        completed_rounds: int = 0,
        log=NULL_LOG,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.state = state
        self.completed_rounds = completed_rounds
        self.log = log
        self._round_started_at: dict[int, float] = {}

    def global_round(self, server_round: int) -> int:
        """Translate this run's local round number into the global one."""
        return self.completed_rounds + server_round

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: Any, grid: Any
    ) -> Iterable[Message]:
        self._round_started_at[server_round] = time.monotonic()
        self.log.emit(
            ROUND_STARTED,
            round=self.global_round(server_round),
            local_round=server_round,
        )
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(
        self, server_round: int, replies: Iterable[Message]
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        # Materialise first: `replies` is a one-shot iterable and the base
        # implementation consumes it, so the participant list has to be taken
        # before delegating.
        materialised = list(replies)

        arrays, metrics = super().aggregate_train(server_round, materialised)

        if arrays is not None:
            global_round = self.global_round(server_round)
            # The article's "log of participating clients" (Section 5.2).
            clients = sorted(
                str(msg.metadata.src_node_id)
                for msg in materialised
                if msg.has_content()
            )
            save_checkpoint(
                self.state,
                arrays=arrays,
                server_round=global_round,
                clients=clients,
                log=self.log,
            )
            started = self._round_started_at.pop(server_round, None)
            self.log.emit(
                ROUND_COMPLETED,
                round=global_round,
                local_round=server_round,
                clients=len(clients),
                duration_s=None if started is None else time.monotonic() - started,
            )

        return arrays, metrics
