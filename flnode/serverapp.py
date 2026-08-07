"""Flower ServerApp: the FL aggregator.

Launched by the supervisor (`flnode.node`) on whichever node currently holds
the aggregator role, via `flwr run`. It runs in its own process, so it cannot
share the supervisor's replicated dictionary object; instead it joins the RAFT
cluster as a **read-only client** (see `flnode.raft.join_cluster`), which lets
it read the last checkpoint and write new ones without becoming a voting member
or affecting quorum.

The recovery path lives here, in `restore`: whichever node takes the aggregator
role reads the latest replicated checkpoint and resumes from it, rather than
starting over. That is the difference between Level 3 and Level 5 of the
article's taxonomy (Table 1).

Configuration arrives through `context.run_config`, which the supervisor sets
per run with `flwr run --run-config`, because the number of rounds still to do
depends on the replicated checkpoint and is therefore only known at launch.
"""

from __future__ import annotations

import time
from typing import Any

from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp

from flnode import raft
from flnode.checkpoint import (
    Checkpoint,
    completed_rounds,
    load_checkpoint,
    num_rounds,
    remaining_rounds,
)
from flnode.events import (
    CHECKPOINT_RESTORED,
    EventLog,
    NULL_LOG,
    SERVERAPP_READY,
    default_path,
)
from flnode.model import Net

app = ServerApp()


def _split(value: Any) -> list[str]:
    """Parse a comma-separated run-config value."""
    if not value:
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def restore(state: Any, log=NULL_LOG) -> tuple[ArrayRecord, int]:
    """Recover the global model and the round count from the replicated log.

    Returns `(initial_arrays, completed_rounds)`. On a cold start the
    checkpoint is absent and a freshly initialised model is returned with a
    count of zero -- the article's "For the initial launch, this RAFT
    replicated log will be empty" (Section 6.2).
    """
    started = time.monotonic()
    checkpoint: Checkpoint | None = load_checkpoint(state)

    if checkpoint is None:
        log.emit(CHECKPOINT_RESTORED, found=False, round=0)
        return ArrayRecord(Net().state_dict()), 0

    arrays = checkpoint.to_array_record()
    log.emit(
        CHECKPOINT_RESTORED,
        found=True,
        round=checkpoint.server_round,
        bytes=checkpoint.size_bytes,
        clients=len(checkpoint.clients),
        duration_s=time.monotonic() - started,
    )
    return arrays, checkpoint.server_round


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = context.run_config

    partners = _split(cfg.get("raft-partners"))
    self_addr = str(cfg.get("raft-self", "") or "")
    results_dir = str(cfg.get("results-dir", "results"))
    local_epochs = int(cfg.get("local-epochs", 1))
    fraction_train = float(cfg.get("fraction-train", 1.0))
    min_nodes = int(cfg.get("min-nodes", 1))

    log = (
        EventLog(default_path(f"{self_addr}-serverapp", results_dir), self_addr)
        if self_addr
        else NULL_LOG
    )

    # Join read-only: replicate and write, but do not vote.
    sync_obj, state = raft.join_cluster(None, partners)
    try:
        if not raft.wait_ready(sync_obj, timeout=60.0):
            raise RuntimeError(
                f"ServerApp could not join the RAFT cluster at {partners}"
            )

        initial_arrays, done = restore(state, log)

        total = num_rounds(state) or int(cfg.get("num-server-rounds", 3))
        rounds_to_run = remaining_rounds(state) if num_rounds(state) else total
        if rounds_to_run <= 0:
            log.emit(SERVERAPP_READY, rounds=0, note="training already complete")
            return

        # Import here so the module can be introspected without the strategy's
        # heavier dependencies present.
        from flnode.strategy import RaftFedAvg

        strategy = RaftFedAvg(
            state=state,
            completed_rounds=done,
            log=log,
            fraction_train=fraction_train,
            fraction_evaluate=fraction_train,
            min_train_nodes=min_nodes,
            min_evaluate_nodes=min_nodes,
            min_available_nodes=min_nodes,
        )

        log.emit(
            SERVERAPP_READY,
            rounds=rounds_to_run,
            resumed_from_round=done,
            total_rounds=total,
        )

        strategy.start(
            grid=grid,
            initial_arrays=initial_arrays,
            num_rounds=rounds_to_run,
            train_config=ConfigRecord({"local-epochs": local_epochs}),
            evaluate_config=ConfigRecord({}),
        )
    finally:
        # Leave the cluster cleanly so the supervisor's liveness view stays
        # accurate; a lingering read-only member is harmless but noisy.
        try:
            sync_obj.destroy()
        except Exception:
            pass
        log.close() if log is not NULL_LOG else None
