"""Tests for flnode.checkpoint.

Covers the two behaviours the original prototype got wrong: model
serialisation that actually round-trips, and a round counter that stays
correct across *repeated* recovery.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from flwr.app import ArrayRecord

torch_no_grad = torch.no_grad

from flnode.checkpoint import (
    KEY_CHECKPOINT,
    Checkpoint,
    completed_rounds,
    dump_arrays,
    get_aggregator,
    is_training_complete,
    load_arrays,
    load_checkpoint,
    num_rounds,
    remaining_rounds,
    save_checkpoint,
    set_aggregator,
    set_num_rounds,
)


class FakeState(dict):
    """Stand-in for pysyncobj's ReplDict (same get/set surface)."""

    def set(self, key, value, **kwargs):  # noqa: A003 - matches ReplDict
        self[key] = value

    def get(self, key, default=None):
        return super().get(key, default)


def cnn_arrays():
    """Shapes of the article's CIFAR-10 CNN (main.py's Net)."""
    shapes = [
        (6, 3, 5, 5), (6,), (16, 6, 5, 5), (16,), (120, 400),
        (120,), (84, 120), (84,), (10, 84), (10,),
    ]
    rng = np.random.default_rng(0)
    return [rng.standard_normal(s).astype(np.float32) for s in shapes]


@pytest.fixture
def state():
    return FakeState()


# ---------------------------------------------------------------------------
# Model serialisation
# ---------------------------------------------------------------------------


class TestArraySerialisation:
    def test_roundtrip_is_exact(self):
        arrays = cnn_arrays()
        record = ArrayRecord.from_numpy_ndarrays(arrays)
        back = load_arrays(dump_arrays(record)).to_numpy_ndarrays()
        assert len(back) == len(arrays)
        for original, restored in zip(arrays, back):
            assert np.array_equal(original, restored)
            assert original.dtype == restored.dtype
            assert original.shape == restored.shape

    def test_order_is_preserved_beyond_ten_arrays(self):
        """Guards the zero-padded naming: 'a10' must not sort before 'a9'."""
        arrays = [np.full((2,), i, dtype=np.float32) for i in range(25)]
        record = ArrayRecord.from_numpy_ndarrays(arrays)
        back = load_arrays(dump_arrays(record)).to_numpy_ndarrays()
        assert [int(a[0]) for a in back] == list(range(25))

    def test_blob_is_self_contained(self):
        """The whole point: unlike ArrayRecord.deflate(), the blob carries the
        data and needs no child objects to reconstruct."""
        record = ArrayRecord.from_numpy_ndarrays(cnn_arrays())
        blob = dump_arrays(record)
        # ~250 kB, as reported in Section 7.1 -- not a 1.4 kB manifest.
        assert len(blob) > 200_000
        del record  # nothing else may be needed to restore it
        assert len(load_arrays(blob).to_numpy_ndarrays()) == 10

    def test_no_pickle_required(self):
        """allow_pickle=False in load_arrays means a malicious checkpoint in
        the replicated log cannot execute code on restore."""
        record = ArrayRecord.from_numpy_ndarrays(cnn_arrays())
        blob = dump_arrays(record)
        assert b"__reduce__" not in blob

    def test_parameter_names_survive_the_round_trip(self):
        """Keys must be preserved, not renumbered.

        `from_numpy_ndarrays` keys arrays "0".."9"; `load_state_dict` matches
        on names like "conv1.weight". Losing them breaks every round *after* a
        recovery, while the first round still works because it uses a freshly
        built model rather than a checkpoint -- so the failure only appears in
        exactly the scenario this system exists for.
        """
        from flnode.model import Net

        record = ArrayRecord(Net().state_dict())
        assert "conv1.weight" in record.keys()

        restored = load_arrays(dump_arrays(record))
        assert list(restored.keys()) == list(record.keys())

    def test_restored_model_loads_into_a_fresh_net(self):
        """End-to-end: the thing the ClientApp actually does on every round."""
        from flnode.model import Net

        trained = Net()
        with torch_no_grad():
            for param in trained.parameters():
                param.add_(1.0)  # make it distinguishable from a fresh init

        restored = load_arrays(dump_arrays(ArrayRecord(trained.state_dict())))

        fresh = Net()
        fresh.load_state_dict(restored.to_torch_state_dict())  # must not raise

        for (name, expected), (_, got) in zip(
            trained.state_dict().items(), fresh.state_dict().items()
        ):
            assert (expected == got).all(), f"{name} differs after restore"

    def test_positional_record_still_restores(self):
        """Records with no meaningful names keep working."""
        record = ArrayRecord.from_numpy_ndarrays(cnn_arrays())
        restored = load_arrays(dump_arrays(record))
        assert len(restored.to_numpy_ndarrays()) == 10


# ---------------------------------------------------------------------------
# Save / restore
# ---------------------------------------------------------------------------


class TestSaveRestore:
    def test_cold_start_has_no_checkpoint(self, state):
        assert load_checkpoint(state) is None
        assert completed_rounds(state) == 0

    def test_roundtrip_through_state(self, state):
        record = ArrayRecord.from_numpy_ndarrays(cnn_arrays())
        save_checkpoint(
            state, arrays=record, server_round=3, clients=["a:5000", "b:5000"]
        )

        restored = load_checkpoint(state)
        assert restored is not None
        assert restored.server_round == 3
        assert restored.clients == ["a:5000", "b:5000"]
        assert restored.size_bytes > 200_000
        assert len(restored.to_array_record().to_numpy_ndarrays()) == 10

    def test_checkpoint_is_a_single_key(self, state):
        """One key means one RAFT log entry, so a checkpoint cannot be torn
        across a crash."""
        record = ArrayRecord.from_numpy_ndarrays(cnn_arrays())
        save_checkpoint(state, arrays=record, server_round=1)
        assert [k for k in state if k.startswith("ckpt:")] == [KEY_CHECKPOINT]

    def test_malformed_checkpoint_reads_as_none(self, state):
        for junk in [None, "nonsense", 42, {}, {"round": 1}, {"arrays": "notbytes", "round": 1}]:
            state.set(KEY_CHECKPOINT, junk)
            assert load_checkpoint(state) is None

    def test_participating_clients_are_recorded(self, state):
        """Section 5.2's 'log of participating clients', never stored before."""
        record = ArrayRecord.from_numpy_ndarrays(cnn_arrays())
        save_checkpoint(state, arrays=record, server_round=1, clients=["n1", "n2", "n3"])
        assert load_checkpoint(state).clients == ["n1", "n2", "n3"]


# ---------------------------------------------------------------------------
# Round bookkeeping -- the arithmetic the original got wrong
# ---------------------------------------------------------------------------


class TestRoundBookkeeping:
    def test_remaining_counts_down(self, state):
        set_num_rounds(state, 10)
        assert remaining_rounds(state) == 10
        record = ArrayRecord.from_numpy_ndarrays(cnn_arrays())
        save_checkpoint(state, arrays=record, server_round=4)
        assert completed_rounds(state) == 4
        assert remaining_rounds(state) == 6

    def test_counter_survives_repeated_recovery(self, state):
        """The regression that broke chained failures.

        Each new aggregator restarts its *local* round index at 1. The global
        completed-round count must keep climbing regardless, so that a cluster
        surviving three successive aggregator crashes still finishes exactly
        `num_rounds` rounds in total.
        """
        set_num_rounds(state, 9)
        record = ArrayRecord.from_numpy_ndarrays(cnn_arrays())

        global_round = 0
        for _ in range(3):  # three aggregators, three rounds each
            local_round = 0
            for _ in range(3):
                local_round += 1          # what the restarted server reports
                global_round += 1         # what must be persisted
                save_checkpoint(state, arrays=record, server_round=global_round)
            assert local_round == 3
            assert completed_rounds(state) == global_round

        assert completed_rounds(state) == 9
        assert remaining_rounds(state) == 0
        assert is_training_complete(state)

    def test_remaining_never_negative(self, state):
        set_num_rounds(state, 2)
        record = ArrayRecord.from_numpy_ndarrays(cnn_arrays())
        save_checkpoint(state, arrays=record, server_round=5)
        assert remaining_rounds(state) == 0

    def test_num_rounds_coerces_string(self, state):
        """The original wrote this straight from argparse, i.e. as a string."""
        state.set("fl:num_rounds", "7")
        assert num_rounds(state) == 7

    def test_num_rounds_survives_garbage(self, state):
        state.set("fl:num_rounds", "not-a-number")
        assert num_rounds(state, default=3) == 3

    def test_not_complete_before_bootstrap(self, state):
        assert not is_training_complete(state)


class TestAggregatorRecord:
    def test_roundtrip(self, state):
        assert get_aggregator(state) is None
        set_aggregator(state, "10.0.0.1:5000")
        assert get_aggregator(state) == "10.0.0.1:5000"

    def test_non_string_reads_as_none(self, state):
        state.set("fl:aggregator", 12345)
        assert get_aggregator(state) is None


# ---------------------------------------------------------------------------
# Against a real RAFT cluster
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_checkpoint_replicates_across_a_real_raft_cluster(tmp_path):
    """End-to-end: a checkpoint written on one node is readable on the others.

    This is the mechanism the whole article rests on, exercised against real
    pysyncobj rather than a dict.
    """
    import time

    from pysyncobj import SyncObj, SyncObjConf
    from pysyncobj.batteries import ReplDict

    ports = [5310, 5311, 5312]
    addrs = [f"localhost:{p}" for p in ports]
    objs, states = [], []
    try:
        for addr in addrs:
            st = ReplDict()
            obj = SyncObj(
                addr,
                [a for a in addrs if a != addr],
                consumers=[st],
                conf=SyncObjConf(appendEntriesUseBatch=False),
            )
            objs.append(obj)
            states.append(st)

        deadline = time.time() + 30
        while time.time() < deadline and not all(o.isReady() for o in objs):
            time.sleep(0.2)
        assert all(o.isReady() for o in objs), "cluster did not converge"

        leader_idx = next(
            i for i, o in enumerate(objs) if str(o.getStatus().get("leader")) == addrs[i]
        )
        record = ArrayRecord.from_numpy_ndarrays(cnn_arrays())
        set_num_rounds(states[leader_idx], 5)
        save_checkpoint(
            states[leader_idx], arrays=record, server_round=2, clients=["w1", "w2"]
        )

        time.sleep(3)  # let followers apply the entry

        for i, st in enumerate(states):
            restored = load_checkpoint(st)
            assert restored is not None, f"node {addrs[i]} has no checkpoint"
            assert restored.server_round == 2
            assert restored.clients == ["w1", "w2"]
            assert completed_rounds(st) == 2
            assert remaining_rounds(st) == 3
            back = restored.to_array_record().to_numpy_ndarrays()
            for original, got in zip(cnn_arrays(), back):
                assert np.array_equal(original, got)
    finally:
        for obj in objs:
            obj.destroy()
