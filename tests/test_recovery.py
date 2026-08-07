"""Recovery-path tests against a real multi-process RAFT cluster.

These exercise the decision logic that runs after an aggregator dies, without
the cost of a full Flower deployment: a node is killed, and the survivors must
(a) notice, (b) agree on a replacement, and (c) agree on the *same* one.

The full end-to-end path (training actually resuming from a replicated
checkpoint under a new aggregator) is exercised by
`experiments/run_cluster.py` + `experiments/kill_aggregator.py`; see the README.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

from flnode import raft
from flnode.context import context_key, read_context_table
from flnode.ranking import select_aggregator

REPO = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.slow


HOLDER = '''
import sys, time
sys.path.insert(0, r"{repo}")
from flnode import raft
from flnode.context import context_key

self_addr = sys.argv[1]
partners = sys.argv[2:]
latency = float(self_addr.rsplit(":", 1)[1])  # deterministic, distinct per node

sync_obj, state = raft.join_cluster(
    self_addr, partners, conf=raft.default_conf(connectionTimeout=1.0)
)
raft.wait_ready(sync_obj, timeout=60)

while True:
    state.set(
        context_key(self_addr),
        {{
            "node": self_addr,
            "ts": time.time(),
            "latency": {{}},
            "unreachable": [],
            "mean_latency": latency,
            "fl_port": 9092,
            "probe_port": int(self_addr.rsplit(":", 1)[1]) + 1000,
        }},
        sync=False,
    )
    time.sleep(0.5)
'''


@pytest.fixture
def cluster(tmp_path):
    """Three RAFT nodes publishing distinct, deterministic context records."""
    script = tmp_path / "holder.py"
    script.write_text(HOLDER.format(repo=str(REPO)), encoding="utf-8")

    ports = [5210, 5211, 5212]
    addrs = [f"localhost:{p}" for p in ports]
    procs = {}
    for addr in addrs:
        peers = [a for a in addrs if a != addr]
        procs[addr] = subprocess.Popen(
            [sys.executable, str(script), addr, *peers],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    observer, state = raft.join_cluster(
        None, addrs, conf=raft.default_conf(connectionTimeout=1.0)
    )
    try:
        assert raft.wait_ready(observer, timeout=60), "cluster did not converge"
        deadline = time.time() + 30
        while time.time() < deadline and len(read_context_table(state)) < 3:
            time.sleep(0.2)
        assert len(read_context_table(state)) == 3, "context records did not arrive"
        yield observer, state, addrs, procs
    finally:
        try:
            observer.destroy()
        except Exception:
            pass
        for proc in procs.values():
            if proc.poll() is None:
                proc.kill()
            try:
                proc.wait(timeout=10)
            except Exception:
                pass


def test_all_nodes_pick_the_same_aggregator(cluster):
    """Every node ranks the same replicated table and must agree."""
    _, state, addrs, _ = cluster
    table = read_context_table(state)
    now = time.time()
    choices = {select_aggregator(table, now=now, ttl=30.0) for _ in range(20)}
    assert len(choices) == 1
    # Lowest mean_latency wins, and latency is seeded from the port number.
    assert choices.pop() == min(addrs, key=lambda a: int(a.rsplit(":", 1)[1]))


def test_survivors_exclude_a_killed_node(cluster):
    """The dead node must drop out of the ranking, and quickly.

    Exclusion is driven by RAFT's partner state rather than by waiting a full
    context TTL, so this must hold well before 30 s have elapsed.
    """
    observer, state, addrs, procs = cluster

    victim = min(addrs, key=lambda a: int(a.rsplit(":", 1)[1]))
    assert select_aggregator(read_context_table(state), now=time.time(), ttl=30.0) == victim

    procs[victim].kill()
    procs[victim].wait(timeout=10)

    deadline = time.time() + 20
    while time.time() < deadline:
        if not raft.is_alive(observer, victim):
            break
        time.sleep(0.1)
    assert not raft.is_alive(observer, victim), "RAFT never noticed the failure"

    dead = {a for a in addrs if not raft.is_alive(observer, a)}
    survivors = [a for a in addrs if a not in dead]

    chosen = select_aggregator(
        read_context_table(state), now=time.time(), ttl=30.0, exclude=dead
    )
    assert chosen != victim
    assert chosen in survivors
    # Deterministic: the best-ranked survivor, not an arbitrary one.
    assert chosen == min(survivors, key=lambda a: int(a.rsplit(":", 1)[1]))


def test_raft_leader_is_not_necessarily_the_aggregator(cluster):
    """The two roles are independent by design (article §6.2.1)."""
    observer, state, addrs, _ = cluster
    leader = raft.wait_for_leader(observer, timeout=30)
    assert leader in addrs

    chosen = select_aggregator(read_context_table(state), now=time.time(), ttl=30.0)
    assert chosen in addrs
    # Both must be well-defined; whether they coincide is incidental.
    assert isinstance(leader, str) and isinstance(chosen, str)


def test_checkpoint_survives_the_loss_of_a_node(cluster):
    """State written before a failure is still readable afterwards."""
    import numpy as np
    from flwr.app import ArrayRecord

    from flnode.checkpoint import load_checkpoint, save_checkpoint, set_num_rounds

    observer, state, addrs, procs = cluster

    set_num_rounds(state, 5)
    record = ArrayRecord.from_numpy_ndarrays(
        [np.arange(12, dtype=np.float32).reshape(3, 4)]
    )
    save_checkpoint(state, arrays=record, server_round=3, clients=["a", "b"])

    victim = addrs[0]
    procs[victim].kill()
    procs[victim].wait(timeout=10)
    time.sleep(3)

    restored = load_checkpoint(state)
    assert restored is not None
    assert restored.server_round == 3
    assert restored.clients == ["a", "b"]
    assert np.array_equal(
        restored.to_array_record().to_numpy_ndarrays()[0],
        np.arange(12, dtype=np.float32).reshape(3, 4),
    )
