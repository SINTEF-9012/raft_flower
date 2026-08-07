# Controlled Self-Recovery of the FL Aggregator using RAFT

Reference implementation for:

> Rustem Dautov and Erik Johannes Husom. *Controlled Self-Recovery of the
> Aggregator in Federated Learning Using RAFT Protocol.* ACM Transactions on
> Autonomous and Adaptive Systems, 21(3), Article 26, June 2026.
> <https://doi.org/10.1145/3785470>

The central aggregator is a single point of failure in federated learning. This
implementation removes it: every node continuously replicates both the global
training state and its own contextual properties through RAFT, and when the
aggregator dies the survivors independently rank each other and the most
suitable one resumes training from the last replicated checkpoint. That is
**Level 5** of the article's taxonomy (Table 1) — stateful self-recovery with a
*controlled*, context-ranked aggregator re-assignment, as opposed to a random
one.

Built on **Flower 1.33** and **PySyncObj 0.3.15**.

> The earlier prototypes described in the SEAMS'24 and IFIP Networking'24
> papers are preserved unchanged in [`legacy/`](legacy/), together with a note
> on their limitations.

---

## Architecture

Every node runs the same program. Its role follows from a deterministic
ranking, not from configuration.

```
flnode supervisor                    always running, never restarted
├── PySyncObj member          :5000  RAFT election, heartbeats, replicated state
├── ProbeListener             :6000  always-on target for peers' latency probes
├── ContextMonitor thread            publishes ctx:<self> every period
├── EventLog                         JSONL measurement trace
└── exactly one child group, by role:
      AGGREGATOR → flower-superlink (Fleet :9092) + flwr run → ServerApp
      WORKER     → flower-supernode --superlink <aggregator>:9092
```

The supervisor holds RAFT membership and contextual monitoring for the life of
the node; only the child group changes when the role moves. That is what lets a
node switch roles without leaving the consensus cluster, and it is why
aggregator liveness can be read straight off RAFT's own heartbeats instead of
being probed separately.

**The SuperLink moves with the aggregator.** Pinning it to one host would
reintroduce exactly the single point of failure this work exists to remove. A
SuperNode's `--superlink` address is fixed at process start, so handover
restarts the SuperNode child against the new aggregator. The aggregator submits
`flwr run` to the SuperLink on its *own* machine, so that address is loopback
and never changes.

### Recovery, in two phases

Kept distinct exactly as the article specifies:

| Phase | What happens | Article |
|---|---|---|
| Detection | Workers see the aggregator drop out of RAFT's partner set | §6.2 |
| (ii) RAFT re-election | The cluster restores a consistent replicated state | §5.3 |
| (iii) Controlled re-assignment | Every node independently ranks peers and decides its own role | §5.4 |
| Resume | The new aggregator restores the checkpoint and continues | §5.2 |

The RAFT leader and the FL aggregator are **separate roles** and frequently end
up on different nodes (§6.2.1). A run from this implementation:

```
  [1] localhost:5000 -> localhost:5001
      detected by         3 node(s)
      RAFT re-election    0.299 s
      re-assignment       0.002 s
      TOTAL recovery      5.318 s
      resumed from round  1
      RAFT leader         localhost:5003  (leader is aggregator: False)
      unanimous ranking   True ['localhost:5001']
```

---

## Quick start

Requires **Python 3.11+** (Flower 1.33's floor — note this rules out the
Raspberry Pi OS *Bullseye* images used for the article's testbed, which ship
Python 3.9; use *Bookworm* or newer).

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e .
python experiments/setup_superlink.py            # once per machine
```

Run a 4-node cluster locally and kill its aggregator mid-training:

```bash
python experiments/run_cluster.py --nodes 4 --rounds 4 --min-nodes 2
```

```bash
python experiments/kill_aggregator.py --after-round 1
```

```bash
python experiments/collect.py --results-dir results --csv recovery.csv
```

Training continues across the failure, resuming from the replicated
checkpoint — rounds 2–4 are completed by a different node than round 1.

### One node per host

```bash
python -m flnode.node --self 10.0.0.1:5000 \
    --peers 10.0.0.2:5000 10.0.0.3:5000 10.0.0.4:5000 --rounds 10
```

Pass `--rounds` on exactly one node to seed the run; an already-replicated
value always wins, so a restarting node never resets the budget.

---

## How the code maps to the paper

| Module | Role | Article |
|---|---|---|
| [`flnode/node.py`](flnode/node.py) | Supervisor, role state machine, recovery | Figure 2 |
| [`flnode/ranking.py`](flnode/ranking.py) | Deterministic context-based ranking | §5.4 |
| [`flnode/context.py`](flnode/context.py) | Continuous contextual monitoring | §5.1 |
| [`flnode/checkpoint.py`](flnode/checkpoint.py) | Replicated training state | §5.2, §4.2 |
| [`flnode/raft.py`](flnode/raft.py) | Consensus, leadership, liveness | §4.1, §5.3 |
| [`flnode/strategy.py`](flnode/strategy.py) | FedAvg + per-round checkpoint | §5.2 |
| [`flnode/serverapp.py`](flnode/serverapp.py) | Aggregator; restores on takeover | §6.2 |
| [`flnode/events.py`](flnode/events.py) | Measurement trace | §7.2 |

### Replicated keys

| Key | Contents |
|---|---|
| `fl:num_rounds` | Total rounds for the run |
| `fl:aggregator` | Current aggregator — **a trace, not the source of truth** |
| `ckpt:latest` | `{arrays, round, clients, ts}` — one key, so one atomic RAFT entry |
| `ctx:<node>` | Per-node contextual record (latency table, mean, uptime, ports) |

No node ever learns its role by reading `fl:aggregator`. Each computes it from
`ctx:*` with the same pure function, which is what §5.4 requires — and what
makes recovery possible when the node that would have written that key is the
one that crashed.

### What makes the ranking deterministic

`rank_nodes` is pure and independently testable, because the article's
correctness claim rests entirely on every node reaching the same answer:

- **Staleness filtering** — a crashed node's record is never removed by
  anything, so without a TTL it stays a candidate and can be "elected" to
  replace itself. Nodes RAFT already reports as gone are excluded immediately,
  so detection speed rather than the TTL bounds recovery.
- **Explicit tie-breaking** on `(score, node)` — sorting floats alone is only
  deterministic when there are no ties.
- **Order-independent arithmetic** via `math.fsum` — two nodes may iterate the
  replicated dictionary in different orders, and floating-point addition is not
  associative; a last-bit difference is enough to flip a ranking and let two
  nodes both claim the role.
- **Unreachable peers are excluded, not scored zero** — `tcppinglib` returns
  `avg_rtt == 0.0` for a host it cannot reach, so counting it would make the
  nodes nearest a failure look like the *best* aggregator candidates.

`tests/test_ranking.py` enforces all four, including replaying the same table
in 100 shuffled orders and asserting bit-identical results.

---

## Reproducing the measurements

`collect.py` rebuilds the recovery breakdown of Figures 6 and 8 from the event
logs:

```
t0  aggregator_down_detected
t1  raft_leader_elected      RAFT re-election    = t1 - t0
t2  ranking_computed         re-assignment       = t2 - t1
t3  aggregator_serving       TOTAL recovery      = t3 - t0
    state_restored           checkpoint restore
    serverapp_ready          serving after
```

t0 is observed on the surviving workers and t3 on whichever node takes over, so
the total spans nodes and uses wall-clock timestamps — keep clocks NTP-synced
on a real testbed.

For the traffic breakdown of Figure 7, the ports are cleanly separated:

```bash
sudo iftop -i eth0 -f "tcp port 9092"   # FL (Fleet API) — the only inter-node FL traffic
```

```bash
sudo iftop -i eth0 -f "tcp port 5000"   # RAFT: elections, heartbeats, state replication
```

```bash
sudo iftop -i eth0 -f "tcp port 6000"   # latency probes, measurable separately
```

### What dominates recovery time

Measured on a local 4-node cluster:

| Component | Time |
|---|---|
| Failure **detection** | ~7 s at `connectionTimeout` 3.5 s, ~0.3 s at 1.0 s |
| RAFT re-election | ~0.01–0.3 s |
| Controlled re-assignment | ~0.002 s |
| Checkpoint restore | ~0.005 s |
| SuperLink startup | ~5 s |

The consensus machinery is not the cost. Recovery time is set by how long a
node waits before declaring a silent partner dead, and by Flower process
startup. Report `--connection-timeout` alongside any recovery-time result:

```bash
python experiments/run_cluster.py --nodes 4 --rounds 4 --connection-timeout 1.0
```

Lowering it scales the whole RAFT timing window together (PySyncObj
cross-validates these), taking total recovery from ~12 s to ~5 s.

---

## Differences from the published description

Recorded here so results from this branch are not compared naively against the
published figures.

1. **The published numbers were measured on the legacy Flower API.** Recovery
   there was a single `start_numpy_client` call at a new address. Here it is:
   kill the SuperNode, respawn it against the new SuperLink, boot
   `flower-superlink`, submit the run. Process startup dominates, and the
   measurement is not like-for-like. `legacy/` remains the artifact of record
   for the published figures.

2. **Ports.** The article uses 8080 for FL; Flower 1.33 uses the Fleet API on
   9092. §7.2.2's `iftop` filter needs updating.

3. **Latency probes have their own port (6000).** The original probed port
   5000, folding probe traffic into the measurement §7.2.2 attributes to
   consensus. They cannot instead target the FL port, which only exists while a
   node happens to be the aggregator — probing it deadlocks the cluster at
   startup, since every node then scores infinity, nobody is rankable, and no
   aggregator ever starts to open the port.

4. **"ICMP echo requests" (§6.1.3).** `tcppinglib.tcpping` performs a TCP
   handshake, not ICMP. The code is TCP; the wording should say so.

5. **"Flower simulation mode" for the 100-node runs (§7.2.3).** Flower's
   simulation runs clients as in-process Ray actors with no real sockets, so it
   cannot host a RAFT member per node. `run_cluster.py` spawns N independent OS
   processes, which is what the description ("each represented as an
   independent Python process") actually matches.

6. **Python 3.11+**, so the Bullseye/3.9 testbed of §7.1 needs Bookworm.

7. **No data partitioning.** §7.1 says each client "processes a local
   partition"; as in the original, `load_data` gives every node the full,
   identical dataset. `flnode/model.py:partition` is the hook, currently unused.

8. **Latency-aware client selection** (IFIP §II-A) is not implemented, as in
   the original.

---

## Tests

```bash
pytest
```

61 tests. `test_ranking.py` covers the determinism claim; `test_checkpoint.py`
covers model serialisation and the round accounting that must survive repeated
recovery, including one test against a real 3-node RAFT cluster.
