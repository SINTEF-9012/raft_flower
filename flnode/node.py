"""The FL node supervisor: one program, run identically on every node.

This implements the algorithm of Figure 2 in the TAAS article. As Section 6.2
puts it: "in our implementation all FL nodes execute exactly the same code
base, and depending on the election/assignment outcome, proceed with either the
aggregator or the worker role."

Process layout on each node:

    flnode supervisor                 (this file -- always running)
    |- PySyncObj member        :5000  RAFT election, heartbeats, replicated state
    |- ContextMonitor thread          publishes ctx:<self> every period
    |- EventLog                       JSONL measurement trace
    `- exactly one child group, according to the current role:
         AGGREGATOR -> flower-superlink  +  flwr run   (the ServerApp)
         WORKER     -> flower-supernode --superlink <aggregator>:<fleet port>

The supervisor never restarts: it holds RAFT membership and contextual
monitoring for the lifetime of the node. Only the child group changes when the
role moves. That separation is what lets a node switch roles without leaving
the consensus cluster, and it is why aggregator liveness can be read straight
off RAFT's own heartbeats rather than probed separately.

Why the SuperLink moves with the aggregator: leaving it pinned to one host
would reintroduce exactly the single point of failure this work exists to
remove. A SuperNode's `--superlink` address is fixed at process start, so
handover restarts the SuperNode child against the new aggregator. The
aggregator itself always submits `flwr run` to the SuperLink on its *own*
machine, so that address is loopback and never changes.

Two phases of recovery, kept distinct exactly as the article specifies:
first RAFT re-elects a leader to restore a consistent replicated state
(Section 5.3), and only then do all nodes independently rank each other to
decide who becomes the aggregator (Section 5.4). The RAFT leader and the FL
aggregator are separate roles and need not be the same node (Section 6.2.1).
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flnode import raft
from flnode.checkpoint import (
    bump_run_generation,
    completed_rounds,
    get_aggregator,
    is_training_complete,
    num_rounds,
    remaining_rounds,
    run_generation,
    set_aggregator,
    set_num_rounds,
)
from flnode.context import (
    PROBE_PORT_OFFSET,
    ContextMonitor,
    ProbeConfig,
    ProbeListener,
    probe_port_of,
    read_context_table,
)
from flnode.events import (
    AGGREGATOR_DOWN_DETECTED,
    AGGREGATOR_SERVING,
    CHILD_EXITED,
    CHILD_SPAWNED,
    MIN_NODES_CLAMPED,
    NODE_STARTED,
    NODE_STOPPING,
    RAFT_LEADER_ELECTED,
    RANKING_COMPUTED,
    ROLE_DECIDED,
    WORKER_SERVING,
    EventLog,
    default_path,
)
from flnode.ranking import DEFAULT_TTL, rank_nodes, select_aggregator

AGGREGATOR = "aggregator"
WORKER = "worker"


#: Offset from a node's RAFT port to its ClientAppIo port. See `Ports`.
CLIENTAPPIO_PORT_OFFSET = 10_000


@dataclass(frozen=True)
class Ports:
    """Flower's ports for this node.

    `fleet` is the only port other nodes ever contact, and every node uses the
    same value (9092 by default) because only one node is the aggregator at a
    time. That is what keeps the Section 7.2.2 traffic split clean: inter-node
    FL traffic is entirely on the Fleet port and RAFT entirely on port 5000.

    `serverappio` and `control` are loopback-only on the aggregator, so they
    can be derived from `fleet` and shared too.

    `clientappio` cannot. Every *worker* runs a SuperNode with its own
    ClientAppIo listener, so when several nodes share a host they would all
    try to bind the same port and all but one would fail to start. It is
    therefore derived from this node's RAFT port, which is unique by
    construction. The offset keeps it clear of the 909x block.
    """

    host: str
    fleet: int
    raft_port: int

    @property
    def serverappio(self) -> int:
        return self.fleet - 1

    @property
    def control(self) -> int:
        return self.fleet + 1

    @property
    def clientappio(self) -> int:
        return self.raft_port + CLIENTAPPIO_PORT_OFFSET


def scripts_dir() -> Path:
    """Directory holding this interpreter's console scripts."""
    # On POSIX venvs the interpreter already lives in bin/ alongside them; on
    # Windows both python.exe and the .exe shims live in Scripts/.
    return Path(sys.executable).parent


def child_env() -> dict[str, str]:
    """Environment for spawned Flower processes.

    The SuperLink and SuperNode spawn *further* helpers (`flwr-serverapp`,
    `flwr-clientapp`) by bare name, so the console-scripts directory has to be
    on their PATH or they die with an unexplained "file not found". Also forces
    UTF-8: Flower's output contains non-ASCII characters that raise
    UnicodeEncodeError on the default Windows console codepage.
    """
    env = dict(os.environ)
    env["PATH"] = f"{scripts_dir()}{os.pathsep}{env.get('PATH', '')}"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def resolve_binary(name: str) -> str:
    """Absolute path to a console script.

    Necessary because on Windows `CreateProcess` resolves the executable using
    the *calling* process's PATH and ignores the PATH we put in the child's
    environment -- so passing a bare name fails with WinError 2 even though
    `child_env()` is correct. Resolving here fixes the launch; `child_env()`
    still matters for the helpers those children spawn in turn.
    """
    found = shutil.which(name, path=str(scripts_dir()))
    if found:
        return found
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(
        f"Could not find '{name}'. Is flwr installed in {scripts_dir()}?"
    )


def effective_min_nodes(
    configured: int, live_members: int, *, strict: bool = False
) -> int:
    """Clamp the strategy's client threshold to the workers that actually exist.

    The article's recovery guarantee holds "as long as ... a sufficient number
    of worker nodes remain operational" (Section 7.1.1). A threshold fixed
    before the failure cannot express that: with 3 nodes and
    `min_train_nodes=2`, losing the aggregator leaves exactly one worker
    against a requirement of two, and the run waits forever for a node that no
    longer exists -- the failure looks like a hang rather than a
    misconfiguration.

    The aggregator itself runs no SuperNode, so `live_members - 1` is the
    number of workers available to it. Never returns less than 1, because
    federated training needs at least one client.

    `strict=True` keeps the configured value regardless, for experiments that
    need the client count pinned across a failure and would rather see the run
    stall than silently train with fewer clients.
    """
    if strict:
        return max(1, configured)
    workers = max(1, live_members - 1)
    return max(1, min(configured, workers))


def port_is_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def wait_for_port(host: str, port: int, timeout: float = 60.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if port_is_open(host, port):
            return True
        time.sleep(0.2)
    return False


class Supervisor:
    """Owns RAFT membership, contextual monitoring, and the child process group."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.self_addr: str = args.self_addr
        self.partners: list[str] = list(args.peers)
        self.app_dir: str = args.app
        self.federation: str = args.federation
        self.results_dir: str = args.results_dir
        self.ttl: float = args.ttl
        self.margin: float = args.margin
        self.local_epochs: int = args.local_epochs
        self.min_nodes: int = args.min_nodes
        self.strict_min_nodes: bool = args.strict_min_nodes
        self.death_confirm: float = args.death_confirm
        self.member_grace: float = args.member_grace
        self.reassign_every_round: bool = args.reassign == "every-round"
        host, raft_port = self.self_addr.rsplit(":", 1)
        self.ports = Ports(host=host, fleet=args.fl_port, raft_port=int(raft_port))

        self.log = EventLog(default_path(self.self_addr, self.results_dir), self.self_addr)
        self._children: list[subprocess.Popen] = []
        self._child_logs: list[Any] = []
        self._shutdown = threading.Event()

        self.probe_offset: int = args.probe_offset
        self.connection_timeout: float | None = args.connection_timeout
        self.election_min: float | None = args.election_min
        self.election_max: float | None = args.election_max
        self.sync_obj = None
        self.state = None
        self.monitor: ContextMonitor | None = None
        self.probe_listener: ProbeListener | None = None

    # -- lifecycle ----------------------------------------------------------

    def start(self, bootstrap_rounds: int | None) -> None:
        # Started before anything else: peers cannot measure latency to this
        # node until it is listening, and a node nobody can measure is a node
        # nobody can rank.
        self.probe_listener = ProbeListener(
            probe_port_of(self.self_addr, self.probe_offset)
        )
        self.probe_listener.start()

        self.sync_obj, self.state = raft.join_cluster(
            self.self_addr,
            self.partners,
            conf=raft.default_conf(
                connectionTimeout=self.connection_timeout,
                raftMinTimeout=self.election_min,
                raftMaxTimeout=self.election_max,
            ),
        )
        if not raft.wait_ready(self.sync_obj, timeout=120.0):
            raise RuntimeError(f"{self.self_addr}: could not join RAFT cluster")

        self.log.emit(
            NODE_STARTED,
            peers=self.partners,
            fleet_port=self.ports.fleet,
            raft_leader=raft.leader_of(self.sync_obj),
        )

        # Only the bootstrapping node sets the round budget, and only if it has
        # not already been set -- on a restart the replicated value wins.
        if bootstrap_rounds is not None and not num_rounds(self.state):
            set_num_rounds(self.state, bootstrap_rounds)

        self.monitor = ContextMonitor(
            self.state,
            self.self_addr,
            self.partners,
            fl_port=self.ports.fleet,
            probe_offset=self.probe_offset,
            config=ProbeConfig(period=self.ttl / 3.0),
            log=self.log,
        )
        self.monitor.start()

        # Publish one record immediately: ranking cannot proceed until every
        # live node has a fresh record, and waiting a whole period here would
        # be counted as recovery time.
        self.monitor.publish(self.monitor.sample())

        # Then wait for everyone else's first record before ranking anything.
        # Ranking is only deterministic across nodes if the nodes are ranking
        # the same table; deciding while records are still arriving lets two
        # nodes see different subsets, reach different answers, and both claim
        # the aggregator role.
        self.wait_for_context(timeout=60.0)

    def wait_for_members(self, timeout: float) -> bool:
        """Block until every configured node is RAFT-reachable, or give up.

        Returns True if the full cluster is present. Returning False is normal
        and expected after a real failure: the dead node will never come back,
        so the caller proceeds with whoever is there.
        """
        expected = {self.self_addr, *self.partners}
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not self._shutdown.is_set():
            if raft.live_members(self.sync_obj, self.self_addr) >= expected:
                return True
            time.sleep(0.2)
        return raft.live_members(self.sync_obj, self.self_addr) >= expected

    def wait_for_context(self, timeout: float) -> bool:
        """Block until every cluster member has published a context record."""
        expected = {self.self_addr, *self.partners}
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not self._shutdown.is_set():
            table = read_context_table(self.state)
            missing = expected - set(table)
            if not missing:
                return True
            # A member that is already unreachable will never publish; do not
            # wait on it. Recovery handles nodes that die later.
            live = raft.live_members(self.sync_obj, self.self_addr)
            if not (missing & live):
                return True
            time.sleep(0.25)
        return False

    def shutdown(self) -> None:
        self._shutdown.set()
        self.log.emit(NODE_STOPPING)
        self.kill_children()
        if self.monitor is not None:
            self.monitor.stop()
        if self.probe_listener is not None:
            self.probe_listener.stop()
        if self.sync_obj is not None:
            try:
                self.sync_obj.destroy()
            except Exception:
                pass
        self.log.close()

    # -- child processes ----------------------------------------------------

    def spawn(self, argv: list[str], name: str) -> subprocess.Popen:
        argv = [resolve_binary(argv[0]), *argv[1:]]
        # Child output goes to its own file rather than being discarded: when
        # a SuperLink or ServerApp fails, its traceback is the only thing that
        # explains why a node dropped out of its role.
        slot = self.self_addr.rsplit(":", 1)[1]
        log_path = Path(self.results_dir) / f"child-{slot}-{name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("a", encoding="utf-8", errors="replace")
        self._child_logs.append(handle)

        popen_kwargs = {}
        if os.name == "posix":
            # Own process group, so the whole tree can be signalled at once.
            popen_kwargs["start_new_session"] = True
        proc = subprocess.Popen(
            argv,
            env=child_env(),
            cwd=self.app_dir,
            stdout=handle,
            stderr=subprocess.STDOUT,
            **popen_kwargs,
        )
        self._children.append(proc)
        self.log.emit(CHILD_SPAWNED, child=name, pid=proc.pid, log=str(log_path))
        return proc

    def kill_children(self) -> None:
        """Terminate the child group, including grandchildren.

        `Popen.kill()` alone is not enough: a SuperLink spawns `flwr-serverapp`
        and a SuperNode spawns `flwr-clientapp`, and killing only the direct
        child orphans those. They then keep holding the Fleet and ClientAppIo
        ports, so the node cannot retake a role, and on the recovery path the
        wait for them to exit lands squarely inside the measured recovery time.
        """
        for proc in self._children:
            if proc.poll() is None:
                try:
                    if os.name == "posix":
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    else:
                        subprocess.run(
                            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                            capture_output=True,
                            timeout=10,
                        )
                except (OSError, subprocess.SubprocessError):
                    try:
                        proc.kill()
                    except OSError:
                        pass
        for proc in self._children:
            try:
                # Short: the tree has already been force-killed, so this only
                # reaps the handle. A long timeout here would be pure latency
                # added to every recovery.
                proc.wait(timeout=3)
            except Exception:
                pass
            self.log.emit(CHILD_EXITED, pid=proc.pid, returncode=proc.returncode)
        self._children.clear()
        for handle in self._child_logs:
            try:
                handle.close()
            except Exception:
                pass
        self._child_logs.clear()

    # -- role decision ------------------------------------------------------

    def decide_role(self) -> tuple[str, str | None]:
        """Rank peers locally and decide this node's own role.

        Every node runs this on its own replicated copy of the context table
        and, by construction, reaches the same answer -- so no node has to be
        *told* its role. The `fl:aggregator` key is written afterwards purely
        as a trace; nothing reads it to make a decision, because the node that
        would have written it may be the one that just crashed.
        """
        table = read_context_table(self.state)
        now = time.time()
        incumbent = get_aggregator(self.state)

        ranking = rank_nodes(table, now=now, ttl=self.ttl)

        # A live incumbent keeps the role. Re-ranking on every iteration would
        # hand the aggregator around on ordinary latency jitter, which is not
        # what the article evaluates: re-assignment is triggered by *failure*
        # (Sections 5.3-5.4), while continuous re-evaluation is the proactive
        # extension discussed in Section 8.1 and enabled by --reassign
        # every-round.
        if incumbent is not None and not self.reassign_every_round:
            if self._is_live(incumbent, table, now):
                self.log.emit(
                    RANKING_COMPUTED,
                    ranking=[[e.node, e.score] for e in ranking],
                    incumbent=incumbent,
                    chosen=incumbent,
                    reason="incumbent alive",
                )
                role = AGGREGATOR if incumbent == self.self_addr else WORKER
                self.log.emit(ROLE_DECIDED, role=role, aggregator=incumbent)
                return role, incumbent

        # Exclude nodes RAFT already knows are gone. Staleness alone would get
        # there eventually, but only after a full TTL, and until then a crashed
        # node stays in the ranking -- so recovery time would be bounded below
        # by the TTL rather than by how fast the failure is detected. RAFT's
        # heartbeats notice within a heartbeat interval, and every node sees
        # the same partner states, so this stays deterministic.
        dead = {
            node
            for node in ({self.self_addr, *self.partners} & set(table))
            if not raft.is_alive(self.sync_obj, node, self.self_addr)
        }

        chosen = select_aggregator(
            table,
            now=now,
            ttl=self.ttl,
            exclude=dead,
            incumbent=incumbent if self.margin > 0 else None,
            margin=self.margin,
            # Keeps a cluster moving when no node has a finite score yet --
            # at startup, or with a single node and no peers to measure.
            fallback=True,
        )

        self.log.emit(
            RANKING_COMPUTED,
            ranking=[[entry.node, entry.score] for entry in ranking],
            incumbent=incumbent,
            chosen=chosen,
            excluded=sorted(dead),
            reason="ranked",
        )

        role = AGGREGATOR if chosen == self.self_addr else WORKER
        self.log.emit(ROLE_DECIDED, role=role, aggregator=chosen)

        # The node that ranks itself first claims the role by recording it.
        # Ranking is deterministic, so no two nodes should claim concurrently;
        # RAFT linearises the write regardless. This key is a trace, not the
        # source of truth -- a node never learns its role by reading it, so a
        # refused write (routine while RAFT is still settling) is not fatal and
        # is simply retried on the next iteration.
        if chosen == self.self_addr and incumbent != chosen:
            if not set_aggregator(self.state, chosen):
                self.log.emit(
                    "aggregator_record_deferred",
                    aggregator=chosen,
                    reason="RAFT write refused, retrying next iteration",
                )

        return role, chosen

    def confirm_dead(self, node: str, poll: float = 0.1) -> bool:
        """Is `node` still absent after watching it for `self.death_confirm`?

        A failure detector built on one sample cannot distinguish a crash from
        a scheduling hiccup. This requires the node to stay unreachable for the
        whole confirmation window, and returns `False` the moment it reappears.

        The cost is that the window is added to every recovery, so it is kept
        short and is reported in the event log; the benefit is that transient
        load does not trigger a spurious re-assignment. It should be at least
        one RAFT heartbeat interval.
        """
        if self.death_confirm <= 0:
            return True
        deadline = time.monotonic() + self.death_confirm
        while time.monotonic() < deadline:
            if self._shutdown.is_set():
                return False
            if raft.is_alive(self.sync_obj, node, self.self_addr):
                self.log.emit(
                    "liveness_recovered",
                    node=node,
                    note="transient disconnect, not a failure",
                )
                return False
            time.sleep(poll)
        return not raft.is_alive(self.sync_obj, node, self.self_addr)

    def _is_live(self, node: str, table: dict, now: float) -> bool:
        """Is `node` both RAFT-reachable and still publishing fresh context?

        Both signals are needed. RAFT reachability alone can lag a process that
        has stopped doing useful work, and context freshness alone lags a hard
        crash by up to one TTL.
        """
        if node == self.self_addr:
            return True
        if not raft.is_alive(self.sync_obj, node, self.self_addr):
            return False
        record = table.get(node)
        if not isinstance(record, dict):
            return False
        ts = record.get("ts")
        return isinstance(ts, (int, float)) and (now - float(ts)) <= self.ttl

    def recover(self, failed: str | None) -> None:
        """The two-phase recovery flow of Sections 5.3 and 5.4.

        Timestamps emitted here are what Figures 6 and 8 are built from:
        `aggregator_down_detected` is t0, `raft_leader_elected` is t1 (RAFT
        re-election time), and `ranking_computed` is t2 (controlled
        re-assignment time).
        """
        self.log.emit(AGGREGATOR_DOWN_DETECTED, failed=failed)
        self.kill_children()

        # Phase (ii): let RAFT settle before ranking, so every node ranks
        # against a consistent replicated state. self_addr is required so that
        # a node which is itself the new leader counts as live, and so that a
        # stale report naming the node that just died is not accepted.
        leader = raft.wait_for_leader(
            self.sync_obj, timeout=60.0, self_addr=self.self_addr
        )
        self.log.emit(
            RAFT_LEADER_ELECTED,
            leader=leader,
            is_self=leader == self.self_addr,
            live_members=sorted(raft.live_members(self.sync_obj, self.self_addr)),
        )

        # Phase (iii) happens in decide_role, on the next loop iteration.

    # -- roles --------------------------------------------------------------

    def run_aggregator(self) -> None:
        """Host the SuperLink and drive a run for the remaining rounds."""
        superlink = self.spawn(
            [
                "flower-superlink",
                "--insecure",
                "--serverappio-api-address", f"0.0.0.0:{self.ports.serverappio}",
                "--fleet-api-address", f"0.0.0.0:{self.ports.fleet}",
                "--control-api-address", f"0.0.0.0:{self.ports.control}",
            ],
            "superlink",
        )
        if not wait_for_port("127.0.0.1", self.ports.control, timeout=60.0):
            raise RuntimeError("SuperLink control API did not come up")

        remaining = remaining_rounds(self.state)

        # Give slow peers a moment to appear before sizing the federation, but
        # only at cold start. RAFT connections are still being established in
        # the first seconds after launch, so reading membership immediately can
        # miss a node that is merely late, and the threshold would then be set
        # too low for the whole run.
        #
        # On the recovery path this must be skipped. The missing node is known
        # dead -- its absence was just confirmed over `--death-confirm` -- so
        # waiting for it to reappear only adds the full grace period to every
        # recovery, which is the headline metric. A non-zero run generation
        # means a run has already started, i.e. membership is known.
        if run_generation(self.state) == 0:
            self.wait_for_members(timeout=self.member_grace)

        # Size the client threshold to the cluster that actually exists now,
        # not to the one that existed at launch. Recomputed every time this
        # node takes the role, so each recovery adapts to the survivors.
        live = raft.live_members(self.sync_obj, self.self_addr)
        min_nodes = effective_min_nodes(
            self.min_nodes, len(live), strict=self.strict_min_nodes
        )
        if min_nodes != self.min_nodes:
            self.log.emit(
                MIN_NODES_CLAMPED,
                configured=self.min_nodes,
                effective=min_nodes,
                live_members=sorted(live),
            )

        self.log.emit(
            AGGREGATOR_SERVING,
            rounds=remaining,
            resumed_from_round=completed_rounds(self.state),
            fleet_port=self.ports.fleet,
            min_nodes=min_nodes,
        )

        run_config = " ".join(
            [
                f'raft-partners="{",".join([self.self_addr, *self.partners])}"',
                f'raft-self="{self.self_addr}"',
                f'results-dir="{self.results_dir}"',
                f"min-nodes={min_nodes}",
                f"local-epochs={self.local_epochs}",
            ]
        )
        # Tell the workers a new run is starting, so they relaunch their
        # SuperNodes against this SuperLink. Done before the run is submitted
        # so nobody attaches to the previous generation.
        generation = bump_run_generation(self.state)
        self.log.emit("run_generation", generation=generation, min_nodes=min_nodes)

        run = self.spawn(
            [
                "flwr", "run", self.app_dir, self.federation,
                "--run-config", run_config,
                # Without --stream, `flwr run` returns as soon as the run is
                # *submitted*, and this supervisor would tear the SuperLink
                # down underneath a run that had only just started.
                "--stream",
            ],
            "flwr-run",
        )

        # Block until the run finishes. A crash of this node kills the whole
        # process group, which is what the failure injection exercises.
        low_since: float | None = None
        while run.poll() is None and not self._shutdown.is_set():
            if self.reassign_every_round and self._better_aggregator_exists():
                break

            # A worker can die *during* a run, and Flower fixes the client
            # threshold for the lifetime of a run -- so the strategy would go
            # on waiting for a node that no longer exists. Clamping only at
            # role-assumption time covers aggregator failure but not this.
            # Restarting the run is what applies the lower threshold; progress
            # is not lost, because the new run resumes from the last replicated
            # checkpoint. The in-flight round's client work is discarded, which
            # is the same limitation the article notes in Section 8.2.
            if not self.strict_min_nodes:
                current = effective_min_nodes(
                    self.min_nodes,
                    len(raft.live_members(self.sync_obj, self.self_addr)),
                )
                if current < min_nodes:
                    # Require the drop to persist, for the same reason the
                    # liveness check does: membership read at one instant can
                    # dip under load, and each spurious dip would restart a
                    # perfectly healthy run, losing the round in flight and
                    # inflating any measured overhead.
                    if low_since is None:
                        low_since = time.monotonic()
                    elif time.monotonic() - low_since >= self.death_confirm:
                        self.log.emit(
                            MIN_NODES_CLAMPED,
                            configured=min_nodes,
                            effective=current,
                            reason="worker lost mid-run; restarting to apply it",
                            live_members=sorted(
                                raft.live_members(self.sync_obj, self.self_addr)
                            ),
                        )
                        break
                else:
                    low_since = None

            time.sleep(0.5)

        self.kill_children()
        del superlink

    def _better_aggregator_exists(self) -> bool:
        """Proactive re-assignment check (the article's Section 8.1 extension).

        Only consulted when `--reassign every-round` is set. The default is
        `on-failure`, which is the behaviour the article evaluates.
        """
        table = read_context_table(self.state)
        chosen = select_aggregator(
            table,
            now=time.time(),
            ttl=self.ttl,
            incumbent=self.self_addr,
            margin=max(self.margin, 0.10),
        )
        return chosen is not None and chosen != self.self_addr

    def run_worker(self, aggregator: str) -> None:
        """Attach a SuperNode to the aggregator and watch that it stays up."""
        host = aggregator.rsplit(":", 1)[0]
        table = read_context_table(self.state)
        record = table.get(aggregator) or {}
        fleet = record.get("fl_port")
        if not isinstance(fleet, int):
            fleet = self.ports.fleet

        # Captured before spawning: if the aggregator starts a newer run while
        # this SuperNode is attaching, the mismatch is caught on the next poll.
        generation = run_generation(self.state)

        supernode = self.spawn(
            [
                "flower-supernode",
                "--insecure",
                "--superlink", f"{host}:{fleet}",
                "--clientappio-api-address", f"127.0.0.1:{self.ports.clientappio}",
                "--node-config", f'partition-id={self._partition_id()} '
                                 f"num-partitions={len(self.partners) + 1}",
            ],
            "supernode",
        )
        self.log.emit(
            WORKER_SERVING,
            aggregator=aggregator,
            fleet_port=fleet,
            generation=generation,
        )

        # This is the check Section 6.2 requires of every worker. It reads the
        # partner state RAFT already maintains from its heartbeats, so it costs
        # no extra traffic -- and unlike the original's bare `except:` around a
        # gRPC call, it detects the failure even while the node sits idle.
        while not self._shutdown.is_set():
            if is_training_complete(self.state):
                return

            if not raft.is_alive(self.sync_obj, aggregator, self.self_addr):
                # Confirm before acting. A single negative reading is not
                # evidence of death: under load, one missed heartbeat window
                # is enough for pysyncobj to mark a perfectly healthy partner
                # disconnected, and tearing the SuperNode down on that alone
                # produces churn that looks exactly like the failure it was
                # meant to handle.
                if self.confirm_dead(aggregator):
                    self.recover(failed=aggregator)
                    return

            # The aggregator replaced its run, so this SuperNode is attached to
            # a SuperLink that no longer exists. The aggregator never looked
            # unhealthy, so the liveness check above cannot catch this.
            current = run_generation(self.state)
            if current != generation:
                self.log.emit(
                    "run_generation_changed",
                    was=generation,
                    now=current,
                    note="relaunching SuperNode against the new run",
                )
                self.kill_children()
                return

            # Safety net: a SuperNode that died for any other reason must be
            # replaced, or this node silently stops participating while still
            # looking healthy to everyone else.
            if supernode.poll() is not None:
                self.log.emit(
                    CHILD_EXITED,
                    pid=supernode.pid,
                    returncode=supernode.returncode,
                    note="SuperNode exited unexpectedly; relaunching",
                )
                self.kill_children()
                return

            time.sleep(0.2)

    def _partition_id(self) -> int:
        """Stable index of this node among all members, for data partitioning."""
        members = sorted([self.self_addr, *self.partners])
        return members.index(self.self_addr)

    # -- main loop ----------------------------------------------------------

    def loop(self) -> None:
        failures = 0
        while not self._shutdown.is_set():
            if is_training_complete(self.state):
                self.log.emit(
                    NODE_STOPPING,
                    reason="training complete",
                    completed=completed_rounds(self.state),
                )
                return

            try:
                role, aggregator = self.decide_role()

                if aggregator is None:
                    # Nobody is rankable yet: either the cluster is still
                    # starting or every context record is stale. Wait for the
                    # monitors rather than guessing.
                    time.sleep(1.0)
                    continue

                if role == AGGREGATOR:
                    self.run_aggregator()
                else:
                    self.run_worker(aggregator)
                failures = 0
            except Exception as exc:
                # A supervisor that dies on a transient error is itself a node
                # failure, which is the thing this system exists to survive.
                # Recovery is precisely when RAFT writes are most likely to be
                # refused (leader in flux), so one iteration failing must never
                # take the node down with it.
                failures += 1
                self.log.emit(
                    "loop_error",
                    error=f"{type(exc).__name__}: {exc}",
                    consecutive=failures,
                )
                self.kill_children()
                time.sleep(min(1.0 * failures, 10.0))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flnode",
        description=(
            "RAFT-enhanced federated learning node with controlled aggregator "
            "self-recovery. Every node runs this same command."
        ),
    )
    parser.add_argument(
        "-s", "--self", dest="self_addr", required=True,
        help="This node's RAFT address, e.g. 10.0.0.1:5000",
    )
    parser.add_argument(
        "-p", "--peers", nargs="+", required=True,
        help="RAFT addresses of the other nodes",
    )
    parser.add_argument(
        "-r", "--rounds", type=int, default=None,
        help="Total FL rounds. Set on exactly one node to bootstrap the run; "
             "an already-replicated value always wins.",
    )
    parser.add_argument(
        "--fl-port", type=int, default=9092,
        help="Flower Fleet API port (default: 9092). The SuperLink's other "
             "ports are derived from it and stay loopback-only.",
    )
    parser.add_argument(
        "--reassign", choices=["on-failure", "every-round"], default="on-failure",
        help="on-failure reproduces the article's evaluated behaviour; "
             "every-round enables the proactive re-assignment of Section 8.1.",
    )
    parser.add_argument(
        "--margin", type=float, default=0.0,
        help="Relative improvement a challenger must show before the "
             "aggregator role moves. Guards against flapping on jitter.",
    )
    parser.add_argument(
        "--ttl", type=float, default=DEFAULT_TTL,
        help="Seconds after which a context record is considered stale and "
             "its node treated as dead (default: %(default)s).",
    )
    parser.add_argument(
        "--probe-offset", type=int, default=PROBE_PORT_OFFSET,
        help="Latency probe port = RAFT port + this offset (default: "
             "%(default)s). Probes need a target that listens for the whole "
             "life of the node, which the Fleet port is not.",
    )
    parser.add_argument(
        "--connection-timeout", type=float, default=None,
        help="RAFT: seconds before a silent partner is declared disconnected "
             "(pysyncobj default 3.5). This is how long after a crash the "
             "survivors take to NOTICE, and it dominates measured recovery "
             "time -- detection lag was ~7 s at the default, against ~0.3 s "
             "for re-election plus re-assignment combined. Report it with any "
             "recovery-time result.",
    )
    parser.add_argument(
        "--member-grace", type=float, default=15.0,
        help="Seconds to wait for every configured node to become reachable "
             "before sizing the client threshold (default: %(default)s). "
             "Prevents a late-joining node from being excluded at startup, "
             "which would run the whole federation with fewer clients than "
             "requested. After a genuine failure this simply times out.",
    )
    parser.add_argument(
        "--death-confirm", type=float, default=1.0,
        help="Seconds a node must stay unreachable before it is declared "
             "failed (default: %(default)s). Guards against acting on a single "
             "missed heartbeat, which under load produces spurious "
             "re-assignment. Adds to measured recovery time, so report it "
             "alongside recovery results. Set 0 to react immediately.",
    )
    parser.add_argument(
        "--election-min", type=float, default=None,
        help="RAFT randomised election timeout, lower bound (default 0.4).",
    )
    parser.add_argument(
        "--election-max", type=float, default=None,
        help="RAFT randomised election timeout, upper bound (default 1.4).",
    )
    parser.add_argument("--app", default=".", help="Path to the Flower app")
    parser.add_argument("--federation", default="flnode-local")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument(
        "--min-nodes", type=int, default=1,
        help="Desired number of clients per round. Treated as a maximum: it is "
             "automatically reduced to the number of surviving workers each "
             "time a node takes the aggregator role, so losing a node shrinks "
             "the federation instead of stalling it. A min_nodes_clamped event "
             "records any reduction.",
    )
    parser.add_argument(
        "--strict-min-nodes", action="store_true",
        help="Never reduce --min-nodes. Keeps the client count pinned across a "
             "failure, at the cost of stalling if too few workers remain. For "
             "experiments where a varying client count would confound results.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    supervisor = Supervisor(args)

    def handle_signal(signum, frame):  # noqa: ARG001
        supervisor._shutdown.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, handle_signal)
        except (ValueError, OSError):
            pass

    try:
        supervisor.start(args.rounds)
        supervisor.loop()
    except KeyboardInterrupt:
        pass
    finally:
        supervisor.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
