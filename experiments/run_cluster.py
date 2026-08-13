"""Launch an N-node FL cluster on one host.

Replaces the original `run.sh`, which was hard-coded to exactly four nodes on
ports 5000-5003. The article's experiments sweep 1 to 10 workers on physical
boards (Section 7.2) and 10 to 100 simulated nodes (Section 7.2.3), so the node
count has to be a parameter.

A note on what "simulated" means here. Section 7.2.3 describes the large-scale
runs as using "the simulation mode offered by the Flower framework ... each
represented as an independent Python process". Flower's actual simulation mode
runs clients as in-process Ray actors and gives them no real sockets, so it
cannot host a RAFT member per node. What this script does -- N independent OS
processes on one host -- is what those experiments actually require, and what
the quoted description matches.

Writes `results/cluster.json` mapping each node's RAFT address to its PID, so
`kill_aggregator.py` can terminate a specific node's process tree.

    python experiments/run_cluster.py --nodes 4 --rounds 3
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an N-node flnode cluster locally")
    parser.add_argument("--nodes", type=int, default=4, help="Total nodes (1 aggregator + N-1 workers)")
    parser.add_argument("--rounds", type=int, default=3, help="FL rounds")
    parser.add_argument("--raft-base", type=int, default=5000, help="First RAFT port")
    parser.add_argument(
        "--fl-port", type=int, default=9092,
        help="Fleet API port, shared by all nodes. Only one node is the "
             "aggregator at any moment, so only one SuperLink is bound at a "
             "time -- and this matches the one-node-per-host deployment the "
             "article evaluates, where every node uses the same port.",
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--ttl", type=float, default=30.0)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--reassign", choices=["on-failure", "every-round"], default="on-failure")
    parser.add_argument(
        "--min-nodes", type=int, default=None,
        help="min_train_nodes for the strategy. Defaults to nodes-2, NOT "
             "nodes-1: the aggregator does not run a SuperNode, so N nodes "
             "give N-1 workers, and one aggregator failure leaves N-2. A "
             "higher value makes the run block after the first failure.",
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument(
        "--connection-timeout", type=float, default=None,
        help="RAFT connectionTimeout, passed through to each node. Dominates "
             "recovery time -- see flnode.raft.default_conf.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    addrs = [f"{args.host}:{args.raft_base + i}" for i in range(args.nodes)]
    fl_ports = [args.fl_port] * args.nodes
    # N nodes -> N-1 workers (the aggregator runs no SuperNode) -> N-2 after
    # one aggregator failure. Requiring more than that would stall recovery.
    min_nodes = args.min_nodes if args.min_nodes is not None else max(1, args.nodes - 2)

    # N nodes give N-1 workers, and one aggregator failure leaves N-2. Nodes
    # clamp the threshold down at run time so this no longer stalls, but an
    # explicit value that can never be met is still worth flagging: it means
    # the run will silently train with fewer clients than asked for.
    if args.min_nodes is not None and args.min_nodes > args.nodes - 2:
        survivable = max(0, args.nodes - 1 - args.min_nodes)
        print(
            f"NOTE: --min-nodes {args.min_nodes} with {args.nodes} nodes tolerates "
            f"{survivable} aggregator failure(s) at full client count.\n"
            f"      Nodes will reduce the threshold automatically after that "
            f"(min_nodes_clamped events); pass --strict-min-nodes to a node to "
            f"disable clamping.\n",
            flush=True,
        )

    results = REPO / args.results_dir
    results.mkdir(parents=True, exist_ok=True)
    for stale in results.glob("events-*.jsonl"):
        stale.unlink()

    procs: dict[str, subprocess.Popen] = {}
    logs: dict[str, object] = {}

    print(f"Starting {args.nodes} nodes, {args.rounds} rounds, min_nodes={min_nodes}")
    for i, addr in enumerate(addrs):
        peers = [a for a in addrs if a != addr]
        cmd = [
            sys.executable, "-m", "flnode.node",
            "--self", addr,
            "--peers", *peers,
            "--fl-port", str(fl_ports[i]),
            "--results-dir", args.results_dir,
            "--ttl", str(args.ttl),
            "--margin", str(args.margin),
            "--reassign", args.reassign,
            "--min-nodes", str(min_nodes),
            "--app", str(REPO),
        ]
        if args.connection_timeout is not None:
            cmd += ["--connection-timeout", str(args.connection_timeout)]
        # Exactly one node seeds the round budget; the replicated value wins
        # everywhere else.
        if i == 0:
            cmd += ["--rounds", str(args.rounds)]

        handle = (results / f"node-{args.raft_base + i}.log").open("w", encoding="utf-8")
        logs[addr] = handle
        kwargs = {}
        if os.name == "posix":
            kwargs["start_new_session"] = True  # own process group, so we can kill the tree
        procs[addr] = subprocess.Popen(
            cmd, cwd=str(REPO), stdout=handle, stderr=subprocess.STDOUT, **kwargs
        )
        print(f"  {addr}  fleet={fl_ports[i]}  pid={procs[addr].pid}")

    (results / "cluster.json").write_text(
        json.dumps(
            {
                "nodes": {a: p.pid for a, p in procs.items()},
                "fl_ports": dict(zip(addrs, fl_ports)),
                "rounds": args.rounds,
                "started": time.time(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nCluster map: {results / 'cluster.json'}")
    print("Logs:        %s/node-*.log" % results)
    print("Events:      %s/events-*.jsonl" % results)
    print("\nCtrl-C to stop.\n")

    try:
        while any(p.poll() is None for p in procs.values()):
            time.sleep(1)
        print("All nodes exited.")
    except KeyboardInterrupt:
        print("\nStopping cluster...")
    finally:
        for addr, proc in procs.items():
            if proc.poll() is None:
                try:
                    if os.name == "posix":
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    else:
                        subprocess.run(
                            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                            capture_output=True,
                        )
                except OSError:
                    pass
        for handle in logs.values():
            try:
                handle.close()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
