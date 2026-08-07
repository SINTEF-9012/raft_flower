"""Inject an aggregator failure, the way the article's experiments do.

Section 7.1.1: "we simulate aggregator failures during training by manually
terminating the aggregator's Python process ... performed at controlled points
in time -- specifically, when the aggregator is in an idle state, waiting to
receive training updates from the client nodes."

This joins the RAFT cluster read-only to find out which node currently holds
the aggregator role, then kills that node's entire process tree (supervisor,
SuperLink and ServerApp together), which is what a real node failure looks
like to the rest of the cluster.

    python experiments/kill_aggregator.py                 # kill now
    python experiments/kill_aggregator.py --after-round 2 # wait for a round first

Run it repeatedly to exercise the chained failures of Section 7.1.1.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flnode import raft  # noqa: E402
from flnode.checkpoint import completed_rounds, get_aggregator  # noqa: E402


def pid_alive(pid: int) -> bool:
    if os.name == "posix":
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
    out = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True
    ).stdout
    return str(pid) in out


def kill_tree(pid: int, timeout: float = 10.0) -> bool:
    """Kill a node's whole process group. SIGKILL-equivalent, no cleanup.

    Verified by polling rather than by the killer's exit status: `taskkill /T`
    reports failure when any member of the tree has already exited, which is
    routine here and does not mean the target survived.
    """
    try:
        if os.name == "posix":
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        else:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
    except OSError:
        pass

    deadline = time.time() + timeout
    while time.time() < deadline:
        if not pid_alive(pid):
            return True
        time.sleep(0.2)
    return not pid_alive(pid)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Kill the current FL aggregator")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--after-round", type=int, default=None,
        help="Wait until this many rounds have completed before killing.",
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args(argv)

    repo = Path(__file__).resolve().parent.parent
    cluster_file = repo / args.results_dir / "cluster.json"
    if not cluster_file.exists():
        print(f"No cluster map at {cluster_file}; is the cluster running?")
        return 1
    cluster = json.loads(cluster_file.read_text(encoding="utf-8"))
    pids: dict[str, int] = {a: int(p) for a, p in cluster["nodes"].items()}

    # Read-only member: observes the replicated state without voting.
    sync_obj, state = raft.join_cluster(None, list(pids))
    try:
        if not raft.wait_ready(sync_obj, timeout=60.0):
            print("Could not join the RAFT cluster")
            return 1

        deadline = time.time() + args.timeout
        while time.time() < deadline:
            aggregator = get_aggregator(state)
            done = completed_rounds(state)
            if aggregator is None:
                time.sleep(0.5)
                continue
            if args.after_round is not None and done < args.after_round:
                time.sleep(0.5)
                continue
            break
        else:
            print("Timed out waiting for the kill condition")
            return 1

        aggregator = get_aggregator(state)
        pid = pids.get(aggregator)
        if pid is None:
            print(f"Aggregator {aggregator} is not in the cluster map")
            return 1

        print(
            f"Killing aggregator {aggregator} (pid {pid}) "
            f"after {completed_rounds(state)} completed round(s)"
        )
        ok = kill_tree(pid)
        print("killed" if ok else "kill failed (already gone?)")
        return 0 if ok else 1
    finally:
        try:
            sync_obj.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
