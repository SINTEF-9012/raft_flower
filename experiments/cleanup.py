"""Kill leftover cluster processes between runs.

A run that is interrupted rather than shut down cleanly can leave orphaned
`flower-superlink`, `flower-supernode`, `flwr-serverapp` or `flwr-clientapp`
processes behind. They keep holding the Fleet API port, so the *next* run fails
with a connection or bind error that looks like a new bug rather than residue
from the previous attempt.

    python experiments/cleanup.py           # kill leftovers, report what went
    python experiments/cleanup.py --dry-run # list them without killing

Dependency-free by design: this has to work when the environment is in a
half-broken state, so it shells out to the platform's own tools rather than
importing anything.
"""

from __future__ import annotations

import argparse
import os
import re
import socket
import subprocess
import sys

#: Command-line fragments identifying a process this project started.
PATTERNS = (
    "flnode.node",
    "experiments/run_cluster.py",
    "experiments\\run_cluster.py",
    "raft_holder",
)

#: Flower's own executables, which our supervisors spawn as children.
EXECUTABLES = (
    "flower-superlink",
    "flower-supernode",
    "flwr-serverapp",
    "flwr-clientapp",
    "flwr-superexec",
    "flower-superexec",
)

#: Ports worth reporting on afterwards: RAFT, probe listeners, Flower.
CHECK_PORTS = (5000, 5001, 5002, 5003, 6000, 6001, 6002, 6003, 9091, 9092, 9093)


def _windows_matches() -> list[tuple[int, str]]:
    """(pid, command line) for python processes belonging to this project."""
    out = subprocess.run(
        ["wmic", "process", "where", "name='python.exe'",
         "get", "ProcessId,CommandLine", "/format:csv"],
        capture_output=True, text=True,
    ).stdout
    found = []
    for line in out.splitlines():
        if not any(p in line for p in PATTERNS):
            continue
        match = re.search(r",(\d+)\s*$", line.strip())
        if match:
            found.append((int(match.group(1)), line.strip()))
    return found


def _posix_matches() -> list[tuple[int, str]]:
    out = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True).stdout
    found = []
    for line in out.splitlines():
        line = line.strip()
        if not any(p in line for p in PATTERNS):
            continue
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[0].isdigit():
            found.append((int(parts[0]), parts[1]))
    return found


def kill_pid_tree(pid: int) -> bool:
    try:
        if os.name == "posix":
            os.killpg(os.getpgid(pid), 9)
        else:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
        return True
    except OSError:
        return False


def kill_by_name(name: str) -> int:
    """Kill every process running `name`. Returns how many were killed."""
    if os.name == "posix":
        result = subprocess.run(["pkill", "-9", "-f", name], capture_output=True)
        return 1 if result.returncode == 0 else 0
    result = subprocess.run(
        ["taskkill", "/F", "/IM", f"{name}.exe"], capture_output=True, text=True
    )
    return result.stdout.count("SUCCESS")


def port_busy(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex((host, port)) == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Kill leftover flnode/Flower processes")
    parser.add_argument("--dry-run", action="store_true", help="List, do not kill")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    matches = _posix_matches() if os.name == "posix" else _windows_matches()

    if args.dry_run:
        if not matches:
            print("No flnode processes running.")
        for pid, cmd in matches:
            print(f"  would kill {pid}: {cmd[:110]}")
        print(f"  would also kill any of: {', '.join(EXECUTABLES)}")
        return 0

    killed = 0
    for pid, cmd in matches:
        if kill_pid_tree(pid):
            killed += 1
            if not args.quiet:
                print(f"  killed {pid}: {cmd[:100]}")

    children = 0
    for name in EXECUTABLES:
        children += kill_by_name(name)

    if not args.quiet:
        print(f"Killed {killed} node process(es) and {children} Flower process(es).")
        busy = [p for p in CHECK_PORTS if port_busy(p)]
        if busy:
            print(f"WARNING: still listening: {busy}")
            print("Something outside this project may be holding them.")
        else:
            print("All project ports are free.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
