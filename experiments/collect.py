"""Turn the per-node event logs into the article's recovery-time breakdown.

Rebuilds the quantities plotted in Figures 6 and 8 from the JSONL traces, so
those figures can be regenerated rather than transcribed by hand.

For each recovery episode, on each surviving node:

    t0  aggregator_down_detected
    t1  raft_leader_elected        RAFT re-election    = t1 - t0
    t2  ranking_computed           re-assignment       = t2 - t1
    t3  state_restored             checkpoint restore  = t3 - t2
    t4  aggregator_serving         total recovery      = t4 - t0

t0 is observed on the surviving workers and t4 on whichever node took over, so
the total is measured across nodes and uses wall-clock rather than monotonic
timestamps. Keep the clocks reasonably synchronised (NTP) on a real testbed.

    python experiments/collect.py --results-dir results
    python experiments/collect.py --results-dir results --csv recovery.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any


def load_events(results_dir: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("events-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    events.sort(key=lambda e: e.get("ts", 0.0))
    return events


def first_after(events, name, after, node=None):
    for event in events:
        if event.get("event") != name:
            continue
        if event.get("ts", 0.0) < after:
            continue
        if node is not None and event.get("node") != node:
            continue
        return event
    return None


def episodes(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One record per aggregator failure.

    Every surviving node detects the same failure independently, and they do
    so at slightly different moments (a worker sitting idle notices as soon as
    RAFT drops the partner; one mid-round notices when its loop next comes
    round). All detections naming the same failed node belong to one episode
    until a new aggregator actually takes over, so episodes are delimited by
    the following `aggregator_serving`, not by a fixed time window.
    """
    out = []
    downs = [e for e in events if e.get("event") == "aggregator_down_detected"]

    grouped: list[list[dict]] = []
    for event in downs:
        if grouped:
            previous = grouped[-1][0]
            same_target = event.get("failed") == previous.get("failed")
            # A new aggregator taking over closes the previous episode.
            takeover = first_after(
                events, "aggregator_serving", previous["ts"]
            )
            still_open = takeover is None or event["ts"] <= takeover["ts"]
            if same_target and still_open:
                grouped[-1].append(event)
                continue
        grouped.append([event])

    for group in grouped:
        t0 = min(e["ts"] for e in group)
        failed = group[0].get("failed")
        detectors = sorted({e["node"] for e in group})

        elected = first_after(events, "raft_leader_elected", t0)
        t1 = elected["ts"] if elected else None

        # Only ranking decisions taken *after* the failure count, and only
        # those that actually ranked -- entries where a live incumbent simply
        # kept the role are not re-assignment decisions.
        ranked = next(
            (
                e for e in events
                if e.get("event") == "ranking_computed"
                and e.get("reason") == "ranked"
                and e.get("ts", 0) >= (t1 or t0)
            ),
            None,
        )
        t2 = ranked["ts"] if ranked else None

        serving = first_after(events, "aggregator_serving", t0)
        t3 = serving["ts"] if serving else None
        new_aggregator = serving.get("node") if serving else None

        # Checkpoint restore and readiness happen in the ServerApp process,
        # after SuperNodes reconnect -- reported separately so client
        # reconnection is not charged to the recovery interval.
        restored = first_after(events, "state_restored", t3 or t0)
        ready = first_after(events, "serverapp_ready", t3 or t0)

        record = {
            "failed_aggregator": failed,
            "new_aggregator": new_aggregator,
            "detected_by": len(detectors),
            "t0_detected": t0,
            "raft_election_s": (t1 - t0) if t1 else None,
            "reassignment_s": (t2 - t1) if (t2 and t1) else None,
            # The quantity Figure 6 reports: RAFT re-election plus controlled
            # re-assignment, ending when the new aggregator takes the role.
            "total_recovery_s": (t3 - t0) if t3 else None,
            "restore_s": restored.get("duration_s") if restored else None,
            "serving_after_s": (ready["ts"] - t0) if ready else None,
            "resumed_from_round": ready.get("resumed_from_round") if ready else None,
            "raft_leader": elected.get("leader") if elected else None,
            "leader_is_aggregator": (
                elected.get("leader") == new_aggregator if elected else None
            ),
        }

        # Did every node independently reach the same answer? This is the
        # Section 5.4 determinism claim, observed at run time rather than
        # assumed.
        #
        # Compared on each node's *final* decision in the episode, not on every
        # intermediate one. A node that ranks in the first moments after a
        # crash may still see the dead node as a candidate and pick it, then
        # correct itself once RAFT reports the partner gone. That convergence
        # is correct behaviour; only a lasting disagreement would not be.
        window_end = (t3 if t3 else t0 + 60) + 5
        final_choice: dict[str, str] = {}
        for e in events:
            if (
                e.get("event") == "ranking_computed"
                and e.get("reason") == "ranked"
                and t0 <= e.get("ts", 0) <= window_end
                and e.get("chosen")
            ):
                final_choice[e["node"]] = e["chosen"]

        choices = set(final_choice.values())
        record["unanimous_choice"] = len(choices) <= 1
        record["choices_seen"] = sorted(choices)
        record["nodes_deciding"] = len(final_choice)
        out.append(record)
    return out


def summarise(events: list[dict[str, Any]]) -> dict[str, Any]:
    rounds = [e for e in events if e.get("event") == "round_completed"]
    ckpts = [e for e in events if e.get("event") == "checkpoint_written"]
    ctxs = [e for e in events if e.get("event") == "context_written"]
    return {
        "rounds_completed": len(rounds),
        "mean_round_s": statistics.mean([r["duration_s"] for r in rounds if r.get("duration_s")])
        if any(r.get("duration_s") for r in rounds) else None,
        "checkpoints": len(ckpts),
        "mean_checkpoint_bytes": statistics.mean([c["bytes"] for c in ckpts]) if ckpts else None,
        "mean_checkpoint_s": statistics.mean([c["duration_s"] for c in ckpts if c.get("duration_s")])
        if any(c.get("duration_s") for c in ckpts) else None,
        "context_writes": len(ctxs),
        "mean_context_bytes": statistics.mean([c["bytes"] for c in ctxs]) if ctxs else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarise flnode event logs")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--csv", default=None, help="Write per-episode rows to this CSV")
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = Path(__file__).resolve().parent.parent / results_dir

    events = load_events(results_dir)
    if not events:
        print(f"No events found in {results_dir}")
        return 1

    nodes = sorted({e.get("node") for e in events if e.get("node")})
    print(f"{len(events)} events from {len(nodes)} node logs in {results_dir}\n")

    stats = summarise(events)
    print("Training")
    print(f"  rounds completed        {stats['rounds_completed']}")
    if stats["mean_round_s"]:
        print(f"  mean round time         {stats['mean_round_s']:.1f} s")
    print(f"  checkpoints written     {stats['checkpoints']}")
    if stats["mean_checkpoint_bytes"]:
        print(f"  mean checkpoint size    {stats['mean_checkpoint_bytes']:,.0f} bytes")
    if stats["mean_checkpoint_s"]:
        print(f"  mean replication time   {stats['mean_checkpoint_s'] * 1000:.0f} ms")
    if stats["mean_context_bytes"]:
        print(f"  mean context record     {stats['mean_context_bytes']:,.0f} bytes"
              f"  ({stats['context_writes']} writes)")

    eps = episodes(events)
    print(f"\nRecovery episodes: {len(eps)}")
    if not eps:
        print("  (no aggregator failures recorded)")

    for i, ep in enumerate(eps, 1):
        print(f"\n  [{i}] {ep['failed_aggregator']} -> {ep['new_aggregator']}")
        print(f"      detected by         {ep['detected_by']} node(s)")
        for label, key in [
            ("RAFT re-election", "raft_election_s"),
            ("re-assignment", "reassignment_s"),
            ("TOTAL recovery", "total_recovery_s"),
            ("checkpoint restore", "restore_s"),
            ("serving after", "serving_after_s"),
        ]:
            value = ep[key]
            print(f"      {label:<19} {value:.3f} s" if value is not None
                  else f"      {label:<19} n/a")
        print(f"      resumed from round  {ep['resumed_from_round']}")
        print(f"      RAFT leader         {ep['raft_leader']}"
              f"  (leader is aggregator: {ep['leader_is_aggregator']})")
        print(f"      unanimous ranking   {ep['unanimous_choice']} {ep['choices_seen']}")

    if args.csv and eps:
        out = Path(args.csv)
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(eps[0]))
            writer.writeheader()
            for ep in eps:
                writer.writerow(ep)
        print(f"\nWrote {len(eps)} rows to {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
