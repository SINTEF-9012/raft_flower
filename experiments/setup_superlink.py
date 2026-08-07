"""Register the local SuperLink connection Flower needs, idempotently.

As of Flower 1.26 the SuperLink connection is configured in the user-level file
`~/.flwr/config.toml`, not in the project's `pyproject.toml`. This project needs
one entry pointing at loopback, because the aggregator always submits its run to
the SuperLink on its own machine.

Run once per machine:

    python experiments/setup_superlink.py

Adds nothing if the entry is already present, and never touches other entries.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    print("Python 3.11+ required")
    raise SystemExit(1)

NAME = "flnode-local"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Register the flnode SuperLink connection")
    parser.add_argument("--control-port", type=int, default=9093)
    parser.add_argument("--name", default=NAME)
    parser.add_argument("--make-default", action="store_true")
    args = parser.parse_args(argv)

    config_path = Path.home() / ".flwr" / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    text = ""
    if config_path.exists():
        text = config_path.read_text(encoding="utf-8")
        try:
            existing = tomllib.loads(text)
        except tomllib.TOMLDecodeError as exc:
            print(f"Could not parse {config_path}: {exc}")
            return 1

    superlinks = existing.get("superlink", {})
    if args.name in superlinks:
        print(f"'{args.name}' already registered in {config_path}")
        print(f"  address = {superlinks[args.name].get('address')}")
        return 0

    block = (
        f'\n[superlink.{args.name}]\n'
        f'address = "127.0.0.1:{args.control_port}"\n'
        f"insecure = true\n"
    )
    if not text:
        text = f'[superlink]\ndefault = "{args.name}"\n'
    elif args.make_default and "default" not in superlinks:
        text = f'[superlink]\ndefault = "{args.name}"\n\n' + text

    config_path.write_text(text.rstrip("\n") + "\n" + block, encoding="utf-8")
    print(f"Registered '{args.name}' -> 127.0.0.1:{args.control_port} in {config_path}")
    print("Verify with: flwr config list")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
