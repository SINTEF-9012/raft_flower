# Legacy artifacts (SEAMS'24 / IFIP Networking'24)

This directory holds the original prototypes **exactly as they were when the
papers below were written**. They are kept unchanged so that the source-code
links printed in those papers keep resolving to the code they actually describe.

Do not develop here. The maintained implementation lives in `../flnode/`.

| File | Paper |
|---|---|
| `main.py`, `raft_strategy.py`, `run.sh` | Dautov & Husom, *RAFT Protocol for Fault Tolerance and Self-Recovery in Federated Learning*, SEAMS'24 |
| `latency_aware/` | Dautov & Husom, *Latency-Aware Node Selection in Federated Learning*, IFIP Networking'24 |

## Running these

They require the **legacy Flower API** (`start_server` / `start_numpy_client`),
which was removed from Flower after 1.14. They will not run against a current
`flwr` release.

```bash
python3.9 -m venv .venv-legacy && . .venv-legacy/bin/activate
pip install -r requirements.txt
./run.sh
```

`requirements.txt` here pins `flwr==1.9.0` — the original pin was
`flwr>=1.0, <2.0`, which today resolves to a version where the entry points
these scripts call no longer exist. 1.9.0 is the version that satisfies both
`main.py` (`start_server` / `start_numpy_client`) and
`latency_aware/la_main.py` (`start_client` + `NumPyClient.to_client()`,
added in 1.7).

## Known limitations (why this was rewritten)

These are recorded here because they are the reason the `unified`
implementation exists, and because two of them affect how the published
results should be read.

- **`main.py`** conflates the RAFT leader with the FL aggregator
  (`main.py:171` uses `sync_obj._isLeader()` to decide the role). The TAAS
  article treats these as deliberately separate roles (§6.2.1).
- **`latency_aware/la_main.py`** never consults RAFT leadership at all, and
  performs aggregator re-assignment *only* inside the aggregator's own branch
  after a round completes successfully (`la_main.py:218-230`). If the
  aggregator crashes, nothing executes that code, so this variant cannot
  recover from aggregator failure — it implements the article's §8.1 future
  work (proactive, fault-free re-assignment) rather than its main contribution.
- **`main.py:175`** computes `rounds_left` from `configure_fit_round`, which is
  the restarted server's *local* round index and resets to 1. This is correct
  on the first recovery and wrong on every subsequent one, so chained
  aggregator failures (§7.1.1) are not actually supported.
- **`la_main.py:202`** reuses the loop variable `i` from the enclosing round
  loop at `la_main.py:195`, clobbering the round counter.
- **`la_main.py:213`** binds the Flower server to the *RAFT* address, so FL
  traffic and RAFT traffic share port 5000. The traffic separation the article
  relies on in §7.2.2 (port 8080 = FL, port 5000 = RAFT) does not hold in this
  code, and latency probes (`la_main.py:185`) also target port 5000.
- **Four tensor families** (`initialize_parameters_*`, `configure_fit_*`,
  `configure_evaluate_*`, `aggregate_fit_*`) are replicated every round, one
  RAFT entry per tensor, but only `configure_fit_*` is ever read back during
  recovery.
- **`la_criterion.py:35`** — `select()` returns `True` unconditionally; the
  class is never instantiated and never passed to `client_manager.sample()`.
  Latency-aware *client* selection (IFIP §II-A) is therefore not wired in.
- **No data partitioning** — `load_data()` gives every node the full, identical
  CIFAR-10 train and test set.
- State is in-memory only; no `journalFile` / `fullDumpFile` is configured, so
  nothing survives a whole-cluster restart. (The SEAMS paper states this
  explicitly in §6.2.1.)

Deleted on the `unified` branch, as dead code with no call sites:
`latency_aware/la_client.py` (hard-coded IP `172.17.192.1`, entire main loop
commented out inside a string literal) and `latency_aware/la_script.py`
(11-line scratch file).
