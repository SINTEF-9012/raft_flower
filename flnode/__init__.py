"""Controlled self-recovery of the FL aggregator using the RAFT protocol.

Every node in the cluster runs the same program (`flnode.node`); the role it
takes -- aggregator or worker -- follows from a deterministic ranking of
replicated contextual properties, exactly as the accompanying article
describes. See README.md for the architecture and how it maps onto the paper.
"""

__version__ = "2.0.0"
