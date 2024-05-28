from flwr.server.client_manager import ClientManager, SimpleClientManager
import threading
from typing import Dict, List, Optional
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion

class LatencyAwareClientManager(SimpleClientManager):
    """Provides a pool of available clients."""

    def __init__(
        self,        
        la_criterion: Criterion
    ) -> None:
        """
        Parameters
        ----------        
        la_criterion : Criterion
            Criterion to select clients based on latency
        """
        super().__init__()        
        self.la_criterion = la_criterion

    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
    ) -> List[ClientProxy]:
        """Sample a number of Flower ClientProxy instances."""
        # Block until at least num_clients are connected.
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)
        # Sample clients which meet the criterion
        available_cids = list(self.clients)
        if criterion is not None:
            available_cids = [
                cid for cid in available_cids if criterion.select(self.clients[cid])
            ]

        if num_clients > len(available_cids):
            log(
                INFO,
                "Sampling failed: number of available clients"
                " (%s) is less than number of requested clients (%s).",
                len(available_cids),
                num_clients,
            )
            return []

        sampled_cids = random.sample(available_cids, num_clients)
        return [self.clients[cid] for cid in sampled_cids]