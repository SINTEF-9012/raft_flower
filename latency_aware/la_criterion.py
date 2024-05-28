from pysyncobj.batteries import ReplDict
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion

class LatencyAwareCriterion(Criterion):
    """Provides a criterion for selecting clients based on latency."""

    def __init__(
        self,        
        replicated_state: ReplDict,
        #self_address: str
    ) -> None:
        """
        Parameters
        ----------        
        replicated_state : ReplDict
            The replicated state of the federated system containing latency of all nodes
        self_address: str
            The current host/aggregator
        """
        super().__init__()        
        self.replicated_state = replicated_state
        #self.self_address = self_address

    def select(self, client: ClientProxy):
        """Decide whether a client should be eligible for sampling or not."""
        
        print("client.cid", client.cid)
        client_ip = client.cid.split(':')[1]
        if hasattr(client, 'node_id'):    
            print("client.node_id", client.node_id)
        #for key in self.replicated_state.keys():
        #    if self.self_address in key:
        #        print(key + " - " + str(self.replicated_state.get(key, None)))
        return True

    