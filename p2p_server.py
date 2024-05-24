from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics

from p2p_node import MyOwnPeer2PeerNode
import time
from p2p_strategy import P2PStrategy

###############################################################################

node = MyOwnPeer2PeerNode("127.0.0.1", 10001)
time.sleep(5)

# Do not forget to start your node!
node.start()
time.sleep(1)

# Connect with another node, otherwise you do not create any network!
node.connect_with_node('127.0.0.1', 10002)
node.connect_with_node('127.0.0.1', 10003)
time.sleep(2)

# Example of sending a message to the nodes (dict).
# node.send_to_nodes({"message": "Hi there!"})

#time.sleep(5) # Create here your main loop of the application

#node.stop()

###############################################################################


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


# Define strategy
strategy = P2PStrategy(evaluate_metrics_aggregation_fn=weighted_average, min_available_clients=3, min_evaluate_clients=3, min_fit_clients=3, p2p_node=node)

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=1),
    strategy=strategy,
)

node.stop()