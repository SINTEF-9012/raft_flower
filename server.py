from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics
from raft_strategy import RaftStrategy
from pysyncobj import SyncObj, SyncObjConf
from pysyncobj.batteries import ReplDict
import time

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

replicated_state = ReplDict()
config=SyncObjConf(appendEntriesUseBatch = False, appendEntriesBatchSizeBytes = 2**20, logCompactionBatchSize = 2**20, recvBufferSize = 2**20, sendBufferSize = 2**20)
sync_obj = SyncObj('localhost:5000', ['localhost:5001', 'localhost:5002', 'localhost:5003'], consumers=[replicated_state], conf=config)
num_rounds=3
replicated_state.set('num_rounds', num_rounds, sync=True)

while not sync_obj.isReady():
    print("RAFT cluster is not ready...")
    time.sleep(3)

# Define strategy
strategy = RaftStrategy(evaluate_metrics_aggregation_fn=weighted_average, replicated_state=replicated_state, sync_obj=sync_obj, min_fit_clients=3, min_evaluate_clients=3, min_available_clients=3)

rounds_left = replicated_state.get('num_rounds', 0) - replicated_state.get('configure_fit_round', 0)

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=rounds_left),
    strategy=strategy,
)
