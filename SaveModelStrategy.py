from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

import flwr as fl
import numpy as np

from flwr.common import (
    # EvaluateIns,
    # EvaluateRes,
    # FitIns,
    FitRes,
    # MetricsAggregationFn,
    # NDArrays,
    Parameters,
    Scalar,
    # ndarrays_to_parameters,
    # parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

class SaveModelStrategy(fl.server.strategy.FedAvg):
    
    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        ## TODO: write an read parameters
        ##net = Net()
        ##ndarrays = get_parameters(net)
        ##return fl.common.ndarrays_to_parameters(ndarrays)
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Save aggregated_ndarrays
            print(f"Saving round {server_round} aggregated_ndarrays...")
            np.savez(f"round-{server_round}-weights.npz", *aggregated_ndarrays)

        return aggregated_parameters, aggregated_metrics

# Create strategy and run server
# strategy = SaveModelStrategy(
    # (same arguments as FedAvg here)
#)
#fl.server.start_server(strategy=strategy)