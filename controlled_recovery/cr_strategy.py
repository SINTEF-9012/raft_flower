# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""This is a Raft-enhanced extension to Federated Averaging (FedAvg) strategy described in [McMahan et al., 2016].

Paper: arxiv.org/abs/1602.05629
"""


from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.strategy import Strategy

from pysyncobj import SyncObj, SyncObjConf
from pysyncobj.batteries import ReplDict

from colorama import Fore
from colorama import init

import objsize
import time

## from replicated_state import ReplicatedState

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""
class RaftStrategy(Strategy):
    """Configurable Raft strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes, line-too-long
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        replicated_state: Optional[ReplDict] = None
    ) -> None:
        """Federated Averaging strategy.

        Implementation based on https://arxiv.org/abs/1602.05629

        Parameters
        ----------
        fraction_fit : float, optional
            Fraction of clients used during training. In case `min_fit_clients`
            is larger than `fraction_fit * available_clients`, `min_fit_clients`
            will still be sampled. Defaults to 1.0.
        fraction_evaluate : float, optional
            Fraction of clients used during validation. In case `min_evaluate_clients`
            is larger than `fraction_evaluate * available_clients`,
            `min_evaluate_clients` will still be sampled. Defaults to 1.0.
        min_fit_clients : int, optional
            Minimum number of clients used during training. Defaults to 2.
        min_evaluate_clients : int, optional
            Minimum number of clients used during validation. Defaults to 2.
        min_available_clients : int, optional
            Minimum number of total clients in the system. Defaults to 2.
        evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]]
            Optional function used for validation. Defaults to None.
        on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure training. Defaults to None.
        on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure validation. Defaults to None.
        accept_failures : bool, optional
            Whether or not accept rounds containing failures. Defaults to True.
        initial_parameters : Parameters, optional
            Initial global model parameters.
        fit_metrics_aggregation_fn : Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        evaluate_metrics_aggregation_fn : Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        replicated_state : Optional[ReplDict]
            Raft-based replicated dictionary for storing the state, optional
        """
        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.replicated_state = replicated_state

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"RaftStrategy(accept_failures={self.accept_failures})"
        return rep

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory

        ########################################

        #TODO: uncomment this code if you also need to store and replicate the intial parameters
        if (initial_parameters):
           i = 0
           for tensor in initial_parameters.tensors:
               #print("initialize_parameters_tensor" + str(i), objsize.get_deep_size(tensor))
               self.replicated_state.set("initialize_parameters_tensor" + str(i), tensor, sync=True)
               i = i + 1
           #print("initialize_parameters_tensor_type", initial_parameters.tensor_type)        
           self.replicated_state.set("initialize_parameters_tensor_type", initial_parameters.tensor_type, sync=True)
           #print("initialize_parameters_tensor_length", len(initial_parameters.tensors))
           self.replicated_state.set("initialize_parameters_tensor_length", len(initial_parameters.tensors), sync=True)

        ########################################

        return initial_parameters

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res

        ########################################

        i = 0
        for tensor in parameters.tensors:
            #print("evaluate_tensor" + str(i), objsize.get_deep_size(tensor))
            self.replicated_state.set("evaluate_tensor" + str(i), tensor, sync=True)
            i = i + 1        
        #print("evaluate_tensor_type", parameters.tensor_type)        
        self.replicated_state.set("evaluate_tensor_type", parameters.tensor_type, sync=True)        
        #print("evaluate_tensor_length", len(parameters.tensors))
        self.replicated_state.set("evaluate_tensor_length", len(parameters.tensors), sync=True)
        #print("evaluate_round", server_round)
        self.replicated_state.set("evaluate_round", server_round, sync=True)

        ########################################

        return loss, metrics

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        ########################################

        i = 0
        for tensor in parameters.tensors:
            ##print("configure_fit_tensor" + str(i), objsize.get_deep_size(tensor), hash(tensor))
            self.replicated_state.set("configure_fit_tensor" + str(i), tensor, sync=True)
            i = i + 1        
        ##print("configure_fit_tensor_type", parameters.tensor_type)        
        self.replicated_state.set("configure_fit_tensor_type", parameters.tensor_type, sync=True)        
        ##print("configure_fit_tensor_length", len(parameters.tensors))
        self.replicated_state.set("configure_fit_tensor_length", len(parameters.tensors), sync=True)
        ##print("configure_fit_round", server_round)
        self.replicated_state.set("configure_fit_round", server_round, sync=True)
        #print("Fit parameters successfully saved an replicated!", "Starting next round: " + str(server_round))

        ########################################

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        ########################################

        #TODO: uncomment this code if you also need to store and replicate the evaluation parameters
        i = 0
        for tensor in parameters.tensors:
            #print("configure_evaluate_tensor" + str(i), objsize.get_deep_size(tensor), hash(tensor))
            self.replicated_state.set("configure_evaluate_tensor" + str(i), tensor, sync=True)
            i = i + 1        
        #print("configure_evaluate_tensor_type", parameters.tensor_type)        
        self.replicated_state.set("configure_evaluate_tensor_type", parameters.tensor_type, sync=True)        
        #print("configure_evaluate_tensor_length", len(parameters.tensors))
        self.replicated_state.set("configure_evaluate_tensor_length", len(parameters.tensors), sync=True)
        #print("configure_evaluate_round", server_round)
        self.replicated_state.set("configure_evaluate_round", server_round, sync=True)
        #print("Evaluate parameters successfully saved an replicated!", "Starting next round: " + str(server_round))

        ########################################

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        ########################################

        i = 0
        for tensor in parameters_aggregated.tensors:
            #print("aggregate_fit_tensor" + str(i), objsize.get_deep_size(tensor), hash(tensor))
            self.replicated_state.set("aggregate_fit_tensor" + str(i), tensor, sync=True)
            i = i + 1        
        #print("configure_evaluate_tensor_type", parameters.tensor_type)        
        self.replicated_state.set("aggregate_fit_tensor_type", parameters_aggregated.tensor_type, sync=True)        
        #print("configure_evaluate_tensor_length", len(parameters.tensors))
        self.replicated_state.set("aggregate_fit_tensor_length", len(parameters_aggregated.tensors), sync=True)
        #print("configure_evaluate_round", server_round)
        self.replicated_state.set("aggregate_fit_round", server_round, sync=True)
        #print("Evaluate parameters successfully saved an replicated!", "Starting next round: " + str(server_round))

        ########################################

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")
        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        print(Fore.RED + "server round: ", server_round)
        print(Fore.RED + "global num_rounds", self.replicated_state.get('num_rounds'))

        num_rounds=int(self.replicated_state.get('num_rounds')) - server_round
        self.replicated_state.set('num_rounds', num_rounds)

        return loss_aggregated, metrics_aggregated