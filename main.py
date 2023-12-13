from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics
from raft_strategy import RaftStrategy
from pysyncobj import SyncObj, SyncObjConf
from pysyncobj.batteries import ReplDict
import time, sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
import warnings
from collections import OrderedDict

from flwr.common import Parameters

# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def train(net, trainloader, epochs):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for images, labels in tqdm(trainloader):
            optimizer.zero_grad()
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()


def test(net, testloader):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            outputs = net(images.to(DEVICE))
            labels = labels.to(DEVICE)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


def load_data():
    """Load CIFAR-10 (training and test set)."""
    trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CIFAR10("./data", train=True, download=True, transform=trf)
    testset = CIFAR10("./data", train=False, download=True, transform=trf)
    return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Load model and data (simple CNN, CIFAR-10)
net = Net().to(DEVICE)
trainloader, testloader = load_data()


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, epochs=1)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

if __name__ == '__main__':

    if len(sys.argv) < 3:
        print('Usage: %s selfHost:port partner1Host:port partner2Host:port ...')
        sys.exit(-1)

    selfAddr = sys.argv[1]
    if selfAddr == 'readonly':
        selfAddr = None
    partners = sys.argv[2:]

    print(selfAddr)
    print(partners)

    replicated_state = ReplDict()
    config=SyncObjConf(appendEntriesUseBatch = False, appendEntriesBatchSizeBytes = 2**20, logCompactionBatchSize = 2**20, recvBufferSize = 2**20, sendBufferSize = 2**20)
    sync_obj = SyncObj(selfAddr, partners, consumers=[replicated_state], conf=config)
    
    ##check if this is the very first launch or a re-start (i.e. num_rounds has already been defined)
    if not replicated_state.get('num_rounds', None):
        num_rounds = 1
        replicated_state.set('num_rounds', num_rounds, sync=True)

    while True:
        print(sync_obj.isReady())
        for key in replicated_state.keys():
            print(key + " - " + str(replicated_state.get(key, None).__class__))
        time.sleep(3)     
        if sync_obj._isLeader():            
            print("I am the new leader, so I am starting the FL server")
            
            ## restore the number of remaining rounds
            rounds_left = replicated_state.get('num_rounds', 0) - max(0, (replicated_state.get('configure_fit_round', 0) - 1)) # this is needed because by default we write into the replicated state the current round, which is -1 lower the number of completed rounds 
            print("Remaining rounds:", rounds_left)
            
            ## restore the latest parameters (if any) and make them initial parameters for the re-start
            tensors = []
            for i in range(0, replicated_state.get('configure_fit_tensor_length', 0)):
                tensors.append(replicated_state.get('configure_fit_tensor' + str(i), None))
            tensor_type=replicated_state.get('configure_fit_tensor_type', None)
            initial_parameters = None if tensor_type==None else Parameters(tensors=tensors, tensor_type=tensor_type)
            print("Previously stored parameters", initial_parameters.__class__)

            # Define strategy
            strategy = RaftStrategy(initial_parameters=initial_parameters, evaluate_metrics_aggregation_fn=weighted_average, replicated_state=replicated_state, min_fit_clients=1, min_evaluate_clients=1, min_available_clients=1)

            # Start Flower server
            fl.server.start_server(
                server_address="0.0.0.0:8080",
                config=fl.server.ServerConfig(num_rounds=rounds_left),
                strategy=strategy,
            )
            break
        elif (sync_obj._getLeader() is None):
            print("There is no RAFT leader currently, let's wait a bit and re-try.")                   
        else:
            ##TODO: check if sync_obj is ready()
            print("I am a worker, so I will be doing my local training and send the updates to: ", sync_obj._getLeader().address)
            try:
                fl.client.start_numpy_client(server_address=sync_obj._getLeader().host + ":8080", client=FlowerClient())
                break
            except:
                print("Error: server disconnected. Initiating re-start...")
    
    print(sync_obj.isReady())
    for key in replicated_state.keys():
        print(key + " - " + str(replicated_state.get(key, None).__class__))