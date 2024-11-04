from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics
from flwr.server.client_manager import ClientManager, SimpleClientManager
from la_strategy import LatencyAwareStrategy
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

from tcppinglib import tcpping
from threading import *
from tcppinglib import async_tcpping
import statistics

from flwr.common import Parameters
import objsize

import argparse
from colorama import Fore, Back, Style
from colorama import init
init(autoreset=True)

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

    def __init__(self, replicated_state, nodes):
        super().__init__()
        self.nodes = nodes
        self.replicated_state = replicated_state

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

    parser = argparse.ArgumentParser(description='Latency-aware node selection in Federated Learning')
    parser.add_argument('-a','--aggr', help='Specify whether this node will start as the initial aggregator ("aggr") or a client (then specifiy the address of the aggregator)', required=True)
    parser.add_argument('-r','--rounds', help='Specify how many training rounds are needed)', required=False)
    parser.add_argument('-s','--self', help='Specify own address', required=True)
    parser.add_argument('-n', '--nodes', nargs='+', default=[], help='Specify the addresses of other nodes', required=True)
    args = vars(parser.parse_args())
    
    if args['aggr'] == 'self':
        aggregator = args['self']
        num_rounds = args['rounds']        
    else:
        aggregator = args['aggr']  
    selfAddr = args['self']
    partners = args['nodes']
   
    #print(Fore.GREEN + aggregator)
    #print(Fore.GREEN + num_rounds)
    #print(Fore.GREEN + selfAddr)
    #for node in partners:
    #    print(Fore.GREEN + node)

    replicated_state = ReplDict()
    config=SyncObjConf(appendEntriesUseBatch = False, appendEntriesBatchSizeBytes = 2**20, logCompactionBatchSize = 2**20, recvBufferSize = 2**20, sendBufferSize = 2**20)
    sync_obj = SyncObj(selfAddr, partners, consumers=[replicated_state], conf=config)

    ## Wait until all nodes join pysyncobj
    while True:            
        if sync_obj.isReady():
            if args['aggr'] == 'self':
                replicated_state.set('aggregator', aggregator, sync=True)
                replicated_state.set('num_rounds', num_rounds, sync=True)
            time.sleep(1)
            break
        print(Fore.RED + "Waiting for pysyncobj...")
        time.sleep(3)    
    
    def measure_latency(nodes):
        #print(nodes)
        while True:
            latency=[]
            for node in nodes:
                #print(node)
                #print(Fore.GREEN + "Measuring latency towards: " + node)
                host=node.split(':')[0]
                port=int(node.split(':')[1])
                ping = tcpping(host, port=port, interval=1.0)
                print(Fore.GREEN + "Latency towards " + node, ping.avg_rtt)                
                replicated_state.set(selfAddr + "_" + node + "_latency", ping.avg_rtt, sync=False)
                latency.append(ping.avg_rtt)
            replicated_state.set(selfAddr + "_mean_latency", statistics.mean(latency), sync=False)
            time.sleep(10)     
   
    Thread(target=measure_latency, args=(partners,), daemon=True).start()
    
    for i in range(int(replicated_state.get('num_rounds'))):
        print(Back.RED + "Check: " + selfAddr + " and " + replicated_state.get('aggregator'))
        if selfAddr == replicated_state.get('aggregator'):
            print(Fore.GREEN + "I am the current aggregator!")
                                
            ## restore the latest parameters (if any) and make them initial parameters for the re-start
            tensors = []
            for i in range(0, replicated_state.get('configure_fit_tensor_length', 0)):
                tensors.append(replicated_state.get('configure_fit_tensor' + str(i), None))
            tensor_type=replicated_state.get('configure_fit_tensor_type', None)
            initial_parameters = None if tensor_type==None else Parameters(tensors=tensors, tensor_type=tensor_type)
            print(Fore.GREEN + "Previously stored parameters", initial_parameters.__class__)

            # Define strategy
            strategy = LatencyAwareStrategy(initial_parameters=initial_parameters, evaluate_metrics_aggregation_fn=weighted_average, replicated_state=replicated_state, min_fit_clients=2, min_evaluate_clients=2, min_available_clients=2)
                    
            # Start Flower server with a single training round
            fl.server.start_server(
                server_address=replicated_state.get('aggregator'),
                config=fl.server.ServerConfig(num_rounds=1), #(num_rounds=rounds_left),
                strategy=strategy,
            )

            mean_latency = {key: value for (key, value) in replicated_state.items() if '_mean_latency' in key}        
            #print(mean_latency)
            mean_latency_sorted = {key: value for (key, value) in sorted(mean_latency.items(), key=lambda x: x[1])}
            print(mean_latency_sorted)
            node_with_lowest_latency = (list(mean_latency_sorted)[0]).split('_')[0]
            #print(Back.RED + node_with_lowest_latency)
            if replicated_state.get('aggregator') != node_with_lowest_latency:
                print(Back.RED + 'We are re-assigning the aggregator!')
                print(Back.RED + 'Current aggregator: ' + replicated_state.get('aggregator'))
                print(Back.RED + 'New aggregator: ' + node_with_lowest_latency)
                replicated_state.set('aggregator', node_with_lowest_latency, sync=True)
            else:
                print(Back.GREEN + 'The current aggregator still has the lowest average latency!')            
        else:
            ## start as a client
            print(Fore.GREEN + "I am a client!")
            try:
                #fl.client.start_numpy_client(server_address=replicated_state.get('aggregator'), client=FlowerClient(replicated_state=replicated_state, nodes=partners))      
                fl.client.start_client(server_address=replicated_state.get('aggregator'),client=FlowerClient(replicated_state=replicated_state, nodes=partners).to_client())      
            except:
                print(Fore.RED + "Error: server disconnected. Initiating re-start...") 
        time.sleep(2)       
                
        #for key in mean_latency.keys():
            #print(key + " - " + str(mean_latency.get(key, None)))
            #mean_latency=dict()           
            #if '_mean_latency' in key:                
            #    print(Fore.GREEN + key + " - " + str(replicated_state.get(key, None)))
            #    mean_latency.append(replicated_state.get(key))
     
    for key in replicated_state.keys():
    #    ## print(key + " - " + str(replicated_state.get(key, None).__class__))
        if 'latency' in key:
            print(Fore.GREEN + key + " - " + str(replicated_state.get(key, None)))