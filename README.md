# Raft Protocol for Fault Tolerance and Self-Recovery in Federated Learning

While Federated Learning (FL) is designed to be decentralised in terms of data storage and privacy, the central aggregator plays a crucial role in coordinating the model updates across different devices or servers. If the central aggregator becomes unavailable or compromised, it can disrupt the entire FL process, making it a single point of failure for the whole system. To this end, ensuring fault tolerance and self-recovery in such environments remains a critical challenge.

State replication can enable every node in the federated system to maintain a consistent and up-to-date information about the training progress making it suitable to become the new aggregator and continue from the latest stored state in the event of failures or disruptions. In particular, the Raft consensus protocol is renowned for its simplicity and effectiveness, and can be employed for replicating the state of the FL system across all participating nodes. Our proposed approach leverages Raft's leader election and log replication mechanisms to enhance fault tolerance and enable self-recovery in the event of aggregator failures.

The proof of concept was built on top of the Flower framework enhanced with PySyncObj used for aggregator election and state replication. Normally, Flower requires the developers to implement at least two separate Python classes - i.e., one for the server, and the other for the clients. However, in order to replace a faulty aggregator, a worker node must not only be able to recover the state, but also be equipped with the required aggregator execution logic. In other words, in our implementation all FL nodes execute exactly the same code base, and, depending on the election outcome, proceed with either the aggregator or the worker role. This logic is depicted in the two diagrams below:

![Algorithm](https://github.com/SINTEF-9012/raft_flower/blob/master/img/algo.png?raw=true)

![Sequence](https://github.com/SINTEF-9012/raft_flower/blob/master/img/sequence.png?raw=true)

Our proof of concept extends the [quick-start tutorial](https://flower.dev/docs/framework/tutorial-quickstart-pytorch.html) on how to use Flower together with PyTorch. The experiments involve training a convolutional neural network (CNN) on the CIFAR-10 dataset – a widely recognised benchmark dataset in ML for image classification – in a federated setup. The full dataset consists of 60,000 32x32 colour images, divided into 10 classes with 6,000 images per class.

## Running the code

### Pre-requisites

A Python3 installation is required to run the code. Required libraries canbe installed via the following command:

```shell
pip install -r requirements.txt
```

### Using the convenience script (1 aggregator + 3 worker nodes)

This script below will start four identical Python processes on localhost, which will eventually go through the election procedure to select an aggregator, while the other three will become worker nodes.

```shell
run.sh
```

### Starting nodes separately

For running the code on different network nodes, it is required to run the following command on each of the nodes

```shell
python main.py HOST_1:5000 HOST_2:5000 HOST_3:5000 HOST_4:5000
```
HOST_1 is the address of the current node, whereas HOST_2, HOST_3, and HOST_4 are the addresses of other network nodes. Accordingly, for HOST_2 the command will be the following:

```shell
python main.py HOST_2:5000 HOST_1:5000 HOST_3:5000 HOST_4:5000
```
Similar adjustments need to be made for HOST_3 and HOST_4. Port 5000 can be changed to any other un-occupied network port.

### Configuring the number of nodes and trinaing rounds

Change this value to increase the number of training rounds:

```shell
num_rounds = 1
```

Change these values to increase the number of federated nodes:

```shell
min_fit_clients=3, min_evaluate_clients=3, min_available_clients=3
```

**NB: Do not forget to start the corresonding number of nodes!**

### Measuring network traffic

The traffic overheads associated with the state replication can be measured using the (iftop)[https://pdw.ex-parrot.com/iftop/] utility on each of the federated nodes. For example, the following commands assume that federated nodes are inter-connected via Ethernet will measure: the native Flower traffic on port 8080, the Raft-related overheads on port 5000, and the total TCP traffic:

```shell
sudo iftop -i eth0 -f "tcp port 8080"
sudo iftop -i eth0 -f "tcp port 5000"
sudo iftop -i eth0 -f "tcp"
```

Some additinal flags and arguments can be added to control the results. See more details [here](https://man.freebsd.org/cgi/man.cgi?query=iftop).


