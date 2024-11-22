# Controlled Self-Recovery of the Aggregator in Federated Learning Using RAFT Protocol

Federated Learning (FL) has emerged as a decentralised machine learning paradigm for distributed systems, particularly in edge and IoT environments. However, achieving fault tolerance and self-recovery in such scenarios is challenging due to the centralised model aggregation, which poses a single point of failure. This paper focuses on the self-recovery of the aggregator, specifically the controlled re-assignment of the aggregator role to the most suitable node. Our proposed solution leverages the RAFT consensus algorithm to facilitate consistent state replication and leader election within the FL system. This is complemented by controlled aggregator re-assignment, which considers various contextual properties to select the optimal node, enhancing the system's robustness, especially in dynamic and unreliable cyber-physical environments. We implement a proof of concept using the Flower FL framework and conduct experiments to evaluate aggregator recovery time and the traffic overhead associated with state replication. While the traffic overhead scales with the number of FL nodes, our results demonstrate a resilient, self-recovering system capable of handling node failures while maintaining model consistency.

Our proof of concept extends the [quick-start tutorial](https://flower.dev/docs/framework/tutorial-quickstart-pytorch.html) on how to use Flower together with PyTorch. The experiments involve training a convolutional neural network (CNN) on the CIFAR-10 dataset – a widely recognised benchmark dataset in ML for image classification – in a federated setup. The full dataset consists of 60,000 32x32 colour images, divided into 10 classes with 6,000 images per class.

## Running the code

### Pre-requisites

A Python3 installation is required to run the code. Required libraries canbe installed via the following command:

```shell
pip install -r requirements.txt
```

### Using the convenience script (1 aggregator + 3 worker nodes)

Work in progress...

### Starting nodes separately

For running the code as three separate Python process in different terminals (emulating 3 separate network nodes):

```shell
python main.py -a self -r 3 -s localhost:5000 -n localhost:5001 localhost:5002
```

```shell
python main.py -a localhost:5000 -s localhost:5001 -n localhost:5000 localhost:5002
```

```shell
python main.py -a localhost:5000 -s localhost:5002 -n localhost:5000 localhost:5001
```
Start the three processes one by one. The three nodes will first establish a RAFT cluster with an elected leader. Then, Node-1 will act as the Flower server, while Node-2 and Node-3 will start as clients. In parallel to this, the nodes will measure latency towards each other using simple ping. Global FL progress and latency measurements are replicated using RAFT across all node.

Half-way through the execution kill Node-1 and see how the remaining two nodes will react. They will detect the absence of the aggregator and then proceed with selecting a replacement among themselves based on the mean network measurements. Depending on these, either Node-2 or Node-3 will step forward as the new aggregator and will complete the remaining FL rounds with a single worker.

### Measuring network traffic

The traffic overheads associated with the state replication can be measured using the (iftop)[https://pdw.ex-parrot.com/iftop/] utility on each of the federated nodes. For example, the following commands assume that federated nodes are inter-connected via Ethernet will measure: the native Flower traffic on port 8080, the Raft-related overheads on port 5000, and the total TCP traffic:

```shell
sudo iftop -i eth0 -f "tcp port 8080"
sudo iftop -i eth0 -f "tcp port 5000"
sudo iftop -i eth0 -f "tcp"
```

Some additinal flags and arguments can be added to control the results. See more details [here](https://man.freebsd.org/cgi/man.cgi?query=iftop).


