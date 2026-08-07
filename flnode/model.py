"""The CIFAR-10 CNN, training and evaluation loops.

Unchanged in substance from Flower's PyTorch quickstart, which is what the
article extends. As Section 7.2.1 notes, the recovery mechanism "does not
affect the internal ML training logic or data, which means the model accuracy
and loss remained the same as in the original baseline implementation" -- so
this file exists to remove duplication, not to change behaviour.

The original repository carried three near-identical copies of this code
(`main.py`, `latency_aware/la_main.py`, `latency_aware/la_client.py`).

Known divergence from the article, left in place deliberately: Section 7.1
states that each client "processes a local partition of the CIFAR-10 dataset",
but `load_data` gives every node the full, identical training and test set, as
the original did. Partitioning was out of scope for this branch; see
`partition` below for the hook, currently unused.
"""

from __future__ import annotations

import warnings
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

warnings.filterwarnings("ignore", category=UserWarning)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    """Simple CNN adapted from 'PyTorch: A 60 Minute Blitz'."""

    def __init__(self) -> None:
        super().__init__()
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


def train(net: Net, trainloader: DataLoader, epochs: int, device=DEVICE) -> float:
    """Train locally. Returns mean loss per batch over the whole run."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    net.to(device)
    net.train()
    total, batches = 0.0, 0
    for _ in range(epochs):
        for images, labels in trainloader:
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            batches += 1
    return total / max(1, batches)


def test(net: Net, testloader: DataLoader, device=DEVICE) -> tuple[float, float]:
    """Evaluate on the local test set. Returns `(loss, accuracy)`."""
    criterion = torch.nn.CrossEntropyLoss()
    net.to(device)
    net.eval()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = net(images.to(device))
            labels = labels.to(device)
            loss += float(criterion(outputs, labels).item())
            correct += int((torch.max(outputs.data, 1)[1] == labels).sum().item())
    return loss, correct / len(testloader.dataset)


def partition(dataset, index: int, count: int) -> Subset:
    """Contiguous shard `index` of `count`. Currently unused -- see module note."""
    if count <= 1:
        return Subset(dataset, range(len(dataset)))
    per = len(dataset) // count
    start = index * per
    end = len(dataset) if index == count - 1 else start + per
    return Subset(dataset, range(start, end))


def load_data(
    root: str = "./data",
    batch_size: int = 32,
    *,
    partition_id: int = 0,
    num_partitions: int = 1,
) -> tuple[DataLoader, DataLoader]:
    """Load CIFAR-10.

    With the default `num_partitions=1` every node gets the full dataset, which
    is the behaviour the published experiments used.
    """
    transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CIFAR10(root, train=True, download=True, transform=transform)
    testset = CIFAR10(root, train=False, download=True, transform=transform)
    if num_partitions > 1:
        trainset = partition(trainset, partition_id, num_partitions)
        testset = partition(testset, partition_id, num_partitions)
    return (
        DataLoader(trainset, batch_size=batch_size, shuffle=True),
        DataLoader(testset, batch_size=batch_size),
    )


# --- Conversion between the model and Flower's ArrayRecord ------------------


def get_weights(net: Net) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(net.state_dict())


def set_weights(net: Net, state_dict) -> None:
    net.load_state_dict(
        OrderedDict({k: torch.as_tensor(v) for k, v in state_dict.items()}),
        strict=True,
    )
