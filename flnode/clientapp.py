"""Flower ClientApp: the FL worker.

Plain local training and evaluation -- no RAFT here. Worker nodes participate
in consensus through the supervisor process (`flnode.node`), which is what
publishes their contextual properties and watches the aggregator. Keeping the
ClientApp free of consensus concerns means the SuperNode can be restarted and
re-pointed at a new aggregator without disturbing cluster membership.

Note the article's Section 8.2 caveat, which still applies: "the only
information potentially lost during an aggregator crash is the most recent set
of local training updates sent by clients in the current training round",
because a client finishes training before discovering the aggregator has gone.
"""

from __future__ import annotations

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from flnode.model import DEVICE, Net, load_data, test, train

app = ClientApp()

# Built once per process rather than per message: the CIFAR-10 loaders are the
# expensive part, and the article's testbed is resource-constrained.
_net: Net | None = None
_trainloader = None
_testloader = None


def _setup(context: Context):
    global _net, _trainloader, _testloader
    if _net is None:
        _net = Net().to(DEVICE)
        partition_id = int(context.node_config.get("partition-id", 0))
        num_partitions = int(context.node_config.get("num-partitions", 1))
        _trainloader, _testloader = load_data(
            partition_id=partition_id, num_partitions=num_partitions
        )
    return _net, _trainloader, _testloader


@app.train()
def train_fn(msg: Message, context: Context) -> Message:
    net, trainloader, _ = _setup(context)
    net.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    epochs = int(msg.content["config"].get("local-epochs", 1))
    loss = train(net, trainloader, epochs=epochs)

    reply = RecordDict(
        {
            "arrays": ArrayRecord(net.state_dict()),
            "metrics": MetricRecord(
                {"train_loss": loss, "num-examples": len(trainloader.dataset)}
            ),
        }
    )
    return Message(reply, reply_to=msg)


@app.evaluate()
def evaluate_fn(msg: Message, context: Context) -> Message:
    net, _, testloader = _setup(context)
    net.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    loss, accuracy = test(net, testloader)

    reply = RecordDict(
        {
            "metrics": MetricRecord(
                {
                    "eval_loss": loss,
                    "accuracy": accuracy,
                    "num-examples": len(testloader.dataset),
                }
            )
        }
    )
    return Message(reply, reply_to=msg)
