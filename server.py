import flwr as fl
from SaveModelStrategy import SaveModelStrategy

# Start Flower server
hist = fl.server.start_server(
    server_address="0.0.0.0:8080",
    strategy=SaveModelStrategy(),
    config=fl.server.ServerConfig(num_rounds=3),
)

print(hist)
