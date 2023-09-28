import os

import sys, time
import flwr as fl
import tensorflow as tf
from pysyncobj import SyncObj, replicated
from SaveModelStrategy import SaveModelStrategy

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Load model and data (MobileNetV2, CIFAR-10)
model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


# Define Flower client
class RaftCifarClient(fl.client.NumPyClient, SyncObj):
    
    
    ## Flower part
    
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=1, batch_size=32)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}
    
    ## RAFT part  
    ## TODO

# Start Flower client
port = int(sys.argv[1])
partners = ['localhost:%d' % int(p) for p in sys.argv[2:]]
o = RaftCifarClient('localhost:%d' % port, partners)
while True:
    time.sleep(5)
    if o._isLeader():            
        print("I am the new leader, so I am starting the FL server")
        # Start Flower server
        fl.server.start_server(
            server_address="127.0.0.1:8080",
            strategy=SaveModelStrategy(),
            config=fl.server.ServerConfig(num_rounds=3),
            )
        # TODO: fetch the latest replicated model, merge new inputs, and save it back into the replicated object
    elif (o._getLeader() is None):
        print("Something is wrong, there is no leader currently, let's wait a bit and re-try.")        
    else:
        print("I am a worker, so I will be doing my local training and send the updates to: ", o._getLeader())
        try:
            fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=o)
        except:
            print("Error: server disconnected.")